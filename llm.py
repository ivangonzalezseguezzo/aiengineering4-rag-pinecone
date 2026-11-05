"""Fábrica del chat model de LangChain según el proveedor configurado en el entorno.

Es el mismo módulo de las pre-entregas 2, 3 y 5: las variables de entorno se validan con
Pydantic y el resto del código no sabe qué proveedor hay detrás (Gemini, OpenAI o Anthropic).
Además centraliza:
- la traducción de `temperature` por proveedor (`parametros_sampling`);
- el rate limiter compartido (`InMemoryRateLimiter`, una instancia por cupo);
- la política de reintentos ante errores del proveedor (`es_reintentable`). El SDK no
  reintenta solo, porque no distingue un 429 por minuto (conviene esperar) de un 429 por
  cupo diario (no tiene sentido reintentar).

Solo lo usa la generación opcional (`consultar.py --responder`); la recuperación y la
evaluación no llaman a ningún LLM.
"""

import logging
import os
from functools import cache
from typing import Any, Literal

from langchain_core.exceptions import ModelError, ModelRateLimitError
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from pydantic import BaseModel, Field, SecretStr, model_validator

logger = logging.getLogger(__name__)

Provider = Literal["gemini", "openai", "anthropic"]
ThinkingLevel = Literal["low", "medium", "high"]

# IDs reales de cada API (tabla "Modelos por defecto" del README, con fuentes):
# - gemini-3.5-flash-lite: Flash Lite tiene 15 requests/min y 500/día en el plan gratuito
#   (los Flash, 5/min y 20/día). Respaldo: gemini-3.1-flash-lite.
#   https://ai.google.dev/gemini-api/docs/models
# - gpt-4o-mini: deprecado, se da de baja el 2027-01-14 (reemplazo: gpt-6-luna)
#   https://developers.openai.com/api/docs/deprecations
# - claude-sonnet-5-5: retiro "no antes del 2027-09-28"
#   https://platform.claude.com/docs/en/about-claude/models/overview
# Se pueden cambiar sin tocar código con GEMINI_MODEL / OPENAI_MODEL / ANTHROPIC_MODEL.
DEFAULT_MODELS: dict[str, str] = {
    "gemini": "gemini-3.5-flash-lite",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-5-5",
}

API_KEY_ENVS: dict[str, tuple[str, ...]] = {
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
}

ERROR_MAX_CHARS = 200


class LLMSettings(BaseModel):
    provider: Provider = "gemini"
    model: str | None = Field(default=None, description="Si es None se usa DEFAULT_MODELS[provider]")
    api_key: SecretStr | None = Field(default=None, repr=False)
    temperature: float | None = Field(
        default=None, ge=0.0, le=2.0, description="None = valor por defecto del proveedor"
    )
    # Solo aplica a Gemini 3: los modelos que "piensan" gastan parte de max_tokens razonando.
    thinking_level: ThinkingLevel = "low"
    max_tokens: int = Field(default=2048, gt=0, le=32_000)
    timeout: float = Field(default=30.0, gt=0)
    max_retries: int = Field(
        default=2, ge=0, le=10, description="Reintentos de la cadena ante errores transitorios (el SDK no reintenta)"
    )
    # Cupo de requests por minuto (plan gratuito de Gemini: 15 en Flash Lite, 5 en Flash). None = sin límite.
    requests_per_minute: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def completar_y_verificar(self) -> "LLMSettings":
        if not self.model:
            self.model = DEFAULT_MODELS[self.provider]
        if self.api_key is None or not self.api_key.get_secret_value():
            nombres = " o ".join(API_KEY_ENVS[self.provider])
            raise ValueError(f"falta la API key de {self.provider}: definí {nombres} en el .env")
        return self


def settings_from_env(provider: str | None = None, **overrides: Any) -> LLMSettings:
    """Arma LLMSettings desde variables de entorno (ver .env.example); `overrides` tiene prioridad."""
    provider = (provider or os.getenv("LLM_PROVIDER") or "gemini").strip().lower()
    api_key = next((os.getenv(n) for n in API_KEY_ENVS.get(provider, ()) if os.getenv(n)), None)
    data: dict[str, Any] = {
        "provider": provider,
        "model": os.getenv(f"{provider.upper()}_MODEL"),
        "api_key": api_key,
        "temperature": os.getenv("LLM_TEMPERATURE"),
        "thinking_level": os.getenv("LLM_THINKING_LEVEL"),
        "max_tokens": os.getenv("LLM_MAX_TOKENS"),
        "timeout": os.getenv("LLM_TIMEOUT"),
        "max_retries": os.getenv("LLM_MAX_RETRIES"),
        "requests_per_minute": os.getenv("LLM_REQUESTS_PER_MINUTE"),
    }
    data = {k: v for k, v in data.items() if v not in (None, "")}
    data.update(overrides)
    return LLMSettings.model_validate(data)


def _es_gemini_3(s: LLMSettings) -> bool:
    return s.provider == "gemini" and bool(s.model) and s.model.startswith("gemini-3")


def parametros_sampling(s: LLMSettings) -> dict[str, Any]:
    """Traduce LLM_TEMPERATURE a lo que acepta cada proveedor (ver tabla del README).

    - OpenAI y Gemini 2.x: se envía tal cual, solo si se configuró.
    - Gemini 3.x: Google la declaró deprecada e ignorada desde gemini-3.6-flash y
      gemini-3.5-flash-lite; se omite en toda la familia gemini-3*.
    - Anthropic (Claude Opus 4.7 en adelante, incluida la familia 5.5): la API responde 400.
    En los dos últimos casos se avisa en el log en lugar de ignorarla en silencio, y el estilo
    de la respuesta queda fijado por el prompt grounded y la salida estructurada.
    """
    if s.temperature is None:
        return {}
    if s.provider == "anthropic" or _es_gemini_3(s):
        logger.warning("LLM_TEMPERATURE=%s se ignora: %s no acepta parámetros de sampling", s.temperature, s.model)
        return {}
    return {"temperature": s.temperature}


def reintentos_del_sdk(provider: str) -> int:
    """Valor de `max_retries` que desactiva los reintentos internos del SDK.

    En langchain-google-genai, `max_retries` es la cantidad total de intentos y la propia
    librería documenta que hay que pasar 1 (no 0) para que el SDK de Google no reintente.
    En OpenAI y Anthropic, `max_retries` cuenta reintentos: 0 = ninguno.
    """
    return 1 if provider == "gemini" else 0


@cache
def _rate_limiter(requests_per_minute: float) -> InMemoryRateLimiter:
    """Un único limitador por cupo: todos los modelos creados comparten el mismo balde de requests."""
    return InMemoryRateLimiter(requests_per_second=requests_per_minute / 60, max_bucket_size=1)


def get_chat_model(provider: str | None = None, **overrides: Any) -> BaseChatModel:
    """Devuelve ChatGoogleGenerativeAI, ChatOpenAI o ChatAnthropic según la configuración.

    Los reintentos NO los hace el SDK: los aplica la cadena con `es_reintentable` (ver rag_system.py).
    """
    s = settings_from_env(provider, **overrides)

    comunes: dict[str, Any] = {"timeout": s.timeout, "max_retries": reintentos_del_sdk(s.provider)}
    if s.requests_per_minute is not None:
        comunes["rate_limiter"] = _rate_limiter(s.requests_per_minute)
    sampling = parametros_sampling(s)

    # Imports diferidos: solo hace falta instalar el paquete del proveedor que se usa.
    match s.provider:
        case "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            # Gemini 3 razona antes de responder y ese razonamiento consume max_tokens.
            # Responder con el contexto ya recuperado no requiere razonamiento profundo:
            # "low" reduce latencia y evita que el razonamiento agote max_tokens.
            thinking = {"thinking_level": s.thinking_level} if _es_gemini_3(s) else {}
            return ChatGoogleGenerativeAI(
                model=s.model,
                google_api_key=s.api_key,
                max_output_tokens=s.max_tokens,
                **comunes,
                **sampling,
                **thinking,
            )
        case "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(model=s.model, api_key=s.api_key, max_tokens=s.max_tokens, **comunes, **sampling)
        case "anthropic":
            from langchain_anthropic import ChatAnthropic

            # `sampling` siempre está vacío acá (ver parametros_sampling).
            return ChatAnthropic(model=s.model, api_key=s.api_key, max_tokens=s.max_tokens, **comunes)


# --- Errores del proveedor ----------------------------------------------------------------


def resumir_error(exc: BaseException) -> str:
    """Primera línea del error, recortada: Google incluye el JSON completo de la respuesta."""
    linea = (str(exc).strip().splitlines() or [""])[0]
    return linea if len(linea) <= ERROR_MAX_CHARS else linea[:ERROR_MAX_CHARS] + "..."


def cuota_diaria_agotada(exc: BaseException) -> bool:
    """429 por cupo diario (quotaId GenerateRequestsPerDayPerProjectPerModel-FreeTier en Gemini).

    A diferencia del límite por minuto, no se resuelve esperando unos segundos.
    """
    return isinstance(exc, ModelRateLimitError) and "PerDay" in str(exc)


def es_reintentable(exc: BaseException) -> bool:
    """Política de reintento ante errores del proveedor.

    Se reintentan los transitorios que LangChain marca como tales (rate limit por minuto,
    timeout, conexión, 5xx). No se reintentan la cuota diaria ni los permanentes
    (autenticación, modelo inexistente, request inválido).
    """
    if cuota_diaria_agotada(exc):
        return False
    return isinstance(exc, ModelError) and exc.is_retryable
