"""Configuración del LLM: modelos, sampling por proveedor, rate limiter y reintentos (sin red)."""

import logging

import pytest
from langchain_core.exceptions import ModelAuthenticationError, ModelRateLimitError, ModelTimeoutError
from pydantic import ValidationError

from llm import (
    DEFAULT_MODELS,
    LLMSettings,
    cuota_diaria_agotada,
    es_reintentable,
    get_chat_model,
    parametros_sampling,
    resumir_error,
    settings_from_env,
)

DIARIO = "429 RESOURCE_EXHAUSTED quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier"
POR_MINUTO = "429 RESOURCE_EXHAUSTED quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier"


@pytest.fixture(autouse=True)
def entorno_limpio(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "LLM_PROVIDER", "LLM_TEMPERATURE", "LLM_THINKING_LEVEL", "LLM_REQUESTS_PER_MINUTE", "LLM_MAX_RETRIES",
        "GEMINI_MODEL", "OPENAI_MODEL", "ANTHROPIC_MODEL", "GEMINI_API_KEY", "GOOGLE_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.mark.parametrize("provider", ["gemini", "openai", "anthropic"])
def test_modelo_por_defecto(provider: str) -> None:
    s = LLMSettings(provider=provider, api_key="clave")
    assert s.model == DEFAULT_MODELS[provider]


def test_default_de_gemini_es_flash_lite() -> None:
    # Flash Lite: 500 requests/día en el plan gratuito (los Flash, 20). Igual que en las PE2, PE3 y PE5.
    assert DEFAULT_MODELS["gemini"] == "gemini-3.5-flash-lite"


def test_falta_api_key() -> None:
    with pytest.raises(ValidationError, match="falta la API key de openai"):
        LLMSettings(provider="openai")


def test_temperature_se_envia_a_openai_y_gemini_2() -> None:
    assert parametros_sampling(LLMSettings(provider="openai", api_key="k", temperature=0.2)) == {"temperature": 0.2}
    gemini_2 = LLMSettings(provider="gemini", model="gemini-2.5-flash", api_key="k", temperature=0.2)
    assert parametros_sampling(gemini_2) == {"temperature": 0.2}


@pytest.mark.parametrize("provider", ["anthropic", "gemini"])
def test_temperature_no_se_envia_si_el_modelo_la_rechaza_y_avisa(
    provider: str, caplog: pytest.LogCaptureFixture
) -> None:
    s = LLMSettings(provider=provider, api_key="clave", temperature=0.2)  # claude-sonnet-5-5 / gemini-3.5-flash-lite
    with caplog.at_level(logging.WARNING, logger="llm"):
        assert parametros_sampling(s) == {}
    assert "se ignora" in caplog.text


def test_sin_temperature_usa_default_del_modelo() -> None:
    assert parametros_sampling(LLMSettings(provider="gemini", api_key="clave")) == {}


def test_settings_desde_entorno(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "clave")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-haiku-5-5")
    s = settings_from_env()
    assert (s.provider, s.model) == ("anthropic", "claude-haiku-5-5")


def test_modelo_gemini_llega_configurado(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "clave-de-prueba")
    monkeypatch.setenv("LLM_REQUESTS_PER_MINUTE", "12")
    monkeypatch.setenv("LLM_THINKING_LEVEL", "medium")
    modelo = get_chat_model()
    assert modelo.model == "gemini-3.5-flash-lite"
    assert modelo.thinking_level == "medium"
    # En langchain-google-genai max_retries cuenta intentos totales: 1 = el SDK no reintenta.
    assert modelo.max_retries == 1
    assert modelo.rate_limiter is not None
    assert get_chat_model().rate_limiter is modelo.rate_limiter  # un solo balde por cupo


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_openai_y_anthropic_sin_reintentos_del_sdk(provider: str) -> None:
    assert get_chat_model(provider, api_key="clave").max_retries == 0


def test_cuota_diaria_no_se_reintenta_y_la_por_minuto_si() -> None:
    assert cuota_diaria_agotada(ModelRateLimitError(DIARIO))
    assert not es_reintentable(ModelRateLimitError(DIARIO))
    assert es_reintentable(ModelRateLimitError(POR_MINUTO))
    assert es_reintentable(ModelTimeoutError("timeout"))
    assert not es_reintentable(ModelAuthenticationError("401"))
    assert not es_reintentable(KeyError("x"))


def test_mensaje_de_error_recortado() -> None:
    assert len(resumir_error(ModelRateLimitError("429 " + "x" * 5000 + "\nsegunda línea"))) <= 203
