"""Fábrica del modelo de embeddings según el proveedor configurado en el entorno.

Regla de oro: el MISMO modelo indexa y consulta. Por eso cada configuración tiene un
`identificador` (`proveedor:modelo`) que se guarda como tag del índice de Pinecone y se
verifica al abrirlo, y la dimensión del índice se toma del propio modelo (ver
pinecone_store.py). Así no hay forma de subir vectores de 384 a un índice de 1536.

Por defecto se usa un modelo local de HuggingFace: no necesita API key ni tiene costo.
"""

import logging
import os
from typing import Any, Literal

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field, SecretStr, model_validator

logger = logging.getLogger(__name__)

EmbeddingsProvider = Literal["huggingface", "gemini", "openai"]

DEFAULT_EMBEDDING_MODELS: dict[str, str] = {
    # Multilingüe (el corpus está en español) y con ventana de 512 tokens, acorde a chunks de 500.
    # Genera vectores de 384 dimensiones: el índice de Pinecone se crea con esa dimensión.
    # Modelos más chicos como all-MiniLM-L6-v2 truncan el texto a 256 tokens y están entrenados en inglés.
    "huggingface": "intfloat/multilingual-e5-small",
    "gemini": "gemini-embedding-001",
    "openai": "text-embedding-3-small",
}

API_KEY_ENVS: dict[str, tuple[str, ...]] = {
    "huggingface": (),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
}

# Los modelos E5 se entrenaron con prefijos distintos para consultas y para pasajes.
# Sin ellos la similitud se degrada, así que se agregan solos cuando el modelo es de esa familia.
E5_PREFIJO_CONSULTA = "query: "
E5_PREFIJO_PASAJE = "passage: "


class EmbeddingsSettings(BaseModel):
    provider: EmbeddingsProvider = "huggingface"
    model: str | None = Field(default=None, description="Si es None se usa DEFAULT_EMBEDDING_MODELS[provider]")
    api_key: SecretStr | None = Field(default=None, repr=False)

    @model_validator(mode="after")
    def completar_y_verificar(self) -> "EmbeddingsSettings":
        if not self.model:
            self.model = DEFAULT_EMBEDDING_MODELS[self.provider]
        nombres = API_KEY_ENVS[self.provider]
        if nombres and (self.api_key is None or not self.api_key.get_secret_value()):
            raise ValueError(f"falta la API key de embeddings de {self.provider}: definí {' o '.join(nombres)}")
        return self

    @property
    def identificador(self) -> str:
        """Identifica unívocamente el espacio vectorial: dos índices con distinto id no son comparables."""
        return f"{self.provider}:{self.model}"


def embeddings_settings_from_env(**overrides: Any) -> EmbeddingsSettings:
    provider = (os.getenv("EMBEDDINGS_PROVIDER") or "huggingface").strip().lower()
    api_key = next((os.getenv(n) for n in API_KEY_ENVS.get(provider, ()) if os.getenv(n)), None)
    data: dict[str, Any] = {"provider": provider, "model": os.getenv("EMBEDDINGS_MODEL"), "api_key": api_key}
    data = {k: v for k, v in data.items() if v not in (None, "")}
    data.update(overrides)
    return EmbeddingsSettings.model_validate(data)


def get_embeddings(settings: EmbeddingsSettings | None = None) -> Embeddings:
    """Devuelve HuggingFaceEmbeddings, GoogleGenerativeAIEmbeddings u OpenAIEmbeddings."""
    s = settings or embeddings_settings_from_env()

    # Imports diferidos: solo hace falta instalar el paquete del proveedor que se usa.
    match s.provider:
        case "huggingface":
            from langchain_huggingface import HuggingFaceEmbeddings

            es_e5 = "e5" in s.model.lower()
            return HuggingFaceEmbeddings(
                model_name=s.model,
                # Vectores normalizados: la similitud coseno queda en una escala estable.
                encode_kwargs={"normalize_embeddings": True, **({"prompt": E5_PREFIJO_PASAJE} if es_e5 else {})},
                query_encode_kwargs={
                    "normalize_embeddings": True,
                    **({"prompt": E5_PREFIJO_CONSULTA} if es_e5 else {}),
                },
            )
        case "gemini":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            return GoogleGenerativeAIEmbeddings(model=s.model, google_api_key=s.api_key)
        case "openai":
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(model=s.model, api_key=s.api_key)


def dimension(embeddings: Embeddings) -> int:
    """Dimensión real del modelo, medida embebiendo un texto de prueba (no se hardcodea)."""
    return len(embeddings.embed_query("dimension"))


def advertir_truncamiento(embeddings: Embeddings, chunks: list[Document]) -> int:
    """Avisa si algún chunk supera la ventana del modelo local y se embebería truncado.

    Los chunks se miden con tiktoken, pero cada modelo tiene su propio tokenizador. Solo se
    puede verificar en los modelos de sentence-transformers; para el resto devuelve 0.
    """
    cliente = getattr(embeddings, "_client", None)
    tokenizer, limite = getattr(cliente, "tokenizer", None), getattr(cliente, "max_seq_length", None)
    if tokenizer is None or not limite:
        return 0
    prefijo = (getattr(embeddings, "encode_kwargs", None) or {}).get("prompt", "")
    largos = [len(tokenizer(prefijo + c.page_content)["input_ids"]) for c in chunks]
    excedidos = sum(largo > limite for largo in largos)
    if excedidos:
        logger.warning(
            "%d de %d chunks superan la ventana de %d tokens del modelo de embeddings (máx. %d) y se "
            "embeberán truncados: bajá RAG_CHUNK_SIZE o usá un modelo con más contexto",
            excedidos,
            len(chunks),
            limite,
            max(largos),
        )
    else:
        logger.info("Todos los chunks entran en la ventana del modelo de embeddings (máx. %d/%d)", max(largos), limite)
    return excedidos
