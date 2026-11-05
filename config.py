"""Configuración del sistema: infraestructura de Pinecone y parámetros de recuperación.

Se separa en dos modelos porque no siempre hacen falta los dos: la evaluación con
`--backend memoria` y los tests no necesitan credenciales de Pinecone.
Todo se lee de variables de entorno (ver .env.example) y se valida con Pydantic.
"""

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator

BASE_DIR = Path(__file__).resolve().parent


def _desde_entorno(variables: dict[str, str], overrides: dict[str, Any]) -> dict[str, Any]:
    """{campo: variable de entorno} -> dict con los valores definidos; `overrides` tiene prioridad."""
    data = {campo: os.getenv(var) for campo, var in variables.items()}
    data = {k: v for k, v in data.items() if v not in (None, "")}
    data.update(overrides)
    return data


class PineconeSettings(BaseModel):
    api_key: SecretStr = Field(repr=False)
    # Pinecone exige minúsculas, números y guiones (máximo 45 caracteres).
    index_name: str = Field(default="orbita-handbook", pattern=r"^[a-z0-9]([a-z0-9-]{0,43}[a-z0-9])?$")
    # Namespace = partición lógica del índice: permite convivir varias versiones del corpus o varios tenants.
    namespace: str = Field(default="handbook-v1", min_length=1, max_length=64)
    cloud: Literal["aws", "gcp", "azure"] = "aws"
    region: str = "us-east-1"
    metric: Literal["cosine", "dotproduct", "euclidean"] = "cosine"
    batch_size: int = Field(default=100, ge=1, le=1000, description="Vectores por upsert")

    @field_validator("api_key")
    @classmethod
    def api_key_presente(cls, valor: SecretStr) -> SecretStr:
        if not valor.get_secret_value().strip():
            raise ValueError("falta PINECONE_API_KEY en el .env")
        return valor


class RetrievalSettings(BaseModel):
    docs_dir: Path = BASE_DIR / "data" / "docs"
    golden_set: Path = BASE_DIR / "eval" / "golden_set.json"
    reports_dir: Path = BASE_DIR / "reports"
    # La consigna sugiere 500-800 tokens. 500 es el valor que entra completo en la ventana de
    # 512 tokens del modelo local de embeddings; con OpenAI o Gemini se puede subir.
    chunk_size: int = Field(default=500, ge=500, le=800, description="Tamaño máximo del chunk, en tokens")
    chunk_overlap: int = Field(default=75, ge=50, le=200, description="Tokens compartidos entre chunks")
    token_encoding: str = "cl100k_base"
    top_k: int = Field(default=5, ge=1, le=10, description="Documentos que devuelve el sistema")
    candidatos: int = Field(default=10, ge=1, le=50, description="Documentos que aporta cada retriever a la fusión")
    peso_bm25: float = Field(default=0.5, ge=0.0, le=1.0, description="Peso del retriever léxico en el ensemble")

    @field_validator("docs_dir", "golden_set", "reports_dir")
    @classmethod
    def resolver_ruta(cls, ruta: Path) -> Path:
        """Las rutas relativas se interpretan desde la carpeta del proyecto, no desde el cwd."""
        return ruta if ruta.is_absolute() else BASE_DIR / ruta

    @model_validator(mode="after")
    def validar(self) -> "RetrievalSettings":
        if self.candidatos < self.top_k:
            raise ValueError("candidatos tiene que ser mayor o igual que top_k")
        if self.chunk_overlap >= self.chunk_size // 2:
            raise ValueError("chunk_overlap tiene que ser menor que la mitad de chunk_size")
        return self

    @property
    def pesos(self) -> list[float]:
        """Pesos del EnsembleRetriever en el orden [bm25, vectorial]."""
        return [self.peso_bm25, round(1.0 - self.peso_bm25, 4)]


def pinecone_settings_from_env(**overrides: Any) -> PineconeSettings:
    variables = {
        "api_key": "PINECONE_API_KEY",
        "index_name": "INDEX_NAME",
        "namespace": "PINECONE_NAMESPACE",
        "cloud": "PINECONE_CLOUD",
        "region": "PINECONE_REGION",
        "metric": "PINECONE_METRIC",
        "batch_size": "PINECONE_BATCH_SIZE",
    }
    data = _desde_entorno(variables, overrides)
    data.setdefault("api_key", "")
    return PineconeSettings.model_validate(data)


def retrieval_settings_from_env(**overrides: Any) -> RetrievalSettings:
    variables = {
        "docs_dir": "RAG_DOCS_DIR",
        "chunk_size": "RAG_CHUNK_SIZE",
        "chunk_overlap": "RAG_CHUNK_OVERLAP",
        "top_k": "RAG_TOP_K",
        "candidatos": "RAG_CANDIDATOS",
        "peso_bm25": "RAG_PESO_BM25",
    }
    return RetrievalSettings.model_validate(_desde_entorno(variables, overrides))
