"""Fixtures compartidas. Ningún test usa red: embeddings falsos y un índice de Pinecone simulado."""

# FakeIndex define un método `list` (como pinecone.Index), que taparía al tipo list en las anotaciones.
from __future__ import annotations

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from config import RetrievalSettings
from documents import construir_chunks


@pytest.fixture(scope="session")
def settings() -> RetrievalSettings:
    return RetrievalSettings()


@pytest.fixture(scope="session")
def chunks(settings: RetrievalSettings):
    """El corpus real de data/docs, fragmentado igual que en la ingesta."""
    return construir_chunks(settings)


@pytest.fixture
def embeddings() -> DeterministicFakeEmbedding:
    return DeterministicFakeEmbedding(size=32)


class FakeIndex:
    """Doble de prueba de pinecone.Index: registra upserts y deletes en memoria."""

    def __init__(self, existentes: list[str] | None = None) -> None:
        self.vectores: dict[str, dict] = {vid: {} for vid in existentes or []}
        self.llamadas_upsert: list[int] = []
        self.borrados: list[str] = []

    def upsert(self, vectors: list[dict], namespace: str) -> None:
        self.llamadas_upsert.append(len(vectors))
        for vector in vectors:
            self.vectores[vector["id"]] = vector

    def list(self, namespace: str):
        ids = list(self.vectores)
        for inicio in range(0, len(ids), 100):
            yield ids[inicio : inicio + 100]

    def delete(self, ids: list[str], namespace: str) -> None:
        self.borrados.extend(ids)
        for vid in ids:
            self.vectores.pop(vid, None)


@pytest.fixture
def fake_index() -> FakeIndex:
    return FakeIndex(existentes=["doc_viejo::000", "doc_viejo::001"])
