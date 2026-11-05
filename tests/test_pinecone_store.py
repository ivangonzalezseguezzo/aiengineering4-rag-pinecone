from types import SimpleNamespace

import pytest

from config import PineconeSettings
from pinecone_store import IndexMismatchError, asegurar_indice

EMB_ID = "huggingface:intfloat/multilingual-e5-small"


class FakePinecone:
    """Doble del cliente de Pinecone: guarda los índices creados en un diccionario."""

    def __init__(self, indices: dict | None = None) -> None:
        self.indices = indices or {}
        self.creados: list[dict] = []

    def has_index(self, nombre: str) -> bool:
        return nombre in self.indices

    def create_index(self, **kwargs) -> None:
        self.creados.append(kwargs)
        self.indices[kwargs["name"]] = {"dimension": kwargs["dimension"], "metric": kwargs["metric"], "tags": kwargs["tags"]}

    def describe_index(self, nombre: str):
        datos = self.indices[nombre]
        return SimpleNamespace(status=SimpleNamespace(ready=True), host="fake-host", **datos)

    def Index(self, nombre: str) -> str:  # noqa: N802 - mismo nombre que el SDK
        return f"index:{nombre}"


@pytest.fixture
def settings() -> PineconeSettings:
    return PineconeSettings(api_key="pc-test", index_name="orbita-test")


def test_crea_el_indice_si_no_existe(settings):
    pc = FakePinecone()
    index, creado = asegurar_indice(pc, settings, 384, EMB_ID)
    assert creado and index == "index:orbita-test"
    [llamada] = pc.creados
    assert llamada["dimension"] == 384 and llamada["metric"] == "cosine"
    assert llamada["spec"].cloud == "aws" and llamada["spec"].region == "us-east-1"
    assert llamada["tags"]["embeddings"] == "huggingface-intfloat-multilingual-e5-small"


def test_no_recrea_un_indice_existente(settings):
    pc = FakePinecone()
    asegurar_indice(pc, settings, 384, EMB_ID)
    _, creado = asegurar_indice(pc, settings, 384, EMB_ID)
    assert not creado and len(pc.creados) == 1


def test_rechaza_dimension_distinta(settings):
    pc = FakePinecone({"orbita-test": {"dimension": 1536, "metric": "cosine", "tags": None}})
    with pytest.raises(IndexMismatchError, match="dimensión 1536"):
        asegurar_indice(pc, settings, 384, EMB_ID)


def test_rechaza_otro_modelo_de_embeddings(settings):
    pc = FakePinecone({"orbita-test": {"dimension": 384, "metric": "cosine", "tags": {"embeddings": "otro-modelo"}}})
    with pytest.raises(IndexMismatchError, match="otro-modelo"):
        asegurar_indice(pc, settings, 384, EMB_ID)


def test_modo_consulta_no_crea_indices(settings):
    with pytest.raises(IndexMismatchError, match="no existe"):
        asegurar_indice(FakePinecone(), settings, 384, EMB_ID, crear=False)


def test_settings_validan_nombre_de_indice_y_api_key():
    with pytest.raises(ValueError):
        PineconeSettings(api_key="pc-test", index_name="Nombre_Invalido")
    with pytest.raises(ValueError):
        PineconeSettings(api_key="  ")


def test_tags_del_sdk_real_se_leen(settings):
    """Regresión: el SDK devuelve IndexTags (modelo OpenAPI sin .keys()), no un dict."""
    from pinecone.core.openapi.db_control.model.index_tags import IndexTags

    tags = IndexTags(embeddings="otro-modelo")
    pc = FakePinecone({"orbita-test": {"dimension": 384, "metric": "cosine", "tags": tags}})
    with pytest.raises(IndexMismatchError, match="otro-modelo"):
        asegurar_indice(pc, settings, 384, EMB_ID)
