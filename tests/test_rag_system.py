import pytest
from langchain_core.exceptions import ModelAuthenticationError, ModelRateLimitError, OutputParserException
from langchain_core.runnables import RunnableLambda

import rag_system as rag_module
from evaluate import cargar_golden_set
from pinecone_store import TEXT_KEY, borrar_obsoletos, upsert_en_lotes
from rag_system import MemoriaBackend, PineconeBackend, RAGSystem
from schemas import RESPUESTA_SIN_CONTEXTO, RespuestaGenerada

VALIDA = RespuestaGenerada(respuesta="Clave reutilizada.", respuesta_encontrada=True, fragmentos_usados=[1])


class ModeloGuionado:
    """Imita `model.with_structured_output(...)`: devuelve o lanza, en orden, lo que recibe."""

    def __init__(self, *guion) -> None:
        self.guion = list(guion)
        self.llamadas = 0
        self.method = None

    def with_structured_output(self, schema, method=None):
        assert schema is RespuestaGenerada
        self.method = method

        def paso(_prompt):
            self.llamadas += 1
            item = self.guion.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item

        return RunnableLambda(paso)


@pytest.fixture
def sin_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rag_module, "BACKOFF", {"initial": 0, "max": 0, "exp_base": 1, "jitter": 0})


@pytest.fixture
def rag(chunks, embeddings) -> RAGSystem:
    return RAGSystem(chunks, MemoriaBackend(chunks, embeddings), top_k=5, candidatos=10)


def test_bm25_encuentra_terminos_exactos(rag):
    # Con embeddings falsos la parte vectorial es ruido: lo que se prueba es el retriever léxico.
    resultados = rag.buscar("¿Qué significa el error PAY-4091?", modo="bm25")
    assert resultados[0].metadata["doc_id"] == "catalogo_errores#PAY-4091"


def test_hibrido_devuelve_top_k_sin_duplicados(rag):
    resultados = rag.buscar("rollback automático del canary", modo="hibrido")
    ids = [d.metadata["chunk_id"] for d in resultados]
    assert len(ids) == 5
    assert len(set(ids)) == 5
    # Lo que BM25 pone primero sobrevive a la fusión dentro del top-5.
    mejor_bm25 = rag.buscar("rollback automático del canary", modo="bm25")[0].metadata["chunk_id"]
    assert mejor_bm25 in ids


def test_filtro_por_categoria(rag):
    for modo in ("bm25", "vectorial", "hibrido"):
        resultados = rag.buscar("rotación de secretos y accesos", modo=modo, categoria="seguridad")
        assert resultados
        assert {d.metadata["categoria"] for d in resultados} == {"seguridad"}


def test_consulta_vacia(rag):
    with pytest.raises(ValueError):
        rag.buscar("  ")


async def test_abuscar_es_asincronica(rag):
    resultados = await rag.abuscar("PAY-4091", modo="bm25")
    assert resultados[0].metadata["doc_id"] == "catalogo_errores#PAY-4091"


async def test_aresponder_mapea_fuentes(rag):
    rag._cadena = RunnableLambda(
        lambda _: RespuestaGenerada(respuesta="Clave reutilizada.", respuesta_encontrada=True, fragmentos_usados=[1, 9])
    )
    respuesta = await rag.aresponder("¿Qué significa PAY-4091?")
    assert respuesta.respuesta_encontrada
    assert [f.fragmento for f in respuesta.fuentes] == [1]  # el 9 no existe y se descarta


async def test_aresponder_sin_respuesta(rag):
    rag._cadena = RunnableLambda(
        lambda _: RespuestaGenerada(respuesta="No está.", respuesta_encontrada=False, fragmentos_usados=[])
    )
    respuesta = await rag.aresponder("¿Cuál es el precio de la acción?")
    assert not respuesta.respuesta_encontrada
    assert respuesta.respuesta.startswith(RESPUESTA_SIN_CONTEXTO)
    assert respuesta.fuentes == []


async def test_generacion_usa_json_schema_y_reintenta_salida_invalida_y_rate_limit(chunks, embeddings, sin_backoff):
    modelo = ModeloGuionado(OutputParserException("json inválido"), ModelRateLimitError("429 por minuto"), VALIDA)
    rag = RAGSystem(chunks, MemoriaBackend(chunks, embeddings), model=modelo)
    respuesta = await rag.aresponder("¿Qué significa PAY-4091?")
    # json_schema: en Anthropic el default (tool calling forzado) da 400 con los modelos 5.5.
    assert modelo.method == "json_schema"
    assert modelo.llamadas == 3
    assert respuesta.respuesta_encontrada


@pytest.mark.parametrize(
    "error",
    [
        ModelRateLimitError("429 RESOURCE_EXHAUSTED quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier"),
        ModelAuthenticationError("401 API key inválida"),
    ],
)
async def test_generacion_no_reintenta_cuota_diaria_ni_errores_permanentes(chunks, embeddings, sin_backoff, error):
    modelo = ModeloGuionado(error, VALIDA)
    rag = RAGSystem(chunks, MemoriaBackend(chunks, embeddings), model=modelo)
    with pytest.raises(type(error)):
        await rag.aresponder("¿Qué significa PAY-4091?")
    assert modelo.llamadas == 1


def test_pinecone_backend_arma_namespace_y_filtro():
    class StoreEspia:
        def as_retriever(self, search_kwargs):
            self.search_kwargs = search_kwargs
            return search_kwargs

    store = StoreEspia()
    PineconeBackend(store, "handbook-v1").retriever(10, "seguridad")
    assert store.search_kwargs == {"k": 10, "namespace": "handbook-v1", "filter": {"categoria": {"$eq": "seguridad"}}}


def test_upsert_por_lotes_con_texto_en_metadata(chunks, embeddings, fake_index):
    subidos = upsert_en_lotes(fake_index, chunks, embeddings, "ns", batch_size=10)
    assert subidos == len(chunks)
    assert fake_index.llamadas_upsert == [10] * (len(chunks) // 10) + ([len(chunks) % 10] if len(chunks) % 10 else [])
    vector = fake_index.vectores[chunks[0].id]
    assert vector["metadata"][TEXT_KEY] == chunks[0].page_content
    assert vector["metadata"]["fuente"] == chunks[0].metadata["fuente"]
    assert len(vector["values"]) == 32


def test_borra_solo_vectores_obsoletos(chunks, embeddings, fake_index):
    upsert_en_lotes(fake_index, chunks, embeddings, "ns", batch_size=100)
    eliminados = borrar_obsoletos(fake_index, "ns", {c.id for c in chunks})
    assert eliminados == 2
    assert sorted(fake_index.borrados) == ["doc_viejo::000", "doc_viejo::001"]
    assert set(fake_index.vectores) == {c.id for c in chunks}


def test_golden_set_referencia_documentos_existentes(settings, chunks):
    golden = cargar_golden_set(settings.golden_set, {c.metadata["doc_id"] for c in chunks})
    assert len(golden.preguntas) >= 5
    assert {"semantica", "termino_exacto"} <= {p.tipo for p in golden.preguntas}


def test_golden_set_con_documento_inexistente_falla(tmp_path, chunks):
    ruta = tmp_path / "golden.json"
    preguntas = [
        {"id": f"q{i}", "pregunta": "¿pregunta?", "documento_id_esperado": "no_existe", "documentos_relevantes": ["no_existe"]}
        for i in range(5)
    ]
    ruta.write_text('{"preguntas": ' + __import__("json").dumps(preguntas) + "}", encoding="utf-8")
    with pytest.raises(ValueError, match="no existen en el corpus"):
        cargar_golden_set(ruta, {c.metadata["doc_id"] for c in chunks})
