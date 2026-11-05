from collections import Counter

import pytest
from langchain_core.documents import Document

from metrics import evaluar_pregunta, hit_at_k, precision_at_k, promedios, recall_at_k, reciprocal_rank
from schemas import CasoGolden

RECUPERADOS = ["a", "x", "a", "b", "y"]


def test_recall_es_fraccion_de_documentos_relevantes_encontrados():
    assert recall_at_k(RECUPERADOS, {"a"}) == 1.0
    assert recall_at_k(RECUPERADOS, {"a", "c"}) == 0.5


def test_precision_cuenta_chunks_relevantes_sobre_k():
    assert precision_at_k(RECUPERADOS, {"a", "b"}, k=5) == pytest.approx(0.6)
    assert precision_at_k(RECUPERADOS, {"z"}, k=5) == 0.0


def test_hit_y_reciprocal_rank():
    assert hit_at_k(RECUPERADOS, "b") == 1.0
    assert hit_at_k(RECUPERADOS, "c") == 0.0
    assert reciprocal_rank(RECUPERADOS, {"b"}) == pytest.approx(1 / 4)
    assert reciprocal_rank(RECUPERADOS, {"c"}) == 0.0


def test_evaluar_pregunta_y_precision_maxima():
    caso = CasoGolden(id="q1", pregunta="¿pregunta de prueba?", documento_id_esperado="a", documentos_relevantes=["a"])
    documentos = [Document(page_content=d, metadata={"doc_id": d, "chunk_id": f"{d}::{i}"}) for i, d in enumerate(RECUPERADOS)]
    resultado = evaluar_pregunta(caso, documentos, k=5, chunks_por_doc=Counter({"a": 2, "b": 3}))

    assert resultado.hit == 1.0 and resultado.recall == 1.0
    assert resultado.precision == pytest.approx(0.4)
    assert resultado.precision_maxima == pytest.approx(0.4)  # "a" solo tiene 2 chunks en el corpus
    assert resultado.rr == 1.0
    assert promedios([resultado])["precision@k"] == pytest.approx(0.4)


def test_el_esperado_siempre_cuenta_como_relevante():
    caso = CasoGolden(id="q1", pregunta="¿pregunta de prueba?", documento_id_esperado="a", documentos_relevantes=["b"])
    assert caso.documentos_relevantes == ["a", "b"]
