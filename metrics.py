"""Métricas de recuperación a nivel documento.

Un chunk recuperado cuenta como relevante si pertenece a uno de los documentos que
contienen la respuesta (`documentos_relevantes` del golden set).

- Recall@k: qué fracción de los documentos relevantes aparece entre los k recuperados.
  Con un solo documento relevante vale 0 o 1 ("¿está el documento correcto entre los 5?").
- Precision@k: qué fracción de los k recuperados es relevante ("¿qué % de los 5 sirve?").
- Hit@k: si el documento esperado (la fuente principal) aparece entre los k.
- MRR: 1 / posición del primer resultado relevante. Premia que lo útil aparezca arriba.

Precision@k tiene un techo: si un documento relevante solo genera 2 chunks, el máximo
posible con k=5 es 0,4. Por eso también se informa `precision_maxima`.
"""

from collections import Counter
from dataclasses import asdict, dataclass

from langchain_core.documents import Document

from schemas import CasoGolden


def recall_at_k(recuperados: list[str], relevantes: set[str]) -> float:
    return len(set(recuperados) & relevantes) / len(relevantes)


def precision_at_k(recuperados: list[str], relevantes: set[str], k: int) -> float:
    return sum(doc_id in relevantes for doc_id in recuperados[:k]) / k


def hit_at_k(recuperados: list[str], esperado: str) -> float:
    return 1.0 if esperado in recuperados else 0.0


def reciprocal_rank(recuperados: list[str], relevantes: set[str]) -> float:
    return next((1.0 / posicion for posicion, doc_id in enumerate(recuperados, start=1) if doc_id in relevantes), 0.0)


@dataclass(frozen=True)
class ResultadoPregunta:
    id: str
    pregunta: str
    tipo: str
    esperado: str
    relevantes: list[str]
    recuperados: list[str]
    chunk_ids: list[str]
    recall: float
    precision: float
    precision_maxima: float
    hit: float
    rr: float

    def to_dict(self) -> dict:
        return asdict(self)


def evaluar_pregunta(
    caso: CasoGolden, documentos: list[Document], k: int, chunks_por_doc: Counter[str]
) -> ResultadoPregunta:
    recuperados = [d.metadata["doc_id"] for d in documentos[:k]]
    relevantes = set(caso.documentos_relevantes)
    chunks_relevantes = sum(chunks_por_doc[doc_id] for doc_id in relevantes)
    return ResultadoPregunta(
        id=caso.id,
        pregunta=caso.pregunta,
        tipo=caso.tipo,
        esperado=caso.documento_id_esperado,
        relevantes=sorted(relevantes),
        recuperados=recuperados,
        chunk_ids=[d.metadata["chunk_id"] for d in documentos[:k]],
        recall=recall_at_k(recuperados, relevantes),
        precision=precision_at_k(recuperados, relevantes, k),
        precision_maxima=min(k, chunks_relevantes) / k,
        hit=hit_at_k(recuperados, caso.documento_id_esperado),
        rr=reciprocal_rank(recuperados, relevantes),
    )


def promedios(resultados: list[ResultadoPregunta]) -> dict[str, float]:
    n = len(resultados)
    return {
        "recall@k": sum(r.recall for r in resultados) / n,
        "precision@k": sum(r.precision for r in resultados) / n,
        "precision_maxima@k": sum(r.precision_maxima for r in resultados) / n,
        "hit@k": sum(r.hit for r in resultados) / n,
        "mrr": sum(r.rr for r in resultados) / n,
    }
