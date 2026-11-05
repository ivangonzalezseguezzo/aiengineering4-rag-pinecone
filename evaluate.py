"""Evaluación cuantitativa del recuperador: Recall@k, Precision@k, Hit@k y MRR.

Corre el golden set (eval/golden_set.json) contra los tres modos de recuperación (BM25,
vectorial e híbrido) para que el número muestre si la búsqueda híbrida realmente mejora.
Imprime el reporte en consola y lo guarda en reports/evaluacion_<backend>.json.

Uso:
    python evaluate.py                      # contra Pinecone (requiere haber corrido ingest.py)
    python evaluate.py --backend memoria    # mismos chunks y embeddings en memoria, sin Pinecone
    python evaluate.py --modos hibrido --k 5
"""

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import datetime

from dotenv import load_dotenv
from pydantic import ValidationError

from config import retrieval_settings_from_env
from embeddings import embeddings_settings_from_env
from logging_setup import configurar_logging
from metrics import ResultadoPregunta, evaluar_pregunta, promedios
from rag_system import MODOS, RAGSystem, construir_rag_system
from schemas import GoldenSet

logger = logging.getLogger(__name__)


def cargar_golden_set(ruta, doc_ids_corpus: set[str]) -> GoldenSet:
    golden = GoldenSet.model_validate_json(ruta.read_text(encoding="utf-8"))
    # Un id mal escrito daría Recall 0 "por error de tipeo": se valida contra el corpus real.
    desconocidos = {d for caso in golden.preguntas for d in caso.documentos_relevantes} - doc_ids_corpus
    if desconocidos:
        raise ValueError(f"el golden set referencia documentos que no existen en el corpus: {sorted(desconocidos)}")
    return golden


def evaluar(rag: RAGSystem, golden: GoldenSet, modo: str, k: int) -> list[ResultadoPregunta]:
    chunks_por_doc = Counter(c.metadata["doc_id"] for c in rag.chunks)
    return [evaluar_pregunta(caso, rag.buscar(caso.pregunta, modo=modo), k, chunks_por_doc) for caso in golden.preguntas]


def imprimir_detalle(modo: str, resultados: list[ResultadoPregunta]) -> None:
    print(f"\n--- Detalle por pregunta (modo {modo}) ---")
    print(f"{'id':<4} {'tipo':<15} {'hit':>3} {'recall':>6} {'prec':>5} {'máx':>5} {'RR':>5}  recuperados (★ = relevante)")
    for r in resultados:
        marcados = " ".join(("★" if d in r.relevantes else "·") + d.split("#")[-1][:18] for d in r.recuperados)
        print(
            f"{r.id:<4} {r.tipo:<15} {'✔' if r.hit else '✘':>3} {r.recall:>6.2f} {r.precision:>5.2f} "
            f"{r.precision_maxima:>5.2f} {r.rr:>5.2f}  {marcados}"
        )
        if not r.hit:
            print(f"     ↳ esperado: {r.esperado} | pregunta: {r.pregunta}")


def imprimir_resumen(resumen: dict[str, dict[str, float]], k: int) -> None:
    print(f"\n=== Resumen (promedio sobre el golden set, k={k}) ===")
    print(f"{'modo':<10} {'Recall@'+str(k):>9} {'Precision@'+str(k):>12} {'(máx. posible)':>15} {'Hit@'+str(k):>7} {'MRR':>6}")
    for modo, m in resumen.items():
        print(
            f"{modo:<10} {m['recall@k']:>9.3f} {m['precision@k']:>12.3f} {m['precision_maxima@k']:>15.3f} "
            f"{m['hit@k']:>7.3f} {m['mrr']:>6.3f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluación del recuperador híbrido")
    parser.add_argument("--backend", choices=["pinecone", "memoria"], default="pinecone")
    parser.add_argument("--modos", nargs="+", choices=MODOS, default=["bm25", "vectorial", "hibrido"])
    parser.add_argument("--k", type=int, default=None, help="por defecto, RAG_TOP_K (5)")
    args = parser.parse_args()

    settings = retrieval_settings_from_env(**({"top_k": args.k} if args.k else {}))
    rag = construir_rag_system(args.backend)
    rag.top_k = settings.top_k
    golden = cargar_golden_set(settings.golden_set, {c.metadata["doc_id"] for c in rag.chunks})

    print(
        f"\n=== Evaluación de recuperación | backend={args.backend} | k={settings.top_k} | "
        f"{len(golden.preguntas)} preguntas | {len(rag.chunks)} chunks ==="
    )
    detalle: dict[str, list[dict]] = {}
    resumen: dict[str, dict[str, float]] = {}
    for modo in args.modos:
        resultados = evaluar(rag, golden, modo, settings.top_k)
        imprimir_detalle(modo, resultados)
        detalle[modo] = [r.to_dict() for r in resultados]
        resumen[modo] = promedios(resultados)
    imprimir_resumen(resumen, settings.top_k)

    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    ruta = settings.reports_dir / f"evaluacion_{args.backend}.json"
    reporte = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "backend": args.backend,
        "embeddings": embeddings_settings_from_env().identificador,
        "k": settings.top_k,
        "candidatos_por_retriever": settings.candidatos,
        "pesos_bm25_vectorial": settings.pesos,
        "chunking": {"chunk_size": settings.chunk_size, "chunk_overlap": settings.chunk_overlap},
        "resumen": resumen,
        "detalle": detalle,
    }
    ruta.write_text(json.dumps(reporte, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReporte guardado en {ruta.relative_to(settings.reports_dir.parent)}")
    return 0


if __name__ == "__main__":
    load_dotenv()
    configurar_logging()
    try:
        sys.exit(main())
    except ValidationError as exc:
        sys.exit("Configuración inválida: " + exc.errors()[0]["msg"])
    except (RuntimeError, ValueError) as exc:  # índice inexistente/incompatible, namespace vacío, golden set
        sys.exit(f"Error: {exc}")
