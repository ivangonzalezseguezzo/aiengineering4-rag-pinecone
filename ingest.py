"""Pipeline de ingesta a Pinecone Serverless.

    data/docs (MD, JSON, PDF) ──► limpieza ──► chunks de ≤500 tokens con metadata
        ──► embeddings por lotes ──► upsert por lotes al namespace ──► borrado de obsoletos

Es idempotente: los ids de los chunks son deterministas, así que volver a correrla
actualiza los vectores existentes en lugar de duplicarlos, y los chunks que ya no están en
el corpus se eliminan del namespace.

Uso:
    python ingest.py            # sincroniza el namespace con data/docs
    python ingest.py --reset    # vacía el namespace antes de subir
"""

import argparse
import logging
import sys
import time
from collections import Counter

from dotenv import load_dotenv
from pydantic import ValidationError

from config import pinecone_settings_from_env, retrieval_settings_from_env
from documents import construir_chunks
from embeddings import advertir_truncamiento, dimension, embeddings_settings_from_env, get_embeddings
from logging_setup import configurar_logging
from pinecone_store import (
    IndexMismatchError,
    asegurar_indice,
    borrar_namespace,
    borrar_obsoletos,
    conectar,
    esperar_conteo,
    upsert_en_lotes,
)

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingesta del corpus a Pinecone")
    parser.add_argument("--reset", action="store_true", help="vacía el namespace antes de subir")
    args = parser.parse_args()

    pc_settings = pinecone_settings_from_env()
    settings = retrieval_settings_from_env()
    emb_settings = embeddings_settings_from_env()
    embeddings = get_embeddings(emb_settings)

    # 1. Corpus -> chunks con metadata validada
    chunks = construir_chunks(settings)
    advertir_truncamiento(embeddings, chunks)

    # 2. Infraestructura: crea el índice si no existe y verifica dimensión/modelo
    try:
        index, _ = asegurar_indice(conectar(pc_settings), pc_settings, dimension(embeddings), emb_settings.identificador)
    except IndexMismatchError as exc:
        logger.error("%s", exc)
        return 1
    if args.reset:
        borrar_namespace(index, pc_settings.namespace)

    # 3. Upsert por lotes + limpieza de vectores que ya no existen en el corpus
    inicio = time.perf_counter()
    subidos = upsert_en_lotes(index, chunks, embeddings, pc_settings.namespace, pc_settings.batch_size)
    eliminados = borrar_obsoletos(index, pc_settings.namespace, {c.id for c in chunks})
    duracion = time.perf_counter() - inicio

    # 4. Verificación: las estadísticas del índice deben reflejar el corpus
    stats = esperar_conteo(index, pc_settings.namespace, len(chunks))
    por_categoria = Counter(c.metadata["categoria"] for c in chunks)
    tokens = [c.metadata["n_tokens"] for c in chunks]

    print("\n=== Resumen de la ingesta ===")
    print(f"Índice / namespace:   {pc_settings.index_name} / {pc_settings.namespace}")
    print(f"Embeddings:           {emb_settings.identificador} (dim {stats['dimension']})")
    print(f"Documentos lógicos:   {len({c.metadata['doc_id'] for c in chunks})}")
    print(f"Chunks:               {len(chunks)} (tokens: min {min(tokens)}, máx {max(tokens)}, "
          f"promedio {sum(tokens) / len(tokens):.0f})")
    print("Por categoría:        " + ", ".join(f"{k}={v}" for k, v in sorted(por_categoria.items())))
    print(f"Upsert:               {subidos} vectores en lotes de {pc_settings.batch_size} ({duracion:.1f}s)")
    print(f"Obsoletos eliminados: {eliminados}")
    print(f"Vectores en Pinecone: {stats['vectores_en_namespace']} en el namespace, {stats['total_vectores']} en total")
    if stats["vectores_en_namespace"] != len(chunks):
        print("Aviso: las estadísticas de Pinecone tardan unos segundos en actualizarse; volvé a consultar con setup_index.py")
    return 0


if __name__ == "__main__":
    load_dotenv()
    configurar_logging()
    try:
        sys.exit(main())
    except ValidationError as exc:
        sys.exit("Configuración inválida: " + exc.errors()[0]["msg"])
