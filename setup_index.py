"""Inicialización de la infraestructura: verifica si el índice serverless existe y lo crea si no.

La dimensión se obtiene del modelo de embeddings configurado (384 con el modelo local por
defecto, 1536 con text-embedding-3-small de OpenAI), así el índice siempre coincide con
los vectores que se van a subir.

Uso:  python setup_index.py
"""

import json
import logging
import sys

from dotenv import load_dotenv
from pydantic import ValidationError

from config import pinecone_settings_from_env
from embeddings import dimension, embeddings_settings_from_env, get_embeddings
from logging_setup import configurar_logging
from pinecone_store import IndexMismatchError, asegurar_indice, conectar, estadisticas

logger = logging.getLogger(__name__)


def main() -> int:
    settings = pinecone_settings_from_env()
    emb_settings = embeddings_settings_from_env()
    dim = dimension(get_embeddings(emb_settings))
    logger.info("Modelo de embeddings %s -> dimensión %d", emb_settings.identificador, dim)

    pc = conectar(settings)
    try:
        index, creado = asegurar_indice(pc, settings, dim, emb_settings.identificador)
    except IndexMismatchError as exc:
        logger.error("%s", exc)
        return 1

    descripcion = pc.describe_index(settings.index_name)
    print(f"\nÍndice '{settings.index_name}' {'creado' if creado else 'ya existente'}:")
    print(f"  host:      {descripcion.host}")
    print(f"  dimensión: {descripcion.dimension} | métrica: {descripcion.metric}")
    print(f"  spec:      serverless {settings.cloud}/{settings.region}")
    print("  stats:     " + json.dumps(estadisticas(index, settings.namespace), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    load_dotenv()
    configurar_logging()
    try:
        sys.exit(main())
    except ValidationError as exc:
        sys.exit("Configuración inválida: " + exc.errors()[0]["msg"])
