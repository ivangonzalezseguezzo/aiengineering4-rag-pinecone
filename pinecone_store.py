"""Infraestructura de Pinecone Serverless: índice, ingesta por lotes y vector store.

Garantías que da este módulo:
- El índice se crea una sola vez (verifica si existe) y su dimensión sale del modelo de
  embeddings, no de un número escrito a mano: evita el "mismatch de dimensiones".
- El índice guarda como tag el modelo con el que se indexó y se rechaza abrirlo con otro.
- La ingesta sube vectores en lotes (no de a uno) y es idempotente: los ids son
  deterministas y los chunks que ya no existen en el corpus se borran.
"""

import logging
import re
import time
from collections.abc import Callable
from typing import Any, TypeVar

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from pinecone.db_data import Index

from config import PineconeSettings

logger = logging.getLogger(__name__)

TEXT_KEY = "text"  # clave de metadata donde se guarda el contenido del chunk
TAG_EMBEDDINGS = "embeddings"
T = TypeVar("T")


class IndexMismatchError(RuntimeError):
    """El índice existente no es compatible con el modelo de embeddings o la métrica configurados."""


def _tag(valor: str) -> str:
    """Los tags de Pinecone solo admiten letras, números, '_' y '-'."""
    return re.sub(r"[^a-zA-Z0-9_-]", "-", valor)[:120]


def _tags_como_dict(tags: Any) -> dict[str, str]:
    """El SDK devuelve los tags como `IndexTags` (modelo OpenAPI, sin `.keys()`), no como dict."""
    if not tags:
        return {}
    if hasattr(tags, "to_dict"):
        return dict(tags.to_dict())
    return dict(tags)


def _con_reintentos(operacion: Callable[[], T], descripcion: str, intentos: int = 3) -> T:
    """Reintenta errores transitorios de red/servidor con backoff exponencial."""
    for intento in range(1, intentos + 1):
        try:
            return operacion()
        except Exception as exc:  # el SDK lanza distintos tipos según el transporte
            if intento == intentos:
                raise
            espera = 2**intento
            logger.warning("%s falló (%s); reintento %d en %ds", descripcion, exc, intento, espera)
            time.sleep(espera)
    raise AssertionError("inalcanzable")


def conectar(settings: PineconeSettings) -> Pinecone:
    return Pinecone(api_key=settings.api_key.get_secret_value())


def _esperar_listo(pc: Pinecone, nombre: str, timeout_s: int = 120) -> None:
    limite = time.monotonic() + timeout_s
    while time.monotonic() < limite:
        estado = pc.describe_index(nombre).status
        if getattr(estado, "ready", False):
            return
        logger.info("Esperando a que el índice '%s' esté listo...", nombre)
        time.sleep(3)
    raise TimeoutError(f"el índice '{nombre}' no quedó listo en {timeout_s}s")


def asegurar_indice(
    pc: Pinecone, settings: PineconeSettings, dimension: int, embeddings_id: str, *, crear: bool = True
) -> tuple[Index, bool]:
    """Devuelve el índice (creándolo si no existe y `crear` es True) y si fue creado en esta llamada."""
    creado = False
    if not pc.has_index(settings.index_name):
        if not crear:
            raise IndexMismatchError(
                f"el índice '{settings.index_name}' no existe: corré  python setup_index.py  y  python ingest.py"
            )
        logger.info(
            "Creando índice serverless '%s' (dim=%d, metric=%s, %s/%s)",
            settings.index_name,
            dimension,
            settings.metric,
            settings.cloud,
            settings.region,
        )
        pc.create_index(
            name=settings.index_name,
            dimension=dimension,
            metric=settings.metric,
            spec=ServerlessSpec(cloud=settings.cloud, region=settings.region),
            tags={TAG_EMBEDDINGS: _tag(embeddings_id), "proyecto": "orbita-rag"},
        )
        creado = True
    else:
        logger.info("El índice '%s' ya existe: no se vuelve a crear", settings.index_name)

    _esperar_listo(pc, settings.index_name)
    descripcion = pc.describe_index(settings.index_name)

    if descripcion.dimension != dimension:
        raise IndexMismatchError(
            f"el índice '{settings.index_name}' tiene dimensión {descripcion.dimension} y el modelo de "
            f"embeddings genera {dimension}. Usá otro INDEX_NAME o borrá el índice desde la consola de Pinecone."
        )
    if str(descripcion.metric) != settings.metric:
        raise IndexMismatchError(f"el índice usa la métrica {descripcion.metric} y se configuró {settings.metric}")
    tags = _tags_como_dict(getattr(descripcion, "tags", None))
    if tags.get(TAG_EMBEDDINGS) and tags[TAG_EMBEDDINGS] != _tag(embeddings_id):
        raise IndexMismatchError(
            f"el índice se indexó con '{tags[TAG_EMBEDDINGS]}' y el entorno configura '{embeddings_id}'"
        )

    return pc.Index(settings.index_name), creado


def borrar_namespace(index: Index, namespace: str) -> None:
    try:
        index.delete(delete_all=True, namespace=namespace)
        logger.info("Namespace '%s' vaciado", namespace)
    except Exception as exc:  # Pinecone responde 404 si el namespace todavía no existe
        logger.info("No se vació el namespace '%s' (%s)", namespace, str(exc).splitlines()[0])


def upsert_en_lotes(
    index: Index, chunks: list[Document], embeddings: Embeddings, namespace: str, batch_size: int
) -> int:
    """Embebe y sube los chunks de a `batch_size`. El texto viaja en la metadata (clave `text`)."""
    total = 0
    lotes = range(0, len(chunks), batch_size)
    for numero, inicio in enumerate(lotes, start=1):
        lote = chunks[inicio : inicio + batch_size]
        vectores = embeddings.embed_documents([c.page_content for c in lote])
        payload = [
            {"id": chunk.id, "values": vector, "metadata": {**chunk.metadata, TEXT_KEY: chunk.page_content}}
            for chunk, vector in zip(lote, vectores)
        ]
        _con_reintentos(lambda: index.upsert(vectors=payload, namespace=namespace), f"upsert del lote {numero}")
        total += len(payload)
        logger.info("Lote %d/%d: %d vectores subidos", numero, len(lotes), len(payload))
    return total


def borrar_obsoletos(index: Index, namespace: str, ids_vigentes: set[str]) -> int:
    """Elimina los vectores del namespace cuyo id ya no corresponde a ningún chunk del corpus."""
    existentes = [vector_id for pagina in index.list(namespace=namespace) for vector_id in pagina]
    obsoletos = sorted(set(existentes) - ids_vigentes)
    for inicio in range(0, len(obsoletos), 1000):
        index.delete(ids=obsoletos[inicio : inicio + 1000], namespace=namespace)
    if obsoletos:
        logger.info("Eliminados %d vectores obsoletos", len(obsoletos))
    return len(obsoletos)


def estadisticas(index: Index, namespace: str) -> dict[str, Any]:
    stats = index.describe_index_stats()
    resumen = stats.namespaces.get(namespace) if stats.namespaces else None
    return {
        "dimension": stats.dimension,
        "total_vectores": stats.total_vector_count,
        "vectores_en_namespace": resumen.vector_count if resumen else 0,
        "namespaces": sorted(stats.namespaces or {}),
    }


def esperar_conteo(index: Index, namespace: str, esperado: int, timeout_s: int = 60) -> dict[str, Any]:
    """Las estadísticas de Pinecone son eventualmente consistentes: espera a que reflejen la ingesta."""
    limite = time.monotonic() + timeout_s
    stats = estadisticas(index, namespace)
    while stats["vectores_en_namespace"] != esperado and time.monotonic() < limite:
        time.sleep(3)
        stats = estadisticas(index, namespace)
    return stats


def vector_store(index: Index, embeddings: Embeddings, namespace: str) -> PineconeVectorStore:
    return PineconeVectorStore(index=index, embedding=embeddings, text_key=TEXT_KEY, namespace=namespace)
