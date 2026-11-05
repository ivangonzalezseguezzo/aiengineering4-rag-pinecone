"""Recuperador híbrido: BM25 (léxico) + búsqueda vectorial en Pinecone, fusionados con RRF.

    consulta ──► BM25Retriever (k candidatos) ──┐
             └─► retriever vectorial (k)  ──────┴─► EnsembleRetriever (Reciprocal Rank Fusion) ──► top_k

- BM25 encuentra términos exactos que los embeddings diluyen: códigos de error (PAY-4091),
  headers (Idempotency-Key), nombres propios (Sirena, Argo Rollouts).
- La búsqueda vectorial encuentra paráfrasis: "renovar contraseñas" ≈ "rotar secretos".
- EnsembleRetriever combina los rankings por POSICIÓN (RRF), no por score: los scores de
  BM25 y de coseno viven en escalas incomparables y sumarlos directamente sería un error.

El backend vectorial es intercambiable: Pinecone en producción, o un InMemoryVectorStore con
los mismos chunks y embeddings para desarrollar y evaluar sin credenciales.
"""

import asyncio
import logging
import time
import warnings
from collections.abc import Sequence
from typing import Literal, Protocol

from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
from pydantic import ValidationError

from lexico import tokenizar
from llm import es_reintentable, get_chat_model, resumir_error, settings_from_env
from prompts import rag_prompt
from schemas import RESPUESTA_SIN_CONTEXTO, Fuente, RespuestaGenerada, RespuestaRAG

with warnings.catch_warnings():
    # langchain-community anuncia su "sunset"; BM25Retriever sigue siendo el retriever léxico de LangChain.
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from langchain_community.retrievers import BM25Retriever

logger = logging.getLogger(__name__)

Modo = Literal["hibrido", "vectorial", "bm25"]
MODOS: tuple[Modo, ...] = ("hibrido", "vectorial", "bm25")
MAX_ATTEMPTS = 3
# Backoff exponencial con jitter entre intentos: ~4 s y ~8 s, igual que en las pre-entregas 2 y 3.
BACKOFF = {"initial": 4, "max": 30, "exp_base": 2, "jitter": 1}


def es_reintentable_generacion(exc: BaseException) -> bool:
    """Política de reintento de la generación (ver tabla de errores del README).

    - Salida que no cumple RespuestaGenerada (JSON inválido o incoherente): sí.
    - Error del proveedor: solo si es transitorio y no es la cuota diaria (`llm.es_reintentable`).
    """
    reintentar = isinstance(exc, (OutputParserException, ValidationError)) or es_reintentable(exc)
    logger.warning(
        "Intento de generación fallido (%s: %s) -> %s",
        type(exc).__name__,
        resumir_error(exc),
        "reintentable" if reintentar else "no reintentable",
    )
    return reintentar


class BackendVectorial(Protocol):
    nombre: str

    def retriever(self, k: int, categoria: str | None) -> BaseRetriever: ...


class PineconeBackend:
    """Búsqueda vectorial en Pinecone, acotada al namespace y con filtro de metadata opcional."""

    nombre = "pinecone"

    def __init__(self, store: VectorStore, namespace: str) -> None:
        self.store = store
        self.namespace = namespace

    def retriever(self, k: int, categoria: str | None) -> BaseRetriever:
        search_kwargs: dict = {"k": k, "namespace": self.namespace}
        if categoria:
            # Hard filter: se aplica dentro de Pinecone, antes de rankear por similitud.
            search_kwargs["filter"] = {"categoria": {"$eq": categoria}}
        return self.store.as_retriever(search_kwargs=search_kwargs)


class MemoriaBackend:
    """Mismos chunks y embeddings en memoria: permite evaluar y probar sin Pinecone."""

    nombre = "memoria"

    def __init__(self, chunks: Sequence[Document], embeddings: Embeddings) -> None:
        self.store = InMemoryVectorStore(embeddings)
        self.store.add_documents(list(chunks), ids=[c.id for c in chunks])

    def retriever(self, k: int, categoria: str | None) -> BaseRetriever:
        search_kwargs: dict = {"k": k}
        if categoria:
            search_kwargs["filter"] = lambda doc: doc.metadata.get("categoria") == categoria
        return self.store.as_retriever(search_kwargs=search_kwargs)


class RAGSystem:
    """Encapsula el EnsembleRetriever (BM25 + vectorial) y la generación grounded opcional."""

    def __init__(
        self,
        chunks: Sequence[Document],
        backend: BackendVectorial,
        *,
        top_k: int = 5,
        candidatos: int = 10,
        pesos: Sequence[float] = (0.5, 0.5),
        model: BaseChatModel | None = None,
    ) -> None:
        if not chunks:
            raise ValueError("RAGSystem necesita el corpus de chunks para construir el índice BM25")
        self.chunks = list(chunks)
        self.backend = backend
        self.top_k = top_k
        self.candidatos = candidatos
        self.pesos = list(pesos)
        self._model = model
        self._bm25: dict[str | None, BM25Retriever] = {}
        self._retrievers: dict[tuple[Modo, str | None], BaseRetriever] = {}
        self._cadena: Runnable | None = None

    # --- Recuperación -------------------------------------------------------------------

    def _bm25_retriever(self, categoria: str | None) -> BM25Retriever:
        if categoria not in self._bm25:
            corpus = [c for c in self.chunks if categoria is None or c.metadata.get("categoria") == categoria]
            if not corpus:
                raise ValueError(f"no hay documentos de la categoría '{categoria}'")
            self._bm25[categoria] = BM25Retriever.from_documents(corpus, preprocess_func=tokenizar, k=self.candidatos)
        return self._bm25[categoria]

    def retriever(self, modo: Modo = "hibrido", categoria: str | None = None) -> BaseRetriever:
        clave = (modo, categoria)
        if clave not in self._retrievers:
            match modo:
                case "bm25":
                    retriever = self._bm25_retriever(categoria)
                case "vectorial":
                    retriever = self.backend.retriever(self.candidatos, categoria)
                case "hibrido":
                    retriever = EnsembleRetriever(
                        retrievers=[self._bm25_retriever(categoria), self.backend.retriever(self.candidatos, categoria)],
                        weights=self.pesos,
                        # Deduplica por chunk_id: el mismo chunk puede venir de los dos retrievers.
                        id_key="chunk_id",
                    )
                case _:
                    raise ValueError(f"modo desconocido: {modo}")
            self._retrievers[clave] = retriever
        return self._retrievers[clave]

    def buscar(self, query: str, *, modo: Modo = "hibrido", categoria: str | None = None) -> list[Document]:
        """Top-k documentos para la consulta (por defecto, combinando léxico y semántico)."""
        if not query or not query.strip():
            raise ValueError("la consulta no puede estar vacía")
        return self.retriever(modo, categoria).invoke(query.strip())[: self.top_k]

    async def abuscar(self, query: str, *, modo: Modo = "hibrido", categoria: str | None = None) -> list[Document]:
        # BM25 y el cliente de Pinecone son sincrónicos: se ejecutan en un hilo para no bloquear el event loop.
        return await asyncio.to_thread(self.buscar, query, modo=modo, categoria=categoria)

    # --- Generación ---------------------------------------------------------------------

    def _cadena_generacion(self) -> Runnable:
        if self._cadena is None:
            intentos = MAX_ATTEMPTS
            if self._model is None:
                # Los reintentos los hace la cadena (el SDK no reintenta): 1 intento + LLM_MAX_RETRIES.
                intentos = settings_from_env().max_retries + 1
                self._model = get_chat_model()
            # json_schema = salida estructurada nativa del proveedor (igual que en la PE2). En Anthropic
            # el default de with_structured_output es tool calling forzado, que los modelos 5.5 rechazan.
            estructurado = self._model.with_structured_output(RespuestaGenerada, method="json_schema")
            self._cadena = rag_prompt | estructurado.with_retry(
                retry_if_exception_type=es_reintentable_generacion,
                stop_after_attempt=intentos,
                wait_exponential_jitter=True,
                exponential_jitter_params=BACKOFF,
            )
        return self._cadena

    async def aresponder(
        self, query: str, *, categoria: str | None = None, documentos: list[Document] | None = None
    ) -> RespuestaRAG:
        """Flujo RAG completo: recuperación híbrida + respuesta grounded con fuentes.

        Si ya se recuperaron los documentos (p. ej. para mostrarlos), se pasan en `documentos`
        y no se vuelve a consultar Pinecone.
        """
        inicio = time.perf_counter()
        if documentos is None:
            documentos = await self.abuscar(query, categoria=categoria)
        if not documentos:
            return RespuestaRAG(
                pregunta=query,
                respuesta=f"{RESPUESTA_SIN_CONTEXTO} No se recuperaron documentos.",
                respuesta_encontrada=False,
                documentos_recuperados=0,
                latencia_s=round(time.perf_counter() - inicio, 3),
            )

        contexto = "\n\n---\n\n".join(
            f"[{n}] Fuente: {d.metadata['fuente']} — {d.metadata['seccion']}\n{d.page_content}"
            for n, d in enumerate(documentos, start=1)
        )
        generada: RespuestaGenerada = await self._cadena_generacion().ainvoke({"contexto": contexto, "pregunta": query})

        fuentes = [
            Fuente(
                fragmento=n,
                doc_id=documentos[n - 1].metadata["doc_id"],
                fuente=documentos[n - 1].metadata["fuente"],
                seccion=documentos[n - 1].metadata["seccion"],
                chunk_id=documentos[n - 1].metadata["chunk_id"],
            )
            for n in generada.fragmentos_usados
            if 1 <= n <= len(documentos)
        ]
        return RespuestaRAG(
            pregunta=query,
            respuesta=generada.respuesta,
            respuesta_encontrada=generada.respuesta_encontrada and bool(fuentes),
            fuentes=fuentes,
            documentos_recuperados=len(documentos),
            latencia_s=round(time.perf_counter() - inicio, 3),
        )


def construir_rag_system(backend: Literal["pinecone", "memoria"] = "pinecone") -> RAGSystem:
    """Arma el RAGSystem desde la configuración del entorno.

    - pinecone: abre el índice existente (no lo crea) y verifica que el namespace tenga vectores.
    - memoria: indexa el corpus en un InMemoryVectorStore; no necesita PINECONE_API_KEY.
    """
    from config import pinecone_settings_from_env, retrieval_settings_from_env
    from documents import construir_chunks
    from embeddings import dimension, embeddings_settings_from_env, get_embeddings
    from pinecone_store import asegurar_indice, conectar, estadisticas, vector_store

    settings = retrieval_settings_from_env()
    emb_settings = embeddings_settings_from_env()
    embeddings = get_embeddings(emb_settings)
    chunks = construir_chunks(settings)

    if backend == "memoria":
        vectorial: BackendVectorial = MemoriaBackend(chunks, embeddings)
    else:
        pc_settings = pinecone_settings_from_env()
        index, _ = asegurar_indice(
            conectar(pc_settings), pc_settings, dimension(embeddings), emb_settings.identificador, crear=False
        )
        stats = estadisticas(index, pc_settings.namespace)
        if stats["vectores_en_namespace"] == 0:
            raise RuntimeError(f"el namespace '{pc_settings.namespace}' está vacío: corré  python ingest.py")
        if stats["vectores_en_namespace"] != len(chunks):
            logger.warning(
                "Pinecone tiene %d vectores y el corpus local %d chunks: volvé a correr python ingest.py",
                stats["vectores_en_namespace"],
                len(chunks),
            )
        vectorial = PineconeBackend(vector_store(index, embeddings, pc_settings.namespace), pc_settings.namespace)

    logger.info(
        "RAGSystem listo (backend=%s, %d chunks, pesos bm25/vectorial=%s)", vectorial.nombre, len(chunks), settings.pesos
    )
    return RAGSystem(chunks, vectorial, top_k=settings.top_k, candidatos=settings.candidatos, pesos=settings.pesos)
