"""Carga, limpieza y fragmentación del corpus, con un esquema de metadatos estricto.

Formatos soportados:
- Markdown/TXT con front matter opcional (titulo, categoria, etiquetas).
- JSON con una lista de entradas (p. ej. el catálogo de errores): cada entrada es un documento.
- PDF: cada página es un documento y conserva su número de página en la metadata.

La salida es la misma para todos: una lista de `Document` con id determinista y metadata
validada por `ChunkMetadata`. Tanto la ingesta a Pinecone como el BM25 local se arman con
esta función, así ambos retrievers ven exactamente los mismos chunks.
"""

import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, ConfigDict, Field

from config import RetrievalSettings

logger = logging.getLogger(__name__)

EXTENSIONES_SOPORTADAS = (".md", ".txt", ".json", ".pdf")

_CARACTERES_BASURA = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​-‍﻿■□▪●]")
_ENCABEZADO_MD = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$", re.MULTILINE)
_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)

# Jerarquía del splitter: fin de párrafo u oración > línea > palabra > carácter.
# La pieza mínima es la oración para que el overlap repita oraciones completas.
SEPARADORES = [r"\n{2,}|(?<=\D[.!?:;])[ \t\n]+", r"\n", r" ", r""]


class ChunkMetadata(BaseModel):
    """Interfaz de metadatos: un único esquema para todo el corpus (evita el "schema drift").

    Todos los tipos son aceptados por Pinecone (string, número, booleano o lista de strings)
    y ningún campo puede quedar en null: los opcionales se omiten.
    """

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    doc_id: str = Field(description="Documento lógico: nombre de archivo sin extensión, o archivo#clave en JSON")
    fuente: str = Field(description="Archivo de origen")
    titulo: str
    categoria: str
    etiquetas: list[str] = Field(default_factory=list)
    seccion: str
    pagina: int | None = Field(default=None, description="Solo para PDF")
    chunk_index: int = Field(ge=0)
    n_tokens: int = Field(gt=0)

    def to_metadata(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


@dataclass(frozen=True)
class DocumentoFuente:
    doc_id: str
    fuente: str
    titulo: str
    categoria: str
    etiquetas: tuple[str, ...]
    texto: str
    pagina: int | None = None


def limpiar_texto(texto: str) -> str:
    """Normaliza el texto sin destruir la estructura que usa el splitter (párrafos y encabezados)."""
    texto = unicodedata.normalize("NFC", texto)
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")
    texto = _CARACTERES_BASURA.sub("", texto)
    texto = re.sub(r"[ \t]+\n", "\n", texto)
    texto = re.sub(r"(?<=\S)[ \t]{2,}", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


# --- Loaders -----------------------------------------------------------------------------


def _parsear_front_matter(texto: str) -> tuple[dict[str, Any], str]:
    """Front matter mínimo `clave: valor` / `clave: [a, b]`, sin depender de PyYAML."""
    match = _FRONT_MATTER.match(texto)
    if not match:
        return {}, texto
    datos: dict[str, Any] = {}
    for linea in match.group(1).splitlines():
        if ":" not in linea:
            continue
        clave, valor = (parte.strip() for parte in linea.split(":", 1))
        if valor.startswith("[") and valor.endswith("]"):
            datos[clave] = [v.strip() for v in valor[1:-1].split(",") if v.strip()]
        else:
            datos[clave] = valor
    return datos, texto[match.end() :]


def _cargar_markdown(ruta: Path) -> list[DocumentoFuente]:
    meta, cuerpo = _parsear_front_matter(ruta.read_text(encoding="utf-8").replace("\r\n", "\n"))
    texto = limpiar_texto(cuerpo)
    h1 = next((m.group(2) for m in _ENCABEZADO_MD.finditer(texto) if len(m.group(1)) == 1), None)
    return [
        DocumentoFuente(
            doc_id=ruta.stem,
            fuente=ruta.name,
            titulo=meta.get("titulo") or h1 or ruta.stem.replace("_", " ").capitalize(),
            categoria=meta.get("categoria", "general"),
            etiquetas=tuple(meta.get("etiquetas", [])),
            texto=texto,
        )
    ]


_ETIQUETAS_JSON = {
    "codigo": "Código",
    "titulo": "Título",
    "http_status": "Estado HTTP",
    "descripcion": "Descripción",
    "causa_probable": "Causa probable",
    "accion_recomendada": "Acción recomendada",
}


def _renderizar_entrada(entrada: dict[str, Any]) -> str:
    """Convierte una entrada JSON en texto legible 'Clave: valor' (lo que se embebe y se indexa)."""
    lineas = []
    for clave, valor in entrada.items():
        etiqueta = _ETIQUETAS_JSON.get(clave, clave.replace("_", " ").capitalize())
        lineas.append(f"{etiqueta}: {', '.join(map(str, valor)) if isinstance(valor, list) else valor}")
    return "\n".join(lineas)


def _cargar_json(ruta: Path) -> list[DocumentoFuente]:
    """JSON con metadatos a nivel raíz y una lista de entradas; cada entrada es un documento."""
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    entradas = next((v for v in datos.values() if isinstance(v, list) and v and isinstance(v[0], dict)), None)
    if entradas is None:
        raise ValueError(f"{ruta.name}: se esperaba un objeto con una lista de entradas")

    documentos = []
    for posicion, entrada in enumerate(entradas):
        clave = str(entrada.get("codigo") or entrada.get("id") or posicion)
        titulo = entrada.get("titulo", clave)
        etiquetas = [clave] + [str(entrada[c]) for c in ("servicio", "http_status") if c in entrada]
        documentos.append(
            DocumentoFuente(
                doc_id=f"{ruta.stem}#{clave}",
                fuente=ruta.name,
                titulo=f"{clave} — {titulo}" if clave != titulo else titulo,
                categoria=datos.get("categoria", "general"),
                etiquetas=tuple(etiquetas),
                texto=limpiar_texto(_renderizar_entrada(entrada)),
            )
        )
    return documentos


def _cargar_pdf(ruta: Path) -> list[DocumentoFuente]:
    from pypdf import PdfReader

    lector = PdfReader(ruta)
    titulo = (lector.metadata.title if lector.metadata else None) or ruta.stem.replace("_", " ").capitalize()
    documentos = []
    for numero, pagina in enumerate(lector.pages, start=1):
        texto = limpiar_texto(pagina.extract_text() or "")
        if texto:
            documentos.append(
                DocumentoFuente(ruta.stem, ruta.name, titulo, "general", (), texto, pagina=numero)
            )
    return documentos


_LOADERS = {".md": _cargar_markdown, ".txt": _cargar_markdown, ".json": _cargar_json, ".pdf": _cargar_pdf}


def cargar_documentos(docs_dir: Path) -> list[DocumentoFuente]:
    if not docs_dir.is_dir():
        raise FileNotFoundError(f"no existe la carpeta de documentos: {docs_dir}")
    documentos: list[DocumentoFuente] = []
    for ruta in sorted(docs_dir.iterdir()):
        if ruta.is_file() and ruta.suffix.lower() in _LOADERS:
            nuevos = [d for d in _LOADERS[ruta.suffix.lower()](ruta) if d.texto]
            documentos.extend(nuevos)
            logger.debug("Cargado %s: %d documentos", ruta.name, len(nuevos))
    return documentos


# --- Chunking ----------------------------------------------------------------------------


def crear_splitter(settings: RetrievalSettings) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=settings.token_encoding,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=SEPARADORES,
        is_separator_regex=True,
    )


def _posiciones(texto: str, chunks: list[str]) -> list[int]:
    """Posición de cada chunk en el texto (add_start_index falla cuando el overlap se mide en tokens)."""
    posiciones, desde = [], 0
    for chunk in chunks:
        pos = texto.find(chunk, desde)
        posiciones.append(pos)
        desde = pos + 1
    return posiciones


def _seccion_dominante(texto: str, inicio: int, fin: int, por_defecto: str) -> str:
    """El encabezado Markdown que más caracteres aporta al chunk [inicio, fin)."""
    secciones = [(m.start(), m.group(2)) for m in _ENCABEZADO_MD.finditer(texto)]
    mejor, mejor_solape = por_defecto, 0
    for i, (pos, titulo) in enumerate(secciones):
        fin_seccion = secciones[i + 1][0] if i + 1 < len(secciones) else fin
        solape = min(fin, fin_seccion) - max(inicio, pos)
        if solape > mejor_solape:
            mejor, mejor_solape = titulo, solape
    return mejor


def fragmentar(documento: DocumentoFuente, settings: RetrievalSettings) -> list[Document]:
    splitter = crear_splitter(settings)
    tokenizer = tiktoken.get_encoding(settings.token_encoding)
    textos = splitter.split_text(documento.texto)
    prefijo_id = documento.doc_id if documento.pagina is None else f"{documento.doc_id}::p{documento.pagina:03d}"

    chunks = []
    for indice, (texto, inicio) in enumerate(zip(textos, _posiciones(documento.texto, textos))):
        seccion = (
            f"Página {documento.pagina}"
            if documento.pagina is not None
            else _seccion_dominante(documento.texto, inicio, inicio + len(texto), documento.titulo)
        )
        metadata = ChunkMetadata(
            chunk_id=f"{prefijo_id}::{indice:03d}",
            doc_id=documento.doc_id,
            fuente=documento.fuente,
            titulo=documento.titulo,
            categoria=documento.categoria,
            etiquetas=list(documento.etiquetas),
            seccion=seccion,
            pagina=documento.pagina,
            chunk_index=indice,
            n_tokens=len(tokenizer.encode(texto)),
        )
        # Id determinista: reingestar el mismo documento pisa sus vectores en lugar de duplicarlos.
        chunks.append(Document(id=metadata.chunk_id, page_content=texto, metadata=metadata.to_metadata()))
    return chunks


def construir_chunks(settings: RetrievalSettings) -> list[Document]:
    """Corpus completo listo para indexar: carga + limpieza + chunking + metadata."""
    documentos = cargar_documentos(settings.docs_dir)
    if not documentos:
        raise FileNotFoundError(f"no hay documentos {EXTENSIONES_SOPORTADAS} en {settings.docs_dir}")
    chunks = [chunk for documento in documentos for chunk in fragmentar(documento, settings)]
    ids = [c.id for c in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("hay chunk_ids duplicados en el corpus: revisá doc_ids repetidos")
    logger.info("Corpus: %d documentos -> %d chunks", len(documentos), len(chunks))
    return chunks
