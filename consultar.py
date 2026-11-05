"""Consulta el sistema RAG desde la terminal.

Uso:
    python consultar.py "¿Qué significa PAY-4091?"                      # top-5 híbrido
    python consultar.py "¿Qué significa PAY-4091?" --modo bm25          # solo léxico
    python consultar.py "rotación de secretos" --categoria seguridad    # con filtro de metadata
    python consultar.py "¿Qué significa PAY-4091?" --responder          # además genera la respuesta con el LLM
    python consultar.py "..." --backend memoria                          # sin Pinecone
"""

import argparse
import asyncio
import sys

from dotenv import load_dotenv
from langchain_core.exceptions import ModelError, OutputParserException
from pydantic import ValidationError

from llm import es_reintentable, resumir_error
from logging_setup import configurar_logging
from rag_system import MODOS, construir_rag_system


async def main() -> int:
    parser = argparse.ArgumentParser(description="Consulta al recuperador híbrido")
    parser.add_argument("pregunta")
    parser.add_argument("--modo", choices=MODOS, default="hibrido")
    parser.add_argument("--categoria", default=None, help="filtra por la metadata 'categoria'")
    parser.add_argument("--backend", choices=["pinecone", "memoria"], default="pinecone")
    parser.add_argument("--responder", action="store_true", help="genera la respuesta con el LLM")
    args = parser.parse_args()

    rag = construir_rag_system(args.backend)
    documentos = await rag.abuscar(args.pregunta, modo=args.modo, categoria=args.categoria)

    print(f"\nTop {len(documentos)} ({args.modo}, backend {args.backend}):")
    for posicion, doc in enumerate(documentos, start=1):
        meta = doc.metadata
        extracto = " ".join(doc.page_content.split())[:160]
        print(f"\n {posicion}. [{meta['categoria']}] {meta['fuente']} › {meta['seccion']}  ({meta['chunk_id']})")
        print(f"    {extracto}...")

    if args.responder:
        # Reutiliza el top-k que se acaba de mostrar: no repite la consulta a Pinecone.
        respuesta = await rag.aresponder(args.pregunta, documentos=documentos)
        print("\nRespuesta: " + respuesta.respuesta)
        for fuente in respuesta.fuentes:
            print(f"  [{fuente.fragmento}] {fuente.fuente} › {fuente.seccion}")
    return 0


if __name__ == "__main__":
    load_dotenv()
    configurar_logging()
    try:
        sys.exit(asyncio.run(main()))
    except ValidationError as exc:
        sys.exit("Configuración inválida: " + exc.errors()[0]["msg"])
    except (RuntimeError, ValueError) as exc:  # índice inexistente/incompatible, namespace vacío, categoría
        sys.exit(f"Error: {exc}")
    except (ModelError, OutputParserException) as exc:  # --responder: cupo, API key, modelo o salida inválida
        transitorio = es_reintentable(exc)  # False para la cuota diaria y los errores permanentes
        pista = "Es transitorio: probá de nuevo más tarde." if transitorio else "Revisá la configuración del LLM en el .env."
        sys.exit(f"Error del LLM ({type(exc).__name__}): {resumir_error(exc)}\n{pista}")
