import json
from collections import Counter

from config import RetrievalSettings
from documents import ChunkMetadata, cargar_documentos, construir_chunks, limpiar_texto


def test_corpus_tiene_markdown_y_entradas_json(chunks):
    doc_ids = {c.metadata["doc_id"] for c in chunks}
    assert "politica_despliegues" in doc_ids
    assert "catalogo_errores#PAY-4091" in doc_ids
    assert len(doc_ids) >= 15


def test_chunks_respetan_tamano_y_tienen_ids_unicos(chunks, settings: RetrievalSettings):
    assert all(c.metadata["n_tokens"] <= settings.chunk_size for c in chunks)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(c.id == c.metadata["chunk_id"] for c in chunks)


def test_metadata_cumple_el_esquema_y_es_compatible_con_pinecone(chunks):
    for chunk in chunks:
        ChunkMetadata.model_validate(chunk.metadata)  # extra="forbid": sin campos inesperados
        for valor in chunk.metadata.values():
            assert valor is not None
            assert isinstance(valor, (str, int, float, bool)) or (
                isinstance(valor, list) and all(isinstance(v, str) for v in valor)
            )


def test_front_matter_define_categoria_y_etiquetas(chunks):
    despliegues = next(c for c in chunks if c.metadata["doc_id"] == "politica_despliegues")
    assert despliegues.metadata["categoria"] == "operaciones"
    assert "canary" in despliegues.metadata["etiquetas"]
    assert despliegues.metadata["titulo"] == "Política de despliegues a producción"
    # El front matter no se indexa como contenido.
    assert all("categoria:" not in c.page_content for c in chunks)


def test_entrada_json_se_renderiza_como_texto(chunks):
    error = next(c for c in chunks if c.metadata["doc_id"] == "catalogo_errores#PAY-4091")
    assert error.metadata["categoria"] == "errores"
    assert "PAY-4091" in error.metadata["etiquetas"]
    assert error.page_content.startswith("Código: PAY-4091")
    assert "Acción recomendada:" in error.page_content


def test_documentos_largos_se_parten_con_overlap(chunks):
    por_doc = Counter(c.metadata["doc_id"] for c in chunks)
    assert max(por_doc.values()) >= 2
    runbook = sorted((c for c in chunks if c.metadata["doc_id"] == "runbook_incidentes"), key=lambda c: c.id)
    # La última oración de un chunk reaparece al comienzo del siguiente.
    final_primero = runbook[0].page_content.split("\n")[-1][-40:]
    assert final_primero in runbook[1].page_content


def test_construccion_es_determinista(settings: RetrievalSettings, chunks):
    otra = construir_chunks(settings)
    assert [c.id for c in otra] == [c.id for c in chunks]
    assert [c.page_content for c in otra] == [c.page_content for c in chunks]


def test_loader_json_generico(tmp_path):
    (tmp_path / "faq.json").write_text(
        json.dumps({"categoria": "faq", "items": [{"id": "f1", "pregunta": "¿Qué es?", "respuesta": "Algo."}]}),
        encoding="utf-8",
    )
    [documento] = cargar_documentos(tmp_path)
    assert documento.doc_id == "faq#f1"
    assert documento.categoria == "faq"
    assert "Respuesta: Algo." in documento.texto


def test_limpiar_texto():
    assert limpiar_texto("a■  b\r\n\r\n\r\n\r\nc  \n") == "a b\n\nc"


def _pdf_minimo(paginas: list[str]) -> bytes:
    """Genera un PDF válido de texto plano (una línea por página) sin dependencias externas."""
    n = len(paginas)
    objetos = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [" + b" ".join(b"%d 0 R" % (4 + 2 * i) for i in range(n)) + b"] /Count %d >>" % n,
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i, texto in enumerate(paginas):
        stream = b"BT /F1 12 Tf 72 720 Td (" + texto.encode("latin-1") + b") Tj ET"
        objetos.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> "
            b"/Contents %d 0 R >>" % (5 + 2 * i)
        )
        objetos.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    salida, offsets = bytearray(b"%PDF-1.4\n"), []
    for numero, objeto in enumerate(objetos, start=1):
        offsets.append(len(salida))
        salida += b"%d 0 obj\n" % numero + objeto + b"\nendobj\n"
    xref = len(salida)
    salida += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objetos) + 1)
    salida += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    salida += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objetos) + 1, xref)
    return bytes(salida)


def test_pdf_conserva_el_numero_de_pagina_en_la_metadata(tmp_path):
    (tmp_path / "manual_sdk.pdf").write_bytes(
        _pdf_minimo(["Instalacion del SDK de pagos.", "Configuracion del webhook de cobros."])
    )
    chunks = construir_chunks(RetrievalSettings(docs_dir=tmp_path))
    assert [c.metadata["pagina"] for c in chunks] == [1, 2]
    assert [c.metadata["seccion"] for c in chunks] == ["Página 1", "Página 2"]
    assert [c.id for c in chunks] == ["manual_sdk::p001::000", "manual_sdk::p002::000"]
    assert "webhook" in chunks[1].page_content
    for chunk in chunks:
        ChunkMetadata.model_validate(chunk.metadata)


def test_markdown_y_json_omiten_pagina_en_lugar_de_mandar_null(chunks):
    # Pinecone rechaza metadata con null: el campo opcional directamente no se envía.
    assert all("pagina" not in c.metadata for c in chunks)
