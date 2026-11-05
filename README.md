# Sistema RAG Escalable en la Nube con Pinecone

Pre-entrega 4 de **AI Engineering (Coderhouse)**: el RAG local de la pre-entrega 3 pasa a **Pinecone Serverless**. Incorpora:
- un **recuperador híbrido** (BM25 + vectorial, con `EnsembleRetriever`);
- un **pipeline de ingesta por lotes con metadatos estructurados**;
- un **script de evaluación** que mide Recall@5 y Precision@5 sobre un golden set.

El corpus es la documentación de ingeniería de *Órbita Pagos* (fintech ficticia):
- 8 documentos Markdown: incidentes, despliegues, guardias, APIs, arquitectura, observabilidad, seguridad y bases de datos;
- un catálogo JSON con 13 códigos de error.

En total son 21 documentos lógicos y 35 chunks.

## Arquitectura

```
                         INGESTA (setup_index.py + ingest.py)
 data/docs/*.md  ─┐
 data/docs/*.json ├─► limpieza ─► chunks ≤500 tokens ─► ChunkMetadata ─► embeddings ─► upsert por lotes
 data/docs/*.pdf ─┘   (front matter, (overlap 75,       (esquema único,   (por lotes)   (100 vectores/lote,
                       entradas JSON, cortes en          texto incluido)                 ids deterministas)
                       páginas PDF)   oraciones)                                                 │
                                                                                                 ▼
                                                                   Pinecone Serverless (aws/us-east-1)
                                                                   índice orbita-handbook · namespace handbook-v1

                         CONSULTA (RAGSystem)
 pregunta ─┬─► BM25Retriever (tokenizador en español, 10 candidatos) ──┐
           │                                                           ├─► EnsembleRetriever (RRF, 0.5/0.5,
           └─► PineconeVectorStore (namespace + filtro opcional, 10) ──┘      dedup por chunk_id) ─► top-5
                                                                                     │
                                                         (opcional) LLM grounded ─► respuesta + fuentes
```

## Estructura

```
├── data/docs/              # Corpus: 8 .md con front matter + catalogo_errores.json
├── eval/golden_set.json    # Benchmark: 10 preguntas con su documento fuente conocido
├── config.py               # PineconeSettings + RetrievalSettings (Pydantic, desde el .env)
├── documents.py            # Loaders MD/JSON/PDF, limpieza, chunking y esquema ChunkMetadata
├── embeddings.py           # Fábrica de embeddings, dimensión real y chequeo de truncamiento
├── lexico.py               # Tokenizador para BM25 (minúsculas, sin tildes, códigos, stopwords)
├── pinecone_store.py       # Índice serverless, upsert por lotes, borrado de obsoletos, stats
├── setup_index.py          # Script: verifica si el índice existe y lo crea si no
├── ingest.py               # Script: pipeline de ingesta completo
├── rag_system.py           # Clase RAGSystem (EnsembleRetriever) + backends Pinecone / memoria
├── metrics.py              # Recall@k, Precision@k, Hit@k, MRR
├── evaluate.py             # Script de evaluación: compara BM25, vectorial e híbrido
├── consultar.py            # CLI para consultar (y opcionalmente generar la respuesta con el LLM)
├── schemas.py / prompts.py # Contratos Pydantic y prompt grounded
├── llm.py                  # Fábrica del chat model, sampling, cupo y reintentos (igual que en las PE2, PE3 y PE5)
├── reports/                # Reportes JSON de evaluate.py
└── tests/                  # Tests offline con pytest (sin Pinecone ni API keys)
```

## Cómo replicar el índice de Pinecone

1. **Cuenta y API key**: creá una cuenta gratuita en [app.pinecone.io](https://app.pinecone.io). Copiá la API key desde *API Keys*.
2. **Entorno** (Python 3.12 o superior):
   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate    |    Linux/macOS: source .venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env      # completá PINECONE_API_KEY (y la key del LLM si vas a usar --responder)
   ```
3. **Índice serverless**: el script verifica si existe y, si no, lo crea con la dimensión del modelo de embeddings configurado, métrica coseno y `ServerlessSpec(aws, us-east-1)`.
   ```bash
   python setup_index.py
   ```
4. **Ingesta**: fragmenta `data/docs`, genera los embeddings y sube los vectores en lotes de 100 al namespace `handbook-v1`.
   ```bash
   python ingest.py            # idempotente: se puede volver a correr sin duplicar
   python ingest.py --reset    # vacía el namespace antes de subir
   ```
   Al final se imprime un resumen: chunks, tokens, categorías, vectores subidos y estadísticas del índice. El esperado es 35 vectores en el namespace.
5. **Evaluación**:
   ```bash
   python evaluate.py
   ```

La primera ejecución descarga el modelo de embeddings (~470 MB). En el plan gratuito, Pinecone crea el índice en unos segundos.

## Decisiones de diseño

### Infraestructura (Pinecone Serverless)
- **El índice se crea una sola vez.** `asegurar_indice` consulta `has_index` antes de crear y espera a que el índice quede `ready`.
- **La dimensión no se escribe a mano.** Se mide embebiendo un texto de prueba: 384 con el modelo local, 1536 con `text-embedding-3-small`. Así es imposible el *mismatch de dimensiones*. Si el índice existe con otra dimensión u otra métrica, el script se detiene con un mensaje claro.
- **El índice recuerda su modelo.** Guarda como *tag* el modelo de embeddings con el que se creó y se rechaza abrirlo con otro, aunque la dimensión coincida. Indexar con un modelo y consultar con otro devuelve ruido.
- **Namespace**: el corpus vive en `handbook-v1`, lo que permite cargar una versión nueva del corpus (o los datos de otro tenant) en el mismo índice sin mezclarlos.
- **Credenciales**: solo por `.env`, validadas con Pydantic. `PineconeSettings` no exige la key cuando se evalúa con `--backend memoria`.

### Ingesta y metadatos
- **Tres formatos**:
  - **Markdown con front matter** (`titulo`, `categoria`, `etiquetas`);
  - **JSON**: cada código de error es un documento propio, con id `catalogo_errores#PAY-4091`;
  - **PDF**: cada página conserva su número en la metadata `pagina`.
- **Chunking**: `RecursiveCharacterTextSplitter` medido en tokens (`tiktoken`), con 500 tokens y 75 de overlap, cortando en oraciones como en la PE3. Se eligió 500 dentro del rango de 500 a 800 que sugiere la consigna porque es lo que entra completo en la ventana de 512 tokens del modelo local. `ingest.py` verifica con el tokenizador del propio modelo que ningún chunk se trunque; el máximo actual es 503. Con embeddings de OpenAI o Gemini se puede subir `RAG_CHUNK_SIZE`.
- **Esquema de metadatos único (`ChunkMetadata`, `extra="forbid"`)**: `chunk_id`, `doc_id`, `fuente`, `titulo`, `categoria`, `etiquetas`, `seccion`, `pagina` (solo en PDF), `chunk_index` y `n_tokens`. Así se evita el *schema drift*. Ningún valor es `null`, porque Pinecone no lo acepta: los opcionales se omiten.
- **El texto viaja en la metadata** (clave `text`): una sola consulta a Pinecone trae vector, contenido y fuente, sin una base relacional aparte.
- **Upsert por lotes de 100 con reintentos.** Los ids son deterministas (`doc_id::NNN`), así que reingestar actualiza en lugar de duplicar. Al final se listan los ids del namespace y se borran los que ya no corresponden a ningún chunk (`borrar_obsoletos`).

### Recuperador híbrido (`RAGSystem`)
- **BM25** (`BM25Retriever`) encuentra términos exactos que los embeddings diluyen: `PAY-4091`, `Idempotency-Key`, `DB-1105`, "Sirena", "14,4".
- **Vectorial** (`PineconeVectorStore`) encuentra paráfrasis: "renovar contraseñas" ≈ "rotar secretos".
- **EnsembleRetriever** fusiona los rankings con *Reciprocal Rank Fusion*:
  - combina **posiciones, no scores**, porque un score BM25 de 12 y una similitud coseno de 0,85 no son comparables;
  - deduplica por `chunk_id`, porque el mismo chunk puede venir de los dos lados;
  - cada retriever aporta 10 candidatos y se devuelven los 5 mejores.
- **Tokenizador de BM25 propio** (`lexico.py`). El de LangChain por defecto es `str.split()`: distingue mayúsculas y deja la puntuación pegada ("PAY-4091?" ≠ "PAY-4091"). El propio pasa a minúsculas, quita tildes, conserva los códigos con guion y también sus partes, y descarta stopwords del español.
- **Filtro de metadata opcional** (`categoria`): se aplica dentro de Pinecone con `{"categoria": {"$eq": ...}}` y BM25 se construye solo sobre esa categoría.
- **Backend intercambiable**:
  - `PineconeBackend` en producción;
  - `MemoriaBackend`, un `InMemoryVectorStore` con los mismos chunks y embeddings, para desarrollar, testear y evaluar sin credenciales.
- **Generación opcional** (`aresponder`): respuesta grounded con salida estructurada y fuentes verificadas, igual que en la PE3. Se usa con `consultar.py --responder`.
  - `with_structured_output(RespuestaGenerada, method="json_schema")`: salida estructurada nativa de cada proveedor, como en la PE2. El método va explícito porque en Anthropic el default es *tool calling* forzado, que los modelos 5.5 rechazan con un 400.
  - `.with_retry()` con la misma política que las PE2, PE3 y PE5: se reintentan la salida fuera de esquema y los errores transitorios del proveedor; nunca la cuota diaria ni los errores permanentes.

### Modelos por defecto (IDs reales, verificados el 2026-11-04)

Todos se pueden cambiar desde el `.env` sin tocar código.

| Uso | Proveedor | Modelo por defecto | Variable | Fuente |
|---|---|---|---|---|
| Embeddings | HuggingFace (local) | `intfloat/multilingual-e5-small` (384 dim) | `EMBEDDINGS_MODEL` | [Hugging Face](https://huggingface.co/intfloat/multilingual-e5-small) |
| Embeddings | OpenAI | `text-embedding-3-small` (1536 dim) | `EMBEDDINGS_MODEL` | [OpenAI](https://platform.openai.com/docs/guides/embeddings) |
| Embeddings | Gemini | `gemini-embedding-001` | `EMBEDDINGS_MODEL` | [Google AI](https://ai.google.dev/gemini-api/docs/embeddings) |
| LLM (`--responder`) | Gemini | `gemini-3.5-flash-lite` (respaldo: `gemini-3.1-flash-lite`) | `GEMINI_MODEL` | [Google AI](https://ai.google.dev/gemini-api/docs/models) |
| LLM (`--responder`) | OpenAI | `gpt-4o-mini` (deprecado: se da de baja el 2027-01-14, reemplazo `gpt-6-luna`) | `OPENAI_MODEL` | [OpenAI](https://developers.openai.com/api/docs/deprecations) |
| LLM (`--responder`) | Anthropic | `claude-sonnet-5-5` (retiro no antes de 2027-09-28) | `ANTHROPIC_MODEL` | [Anthropic](https://platform.claude.com/docs/en/about-claude/models/overview) |

Son los mismos modelos por defecto que en las pre-entregas 2, 3 y 5.

**Cupo del plan gratuito de Gemini.** El límite es por proyecto **y por modelo**, y el cupo diario se reinicia a la medianoche del Pacífico (≈ 05:00 ART). Valores del panel de AI Studio en noviembre de 2026:

| Familia | Modelos | Requests/min | Tokens/min | Requests/día |
|---|---|---|---|---|
| Flash | `gemini-3.8-flash`, `gemini-3.6-flash`, `gemini-3-flash`, `gemini-2.5-flash` | 5 | 250K | 20 |
| **Flash Lite** | **`gemini-3.5-flash-lite`**, `gemini-3.1-flash-lite` | 15 | 250K | 500 |

Por eso el default es Flash Lite. `LLM_REQUESTS_PER_MINUTE=12` activa un `InMemoryRateLimiter` compartido por todos los modelos del proceso. La recuperación y `evaluate.py` no consumen cupo del LLM: solo lo usa `consultar.py --responder` (una request por pregunta).

**Razonamiento de Gemini 3 (`LLM_THINKING_LEVEL`).** El valor por defecto es `low`, probado con `gemini-3.5-flash-lite`. Si se cambia a `gemini-3.6-flash`, hay que usar `medium` o `high`, que son los niveles que acepta ese modelo.

**Sampling por proveedor.** `LLM_TEMPERATURE` es opcional y se traduce así (`llm.parametros_sampling`):

| Proveedor | `temperature` | `top_p` / `top_k` |
|---|---|---|
| OpenAI, Gemini 2.x | Se envía si está definida; si no, rige el default del modelo | No se envían |
| Gemini 3.x (incluye el default) | **No se envía**: Google la declaró deprecada e ignorada desde `gemini-3.6-flash` y `gemini-3.5-flash-lite`. Si está definida, se loguea un aviso | No se envían |
| Anthropic | **No se envía**: desde Claude Opus 4.7 (incluida la familia 5.5) la API responde 400. Si está definida, se loguea un aviso en vez de ignorarla en silencio | No se envían (mismo motivo) |

Para que el comportamiento sea equivalente entre proveedores, lo que fija el estilo de la respuesta es el **prompt grounded** (máximo 4 oraciones, solo datos del contexto, "No lo sé." si falta información) y la **salida estructurada** validada con Pydantic, no la temperatura.

`langchain-community` (de donde viene `BM25Retriever`) anunció su *sunset*. Se sigue usando porque es el retriever léxico que pide la consigna y el aviso de deprecación se silencia en el import. Si en el futuro deja de instalarse, reemplazarlo por un `BaseRetriever` propio sobre `rank_bm25` es un cambio de un solo archivo.

### Evaluación (`evaluate.py`)
- **Golden set** (`eval/golden_set.json`): 10 preguntas con su documento fuente conocido de antemano. Hay 6 semánticas (paráfrasis) y 4 de término exacto (códigos de error, valores numéricos). Antes de evaluar se valida que todos los ids existan en el corpus: un id mal escrito no puede dar un Recall de 0 "por tipeo".
- **Métricas a nivel documento:**
  - **Recall@5**: fracción de los documentos relevantes que aparece entre los 5 recuperados.
  - **Precision@5**: fracción de los 5 chunks recuperados que pertenece a un documento relevante.
  - **Hit@5**: si el documento esperado está entre los 5.
  - **MRR**: 1 / posición del primer resultado relevante.
- **Precision@5 tiene un techo.** Si el documento relevante solo tiene 2 chunks, la precisión máxima posible es 0,4. Por eso el reporte informa también la "precisión máxima alcanzable".
- **Se evalúan los tres modos** (BM25, vectorial e híbrido) con la misma configuración, para que el número muestre si la búsqueda híbrida aporta.

## Resultados

### Backend en memoria (mismos 35 chunks y el mismo modelo de embeddings, sin Pinecone)

Salida real de `python evaluate.py --backend memoria` (detalle en `reports/evaluacion_memoria.json`):

| Modo | Recall@5 | Precision@5 | (máx. posible) | Hit@5 | MRR |
|---|---|---|---|---|---|
| BM25 | 0,917 | 0,480 | 0,700 | 1,000 | 0,950 |
| Vectorial | 0,850 | 0,420 | 0,700 | 0,800 | 0,900 |
| **Híbrido** | **0,917** | **0,500** | 0,700 | **1,000** | **0,950** |

Lectura:
- **La búsqueda vectorial falla dos preguntas.**
  - q01 es una paráfrasis ("se caen los cobros"): no recupera el runbook de incidentes.
  - q03 pregunta por `PAY-4091`: trae la guía de APIs, pero no la entrada del catálogo con ese código exacto.
- **BM25 encuentra ambas**, y el híbrido conserva lo mejor de los dos: no pierde ninguna pregunta (Hit@5 = 1,0) y tiene la mejor precisión.
- **Las dos preguntas con Recall < 1** (q05 y q06) tienen la respuesta repartida en varios documentos y el top-5 no alcanzó a cubrirlos todos.

### Backend Pinecone

Salida real de `python evaluate.py` contra el índice serverless `orbita-handbook` (aws/us-east-1, dim 384, namespace `handbook-v1` con 35 vectores), corrida el 2026-11-04. El detalle está en `reports/evaluacion_pinecone.json`.

| Modo | Recall@5 | Precision@5 | (máx. posible) | Hit@5 | MRR |
|---|---|---|---|---|---|
| BM25 | 0,917 | 0,480 | 0,700 | 1,000 | 0,950 |
| Vectorial | 0,850 | 0,420 | 0,700 | 0,800 | 0,900 |
| **Híbrido** | **0,917** | **0,500** | 0,700 | **1,000** | **0,950** |

Los números son idénticos a los del backend en memoria, pregunta por pregunta. Es lo esperado: los dos usan los mismos chunks y el mismo modelo de embeddings, y lo único que cambia es dónde viven los vectores. Esto confirma que la ingesta subió el corpus completo y sin alteraciones (texto, ids y metadata).

El 2026-11-05 se repitió todo desde un clon limpio del repositorio: `setup_index.py` detectó el índice existente sin recrearlo, dos `ingest.py` seguidos dejaron los mismos 35 vectores (sin duplicados ni obsoletos) y `evaluate.py` volvió a dar exactamente estas métricas, pregunta por pregunta.

## Uso

```bash
python consultar.py "¿Qué significa el error PAY-4091?"
python consultar.py "rotación de secretos" --categoria seguridad
python consultar.py "¿Qué significa el error PAY-4091?" --modo bm25
python consultar.py "¿Qué significa el error PAY-4091?" --responder    # genera la respuesta con el LLM
python consultar.py "..." --backend memoria                              # sin Pinecone
```

Salida real de `--responder` contra Pinecone y `gemini-3.5-flash-lite` (2026-11-05, una request al LLM por pregunta):

```text
$ python consultar.py "¿Qué significa el error PAY-4091 y qué tengo que hacer?" --responder
Respuesta: El error PAY-4091 significa que llegó una Idempotency-Key que ya se había usado en las últimas
24 horas, pero con un cuerpo de solicitud distinto al original. La acción recomendada es no reintentar con
la misma clave, generar una clave nueva para la operación nueva o enviar exactamente el mismo cuerpo si se
trata de un reintento.
  [4] catalogo_errores.json › PAY-4091 — Clave de idempotencia reutilizada con otro cuerpo

$ python consultar.py "¿Cuántos días de vacaciones anuales tienen los ingenieros que hacen guardias?" --responder
Respuesta: No lo sé. Falta información sobre los días de vacaciones anuales para los ingenieros que hacen
guardias en el contexto provisto.
```

En la segunda, el top-5 trae la sección "Compensación" de la guía de guardias (francos compensatorios), pero el modelo no la confunde con vacaciones y no cita ninguna fuente.

Desde código:

```python
from rag_system import construir_rag_system

rag = construir_rag_system("pinecone")
for doc in rag.buscar("¿Qué header evita cobros duplicados?"):
    print(doc.metadata["fuente"], doc.metadata["seccion"])
```

## Errores y cómo reaccionar

Ningún script termina con un *traceback* ante errores de configuración: imprimen un mensaje y salen con código distinto de 0.

| Error | Dónde | Cuándo ocurre | Qué hacer |
|---|---|---|---|
| `pydantic.ValidationError` → `Configuración inválida: ...` | todos los scripts | Falta `PINECONE_API_KEY` o la key del LLM/embeddings, `INDEX_NAME` con mayúsculas, `RAG_CHUNK_SIZE` fuera de 500–800, `RAG_CANDIDATOS` < `RAG_TOP_K`, etc. | Corregir el `.env` según el mensaje. |
| `IndexMismatchError` (exit 1) | `setup_index.py`, `ingest.py`, `RAGSystem` | El índice existe con otra dimensión, otra métrica u otro modelo de embeddings (tag), o no existe y se intenta consultar. | Usar otro `INDEX_NAME`, volver al modelo original o borrar el índice desde la consola de Pinecone. Si no existe: `python setup_index.py`. |
| `RuntimeError: el namespace ... está vacío` | `evaluate.py`, `consultar.py` | El índice existe pero nunca se corrió la ingesta en ese namespace. | `python ingest.py`. |
| Aviso `Pinecone tiene N vectores y el corpus local M` | `RAGSystem` | Cambió `data/docs` o el chunking desde la última ingesta. | `python ingest.py` (sincroniza y borra obsoletos). |
| `TimeoutError` | `setup_index.py`, `ingest.py` | El índice no quedó `ready` en 120 s. | Reintentar; revisar el estado en la consola de Pinecone. |
| Errores de red en el upsert | `ingest.py` | Fallo transitorio de Pinecone. | Se reintenta solo 3 veces con backoff (2 s, 4 s). Si persiste, volver a correr: la ingesta es idempotente. |
| `ValueError: el golden set referencia documentos que no existen` | `evaluate.py` | Un `documento_id_esperado` mal escrito en `eval/golden_set.json`. | Corregir el id (nombre de archivo sin extensión, o `catalogo_errores#CODIGO`). |
| `ValueError: no hay documentos de la categoría ...` | `consultar.py --categoria` | La categoría no existe en el corpus. | Usar una categoría existente (la lista sale en el resumen de `ingest.py`). |
| `OutputParserException` / `ValidationError` del LLM | `consultar.py --responder` | El modelo devolvió una salida que no cumple `RespuestaGenerada`. | Se reintenta (1 intento + `LLM_MAX_RETRIES`, con backoff de unos 4 s y 8 s). Si se agotan los intentos: `Error del LLM (...)`. |
| `ModelError` transitorio (429 por minuto, timeout, 5xx) | `consultar.py --responder` | Falla pasajera del proveedor. | Se reintenta igual que el caso anterior. Si persiste: `Error del LLM (...) Es transitorio: probá de nuevo más tarde.` |
| `ModelRateLimitError` con `...PerDay...` | `consultar.py --responder` | Se agotó el cupo diario del modelo. | **No se reintenta.** Esperar al reinicio (≈ 05:00 ART) o pasar a `GEMINI_MODEL=gemini-3.1-flash-lite`. |
| `ModelError` permanente (API key, permisos, modelo inexistente) | `consultar.py --responder` | Error de configuración. | No se reintenta: corregir el `.env`. |

Los reintentos son de la cadena, no del SDK: los chat models se crean sin reintentos internos (`llm.reintentos_del_sdk`), porque el SDK de Gemini reintenta cualquier 429, incluido el de cupo diario, sin distinguirlos.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

56 tests offline, sin Pinecone ni API keys:
- **Chunking y metadatos**: límite de tokens, overlap, esquema compatible con Pinecone y loaders MD/JSON/PDF (el PDF conserva el número de página en `pagina`).
- **Tokenizador léxico**.
- **Métricas**: Recall, Precision, Hit, MRR y precisión máxima.
- **Recuperación**: BM25 encuentra códigos exactos, el híbrido devuelve un top-5 sin duplicados y el filtro por categoría funciona en los tres modos.
- **Ingesta**: upsert por lotes con el texto en la metadata y borrado de obsoletos (contra un índice simulado).
- **Infraestructura**: creación del índice solo si no existe y rechazo de dimensión o modelo distintos.
- **Golden set**: validación de los ids contra el corpus.
- **LLM**: modelos por defecto, key faltante, traducción de `temperature` por proveedor (Gemini 3 y Anthropic no la reciben y se avisa), `thinking_level`, rate limiter compartido y SDK sin reintentos internos.
- **Reintentos de la generación**: usa `method="json_schema"`; una salida inválida y un 429 por minuto se reintentan, mientras que el cupo diario y una API key inválida generan una sola request.

## Cómo cubre la rúbrica

| Criterio | Dónde |
|---|---|
| Infraestructura cloud (15 %) | `setup_index.py` + `pinecone_store.asegurar_indice`: serverless, verificación de existencia, dimensión del modelo, tag de embeddings, `.env` |
| Ingesta y metadatos (25 %) | `documents.py` (MD/JSON/PDF, chunking por tokens, `ChunkMetadata`) + `ingest.py` (lotes, ids deterministas, obsoletos, namespace) |
| Recuperador híbrido (35 %) | `rag_system.RAGSystem`: `EnsembleRetriever` con `BM25Retriever` + `PineconeVectorStore`, top-5, filtro por categoría |
| Evaluación cuantitativa (25 %) | `evaluate.py` + `metrics.py`: Recall@5 y Precision@5 (más Hit@5 y MRR) sobre `eval/golden_set.json`, comparando 3 modos |
