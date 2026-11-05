"""Preprocesamiento léxico para BM25.

El preprocesamiento por defecto de BM25Retriever es `texto.split()`: distingue mayúsculas,
deja la puntuación pegada ("PAY-4091." no coincide con "PAY-4091") y cuenta las palabras
vacías ("de", "la", "que"). Esta versión:
- pasa a minúsculas y quita tildes ("Política" == "politica");
- conserva los códigos con guion como un token ("pay-4091") y además sus partes ("pay", "4091");
- descarta stopwords del español.
"""

import re
import unicodedata

STOPWORDS_ES = frozenset(
    """
    a al algo algun alguna algunas alguno algunos ante antes asi aun aunque cada como con contra cual
    cuales cuando de del desde donde dos e el ella ellas ellos en entre era es esa esas ese eso esos
    esta estan estas este esto estos fue fueron ha han hasta hay la las le les lo los mas me mi muy
    nada ni no nos o otra otras otro otros para pero poco por porque que quien se sea segun ser si
    sido sin sobre son su sus tambien tan tanto te tiene tienen todo todos tu un una unas uno unos y
    ya yo cual cuanto cuantos cuanta cuantas tengo tenes hace hacer hago debe deben puede pueden
    """.split()
)

_TOKEN = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")


def normalizar(texto: str) -> str:
    """Minúsculas y sin tildes (NFKD + descarte de marcas diacríticas)."""
    descompuesto = unicodedata.normalize("NFKD", texto.casefold())
    return "".join(c for c in descompuesto if not unicodedata.combining(c))


def tokenizar(texto: str) -> list[str]:
    tokens: list[str] = []
    for token in _TOKEN.findall(normalizar(texto)):
        if token in STOPWORDS_ES:
            continue
        tokens.append(token)
        if "-" in token or "_" in token:
            tokens.extend(parte for parte in re.split(r"[-_]", token) if len(parte) > 1 and parte not in STOPWORDS_ES)
    return tokens
