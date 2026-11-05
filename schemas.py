"""Contratos de datos: respuesta generada, resultado del RAG y golden set de evaluación."""

from pydantic import BaseModel, Field, field_validator, model_validator

RESPUESTA_SIN_CONTEXTO = "No lo sé."


# --- Generación -----------------------------------------------------------------------


class RespuestaGenerada(BaseModel):
    """Respuesta del asistente basada exclusivamente en los fragmentos del contexto."""

    respuesta: str = Field(
        description=(
            "Respuesta en español, concisa (máximo 4 oraciones), usando solo datos del contexto. "
            f"Si el contexto no alcanza, empezá exactamente con '{RESPUESTA_SIN_CONTEXTO}' y aclarás qué falta."
        )
    )
    respuesta_encontrada: bool = Field(description="true si el contexto contiene la información para responder.")
    fragmentos_usados: list[int] = Field(
        default_factory=list,
        description="Números [n] de los fragmentos de los que sale la respuesta. Vacío si no se encontró.",
    )

    @field_validator("respuesta")
    @classmethod
    def respuesta_no_vacia(cls, valor: str) -> str:
        limpio = " ".join(valor.split())
        if not limpio:
            raise ValueError("la respuesta no puede estar vacía")
        return limpio

    @model_validator(mode="after")
    def coherencia(self) -> "RespuestaGenerada":
        if self.respuesta_encontrada and not self.fragmentos_usados:
            raise ValueError("si respuesta_encontrada es true hay que citar al menos un fragmento")
        if not self.respuesta_encontrada:
            self.fragmentos_usados = []
            if not self.respuesta.startswith(RESPUESTA_SIN_CONTEXTO):
                self.respuesta = f"{RESPUESTA_SIN_CONTEXTO} {self.respuesta}"
        self.fragmentos_usados = sorted(set(self.fragmentos_usados))
        return self


class Fuente(BaseModel):
    fragmento: int
    doc_id: str
    fuente: str
    seccion: str
    chunk_id: str


class RespuestaRAG(BaseModel):
    pregunta: str
    respuesta: str
    respuesta_encontrada: bool
    fuentes: list[Fuente] = Field(default_factory=list)
    documentos_recuperados: int = Field(ge=0)
    latencia_s: float = Field(ge=0)


# --- Evaluación -------------------------------------------------------------------------


class CasoGolden(BaseModel):
    id: str
    pregunta: str = Field(min_length=5)
    documento_id_esperado: str
    documentos_relevantes: list[str] = Field(min_length=1)
    tipo: str = "semantica"

    @model_validator(mode="after")
    def esperado_es_relevante(self) -> "CasoGolden":
        if self.documento_id_esperado not in self.documentos_relevantes:
            self.documentos_relevantes.insert(0, self.documento_id_esperado)
        return self


class GoldenSet(BaseModel):
    descripcion: str = ""
    preguntas: list[CasoGolden] = Field(min_length=5)

    @field_validator("preguntas")
    @classmethod
    def ids_unicos(cls, preguntas: list[CasoGolden]) -> list[CasoGolden]:
        ids = [p.id for p in preguntas]
        if len(ids) != len(set(ids)):
            raise ValueError("hay ids de pregunta repetidos en el golden set")
        return preguntas
