from lexico import normalizar, tokenizar


def test_normaliza_mayusculas_y_tildes():
    assert normalizar("Política de Migración") == "politica de migracion"


def test_conserva_codigos_y_sus_partes():
    tokens = tokenizar("¿Qué significa el error PAY-4091?")
    assert "pay-4091" in tokens
    assert "4091" in tokens
    assert "pay" in tokens


def test_descarta_stopwords_y_puntuacion():
    tokens = tokenizar("El header Idempotency-Key, de la API.")
    assert "el" not in tokens and "de" not in tokens and "la" not in tokens
    assert "idempotency-key" in tokens
    assert "api" in tokens
