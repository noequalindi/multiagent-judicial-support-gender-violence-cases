from __future__ import annotations

import re

from src.models.contracts import ExtractedCase


def assert_safe_for_external_llm(extracted: ExtractedCase) -> None:
    text = extracted.anonymized_text or ""
    suspicious_patterns = [
        r"\b[A-ZÁÉÍÓÚÑ]{2,}(?:\s+[A-ZÁÉÍÓÚÑ]{2,}){1,3}\b",
        r"\b(?:calle|domicilio|localidad|partido)\s+[A-Za-zÁÉÍÓÚÑáéíóúñ0-9 ]{4,}",
        r"\b(?:FV|FVO|EXP|PP)\s*[A-Z0-9\-/]{4,}\b",
    ]
    placeholders = {
        "[PERSONA_REDACTED]",
        "[DOMICILIO_REDACTED]",
        "[ID_REDACTED]",
        "[PHONE_REDACTED]",
        "[EMAIL_REDACTED]",
        "[DNI_REDACTED]",
        "[LOCALIDAD_REDACTED]",
    }
    sanitized = text
    for token in placeholders:
        sanitized = sanitized.replace(token, " ")
    for pattern in suspicious_patterns:
        if re.search(pattern, sanitized):
            raise RuntimeError(
                "El caso anonimizado aun contiene posibles datos sensibles. Revisar limpieza y anonimización antes de usar proveedor externo."
            )
