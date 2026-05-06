from __future__ import annotations

import re
import unicodedata

from src.models.contracts import ExtractedCase


class UnsupportedDocumentError(RuntimeError):
    pass


def _normalize_text(text: str) -> str:
    lowered = (text or "").lower()
    normalized = unicodedata.normalize("NFD", lowered)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def assert_supported_case_document(text: str) -> None:
    normalized = _normalize_text(text)
    category_patterns = {
        "violence": [
            r"\bviolencia\b",
            r"\bviolencia domestica\b",
            r"\bviolencia familiar\b",
            r"\bviolencia de genero\b",
            r"\bviolencia contra la mujer\b",
            r"\bmaltrato\b",
            r"\bagresion\b",
            r"\bamenazas?\b",
            r"\bhostigamiento\b",
            r"\blesiones?\b",
            r"\bgolpes?\b",
            r"\bviolento\b",
            r"\bacoso\b",
            r"\babuso\b",
            r"\bcoaccion\b",
        ],
        "judicial": [
            r"\bdenuncia\b",
            r"\bexpediente\b",
            r"\bjuzgado\b",
            r"\bfiscalia\b",
            r"\bcomisaria\b",
            r"\bovd\b",
            r"\boficina de violencia domestica\b",
            r"\bmedidas?\b",
            r"\bmedida cautelar\b",
            r"\bsecretaria\b",
            r"\bfuero\b",
            r"\bcausa\b",
            r"\bactuaciones?\b",
            r"\bintervencion judicial\b",
            r"\bdenuncias?\s*n[o0°º]?\b",
        ],
        "people": [
            r"\bvictima\b",
            r"\bdenunciad[oa]\b",
            r"\bdenunciante\b",
            r"\bagresor\b",
            r"\bimputad[oa]\b",
            r"\bdamnificad[oa]\b",
            r"\brequirente\b",
            r"\bpersona agresora\b",
            r"\bintervinientes?\b",
        ],
        "risk_or_protection": [
            r"\briesgo\b",
            r"\bperimetral\b",
            r"\bprohibicion de acercamiento\b",
            r"\bimpedimento de contacto\b",
            r"\bexclusion del hogar\b",
            r"\bproteccion\b",
            r"\brestriccion\b",
            r"\bmedidas? de proteccion\b",
            r"\bboton antipanic[oa]\b",
            r"\bcese de actos de perturbacion\b",
            r"\babstencion de\b",
        ],
        "family_or_context": [
            r"\bgrupo familiar\b",
            r"\bambito familiar\b",
            r"\bconvivient[ea]s?\b",
            r"\bconvivencia\b",
            r"\bpareja\b",
            r"\bex pareja\b",
            r"\bhij[oa]s?\b",
            r"\bprogenitor[ae]?\b",
        ],
        "intake_form": [
            r"\brelato de los hechos\b",
            r"\bhechos\b",
            r"\bfecha del hecho\b",
            r"\bcontacto\b",
            r"\bdomicilio\b",
            r"\borganismo receptor\b",
            r"\bnumero de denuncia\b",
            r"\bnumero de expediente\b",
        ],
    }
    hits_by_category = {
        name: sum(1 for pattern in patterns if re.search(pattern, normalized)) for name, patterns in category_patterns.items()
    }
    matches = {name: hits > 0 for name, hits in hits_by_category.items()}
    strong_negative_patterns = [
        r"\bfactura\b",
        r"\bpresupuesto\b",
        r"\bcurriculum\b",
        r"\bcv\b",
        r"\bboleta\b",
        r"\breceta\b",
        r"\binvoice\b",
        r"\bremito\b",
        r"\bcertificado de trabajo\b",
        r"\bconstancia de cbu\b",
    ]
    negative_hits = sum(1 for pattern in strong_negative_patterns if re.search(pattern, normalized))
    category_score = sum(1 for matched in matches.values() if matched)
    signal_score = sum(hits_by_category.values())
    has_case_number = bool(
        re.search(
            r"\b(?:fv|fvo|fvdo|fvodo|exp|ipp|pp)[a-z0-9/-]{4,}\b",
            normalized,
        )
        or re.search(r"\b(?:numero|n[o0°º])\s+de\s+(?:denuncia|expediente)\b", normalized)
    )
    is_supported = (
        matches["judicial"]
        and (matches["violence"] or matches["risk_or_protection"])
        and category_score >= 2
    ) or (
        matches["judicial"]
        and matches["people"]
        and matches["risk_or_protection"]
    ) or (
        matches["judicial"]
        and matches["people"]
        and matches["family_or_context"]
        and signal_score >= 4
    ) or (
        matches["violence"]
        and matches["people"]
        and (matches["family_or_context"] or matches["intake_form"])
        and signal_score >= 4
    ) or (
        has_case_number
        and matches["people"]
        and (matches["violence"] or matches["risk_or_protection"] or matches["family_or_context"])
    )
    if negative_hits:
        is_supported = is_supported and category_score >= 3 and signal_score >= 5
    if not is_supported:
        raise UnsupportedDocumentError(
            "El sistema solo acepta denuncias, expedientes o documentación vinculada a violencia de género o violencia doméstica. "
            "No se procesan otros tipos de documentos."
        )


def assert_safe_for_external_llm(extracted: ExtractedCase) -> None:
    text = extracted.anonymized_text or ""
    suspicious_patterns = [
        r"\b[A-ZÁÉÍÓÚÑ]{2,}(?:\s+[A-ZÁÉÍÓÚÑ]{2,}){1,3}\b",
        r"\b(?:calle|avenida|av\.?|pasaje|pje\.?|ruta|camino)\s+[A-Za-zÁÉÍÓÚÑáéíóúñ]{2,}(?:[\s,.-]+\d{1,5})\b",
        r"\b(?:manzana|mz\.?)\s+[A-Za-z0-9-]{1,12}\b",
        r"\b(?:casa|piso|depto\.?|departamento)\s+\d{1,5}[A-Za-z]?\b",
        r"\b\d{2}\.?\d{3}\.?\d{3}\b",
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
