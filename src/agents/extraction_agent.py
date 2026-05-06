from __future__ import annotations

import unicodedata

from src.models.contracts import ExtractedCase


class ExtractionAgent:
    """Initial rule-based extraction (placeholder)."""

    _CABA_HINTS = {
        "ciudad autonoma de buenos aires",
        "ciudad de buenos aires",
        "caba",
        "policia de la ciudad",
        "ovd",
        "oficina de violencia domestica",
        "comuna",
        "palermo",
        "flores",
        "barracas",
        "caballito",
        "belgrano",
        "almagro",
        "villa urquiza",
        "recoleta",
        "liniers",
        "mataderos",
        "boedo",
        "once",
        "constitucion",
        "san cristobal",
        "villa lugano",
    }

    @staticmethod
    def _normalize(text: str) -> str:
        text = (text or "").lower()
        return "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")

    @classmethod
    def _detect_origin_jurisdiction(cls, lower: str) -> str | None:
        for hint in cls._CABA_HINTS:
            if hint in lower:
                return "Ciudad Autónoma de Buenos Aires"
        return None

    def invoke(self, case_id: str, anonymized_text: str) -> ExtractedCase:
        lower = self._normalize(anonymized_text)
        origin_jurisdiction = self._detect_origin_jurisdiction(lower)

        facts = []
        if "amenaza" in lower:
            facts.append("Se reportan amenazas.")
        if "golpe" in lower or "lesion" in lower:
            facts.append("Se reportan agresiones fisicas o lesiones.")
        if "hostig" in lower:
            facts.append("Se reportan hechos de hostigamiento.")
        if not facts:
            facts.append("Hechos a validar por operador humano.")

        active_measures = []
        if "perimetral" in lower or "prohibicion de acercamiento" in lower:
            active_measures.append("Prohibicion de acercamiento")
        if "boton antipanico" in lower:
            active_measures.append("Boton antipanico")
        if "exclusion del hogar" in lower:
            active_measures.append("Exclusion del hogar del agresor")
        if not active_measures:
            active_measures.append("Evaluar medidas cautelares urgentes")

        risk_factors = []
        if "arma" in lower:
            risk_factors.append("Acceso reportado a arma")
        if "incumpl" in lower:
            risk_factors.append("Antecedente de incumplimiento")
        if "nino" in lower or "nina" in lower or "hijo" in lower or "menor" in lower:
            risk_factors.append("NNA involucrados")
        if "convivencia" in lower:
            risk_factors.append("Convivencia actual o reciente")
        if not risk_factors:
            risk_factors.append("Sin factores automaticos concluyentes")

        timeline = ["Ingreso de denuncia", "Evaluacion interdisciplinaria inicial"]
        return ExtractedCase(
            case_id=case_id,
            anonymized_text=anonymized_text,
            facts=facts,
            active_measures=active_measures,
            risk_factors=risk_factors,
            timeline=timeline,
            origin_jurisdiction=origin_jurisdiction,
        )
