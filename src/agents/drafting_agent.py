from __future__ import annotations

from src.models.contracts import Citation, DraftDecision, ExtractedCase, RetrievalResult
from src.agents.template_agent import TemplateAgent


class DraftingAgent:
    """Generates a non-binding draft in a structured format."""

    def __init__(self) -> None:
        self.template_agent = TemplateAgent()

    @staticmethod
    def _normalize(text: str) -> str:
        return (text or "").strip().lower()

    def _select_template_with_jurisprudence(
        self,
        extracted: ExtractedCase,
        citations: list[Citation],
    ):
        selected_template = self.template_agent.select(extracted)
        pba_juris = [
            c
            for c in citations
            if (c.source_type or "").lower() == "jurisprudencia"
            and "provincia de buenos aires" in self._normalize(c.jurisdiction or "")
        ]
        if not pba_juris:
            return selected_template

        excerpt_text = " ".join(self._normalize(c.excerpt) for c in pba_juris[:5])
        if "exclusion del hogar" in excerpt_text:
            return self.template_agent.get_catalog()["medida_exclusion"]
        if "prohibicion de acercamiento" in excerpt_text or "perimetro" in excerpt_text:
            return self.template_agent.get_catalog()["medida_perimetro"]
        if "impedimento de contacto" in excerpt_text:
            return self.template_agent.get_catalog()["medida_impedimento_contacto"]
        if "abstencion de actos de violencia" in excerpt_text:
            return self.template_agent.get_catalog()["medida_abstencion_violencia"]
        return selected_template

    def invoke(self, extracted: ExtractedCase, retrieval: list[RetrievalResult]) -> DraftDecision:
        all_citations: list[Citation] = []
        for item in retrieval:
            all_citations.extend(item.hits)

        high_risk = any(
            token in " ".join(extracted.risk_factors).lower()
            for token in ["arma", "incumplimiento", "nna", "menor"]
        )
        risk_band = "high" if high_risk else "medium"
        selected_template = self._select_template_with_jurisprudence(extracted, all_citations)

        draft = (
            "Borrador no vinculante: se sugiere evaluar medidas cautelares urgentes "
            "con enfoque de proteccion integral y seguimiento interdisciplinario. "
            f"Template sugerido: {selected_template.name} ({selected_template.template_id})."
        )
        checklist = [
            "Verificar hechos y medidas vigentes en el expediente digital.",
            "Confirmar competencia y jurisdiccion.",
            "Revisar procedencia normativa de cada medida sugerida.",
            "Completar los placeholders del template con datos del expediente.",
            "Validar el borrador por operador humano antes de firma.",
        ]

        return DraftDecision(
            decision_draft=draft,
            checklist=checklist,
            suggested_measures=selected_template.measures,
            selected_template_id=selected_template.template_id,
            selected_template_name=selected_template.name,
            template_text=selected_template.text,
            fields_to_fill=selected_template.fields_to_fill,
            risk_band=risk_band,
            citations=all_citations[:8],
        )
