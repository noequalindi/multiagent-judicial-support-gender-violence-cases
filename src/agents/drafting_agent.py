from __future__ import annotations

from src.models.contracts import Citation, DraftDecision, ExtractedCase, RetrievalResult
from src.agents.template_agent import TemplateAgent


class DraftingAgent:
    """Generates a non-binding draft in a structured format."""

    def __init__(self) -> None:
        self.template_agent = TemplateAgent()

    def invoke(self, extracted: ExtractedCase, retrieval: list[RetrievalResult]) -> DraftDecision:
        all_citations: list[Citation] = []
        for item in retrieval:
            all_citations.extend(item.hits)

        high_risk = any(
            token in " ".join(extracted.risk_factors).lower()
            for token in ["arma", "incumplimiento", "nna", "menor"]
        )
        risk_band = "high" if high_risk else "medium"
        selected_template = self.template_agent.select(extracted)

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
