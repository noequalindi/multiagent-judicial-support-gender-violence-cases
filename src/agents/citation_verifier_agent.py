from __future__ import annotations

from src.models.contracts import DraftDecision


class CitationVerifierAgent:
    """Applies a minimum support rule: enforce abstention if citations are missing."""

    def invoke(self, draft: DraftDecision) -> DraftDecision:
        if draft.citations:
            return draft

        draft.abstention = True
        draft.abstention_reason = "Soporte normativo/jurisprudencial insuficiente para sostener el borrador."
        draft.decision_draft = (
            "ABSTENCION: el sistema no emite recomendacion automatica por falta de evidencia recuperada."
        )
        return draft
