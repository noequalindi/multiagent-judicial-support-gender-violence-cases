from __future__ import annotations

from src.models.contracts import DraftDecision


class AlertingAgent:
    def invoke(self, draft: DraftDecision) -> list[str]:
        alerts: list[str] = []
        if draft.risk_band == "high":
            alerts.append("ALERTA_CRITICA: priorizar intervencion y notificacion con acuse.")
        if draft.abstention:
            alerts.append("ALERTA_CALIDAD: salida en abstencion, requiere revision manual reforzada.")
        if len(draft.citations) < 2:
            alerts.append("ALERTA_CITAS: bajo soporte documental para el borrador.")
        return alerts
