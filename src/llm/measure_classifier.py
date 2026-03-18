from __future__ import annotations

import json
import re

from src.agents.template_agent import TemplateAgent
from src.llm.openai_compat_client import OpenAICompatClient, OpenAICompatConfig
from src.models.contracts import ExtractedCase, MeasureClassification, RetrievalResult


class MeasureClassifierService:
    def __init__(self) -> None:
        self.template_agent = TemplateAgent()

    @staticmethod
    def _assert_safe_for_external_llm(extracted: ExtractedCase) -> None:
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

    def classify(
        self,
        extracted: ExtractedCase,
        retrieval: list[RetrievalResult],
        provider: str,
        model: str | None = None,
        include_draft: bool = False,
    ) -> MeasureClassification:
        cfg = OpenAICompatConfig.from_provider(provider=provider, model=model)
        if not cfg:
            raise RuntimeError(f"Provider '{provider}' is not configured in environment variables.")
        self._assert_safe_for_external_llm(extracted)

        client = OpenAICompatClient(cfg)
        catalog = [
            {
                "template_id": template.template_id,
                "name": template.name,
                "measures": template.measures,
                "fields_to_fill": template.fields_to_fill,
                "template_text": template.text,
            }
            for template in self.template_agent._catalog.values()
        ]

        retrieval_support = []
        for item in retrieval[:6]:
            retrieval_support.append(
                {
                    "query": item.query,
                    "hits": [
                        {
                            "source_id": hit.source_id,
                            "excerpt": hit.excerpt[:240],
                            "relevance_score": hit.relevance_score,
                        }
                        for hit in item.hits[:3]
                    ],
                }
            )

        system_prompt = (
            "Sos un asistente juridico. Trabajas solo con texto anonimizado. "
            "Debes elegir una medida cautelar de una lista cerrada de plantillas judiciales. "
            "No inventes normas ni hechos. "
            "Respondé exclusivamente en JSON con estas claves: "
            "selected_template_id, selected_template_name, rationale, confidence, supporting_source_ids"
        )
        if include_draft:
            system_prompt += ", draft_text"

        user_prompt = json.dumps(
            {
                "task": "Elegir la medida cautelar mas adecuada a partir del caso anonimizado.",
                "case": {
                    "case_id": extracted.case_id,
                    "anonymized_text": extracted.anonymized_text[:3000],
                    "facts": extracted.facts,
                    "active_measures": extracted.active_measures,
                    "risk_factors": extracted.risk_factors,
                    "timeline": extracted.timeline,
                },
                "retrieval_support": retrieval_support,
                "candidate_templates": catalog,
                "instructions": {
                    "choose_from_closed_list": True,
                    "return_json_only": True,
                    "confidence_range": "0.0 to 1.0",
                    "include_draft": include_draft,
                },
            },
            ensure_ascii=False,
        )

        response = client.chat_json(system_prompt=system_prompt, user_prompt=user_prompt)
        template_id = str(response.get("selected_template_id", "")).strip()
        if template_id not in self.template_agent._catalog:
            fallback = self.template_agent.select(extracted)
            return MeasureClassification(
                provider=cfg.provider,
                model=cfg.model,
                selected_template_id=fallback.template_id,
                selected_template_name=fallback.name,
                rationale="Fallback local por salida invalida del proveedor externo.",
                confidence=0.0,
                supporting_source_ids=[],
                draft_text=fallback.text if include_draft else None,
            )

        template = self.template_agent._catalog[template_id]
        supporting_source_ids = response.get("supporting_source_ids", [])
        if not isinstance(supporting_source_ids, list):
            supporting_source_ids = []

        return MeasureClassification(
            provider=cfg.provider,
            model=cfg.model,
            selected_template_id=template.template_id,
            selected_template_name=template.name,
            rationale=str(response.get("rationale", "")).strip() or "Sin justificacion provista por el modelo.",
            confidence=float(response.get("confidence", 0.0) or 0.0),
            supporting_source_ids=[str(x) for x in supporting_source_ids[:8]],
            draft_text=str(response.get("draft_text", "")).strip() or None,
        )
