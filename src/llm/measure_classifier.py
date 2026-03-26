from __future__ import annotations
import os

from src.agents.template_agent import TemplateAgent
from src.llm.client_factory import build_json_llm_client
from src.llm.measure_classifier_prompts import (
    CLAUDE_MEASURE_CLASSIFICATION_SYSTEM_PROMPT,
    build_claude_measure_classification_user_prompt,
    build_generic_measure_classification_system_prompt,
    build_generic_measure_classification_user_prompt,
)
from src.llm.safety import assert_safe_for_external_llm
from src.logging import get_logger
from src.models.contracts import (
    ApplicableArticle,
    ExtractedCase,
    LegalBasisItem,
    MeasureClassification,
    RetrievalResult,
)


class MeasureClassifierService:
    def __init__(self) -> None:
        self.template_agent = TemplateAgent()
        self.logger = get_logger("measure_classifier")

    @staticmethod
    def _dedupe_preserve_order(items: list[str]) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for item in items:
            value = str(item).strip()
            if not value or value in seen:
                continue
            seen.add(value)
            deduped.append(value)
        return deduped

    @staticmethod
    def _confidence_to_float(raw: str | float | int | None) -> float:
        if isinstance(raw, (int, float)):
            return max(0.0, min(1.0, float(raw)))
        mapping = {
            "alta": 0.9,
            "media": 0.6,
            "baja": 0.3,
        }
        return mapping.get(str(raw or "").strip().lower(), 0.0)

    @staticmethod
    def _normalize_measure_name(raw: str) -> str:
        mapping = {
            "perimetro": "perimetro",
            "exclusion": "exclusion",
            "impedimento_contacto": "impedimento_contacto",
            "abstencion_violencia": "abstencion_violencia",
        }
        return mapping.get(str(raw).strip().lower(), "")

    def _template_id_from_suggested_measures(self, measures: list[str]) -> str | None:
        normalized = [self._normalize_measure_name(item) for item in measures]
        normalized = [item for item in normalized if item]
        if "exclusion" in normalized:
            return "medida_exclusion"
        if "perimetro" in normalized:
            return "medida_perimetro"
        if "impedimento_contacto" in normalized:
            return "medida_impedimento_contacto"
        if "abstencion_violencia" in normalized:
            return "medida_abstencion_violencia"
        return None

    @staticmethod
    def _format_rag_fragments(retrieval: list[RetrievalResult]) -> str:
        fragments: list[str] = []
        seen_source_ids: set[str] = set()
        for item in retrieval:
            for hit in item.hits:
                if hit.source_id in seen_source_ids:
                    continue
                seen_source_ids.add(hit.source_id)
                title = hit.title or hit.source_id
                law = hit.law or "No informada"
                article = hit.article or "No informado"
                source_type = hit.source_type or "desconocido"
                excerpt = hit.excerpt.strip()
                fragments.append(
                    "\n".join(
                        [
                            f"[{len(fragments) + 1}] {title}",
                            f"Fuente: {source_type}",
                            f"Ley: {law} | Artículo: {article}",
                            f"Texto: {excerpt}",
                        ]
                    )
                )
                if len(fragments) >= 8:
                    return "\n\n".join(fragments)
        return "\n\n".join(fragments)

    def _classify_with_claude(
        self,
        client: object,
        cfg: object,
        extracted: ExtractedCase,
        retrieval: list[RetrievalResult],
    ) -> MeasureClassification:
        self.logger.info(
            "claude_classification_start case_id={} provider={} model={}",
            extracted.case_id,
            cfg.provider,
            cfg.model,
        )
        fragmentos_rag = self._format_rag_fragments(retrieval) or "Sin fragmentos normativos suficientes."

        thinking_budget = int(os.getenv("ANTHROPIC_THINKING_BUDGET_TOKENS", "0") or "0")
        response = client.chat_json(
            system_prompt=CLAUDE_MEASURE_CLASSIFICATION_SYSTEM_PROMPT,
            user_prompt=build_claude_measure_classification_user_prompt(
                extracted=extracted,
                fragmentos_rag=fragmentos_rag,
            ),
            max_tokens=int(os.getenv("ANTHROPIC_MAX_OUTPUT_TOKENS", "4000") or "4000"),
            thinking_budget_tokens=thinking_budget or None,
        )

        suggested_measures = response.get("medidas_sugeridas", [])
        if not isinstance(suggested_measures, list):
            suggested_measures = []
        suggested_measures = [
            self._normalize_measure_name(str(item))
            for item in suggested_measures
        ]
        suggested_measures = [item for item in suggested_measures if item]

        template_id = self._template_id_from_suggested_measures(suggested_measures)
        if template_id not in self.template_agent._catalog:
            fallback = self.template_agent.select(extracted)
            template = fallback
            template_id = fallback.template_id
        else:
            template = self.template_agent._catalog[template_id]

        fundamentacion = response.get("fundamentacion", {})
        if not isinstance(fundamentacion, dict):
            fundamentacion = {}

        def parse_legal_basis(raw_items: object) -> list[LegalBasisItem]:
            parsed: list[LegalBasisItem] = []
            if not isinstance(raw_items, list):
                return parsed
            for item in raw_items[:8]:
                if not isinstance(item, dict):
                    continue
                parsed.append(
                    LegalBasisItem(
                        ley=str(item.get("ley", "")).strip(),
                        articulo=str(item.get("articulo", "")).strip(),
                        motivo=str(item.get("motivo", "")).strip(),
                    )
                )
            return parsed

        articulos = fundamentacion.get("articulos_aplicables", [])
        applicable_articles: list[ApplicableArticle] = []
        if isinstance(articulos, list):
            for item in articulos[:8]:
                if not isinstance(item, dict):
                    continue
                applicable_articles.append(
                    ApplicableArticle(
                        ley=str(item.get("ley", "")).strip(),
                        articulo=str(item.get("articulo", "")).strip(),
                        relevancia=str(item.get("relevancia", "")).strip(),
                    )
                )
        normative_basis = parse_legal_basis(response.get("normative_basis", []))
        procedural_basis = parse_legal_basis(response.get("procedural_basis", []))

        indicators = response.get("indicadores_riesgo", [])
        if not isinstance(indicators, list):
            indicators = []
        alerts = response.get("alertas", [])
        if not isinstance(alerts, list):
            alerts = []

        rationale_parts = [
            str(fundamentacion.get("sintesis_hecho", "")).strip(),
            str(fundamentacion.get("razonamiento", "")).strip(),
        ]
        rationale = " ".join(part for part in rationale_parts if part).strip()
        supporting_source_ids = self._dedupe_preserve_order(
            [hit.source_id for item in retrieval for hit in item.hits[:3]]
        )[:8]

        return MeasureClassification(
            provider=cfg.provider,
            model=cfg.model,
            selected_template_id=template.template_id,
            selected_template_name=template.name,
            rationale=rationale or "Sin fundamentación provista por Claude.",
            confidence=self._confidence_to_float(response.get("confianza")),
            supporting_source_ids=supporting_source_ids,
            suggested_measures=suggested_measures,
            risk_level=str(response.get("nivel_riesgo", "")).strip().lower() or None,
            risk_indicators=[str(item).strip() for item in indicators if str(item).strip()],
            alerts=[str(item).strip() for item in alerts if str(item).strip()],
            applicable_articles=applicable_articles,
            normative_basis=normative_basis,
            procedural_basis=procedural_basis,
            low_confidence_reason=str(response.get("motivo_baja_confianza", "")).strip() or None,
            draft_text=str(response.get("borrador_resolucion", "")).strip() or None,
        )

    def classify(
        self,
        extracted: ExtractedCase,
        retrieval: list[RetrievalResult],
        provider: str,
        model: str | None = None,
        include_draft: bool = False,
    ) -> MeasureClassification:
        cfg, client = build_json_llm_client(provider=provider, model=model)
        assert_safe_for_external_llm(extracted)
        self.logger.info(
            "classification_start case_id={} provider={} model={}",
            extracted.case_id,
            cfg.provider,
            cfg.model,
        )
        if provider == "anthropic":
            classification = self._classify_with_claude(client=client, cfg=cfg, extracted=extracted, retrieval=retrieval)
            self.logger.info(
                "classification_done case_id={} provider={} model={} template={} risk_level={}",
                extracted.case_id,
                classification.provider,
                classification.model,
                classification.selected_template_id,
                classification.risk_level or "-",
            )
            return classification

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

        system_prompt = build_generic_measure_classification_system_prompt(include_draft=include_draft)
        user_prompt = build_generic_measure_classification_user_prompt(
            extracted=extracted,
            retrieval_support=retrieval_support,
            catalog=catalog,
            include_draft=include_draft,
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
                suggested_measures=[],
                draft_text=fallback.text if include_draft else None,
            )

        template = self.template_agent._catalog[template_id]
        supporting_source_ids = response.get("supporting_source_ids", [])
        if not isinstance(supporting_source_ids, list):
            supporting_source_ids = []
        supporting_source_ids = self._dedupe_preserve_order([str(x) for x in supporting_source_ids])[:8]

        classification = MeasureClassification(
            provider=cfg.provider,
            model=cfg.model,
            selected_template_id=template.template_id,
            selected_template_name=template.name,
            rationale=str(response.get("rationale", "")).strip() or "Sin justificacion provista por el modelo.",
            confidence=float(response.get("confidence", 0.0) or 0.0),
            supporting_source_ids=supporting_source_ids,
            suggested_measures=[],
            draft_text=str(response.get("draft_text", "")).strip() or None,
        )
        self.logger.info(
            "classification_done case_id={} provider={} model={} template={}",
            extracted.case_id,
            classification.provider,
            classification.model,
            classification.selected_template_id,
        )
        return classification
