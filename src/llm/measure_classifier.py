from __future__ import annotations
import os

from src.agents.template_agent import TemplateAgent
from src.legal.source_catalog import LegalSourceCatalog
from src.llm.client_factory import build_json_llm_client
from src.llm.measure_classifier_prompts import (
    CLAUDE_MEASURE_CLASSIFICATION_SYSTEM_PROMPT,
    build_claude_measure_classification_user_prompt,
    build_generic_measure_classification_system_prompt,
    build_generic_measure_classification_user_prompt,
    normalize_prompt_variant,
)
from src.llm.safety import assert_safe_for_external_llm
from src.logging import get_logger
from src.models.contracts import (
    ApplicableArticle,
    ExtractedCase,
    LegalBasisItem,
    MeasureClassification,
    OfficialSource,
    RetrievalResult,
)


class MeasureClassifierService:
    def __init__(self) -> None:
        self.template_agent = TemplateAgent()
        self.source_catalog = LegalSourceCatalog()
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

    def _build_official_sources(
        self,
        retrieval: list[RetrievalResult],
        supporting_source_ids: list[str],
    ) -> list[OfficialSource]:
        by_id = {}
        ordered_hits = [hit for item in retrieval for hit in item.hits]
        prioritized_ids = supporting_source_ids or [hit.source_id for hit in ordered_hits]
        for source_id in prioritized_ids:
            hit = next((item for item in ordered_hits if item.source_id == source_id), None)
            row = self.source_catalog.resolve(source_id)
            title = (
                (hit.title if hit else None)
                or (str(row.get("title", "")).strip() if row else "")
                or source_id
            )
            source = OfficialSource(
                source_id=source_id,
                title=title,
                law=(hit.law if hit else None) or (str(row.get("law", "")).strip() if row else None) or None,
                article=(hit.article if hit else None) or (str(row.get("article", "")).strip() if row else None) or None,
                source_type=(hit.source_type if hit else None) or (str(row.get("source_type", "")).strip() if row else None) or None,
                jurisdiction=(hit.jurisdiction if hit else None) or (str(row.get("jurisdiction", "")).strip() if row else None) or None,
                summary=(hit.summary if hit else None) or (str(row.get("summary", "")).strip() if row else None) or None,
                official_url=(hit.official_url if hit else None) or (str(row.get("official_url", "")).strip() if row else None) or None,
                relevance_score=hit.relevance_score if hit else None,
            )
            if source.source_id not in by_id:
                by_id[source.source_id] = source
            if len(by_id) >= 4:
                break
        return list(by_id.values())

    @staticmethod
    def _parse_legal_basis(raw_items: object) -> list[LegalBasisItem]:
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

    @staticmethod
    def _parse_applicable_articles(raw_items: object) -> list[ApplicableArticle]:
        parsed: list[ApplicableArticle] = []
        if not isinstance(raw_items, list):
            return parsed
        for item in raw_items[:8]:
            if not isinstance(item, dict):
                continue
            parsed.append(
                ApplicableArticle(
                    ley=str(item.get("ley", "")).strip(),
                    articulo=str(item.get("articulo", "")).strip(),
                    relevancia=str(item.get("relevancia", "")).strip(),
                )
            )
        return parsed

    @staticmethod
    def _build_explanation_summary(
        extracted: ExtractedCase,
        classification: MeasureClassification,
        official_sources: list[OfficialSource],
    ) -> str:
        segments: list[str] = []
        if classification.selected_template_name:
            segments.append(f"Se sugiere {classification.selected_template_name.lower()}")
        if classification.risk_level:
            segments.append(f"por nivel de riesgo {classification.risk_level}")
        if extracted.risk_factors:
            segments.append(
                "considerando "
                + ", ".join(extracted.risk_factors[:3])
            )
        if official_sources:
            source_titles = [src.title for src in official_sources[:2] if src.title]
            if source_titles:
                segments.append("con sustento en " + " y ".join(source_titles))
        text = " ".join(segment.strip().rstrip(".") for segment in segments if segment).strip()
        return (text + ".") if text else "La medida se sugiere por la combinación de hechos relevantes, riesgo detectado y sustento jurídico recuperado."

    def _classify_with_claude(
        self,
        client: object,
        cfg: object,
        extracted: ExtractedCase,
        retrieval: list[RetrievalResult],
        prompt_variant: str,
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
                prompt_variant=prompt_variant,
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

        articulos = fundamentacion.get("articulos_aplicables", [])
        applicable_articles = self._parse_applicable_articles(articulos)
        normative_basis = self._parse_legal_basis(response.get("normative_basis", []))
        procedural_basis = self._parse_legal_basis(response.get("procedural_basis", []))

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

        base = MeasureClassification(
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
        official_sources = self._build_official_sources(retrieval, supporting_source_ids)
        base.official_sources = official_sources
        base.explanation_summary = self._build_explanation_summary(extracted, base, official_sources)
        return base

    def classify(
        self,
        extracted: ExtractedCase,
        retrieval: list[RetrievalResult],
        provider: str,
        model: str | None = None,
        include_draft: bool = False,
        prompt_variant: str | None = None,
    ) -> MeasureClassification:
        normalized_prompt_variant = normalize_prompt_variant(prompt_variant)
        cfg, client = build_json_llm_client(provider=provider, model=model)
        assert_safe_for_external_llm(extracted)
        self.logger.info(
            "classification_start case_id={} provider={} model={} prompt_variant={}",
            extracted.case_id,
            cfg.provider,
            cfg.model,
            normalized_prompt_variant,
        )
        if provider == "anthropic":
            classification = self._classify_with_claude(
                client=client,
                cfg=cfg,
                extracted=extracted,
                retrieval=retrieval,
                prompt_variant=normalized_prompt_variant,
            )
            self.logger.info(
                "classification_done case_id={} provider={} model={} template={} risk_level={} prompt_variant={}",
                extracted.case_id,
                classification.provider,
                classification.model,
                classification.selected_template_id,
                classification.risk_level or "-",
                normalized_prompt_variant,
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

        system_prompt = build_generic_measure_classification_system_prompt(
            include_draft=include_draft,
            prompt_variant=normalized_prompt_variant,
        )
        user_prompt = build_generic_measure_classification_user_prompt(
            extracted=extracted,
            retrieval_support=retrieval_support,
            catalog=catalog,
            include_draft=include_draft,
            prompt_variant=normalized_prompt_variant,
        )

        response = client.chat_json(system_prompt=system_prompt, user_prompt=user_prompt)
        template_id = str(response.get("selected_template_id", "")).strip()
        if template_id not in self.template_agent._catalog:
            fallback = self.template_agent.select(extracted)
            fallback_classification = MeasureClassification(
                provider=cfg.provider,
                model=cfg.model,
                selected_template_id=fallback.template_id,
                selected_template_name=fallback.name,
                rationale="Fallback local por salida invalida del proveedor externo.",
                confidence=0.0,
                supporting_source_ids=[],
                suggested_measures=[],
                draft_text=None,
            )
            fallback_classification.explanation_summary = self._build_explanation_summary(
                extracted,
                fallback_classification,
                [],
            )
            return fallback_classification

        template = self.template_agent._catalog[template_id]
        supporting_source_ids = response.get("supporting_source_ids", [])
        if not isinstance(supporting_source_ids, list):
            supporting_source_ids = []
        supporting_source_ids = self._dedupe_preserve_order([str(x) for x in supporting_source_ids])[:8]

        risk_level = str(response.get("risk_level", "")).strip().lower() or None
        if risk_level not in {"alto", "medio", "bajo"}:
            risk_level = None
        risk_indicators = response.get("risk_indicators", [])
        if not isinstance(risk_indicators, list):
            risk_indicators = []
        alerts = response.get("alerts", [])
        if not isinstance(alerts, list):
            alerts = []
        normative_basis = self._parse_legal_basis(response.get("normative_basis", []))
        procedural_basis = self._parse_legal_basis(response.get("procedural_basis", []))
        applicable_articles = self._parse_applicable_articles(response.get("applicable_articles", []))

        classification = MeasureClassification(
            provider=cfg.provider,
            model=cfg.model,
            selected_template_id=template.template_id,
            selected_template_name=template.name,
            rationale=str(response.get("rationale", "")).strip() or "Sin justificacion provista por el modelo.",
            confidence=float(response.get("confidence", 0.0) or 0.0),
            supporting_source_ids=supporting_source_ids,
            suggested_measures=[],
            risk_level=risk_level,
            risk_indicators=[str(item).strip() for item in risk_indicators if str(item).strip()],
            alerts=[str(item).strip() for item in alerts if str(item).strip()],
            applicable_articles=applicable_articles,
            normative_basis=normative_basis,
            procedural_basis=procedural_basis,
            low_confidence_reason=str(response.get("low_confidence_reason", "")).strip() or None,
            draft_text=str(response.get("draft_text", "")).strip() or None,
        )
        official_sources = self._build_official_sources(retrieval, supporting_source_ids)
        classification.official_sources = official_sources
        classification.explanation_summary = self._build_explanation_summary(
            extracted,
            classification,
            official_sources,
        )
        self.logger.info(
            "classification_done case_id={} provider={} model={} template={} prompt_variant={}",
            extracted.case_id,
            classification.provider,
            classification.model,
            classification.selected_template_id,
            normalized_prompt_variant,
        )
        return classification
