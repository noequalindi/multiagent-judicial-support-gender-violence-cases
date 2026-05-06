from __future__ import annotations

import json

from src.llm.client_factory import build_json_llm_client
from src.llm.safety import assert_safe_for_external_llm
from src.logging import get_logger
from src.models.contracts import Citation, ExtractedCase, RetrievalResult
from src.rag.local_retriever import LocalLegalRetriever
from src.rag.pinecone_client import PineconeConfig, PineconeTextIndexClient


class RetrievalAgent:
    """RAG retrieval over Pinecone when configured, with local hybrid fallback."""

    def __init__(self) -> None:
        self.logger = get_logger("retrieval")
        self._pinecone: PineconeTextIndexClient | None = None
        self._local_retriever = LocalLegalRetriever()
        cfg = PineconeConfig.from_env()
        if cfg:
            self._pinecone = PineconeTextIndexClient(cfg)

    @staticmethod
    def _build_query_bundle(extracted: ExtractedCase) -> list[str]:
        queries = extracted.facts + extracted.active_measures + extracted.risk_factors
        combined = " ".join(
            part for part in extracted.facts + extracted.active_measures + extracted.risk_factors if part
        ).strip()
        if combined:
            queries.append(combined)
            queries.append(
                f"{combined} jurisprudencia de violencia familiar La Matanza Provincia de Buenos Aires"
            )

        if extracted.origin_jurisdiction == "Ciudad Autónoma de Buenos Aires":
            queries.append(
                " ".join(
                    part
                    for part in [
                        combined,
                        "normativa",
                        "protocolos",
                        "Ciudad Autónoma de Buenos Aires",
                        "OVD",
                    ]
                    if part
                ).strip()
            )
        seen: set[str] = set()
        deduped: list[str] = []
        for q in queries:
            normalized = q.strip().lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(q.strip())
        return deduped[:7]

    def _build_query_bundle_with_llm(
        self,
        extracted: ExtractedCase,
        provider: str,
        model: str | None = None,
    ) -> list[str]:
        assert_safe_for_external_llm(extracted)
        _, client = build_json_llm_client(provider=provider, model=model)
        system_prompt = (
            "Sos un asistente juridico argentino. Recibis un caso ya anonimizado. "
            "Tu unica tarea es proponer consultas de busqueda para recuperar normativa, "
            "jurisprudencia y criterios de medidas cautelares. "
            "No inventes hechos y no agregues datos personales. "
            "Respondé exclusivamente en JSON con la clave queries."
        )
        user_prompt = json.dumps(
            {
                "task": "Generar consultas de recuperacion del marco juridico.",
                "case": {
                    "case_id": extracted.case_id,
                    "anonymized_text": extracted.anonymized_text[:2500],
                    "facts": extracted.facts,
                    "active_measures": extracted.active_measures,
                    "risk_factors": extracted.risk_factors,
                    "timeline": extracted.timeline,
                },
                "instructions": {
                    "max_queries": 5,
                    "focus": [
                        "normativa aplicable",
                        "medidas cautelares compatibles con los hechos",
                        "factores de riesgo y urgencia",
                    ],
                    "return_json_only": True,
                },
            },
            ensure_ascii=False,
        )
        response = client.chat_json(system_prompt=system_prompt, user_prompt=user_prompt)
        queries = response.get("queries", [])
        if not isinstance(queries, list):
            return self._build_query_bundle(extracted)

        seen: set[str] = set()
        deduped: list[str] = []
        for item in queries:
            query = str(item).strip()
            normalized = query.lower()
            if not query or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(query)
        return deduped[:5] or self._build_query_bundle(extracted)

    def invoke(
        self,
        extracted: ExtractedCase,
        provider: str | None = None,
        model: str | None = None,
    ) -> list[RetrievalResult]:
        queries = self._build_query_bundle(extracted)
        if provider:
            try:
                queries = self._build_query_bundle_with_llm(extracted, provider=provider, model=model)
            except Exception:
                # Keep retrieval resilient: use the local deterministic query bundle.
                queries = self._build_query_bundle(extracted)
        results: list[RetrievalResult] = []
        for q in queries:
            hits = self._search_with_pinecone_or_fallback(q)
            results.append(RetrievalResult(query=q, hits=hits[:3]))
        return results

    def _search_with_pinecone_or_fallback(self, query: str) -> list[Citation]:
        if self._pinecone:
            try:
                pine_hits = self._pinecone.search_text(query_text=query, top_k=5)
                citations: list[Citation] = []
                for h in pine_hits:
                    fields = h.get("fields", {}) if isinstance(h, dict) else {}
                    excerpt = fields.get("chunk_text") or fields.get("text") or ""
                    source_id = fields.get("source_id") or h.get("_id") or "pinecone_record"
                    score = float(h.get("_score", 0.0))
                    citations.append(
                        Citation(
                            source_id=str(source_id),
                            excerpt=str(excerpt),
                            title=str(fields.get("title", "")).strip() or None,
                            law=str(fields.get("law", "")).strip() or None,
                            article=str(fields.get("article", "")).strip() or None,
                            source_type=str(fields.get("source_type", "")).strip() or None,
                            jurisdiction=str(fields.get("jurisdiction", "")).strip() or None,
                            official_url=(
                                str(fields.get("official_source_url", "")).strip()
                                or str(fields.get("official_url", "")).strip()
                                or None
                            ),
                            summary=str(fields.get("summary", "")).strip() or None,
                            relevance_score=max(0.0, min(1.0, score)),
                        )
                    )
                if citations:
                    return citations
            except Exception as exc:
                self.logger.warning("pinecone_search_failed query='{}' error={}", query, str(exc))

        local_hits = self._local_retriever.search(query, top_k=5)
        if local_hits:
            self.logger.info("retrieval_local_fallback query='{}' hits={}", query, len(local_hits))
        return local_hits
