from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

from src.llm.client_factory import build_json_llm_client
from src.llm.safety import assert_safe_for_external_llm
from src.models.contracts import Citation, ExtractedCase, RetrievalResult
from src.rag.pinecone_client import PineconeConfig, PineconeTextIndexClient


class RetrievalAgent:
    """RAG retrieval over Pinecone when configured, with lexical fallback on the local legal corpus."""

    def __init__(self) -> None:
        self._pinecone: PineconeTextIndexClient | None = None
        self._local_corpus = self._load_local_corpus()
        cfg = PineconeConfig.from_env()
        if cfg:
            self._pinecone = PineconeTextIndexClient(cfg)

    @staticmethod
    def _stem(token: str) -> str:
        for suffix in ("mientos", "miento", "ciones", "cion", "mente", "es", "s"):
            if token.endswith(suffix) and len(token) > len(suffix) + 2:
                return token[: -len(suffix)]
        return token

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        normalized = "".join(
            ch
            for ch in unicodedata.normalize("NFD", (text or "").lower())
            if unicodedata.category(ch) != "Mn"
        )
        raw_tokens = re.findall(r"[a-z0-9]+", normalized)
        stopwords = {"la", "el", "los", "las", "de", "del", "y", "en", "se", "ante", "con", "para"}
        tokens = [tok for tok in raw_tokens if tok not in stopwords]
        return {RetrievalAgent._stem(tok) for tok in tokens}

    @staticmethod
    def _corpus_path() -> Path:
        configured = os.getenv("LEGAL_CORPUS_PATH", "").strip()
        if configured:
            return Path(configured).expanduser().resolve()
        return Path(__file__).resolve().parents[2] / "data" / "legal" / "templates" / "measure_corpus.jsonl"

    @classmethod
    def _load_local_corpus(cls) -> list[dict[str, Any]]:
        path = cls._corpus_path()
        if not path.exists():
            return []

        rows: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    continue

                base_text = " ".join(
                    str(row.get(field, "")).strip()
                    for field in ("title", "summary", "text", "law", "article", "measure_type")
                    if str(row.get(field, "")).strip()
                )
                row["_search_text"] = base_text
                rows.append(row)
        return rows

    @staticmethod
    def _build_query_bundle(extracted: ExtractedCase) -> list[str]:
        queries = extracted.facts + extracted.active_measures + extracted.risk_factors
        combined = " ".join(
            part for part in extracted.facts + extracted.active_measures + extracted.risk_factors if part
        ).strip()
        if combined:
            queries.append(combined)
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
                            relevance_score=max(0.0, min(1.0, score)),
                        )
                    )
                if citations:
                    return citations
            except Exception:
                # Keep system resilient: fallback to local lexical corpus.
                pass

        hits = []
        q_terms = self._tokenize(query)
        for doc in self._local_corpus:
            d_terms = self._tokenize(str(doc.get("_search_text", "")))
            overlap = len(q_terms.intersection(d_terms))
            if overlap == 0:
                continue
            score = overlap / max(1, len(q_terms))
            if doc.get("source_type") == "template":
                score += 0.05
            if doc.get("measure_type") and doc.get("measure_type") in query.lower():
                score += 0.05

            excerpt = str(doc.get("text") or doc.get("summary") or "")[:500]
            hits.append(
                Citation(
                    source_id=str(doc.get("_id") or doc.get("source_id") or "local_record"),
                    excerpt=excerpt,
                    title=str(doc.get("title", "")).strip() or None,
                    law=str(doc.get("law", "")).strip() or None,
                    article=str(doc.get("article", "")).strip() or None,
                    source_type=str(doc.get("source_type", "")).strip() or None,
                    jurisdiction=str(doc.get("jurisdiction", "")).strip() or None,
                    relevance_score=round(max(0.0, min(1.0, score)), 3),
                )
            )
        hits.sort(key=lambda x: x.relevance_score, reverse=True)
        return hits
