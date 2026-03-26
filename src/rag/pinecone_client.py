from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class PineconeConfig:
    api_key: str
    host: str
    namespace: str = "__default__"
    api_version: str = "2025-10"
    text_field: str = "text"
    embedding_model: str | None = None

    @classmethod
    def from_env(cls) -> "PineconeConfig | None":
        api_key = os.getenv("PINECONE_API_KEY", "").strip()
        host = os.getenv("PINECONE_INDEX_HOST", "").strip()
        if not api_key or not host:
            return None
        return cls(
            api_key=api_key,
            host=host.replace("https://", "").rstrip("/"),
            namespace=os.getenv("PINECONE_NAMESPACE", "__default__").strip() or "__default__",
            api_version=os.getenv("PINECONE_API_VERSION", "2025-10").strip() or "2025-10",
            text_field=os.getenv("PINECONE_TEXT_FIELD", "text").strip() or "text",
            embedding_model=os.getenv("PINECONE_EMBEDDING_MODEL", "").strip() or None,
        )


class PineconeTextIndexClient:
    """Minimal REST client for Pinecone integrated-embedding text indexes."""

    def __init__(self, config: PineconeConfig):
        self.cfg = config
        self.base = f"https://{self.cfg.host}"
        self.headers = {
            "Api-Key": self.cfg.api_key,
            "X-Pinecone-Api-Version": self.cfg.api_version,
        }

    def search_text(self, query_text: str, top_k: int = 5) -> list[dict[str, Any]]:
        url = f"{self.base}/records/namespaces/{self.cfg.namespace}/search"
        payload = {
            "query": {
                "inputs": {"text": query_text},
                "top_k": top_k,
            },
            "fields": [
                self.cfg.text_field,
                "source_pdf",
                "source_id",
                "category",
                "source_type",
                "title",
                "summary",
                "law",
                "article",
                "jurisdiction",
            ],
        }
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, headers={**self.headers, "Content-Type": "application/json"}, json=payload)
            resp.raise_for_status()
            data = resp.json()

        result = data.get("result", {}) if isinstance(data, dict) else {}
        hits = result.get("hits", []) if isinstance(result, dict) else []
        return hits if isinstance(hits, list) else []

    def upsert_records(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        if not records:
            return {"upserted": 0}
        url = f"{self.base}/records/namespaces/{self.cfg.namespace}/upsert"
        ndjson = "\n".join(json.dumps(rec, ensure_ascii=False) for rec in records)
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                url,
                headers={**self.headers, "Content-Type": "application/x-ndjson"},
                content=ndjson.encode("utf-8"),
            )
            resp.raise_for_status()
            return resp.json() if resp.content else {"status": "ok"}
