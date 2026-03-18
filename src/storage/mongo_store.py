from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

try:
    from pymongo import MongoClient
except ModuleNotFoundError:  # pragma: no cover
    MongoClient = None  # type: ignore


class MongoRunStore:
    def __init__(self) -> None:
        self.uri = os.getenv("MONGODB_URI", "").strip()
        self.db_name = os.getenv("MONGODB_DB", "violence_judicial_ai").strip() or "violence_judicial_ai"
        self.collection_name = os.getenv("MONGODB_COLLECTION", "processing_runs").strip() or "processing_runs"

    def enabled(self) -> bool:
        return bool(self.uri and MongoClient is not None)

    def save_run(self, payload: dict[str, Any]) -> str | None:
        if not self.enabled():
            return None
        client = MongoClient(self.uri)
        try:
            result = client[self.db_name][self.collection_name].insert_one(
                {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    **payload,
                }
            )
            return str(result.inserted_id)
        finally:
            client.close()


def build_run_payload(
    *,
    route: str,
    filename: str | None = None,
    ocr_backend: str | None = None,
    tesseract_lang: str | None = None,
    ollama_model: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    pipeline_result: dict[str, Any] | None = None,
    classification: dict[str, Any] | None = None,
    generated_draft: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "route": route,
        "filename": filename,
        "runtime": {
            "ocr_backend": ocr_backend,
            "tesseract_lang": tesseract_lang,
            "ollama_model": ollama_model,
            "provider": provider,
            "model": model,
        },
        "pipeline_result": pipeline_result,
        "classification": classification,
        "generated_draft": generated_draft,
        "error": error,
    }
