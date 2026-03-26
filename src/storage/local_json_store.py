from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class LocalClassificationStore:
    def __init__(self) -> None:
        self.base_dir = Path(
            os.getenv("LOCAL_CLASSIFIED_CASES_DIR", "data/classified_cases")
        ).expanduser()

    def save_case(self, payload: dict[str, Any], case_id: str) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        safe_case_id = self._safe_file_token(case_id)
        file_path = self.base_dir / f"{timestamp}_{safe_case_id}.json"
        file_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        manifest_path = self.base_dir / "manifest.jsonl"
        with open(manifest_path, "a", encoding="utf-8") as manifest_file:
            manifest_file.write(
                json.dumps(
                    {
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "case_id": case_id,
                        "file_path": str(file_path),
                        "selected_template_id": payload.get("selected_template_id"),
                        "provider": payload.get("provider"),
                        "model": payload.get("model"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        return file_path

    @staticmethod
    def _safe_file_token(value: str) -> str:
        return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in (value or "case"))


def build_classified_case_payload(
    *,
    route: str,
    filename: str | None,
    ocr_backend: str | None,
    tesseract_lang: str | None,
    ollama_model: str | None,
    provider: str | None,
    model: str | None,
    pipeline_result: dict[str, Any],
    classification: dict[str, Any],
) -> dict[str, Any]:
    payload = dict(classification) if isinstance(classification, dict) else {}
    payload["provider"] = provider or payload.get("provider")
    payload["model"] = model or payload.get("model")
    payload["case_id"] = pipeline_result.get("case_id")
    payload["created_at"] = datetime.now(timezone.utc).isoformat()
    payload["route"] = route
    payload["filename"] = filename
    payload["runtime"] = {
        "ocr_backend": ocr_backend,
        "tesseract_lang": tesseract_lang,
        "ollama_model": ollama_model,
        "provider": provider,
        "model": model,
    }
    return payload
