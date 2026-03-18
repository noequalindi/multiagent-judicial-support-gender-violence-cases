from __future__ import annotations

from src.guardrails.schemas.pipeline_output_schema import PIPELINE_OUTPUT_SCHEMA


def validate_output(payload: dict) -> None:
    try:
        import jsonschema  # type: ignore
    except Exception:
        # Minimal fallback without external dependency.
        required = {"case_id", "extracted_case", "retrieval", "draft", "alerts"}
        missing = required - set(payload.keys())
        if missing:
            raise ValueError(f"Missing required keys: {sorted(missing)}")
        return

    jsonschema.validate(instance=payload, schema=PIPELINE_OUTPUT_SCHEMA)
