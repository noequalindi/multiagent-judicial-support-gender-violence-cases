PIPELINE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "case_id": {"type": "string"},
        "extracted_case": {"type": "object"},
        "retrieval": {"type": "array"},
        "draft": {
            "type": "object",
            "properties": {
                "decision_draft": {"type": "string", "minLength": 1},
                "checklist": {"type": "array", "items": {"type": "string"}},
                "suggested_measures": {"type": "array", "items": {"type": "string"}},
                "selected_template_id": {"type": ["string", "null"]},
                "selected_template_name": {"type": ["string", "null"]},
                "template_text": {"type": ["string", "null"]},
                "fields_to_fill": {"type": "array", "items": {"type": "string"}},
                "risk_band": {"type": "string", "enum": ["low", "medium", "high"]},
                "abstention": {"type": "boolean"},
                "abstention_reason": {"type": ["string", "null"]},
                "citations": {"type": "array"},
            },
            "required": ["decision_draft", "risk_band", "abstention", "citations"],
        },
        "alerts": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["case_id", "extracted_case", "retrieval", "draft", "alerts"],
}
