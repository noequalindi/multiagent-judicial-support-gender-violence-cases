from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from src.storage.local_json_store import LocalClassificationStore, build_classified_case_payload


class LocalJsonStoreTests(unittest.TestCase):
    def test_build_payload_keeps_frontend_shape_and_metadata(self) -> None:
        payload = build_classified_case_payload(
            route="/classify-measure-pdf",
            filename="caso.pdf",
            ocr_backend="tesseract",
            tesseract_lang="spa",
            ollama_model="llama3.2-vision",
            provider="anthropic",
            model="claude-sonnet-4-6",
            pipeline_result={"case_id": "FVODO02452-0050264/2026"},
            classification={"selected_template_id": "medida_exclusion", "risk_level": "alto"},
        )

        self.assertEqual(payload["case_id"], "FVODO02452-0050264/2026")
        self.assertEqual(payload["provider"], "anthropic")
        self.assertEqual(payload["model"], "claude-sonnet-4-6")
        self.assertEqual(payload["route"], "/classify-measure-pdf")
        self.assertEqual(payload["filename"], "caso.pdf")
        self.assertEqual(payload["runtime"]["ocr_backend"], "tesseract")

    def test_save_case_writes_json_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_dir = os.environ.get("LOCAL_CLASSIFIED_CASES_DIR")
            os.environ["LOCAL_CLASSIFIED_CASES_DIR"] = tmp
            try:
                store = LocalClassificationStore()
                saved_path = store.save_case(
                    {
                        "provider": "anthropic",
                        "model": "claude-sonnet-4-6",
                        "selected_template_id": "medida_exclusion",
                    },
                    case_id="FVODO02452-0050264/2026",
                )
            finally:
                if old_dir is None:
                    os.environ.pop("LOCAL_CLASSIFIED_CASES_DIR", None)
                else:
                    os.environ["LOCAL_CLASSIFIED_CASES_DIR"] = old_dir

            self.assertTrue(saved_path.exists())
            self.assertIn("FVODO02452-0050264_2026", saved_path.name)

            payload = json.loads(saved_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["provider"], "anthropic")

            manifest_path = Path(tmp) / "manifest.jsonl"
            self.assertTrue(manifest_path.exists())
            lines = manifest_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            manifest_row = json.loads(lines[0])
            self.assertEqual(manifest_row["case_id"], "FVODO02452-0050264/2026")


if __name__ == "__main__":
    unittest.main()
