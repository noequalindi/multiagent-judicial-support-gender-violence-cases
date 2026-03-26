from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.orchestration.batch_runner import BatchConfig
from src.orchestration.langgraph_batch import build_batch_graph, run_batch_graph


class LangGraphBatchTests(unittest.TestCase):
    def test_build_batch_graph_requires_langgraph(self) -> None:
        with patch("src.orchestration.langgraph_batch._LANGGRAPH_IMPORT_ERROR", ImportError("missing")):
            with self.assertRaises(RuntimeError):
                build_batch_graph()

    def test_run_batch_graph_collects_rows_and_manifest(self) -> None:
        try:
            build_batch_graph()
        except RuntimeError:
            self.skipTest("langgraph is not installed in the current environment")

        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp) / "incoming"
            out_dir = Path(tmp) / "processed"
            input_dir.mkdir()
            (input_dir / "case_a.pdf").write_bytes(b"%PDF-1.4")
            (input_dir / "case_b.pdf").write_bytes(b"%PDF-1.4")

            cfg = BatchConfig(
                input_dir=input_dir,
                out_dir=out_dir,
                glob_pattern="*.pdf",
                workers=2,
                ocr_backend="tesseract",
                tesseract_lang="spa",
                ollama_model="llama3.2-vision",
                chunk_size=1200,
                overlap=200,
                upsert_pinecone=False,
                pinecone_batch_size=96,
            )

            def fake_process(pdf_path: Path, _cfg: BatchConfig) -> dict[str, object]:
                return {
                    "status": "ok",
                    "pdf_path": str(pdf_path),
                    "selected_template_id": "medida_perimetro",
                    "chunks": 1,
                }

            with patch("src.orchestration.langgraph_batch.process_one_pdf_safe", side_effect=fake_process):
                result = run_batch_graph(cfg)

            self.assertEqual(result["processed"], 2)
            self.assertEqual(result["failed"], 0)
            self.assertEqual(len(result["results"]), 2)
            self.assertTrue(Path(result["manifest_path"]).exists())


if __name__ == "__main__":
    unittest.main()
