from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from src.scripts.build_measure_corpus import load_external_corpora, resolve_external_corpus_dirs


class BuildMeasureCorpusTests(unittest.TestCase):
    def test_resolve_external_corpus_dirs_supports_normativas_plural(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data" / "legal" / "normativas").mkdir(parents=True)
            (root / "data" / "legal" / "protocolos").mkdir(parents=True)
            resolved = resolve_external_corpus_dirs(root)

            self.assertIn(root / "data" / "legal" / "normativas", resolved)
            self.assertIn(root / "data" / "legal" / "protocolos", resolved)

    def test_load_external_corpora_merges_normativas_protocolos_and_jurisprudencia(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_cwd = Path.cwd()
            os.chdir(root)
            try:
                normativas = root / "data" / "legal" / "normativas"
                protocolos = root / "data" / "legal" / "protocolos"
                jurisprudencia = root / "data" / "legal" / "jurisprudencia"
                normativas.mkdir(parents=True)
                protocolos.mkdir(parents=True)
                jurisprudencia.mkdir(parents=True)

                (normativas / "leyes.jsonl").write_text(
                    json.dumps(
                        {
                            "doc_id": "ley_1",
                            "source_type": "normativa",
                            "title": "Ley 1",
                            "summary": "Resumen ley 1",
                            "text": "Texto ley 1",
                        },
                        ensure_ascii=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                (protocolos / "protocolos.jsonl").write_text(
                    json.dumps(
                        {
                            "doc_id": "proto_1",
                            "source_type": "protocolo",
                            "title": "Protocolo 1",
                            "summary": "Resumen protocolo 1",
                            "text": "Texto protocolo 1",
                        },
                        ensure_ascii=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                (jurisprudencia / "fallos.jsonl").write_text(
                    json.dumps(
                        {
                            "doc_id": "fallo_1",
                            "source_type": "jurisprudencia",
                            "embedding_text": "Criterio jurisprudencial 1",
                        },
                        ensure_ascii=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )

                rows = load_external_corpora(
                    [normativas, protocolos, jurisprudencia]
                )
            finally:
                os.chdir(old_cwd)

            self.assertEqual(len(rows), 3)
            by_id = {row["_id"]: row for row in rows}
            self.assertEqual(by_id["ley_1"]["source_file"], "data/legal/normativas/leyes.jsonl")
            self.assertEqual(by_id["proto_1"]["source_file"], "data/legal/protocolos/protocolos.jsonl")
            self.assertEqual(by_id["fallo_1"]["text"], "Criterio jurisprudencial 1")


if __name__ == "__main__":
    unittest.main()
