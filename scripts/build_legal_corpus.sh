#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
LEGAL_CATALOG="${LEGAL_CATALOG:-data/legal/templates/measure_corpus_catalog.jsonl}"
LEGAL_CORPUS="${LEGAL_CORPUS:-data/legal/templates/measure_corpus.jsonl}"
LEGAL_MANIFEST="${LEGAL_MANIFEST:-data/legal/templates/measure_corpus.manifest.json}"

exec "${PYTHON_BIN}" -m src.scripts.build_measure_corpus \
  --catalog "${LEGAL_CATALOG}" \
  --out "${LEGAL_CORPUS}" \
  --manifest-out "${LEGAL_MANIFEST}"
