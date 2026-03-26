#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
LEGAL_CORPUS="${LEGAL_CORPUS:-data/legal/templates/measure_corpus.jsonl}"
PINECONE_BATCH_SIZE="${PINECONE_BATCH_SIZE:-50}"

exec "${PYTHON_BIN}" -m src.scripts.upsert_chunks_to_pinecone \
  --jsonl "${LEGAL_CORPUS}" \
  --batch-size "${PINECONE_BATCH_SIZE}"
