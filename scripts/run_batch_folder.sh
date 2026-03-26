#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
INPUT_DIR="${INPUT_DIR:-${1:-}}"
OUT_DIR="${OUT_DIR:-}"
WORKERS="${WORKERS:-4}"
OCR_BACKEND="${OCR_BACKEND:-tesseract}"
TESSERACT_LANG="${TESSERACT_LANG:-spa}"
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.2-vision}"
CHUNK_SIZE="${CHUNK_SIZE:-1200}"
OVERLAP="${OVERLAP:-200}"
UPSERT_PINECONE="${UPSERT_PINECONE:-0}"
PINECONE_BATCH_SIZE="${PINECONE_BATCH_SIZE:-96}"

if [[ -z "${INPUT_DIR}" ]]; then
  echo "Uso: INPUT_DIR=/ruta/denuncias bash scripts/run_batch_folder.sh"
  echo "  o  bash scripts/run_batch_folder.sh /ruta/denuncias"
  exit 1
fi

CMD=(
  "${PYTHON_BIN}" -m src.scripts.process_batch_folder
  --input-dir "${INPUT_DIR}"
  --workers "${WORKERS}"
  --ocr-backend "${OCR_BACKEND}"
  --tesseract-lang "${TESSERACT_LANG}"
  --ollama-model "${OLLAMA_MODEL}"
  --chunk-size "${CHUNK_SIZE}"
  --overlap "${OVERLAP}"
  --pinecone-batch-size "${PINECONE_BATCH_SIZE}"
)

if [[ -n "${OUT_DIR}" ]]; then
  CMD+=(--out-dir "${OUT_DIR}")
fi

if [[ "${UPSERT_PINECONE}" == "1" ]]; then
  CMD+=(--upsert-pinecone)
fi

exec "${CMD[@]}"
