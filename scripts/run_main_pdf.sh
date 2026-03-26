#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
PDF_PATH="${PDF_PATH:-${1:-}}"
OCR_BACKEND="${OCR_BACKEND:-tesseract}"
TESSERACT_LANG="${TESSERACT_LANG:-spa}"
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.2-vision}"

if [[ -z "${PDF_PATH}" ]]; then
  echo "Uso: PDF_PATH=/ruta/denuncia.pdf bash scripts/run_main_pdf.sh"
  echo "  o  bash scripts/run_main_pdf.sh /ruta/denuncia.pdf"
  exit 1
fi

exec "${PYTHON_BIN}" -m src.main \
  --pdf "${PDF_PATH}" \
  --ocr-backend "${OCR_BACKEND}" \
  --tesseract-lang "${TESSERACT_LANG}" \
  --ollama-model "${OLLAMA_MODEL}"
