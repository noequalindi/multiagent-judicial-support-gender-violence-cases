PYTHON ?= python3
PDF ?=
INPUT_DIR ?=
WORKERS ?= 4
OCR_BACKEND ?= tesseract
TESSERACT_LANG ?= spa
OLLAMA_MODEL ?= llama3.2-vision
LEGAL_CATALOG ?= data/legal/templates/measure_corpus_catalog.jsonl
LEGAL_CORPUS ?= data/legal/templates/measure_corpus.jsonl
PINECONE_BATCH_SIZE ?= 50

.PHONY: help run console ocr-chunk batch batch-langgraph build-legal-corpus upsert-legal-corpus api ui test

help:
	@echo "Targets disponibles:"
	@echo "  make run PDF=/ruta/denuncia.pdf"
	@echo "  make console PDF=/ruta/denuncia.pdf"
	@echo "  make ocr-chunk PDF=/ruta/denuncia.pdf"
	@echo "  make batch INPUT_DIR=/ruta/denuncias"
	@echo "  make batch-langgraph INPUT_DIR=data/incoming"
	@echo "  make build-legal-corpus"
	@echo "  make upsert-legal-corpus"
	@echo "  make api"
	@echo "  make ui"
	@echo "  make test"

run:
	PYTHON_BIN="$(PYTHON)" PDF_PATH="$(PDF)" OCR_BACKEND="$(OCR_BACKEND)" TESSERACT_LANG="$(TESSERACT_LANG)" OLLAMA_MODEL="$(OLLAMA_MODEL)" bash scripts/run_main_pdf.sh

console:
	PYTHON_BIN="$(PYTHON)" PDF_PATH="$(PDF)" OCR_BACKEND="$(OCR_BACKEND)" TESSERACT_LANG="$(TESSERACT_LANG)" OLLAMA_MODEL="$(OLLAMA_MODEL)" bash scripts/run_console_pdf.sh

ocr-chunk:
	PYTHON_BIN="$(PYTHON)" PDF_PATH="$(PDF)" OCR_BACKEND="$(OCR_BACKEND)" TESSERACT_LANG="$(TESSERACT_LANG)" OLLAMA_MODEL="$(OLLAMA_MODEL)" bash scripts/run_ocr_chunk.sh

batch:
	PYTHON_BIN="$(PYTHON)" INPUT_DIR="$(INPUT_DIR)" WORKERS="$(WORKERS)" OCR_BACKEND="$(OCR_BACKEND)" TESSERACT_LANG="$(TESSERACT_LANG)" OLLAMA_MODEL="$(OLLAMA_MODEL)" bash scripts/run_batch_folder.sh

batch-langgraph:
	PYTHON_BIN="$(PYTHON)" INPUT_DIR="$(INPUT_DIR)" WORKERS="$(WORKERS)" OCR_BACKEND="$(OCR_BACKEND)" TESSERACT_LANG="$(TESSERACT_LANG)" OLLAMA_MODEL="$(OLLAMA_MODEL)" bash scripts/run_batch_langgraph.sh

build-legal-corpus:
	PYTHON_BIN="$(PYTHON)" LEGAL_CATALOG="$(LEGAL_CATALOG)" LEGAL_CORPUS="$(LEGAL_CORPUS)" bash scripts/build_legal_corpus.sh

upsert-legal-corpus:
	PYTHON_BIN="$(PYTHON)" LEGAL_CORPUS="$(LEGAL_CORPUS)" PINECONE_BATCH_SIZE="$(PINECONE_BATCH_SIZE)" bash scripts/upsert_legal_corpus.sh

api:
	PYTHON_BIN="$(PYTHON)" bash scripts/run_api.sh

ui:
	PYTHON_BIN="$(PYTHON)" bash scripts/run_ui.sh

test:
	PYTHON_BIN="$(PYTHON)" bash scripts/run_tests.sh
