PYTHON ?= python3
PDF ?=
INPUT_DIR ?=
WORKERS ?= 4
OCR_BACKEND ?= tesseract
TESSERACT_LANG ?= spa
OLLAMA_MODEL ?= llama3.2-vision
LEGAL_CATALOG ?= data/legal/templates/measure_corpus_catalog.json
LEGAL_CORPUS ?= data/legal/templates/measure_corpus.jsonl
PINECONE_BATCH_SIZE ?= 50

.PHONY: help run ocr-chunk batch build-legal-corpus upsert-legal-corpus api ui

help:
	@echo "Targets disponibles:"
	@echo "  make run PDF=/ruta/denuncia.pdf"
	@echo "  make ocr-chunk PDF=/ruta/denuncia.pdf"
	@echo "  make batch INPUT_DIR=/ruta/denuncias"
	@echo "  make build-legal-corpus"
	@echo "  make upsert-legal-corpus"
	@echo "  make api"
	@echo "  make ui"

run:
	$(PYTHON) -m src.main --pdf "$(PDF)" --ocr-backend "$(OCR_BACKEND)" --tesseract-lang "$(TESSERACT_LANG)" --ollama-model "$(OLLAMA_MODEL)"

ocr-chunk:
	$(PYTHON) -m src.scripts.ocr_and_chunk_pdf --pdf "$(PDF)" --backend "$(OCR_BACKEND)" --tesseract-lang "$(TESSERACT_LANG)" --ollama-model "$(OLLAMA_MODEL)"

batch:
	$(PYTHON) -m src.scripts.process_batch_folder --input-dir "$(INPUT_DIR)" --workers "$(WORKERS)" --ocr-backend "$(OCR_BACKEND)" --tesseract-lang "$(TESSERACT_LANG)" --ollama-model "$(OLLAMA_MODEL)"

build-legal-corpus:
	$(PYTHON) -m src.scripts.build_measure_corpus --catalog "$(LEGAL_CATALOG)" --out "$(LEGAL_CORPUS)"

upsert-legal-corpus:
	$(PYTHON) -m src.scripts.upsert_chunks_to_pinecone --jsonl "$(LEGAL_CORPUS)" --batch-size "$(PINECONE_BATCH_SIZE)"

api:
	$(PYTHON) -m uvicorn src.api.app:app --reload

ui:
	$(PYTHON) -m streamlit run streamlit_app.py
