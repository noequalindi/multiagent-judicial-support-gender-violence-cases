PYTHON ?= python3
PDF ?=
INPUT_DIR ?=
WORKERS ?= 4
OCR_BACKEND ?= tesseract
TESSERACT_LANG ?= spa
OLLAMA_MODEL ?= llama3.2-vision
LEGAL_CATALOG ?= data/legal/templates/measure_corpus_catalog.jsonl
LEGAL_CORPUS ?= data/legal/templates/measure_corpus.jsonl
LEGAL_MANIFEST ?= data/legal/templates/measure_corpus.manifest.json
PINECONE_BATCH_SIZE ?= 50

.PHONY: help run console ocr-chunk batch batch-langgraph build-legal-corpus upsert-legal-corpus refresh-legal-corpus generate-synthetic-cases api ui test hash-password benchmark benchmark-model-grid benchmark-experiments benchmark-legal-corpus export-gold export-sft-openai export-sft-qwen export-sft-augmented split-sft create-openai-finetune train-qwen-qlora

help:
	@echo "Targets disponibles:"
	@echo "  make run PDF=/ruta/denuncia.pdf"
	@echo "  make console PDF=/ruta/denuncia.pdf"
	@echo "  make ocr-chunk PDF=/ruta/denuncia.pdf"
	@echo "  make batch INPUT_DIR=/ruta/denuncias"
	@echo "  make batch-langgraph INPUT_DIR=data/incoming"
	@echo "  make build-legal-corpus"
	@echo "  make upsert-legal-corpus"
	@echo "  make refresh-legal-corpus CLEAR_ON_STALE=1"
	@echo "  make generate-synthetic-cases"
	@echo "  make api"
	@echo "  make ui"
	@echo "  make test"
	@echo "  make hash-password PASSWORD=clave"
	@echo "  make benchmark GOLD_PATH=data/evals/gold_cases.sample.jsonl LABEL=baseline"
	@echo "  make benchmark-model-grid GOLD_PATH=data/evals/gold_cases.jsonl MODEL_SPECS=\"openai:gpt-5-mini anthropic:claude-sonnet-4-6@balanced_measures ollama:qwen2.5:7b-instruct\" LABEL_PREFIX=model_grid"
	@echo "  make benchmark-experiments GOLD_PATH=data/evals/gold_cases.jsonl BASELINE_SPECS=\"openai:gpt-5-mini anthropic:claude-sonnet-4-6 ollama:qwen2.5:7b-instruct\" PROMPT_SPECS=\"anthropic:claude-sonnet-4-6@strict_rag anthropic:claude-sonnet-4-6@balanced_measures\" TUNED_SPECS=\"openai:ft:... ollama:qwen2.5-7b-qlora:latest\" LABEL_PREFIX=experiment_matrix"
	@echo "  make benchmark-legal-corpus GOLD_PATH=data/evals/gold_cases.jsonl LABEL=legal_alignment_v1"
	@echo "  make export-gold OUT=data/evals/gold_cases.jsonl VALIDATED_ONLY=1"
	@echo "  make export-sft-openai OUT=data/evals/openai_sft.jsonl VALIDATED_ONLY=1"
	@echo "  make export-sft-qwen OUT=data/evals/qwen_sft.jsonl VALIDATED_ONLY=1"
	@echo "  make export-sft-augmented OUT=data/evals/augmented_sft.jsonl SYNTHETIC_JSONL=data/synthetic/synthetic_cases.jsonl VALIDATED_ONLY=1"
	@echo "  make split-sft INPUT=data/evals/openai_sft.jsonl TRAIN_OUT=data/evals/openai_train.jsonl VALID_OUT=data/evals/openai_valid.jsonl TEST_OUT=data/evals/openai_test.jsonl"
	@echo "  make create-openai-finetune TRAIN_FILE=data/evals/openai_train.jsonl VALID_FILE=data/evals/openai_valid.jsonl MODEL=gpt-4.1-mini-2025-04-14"
	@echo "  make train-qwen-qlora DATASET=data/evals/qwen_sft.jsonl OUTPUT_DIR=artifacts/qwen-qlora MODEL=Qwen/Qwen2.5-7B-Instruct USE_QLORA=1"

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
	PYTHON_BIN="$(PYTHON)" LEGAL_CATALOG="$(LEGAL_CATALOG)" LEGAL_CORPUS="$(LEGAL_CORPUS)" LEGAL_MANIFEST="$(LEGAL_MANIFEST)" bash scripts/build_legal_corpus.sh

upsert-legal-corpus:
	PYTHON_BIN="$(PYTHON)" LEGAL_CORPUS="$(LEGAL_CORPUS)" PINECONE_BATCH_SIZE="$(PINECONE_BATCH_SIZE)" bash scripts/upsert_legal_corpus.sh

refresh-legal-corpus:
	$(PYTHON) -m src.scripts.refresh_legal_corpus --catalog "$(LEGAL_CATALOG)" --jsonl "$(LEGAL_CORPUS)" --manifest "$(LEGAL_MANIFEST)" --batch-size "$(PINECONE_BATCH_SIZE)" $(if $(CLEAR_FIRST),--clear-first,) $(if $(CLEAR_ON_STALE),--clear-on-stale,)

generate-synthetic-cases:
	PYTHON_BIN="$(PYTHON)" SEED_MANIFEST="$(SEED_MANIFEST)" OUT_JSONL="$(OUT_JSONL)" PER_SEED="$(PER_SEED)" CLAUDE_MODEL="$(CLAUDE_MODEL)" bash scripts/generate_synthetic_cases.sh

api:
	PYTHON_BIN="$(PYTHON)" bash scripts/run_api.sh

ui:
	PYTHON_BIN="$(PYTHON)" bash scripts/run_ui.sh

test:
	PYTHON_BIN="$(PYTHON)" bash scripts/run_tests.sh

hash-password:
	$(PYTHON) -m src.scripts.hash_password "$(PASSWORD)"

benchmark:
	$(PYTHON) -m src.scripts.run_benchmark --gold-path "$(GOLD_PATH)" --label "$(LABEL)"

benchmark-model-grid:
	$(PYTHON) -m src.scripts.benchmark_model_grid --gold-path "$(GOLD_PATH)" --label-prefix "$(LABEL_PREFIX)" $(foreach spec,$(MODEL_SPECS),--model-spec "$(spec)")

benchmark-experiments:
	$(PYTHON) -m src.scripts.run_model_experiments --gold-path "$(GOLD_PATH)" --label-prefix "$(LABEL_PREFIX)" $(if $(LIMIT),--limit "$(LIMIT)",) $(foreach spec,$(BASELINE_SPECS),--baseline-spec "$(spec)") $(foreach spec,$(PROMPT_SPECS),--prompt-spec "$(spec)") $(foreach spec,$(TUNED_SPECS),--tuned-spec "$(spec)")

benchmark-legal-corpus:
	$(PYTHON) -m src.scripts.benchmark_legal_corpus_alignment --gold-path "$(GOLD_PATH)" --label "$(LABEL)" $(if $(LIMIT),--limit "$(LIMIT)",)

export-gold:
	$(PYTHON) -m src.scripts.export_gold_from_mongo --out "$(OUT)" $(if $(VALIDATED_ONLY),--validated-only,) $(if $(INCLUDE_UNVALIDATED),--include-unvalidated,)

export-sft-openai:
	$(PYTHON) -m src.scripts.export_finetuning_dataset --out "$(OUT)" --format openai_sft $(if $(VALIDATED_ONLY),--validated-only,) $(if $(LIMIT),--limit "$(LIMIT)",)

export-sft-qwen:
	$(PYTHON) -m src.scripts.export_finetuning_dataset --out "$(OUT)" --format qwen_chat $(if $(VALIDATED_ONLY),--validated-only,) $(if $(LIMIT),--limit "$(LIMIT)",)

export-sft-augmented:
	$(PYTHON) -m src.scripts.export_augmented_finetuning_dataset --out "$(OUT)" --synthetic-jsonl "$(SYNTHETIC_JSONL)" $(if $(VALIDATED_ONLY),--validated-only,) $(if $(LIMIT),--limit "$(LIMIT)",)

split-sft:
	$(PYTHON) -m src.scripts.split_finetuning_dataset --input "$(INPUT)" --train-out "$(TRAIN_OUT)" --valid-out "$(VALID_OUT)" --test-out "$(TEST_OUT)" $(if $(TRAIN_RATIO),--train-ratio "$(TRAIN_RATIO)",) $(if $(VALID_RATIO),--valid-ratio "$(VALID_RATIO)",) $(if $(SEED),--seed "$(SEED)",)

create-openai-finetune:
	$(PYTHON) -m src.scripts.create_openai_finetune_job --train-file "$(TRAIN_FILE)" $(if $(VALID_FILE),--valid-file "$(VALID_FILE)",) $(if $(MODEL),--model "$(MODEL)",) $(if $(SUFFIX),--suffix "$(SUFFIX)",) $(if $(EPOCHS),--epochs "$(EPOCHS)",)

train-qwen-qlora:
	$(PYTHON) -m src.scripts.train_qwen_qlora --dataset "$(DATASET)" --output-dir "$(OUTPUT_DIR)" $(if $(MODEL),--model "$(MODEL)",) $(if $(EPOCHS),--epochs "$(EPOCHS)",) $(if $(USE_QLORA),--use-qlora,)
