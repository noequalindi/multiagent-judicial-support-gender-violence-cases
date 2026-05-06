# violence_judicial_files_classification

Multi-agent judicial assistant scaffold for the analysis of gender-violence complaints, retrieval of legal support, and generation of non-binding precautionary-measure drafts with mandatory human review.

Current project capabilities:

- OCR from scanned complaints
- anonymization of sensitive fields
- structured extraction of facts, active measures, risk factors, and timeline
- local or Pinecone-backed legal retrieval
- judicial template selection
- non-binding draft generation
- citation support checks, abstention, and alerts
- FastAPI wrapper
- Streamlit demo UI

## Structure

- `src/agents/`: judicial-domain agents
- `src/orchestration/`: flow coordination
- `src/guardrails/`: output validation and policies
- `src/models/`: agent data contracts
- `src/rag/`: retrieval utilities and Pinecone client
- `src/api/`: FastAPI app
- `docs/`: PDF analysis and architecture mapping
- `data/measures/`: court templates
- `data/legal/`: enriched legal corpus catalog and generated JSONL corpus

## Target Flow

1. Case ingestion and anonymization
2. Structured extraction (facts, measures, risk, timeline)
3. RAG retrieval (regulation and case law)
4. Non-binding draft generation
5. Citation verification and support checks
6. Operational alerts and traceability

## Environment Setup

Create the Conda environment:

```bash
cd violence_judicial_files_classification
conda env create -f environment.yml
conda activate violence-judicial-ai
```

This environment includes:

- Python 3.11
- `tesseract`
- `poppler`
- FastAPI, Streamlit, Pydantic, JSON Schema validation, and HTTP client dependencies

Optional for stronger local anonymization in Spanish:

```bash
python -m spacy download es_core_news_md
```

If the spaCy model is not installed, the pipeline keeps working with the
existing regex and Presidio-based anonymization layers.

## Shell Scripts

Operational commands are exposed through `bash` scripts under `scripts/`:

```bash
ls scripts/
```

Main scripts:

- `bash scripts/run_main_pdf.sh /ruta/denuncia.pdf`
- `bash scripts/run_console_pdf.sh /ruta/denuncia.pdf`
- `bash scripts/run_ocr_chunk.sh /ruta/denuncia.pdf`
- `bash scripts/run_batch_folder.sh /ruta/denuncias`
- `bash scripts/run_batch_langgraph.sh data/incoming`
- `bash scripts/build_legal_corpus.sh`
- `bash scripts/upsert_legal_corpus.sh`
- `bash scripts/generate_synthetic_cases.sh`
- `bash scripts/run_api.sh`
- `bash scripts/run_ui.sh`
- `bash scripts/run_tests.sh`

These scripts accept environment variables such as `PYTHON_BIN`, `OCR_BACKEND`, `TESSERACT_LANG`, `OLLAMA_MODEL`, `INPUT_DIR`, `OUT_DIR`, and `WORKERS`.

Relevant optional anonymization variables:

- `SPACY_ANONYMIZATION_ENABLED=1`
- `SPACY_MODEL=es_core_news_md`

## Make Targets

The `Makefile` is still available as a thin wrapper around those scripts:

```bash
make help
```

Main targets:

- `make run PDF=/ruta/denuncia.pdf`
- `make ocr-chunk PDF=/ruta/denuncia.pdf`
- `make batch INPUT_DIR=/ruta/denuncias`
- `make batch-langgraph INPUT_DIR=data/incoming`
- `make build-legal-corpus`
- `make upsert-legal-corpus`
- `make generate-synthetic-cases`
- `make api`
- `make ui`

## Local Run (CLI Demo)

```bash
cd violence_judicial_files_classification
bash scripts/run_main_pdf.sh /path/to/denuncia.pdf
```

You can still run the Python entrypoint directly if needed.

## OCR From PDF

```bash
PDF_PATH="/path/to/denuncia.pdf" bash scripts/run_main_pdf.sh
```

## Legal Corpus

The retrieval layer uses:

- `data/legal/templates/measure_corpus_catalog.json` as the editable source of truth
- `data/legal/templates/measure_corpus.jsonl` as the generated Pinecone/local retrieval corpus
- curated JSONL files under `data/legal/normativa/`, `data/legal/protocolos/`, and `data/legal/jurisprudencia/` which are merged into the generated retrieval corpus

Generate the legal corpus:

```bash
bash scripts/build_legal_corpus.sh
```

The generated corpus contains:

- judicial templates (`source_type=template`)
- normative references (`source_type=normativa`)
- legal metadata and official sources

## Pinecone Retrieval

Set environment variables:

```bash
export PINECONE_API_KEY="..."
export PINECONE_INDEX_HOST="your-index-host.svc....pinecone.io"
export PINECONE_NAMESPACE="__default__"
export PINECONE_TEXT_FIELD="text"
export PINECONE_API_VERSION="2025-10"
export PINECONE_EMBEDDING_MODEL="multilingual-e5-large"
export CHUNK_MAX_TOKENS="300"
export CHUNK_OVERLAP_TOKENS="50"
# Optional: explicit tokenizer override for future embedding models
export CHUNK_TOKENIZER_NAME=""
```

Upsert the legal corpus:

```bash
bash scripts/upsert_legal_corpus.sh
```

You can still generate OCR chunks from complaints and upsert them if needed:

```bash
python -m src.scripts.ocr_and_chunk_pdf --pdf "/path/to/denuncia.pdf" --backend tesseract
python -m src.scripts.upsert_chunks_to_pinecone --jsonl data/processed/denuncia_chunks.jsonl
```

After this, `RetrievalAgent` automatically uses Pinecone when env vars are present.

Chunking is token-aware. By default, the project resolves the tokenizer from
`PINECONE_EMBEDDING_MODEL` and uses `CHUNK_MAX_TOKENS` / `CHUNK_OVERLAP_TOKENS`
to keep complaint fragments aligned with the embedding model context window.

If Pinecone is unavailable or a search request fails, retrieval falls back
automatically to a local hybrid mode over the legal corpus, combining BM25 and
lexical scoring. This keeps the pipeline operational without internet access,
although with lower semantic recall than the vector index.

## Concurrent Batch Processing (Folder)

Process many PDF case files in parallel:

```bash
INPUT_DIR="/path/to/denuncias" WORKERS=4 OCR_BACKEND=tesseract \
  bash scripts/run_batch_folder.sh
```

With Pinecone upsert during batch:

```bash
INPUT_DIR="/path/to/denuncias" WORKERS=4 OCR_BACKEND=tesseract UPSERT_PINECONE=1 \
  bash scripts/run_batch_folder.sh
```

Outputs per file:

- `*_ocr.txt`
- `*_chunks.jsonl`
- `*_result.json`
- `manifest.jsonl` (batch summary)

## LangGraph Batch Processing

There is now a LangGraph-based batch runner to test the same complaint-processing pipeline over `data/incoming` or any other folder.

```bash
bash scripts/run_batch_langgraph.sh data/incoming
```

Equivalent direct command:

```bash
INPUT_DIR="data/incoming" WORKERS=4 OCR_BACKEND=tesseract \
  bash scripts/run_batch_langgraph.sh
```

The graph performs:

1. PDF discovery
2. per-file OCR + anonymization + orchestration
3. manifest generation with success/error rows

Outputs are written under `data/processed/langgraph_batch_<timestamp>/`.

## Synthetic Complaint Generation with Claude

There is now a dedicated synthetic-data generation module that uses Claude to
create new complaint narratives from a small set of real anonymized seed cases
with known applied measures.

Current seeds are declared in:

- `data/synthetic/seed_cases.json`

The default setup uses three court-provided complaint examples:

- `caso_exclusion.pdf` -> `medida_exclusion`
- `caso_perimetro.pdf` -> `medida_perimetro`
- `caso_perimetro_1.pdf` -> `medida_perimetro`

The generator is intentionally scoped to La Matanza / Provincia de Buenos Aires
and does not use the CABA case-law PDFs as synthetic complaint seeds.

Run it with:

```bash
bash scripts/generate_synthetic_cases.sh
```

Or with explicit parameters:

```bash
SEED_MANIFEST="data/synthetic/seed_cases.json" \
OUT_JSONL="data/synthetic/synthetic_cases.jsonl" \
PER_SEED=5 \
CLAUDE_MODEL="claude-sonnet-4-6" \
bash scripts/generate_synthetic_cases.sh
```

Equivalent `make` target:

```bash
make generate-synthetic-cases PER_SEED=5
```

The output is a JSONL dataset of synthetic complaints, each one tagged with:

- synthetic identifier
- seed provenance
- target measure label
- jurisdiction metadata
- violence types
- risk factors
- facts summary
- synthetic complaint text

## Generate Filled Measure Document (TXT + PDF)

From one complaint PDF, select template + fill detectable fields + generate output document:

```bash
python -m src.scripts.generate_measure_document \
  --denuncia-pdf "data/incoming/exp_ejemplo.pdf" \
  --measures-dir "data/measures" \
  --out-dir "data/generated_measures" \
  --ocr-backend tesseract
```

## FastAPI

Run the API:

```bash
bash scripts/run_api.sh
```

Main endpoints:

- `GET /health`
- `POST /process-text`
- `POST /process-pdf`
- `POST /generate-draft`
- `POST /generate-draft-pdf`

## Streamlit UI

Run the demo UI:

```bash
bash scripts/run_ui.sh
```

By default it points to:

- `http://127.0.0.1:8000`

The UI currently supports:

- processing plain text
- processing complaint PDFs
- generating a non-binding draft from a complaint PDF

## Notes

- Sensitive complaint text should be processed locally before calling any external LLM API.
- The legal retrieval corpus is intended for support and traceability, not for autonomous decision-making.
- Judicial templates and legal references should be reviewed by a human operator before operational use.

## External LLM Providers

The API/UI can use external providers for:

- legal-framework retrieval assistance from the anonymized case summary
- closed-list measure classification

Supported environment variables:

```bash
export ANTHROPIC_API_KEY="..."
export ANTHROPIC_MODEL="claude-3-5-sonnet-latest"

export OPENAI_API_KEY="..."
export OPENAI_MODEL="gpt-5-mini"

export DEEPSEEK_API_KEY="..."
export HUGGINGFACE_API_KEY="..."
```

For Anthropic, the `POST /classify-measure` and `POST /classify-measure-pdf` routes now use Claude both to suggest retrieval queries over the local/Pinecone legal corpus and to select the most appropriate judicial template. If the provider fails, retrieval falls back to the deterministic local strategy.

Classified anonymized cases are also stored locally as JSON under `data/classified_cases/` by default, with an append-only `manifest.jsonl`. You can override the output directory with:

```bash
export LOCAL_CLASSIFIED_CASES_DIR="data/mi_export_clasificados"
```

Application logs are stored locally under `logs/app.log` by default. You can adjust verbosity and location with:

```bash
export LOG_LEVEL="INFO"
export LOG_DIR="logs"
```
