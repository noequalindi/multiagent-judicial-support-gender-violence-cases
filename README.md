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

## Make Targets

The project includes a `Makefile` for common actions:

```bash
make help
```

Main targets:

- `make run PDF=/ruta/denuncia.pdf`
- `make ocr-chunk PDF=/ruta/denuncia.pdf`
- `make batch INPUT_DIR=/ruta/denuncias`
- `make build-legal-corpus`
- `make upsert-legal-corpus`
- `make api`
- `make ui`

## Local Run (CLI Demo)

```bash
cd violence_judicial_files_classification
python -m src.main
```

The demo processes a sample case file and returns a structured JSON payload.

## OCR From PDF

```bash
python -m src.main --pdf "/path/to/denuncia.pdf" --ocr-backend tesseract --tesseract-lang spa
```

## Legal Corpus

The retrieval layer uses:

- `data/legal/templates/measure_corpus_catalog.json` as the editable source of truth
- `data/legal/templates/measure_corpus.jsonl` as the generated Pinecone/local retrieval corpus

Generate the legal corpus:

```bash
make build-legal-corpus
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
```

Upsert the legal corpus:

```bash
make upsert-legal-corpus
```

You can still generate OCR chunks from complaints and upsert them if needed:

```bash
python -m src.scripts.ocr_and_chunk_pdf --pdf "/path/to/denuncia.pdf" --backend tesseract
python -m src.scripts.upsert_chunks_to_pinecone --jsonl data/processed/denuncia_chunks.jsonl
```

After this, `RetrievalAgent` automatically uses Pinecone when env vars are present.

## Concurrent Batch Processing (Folder)

Process many PDF case files in parallel:

```bash
python -m src.scripts.process_batch_folder \
  --input-dir "/path/to/denuncias" \
  --workers 4 \
  --ocr-backend tesseract \
  --chunk-size 1200 \
  --overlap 200
```

With Pinecone upsert during batch:

```bash
python -m src.scripts.process_batch_folder \
  --input-dir "/path/to/denuncias" \
  --workers 4 \
  --ocr-backend tesseract \
  --upsert-pinecone
```

Outputs per file:

- `*_ocr.txt`
- `*_chunks.jsonl`
- `*_result.json`
- `manifest.jsonl` (batch summary)

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
make api
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
make ui
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
