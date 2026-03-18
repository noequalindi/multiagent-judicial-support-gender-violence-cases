# OCR + Vector Store Guide (Judicial Spanish PDFs)

## 1) Do you need Pinecone right now?

Short answer: **no**, not for the first phase.

Use local-first for initial development:

1. OCR scanned PDFs
2. Chunk text
3. Test retrieval locally
4. Validate legal output + human review workflow

Use Pinecone when you have:

- many files (thousands of chunks),
- multi-user concurrent querying,
- managed cloud indexing/search requirements.

## 2) Current OCR options in this repo

### A) Tesseract (already available locally)

Works offline and is easy to automate, but scanned forms may have noisy output.

Run full pipeline from PDF:

```bash
python -m src.main --pdf "/path/denuncia.pdf" --ocr-backend tesseract --tesseract-lang spa
```

Run OCR + chunk export:

```bash
python -m src.scripts.ocr_and_chunk_pdf --pdf "/path/denuncia.pdf" --backend tesseract
```

### B) Ollama vision backend (optional)

The OCR backend is already implemented in code. It needs a **vision-capable** model in your Ollama runtime.

Run:

```bash
python -m src.main --pdf "/path/denuncia.pdf" --ocr-backend ollama --ollama-model "llama3.2-vision"
```

## 3) Suggested open-source models (practical shortlist)

For scanned judicial PDFs in Spanish:

1. **Llama 3.2 Vision (Ollama)** for quick multimodal local tests.
2. **Qwen2.5-VL (Ollama/HF variants)** for stronger document understanding.
3. **PaddleOCR** if you want a dedicated OCR pipeline (non-LLM).
4. **Tesseract (spa)** as baseline fallback.

## 4) Recommended rollout

1. Keep Tesseract as guaranteed fallback.
2. Add one Ollama vision model and compare OCR quality in 20 real denunciations.
3. If quality improves, use Ollama as primary OCR and Tesseract fallback.
4. Keep Pinecone optional until corpus scale justifies managed vector infra.

