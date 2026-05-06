from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.agents.ingestion_agent import IngestionAgent
from src.rag.chunking import chunk_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Ruta al PDF escaneado.")
    parser.add_argument("--out", default="data/processed", help="Directorio de salida.")
    parser.add_argument("--backend", default="tesseract", choices=["tesseract", "ollama", "hybrid"])
    parser.add_argument("--tesseract-lang", default="spa")
    parser.add_argument("--ollama-model", default="llama3.2-vision")
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--overlap-tokens", type=int, default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--tokenizer-name", default=None)
    args = parser.parse_args()

    ingestion = IngestionAgent()
    result = ingestion.invoke_pdf(
        pdf_path=args.pdf,
        ocr_backend=args.backend,
        tesseract_lang=args.tesseract_lang,
        ollama_model=args.ollama_model,
    )
    text = result["anonymized_text"]
    chunks = chunk_text(
        text,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
        embedding_model=args.embedding_model or os.getenv("PINECONE_EMBEDDING_MODEL"),
        tokenizer_name=args.tokenizer_name or os.getenv("CHUNK_TOKENIZER_NAME"),
    )

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    base = Path(args.pdf).stem

    ocr_path = out_dir / f"{base}_ocr.txt"
    chunks_path = out_dir / f"{base}_chunks.jsonl"

    ocr_path.write_text(text, encoding="utf-8")
    with open(chunks_path, "w", encoding="utf-8") as f:
        for i, c in enumerate(chunks):
            row = {"chunk_id": f"{base}_{i:04d}", "source_pdf": str(Path(args.pdf)), "text": c}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"OCR saved to: {ocr_path}")
    print(f"Chunks saved to: {chunks_path}")
    print(f"Total chunks: {len(chunks)}")


if __name__ == "__main__":
    main()
