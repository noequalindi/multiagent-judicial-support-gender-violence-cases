from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from src.orchestration.batch_runner import BatchConfig
from src.orchestration.langgraph_batch import run_batch_graph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/incoming", help="Carpeta con denuncias PDF.")
    parser.add_argument("--out-dir", default=None, help="Carpeta de salida para artefactos.")
    parser.add_argument("--glob", default="*.pdf", help="Patron de PDFs a procesar.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ocr-backend", default="tesseract", choices=["tesseract", "ollama"])
    parser.add_argument("--tesseract-lang", default="spa")
    parser.add_argument("--ollama-model", default="llama3.2-vision")
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--upsert-pinecone", action="store_true")
    parser.add_argument("--pinecone-batch-size", type=int, default=96)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else Path("data/processed") / f"langgraph_batch_{run_id}"
    )

    cfg = BatchConfig(
        input_dir=input_dir,
        out_dir=out_dir,
        glob_pattern=args.glob,
        workers=max(1, args.workers),
        ocr_backend=args.ocr_backend,
        tesseract_lang=args.tesseract_lang,
        ollama_model=args.ollama_model,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        upsert_pinecone=args.upsert_pinecone,
        pinecone_batch_size=max(1, args.pinecone_batch_size),
    )

    try:
        result = run_batch_graph(cfg)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    rows = result.get("results", [])

    if not rows:
        print("No PDF files found.")
        print(f"Manifest: {Path(result['manifest_path']).resolve()}")
        return

    for row in rows:
        if row.get("status") == "ok":
            print(
                f"[OK] {Path(row['pdf_path']).name} "
                f"-> template={row['selected_template_id']} chunks={row['chunks']}"
            )
        else:
            print(f"[ERROR] {Path(row['pdf_path']).name}: {row['error']}")

    print(f"Batch done. processed={result['processed']} failed={result['failed']}")
    print(f"Manifest: {Path(result['manifest_path']).resolve()}")


if __name__ == "__main__":
    main()
