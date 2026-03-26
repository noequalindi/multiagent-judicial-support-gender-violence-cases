from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from src.orchestration.batch_runner import BatchConfig, list_pdfs, process_one_pdf, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Carpeta con denuncias PDF.")
    parser.add_argument("--out-dir", default=None, help="Carpeta de salida para artefactos.")
    parser.add_argument("--glob", default="*.pdf", help="Patron de PDFs a procesar.")
    parser.add_argument("--workers", type=int, default=min(8, (os.cpu_count() or 4)))
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
    out_dir = Path(args.out_dir).resolve() if args.out_dir else Path("data/processed") / f"batch_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

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

    files = list_pdfs(cfg.input_dir, cfg.glob_pattern)
    if not files:
        print("No PDF files found.")
        return

    rows: list[dict[str, object]] = []

    with ThreadPoolExecutor(max_workers=cfg.workers) as executor:
        futures = {executor.submit(process_one_pdf, pdf, cfg): pdf for pdf in files}
        for future in as_completed(futures):
            pdf = futures[future]
            try:
                row = future.result()
                rows.append(row)
                print(f"[OK] {pdf.name} -> template={row['selected_template_id']} chunks={row['chunks']}")
            except Exception as exc:
                error_row = {"status": "error", "pdf_path": str(pdf), "error": str(exc)}
                rows.append(error_row)
                print(f"[ERROR] {pdf.name}: {exc}")

    manifest_path, summary = write_manifest(out_dir, rows)
    print(f"Batch done. processed={summary['processed']} failed={summary['failed']}")
    print(f"Manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
