from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

@dataclass
class BatchConfig:
    input_dir: Path
    out_dir: Path
    glob_pattern: str
    workers: int
    ocr_backend: str
    tesseract_lang: str
    ollama_model: str
    chunk_size: int
    overlap: int
    upsert_pinecone: bool
    pinecone_batch_size: int


def list_pdfs(input_dir: Path, glob_pattern: str) -> list[Path]:
    files = sorted(input_dir.rglob(glob_pattern))
    return [p for p in files if p.is_file() and p.suffix.lower() == ".pdf"]


def safe_doc_id(root: Path, file_path: Path) -> str:
    rel = file_path.relative_to(root)
    stem = rel.with_suffix("").as_posix().replace("/", "__")
    return stem.replace(" ", "_")


def batched(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def to_pinecone_records(
    chunks: list[str],
    source_pdf: str,
    doc_id: str,
    text_field: str,
) -> list[dict[str, Any]]:
    records = []
    for i, chunk in enumerate(chunks):
        cid = f"{doc_id}_{i:04d}"
        records.append(
            {
                "_id": cid,
                text_field: chunk,
                "source_pdf": source_pdf,
                "source_id": cid,
                "category": "denuncia",
            }
        )
    return records


def process_one_pdf(pdf_path: Path, cfg: BatchConfig) -> dict[str, Any]:
    from src.agents.ingestion_agent import IngestionAgent
    from src.orchestration.orchestrator import JudicialOrchestrator
    from src.rag.chunking import chunk_text

    doc_id = safe_doc_id(cfg.input_dir, pdf_path)
    out_base = cfg.out_dir / doc_id
    ocr_path = out_base.with_name(f"{doc_id}_ocr.txt")
    chunks_path = out_base.with_name(f"{doc_id}_chunks.jsonl")
    result_path = out_base.with_name(f"{doc_id}_result.json")

    ingestion = IngestionAgent()
    orchestrator = JudicialOrchestrator()

    ingest = ingestion.invoke_pdf(
        pdf_path=pdf_path,
        ocr_backend=cfg.ocr_backend,
        tesseract_lang=cfg.tesseract_lang,
        ollama_model=cfg.ollama_model,
    )
    text = ingest["anonymized_text"]
    chunks = chunk_text(text, chunk_size=cfg.chunk_size, overlap=cfg.overlap)
    result = orchestrator.run_from_ingest_payload(ingest)

    ocr_path.write_text(text, encoding="utf-8")
    with open(chunks_path, "w", encoding="utf-8") as file_handle:
        for i, chunk in enumerate(chunks):
            row = {"chunk_id": f"{doc_id}_{i:04d}", "source_pdf": str(pdf_path), "text": chunk}
            file_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    result_path.write_text(json.dumps(result.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")

    pinecone_upserted = 0
    pinecone_error = None
    if cfg.upsert_pinecone:
        from src.rag.pinecone_client import PineconeConfig, PineconeTextIndexClient

        p_cfg = PineconeConfig.from_env()
        if not p_cfg:
            pinecone_error = "Pinecone env vars not configured."
        else:
            client = PineconeTextIndexClient(p_cfg)
            records = to_pinecone_records(chunks, str(pdf_path), doc_id, p_cfg.text_field)
            try:
                for batch in batched(records, cfg.pinecone_batch_size):
                    client.upsert_records(batch)
                    pinecone_upserted += len(batch)
            except Exception as exc:
                pinecone_error = str(exc)

    return {
        "status": "ok",
        "doc_id": doc_id,
        "pdf_path": str(pdf_path),
        "ocr_path": str(ocr_path),
        "chunks_path": str(chunks_path),
        "result_path": str(result_path),
        "chunks": len(chunks),
        "alerts": result.alerts,
        "selected_template_id": result.draft.selected_template_id,
        "pinecone_upserted": pinecone_upserted,
        "pinecone_error": pinecone_error,
    }


def process_one_pdf_safe(pdf_path: Path, cfg: BatchConfig) -> dict[str, Any]:
    try:
        return process_one_pdf(pdf_path, cfg)
    except Exception as exc:
        return {
            "status": "error",
            "pdf_path": str(pdf_path),
            "error": str(exc),
        }


def write_manifest(out_dir: Path, rows: list[dict[str, Any]]) -> tuple[Path, dict[str, int]]:
    manifest_path = out_dir / "manifest.jsonl"
    summary = {"processed": 0, "failed": 0}

    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        for row in rows:
            if row.get("status") == "ok":
                summary["processed"] += 1
            else:
                summary["failed"] += 1
            manifest_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    return manifest_path, summary
