from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.llm.safety import UnsupportedDocumentError

logger = logging.getLogger(__name__)

BatchProgressCallback = Callable[[str, str, dict[str, Any]], None]
BatchResultCallback = Callable[[dict[str, Any]], None]


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
    retrieval_provider: str | None = None
    retrieval_model: str | None = None
    include_draft: bool = True
    progress_callback: BatchProgressCallback | None = None
    result_callback: BatchResultCallback | None = None


def list_pdfs(input_dir: Path, glob_pattern: str) -> list[Path]:
    files = sorted(input_dir.rglob(glob_pattern))
    return [p for p in files if p.is_file() and p.suffix.lower() == ".pdf"]


def safe_doc_id(root: Path, file_path: Path) -> str:
    rel = file_path.relative_to(root)
    stem = rel.with_suffix("").as_posix().replace("/", "__")
    return stem.replace(" ", "_")


def batched(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    if size <= 0:
        raise ValueError("pinecone_batch_size debe ser mayor a 0.")
    return [items[i : i + size] for i in range(0, len(items), size)]


def to_pinecone_records(
    chunks: list[str],
    source_pdf: str,
    doc_id: str,
    text_field: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

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
    from src.llm.measure_classifier import MeasureClassifierService
    from src.orchestration.orchestrator import JudicialOrchestrator
    from src.rag.chunking import chunk_text

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    doc_id = safe_doc_id(cfg.input_dir, pdf_path)
    filename = pdf_path.name

    out_base = cfg.out_dir / doc_id
    ocr_path = out_base.with_name(f"{doc_id}_ocr.txt")
    chunks_path = out_base.with_name(f"{doc_id}_chunks.jsonl")
    result_path = out_base.with_name(f"{doc_id}_result.json")

    def notify(stage: str, **payload: Any) -> None:
        if cfg.progress_callback:
            cfg.progress_callback(
                filename,
                stage,
                {
                    "doc_id": doc_id,
                    "filename": filename,
                    **payload,
                },
            )

    total_started_at = time.perf_counter()

    notify(
        "processing_started",
        progress=0,
        detail="Iniciando procesamiento.",
    )

    logger.info("Iniciando procesamiento batch para PDF: %s", pdf_path)

    ingestion = IngestionAgent()
    orchestrator = JudicialOrchestrator()
    classifier = MeasureClassifierService()

    notify(
        "ocr_started",
        progress=5,
        detail="Ejecutando OCR.",
    )

    ingest = ingestion.invoke_pdf(
        pdf_path=pdf_path,
        ocr_backend=cfg.ocr_backend,
        tesseract_lang=cfg.tesseract_lang,
        ollama_model=cfg.ollama_model,
    )

    notify(
        "anonymization_done",
        progress=20,
        detail="OCR y anonimización completados.",
        case_id=ingest["case_id"],
        anonymized_text=ingest["anonymized_text"],
        ocr_quality=ingest.get("ocr_quality"),
        ocr_warning=ingest.get("ocr_warning"),
        ocr_duration_ms=ingest.get("ocr_duration_ms"),
        anonymization_audit=ingest.get("anonymization_audit"),
        anonymization_warning=ingest.get("anonymization_warning"),
    )

    text = ingest["anonymized_text"]

    notify(
        "chunking_started",
        progress=25,
        detail="Segmentando texto anonimizado.",
    )

    chunks = chunk_text(
        text,
        chunk_size=cfg.chunk_size,
        overlap=cfg.overlap,
        embedding_model=os.getenv("PINECONE_EMBEDDING_MODEL"),
        tokenizer_name=os.getenv("CHUNK_TOKENIZER_NAME"),
    )

    notify(
        "chunking_done",
        progress=35,
        detail=f"Segmentación completada: {len(chunks)} chunks.",
        chunks=len(chunks),
    )

    notify(
        "orchestration_started",
        progress=40,
        detail="Ejecutando extracción, recuperación y generación preliminar.",
    )

    result = orchestrator.run_from_ingest_payload(
        ingest,
        retrieval_provider=cfg.retrieval_provider,
        retrieval_model=cfg.retrieval_model,
        progress_callback=lambda stage, payload: notify(stage, **payload),
    )

    notify(
        "orchestration_done",
        progress=75,
        detail="Extracción, recuperación y borrador preliminar completados.",
        case_id=result.case_id,
    )

    notify(
        "classification_started",
        progress=85,
        detail="Clasificando plantilla y nivel de riesgo.",
    )

    classification_started_at = time.perf_counter()

    classification = classifier.classify(
        extracted=result.extracted_case,
        retrieval=result.retrieval,
        provider=cfg.retrieval_provider or "openai",
        model=cfg.retrieval_model,
        include_draft=cfg.include_draft,
    )

    classification_duration_ms = int(
        (time.perf_counter() - classification_started_at) * 1000
    )

    notify(
        "classification_done",
        progress=92,
        detail="Clasificación completada.",
        selected_template_name=classification.selected_template_name,
        risk_level=classification.risk_level,
    )

    notify(
        "persist_started",
        progress=94,
        detail="Persistiendo OCR, chunks y resultado.",
    )

    ocr_path.write_text(text, encoding="utf-8")

    with open(chunks_path, "w", encoding="utf-8") as file_handle:
        for i, chunk in enumerate(chunks):
            row = {
                "chunk_id": f"{doc_id}_{i:04d}",
                "source_pdf": str(pdf_path),
                "text": chunk,
            }
            file_handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    result_path.write_text(
        json.dumps(
            {
                "pipeline_result": result.model_dump(),
                "classification": classification.model_dump(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    notify(
        "persist_done",
        progress=97,
        detail="Archivos de salida persistidos.",
        ocr_path=str(ocr_path),
        chunks_path=str(chunks_path),
        result_path=str(result_path),
    )

    pinecone_upserted = 0
    pinecone_error = None

    if cfg.upsert_pinecone:
        notify(
            "pinecone_started",
            progress=98,
            detail="Indexando chunks en Pinecone.",
        )

        from src.rag.pinecone_client import PineconeConfig, PineconeTextIndexClient

        p_cfg = PineconeConfig.from_env()

        if not p_cfg:
            pinecone_error = "Pinecone env vars not configured."
            logger.warning(
                "No se indexó en Pinecone porque faltan variables de entorno."
            )
        else:
            client = PineconeTextIndexClient(p_cfg)
            records = to_pinecone_records(
                chunks=chunks,
                source_pdf=str(pdf_path),
                doc_id=doc_id,
                text_field=p_cfg.text_field,
            )

            try:
                for batch in batched(records, cfg.pinecone_batch_size):
                    client.upsert_records(batch)
                    pinecone_upserted += len(batch)
            except Exception as exc:
                pinecone_error = str(exc)
                logger.exception(
                    "Error al indexar chunks en Pinecone para PDF: %s",
                    pdf_path,
                )

        notify(
            "pinecone_done",
            progress=99,
            detail=(
                f"Indexación en Pinecone completada: {pinecone_upserted} registros."
                if not pinecone_error
                else f"Indexación en Pinecone omitida o fallida: {pinecone_error}"
            ),
            pinecone_upserted=pinecone_upserted,
            pinecone_error=pinecone_error,
        )

    total_duration_ms = int((time.perf_counter() - total_started_at) * 1000)

    row = {
        "status": "ok",
        "doc_id": doc_id,
        "filename": filename,
        "case_id": result.case_id,
        "pdf_path": str(pdf_path),
        "ocr_path": str(ocr_path),
        "chunks_path": str(chunks_path),
        "result_path": str(result_path),
        "chunks": len(chunks),
        "alerts": result.alerts,
        "selected_template_id": result.draft.selected_template_id,
        "selected_template_name": classification.selected_template_name,
        "risk_level": classification.risk_level,
        "anonymized_text": result.extracted_case.anonymized_text,
        "anonymization_audit": ingest.get("anonymization_audit"),
        "anonymization_warning": ingest.get("anonymization_warning"),
        "ocr_duration_ms": ingest.get("ocr_duration_ms"),
        "classification_duration_ms": classification_duration_ms,
        "total_duration_ms": total_duration_ms,
        "classification": classification.model_dump(),
        "pipeline_result": result.model_dump(),
        "pinecone_upserted": pinecone_upserted,
        "pinecone_error": pinecone_error,
    }

    notify(
        "completed",
        progress=100,
        detail="Procesamiento completado.",
        case_id=result.case_id,
        selected_template_id=result.draft.selected_template_id,
        selected_template_name=classification.selected_template_name,
        risk_level=classification.risk_level,
        chunks=len(chunks),
        total_duration_ms=total_duration_ms,
        pinecone_upserted=pinecone_upserted,
        pinecone_error=pinecone_error,
    )

    if cfg.result_callback:
        cfg.result_callback(row)

    logger.info(
        "Procesamiento batch completado para PDF: %s en %sms",
        pdf_path,
        total_duration_ms,
    )

    return row


def process_one_pdf_safe(pdf_path: Path, cfg: BatchConfig) -> dict[str, Any]:
    try:
        return process_one_pdf(pdf_path, cfg)

    except Exception as exc:
        logger.exception("Error procesando PDF en batch: %s", pdf_path)

        retryable = not isinstance(exc, UnsupportedDocumentError)

        row = {
            "status": "error",
            "filename": pdf_path.name,
            "pdf_path": str(pdf_path),
            "error": str(exc),
            "error_type": (
                "unsupported_document"
                if isinstance(exc, UnsupportedDocumentError)
                else "processing_error"
            ),
            "retryable": retryable,
        }

        if cfg.progress_callback:
            cfg.progress_callback(
                pdf_path.name,
                "failed",
                {
                    "filename": pdf_path.name,
                    "pdf_path": str(pdf_path),
                    "progress": 100,
                    "detail": str(exc) or "Error procesando el documento.",
                    "error": str(exc),
                    "error_type": row["error_type"],
                    "retryable": retryable,
                },
            )

        if cfg.result_callback:
            cfg.result_callback(row)

        return row


def write_manifest(out_dir: Path, rows: list[dict[str, Any]]) -> tuple[Path, dict[str, int]]:
    out_dir.mkdir(parents=True, exist_ok=True)

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