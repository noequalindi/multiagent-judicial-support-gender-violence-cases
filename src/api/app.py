from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.agents.ingestion_agent import IngestionAgent
from src.api.auth import AuthManager, AuthSession, auth_manager, get_client_key, require_auth
from src.llm.measure_classifier import MeasureClassifierService
from src.llm.registry import ocr_options, provider_options
from src.llm.safety import UnsupportedDocumentError
from src.api.job_store import job_store
from src.logging import get_logger
from src.models.contracts import MeasureClassification, PipelineOutput
from src.orchestration.batch_runner import BatchConfig, process_one_pdf_safe
from src.orchestration.langgraph_batch import (
    batch_graph_available,
    batch_graph_unavailable_reason,
    run_batch_graph,
)
from src.orchestration.orchestrator import JudicialOrchestrator
from src.reporting.preliminary_reports import (
    build_pdf_bytes,
    render_general_cases_csv,
    render_general_cases_report_text,
    render_individual_case_csv,
    render_individual_case_report_text,
)
from src.scripts.generate_measure_document import (
    extract_case_fields,
    fill_template_text,
    load_template_text,
    text_to_simple_pdf,
)
from src.storage.mongo_store import (
    MongoRunStore,
    build_anonymization_issue_payload,
    build_run_payload,
)
from src.storage.local_json_store import LocalClassificationStore, build_classified_case_payload


app = FastAPI(title="Violence Judicial Assistant API", version="0.1.0")
mongo_store = MongoRunStore()
local_classification_store = LocalClassificationStore()
api_logger = get_logger("api")

frontend_origins = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_checks() -> None:
    if not mongo_store.enabled():
        api_logger.info("Mongo store disabled: missing URI or pymongo not installed.")
        return
    try:
        result = mongo_store.ensure_indexes()
        api_logger.info(
            "Mongo store ready db={} collection={} indexes={}",
            mongo_store.db_name,
            mongo_store.collection_name,
            len(result.get("created", [])),
        )
        admin_hash = auth_manager.password_hash
        if not admin_hash and auth_manager.password:
            admin_hash = auth_manager.hash_password(auth_manager.password)
        if admin_hash:
            mongo_store.upsert_auth_user(
                username=auth_manager.username,
                password_hash=admin_hash,
                role=auth_manager.role,
                active=True,
            )
            api_logger.info(
                "Auth bootstrap ready auth_collection={} admin={}",
                mongo_store.auth_collection_name,
                auth_manager.username,
            )
        else:
            api_logger.warning(
                "Auth bootstrap skipped: APP_AUTH_PASSWORD o APP_AUTH_PASSWORD_HASH no configurados."
            )
    except Exception as exc:
        api_logger.exception("Mongo bootstrap failed: {}", exc)


def _raise_api_error(exc: Exception, status_code: int = 500) -> None:
    raise HTTPException(status_code=status_code, detail=str(exc))


def _should_persist_input_rejection(exc: Exception) -> bool:
    return not isinstance(exc, UnsupportedDocumentError)


def _persist_anonymization_issue(
    *,
    route: str,
    job_id: str | None,
    case_id: str | None,
    filename: str | None,
    ocr_backend: str | None,
    tesseract_lang: str | None,
    ollama_model: str | None,
    provider: str | None,
    model: str | None,
    anonymization_audit: dict | None,
    anonymization_warning: str | None,
    error: str | None = None,
) -> None:
    if not anonymization_audit or not anonymization_audit.get("has_residuals"):
        return
    mongo_store.save_anonymization_issue(
        build_anonymization_issue_payload(
            route=route,
            job_id=job_id,
            case_id=case_id,
            filename=filename,
            ocr_backend=ocr_backend,
            tesseract_lang=tesseract_lang,
            ollama_model=ollama_model,
            provider=provider,
            model=model,
            anonymization_audit=anonymization_audit,
            anonymization_warning=anonymization_warning,
            error=error,
        )
    )


class ProcessTextRequest(BaseModel):
    text: str = Field(min_length=1)


class ProcessTextResponse(BaseModel):
    result: dict


class GenerateDraftResponse(BaseModel):
    template_id: str
    txt_content: str
    fields: dict[str, str]


class AnalyzeCaseResponse(BaseModel):
    pipeline_result: PipelineOutput
    classification: MeasureClassification


class StartJobResponse(BaseModel):
    job_id: str
    status: str
    kind: str


class ClassifyMeasureRequest(BaseModel):
    text: str = Field(min_length=1)
    provider: str = Field(default="openai")
    model: str | None = None
    include_draft: bool = False


class ValidateRecentRunRequest(BaseModel):
    created_at: str = Field(min_length=1)
    case_id: str | None = None
    filename: str | None = None
    route: str | None = None
    valid: bool = True


class DeleteRecentRunRequest(BaseModel):
    created_at: str = Field(min_length=1)
    case_id: str | None = None
    filename: str | None = None
    route: str | None = None


class LoginRequest(BaseModel):
    username: str = Field(min_length=1)
    password: str = Field(min_length=1)


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=8, max_length=256)


class ExportCaseItemRequest(BaseModel):
    source_label: str | None = None
    result: AnalyzeCaseResponse


class ExportCasesRequest(BaseModel):
    items: list[ExportCaseItemRequest] = Field(min_length=1)


class RetryBatchItemRequest(BaseModel):
    job_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)


def _map_stage_message(stage: str) -> tuple[str, int]:
    mapping = {
        "upload_received": ("Archivo recibido.", 5),
        "ocr_started": ("Extrayendo texto del documento.", 12),
        "anonymization_done": ("Anonimización completada.", 26),
        "ingestion_done": ("Ingesta finalizada.", 34),
        "extraction_done": ("Extracción estructurada completada.", 52),
        "retrieval_done": ("Recuperación de evidencia completada.", 68),
        "drafting_done": ("Borrador preliminar generado.", 82),
        "citation_verifier_done": ("Verificación de soporte completada.", 90),
        "alerting_done": ("Alertas calculadas.", 96),
        "pipeline_done": ("Pipeline finalizado.", 100),
    }
    return mapping.get(stage, (stage, 0))


def _build_progress_callback(job_id: str):
    def callback(stage: str, payload: dict) -> None:
        message, progress = _map_stage_message(stage)
        preview: dict[str, object] = {}
        if stage == "extraction_done":
            preview = {
                "facts": payload.get("facts"),
                "risk_factors": payload.get("risk_factors"),
            }
        if stage == "anonymization_done":
            preview = {
                "anonymization_audit": payload.get("anonymization_audit"),
                "anonymization_warning": payload.get("anonymization_warning"),
            }
        if stage == "drafting_done":
            preview = {
                "selected_template_id": payload.get("selected_template_id"),
            }
        job_store.append_stage(
            job_id,
            stage=stage,
            message=message,
            progress=progress,
            meta=payload,
        )
        if preview:
            job_store.update(job_id, preview={**(job_store.get(job_id) or {}).get("preview", {}), **preview})

    return callback


def _batch_retry_root(job_id: str) -> Path:
    return Path("data/runtime/batch_retry_sources") / job_id


def _batch_runtime_config(
    *,
    input_dir: Path,
    out_dir: Path,
    total: int,
    provider: str,
    model: str | None,
    include_draft: bool,
    ocr_backend: str,
    tesseract_lang: str,
    ollama_model: str,
    progress_callback,
    result_callback,
) -> BatchConfig:
    return BatchConfig(
        input_dir=input_dir,
        out_dir=out_dir,
        glob_pattern="*.pdf",
        workers=min(4, max(1, total)),
        ocr_backend=ocr_backend,
        tesseract_lang=tesseract_lang,
        ollama_model=ollama_model,
        chunk_size=1200,
        overlap=200,
        upsert_pinecone=False,
        pinecone_batch_size=96,
        retrieval_provider=provider,
        retrieval_model=model,
        include_draft=include_draft,
        progress_callback=progress_callback,
        result_callback=result_callback,
    )


def _replace_batch_result_row(
    job_id: str,
    filename: str,
    replacement: dict[str, Any],
) -> dict[str, Any] | None:
    current = job_store.get(job_id)
    if not current:
        return None
    results = [item for item in current.get("results", []) if item.get("filename") != filename]
    results.append(replacement)
    terminal_count = len(
        [
            item
            for item in results
            if item.get("status") in {"ok", "completed", "error", "failed"}
        ]
    )
    total = max(int(current.get("total_files", 0) or 0), 1)
    return job_store.update(
        job_id,
        results=results,
        completed_files=terminal_count,
        progress=min(99, int((terminal_count / total) * 100)),
    )


def _build_batch_callbacks(
    job_id: str,
    *,
    total: int,
    provider: str,
    model: str | None,
    ocr_backend: str,
    tesseract_lang: str,
    ollama_model: str,
):
    def on_progress(filename: str, stage: str, payload: dict[str, Any]) -> None:
        message, progress = _map_stage_message(stage)
        if stage == "anonymization_done":
            message = f"Anonimización lista para {filename}."
        file_progress_weights = {
            "ocr_started": 18,
            "anonymization_done": 34,
            "ingestion_done": 42,
            "extraction_done": 58,
            "retrieval_done": 74,
            "drafting_done": 86,
            "citation_verifier_done": 92,
            "alerting_done": 97,
            "pipeline_done": 100,
        }
        weighted = file_progress_weights.get(stage, progress)
        current_job = job_store.get(job_id) or {}
        completed_files = current_job.get("completed_files", 0)
        overall = int(((completed_files + (weighted / 100)) / max(total, 1)) * 100)
        preview = dict(current_job.get("preview", {}))
        if "anonymized_text" in payload:
            preview["anonymized_text"] = payload["anonymized_text"]
        if "ocr_quality" in payload:
            preview["ocr_quality"] = payload["ocr_quality"]
        if "ocr_warning" in payload:
            preview["ocr_warning"] = payload["ocr_warning"]
        if "ocr_duration_ms" in payload:
            preview["ocr_duration_ms"] = payload["ocr_duration_ms"]
        if "anonymization_audit" in payload:
            preview["anonymization_audit"] = payload["anonymization_audit"]
        if "anonymization_warning" in payload:
            preview["anonymization_warning"] = payload["anonymization_warning"]
        if "case_id" in payload:
            preview["case_id"] = payload["case_id"]
        if "selected_template_id" in payload:
            preview["selected_template_id"] = payload["selected_template_id"]
        if "facts" in payload:
            preview["facts"] = payload["facts"]
        if "risk_factors" in payload:
            preview["risk_factors"] = payload["risk_factors"]
        job_store.update(job_id, current_file=filename, preview=preview)
        job_store.append_stage(
            job_id,
            stage=stage,
            message=message,
            progress=min(99, overall),
            meta={"filename": filename, **payload},
        )

    def on_result(row: dict[str, Any]) -> None:
        current = job_store.get(job_id) or {}
        existing = next(
            (item for item in current.get("results", []) if item.get("filename") == row.get("filename")),
            None,
        )
        retry_count = int((existing or {}).get("retry_count", 0))
        if existing and existing.get("status") == "running":
            retry_count = max(retry_count, 1)
        if retry_count:
            row["retry_count"] = retry_count
            row["retried_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        current = _replace_batch_result_row(job_id, str(row.get("filename", "")), row) or {}
        completed_files = len(
            [
                item
                for item in current.get("results", [])
                if item.get("status") in {"ok", "completed"}
            ]
        )
        failed_files = len(
            [
                item
                for item in current.get("results", [])
                if item.get("status") in {"error", "failed"}
            ]
        )
        terminal_count = completed_files + failed_files
        next_progress = int((terminal_count / max(total, 1)) * 100)
        job_store.update(
            job_id,
            completed_files=terminal_count,
            progress=min(99, next_progress),
            preview={
                **(current.get("preview", {})),
                "ocr_duration_ms": row.get("ocr_duration_ms"),
                "classification_duration_ms": row.get("classification_duration_ms"),
                "total_duration_ms": row.get("total_duration_ms"),
            },
        )
        if row.get("status") == "ok":
            _persist_anonymization_issue(
                route="/jobs/analyze-case-batch/item",
                job_id=job_id,
                case_id=row.get("case_id"),
                filename=row.get("filename"),
                ocr_backend=ocr_backend,
                tesseract_lang=tesseract_lang,
                ollama_model=ollama_model,
                provider=provider,
                model=(row.get("classification") or {}).get("model") or model,
                anonymization_audit=row.get("anonymization_audit"),
                anonymization_warning=row.get("anonymization_warning"),
            )
            local_classification_store.save_case(
                build_classified_case_payload(
                    route="/jobs/analyze-case-batch/item",
                    filename=row.get("filename"),
                    ocr_backend=ocr_backend,
                    tesseract_lang=tesseract_lang,
                    ollama_model=ollama_model,
                    provider=provider,
                    model=(row.get("classification") or {}).get("model"),
                    pipeline_result=row.get("pipeline_result") or {},
                    classification=row.get("classification") or {},
                ),
                case_id=str(row.get("case_id", "unknown")),
            )
            mongo_store.save_run(
                build_run_payload(
                    route="/jobs/analyze-case-batch/item",
                    job_id=job_id,
                    case_id=row.get("case_id"),
                    filename=row.get("filename"),
                    ocr_backend=ocr_backend,
                    tesseract_lang=tesseract_lang,
                    ollama_model=ollama_model,
                    provider=provider,
                    model=(row.get("classification") or {}).get("model"),
                    anonymization_audit=row.get("anonymization_audit"),
                    timings={
                        "ocr_duration_ms": row.get("ocr_duration_ms"),
                        "classification_duration_ms": row.get("classification_duration_ms"),
                        "total_duration_ms": row.get("total_duration_ms"),
                    },
                    pipeline_result=row.get("pipeline_result"),
                    classification=row.get("classification"),
                )
            )
        elif row.get("anonymization_audit"):
            _persist_anonymization_issue(
                route="/jobs/analyze-case-batch/item",
                job_id=job_id,
                case_id=row.get("case_id"),
                filename=row.get("filename"),
                ocr_backend=ocr_backend,
                tesseract_lang=tesseract_lang,
                ollama_model=ollama_model,
                provider=provider,
                model=model,
                anonymization_audit=row.get("anonymization_audit"),
                anonymization_warning=row.get("anonymization_warning"),
                error=row.get("error"),
            )

    return on_progress, on_result


def _run_single_case_job(
    job_id: str,
    *,
    tmp_path: Path,
    filename: str,
    provider: str,
    model: str | None,
    include_draft: bool,
    ocr_backend: str,
    tesseract_lang: str,
    ollama_model: str,
) -> None:
    started_at = time.perf_counter()
    try:
        job_store.append_stage(
            job_id,
            stage="ocr_started",
            message="Extrayendo texto del documento.",
            progress=12,
            meta={"filename": filename},
        )
        ingestion = IngestionAgent()
        orchestrator = JudicialOrchestrator()
        ingest = ingestion.invoke_pdf(
            pdf_path=tmp_path,
            ocr_backend=ocr_backend,
            tesseract_lang=tesseract_lang,
            ollama_model=ollama_model,
        )
        job_store.update(
            job_id,
            preview={
                "anonymized_text": ingest["anonymized_text"],
                "case_id": ingest["case_id"],
                "ocr_quality": ingest.get("ocr_quality"),
                "ocr_warning": ingest.get("ocr_warning"),
                "ocr_duration_ms": ingest.get("ocr_duration_ms"),
                "anonymization_audit": ingest.get("anonymization_audit"),
                "anonymization_warning": ingest.get("anonymization_warning"),
            },
        )
        job_store.append_stage(
            job_id,
            stage="anonymization_done",
            message="Caso anonimizado disponible para revisión.",
            progress=30,
            meta={"case_id": ingest["case_id"]},
        )
        _persist_anonymization_issue(
            route="/jobs/analyze-case-pdf",
            job_id=job_id,
            case_id=ingest.get("case_id"),
            filename=filename,
            ocr_backend=ocr_backend,
            tesseract_lang=tesseract_lang,
            ollama_model=ollama_model,
            provider=provider,
            model=model,
            anonymization_audit=ingest.get("anonymization_audit"),
            anonymization_warning=ingest.get("anonymization_warning"),
        )
        result = orchestrator.run_from_ingest_payload(
            ingest=ingest,
            retrieval_provider=provider,
            retrieval_model=model,
            progress_callback=_build_progress_callback(job_id),
        )
        service = MeasureClassifierService()
        classification_started_at = time.perf_counter()
        classification = service.classify(
            extracted=result.extracted_case,
            retrieval=result.retrieval,
            provider=provider,
            model=model,
            include_draft=include_draft,
        )
        classification_duration_ms = int((time.perf_counter() - classification_started_at) * 1000)
        total_duration_ms = int((time.perf_counter() - started_at) * 1000)
        response = AnalyzeCaseResponse(
            pipeline_result=result,
            classification=classification,
        )
        local_classification_store.save_case(
            build_classified_case_payload(
                route="/jobs/analyze-case-pdf",
                filename=filename,
                ocr_backend=ocr_backend,
                tesseract_lang=tesseract_lang,
                ollama_model=ollama_model,
                provider=provider,
                model=classification.model,
                pipeline_result=result.model_dump(),
                classification=classification.model_dump(),
            ),
            case_id=result.case_id,
        )
        mongo_store.save_run(
            build_run_payload(
                route="/jobs/analyze-case-pdf",
                job_id=job_id,
                case_id=result.case_id,
                filename=filename,
                ocr_backend=ocr_backend,
                tesseract_lang=tesseract_lang,
                ollama_model=ollama_model,
                provider=provider,
                model=classification.model,
                anonymization_audit=ingest.get("anonymization_audit"),
                timings={
                    "ocr_duration_ms": ingest.get("ocr_duration_ms"),
                    "classification_duration_ms": classification_duration_ms,
                    "total_duration_ms": total_duration_ms,
                },
                pipeline_result=result.model_dump(),
                classification=classification.model_dump(),
            )
        )
        job_store.update(
            job_id,
            status="completed",
            stage="completed",
            message="Análisis terminado.",
            progress=100,
            result=response.model_dump(),
            completed_files=1,
            current_file=filename,
            preview={
                **((job_store.get(job_id) or {}).get("preview", {})),
                "ocr_duration_ms": ingest.get("ocr_duration_ms"),
                "classification_duration_ms": classification_duration_ms,
                "total_duration_ms": total_duration_ms,
            },
        )
    except Exception as exc:
        api_logger.exception("single_case_job_error job_id={} filename={} error={}", job_id, filename, exc)
        if "ingest" in locals():
            _persist_anonymization_issue(
                route="/jobs/analyze-case-pdf",
                job_id=job_id,
                case_id=ingest.get("case_id"),
                filename=filename,
                ocr_backend=ocr_backend,
                tesseract_lang=tesseract_lang,
                ollama_model=ollama_model,
                provider=provider,
                model=model,
                anonymization_audit=ingest.get("anonymization_audit"),
                anonymization_warning=ingest.get("anonymization_warning"),
                error=str(exc),
            )
        job_store.update(
            job_id,
            status="failed",
            stage="failed",
            message="La ejecución falló.",
            error=str(exc),
        )
    finally:
        tmp_path.unlink(missing_ok=True)


def _run_batch_case_job(
    job_id: str,
    *,
    files: list[tuple[str, Path]],
    provider: str,
    model: str | None,
    include_draft: bool,
    ocr_backend: str,
    tesseract_lang: str,
    ollama_model: str,
) -> None:
    tmp_root = Path(tempfile.mkdtemp(prefix=f"batch_job_{job_id}_"))
    input_dir = tmp_root / "input"
    out_dir = tmp_root / "output"
    retry_root = _batch_retry_root(job_id)
    input_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    retry_root.mkdir(parents=True, exist_ok=True)
    total = len(files)
    on_progress, on_result = _build_batch_callbacks(
        job_id,
        total=total,
        provider=provider,
        model=model,
        ocr_backend=ocr_backend,
        tesseract_lang=tesseract_lang,
        ollama_model=ollama_model,
    )

    try:
        for filename, tmp_path in files:
            target = input_dir / filename
            retry_target = retry_root / filename
            shutil.copy2(str(tmp_path), retry_target)
            shutil.move(str(tmp_path), target)

        retry_sources = {
            filename: str((retry_root / filename).resolve()) for filename, _ in files
        }
        job_store.update(
            job_id,
            retry_sources=retry_sources,
            retry_config={
                "provider": provider,
                "model": model,
                "include_draft": include_draft,
                "ocr_backend": ocr_backend,
                "tesseract_lang": tesseract_lang,
                "ollama_model": ollama_model,
                "workers": min(4, max(1, total)),
            },
        )
        cfg = _batch_runtime_config(
            input_dir=input_dir,
            out_dir=out_dir,
            total=total,
            provider=provider,
            model=model,
            include_draft=include_draft,
            ocr_backend=ocr_backend,
            tesseract_lang=tesseract_lang,
            ollama_model=ollama_model,
            progress_callback=on_progress,
            result_callback=on_result,
        )
        result = run_batch_graph(cfg)
        persisted_results = [
            item
            for item in result.get("results", [])
            if item.get("error_type") != "unsupported_document"
        ]
        job_store.update(
            job_id,
            status="completed",
            stage="completed",
            message="Batch finalizado.",
            progress=100,
            current_file=None,
            results=result.get("results", []),
            completed_files=result.get("processed", 0) + result.get("failed", 0),
        )
        mongo_store.save_run(
            {
                "route": "/jobs/analyze-case-batch",
                "job_id": job_id,
                "runtime": {
                    "ocr_backend": ocr_backend,
                    "tesseract_lang": tesseract_lang,
                    "ollama_model": ollama_model,
                    "provider": provider,
                    "model": model,
                },
                "job_id": job_id,
                "results": persisted_results,
                "summary": {
                    "total_files": total,
                    "completed_files": result.get("processed", 0),
                    "failed_files": len(
                        [item for item in persisted_results if item.get("status") != "ok"]
                    ),
                },
            }
        )
    except Exception as exc:
        api_logger.exception("batch_case_job_error job_id={} error={}", job_id, exc)
        job_store.update(
            job_id,
            status="failed",
            stage="failed",
            message="El batch falló.",
            error=str(exc),
            results=(job_store.get(job_id) or {}).get("results", []),
        )
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def _retry_batch_case_item(job_id: str, filename: str) -> None:
    current = job_store.get(job_id) or {}
    retry_sources = current.get("retry_sources") or {}
    retry_config = current.get("retry_config") or {}
    source_path = Path(str(retry_sources.get(filename, "")))
    if not source_path.is_file():
        job_store.update(
            job_id,
            status="failed",
            stage="failed",
            message=f"No se encontró el PDF fuente para reintentar {filename}.",
            error=f"No se encontró el PDF fuente para reintentar {filename}.",
            current_file=None,
        )
        return

    out_dir = Path(tempfile.mkdtemp(prefix=f"batch_retry_{job_id}_"))
    total = max(int(current.get("total_files", 0) or 0), 1)
    on_progress, on_result = _build_batch_callbacks(
        job_id,
        total=total,
        provider=str(retry_config.get("provider") or "openai"),
        model=retry_config.get("model"),
        ocr_backend=str(retry_config.get("ocr_backend") or "tesseract"),
        tesseract_lang=str(retry_config.get("tesseract_lang") or "spa"),
        ollama_model=str(retry_config.get("ollama_model") or "llama3.2-vision"),
    )

    try:
        cfg = _batch_runtime_config(
            input_dir=source_path.parent,
            out_dir=out_dir,
            total=total,
            provider=str(retry_config.get("provider") or "openai"),
            model=retry_config.get("model"),
            include_draft=bool(retry_config.get("include_draft", True)),
            ocr_backend=str(retry_config.get("ocr_backend") or "tesseract"),
            tesseract_lang=str(retry_config.get("tesseract_lang") or "spa"),
            ollama_model=str(retry_config.get("ollama_model") or "llama3.2-vision"),
            progress_callback=on_progress,
            result_callback=on_result,
        )
        process_one_pdf_safe(source_path, cfg)
        refreshed = job_store.get(job_id) or {}
        terminal_count = len(
            [
                item
                for item in refreshed.get("results", [])
                if item.get("status") in {"ok", "completed", "error", "failed"}
            ]
        )
        if terminal_count >= total:
            has_failed = any(
                item.get("status") in {"error", "failed"}
                for item in refreshed.get("results", [])
            )
            job_store.update(
                job_id,
                status="completed" if not has_failed else "failed",
                stage="completed" if not has_failed else "failed",
                message="Batch actualizado tras reintento." if not has_failed else "Batch finalizado con errores pendientes.",
                progress=100,
                current_file=None,
                error=None if not has_failed else refreshed.get("error"),
            )
    except Exception as exc:
        api_logger.exception(
            "batch_case_retry_error job_id={} filename={} error={}",
            job_id,
            filename,
            exc,
        )
        _replace_batch_result_row(
            job_id,
            filename,
            {
                "filename": filename,
                "status": "error",
                "error": str(exc),
                "error_type": "processing_error",
                "retry_count": max(
                    1,
                    int(
                        next(
                            (
                                item.get("retry_count", 0)
                                for item in (job_store.get(job_id) or {}).get("results", [])
                                if item.get("filename") == filename
                            ),
                            0,
                        )
                    ),
                ),
            },
        )
        job_store.update(
            job_id,
            status="failed",
            stage="failed",
            message=f"El reintento de {filename} falló.",
            error=str(exc),
            current_file=None,
        )
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
        _start_next_queued_batch_retry(job_id)


def _start_batch_retry(job_id: str, filename: str, retry_count: int) -> None:
    job = job_store.get(job_id) or {}
    _replace_batch_result_row(
        job_id,
        filename,
        {
            **next(
                (
                    item
                    for item in job.get("results", [])
                    if item.get("filename") == filename
                ),
                {"filename": filename},
            ),
            "status": "running",
            "error": None,
            "retry_count": retry_count,
            "retryable": False,
        },
    )
    total_files = max(int(job.get("total_files", 1) or 1), 1)
    progress = max(
        1,
        int((int(job.get("completed_files", 0) or 0) / total_files) * 100),
    )
    job_store.update(
        job_id,
        status="running",
        stage="retrying_item",
        message=f"Reanudando {filename}.",
        current_file=filename,
        error=None,
    )
    job_store.append_stage(
        job_id,
        stage="retrying_item",
        message=f"Reanudando {filename}.",
        progress=progress,
        meta={"filename": filename, "retry_count": retry_count},
    )
    thread = threading.Thread(
        target=_retry_batch_case_item,
        kwargs={"job_id": job_id, "filename": filename},
        daemon=True,
    )
    thread.start()


def _start_next_queued_batch_retry(job_id: str) -> None:
    job = job_store.get(job_id) or {}
    queue = list(job.get("queued_retry_filenames") or [])
    if not queue:
        return
    next_filename = queue[0]
    remaining = queue[1:]
    row = next(
        (item for item in job.get("results", []) if item.get("filename") == next_filename),
        None,
    )
    if not row or row.get("status") not in {"error", "failed"}:
        job_store.update(job_id, queued_retry_filenames=remaining)
        if remaining:
            _start_next_queued_batch_retry(job_id)
        return
    retry_count = int(row.get("retry_count", 0)) + 1
    job_store.update(job_id, queued_retry_filenames=remaining)
    _start_batch_retry(job_id, next_filename, retry_count)


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "mongo": mongo_store.status_snapshot()}


@app.post("/auth/login")
def auth_login(payload: LoginRequest, request: Request, response: Response) -> dict[str, object]:
    if not mongo_store.enabled():
        raise HTTPException(
            status_code=500,
            detail="La autenticación requiere MongoDB habilitado.",
        )
    client_key = get_client_key(request)
    auth_manager.assert_login_allowed(client_key)
    user = mongo_store.fetch_auth_user(payload.username.strip())
    if user is None and mongo_store.last_error:
        raise HTTPException(
            status_code=503,
            detail="No se pudo conectar a MongoDB para autenticar. Revisá DNS y conectividad.",
        )
    if not user or not user.get("active", True):
        auth_manager.record_failed_attempt(client_key)
        raise HTTPException(status_code=401, detail="Credenciales inválidas.")
    if not AuthManager.verify_password_hash(payload.password, str(user.get("password_hash", ""))):
        auth_manager.record_failed_attempt(client_key)
        raise HTTPException(status_code=401, detail="Credenciales inválidas.")
    auth_manager.clear_attempts(client_key)
    token = auth_manager.create_session_token(
        payload.username.strip(),
        str(user.get("role", "user")),
    )
    response.set_cookie(value=token, **auth_manager.cookie_settings())
    return {"ok": True, "username": payload.username.strip(), "role": user.get("role", "user")}


@app.post("/auth/register")
def auth_register(payload: RegisterRequest, request: Request, response: Response) -> dict[str, object]:
    if not mongo_store.enabled():
        raise HTTPException(
            status_code=500,
            detail="La autenticación requiere MongoDB habilitado.",
        )
    if not auth_manager.registration_enabled:
        raise HTTPException(status_code=403, detail="El registro de usuarios está deshabilitado.")
    client_key = get_client_key(request)
    auth_manager.assert_login_allowed(client_key)
    username = payload.username.strip().lower()
    if not username or len(username) < 3:
        raise HTTPException(status_code=422, detail="El usuario debe tener al menos 3 caracteres.")
    if not username.replace(".", "").replace("_", "").replace("-", "").isalnum():
        raise HTTPException(
            status_code=422,
            detail="El usuario solo puede contener letras, números, guiones, puntos o guiones bajos.",
        )
    password_hash = auth_manager.hash_password(payload.password)
    created, reason = mongo_store.create_auth_user(
        username=username,
        password_hash=password_hash,
        role=auth_manager.registration_default_role,
        active=True,
    )
    if not created:
        if reason == "unavailable" or mongo_store.last_error:
            raise HTTPException(
                status_code=503,
                detail="No se pudo conectar a MongoDB para registrar el usuario. Revisá DNS y conectividad.",
            )
        if reason == "duplicate":
            raise HTTPException(status_code=409, detail="Ese usuario ya existe.")
        raise HTTPException(status_code=500, detail="No se pudo registrar el usuario.")
    auth_manager.clear_attempts(client_key)
    token = auth_manager.create_session_token(username, auth_manager.registration_default_role)
    response.set_cookie(value=token, **auth_manager.cookie_settings())
    return {"ok": True, "username": username, "role": auth_manager.registration_default_role}


@app.post("/auth/logout")
def auth_logout(response: Response) -> dict[str, object]:
    response.delete_cookie(
        key=auth_manager.cookie_settings()["key"],
        path=auth_manager.cookie_settings()["path"],
    )
    return {"ok": True}


@app.get("/auth/me")
def auth_me(session: AuthSession = Depends(require_auth)) -> dict[str, object]:
    return {
        "authenticated": True,
        "username": session.username,
        "role": session.role,
        "expires_at": session.expires_at,
    }


@app.post("/exports/preliminary-report.csv")
def export_preliminary_report_csv(
    payload: ExportCasesRequest,
    _: AuthSession = Depends(require_auth),
) -> Response:
    csv_text = render_general_cases_csv(
        [
            (
                item.result.pipeline_result,
                item.result.classification,
                item.source_label,
            )
            for item in payload.items
        ]
    )
    return Response(
        content=csv_text.encode("utf-8"),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="iustiva_informe_preliminar.csv"'},
    )


@app.post("/exports/preliminary-report.pdf")
def export_preliminary_report_pdf(
    payload: ExportCasesRequest,
    _: AuthSession = Depends(require_auth),
) -> Response:
    report_text = render_general_cases_report_text(
        [
            (
                item.result.pipeline_result,
                item.result.classification,
                item.source_label,
            )
            for item in payload.items
        ]
    )
    pdf_bytes = build_pdf_bytes(report_text)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="iustiva_informe_preliminar.pdf"'},
    )


@app.post("/exports/case-detail.csv")
def export_case_detail_csv(
    payload: ExportCaseItemRequest,
    _: AuthSession = Depends(require_auth),
) -> Response:
    csv_text = render_individual_case_csv(
        pipeline=payload.result.pipeline_result,
        classification=payload.result.classification,
        source_label=payload.source_label,
    )
    filename_stem = (payload.source_label or payload.result.pipeline_result.case_id or "caso").replace(" ", "_")
    return Response(
        content=csv_text.encode("utf-8"),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename_stem}_detalle.csv"'},
    )


@app.post("/exports/case-detail.pdf")
def export_case_detail_pdf(
    payload: ExportCaseItemRequest,
    _: AuthSession = Depends(require_auth),
) -> Response:
    report_text = render_individual_case_report_text(
        pipeline=payload.result.pipeline_result,
        classification=payload.result.classification,
        source_label=payload.source_label,
    )
    pdf_bytes = build_pdf_bytes(report_text)
    filename_stem = (payload.source_label or payload.result.pipeline_result.case_id or "caso").replace(" ", "_")
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename_stem}_detalle.pdf"'},
    )


@app.get("/jobs/{job_id}")
def get_job(job_id: str, _: AuthSession = Depends(require_auth)) -> dict:
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@app.post("/jobs/analyze-case-batch/retry-item", response_model=StartJobResponse)
def retry_batch_item(
    payload: RetryBatchItemRequest,
    _: AuthSession = Depends(require_auth),
) -> StartJobResponse:
    job = job_store.get(payload.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.get("kind") != "batch":
        raise HTTPException(status_code=422, detail="El job indicado no corresponde a un batch.")

    retry_sources = job.get("retry_sources") or {}
    source_path = retry_sources.get(payload.filename)
    if not source_path or not Path(str(source_path)).is_file():
        raise HTTPException(
            status_code=404,
            detail="No se encontró el archivo fuente para reanudar este expediente.",
        )

    result_row = next(
        (item for item in job.get("results", []) if item.get("filename") == payload.filename),
        None,
    )
    if not result_row:
        raise HTTPException(
            status_code=404,
            detail="No se encontró una corrida previa para ese expediente dentro del batch.",
        )
    if result_row.get("status") not in {"error", "failed"}:
        raise HTTPException(
            status_code=422,
            detail="Solo se pueden reanudar expedientes que hayan fallado.",
        )

    retry_count = int(result_row.get("retry_count", 0)) + 1
    if job.get("status") == "running":
        if job.get("stage") != "retrying_item":
            raise HTTPException(
                status_code=422,
                detail="Esperá a que termine la ejecución actual del batch antes de reanudar un expediente puntual.",
            )
        queued = list(job.get("queued_retry_filenames") or [])
        if payload.filename not in queued and job.get("current_file") != payload.filename:
            queued.append(payload.filename)
            job_store.update(payload.job_id, queued_retry_filenames=queued)
            job_store.append_stage(
                payload.job_id,
                stage="retry_queued",
                message=f"{payload.filename} quedó en cola para reintento.",
                progress=job.get("progress", 0),
                meta={"filename": payload.filename, "retry_count": retry_count},
            )
        return StartJobResponse(job_id=payload.job_id, status="queued", kind="batch")

    _start_batch_retry(payload.job_id, payload.filename, retry_count)
    return StartJobResponse(job_id=payload.job_id, status="running", kind="batch")


@app.get("/metrics/summary")
def metrics_summary(limit: int = 300, _: AuthSession = Depends(require_auth)) -> dict:
    return mongo_store.metrics_summary(limit=limit)


@app.get("/metrics/recent")
def metrics_recent(limit: int = 20, _: AuthSession = Depends(require_auth)) -> dict:
    rows = mongo_store.fetch_recent_runs(limit=limit)
    return {"items": rows}


@app.get("/cases/search")
def cases_search(q: str, limit: int = 20, _: AuthSession = Depends(require_auth)) -> dict[str, object]:
    query = q.strip()
    if len(query) < 2:
        raise HTTPException(status_code=422, detail="Ingresá al menos 2 caracteres para buscar.")
    rows = mongo_store.search_runs(query=query, limit=limit)
    return {"items": rows, "query": query}


@app.get("/benchmarks/recent")
def benchmarks_recent(limit: int = 10, _: AuthSession = Depends(require_auth)) -> dict[str, object]:
    rows = mongo_store.fetch_benchmark_runs(limit=limit)
    return {"items": rows}


@app.post("/metrics/recent/validate")
def validate_recent_run(
    payload: ValidateRecentRunRequest,
    _: AuthSession = Depends(require_auth),
) -> dict[str, object]:
    updated = mongo_store.mark_run_validated(
        created_at=payload.created_at,
        case_id=payload.case_id,
        filename=payload.filename,
        route=payload.route,
        valid=payload.valid,
    )
    if not updated:
        raise HTTPException(
            status_code=404,
            detail="No se encontró la corrida para registrar la validación.",
        )
    return {"ok": True, "validated": payload.valid}


@app.post("/metrics/recent/delete")
def delete_recent_run(
    payload: DeleteRecentRunRequest,
    _: AuthSession = Depends(require_auth),
) -> dict[str, object]:
    query = {
        "created_at": payload.created_at,
        "case_id": payload.case_id,
        "filename": payload.filename,
        "route": payload.route,
    }
    deleted = mongo_store.delete_one_run(query)
    mongo_store.delete_anonymization_issues(
        {
            "case_id": payload.case_id,
            "filename": payload.filename,
            "route": payload.route,
        }
    )
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="No se encontró la corrida para eliminar.",
        )
    return {"ok": True, "deleted": True}


@app.post("/jobs/analyze-case-pdf", response_model=StartJobResponse)
async def start_analyze_case_pdf_job(
    _: AuthSession = Depends(require_auth),
    file: UploadFile = File(...),
    provider: str = Form("openai"),
    model: str | None = Form(None),
    include_draft: bool = Form(True),
    ocr_backend: str = Form("tesseract"),
    tesseract_lang: str = Form("spa"),
    ollama_model: str = Form("llama3.2-vision"),
) -> StartJobResponse:
    suffix = Path(file.filename or "denuncia.pdf").suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    job = job_store.create(kind="single", filename=file.filename or tmp_path.name)
    job_store.append_stage(
        job["job_id"],
        stage="upload_received",
        message="Archivo recibido.",
        progress=5,
        meta={"filename": file.filename},
    )
    thread = threading.Thread(
        target=_run_single_case_job,
        kwargs={
            "job_id": job["job_id"],
            "tmp_path": tmp_path,
            "filename": file.filename or tmp_path.name,
            "provider": provider,
            "model": model,
            "include_draft": include_draft,
            "ocr_backend": ocr_backend,
            "tesseract_lang": tesseract_lang,
            "ollama_model": ollama_model,
        },
        daemon=True,
    )
    thread.start()
    return StartJobResponse(job_id=job["job_id"], status="queued", kind="single")


@app.post("/jobs/analyze-case-batch", response_model=StartJobResponse)
async def start_analyze_case_batch_job(
    _: AuthSession = Depends(require_auth),
    files: list[UploadFile] = File(...),
    provider: str = Form("openai"),
    model: str | None = Form(None),
    include_draft: bool = Form(True),
    ocr_backend: str = Form("tesseract"),
    tesseract_lang: str = Form("spa"),
    ollama_model: str = Form("llama3.2-vision"),
) -> StartJobResponse:
    if not batch_graph_available():
        raise HTTPException(
            status_code=503,
            detail=batch_graph_unavailable_reason(),
        )
    if not files:
        raise HTTPException(status_code=422, detail="No files uploaded.")
    if len(files) > 20:
        raise HTTPException(status_code=422, detail="Batch limitado a 20 archivos por procesamiento.")

    tmp_files: list[tuple[str, Path]] = []
    for upload in files:
        suffix = Path(upload.filename or "denuncia.pdf").suffix or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await upload.read())
            tmp_files.append((upload.filename or tmp.name, Path(tmp.name)))

    filenames = [item[0] for item in tmp_files]
    job = job_store.create(kind="batch", total_files=len(tmp_files), filenames=filenames)
    job_store.append_stage(
        job["job_id"],
        stage="upload_received",
        message=f"Lote recibido con {len(tmp_files)} archivos.",
        progress=3,
        meta={"total_files": len(tmp_files)},
    )
    thread = threading.Thread(
        target=_run_batch_case_job,
        kwargs={
            "job_id": job["job_id"],
            "files": tmp_files,
            "provider": provider,
            "model": model,
            "include_draft": include_draft,
            "ocr_backend": ocr_backend,
            "tesseract_lang": tesseract_lang,
            "ollama_model": ollama_model,
        },
        daemon=True,
    )
    thread.start()
    return StartJobResponse(job_id=job["job_id"], status="queued", kind="batch")


@app.get("/options/llm")
def llm_options(_: AuthSession = Depends(require_auth)) -> dict[str, object]:
    return {"providers": provider_options()}


@app.get("/options/ocr")
def ocr_runtime_options(_: AuthSession = Depends(require_auth)) -> dict[str, object]:
    return ocr_options()


@app.post("/process-text", response_model=ProcessTextResponse)
def process_text(
    payload: ProcessTextRequest,
    _: AuthSession = Depends(require_auth),
) -> ProcessTextResponse:
    try:
        api_logger.info("process_text_start")
        orchestrator = JudicialOrchestrator()
        result = orchestrator.run(payload.text)
        api_logger.info("process_text_done case_id={}", result.case_id)
        mongo_store.save_run(
            build_run_payload(route="/process-text", pipeline_result=result.model_dump())
        )
        return ProcessTextResponse(result=result.model_dump())
    except HTTPException:
        raise
    except Exception as exc:
        api_logger.exception("process_text_error error={}", exc)
        if _should_persist_input_rejection(exc):
            mongo_store.save_run(build_run_payload(route="/process-text", error=str(exc)))
        _raise_api_error(exc)


@app.post("/classify-measure", response_model=MeasureClassification)
def classify_measure(
    payload: ClassifyMeasureRequest,
    _: AuthSession = Depends(require_auth),
) -> MeasureClassification:
    try:
        api_logger.info(
            "classify_measure_start provider={} model={}",
            payload.provider,
            payload.model or "-",
        )
        orchestrator = JudicialOrchestrator()
        result = orchestrator.run_with_retrieval_provider(
            payload.text,
            retrieval_provider=payload.provider,
            retrieval_model=payload.model,
        )
        service = MeasureClassifierService()
        classification = service.classify(
            extracted=result.extracted_case,
            retrieval=result.retrieval,
            provider=payload.provider,
            model=payload.model,
            include_draft=payload.include_draft,
        )
        api_logger.info(
            "classify_measure_done case_id={} provider={} model={} template={}",
            result.case_id,
            payload.provider,
            classification.model,
            classification.selected_template_id,
        )
        local_classification_store.save_case(
            build_classified_case_payload(
                route="/classify-measure",
                filename=None,
                ocr_backend=None,
                tesseract_lang=None,
                ollama_model=None,
                provider=payload.provider,
                model=classification.model,
                pipeline_result=result.model_dump(),
                classification=classification.model_dump(),
            ),
            case_id=result.case_id,
        )
        mongo_store.save_run(
            build_run_payload(
                route="/classify-measure",
                case_id=result.case_id,
                provider=payload.provider,
                model=classification.model,
                pipeline_result=result.model_dump(),
                classification=classification.model_dump(),
            )
        )
        return classification
    except HTTPException:
        raise
    except Exception as exc:
        api_logger.exception(
            "classify_measure_error provider={} model={} error={}",
            payload.provider,
            payload.model or "-",
            exc,
        )
        if _should_persist_input_rejection(exc):
            mongo_store.save_run(
                build_run_payload(
                    route="/classify-measure",
                    provider=payload.provider,
                    model=payload.model,
                    error=str(exc),
                )
            )
        _raise_api_error(exc, status_code=422)


@app.post("/classify-measure-pdf", response_model=MeasureClassification)
async def classify_measure_pdf(
    _: AuthSession = Depends(require_auth),
    file: UploadFile = File(...),
    provider: str = Form("openai"),
    model: str | None = Form(None),
    include_draft: bool = Form(False),
    ocr_backend: str = Form("tesseract"),
    tesseract_lang: str = Form("spa"),
    ollama_model: str = Form("llama3.2-vision"),
) -> MeasureClassification:
    suffix = Path(file.filename or "denuncia.pdf").suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        api_logger.info(
            "classify_measure_pdf_start filename={} provider={} model={} ocr_backend={}",
            file.filename,
            provider,
            model or "-",
            ocr_backend,
        )
        orchestrator = JudicialOrchestrator()
        result = orchestrator.run_from_pdf_with_retrieval_provider(
            pdf_path=tmp_path,
            ocr_backend=ocr_backend,
            tesseract_lang=tesseract_lang,
            ollama_model=ollama_model,
            retrieval_provider=provider,
            retrieval_model=model,
        )
        service = MeasureClassifierService()
        classification = service.classify(
            extracted=result.extracted_case,
            retrieval=result.retrieval,
            provider=provider,
            model=model,
            include_draft=include_draft,
        )
        api_logger.info(
            "classify_measure_pdf_done case_id={} filename={} provider={} model={} template={}",
            result.case_id,
            file.filename,
            provider,
            classification.model,
            classification.selected_template_id,
        )
        local_classification_store.save_case(
            build_classified_case_payload(
                route="/classify-measure-pdf",
                filename=file.filename,
                ocr_backend=ocr_backend,
                tesseract_lang=tesseract_lang,
                ollama_model=ollama_model,
                provider=provider,
                model=classification.model,
                pipeline_result=result.model_dump(),
                classification=classification.model_dump(),
            ),
            case_id=result.case_id,
        )
        mongo_store.save_run(
            build_run_payload(
                route="/classify-measure-pdf",
                case_id=result.case_id,
                filename=file.filename,
                ocr_backend=ocr_backend,
                tesseract_lang=tesseract_lang,
                ollama_model=ollama_model,
                provider=provider,
                model=classification.model,
                pipeline_result=result.model_dump(),
                classification=classification.model_dump(),
            )
        )
        return classification
    except HTTPException:
        raise
    except Exception as exc:
        api_logger.exception(
            "classify_measure_pdf_error filename={} provider={} model={} error={}",
            file.filename,
            provider,
            model or "-",
            exc,
        )
        if _should_persist_input_rejection(exc):
            mongo_store.save_run(
                build_run_payload(
                    route="/classify-measure-pdf",
                    filename=file.filename,
                    ocr_backend=ocr_backend,
                    tesseract_lang=tesseract_lang,
                    ollama_model=ollama_model,
                    provider=provider,
                    model=model,
                    error=str(exc),
                )
            )
        _raise_api_error(exc, status_code=422)
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/process-pdf")
async def process_pdf(
    _: AuthSession = Depends(require_auth),
    file: UploadFile = File(...),
    ocr_backend: str = Form("tesseract"),
    tesseract_lang: str = Form("spa"),
    ollama_model: str = Form("llama3.2-vision"),
) -> dict:
    suffix = Path(file.filename or "denuncia.pdf").suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        orchestrator = JudicialOrchestrator()
        result = orchestrator.run_from_pdf(
            pdf_path=tmp_path,
            ocr_backend=ocr_backend,
            tesseract_lang=tesseract_lang,
            ollama_model=ollama_model,
        )
        mongo_store.save_run(
            build_run_payload(
                route="/process-pdf",
                case_id=result.case_id,
                filename=file.filename,
                ocr_backend=ocr_backend,
                tesseract_lang=tesseract_lang,
                ollama_model=ollama_model,
                pipeline_result=result.model_dump(),
            )
        )
        return {"result": result.model_dump()}
    except HTTPException:
        raise
    except Exception as exc:
        if _should_persist_input_rejection(exc):
            mongo_store.save_run(
                build_run_payload(
                    route="/process-pdf",
                    filename=file.filename,
                    ocr_backend=ocr_backend,
                    tesseract_lang=tesseract_lang,
                    ollama_model=ollama_model,
                    error=str(exc),
                )
            )
        _raise_api_error(exc)
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/analyze-case-pdf", response_model=AnalyzeCaseResponse)
async def analyze_case_pdf(
    _: AuthSession = Depends(require_auth),
    file: UploadFile = File(...),
    provider: str = Form("openai"),
    model: str | None = Form(None),
    include_draft: bool = Form(True),
    ocr_backend: str = Form("tesseract"),
    tesseract_lang: str = Form("spa"),
    ollama_model: str = Form("llama3.2-vision"),
) -> AnalyzeCaseResponse:
    suffix = Path(file.filename or "denuncia.pdf").suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        api_logger.info(
            "analyze_case_pdf_start filename={} provider={} model={} ocr_backend={}",
            file.filename,
            provider,
            model or "-",
            ocr_backend,
        )
        orchestrator = JudicialOrchestrator()
        result = orchestrator.run_from_pdf_with_retrieval_provider(
            pdf_path=tmp_path,
            ocr_backend=ocr_backend,
            tesseract_lang=tesseract_lang,
            ollama_model=ollama_model,
            retrieval_provider=provider,
            retrieval_model=model,
        )
        service = MeasureClassifierService()
        classification = service.classify(
            extracted=result.extracted_case,
            retrieval=result.retrieval,
            provider=provider,
            model=model,
            include_draft=include_draft,
        )
        response = AnalyzeCaseResponse(
            pipeline_result=result,
            classification=classification,
        )
        local_classification_store.save_case(
            build_classified_case_payload(
                route="/analyze-case-pdf",
                filename=file.filename,
                ocr_backend=ocr_backend,
                tesseract_lang=tesseract_lang,
                ollama_model=ollama_model,
                provider=provider,
                model=classification.model,
                pipeline_result=result.model_dump(),
                classification=classification.model_dump(),
            ),
            case_id=result.case_id,
        )
        mongo_store.save_run(
            build_run_payload(
                route="/analyze-case-pdf",
                case_id=result.case_id,
                filename=file.filename,
                ocr_backend=ocr_backend,
                tesseract_lang=tesseract_lang,
                ollama_model=ollama_model,
                provider=provider,
                model=classification.model,
                pipeline_result=result.model_dump(),
                classification=classification.model_dump(),
            )
        )
        api_logger.info(
            "analyze_case_pdf_done case_id={} filename={} provider={} model={} template={}",
            result.case_id,
            file.filename,
            provider,
            classification.model,
            classification.selected_template_id,
        )
        return response
    except HTTPException:
        raise
    except Exception as exc:
        api_logger.exception(
            "analyze_case_pdf_error filename={} provider={} model={} error={}",
            file.filename,
            provider,
            model or "-",
            exc,
        )
        if _should_persist_input_rejection(exc):
            mongo_store.save_run(
                build_run_payload(
                    route="/analyze-case-pdf",
                    filename=file.filename,
                    ocr_backend=ocr_backend,
                    tesseract_lang=tesseract_lang,
                    ollama_model=ollama_model,
                    provider=provider,
                    model=model,
                    error=str(exc),
                )
            )
        _raise_api_error(exc, status_code=422)
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/generate-draft", response_model=GenerateDraftResponse)
async def generate_draft(
    _: AuthSession = Depends(require_auth),
    file: UploadFile = File(...),
    measures_dir: str = Form("data/measures"),
    ocr_backend: str = Form("tesseract"),
    tesseract_lang: str = Form("spa"),
    ollama_model: str = Form("llama3.2-vision"),
) -> GenerateDraftResponse:
    suffix = Path(file.filename or "denuncia.pdf").suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        orchestrator = JudicialOrchestrator()
        result = orchestrator.run_from_pdf(
            pdf_path=tmp_path,
            ocr_backend=ocr_backend,
            tesseract_lang=tesseract_lang,
            ollama_model=ollama_model,
        )

        template_id = result.draft.selected_template_id
        if not template_id:
            raise HTTPException(status_code=422, detail="No template selected by the pipeline.")

        template_text = load_template_text(template_id, Path(measures_dir).resolve())
        fields = extract_case_fields(result.extracted_case.anonymized_text)
        txt_content = fill_template_text(template_text, fields)

        response = GenerateDraftResponse(
            template_id=template_id,
            txt_content=txt_content,
            fields=fields,
        )
        mongo_store.save_run(
            build_run_payload(
                route="/generate-draft",
                case_id=result.case_id,
                filename=file.filename,
                ocr_backend=ocr_backend,
                tesseract_lang=tesseract_lang,
                ollama_model=ollama_model,
                pipeline_result=result.model_dump(),
                generated_draft=response.model_dump(),
            )
        )
        return response
    except HTTPException:
        raise
    except Exception as exc:
        if _should_persist_input_rejection(exc):
            mongo_store.save_run(
                build_run_payload(
                    route="/generate-draft",
                    filename=file.filename,
                    ocr_backend=ocr_backend,
                    tesseract_lang=tesseract_lang,
                    ollama_model=ollama_model,
                    error=str(exc),
                )
            )
        _raise_api_error(exc, status_code=422)
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/generate-draft-pdf")
async def generate_draft_pdf(
    _: AuthSession = Depends(require_auth),
    file: UploadFile = File(...),
    measures_dir: str = Form("data/measures"),
    ocr_backend: str = Form("tesseract"),
    tesseract_lang: str = Form("spa"),
    ollama_model: str = Form("llama3.2-vision"),
) -> dict:
    suffix = Path(file.filename or "denuncia.pdf").suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        orchestrator = JudicialOrchestrator()
        result = orchestrator.run_from_pdf(
            pdf_path=tmp_path,
            ocr_backend=ocr_backend,
            tesseract_lang=tesseract_lang,
            ollama_model=ollama_model,
        )

        template_id = result.draft.selected_template_id
        if not template_id:
            raise HTTPException(status_code=422, detail="No template selected by the pipeline.")

        template_text = load_template_text(template_id, Path(measures_dir).resolve())
        fields = extract_case_fields(result.extracted_case.anonymized_text)
        txt_content = fill_template_text(template_text, fields)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as pdf_tmp:
            pdf_path = Path(pdf_tmp.name)
        text_to_simple_pdf(txt_content, pdf_path)
        pdf_bytes = pdf_path.read_bytes()
        pdf_path.unlink(missing_ok=True)

        return {
            "template_id": template_id,
            "filename": f"{Path(file.filename or 'denuncia').stem}_{template_id}.pdf",
            "pdf_bytes": pdf_bytes.hex(),
            "fields": fields,
        }
    except HTTPException:
        raise
    except Exception as exc:
        if _should_persist_input_rejection(exc):
            mongo_store.save_run(
                build_run_payload(
                    route="/generate-draft-pdf",
                    filename=file.filename,
                    ocr_backend=ocr_backend,
                    tesseract_lang=tesseract_lang,
                    ollama_model=ollama_model,
                    error=str(exc),
                )
            )
        _raise_api_error(exc, status_code=422)
    finally:
        tmp_path.unlink(missing_ok=True)
