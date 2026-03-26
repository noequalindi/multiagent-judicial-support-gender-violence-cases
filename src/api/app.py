from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.llm.measure_classifier import MeasureClassifierService
from src.llm.registry import ocr_options, provider_options
from src.logging import get_logger
from src.models.contracts import MeasureClassification
from src.orchestration.orchestrator import JudicialOrchestrator
from src.scripts.generate_measure_document import (
    extract_case_fields,
    fill_template_text,
    load_template_text,
    text_to_simple_pdf,
)
from src.storage.mongo_store import MongoRunStore, build_run_payload
from src.storage.local_json_store import LocalClassificationStore, build_classified_case_payload


app = FastAPI(title="Violence Judicial Assistant API", version="0.1.0")
mongo_store = MongoRunStore()
local_classification_store = LocalClassificationStore()
api_logger = get_logger("api")


def _raise_api_error(exc: Exception, status_code: int = 500) -> None:
    raise HTTPException(status_code=status_code, detail=str(exc))


class ProcessTextRequest(BaseModel):
    text: str = Field(min_length=1)


class ProcessTextResponse(BaseModel):
    result: dict


class GenerateDraftResponse(BaseModel):
    template_id: str
    txt_content: str
    fields: dict[str, str]


class ClassifyMeasureRequest(BaseModel):
    text: str = Field(min_length=1)
    provider: str = Field(default="openai")
    model: str | None = None
    include_draft: bool = False


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/options/llm")
def llm_options() -> dict[str, object]:
    return {"providers": provider_options()}


@app.get("/options/ocr")
def ocr_runtime_options() -> dict[str, object]:
    return ocr_options()


@app.post("/process-text", response_model=ProcessTextResponse)
def process_text(payload: ProcessTextRequest) -> ProcessTextResponse:
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
        mongo_store.save_run(build_run_payload(route="/process-text", error=str(exc)))
        _raise_api_error(exc)


@app.post("/classify-measure", response_model=MeasureClassification)
def classify_measure(payload: ClassifyMeasureRequest) -> MeasureClassification:
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


@app.post("/generate-draft", response_model=GenerateDraftResponse)
async def generate_draft(
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
