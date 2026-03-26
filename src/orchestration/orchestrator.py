from __future__ import annotations

from pathlib import Path
from typing import Any

from src.agents.alerting_agent import AlertingAgent
from src.agents.citation_verifier_agent import CitationVerifierAgent
from src.agents.drafting_agent import DraftingAgent
from src.agents.extraction_agent import ExtractionAgent
from src.agents.ingestion_agent import IngestionAgent
from src.agents.retrieval_agent import RetrievalAgent
from src.guardrails.output_validator import validate_output
from src.logging import get_logger
from src.models.contracts import PipelineOutput


class JudicialOrchestrator:
    def __init__(self) -> None:
        self.logger = get_logger("orchestrator")
        self.ingestion = IngestionAgent()
        self.extraction = ExtractionAgent()
        self.retrieval = RetrievalAgent()
        self.drafting = DraftingAgent()
        self.citation_verifier = CitationVerifierAgent()
        self.alerting = AlertingAgent()

    def run(self, raw_text: str) -> PipelineOutput:
        ingest = self.ingestion.invoke(raw_text)
        return self.run_from_ingest_payload(ingest)

    def run_from_pdf(
        self,
        pdf_path: str | Path,
        ocr_backend: str = "tesseract",
        tesseract_lang: str = "spa",
        ollama_model: str = "llama3.2-vision",
    ) -> PipelineOutput:
        ingest = self.ingestion.invoke_pdf(
            pdf_path=pdf_path,
            ocr_backend=ocr_backend,
            tesseract_lang=tesseract_lang,
            ollama_model=ollama_model,
        )
        return self.run_from_ingest_payload(ingest)

    def run_from_ingest_payload(self, ingest: dict) -> PipelineOutput:
        return self._run_pipeline(ingest)

    def trace(
        self,
        raw_text: str,
        retrieval_provider: str | None = None,
        retrieval_model: str | None = None,
    ) -> dict[str, Any]:
        ingest = self.ingestion.invoke(raw_text)
        return self.trace_from_ingest_payload(
            ingest,
            retrieval_provider=retrieval_provider,
            retrieval_model=retrieval_model,
        )

    def trace_from_pdf(
        self,
        pdf_path: str | Path,
        ocr_backend: str = "tesseract",
        tesseract_lang: str = "spa",
        ollama_model: str = "llama3.2-vision",
        retrieval_provider: str | None = None,
        retrieval_model: str | None = None,
    ) -> dict[str, Any]:
        ingest = self.ingestion.invoke_pdf(
            pdf_path=pdf_path,
            ocr_backend=ocr_backend,
            tesseract_lang=tesseract_lang,
            ollama_model=ollama_model,
        )
        return self.trace_from_ingest_payload(
            ingest,
            retrieval_provider=retrieval_provider,
            retrieval_model=retrieval_model,
        )

    def trace_from_ingest_payload(
        self,
        ingest: dict,
        retrieval_provider: str | None = None,
        retrieval_model: str | None = None,
    ) -> dict[str, Any]:
        extracted = self.extraction.invoke(ingest["case_id"], ingest["anonymized_text"])
        retrieved = self.retrieval.invoke(
            extracted,
            provider=retrieval_provider,
            model=retrieval_model,
        )
        draft = self.drafting.invoke(extracted, retrieved)
        verified_draft = self.citation_verifier.invoke(draft)
        alerts = self.alerting.invoke(verified_draft)

        payload = PipelineOutput(
            case_id=ingest["case_id"],
            extracted_case=extracted,
            retrieval=retrieved,
            draft=verified_draft,
            alerts=alerts,
        )
        validate_output(payload.model_dump())
        return {
            "ingestion": ingest,
            "extraction": extracted.model_dump(),
            "retrieval": [item.model_dump() for item in retrieved],
            "drafting": draft.model_dump(),
            "citation_verifier": verified_draft.model_dump(),
            "alerting": alerts,
            "pipeline_output": payload.model_dump(),
        }

    def run_with_retrieval_provider(
        self,
        raw_text: str,
        retrieval_provider: str | None = None,
        retrieval_model: str | None = None,
    ) -> PipelineOutput:
        ingest = self.ingestion.invoke(raw_text)
        return self._run_pipeline(
            ingest,
            retrieval_provider=retrieval_provider,
            retrieval_model=retrieval_model,
        )

    def run_from_pdf_with_retrieval_provider(
        self,
        pdf_path: str | Path,
        ocr_backend: str = "tesseract",
        tesseract_lang: str = "spa",
        ollama_model: str = "llama3.2-vision",
        retrieval_provider: str | None = None,
        retrieval_model: str | None = None,
    ) -> PipelineOutput:
        ingest = self.ingestion.invoke_pdf(
            pdf_path=pdf_path,
            ocr_backend=ocr_backend,
            tesseract_lang=tesseract_lang,
            ollama_model=ollama_model,
        )
        return self._run_pipeline(
            ingest,
            retrieval_provider=retrieval_provider,
            retrieval_model=retrieval_model,
        )

    def _run_pipeline(
        self,
        ingest: dict,
        retrieval_provider: str | None = None,
        retrieval_model: str | None = None,
    ) -> PipelineOutput:
        case_id = str(ingest.get("case_id", "unknown"))
        self.logger.info(
            "pipeline_start case_id={} retrieval_provider={} retrieval_model={}",
            case_id,
            retrieval_provider or "local",
            retrieval_model or "-",
        )
        extracted = self.extraction.invoke(ingest["case_id"], ingest["anonymized_text"])
        self.logger.info(
            "extraction_done case_id={} facts={} risk_factors={}",
            case_id,
            len(extracted.facts),
            len(extracted.risk_factors),
        )
        retrieved = self.retrieval.invoke(
            extracted,
            provider=retrieval_provider,
            model=retrieval_model,
        )
        self.logger.info(
            "retrieval_done case_id={} queries={} hits={}",
            case_id,
            len(retrieved),
            sum(len(item.hits) for item in retrieved),
        )
        draft = self.drafting.invoke(extracted, retrieved)
        self.logger.info(
            "drafting_done case_id={} selected_template_id={}",
            case_id,
            draft.selected_template_id or "-",
        )
        verified_draft = self.citation_verifier.invoke(draft)
        alerts = self.alerting.invoke(verified_draft)
        self.logger.info(
            "pipeline_done case_id={} alerts={} selected_template_id={}",
            case_id,
            len(alerts),
            verified_draft.selected_template_id or "-",
        )

        payload = PipelineOutput(
            case_id=ingest["case_id"],
            extracted_case=extracted,
            retrieval=retrieved,
            draft=verified_draft,
            alerts=alerts,
        )
        validate_output(payload.model_dump())
        return payload
