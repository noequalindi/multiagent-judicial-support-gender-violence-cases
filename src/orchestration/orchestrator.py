from __future__ import annotations

from pathlib import Path

from src.agents.alerting_agent import AlertingAgent
from src.agents.citation_verifier_agent import CitationVerifierAgent
from src.agents.drafting_agent import DraftingAgent
from src.agents.extraction_agent import ExtractionAgent
from src.agents.ingestion_agent import IngestionAgent
from src.agents.retrieval_agent import RetrievalAgent
from src.guardrails.output_validator import validate_output
from src.models.contracts import PipelineOutput


class JudicialOrchestrator:
    def __init__(self) -> None:
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

    def _run_pipeline(self, ingest: dict) -> PipelineOutput:
        extracted = self.extraction.invoke(ingest["case_id"], ingest["anonymized_text"])
        retrieved = self.retrieval.invoke(extracted)
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
        return payload
