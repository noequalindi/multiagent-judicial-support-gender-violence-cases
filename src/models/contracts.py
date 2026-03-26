from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class Citation(BaseModel):
    source_id: str
    excerpt: str
    title: str | None = None
    law: str | None = None
    article: str | None = None
    source_type: str | None = None
    jurisdiction: str | None = None
    relevance_score: float = Field(ge=0.0, le=1.0)


class ExtractedCase(BaseModel):
    case_id: str
    anonymized_text: str
    facts: list[str]
    active_measures: list[str]
    risk_factors: list[str]
    timeline: list[str]


class RetrievalResult(BaseModel):
    query: str
    hits: list[Citation]


class DraftDecision(BaseModel):
    decision_draft: str
    checklist: list[str]
    suggested_measures: list[str]
    selected_template_id: str | None = None
    selected_template_name: str | None = None
    template_text: str | None = None
    fields_to_fill: list[str] = []
    risk_band: Literal["low", "medium", "high"]
    abstention: bool = False
    abstention_reason: str | None = None
    citations: list[Citation] = []


class PipelineOutput(BaseModel):
    case_id: str
    extracted_case: ExtractedCase
    retrieval: list[RetrievalResult]
    draft: DraftDecision
    alerts: list[str]


class ApplicableArticle(BaseModel):
    ley: str
    articulo: str
    relevancia: str


class LegalBasisItem(BaseModel):
    ley: str
    articulo: str
    motivo: str


class MeasureClassification(BaseModel):
    provider: str
    model: str
    selected_template_id: str
    selected_template_name: str
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_source_ids: list[str] = []
    suggested_measures: list[str] = []
    risk_level: Literal["alto", "medio", "bajo"] | None = None
    risk_indicators: list[str] = []
    alerts: list[str] = []
    applicable_articles: list[ApplicableArticle] = []
    normative_basis: list[LegalBasisItem] = []
    procedural_basis: list[LegalBasisItem] = []
    low_confidence_reason: str | None = None
    draft_text: str | None = None
