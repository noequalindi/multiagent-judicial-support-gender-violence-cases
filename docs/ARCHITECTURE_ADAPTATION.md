# Architecture Adaptation: `ui2code-rag` -> Judicial Project

## Direct Pattern Reuse

1. Central orchestrator pattern:
   - In `ui2code-rag`: coordinates Visual Agent + RAG + Code Agent through A2A.
   - In this project: coordinates Ingestion + Extraction + Retrieval + Draft + CitationVerifier + Alerts.

2. Strict JSON contracts:
   - In `ui2code-rag`: schemas for visual/code outputs.
   - In this project: a unified judicial output schema with `decision_draft`, `citations`, `risk_band`, `abstention`.

3. Guardrails:
   - JSON + schema validation.
   - Critical field consistency checks.
   - Fallback and abstention policy.

4. RAG:
   - Hybrid strategy BM25 + embeddings + reranking (target).
   - In this phase, a simplified lexical implementation is used to start integration.

## Domain Changes

- `visual analysis` is replaced by `structured_case`.
- `html generation` is replaced by `precautionary_measures_draft`.
- `web sanitization` is replaced by:
  - PII redaction
  - citation checks
  - non-stigmatizing language checks (advanced rules pending).

## Proposed Operational Sequence

1. `IngestionAgent`: parses text and anonymizes personal data.
2. `ExtractionAgent`: extracts facts, measures, timeline, and risk factors.
3. `RetrievalAgent`: searches regulations/case law for each suggested measure.
4. `DraftingAgent`: builds the structured draft and checklist.
5. `CitationVerifierAgent`: validates that each critical claim has supporting citations.
6. `AlertingAgent`: emits alerts for high-risk bands or insufficient citations.

## Current Status in This Folder

- Functional skeleton ready.
- Executable demo with full pipeline and mock data.
- Ready to integrate your fieldwork findings and real court corpus.
