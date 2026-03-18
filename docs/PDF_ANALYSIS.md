# Initial Analysis of `GdP_Qualindi_Noelia_V6` PDF

## Title and Scope

The project focuses on an interdisciplinary assistant for precautionary measures in gender-violence judicial files, with a RAG-based approach and mandatory human supervision.

## Identified Technical Components

1. Normalized key-value representation + procedural timeline.
2. RAG with jurisdiction/province-specific glossaries.
3. Multi-agent architecture:
   - ingestion/anonymization
   - extraction
   - retrieval + reranking
   - generation with guardrails
   - citation verification
   - alert orchestration
4. Lightweight fine-tuning (LoRA/QLoRA) for format/style adherence.
5. Temporal module for operational prioritization.
6. Ethical-legal protocol (PIA/DPIA, auditing, non-stigmatizing language).

## Relevant Functional Constraints

- Does not replace legal judgment or human signature.
- Output is a non-binding draft.
- Abstention policy when support is insufficient.
- Traceability through citations and decision logs.

## Metrics Mentioned in the Document

- Extraction: entity/relation F1.
- Retrieval: Recall@k, NDCG, MRR.
- Generation: supported answer rate, hallucination rate.
- Temporal risk: C-index, Brier, calibration, and subgroup fairness.
