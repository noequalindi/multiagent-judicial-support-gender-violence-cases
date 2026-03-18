# Corpus Expansion Plan

This file defines the recommended structure for expanding the legal corpus beyond the current measure templates.

## Goals

1. Improve legal retrieval coverage.
2. Separate normative support from drafting templates.
3. Prepare the system for hybrid retrieval with Pinecone and, later, Elasticsearch.

## Recommended corpus families

### 1. Templates

Current source:

- `data/measures/`
- `data/legal/measure_corpus_catalog.json`

Purpose:

- draft generation
- explicit linkage between measure type and legal references

### 2. Normativa

Suggested content:

- Ley 12.569
- Decreto 2875/05
- Ley 26.485
- CPCC Provincia de Buenos Aires
- Codigo Penal (relevant articles)
- CCyCN (relevant articles)
- Constitucion Nacional
- Convencion de Belem do Para
- CADH / Pacto de San Jose de Costa Rica
- Convencion sobre los Derechos del Nino

Recommended granularity:

- one JSON/JSONL row per article
- or one row per short article group when they always apply together

Suggested fields:

- `_id`
- `source_type=normativa`
- `jurisdiction`
- `law`
- `article`
- `title`
- `summary`
- `text`
- `official_source`
- `topic`
- `measure_type` (optional)

### 3. Jurisprudencia

Suggested content:

- curated court decisions related to:
  - exclusion from home
  - no-contact orders
  - proximity restrictions
  - children involved
  - prior non-compliance
  - urgent precautionary measures

Recommended granularity:

- one row per judgment summary
- optionally one row per key reasoning fragment

Suggested fields:

- `_id`
- `source_type=jurisprudencia`
- `court`
- `jurisdiction`
- `date`
- `case_name`
- `measure_type`
- `topic`
- `summary`
- `text`
- `official_source`

## Suggested folders

These can be added progressively:

- `data/legal/normativa/`
- `data/legal/jurisprudencia/`
- `data/legal/protocolos/`

## Indexing strategy

### Pinecone

Use for:

- semantic retrieval
- template-to-law linkage
- related legal support

### Elasticsearch

Use later for:

- BM25 retrieval
- exact article and law matching
- metadata filters
- hybrid retrieval with Pinecone

## Recommendation

Keep `measure_corpus_catalog.json` as the current editable source of truth for template-linked material, and add separate curated corpora for normativa and jurisprudencia as the legal base grows.
