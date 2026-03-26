# Measure Corpus Workflow

This workflow builds a Pinecone-ready corpus from the court measure templates stored in `data/measures/` and merges curated legal JSONL corpora from `data/legal/`.

## Goals

1. Keep complaint OCR and extraction local while the complaint is still sensitive.
2. Build a separate legal-support corpus for retrieval, traceability, and chat citations.
3. Associate each template with the legal references cited inside it.

## Source of truth

- Template files: `data/measures/`
- Metadata catalog: `data/legal/measure_corpus_catalog.json`

Each measure entry contains:

- `template_id`
- `measure_type`
- template summary
- legal references cited in the template
- placeholders for official legal source links/text

## Generated artifact

Run:

```bash
python -m src.scripts.build_measure_corpus
```

This produces:

- `data/legal/measure_corpus.jsonl`

Each JSONL row is either:

- `source_type=template`
- `source_type=normativa`
- `source_type=protocolo`
- `source_type=jurisprudencia`
- `source_type=tratado_internacional`

## Recommended next step

Before upserting to Pinecone, enrich each legal reference with:

- official source URL
- official article text
- short human-written summary

If the official article text is not available yet, the generator falls back to:

1. template excerpt
2. summary only

The `content_status` field makes that explicit for later cleanup.
