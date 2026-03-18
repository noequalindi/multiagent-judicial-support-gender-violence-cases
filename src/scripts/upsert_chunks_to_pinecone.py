from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.rag.pinecone_client import PineconeConfig, PineconeTextIndexClient


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def to_records(rows: list[dict], text_field: str = "chunk_text") -> list[dict]:
    records: list[dict] = []
    for r in rows:
        chunk_id = str(r.get("chunk_id") or r.get("_id") or "")
        if not chunk_id:
            continue
        txt = str(r.get("text") or r.get(text_field) or "").strip()
        if not txt:
            continue
        official_source = r.get("official_source") or {}
        if isinstance(official_source, dict):
            official_institution = str(official_source.get("institution", ""))
            official_title = str(official_source.get("title", ""))
            official_url = str(official_source.get("url", ""))
        else:
            official_institution = ""
            official_title = str(official_source)
            official_url = ""

        rec = {
            "_id": chunk_id,
            text_field: txt,
            "source_pdf": str(r.get("source_pdf", "")),
            "source_id": str(r.get("source_id", chunk_id)),
            "category": str(r.get("category", r.get("source_type", "denuncia"))),
            "source_type": str(r.get("source_type", "")),
            "measure_type": str(r.get("measure_type", "")),
            "template_id": str(r.get("template_id", "")),
            "template_file": str(r.get("template_file", "")),
            "title": str(r.get("title", "")),
            "summary": str(r.get("summary", "")),
            "law": str(r.get("law", "")),
            "article": str(r.get("article", "")),
            "jurisdiction": str(r.get("jurisdiction", "")),
            "content_status": str(r.get("content_status", "")),
            "official_source_institution": official_institution,
            "official_source_title": official_title,
            "official_source_url": official_url,
        }
        records.append(rec)
    return records


def batched(items: list[dict], size: int) -> list[list[dict]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True, help="Archivo jsonl con chunks.")
    parser.add_argument("--batch-size", type=int, default=96)
    args = parser.parse_args()

    cfg = PineconeConfig.from_env()
    if not cfg:
        raise RuntimeError("Missing PINECONE_API_KEY and/or PINECONE_INDEX_HOST environment variables.")
    client = PineconeTextIndexClient(cfg)

    rows = load_jsonl(Path(args.jsonl).resolve())
    records = to_records(rows, text_field=cfg.text_field)
    if not records:
        print("No records to upsert.")
        return

    total = 0
    for i, batch in enumerate(batched(records, args.batch_size), start=1):
        client.upsert_records(batch)
        total += len(batch)
        print(f"Batch {i}: upserted {len(batch)} records (total={total})")

    print(f"Done. Upserted records: {total}")


if __name__ == "__main__":
    main()
