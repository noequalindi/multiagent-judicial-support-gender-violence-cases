from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def resolve_catalog_path(path: Path) -> Path:
    if path.exists():
        return path
    alt_paths = [
        path.with_suffix(".jsonl"),
        path.with_suffix(".json"),
    ]
    for alt_path in alt_paths:
        if alt_path.exists():
            return alt_path
    raise FileNotFoundError(f"Catalog file not found: {path}")


def load_catalog(path: Path) -> list[dict[str, Any]]:
    path = resolve_catalog_path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    measures = data.get("measures")
    if not isinstance(measures, list):
        raise ValueError("Catalog must contain a top-level 'measures' list.")
    return measures


def load_template_text(template_path: Path) -> str:
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")

    if template_path.suffix.lower() == ".txt":
        return template_path.read_text(encoding="utf-8", errors="ignore")

    proc = subprocess.run(["pdftotext", str(template_path), "-"], capture_output=True, text=True, check=True)
    return proc.stdout


def normalize_ws(text: str) -> str:
    return " ".join((text or "").split())


def build_template_record(entry: dict[str, Any], template_text: str) -> dict[str, Any]:
    template_file = str(entry["template_file"])
    return {
        "_id": f"{entry['template_id']}_template",
        "source_type": "template",
        "measure_type": entry["measure_type"],
        "template_id": entry["template_id"],
        "template_file": template_file,
        "title": entry["title"],
        "summary": entry["summary"],
        "text": normalize_ws(template_text),
        "jurisdiction": entry.get("jurisdiction", ""),
        "official_source": {
            "institution": "Juzgado",
            "title": entry["title"],
            "url": "",
        },
        "content_status": "template_text",
    }


def build_reference_record(entry: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    official_text = normalize_ws(str(ref.get("text", "")))
    template_excerpt = normalize_ws(str(ref.get("template_excerpt", "")))
    summary = normalize_ws(str(ref.get("summary", "")))

    official_source = ref.get("official_source", {})
    if not isinstance(official_source, dict):
        official_source = {
            "institution": "Pendiente",
            "title": str(official_source),
            "url": "",
        }

    if official_text:
        text = official_text
        content_status = "official_text"
    elif template_excerpt:
        text = template_excerpt
        content_status = "template_excerpt"
    else:
        text = summary
        content_status = "summary_only"

    return {
        "_id": f"{entry['template_id']}_{ref['id_suffix']}",
        "source_type": "normativa",
        "measure_type": entry["measure_type"],
        "template_id": entry["template_id"],
        "template_file": entry["template_file"],
        "jurisdiction": entry.get("jurisdiction", ""),
        "law": ref.get("law", ""),
        "article": ref.get("article", ""),
        "title": ref.get("title", ""),
        "summary": summary,
        "text": text,
        "official_source": {
            "institution": str(official_source.get("institution", "")).strip(),
            "title": str(official_source.get("title", "")).strip(),
            "url": str(official_source.get("url", "")).strip(),
        },
        "template_excerpt": template_excerpt,
        "content_status": content_status,
    }


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def normalize_external_record(row: dict[str, Any], source_path: Path) -> dict[str, Any]:
    normalized = dict(row)
    record_id = str(normalized.get("_id") or normalized.get("doc_id") or "").strip()
    if not record_id:
        raise ValueError(f"External legal corpus row without id in {source_path}")

    normalized["_id"] = record_id
    normalized["source_id"] = str(normalized.get("source_id") or record_id)
    normalized["text"] = normalize_ws(
        str(
            normalized.get("text")
            or normalized.get("embedding_text")
            or normalized.get("summary")
            or ""
        )
    )
    normalized["summary"] = normalize_ws(str(normalized.get("summary", "")))
    normalized["title"] = normalize_ws(str(normalized.get("title", "")))
    normalized["law"] = normalize_ws(str(normalized.get("law", "")))
    normalized["article"] = normalize_ws(str(normalized.get("article", "")))
    normalized["jurisdiction"] = normalize_ws(str(normalized.get("jurisdiction", "")))
    normalized["content_status"] = str(normalized.get("content_status") or "external_curated")
    normalized["source_file"] = str(source_path.relative_to(Path.cwd()))

    official_url = str(normalized.get("official_url", "")).strip()
    official_source = normalized.get("official_source")
    if not isinstance(official_source, dict):
        official_source = {}
    normalized["official_source"] = {
        "institution": str(official_source.get("institution", "")).strip(),
        "title": str(official_source.get("title", normalized.get("title", ""))).strip(),
        "url": str(official_source.get("url", official_url)).strip(),
    }
    return normalized


def load_external_corpora(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for directory in paths:
        if not directory.exists():
            continue
        for jsonl_path in sorted(directory.glob("*.jsonl")):
            for row in load_jsonl_rows(jsonl_path):
                normalized = normalize_external_record(row, jsonl_path.resolve())
                record_id = str(normalized["_id"])
                if record_id in seen_ids:
                    raise ValueError(f"Duplicate external corpus id '{record_id}' in {jsonl_path}")
                seen_ids.add(record_id)
                rows.append(normalized)
    return rows


def record_checksum(row: dict[str, Any]) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_manifest(rows: list[dict[str, Any]], corpus_path: Path) -> dict[str, Any]:
    manifest_rows: list[dict[str, str]] = []
    for row in rows:
        record_id = str(row.get("_id") or row.get("source_id") or "").strip()
        if not record_id:
            continue
        manifest_rows.append(
            {
                "_id": record_id,
                "source_id": str(row.get("source_id") or record_id).strip(),
                "source_type": str(row.get("source_type", "")).strip(),
                "template_id": str(row.get("template_id", "")).strip(),
                "title": str(row.get("title", "")).strip(),
                "checksum": record_checksum(row),
            }
        )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus_path": str(corpus_path),
        "record_count": len(manifest_rows),
        "records": manifest_rows,
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_external_corpus_dirs(root: Path) -> list[Path]:
    candidates = [
        root / "data" / "legal" / "normativas",
        root / "data" / "legal" / "normativa",
        root / "data" / "legal" / "protocolos",
        root / "data" / "legal" / "jurisprudencia",
    ]
    return [path for path in candidates if path.exists()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a JSONL corpus for Pinecone from judicial measure templates and legal references."
    )
    parser.add_argument(
        "--catalog",
        default="data/legal/templates/measure_corpus_catalog.json",
        help="JSON catalog describing templates and legal references.",
    )
    parser.add_argument(
        "--measures-dir",
        default="data/measures",
        help="Directory containing judicial template files.",
    )
    parser.add_argument(
        "--out",
        default="data/legal/templates/measure_corpus.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--manifest-out",
        default="data/legal/templates/measure_corpus.manifest.json",
        help="Output manifest path with ids and checksums for incremental refresh.",
    )
    args = parser.parse_args()

    root = Path.cwd()
    catalog_path = (root / args.catalog).resolve()
    measures_dir = (root / args.measures_dir).resolve()
    out_path = (root / args.out).resolve()
    manifest_path = (root / args.manifest_out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for entry in load_catalog(catalog_path):
        template_path = measures_dir / str(entry["template_file"])
        template_text = load_template_text(template_path)
        rows.append(build_template_record(entry, template_text))
        for ref in entry.get("legal_references", []):
            rows.append(build_reference_record(entry, ref))

    external_dirs = resolve_external_corpus_dirs(root)
    rows.extend(load_external_corpora(external_dirs))

    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = build_manifest(rows, out_path)
    write_manifest(manifest_path, manifest)

    print(f"Generated corpus rows: {len(rows)}")
    print(f"Output: {out_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
