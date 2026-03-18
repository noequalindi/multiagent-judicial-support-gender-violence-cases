from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def load_catalog(path: Path) -> list[dict[str, Any]]:
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
    args = parser.parse_args()

    root = Path.cwd()
    catalog_path = (root / args.catalog).resolve()
    measures_dir = (root / args.measures_dir).resolve()
    out_path = (root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for entry in load_catalog(catalog_path):
        template_path = measures_dir / str(entry["template_file"])
        template_text = load_template_text(template_path)
        rows.append(build_template_record(entry, template_text))
        for ref in entry.get("legal_references", []):
            rows.append(build_reference_record(entry, ref))

    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Generated corpus rows: {len(rows)}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
