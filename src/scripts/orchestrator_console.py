from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.orchestration.orchestrator import JudicialOrchestrator


def _print_stage(name: str, payload: object) -> None:
    print(f"\n=== {name.upper()} ===")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Console runner to inspect the judicial orchestrator step by step."
    )
    parser.add_argument("--pdf", type=str, default=None, help="Ruta al PDF a procesar.")
    parser.add_argument("--text", type=str, default=None, help="Texto ya anonimizado o de prueba.")
    parser.add_argument("--ocr-backend", type=str, default="tesseract", choices=["tesseract", "ollama"])
    parser.add_argument("--tesseract-lang", type=str, default="spa")
    parser.add_argument("--ollama-model", type=str, default="llama3.2-vision")
    parser.add_argument("--retrieval-provider", type=str, default=None)
    parser.add_argument("--retrieval-model", type=str, default=None)
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=[
            "all",
            "ingestion",
            "extraction",
            "retrieval",
            "drafting",
            "citation_verifier",
            "alerting",
            "pipeline_output",
        ],
        help="Mostrar una etapa puntual o todas.",
    )
    args = parser.parse_args()

    orchestrator = JudicialOrchestrator()
    if args.pdf:
        trace = orchestrator.trace_from_pdf(
            pdf_path=Path(args.pdf),
            ocr_backend=args.ocr_backend,
            tesseract_lang=args.tesseract_lang,
            ollama_model=args.ollama_model,
            retrieval_provider=args.retrieval_provider,
            retrieval_model=args.retrieval_model,
        )
    else:
        text = args.text or (
            "Se denuncia amenaza reiterada, convivencia reciente e incumplimiento de perimetral. "
            "La victima refiere presencia de hijos menores y agresiones previas."
        )
        trace = orchestrator.trace(
            text,
            retrieval_provider=args.retrieval_provider,
            retrieval_model=args.retrieval_model,
        )

    if args.stage == "all":
        for name in (
            "ingestion",
            "extraction",
            "retrieval",
            "drafting",
            "citation_verifier",
            "alerting",
            "pipeline_output",
        ):
            _print_stage(name, trace[name])
        return

    _print_stage(args.stage, trace[args.stage])


if __name__ == "__main__":
    main()
