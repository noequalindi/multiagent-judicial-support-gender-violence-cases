import argparse
import json
from src.orchestration.orchestrator import JudicialOrchestrator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=str, default=None, help="Ruta al PDF de denuncia escaneada.")
    parser.add_argument("--ocr-backend", type=str, default="tesseract", choices=["tesseract", "ollama"])
    parser.add_argument("--tesseract-lang", type=str, default="spa")
    parser.add_argument("--ollama-model", type=str, default="llama3.2-vision")
    args = parser.parse_args()

    orchestrator = JudicialOrchestrator()
    if args.pdf:
        result = orchestrator.run_from_pdf(
            pdf_path=args.pdf,
            ocr_backend=args.ocr_backend,
            tesseract_lang=args.tesseract_lang,
            ollama_model=args.ollama_model,
        )
    else:
        sample_case = (
            "Se denuncia amenaza reiterada e incumplimiento de perimetral. "
            "La victima refiere presencia de hijos menores y episodios de lesion previa."
        )
        result = orchestrator.run(sample_case)

    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
