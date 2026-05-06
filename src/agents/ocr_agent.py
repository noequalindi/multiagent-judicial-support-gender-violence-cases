from __future__ import annotations

import base64
import json
import re
import subprocess
import tempfile
from pathlib import Path

import httpx


class OCRAgent:
    """OCR for image-based PDFs using Tesseract or Ollama vision models."""

    def extract_text_from_pdf(
        self,
        pdf_path: str | Path,
        backend: str = "tesseract",
        tesseract_lang: str = "spa",
        ollama_model: str = "llama3.2-vision",
        dpi: int = 300,
    ) -> str:
        pdf = Path(pdf_path).expanduser().resolve()
        if not pdf.exists():
            raise FileNotFoundError(f"PDF not found: {pdf}")

        with tempfile.TemporaryDirectory(prefix="vjfc_ocr_") as tmpdir:
            page_images = self._render_pdf_to_png(pdf, Path(tmpdir), dpi=dpi)
            if not page_images:
                raise RuntimeError("No pages rendered from PDF for OCR.")

            texts: list[str] = []
            for img in page_images:
                if backend == "tesseract":
                    texts.append(self._ocr_page_tesseract(img, lang=tesseract_lang))
                elif backend == "ollama":
                    texts.append(self._ocr_page_ollama(img, model=ollama_model))
                elif backend == "hybrid":
                    texts.append(
                        self._ocr_page_hybrid(
                            img,
                            lang=tesseract_lang,
                            model=ollama_model,
                        )
                    )
                else:
                    raise ValueError(f"Unsupported OCR backend: {backend}")
            return "\n\n".join(t.strip() for t in texts if t and t.strip())

    @staticmethod
    def _render_pdf_to_png(pdf: Path, out_dir: Path, dpi: int = 300) -> list[Path]:
        prefix = out_dir / "page"
        cmd = [
            "pdftoppm",
            "-png",
            "-r",
            str(dpi),
            str(pdf),
            str(prefix),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return sorted(out_dir.glob("page-*.png"))

    @staticmethod
    def _ocr_page_tesseract(image_path: Path, lang: str = "spa") -> str:
        cmd = [
            "tesseract",
            str(image_path),
            "stdout",
            "-l",
            lang,
            "--psm",
            "6",
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return result.stdout

    @staticmethod
    def _ocr_page_ollama(image_path: Path, model: str) -> str:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Transcribe esta imagen de documento judicial a texto plano en español. "
                        "Conserva saltos de línea y puntuación en lo posible. "
                        "Devuelve solo la transcripción."
                    ),
                    "images": [b64],
                }
            ],
            "stream": False,
            "options": {"temperature": 0},
        }

        with httpx.Client(timeout=120.0) as client:
            resp = client.post("http://127.0.0.1:11434/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, dict) or "message" not in data:
                raise RuntimeError(f"Unexpected Ollama response: {json.dumps(data)[:500]}")
            message = data.get("message") or {}
            content = message.get("content")
            if not isinstance(content, str):
                raise RuntimeError(f"Ollama response without message.content: {json.dumps(data)[:500]}")
            return content

    def _ocr_page_hybrid(self, image_path: Path, lang: str, model: str) -> str:
        try:
            tesseract_text = self._ocr_page_tesseract(image_path, lang=lang)
        except Exception:
            return self._ocr_page_ollama(image_path, model=model)

        if self._is_low_quality_ocr(tesseract_text):
            try:
                return self._ocr_page_ollama(image_path, model=model)
            except Exception:
                return tesseract_text
        return tesseract_text

    @staticmethod
    def _is_low_quality_ocr(text: str) -> bool:
        normalized = (text or "").strip()
        if len(normalized) < 80:
            return True

        lines = [line.strip() for line in normalized.splitlines() if line.strip()]
        if not lines:
            return True

        alpha_count = sum(ch.isalpha() for ch in normalized)
        digit_count = sum(ch.isdigit() for ch in normalized)
        weird_count = sum(
            1
            for ch in normalized
            if not (ch.isalnum() or ch.isspace() or ch in ".,;:()[]{}-_/\"'%+")
        )
        very_short_lines = sum(1 for line in lines if len(line) <= 3)
        garbage_lines = sum(
            1
            for line in lines
            if re.fullmatch(r"[A-Za-z0-9|/\\'\"`´.,()\-]{1,3}", line or "")
        )

        char_count = max(len(normalized), 1)
        weird_ratio = weird_count / char_count
        alpha_ratio = alpha_count / char_count
        short_ratio = very_short_lines / max(len(lines), 1)
        garbage_ratio = garbage_lines / max(len(lines), 1)

        return (
            weird_ratio > 0.08
            or alpha_ratio < 0.35
            or (digit_count > alpha_count and alpha_count < 120)
            or short_ratio > 0.35
            or garbage_ratio > 0.2
        )
