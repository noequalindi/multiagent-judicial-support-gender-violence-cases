import re
from pathlib import Path
from uuid import uuid4

from src.agents.ocr_agent import OCRAgent


class IngestionAgent:
    """Basic ingestion and anonymization for case file text."""

    def __init__(self) -> None:
        self.ocr = OCRAgent()

    @staticmethod
    def _clean_ocr_noise(text: str) -> str:
        text = text or ""
        text = text.replace("\x0c", "\n")
        text = re.sub(r"[·•]{2,}", " ", text)
        text = re.sub(r"[=_]{3,}", " ", text)
        text = re.sub(r"(Código de barras del trámite(?: en números)?)", " ", text, flags=re.I)
        text = re.sub(r"Emitido por el Sistema de Informaci[oó]n Delictual.*?(?=\n|$)", " ", text, flags=re.I)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _redact_form_fields(text: str) -> str:
        replacements = [
            (
                r"((?:Apellido y nombre|APELLIDO y NOMBRE|NOMBRE y APELLIDO|Apellido y Nombre)\s*[:\-]?\s*)([A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑáéíóúñ' ]{2,})",
                r"\1[PERSONA_REDACTED]",
            ),
            (
                r"((?:Tipo de documento\s*DNI.*?N[uú]mero\s*de\s*)([A-Za-z0-9\-\.]+))",
                "[DNI_REDACTED]",
            ),
            (
                r"((?:Domicilio(?: del Actor| de la Parte)?(?: \(Tipo y Nro\))?\s*[:\-]?\s*)([^\n]{4,120}))",
                r"\1[DOMICILIO_REDACTED]",
            ),
            (
                r"((?:N[°º']?\s*de\s*VG|Formulario Policial|EXPEDIENTE\s*N[°º]?|REGISTRO\s*N[°º]?)\s*[:\-]?\s*)([A-Z0-9\-/\.]+)",
                r"\1[ID_REDACTED]",
            ),
            (
                r"((?:N[°º']?\s*de\s*Denuncia|Nro\.?\s*de\s*Denuncia|Número\s*de\s*Denuncia)\s*[:\-]?\s*)([A-Z0-9\-/\.]+)",
                r"\1[ID_REDACTED]",
            ),
            (
                r"((?:Pais|Pa[ií]s|Provincia|Partido|Localidad|Calle|Altura|Entre|Piso|Departamento|Lugar Exacto|Descripci[oó]n)\s*[:\-]?\s*)([^\n]{2,120})",
                r"\1[DOMICILIO_REDACTED]",
            ),
        ]
        for pattern, repl in replacements:
            text = re.sub(pattern, repl, text, flags=re.I)
        return text

    @staticmethod
    def _anonymize(text: str) -> str:
        text = text or ""
        text = IngestionAgent._clean_ocr_noise(text)
        text = re.sub(r"\b\d{7,9}\b", "[DNI_REDACTED]", text)
        text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL_REDACTED]", text)
        text = re.sub(r"\+?\d[\d\s-]{7,}\d", "[PHONE_REDACTED]", text)
        text = re.sub(r"\b(?:FV|FVO|FVDO|FVODO|EXP|PP)[A-Z0-9\-/]{4,}\b", "[ID_REDACTED]", text, flags=re.I)
        text = re.sub(r"\b(?:[A-Z]{2,}\d{3,}[-/][A-Z0-9]+)\b", "[ID_REDACTED]", text)
        text = re.sub(r"\b(?:[A-Z]{1,4}\d{4,}[-/]\d{2,4})\b", "[ID_REDACTED]", text)
        text = IngestionAgent._redact_form_fields(text)
        text = re.sub(
            r"\b([A-ZÁÉÍÓÚÑ]{2,}(?:\s+[A-ZÁÉÍÓÚÑ]{2,}){1,3})\b",
            "[PERSONA_REDACTED]",
            text,
        )
        text = re.sub(r"\b(?:San Justo|Ramos Mej[ií]a|La Matanza|Buenos Aires)\b", "[LOCALIDAD_REDACTED]", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    @staticmethod
    def _extract_case_id(text: str) -> str:
        text = text or ""
        patterns = [
            r"(?:N[°º']?\s*de\s*Denuncia|Nro\.?\s*de\s*Denuncia|Número\s*de\s*Denuncia)\s*[:\-]?\s*([A-Z0-9\-/\.]+)",
            r"(?:EXPEDIENTE\s*N[°º]?|Expediente\s*N[°º]?)\s*[:\-]?\s*([A-Z0-9\-/\.]+)",
            r"\b((?:FV|FVO|FVDO|FVODO|EXP|PP)[A-Z0-9\-/]{4,})\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.I)
            if match:
                return str(match.group(1)).strip()
        return f"case-{uuid4().hex[:8]}"

    def invoke(self, raw_text: str) -> dict:
        return {
            "case_id": self._extract_case_id(raw_text),
            "anonymized_text": self._anonymize(raw_text),
        }

    def invoke_pdf(
        self,
        pdf_path: str | Path,
        ocr_backend: str = "tesseract",
        tesseract_lang: str = "spa",
        ollama_model: str = "llama3.2-vision",
    ) -> dict:
        extracted_text = self.ocr.extract_text_from_pdf(
            pdf_path=pdf_path,
            backend=ocr_backend,
            tesseract_lang=tesseract_lang,
            ollama_model=ollama_model,
        )
        return {
            "case_id": self._extract_case_id(extracted_text),
            "anonymized_text": self._anonymize(extracted_text),
        }
