import re
import time
import unicodedata
from pathlib import Path
from uuid import uuid4

from src.agents.ocr_agent import OCRAgent
from src.privacy.presidio_anonymizer import JudicialPresidioAnonymizer
from src.privacy.spacy_anonymizer import JudicialSpacyAnonymizer


class IngestionAgent:
    """Basic ingestion and anonymization for case file text."""

    _NUMBER_WORD_PATTERN = (
        r"(?:un[oa]?|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce|trece|"
        r"catorce|quince|diecis[eé]is|diecisiete|dieciocho|diecinueve|veinte|treinta|"
        r"cuarenta|cincuenta|sesenta|setenta|ochenta|noventa|cien|ciento|doscientos|"
        r"trescientos|cuatrocientos|quinientos|seiscientos|setecientos|ochocientos|"
        r"novecientos|mil)"
    )

    _ADDRESS_FIELD_PATTERN = (
        r"(?:Pais|Pa[ií]s|Provincia|Partido|Localidad|Barrio|Comuna|Calle|Altura|Entre|"
        r"Piso|Departamento|Lugar Exacto|Descripci[oó]n|Domicilio(?: del Actor| de la Parte)?)"
    )
    _ADDRESS_INLINE_PATTERN = (
        r"\b(?:calle|domicilio|dom\.?|av\.?|avenida|pasaje|pje\.?|ruta|camino|"
        r"barrio|comuna)\s+[A-Za-zÁÉÍÓÚÑáéíóúñ0-9°º'\"./ -]{4,120}"
    )
    _ADDRESS_CONTEXT_PATTERNS = [
        r"\b(?:con\s+domicilio|cuyo\s+domicilio|domicilio)\b\s*[:\-]?\s*(?:(?!tel[eé]fono|correo|email|presentaci[oó]n|nacionalidad|estado civil|condici[oó]n|ocupaci[oó]n|quien|al cual).){0,140}",
        r"\bdomiciliad[oa]\s+en(?:\s+la)?\s+(?:(?!tel[eé]fono|correo|email|presentaci[oó]n|nacionalidad|estado civil|condici[oó]n|ocupaci[oó]n|quien|al cual).){0,140}",
        r"\bcalle\s+(?:(?!tel[eé]fono|correo|email|presentaci[oó]n|nacionalidad|estado civil|condici[oó]n|ocupaci[oó]n|quien|al cual).){0,100}",
    ]
    _PERSON_LABEL_PATTERN = (
        r"(?:Apellido y nombre|APELLIDO y NOMBRE|NOMBRE y APELLIDO|Apellido y Nombre|"
        r"Nombre y apellido|Sr\.?|Sra\.?|Srta\.?|Damnificad[oa]|Denunciad[oa]|Víctima|Victima|"
        r"Imputad[oa]|Actora?|Actor|Progenitor[ae]?|Hijo|Hija)"
    )
    _PHONE_FIELD_PATTERN = (
        r"(?:N[uú]mero\s+de\s+celular|N[uú]mero\s+tel[eé]fono\s+fijo|Tel[eé]fono|"
        r"Celular|Whatsapp|WhatsApp|Contacto telef[oó]nico)"
    )
    _EMAIL_FIELD_PATTERN = r"(?:E-?mail|Correo(?: electr[oó]nico)?)"
    _AUDIT_NAME_STOPWORDS = {
        "argentina",
        "buenos",
        "aires",
        "provincia",
        "partido",
        "localidad",
        "barrio",
        "domicilio",
        "calle",
        "avenida",
        "pasaje",
        "ruta",
        "camino",
        "denuncia",
        "expediente",
        "documento",
        "numero",
        "número",
        "dni",
        "codigo",
        "código",
        "sistema",
        "informacion",
        "información",
        "delictual",
        "tipo",
    }

    def __init__(self) -> None:
        self.ocr = OCRAgent()
        self.presidio = JudicialPresidioAnonymizer()
        self.spacy = JudicialSpacyAnonymizer()

    @staticmethod
    def _repair_common_ocr_mojibake(text: str) -> str:
        def _replace_preserving_case(pattern: str, replacement: str, value: str) -> str:
            def _repl(match: re.Match[str]) -> str:
                token = match.group(0)
                if token.isupper():
                    return replacement.upper()
                if token[:1].isupper():
                    return replacement[:1].upper() + replacement[1:]
                return replacement

            return re.sub(pattern, _repl, value, flags=re.I)

        text = _replace_preserving_case(r"\btravØs\b", "través", text)
        text = re.sub(r"(?<=[A-Za-zÁÉÍÓÚÑáéíóúñ])æ(?=[A-Za-zÁÉÍÓÚÑáéíóúñ])", "ñ", text)
        text = re.sub(r"ciØn\b", "ción", text, flags=re.I)
        text = re.sub(r"siØn\b", "sión", text, flags=re.I)
        text = re.sub(r"giØn\b", "gión", text, flags=re.I)
        return text

    @staticmethod
    def _clean_ocr_noise(text: str) -> str:
        text = text or ""
        text = unicodedata.normalize("NFKC", text)
        text = IngestionAgent._repair_common_ocr_mojibake(text)
        text = text.replace("\x0c", "\n")
        text = text.replace("\u00a0", " ")
        text = text.replace("\u200b", "")
        text = text.replace("\ufeff", "")
        text = text.translate(
            str.maketrans(
                {
                    "“": '"',
                    "”": '"',
                    "‘": "'",
                    "’": "'",
                    "–": "-",
                    "—": "-",
                    "…": "...",
                    "•": " ",
                    "·": " ",
                    "º": "o",
                    "°": "o",
                    "ª": "a",
                    "ﬁ": "fi",
                    "ﬂ": "fl",
                    "�": " ",
                }
            )
        )
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        text = re.sub(r"[^\S\n]+", " ", text)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n[ \t]+", "\n", text)
        text = re.sub(r"[·•]{2,}", " ", text)
        text = re.sub(r"[=_]{3,}", " ", text)
        text = re.sub(r"(Código de barras del trámite(?: en números)?)", " ", text, flags=re.I)
        text = re.sub(r"Emitido por el Sistema de Informaci[oó]n Delictual.*?(?=\n|$)", " ", text, flags=re.I)
        text = re.sub(r"[\t\r]+", " ", text)
        lines: list[str] = []
        for raw_line in text.splitlines():
            line = re.sub(r"\s+", " ", raw_line).strip()
            line = re.sub(r"\s+([,.;:)\]])", r"\1", line)
            line = re.sub(r"([(\[])\s+", r"\1", line)
            line = re.sub(r"([,.;:])([A-Za-zÁÉÍÓÚÑáéíóúñ])", r"\1 \2", line)
            if re.fullmatch(r"[A-Za-z0-9|/\\'\"`´.,()\-]{1,3}", line or ""):
                continue
            lines.append(line)
        text = "\n".join(lines)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _redact_form_fields(text: str) -> str:
        replacements = [
            (
                rf"(({IngestionAgent._PERSON_LABEL_PATTERN})\s*[:\-]?\s*)([A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑáéíóúñ' ]{{2,}})",
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
                rf"(({IngestionAgent._ADDRESS_FIELD_PATTERN})\s*[:\-]?\s*)([^\n]{{2,120}})",
                r"\1[DOMICILIO_REDACTED]",
            ),
            (
                rf"(({IngestionAgent._PHONE_FIELD_PATTERN})\s*[:\-]?\s*)([^\n]{{4,40}})",
                r"\1[PHONE_REDACTED]",
            ),
            (
                rf"(({IngestionAgent._EMAIL_FIELD_PATTERN})\s*[:\-]?\s*)([^\n]{{4,120}})",
                r"\1[EMAIL_REDACTED]",
            ),
        ]
        for pattern, repl in replacements:
            text = re.sub(pattern, repl, text, flags=re.I)
        return text

    def _anonymize(self, text: str) -> str:
        text = text or ""
        text = self._clean_ocr_noise(text)
        text = self._redact_form_fields(text)
        text = self.presidio.anonymize(text)
        text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL_REDACTED]", text)
        text = re.sub(r"\(\d{2,4}\)\s*\d{3,4}[\s-]?\d{4}\b", "[PHONE_REDACTED]", text)
        text = re.sub(r"\+?\d{2,4}[\s-]\d{3,4}[\s-]\d{4}\b", "[PHONE_REDACTED]", text)
        text = re.sub(r"(\[PHONE_REDACTED\])\s*\d{6,10}\b", r"\1", text)
        text = re.sub(r"\b\d{2}\.\d{3}\.\d{3}\b", "[DNI_REDACTED]", text)
        text = re.sub(
            r"\b(?:FV|FVO|FVDO|FVODO|EXP|PP)[A-Z0-9\-/\.]{4,}\b",
            "[ID_REDACTED]",
            text,
            flags=re.I,
        )
        text = re.sub(r"\b\d{7,8}\b", "[DNI_REDACTED]", text)
        text = re.sub(self._ADDRESS_INLINE_PATTERN, "[DOMICILIO_REDACTED]", text, flags=re.I)
        for pattern in self._ADDRESS_CONTEXT_PATTERNS:
            text = re.sub(
                pattern,
                "[DOMICILIO_REDACTED] ",
                text,
                flags=re.I | re.S,
            )
        text = re.sub(
            rf"\b(?:manzana|mz\.?|casa|piso|depto\.?|departamento)\s+(?:\d{{1,5}}[A-Za-z]?|{self._NUMBER_WORD_PATTERN})(?:\s+(?:\d{{1,5}}[A-Za-z]?|{self._NUMBER_WORD_PATTERN})){{0,2}}\b",
            "[DOMICILIO_REDACTED]",
            text,
            flags=re.I,
        )
        text = re.sub(
            r"((?:DNI|Documento|N[úu]mero de documento)\s*[:\-]?\s*)([0-9.\-]{7,12})\b",
            r"\1[DNI_REDACTED]",
            text,
            flags=re.I,
        )
        text = self.spacy.anonymize(text)
        text = re.sub(
            r"\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){1,3})\b",
            "[PERSONA_REDACTED]",
            text,
        )
        text = re.sub(
            r"\b([A-ZÁÉÍÓÚÑ]{2,}(?:\s+[A-ZÁÉÍÓÚÑ]{2,}){1,3})\b",
            "[PERSONA_REDACTED]",
            text,
        )
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    @staticmethod
    def _normalize_for_audit(text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text or "")
        normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        normalized = normalized.lower()
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    @classmethod
    def _extract_name_candidates(cls, text: str) -> list[str]:
        candidates: list[str] = []
        patterns = [
            r"\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ']+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ']+){1,3}\b",
            r"\b[A-ZÁÉÍÓÚÑ]{2,}(?:\s+[A-ZÁÉÍÓÚÑ]{2,}){1,3}\b",
        ]
        for pattern in patterns:
            for match in re.findall(pattern, text or ""):
                phrase = re.sub(r"\s+", " ", match).strip(" ,.;:-")
                tokens = [cls._normalize_for_audit(token) for token in phrase.split()]
                if len(tokens) < 2:
                    continue
                if any(token in cls._AUDIT_NAME_STOPWORDS for token in tokens):
                    continue
                if phrase not in candidates:
                    candidates.append(phrase)
        return candidates[:12]

    @classmethod
    def _extract_address_candidates(cls, text: str) -> list[str]:
        patterns = [
            rf"\b(?:{cls._ADDRESS_FIELD_PATTERN})\s*[:\-]?\s*([^\n]{{4,120}})",
            r"\b(?:calle|domicilio|av\.?|avenida|pasaje|pje\.?|ruta|camino)\s+[A-Za-zÁÉÍÓÚÑáéíóúñ0-9°º'\"./ -]{4,120}",
        ]
        candidates: list[str] = []
        for pattern in patterns:
            for match in re.finditer(pattern, text or "", flags=re.I):
                value = match.group(1) if match.lastindex else match.group(0)
                value = re.sub(r"\s+", " ", value).strip(" ,.;:-")
                if len(value) < 6:
                    continue
                if value not in candidates:
                    candidates.append(value)
        return candidates[:12]

    @staticmethod
    def _extract_dni_candidates(text: str) -> list[str]:
        patterns = [
            r"\b\d{2}\.\d{3}\.\d{3}\b",
            r"\b\d{7,8}\b",
            r"(?:DNI|Documento|N[úu]mero de documento)\s*[:\-]?\s*([0-9.\-]{7,12})\b",
        ]
        candidates: list[str] = []
        for pattern in patterns:
            for match in re.finditer(pattern, text or "", flags=re.I):
                value = match.group(1) if match.lastindex else match.group(0)
                digits = re.sub(r"\D", "", value)
                if 7 <= len(digits) <= 8 and digits not in candidates:
                    candidates.append(digits)
        return candidates[:12]

    @classmethod
    def _audit_anonymization_residuals(
        cls, raw_text: str, anonymized_text: str
    ) -> dict[str, object]:
        normalized_anonymized = cls._normalize_for_audit(anonymized_text)
        digit_anonymized = re.sub(r"\D", "", anonymized_text or "")

        residual_names = [
            candidate
            for candidate in cls._extract_name_candidates(raw_text)
            if cls._normalize_for_audit(candidate) in normalized_anonymized
        ]
        residual_addresses = [
            candidate
            for candidate in cls._extract_address_candidates(raw_text)
            if cls._normalize_for_audit(candidate) in normalized_anonymized
        ]
        residual_dni = [
            candidate
            for candidate in cls._extract_dni_candidates(raw_text)
            if candidate in digit_anonymized
        ]

        return {
            "has_residuals": bool(residual_names or residual_addresses or residual_dni),
            "names": residual_names,
            "addresses": residual_addresses,
            "dni": residual_dni,
            "counts": {
                "names": len(residual_names),
                "addresses": len(residual_addresses),
                "dni": len(residual_dni),
            },
        }

    @staticmethod
    def _normalize_case_id(value: str) -> str:
        cleaned = re.sub(r"\s+", "", (value or "").strip())
        cleaned = cleaned.strip(" .,:;#-")
        return cleaned.upper()

    @classmethod
    def _extract_case_id(cls, text: str, fallback_name: str | None = None) -> str:
        text = text or ""

        patterns = [
            # Legajo OVDyG 679/2022
            r"Legajo\s+([A-Z]{2,8}\s*\d{1,8}/\d{4})",

            # LEGAJO N*: 5439/2022
            # LEGAJO N°: 5439/2022
            # Legajo N" 5439/2022
            # Nro. de Legajo: 5439/2022
            r"(?:Nro\.?\s*de\s*)?Legajo\s*(?:N\s*[°º\"'*]?\s*:?\s*)?(\d{1,8}/\d{4})",

            # N° de Denuncia: 100189/2022
            r"(?:N[°º\"'*]?\s*de\s*Denuncia|Nro\.?\s*de\s*Denuncia|Número\s*de\s*Denuncia)\s*[:\-]?\s*([A-Z0-9\-/\.]+)",

            # Expediente CIV 54560/2022 / EXPEDIENTE N° 5439/2022
            r"(?:EXPEDIENTE|Expediente|EXPTE\.?|Expte\.?)\s*(?:N\s*[°º\"'*]?\s*:?\s*)?(?:[A-Z]{2,5}\s+)?([0-9]+/\d{4})",

            # IDs internos tipo FV...
            r"\b((?:FV|FVO|FVDO|FVODO|EXP|PP)(?=[A-Z0-9\-/]{4,}\d)[A-Z0-9\-/]{4,})\b",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, flags=re.I)
            if match:
                return cls._normalize_case_id(str(match.group(1)))

        return cls._fallback_case_id(fallback_name)

    @classmethod
    def _fallback_case_id(cls, fallback_name: str | None) -> str:
        if fallback_name:
            stem = Path(fallback_name).stem
            stem = re.sub(r"[^\w\-/\.]+", "", stem, flags=re.UNICODE).strip()
            if stem:
                return cls._normalize_case_id(stem)
        return f"CASE-{uuid4().hex[:8].upper()}"

    def invoke(self, raw_text: str) -> dict:
        anonymized_text = self._anonymize(raw_text)
        audit = self._audit_anonymization_residuals(raw_text, anonymized_text)
        return {
            "case_id": self._extract_case_id(raw_text),
            "anonymized_text": anonymized_text,
            "ocr_quality": "ok",
            "ocr_warning": None,
            "ocr_duration_ms": 0,
            "anonymization_audit": audit,
            "anonymization_warning": (
                "Se detectaron posibles datos residuales no anonimizados. Revisá nombres, domicilios y DNI antes de continuar."
                if audit["has_residuals"]
                else None
            ),
        }

    def invoke_pdf(
        self,
        pdf_path: str | Path,
        ocr_backend: str = "tesseract",
        tesseract_lang: str = "spa",
        ollama_model: str = "llama3.2-vision",
    ) -> dict:
        ocr_started = time.perf_counter()

        extracted_text = self.ocr.extract_text_from_pdf(
            pdf_path=pdf_path,
            backend=ocr_backend,
            tesseract_lang=tesseract_lang,
            ollama_model=ollama_model,
        )

        ocr_duration_ms = int((time.perf_counter() - ocr_started) * 1000)
        cleaned_text = self._clean_ocr_noise(extracted_text)
        low_quality = self.ocr._is_low_quality_ocr(cleaned_text)

        anonymized_text = self._anonymize(extracted_text)
        audit = self._audit_anonymization_residuals(extracted_text, anonymized_text)

        case_id = self._extract_case_id(
            extracted_text,
            fallback_name=Path(pdf_path).name,
        )

        return {
            "case_id": case_id,
            "anonymized_text": anonymized_text,
            "ocr_quality": "low" if low_quality else "ok",
            "ocr_warning": (
                "El texto extraído presenta baja calidad OCR. Revisá el documento o usá un modo visual/híbrido antes de confiar en el preview."
                if low_quality
                else None
            ),
            "ocr_duration_ms": ocr_duration_ms,
            "anonymization_audit": audit,
            "anonymization_warning": (
                "Se detectaron posibles datos residuales no anonimizados. Revisá nombres, domicilios y DNI antes de continuar."
                if audit["has_residuals"]
                else None
            ),
        }
