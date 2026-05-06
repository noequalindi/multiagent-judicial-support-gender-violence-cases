from __future__ import annotations

import unittest

from src.agents.ingestion_agent import IngestionAgent


class IngestionAgentTests(unittest.TestCase):
    def test_extract_case_id_prefers_denuncia_number(self) -> None:
        text = "N° de Denuncia: FVODO02452-0050264/2026"
        self.assertEqual(
            IngestionAgent._extract_case_id(text),
            "FVODO02452-0050264/2026",
        )

    def test_extract_case_id_from_legajo_with_prefix(self) -> None:
        text = "Legajo OVDyG 679/2022"
        self.assertEqual(
            IngestionAgent._extract_case_id(text),
            "OVDYG679/2022",
        )

    def test_extract_case_id_from_legajo_number(self) -> None:
        text = "LEGAJO N°: 6308/2024"
        self.assertEqual(
            IngestionAgent._extract_case_id(text),
            "6308/2024",
        )

    def test_extract_case_id_falls_back_to_filename(self) -> None:
        text = "Denuncia sin número identificatorio visible."
        self.assertEqual(
            IngestionAgent._extract_case_id(text, fallback_name="doc1565338807.pdf"),
            "DOC1565338807",
        )

    def test_extract_case_id_does_not_use_exp_ocr_words_without_digits(self) -> None:
        text = "EXPRESANADOLE hechos EXPERIMENTADAS sin identificador útil."
        self.assertEqual(
            IngestionAgent._extract_case_id(text, fallback_name="IDR.pdf"),
            "IDR",
        )

    def test_anonymize_redacts_sensitive_ids_and_addresses(self) -> None:
        agent = IngestionAgent()
        text = (
            "N° de Denuncia: FVODO02452-0050264/2026\n"
            "Pais: Argentina Provincia: Buenos Aires\n"
            "Calle: Almafuerte 123\n"
            "Número de documento: 30111222\n"
        )
        anonymized = agent._anonymize(text)

        self.assertIn("[ID_REDACTED]", anonymized)
        self.assertIn("[DOMICILIO_REDACTED]", anonymized)
        self.assertIn("[DNI_REDACTED]", anonymized)
        self.assertNotIn("FVODO02452", anonymized)
        self.assertNotIn("Almafuerte 123", anonymized)
        self.assertNotIn("30111222", anonymized)

        audit = agent._audit_anonymization_residuals(text, anonymized)
        self.assertFalse(audit["has_residuals"])
        self.assertEqual(audit["counts"]["addresses"], 0)
        self.assertEqual(audit["counts"]["dni"], 0)

    def test_anonymization_audit_detects_residual_names_addresses_and_dni(self) -> None:
        raw = (
            "La denunciante Maria Gomez, con domicilio en Calle Falsa 123, "
            "informó su DNI 30111222."
        )
        anonymized = raw

        audit = IngestionAgent._audit_anonymization_residuals(raw, anonymized)

        self.assertTrue(audit["has_residuals"])
        self.assertGreaterEqual(audit["counts"]["names"], 1)
        self.assertGreaterEqual(audit["counts"]["addresses"], 1)
        self.assertGreaterEqual(audit["counts"]["dni"], 1)

    def test_clean_ocr_noise_repairs_common_mojibake(self) -> None:
        cleaned = IngestionAgent._clean_ocr_noise(
            "Urgente: riesgo de daæo físico inminente. Ambiental a travØs de la rotura."
        )

        self.assertIn("daño físico", cleaned)
        self.assertIn("a través de la rotura", cleaned)

    def test_clean_ocr_noise_repairs_common_ocr_suffixes(self) -> None:
        cleaned = IngestionAgent._clean_ocr_noise(
            "Se requiere protecciØn e intervenciØn institucional."
        )

        self.assertIn("protección", cleaned)
        self.assertIn("intervención", cleaned)


if __name__ == "__main__":
    unittest.main()
