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

    def test_anonymize_redacts_sensitive_ids_and_addresses(self) -> None:
        text = (
            "N° de Denuncia: FVODO02452-0050264/2026\n"
            "Pais: Argentina Provincia: Buenos Aires\n"
            "Calle: Almafuerte 123\n"
            "Número de documento: 30111222\n"
        )
        anonymized = IngestionAgent._anonymize(text)

        self.assertIn("[ID_REDACTED]", anonymized)
        self.assertIn("[DOMICILIO_REDACTED]", anonymized)
        self.assertIn("[DNI_REDACTED]", anonymized)
        self.assertNotIn("FVODO02452", anonymized)
        self.assertNotIn("Almafuerte 123", anonymized)


if __name__ == "__main__":
    unittest.main()
