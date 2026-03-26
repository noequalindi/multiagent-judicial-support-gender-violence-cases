from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.llm.measure_classifier import MeasureClassifierService
from src.models.contracts import Citation, ExtractedCase, RetrievalResult


class MeasureClassifierTests(unittest.TestCase):
    def test_anthropic_classification_maps_response_and_dedupes_supporting_sources(self) -> None:
        extracted = ExtractedCase(
            case_id="FVODO02452-0050264/2026",
            anonymized_text="Texto anonimizado con amenazas, arma y menores.",
            facts=["Se reportan amenazas.", "Se reportan agresiones fisicas o lesiones."],
            active_measures=["Prohibicion de acercamiento"],
            risk_factors=["Acceso reportado a arma", "NNA involucrados", "Convivencia actual o reciente"],
            timeline=["Ingreso de denuncia"],
        )
        retrieval = [
            RetrievalResult(
                query="amenazas y arma",
                hits=[
                    Citation(
                        source_id="ley_26485_art26",
                        excerpt="Medidas preventivas urgentes.",
                        title="Ley 26.485 art. 26",
                        law="Ley 26.485",
                        article="26",
                        source_type="normativa",
                        relevance_score=0.9,
                    ),
                    Citation(
                        source_id="fallo_001",
                        excerpt="Se confirma exclusion del hogar.",
                        title="Fallo 001",
                        law="Jurisprudencia",
                        article="",
                        source_type="jurisprudencia",
                        relevance_score=0.8,
                    ),
                    Citation(
                        source_id="fallo_001",
                        excerpt="Mismo fallo repetido por otra query.",
                        relevance_score=0.7,
                    ),
                ],
            )
        ]

        fake_cfg = SimpleNamespace(provider="anthropic", model="claude-sonnet-4-6")
        fake_response = {
            "nivel_riesgo": "alto",
            "indicadores_riesgo": ["Amenazas", "Arma", "NNA involucrados"],
            "medidas_sugeridas": ["exclusion", "perimetro"],
            "fundamentacion": {
                "sintesis_hecho": "Resumen breve del caso.",
                "razonamiento": "La combinación de amenazas, arma y NNA exige medida intensa.",
                "articulos_aplicables": [
                    {"ley": "Ley 26.485", "articulo": "26", "relevancia": "Habilita medidas urgentes."}
                ],
            },
            "normative_basis": [
                {"ley": "Ley 12.569", "articulo": "7", "motivo": "Habilita exclusión y acercamiento."}
            ],
            "procedural_basis": [
                {"ley": "Ley 12.569", "articulo": "11", "motivo": "Audiencia dentro de 48 horas."}
            ],
            "borrador_resolucion": "BORRADOR DE RESOLUCION",
            "alertas": ["Revisar legibilidad del relato."],
            "confianza": "alta",
            "motivo_baja_confianza": "",
        }
        fake_client = SimpleNamespace(chat_json=lambda **_: fake_response)

        with patch("src.llm.measure_classifier.build_json_llm_client", return_value=(fake_cfg, fake_client)):
            classification = MeasureClassifierService().classify(
                extracted=extracted,
                retrieval=retrieval,
                provider="anthropic",
            )

        self.assertEqual(classification.selected_template_id, "medida_exclusion")
        self.assertEqual(classification.risk_level, "alto")
        self.assertEqual(classification.supporting_source_ids, ["ley_26485_art26", "fallo_001"])
        self.assertEqual(classification.normative_basis[0].ley, "Ley 12.569")
        self.assertEqual(classification.procedural_basis[0].articulo, "11")
        self.assertEqual(classification.draft_text, "BORRADOR DE RESOLUCION")


if __name__ == "__main__":
    unittest.main()
