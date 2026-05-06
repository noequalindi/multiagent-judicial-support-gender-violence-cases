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
            active_measures=["Medida perimetral"],
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

    def test_generic_provider_maps_risk_alerts_and_legal_basis(self) -> None:
        extracted = ExtractedCase(
            case_id="FV00002494-0043221/2026",
            anonymized_text="Texto anonimizado con conflicto, hijos y amenaza de retención.",
            facts=["La denunciante refiere retención de menores.", "Existe denuncia previa."],
            active_measures=["Medida perimetral"],
            risk_factors=["NNA involucrados", "Antecedentes", "Violencia psicologica"],
            timeline=["Denuncia actual"],
        )
        retrieval = [
            RetrievalResult(
                query="retencion de menores",
                hits=[
                    Citation(
                        source_id="ley_12569_art7",
                        excerpt="Medidas urgentes de protección.",
                        title="Ley 12.569 art. 7",
                        law="Ley 12.569",
                        article="7",
                        source_type="normativa",
                        relevance_score=0.9,
                    )
                ],
            )
        ]

        fake_cfg = SimpleNamespace(provider="openai", model="gpt-5-mini")
        fake_response = {
            "selected_template_id": "medida_exclusion",
            "selected_template_name": "Exclusion del hogar + medida perimetral",
            "rationale": "La combinación de antecedentes y NNA involucrados justifica exclusión.",
            "confidence": 0.82,
            "supporting_source_ids": ["ley_12569_art7"],
            "risk_level": "alto",
            "risk_indicators": ["NNA involucrados", "Antecedentes"],
            "alerts": ["ALERTA — evaluar intervención del Ministerio Pupilar."],
            "normative_basis": [
                {"ley": "Ley 12.569", "articulo": "7", "motivo": "Habilita medidas urgentes."}
            ],
            "procedural_basis": [
                {"ley": "Ley 12.569", "articulo": "11", "motivo": "Control judicial posterior."}
            ],
            "low_confidence_reason": "",
            "draft_text": "BORRADOR",
        }
        fake_client = SimpleNamespace(chat_json=lambda **_: fake_response)

        with patch("src.llm.measure_classifier.build_json_llm_client", return_value=(fake_cfg, fake_client)):
            classification = MeasureClassifierService().classify(
                extracted=extracted,
                retrieval=retrieval,
                provider="openai",
            )

        self.assertEqual(classification.selected_template_id, "medida_exclusion")
        self.assertEqual(classification.risk_level, "alto")
        self.assertEqual(classification.alerts, ["ALERTA — evaluar intervención del Ministerio Pupilar."])
        self.assertEqual(classification.normative_basis[0].articulo, "7")
        self.assertEqual(classification.procedural_basis[0].articulo, "11")
        self.assertEqual(classification.draft_text, "BORRADOR")

    def test_generic_provider_passes_prompt_variant_to_prompt_builders(self) -> None:
        extracted = ExtractedCase(
            case_id="FV00002494-0043221/2026",
            anonymized_text="Texto anonimizado.",
            facts=["Hecho."],
            active_measures=["Medida perimetral"],
            risk_factors=["Violencia psicologica"],
            timeline=["Denuncia actual"],
        )
        retrieval = [
            RetrievalResult(
                query="retencion de menores",
                hits=[
                    Citation(
                        source_id="ley_12569_art7",
                        excerpt="Medidas urgentes de protección.",
                        title="Ley 12.569 art. 7",
                        law="Ley 12.569",
                        article="7",
                        source_type="normativa",
                        relevance_score=0.9,
                    )
                ],
            )
        ]
        fake_cfg = SimpleNamespace(provider="openai", model="gpt-5-mini")
        fake_response = {
            "selected_template_id": "medida_perimetro",
            "selected_template_name": "Medida perimetral (radio)",
            "rationale": "Rationale",
            "confidence": 0.5,
            "supporting_source_ids": ["ley_12569_art7"],
            "risk_level": "medio",
            "risk_indicators": [],
            "alerts": [],
            "normative_basis": [],
            "procedural_basis": [],
            "draft_text": "",
        }
        fake_client = SimpleNamespace(chat_json=lambda **_: fake_response)

        with patch("src.llm.measure_classifier.build_json_llm_client", return_value=(fake_cfg, fake_client)):
            with patch("src.llm.measure_classifier.build_generic_measure_classification_system_prompt") as mock_system:
                with patch("src.llm.measure_classifier.build_generic_measure_classification_user_prompt") as mock_user:
                    mock_system.return_value = "SYSTEM"
                    mock_user.return_value = "USER"
                    MeasureClassifierService().classify(
                        extracted=extracted,
                        retrieval=retrieval,
                        provider="openai",
                        prompt_variant="balanced_measures",
                    )

        self.assertEqual(mock_system.call_args.kwargs["prompt_variant"], "balanced_measures")
        self.assertEqual(mock_user.call_args.kwargs["prompt_variant"], "balanced_measures")


if __name__ == "__main__":
    unittest.main()
