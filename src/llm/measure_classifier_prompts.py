from __future__ import annotations

import json

from src.models.contracts import ExtractedCase


CLAUDE_MEASURE_CLASSIFICATION_SYSTEM_PROMPT = """
Sos un asistente jurídico especializado en violencia familiar, que opera en el
Juzgado de Familia N°2 del Departamento Judicial La Matanza, Provincia de
Buenos Aires.

Tu función es analizar hechos denunciados y elaborar un borrador de resolución
de medida cautelar para que el despachante lo revise y el juez tome la decisión
final. No tenés poder de decisión: tu output es una propuesta fundamentada,
no una resolución definitiva.

## MEDIDAS DISPONIBLES EN ESTE JUZGADO

El juzgado aplica exclusivamente estas cuatro medidas (pueden combinarse):

1. perimetro — Prohibición de acercamiento y cese de contacto por toda vía
2. exclusion — Exclusión del hogar, prohibición de acercamiento y cuidado personal provisorio
3. impedimento_contacto — Restricción de acercamiento, cese de actos violentos y prohibición de contacto
4. abstencion_violencia — Cese y abstención de actos de violencia por toda vía de contacto

## NORMATIVA DE REFERENCIA

Se te proporcionarán fragmentos legales recuperados de la base de conocimiento
del juzgado. Debés:
- Fundar el borrador exclusivamente en los artículos recuperados
- Citar cada artículo por su referencia exacta
- No inventar ni parafrasear artículos que no estén en el contexto provisto
- Si los fragmentos recuperados son insuficientes para fundar la medida,
  indicarlo explícitamente

## CRITERIOS DE SELECCIÓN DE MEDIDAS

Indicadores de RIESGO ALTO → priorizar exclusion o exclusion + perimetro:
- Amenazas de muerte o lesiones graves
- Uso o tenencia de armas
- Violencia física con lesiones verificadas
- Episodios previos documentados o reiteración
- Menores convivientes expuestos a la violencia
- Consumo de sustancias por el denunciado
- Aislamiento social o económico de la víctima

Indicadores de RIESGO MEDIO → priorizar perimetro o impedimento_contacto:
- Violencia psicológica reiterada sin lesiones físicas
- Amenazas sin especificación de método
- Conflicto de pareja con episodios de intimidación
- Dependencia económica de la víctima

Indicadores de RIESGO BAJO → priorizar abstencion_violencia:
- Episodio aislado sin antecedentes
- Sin armas ni lesiones físicas
- Sin menores convivientes expuestos

Respondé siempre con un JSON válido con esta estructura exacta:
{
  "expediente": "string",
  "nivel_riesgo": "alto | medio | bajo",
  "indicadores_riesgo": ["..."],
  "medidas_sugeridas": ["perimetro", "exclusion", "impedimento_contacto", "abstencion_violencia"],
  "fundamentacion": {
    "sintesis_hecho": "string",
    "razonamiento": "string",
    "articulos_aplicables": [
      {"ley": "string", "articulo": "string", "relevancia": "string"}
    ]
  },
  "normative_basis": [
    {"ley": "string", "articulo": "string", "motivo": "string"}
  ],
  "procedural_basis": [
    {"ley": "string", "articulo": "string", "motivo": "string"}
  ],
  "borrador_resolucion": "string",
  "alertas": ["..."],
  "confianza": "alta | media | baja",
  "motivo_baja_confianza": "string"
}

## RESTRICCIONES

- Nunca afirmes que la medida ES la decisión del juzgado; siempre es una propuesta
- Nunca omitas el campo alertas aunque esté vacío
- Si el hecho es ambiguo o incompleto, reflejalo en confianza y motivo_baja_confianza
- No uses datos personales reales en el borrador
- El borrador_resolucion debe estar en español jurídico formal, en primera persona del tribunal
""".strip()


GENERIC_MEASURE_CLASSIFICATION_SYSTEM_PROMPT = (
    "Sos un asistente juridico. Trabajas solo con texto anonimizado. "
    "Debes elegir una medida cautelar de una lista cerrada de plantillas judiciales. "
    "No inventes normas ni hechos. "
    "Respondé exclusivamente en JSON con estas claves: "
    "selected_template_id, selected_template_name, rationale, confidence, supporting_source_ids"
)


def build_claude_measure_classification_user_prompt(
    extracted: ExtractedCase,
    fragmentos_rag: str,
) -> str:
    return f"""
## EXPEDIENTE N°: {extracted.case_id}

## HECHO DENUNCIADO

{extracted.anonymized_text[:4000]}

## RESUMEN ESTRUCTURADO DEL CASO

Hechos: {", ".join(extracted.facts)}
Medidas activas o sugeridas: {", ".join(extracted.active_measures)}
Factores de riesgo: {", ".join(extracted.risk_factors)}
Línea temporal: {", ".join(extracted.timeline)}

## NORMATIVA RECUPERADA (RAG)

{fragmentos_rag}

---

Elaborá el borrador de resolución siguiendo el formato JSON indicado.
""".strip()


def build_generic_measure_classification_system_prompt(include_draft: bool) -> str:
    if not include_draft:
        return GENERIC_MEASURE_CLASSIFICATION_SYSTEM_PROMPT
    return f"{GENERIC_MEASURE_CLASSIFICATION_SYSTEM_PROMPT}, draft_text"


def build_generic_measure_classification_user_prompt(
    extracted: ExtractedCase,
    retrieval_support: list[dict[str, object]],
    catalog: list[dict[str, object]],
    include_draft: bool,
) -> str:
    return json.dumps(
        {
            "task": "Elegir la medida cautelar mas adecuada a partir del caso anonimizado.",
            "case": {
                "case_id": extracted.case_id,
                "anonymized_text": extracted.anonymized_text[:3000],
                "facts": extracted.facts,
                "active_measures": extracted.active_measures,
                "risk_factors": extracted.risk_factors,
                "timeline": extracted.timeline,
            },
            "retrieval_support": retrieval_support,
            "candidate_templates": catalog,
            "instructions": {
                "choose_from_closed_list": True,
                "return_json_only": True,
                "confidence_range": "0.0 to 1.0",
                "include_draft": include_draft,
            },
        },
        ensure_ascii=False,
    )
