from __future__ import annotations

import json

from src.models.contracts import ExtractedCase

SUPPORTED_PROMPT_VARIANTS = {
    "default",
    "strict_rag",
    "balanced_measures",
}


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
  "alertas": ["..."],
  "confianza": "alta | media | baja",
  "motivo_baja_confianza": "string"
}

## RESTRICCIONES

- Nunca afirmes que la medida ES la decisión del juzgado; siempre es una propuesta
- Nunca omitas el campo alertas aunque esté vacío
- Si el hecho es ambiguo o incompleto, reflejalo en confianza y motivo_baja_confianza
- No uses datos personales reales en el borrador
""".strip()


GENERIC_MEASURE_CLASSIFICATION_SYSTEM_PROMPT = (
    "Sos un asistente juridico. Trabajas solo con texto anonimizado. "
    "Debes elegir una medida cautelar de una lista cerrada de plantillas judiciales. "
    "No inventes normas ni hechos. "
    "Respondé exclusivamente en JSON con estas claves: "
    "selected_template_id, selected_template_name, rationale, confidence, supporting_source_ids, "
    "risk_level, risk_indicators, alerts, normative_basis, procedural_basis, low_confidence_reason, draft_text. "
    "risk_level debe ser uno de: alto, medio, bajo. "
    "normative_basis y procedural_basis deben ser listas de objetos con ley, articulo y motivo. "
    "supporting_source_ids debe contener solo ids recuperados presentes en retrieval_support."
)


def normalize_prompt_variant(raw: str | None) -> str:
    value = str(raw or "").strip().lower() or "default"
    if value not in SUPPORTED_PROMPT_VARIANTS:
        raise ValueError(
            f"Variant de prompt inválida: {raw}. Opciones: {', '.join(sorted(SUPPORTED_PROMPT_VARIANTS))}"
        )
    return value


def _claude_variant_block(prompt_variant: str) -> str:
    variant = normalize_prompt_variant(prompt_variant)
    if variant == "strict_rag":
        return """
## VARIANTE strict_rag

- No amplíes la fundamentación fuera de las fuentes recuperadas
- Si no hay respaldo suficiente para una medida intensa, señalalo explícitamente
- Priorizá precisión normativa sobre cobertura
""".strip()
    if variant == "balanced_measures":
        return """
## VARIANTE balanced_measures

- No sugieras exclusion por inercia
- Si el caso no presenta indicadores claros de riesgo alto, evaluá primero perimetro, impedimento_contacto o abstencion_violencia
- Si proponés más de una medida, distinguí mentalmente medida principal y medida complementaria
- Evitá sobrepredecir combinaciones exclusion + perimetro si el hecho no lo justifica
""".strip()
    return ""


def _generic_variant_suffix(prompt_variant: str) -> str:
    variant = normalize_prompt_variant(prompt_variant)
    if variant == "strict_rag":
        return (
            " Usa solo supporting_source_ids presentes en retrieval_support. "
            "Si la evidencia normativa es débil, reducí confidence y explicitá low_confidence_reason."
        )
    if variant == "balanced_measures":
        return (
            " Evitá sobreelegir medida_exclusion si el texto no muestra riesgo alto claro. "
            "Si hay más de una medida plausible, elegí una principal y reflejá las complementarias en suggested_measures."
        )
    return ""


def build_claude_measure_classification_user_prompt(
    extracted: ExtractedCase,
    fragmentos_rag: str,
    prompt_variant: str = "default",
) -> str:
    variant = normalize_prompt_variant(prompt_variant)
    variant_block = _claude_variant_block(variant)
    variant_section = f"{variant_block}\n\n---\n" if variant_block else ""
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

{variant_section}

Elaborá el borrador de resolución siguiendo el formato JSON indicado.
""".strip()


def build_generic_measure_classification_system_prompt(
    include_draft: bool,
    prompt_variant: str = "default",
) -> str:
    return GENERIC_MEASURE_CLASSIFICATION_SYSTEM_PROMPT + _generic_variant_suffix(prompt_variant)


def build_generic_measure_classification_user_prompt(
    extracted: ExtractedCase,
    retrieval_support: list[dict[str, object]],
    catalog: list[dict[str, object]],
    include_draft: bool,
    prompt_variant: str = "default",
) -> str:
    variant = normalize_prompt_variant(prompt_variant)
    return json.dumps(
        {
            "task": "Elegir la medida cautelar mas adecuada a partir del caso anonimizado.",
            "prompt_variant": variant,
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
                "include_draft": False,
            },
        },
        ensure_ascii=False,
    )
