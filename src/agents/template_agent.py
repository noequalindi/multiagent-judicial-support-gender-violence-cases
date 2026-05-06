from __future__ import annotations

from dataclasses import dataclass

from src.models.contracts import ExtractedCase


@dataclass(frozen=True)
class JudicialTemplate:
    template_id: str
    name: str
    measures: list[str]
    fields_to_fill: list[str]
    text: str


class TemplateAgent:
    """Selects the most suitable judicial template and exposes fillable fields."""

    _catalog: dict[str, JudicialTemplate] = {
        "medida_perimetro": JudicialTemplate(
            template_id="medida_perimetro",
            name="Medida perimetral (radio)",
            measures=["Medida perimetral"],
            fields_to_fill=[
                "caratula_causa",
                "expediente_numero",
                "registro_numero",
                "fecha_resolucion",
                "nombre_denunciante",
                "nombre_denunciado",
                "domicilio_protegido",
                "radio_metros",
            ],
            text=(
                "RESUELVO: 1) Disponer medida perimetral para que {{nombre_denunciado}} no se aproxime a "
                "{{nombre_denunciante}} hasta un radio de {{radio_metros}} metros "
                "del domicilio {{domicilio_protegido}}, lugar de trabajo, estudio y esparcimiento. "
                "2) Vigencia provisoria por 90 dias, prorrogable. "
                "3) Notifiquese a comisaria interviniente y registrese."
            ),
        ),
        "medida_exclusion": JudicialTemplate(
            template_id="medida_exclusion",
            name="Exclusion del hogar + medida perimetral",
            measures=[
                "Exclusion del hogar del agresor",
                "Medida perimetral",
                "Cuidado personal provisorio de NNA",
            ],
            fields_to_fill=[
                "caratula_causa",
                "expediente_numero",
                "registro_numero",
                "fecha_resolucion",
                "nombre_denunciante",
                "nombre_denunciado",
                "domicilio_hogar_familiar",
                "nombres_nna",
                "radio_metros",
            ],
            text=(
                "RESUELVO: 1) Exclusion de {{nombre_denunciado}} del hogar familiar en "
                "{{domicilio_hogar_familiar}} y atribucion provisoria de la vivienda a "
                "{{nombre_denunciante}}. 2) Disponer medida perimetral en un radio de "
                "{{radio_metros}} metros respecto de {{nombre_denunciante}} y {{nombres_nna}}. "
                "3) Cuidado personal provisorio de NNA por el plazo de vigencia de la medida. "
                "4) Librense oficios de cumplimiento urgente."
            ),
        ),
        "medida_impedimento_contacto": JudicialTemplate(
            template_id="medida_impedimento_contacto",
            name="Impedimento de contacto por todo medio",
            measures=["Impedimento de contacto por cualquier via"],
            fields_to_fill=[
                "caratula_causa",
                "expediente_numero",
                "registro_numero",
                "fecha_resolucion",
                "nombre_denunciante",
                "nombre_denunciado",
                "medios_contacto_alcanzados",
            ],
            text=(
                "RESUELVO: 1) Impedimento de contacto de {{nombre_denunciado}} con "
                "{{nombre_denunciante}} por todo medio, incluyendo {{medios_contacto_alcanzados}}. "
                "2) Cese de hostigamiento y abstencion de actos de violencia fisica, psicologica "
                "o emocional. 3) Vigencia provisoria por 90 dias, prorrogable."
            ),
        ),
        "medida_abstencion_violencia": JudicialTemplate(
            template_id="medida_abstencion_violencia",
            name="Abstencion de actos de violencia",
            measures=["Abstencion de actos de violencia por toda via"],
            fields_to_fill=[
                "caratula_causa",
                "expediente_numero",
                "registro_numero",
                "fecha_resolucion",
                "nombre_denunciante",
                "nombre_denunciado",
            ],
            text=(
                "RESUELVO: 1) Hacer saber a {{nombre_denunciado}} que debera cesar y/o "
                "abstenerse de todo acto violento fisico, psicologico o emocional respecto de "
                "{{nombre_denunciante}}, por toda via de contacto posible. "
                "2) Vigencia provisoria por 90 dias, prorrogable."
            ),
        ),
    }

    def select(self, extracted: ExtractedCase) -> JudicialTemplate:
        measures_text = " ".join(extracted.active_measures).lower()
        risk_text = " ".join(extracted.risk_factors).lower()
        facts_text = " ".join(extracted.facts).lower()

        if "exclusion" in measures_text or "convivencia" in risk_text:
            return self._catalog["medida_exclusion"]

        if "prohibicion de acercamiento" in measures_text or "medida perimetral" in measures_text or "amenaza" in facts_text:
            return self._catalog["medida_perimetro"]

        if "sin factores automaticos concluyentes" in risk_text:
            return self._catalog["medida_abstencion_violencia"]

        return self._catalog["medida_impedimento_contacto"]

    @classmethod
    def available_templates(cls) -> list[str]:
        return list(cls._catalog.keys())

    @classmethod
    def get_catalog(cls) -> dict[str, JudicialTemplate]:
        return cls._catalog.copy()
