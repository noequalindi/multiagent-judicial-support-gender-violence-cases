# Templates Workflow (Spanish Judicial Measures)

## Current templates loaded

The system now includes three templates from your court workflow:

1. `medida_perimetro`
2. `medida_exclusion`
3. `medida_impedimento_contacto`

Defined in:

- `src/agents/template_agent.py`

## How the pipeline uses templates

1. `ExtractionAgent` identifies facts, active measures, and risk factors.
2. `DraftingAgent` calls `TemplateAgent.select(...)`.
3. The output includes:
   - `selected_template_id`
   - `selected_template_name`
   - `template_text`
   - `fields_to_fill`

This allows the clerk to open the selected template and complete only case-specific sensitive fields.

## How to add a new template

1. Open `src/agents/template_agent.py`.
2. Add a new `JudicialTemplate(...)` entry in `_catalog`.
3. Include:
   - a unique `template_id`
   - the legal measure list in `measures`
   - required placeholders in `fields_to_fill`
   - canonical text in `text` using `{{placeholder_name}}`
4. Update `select(...)` rules so the new template can be picked automatically.

## Placeholder convention

Use double braces:

- `{{expediente_numero}}`
- `{{nombre_denunciante}}`
- `{{nombre_denunciado}}`

This keeps templates deterministic and easy to populate by downstream code/UI.
