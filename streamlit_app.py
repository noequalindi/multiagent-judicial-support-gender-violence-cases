from __future__ import annotations

import os

import httpx
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


def _get_json(path: str) -> dict:
    with httpx.Client(timeout=60.0) as client:
        response = client.get(f"{API_BASE_URL}{path}")
        response.raise_for_status()
        return response.json()


def _post_json(path: str, payload: dict) -> dict:
    with httpx.Client(timeout=120.0) as client:
        response = client.post(f"{API_BASE_URL}{path}", json=payload)
        if response.is_error:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            raise RuntimeError(f"{response.status_code}: {detail}")
        return response.json()


def _post_file(path: str, file_name: str, file_bytes: bytes, data: dict[str, str]) -> dict:
    with httpx.Client(timeout=300.0) as client:
        response = client.post(
            f"{API_BASE_URL}{path}",
            data=data,
            files={"file": (file_name, file_bytes, "application/pdf")},
        )
        if response.is_error:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            raise RuntimeError(f"{response.status_code}: {detail}")
        return response.json()


st.set_page_config(page_title="Judicial Assistant Demo", layout="wide")
st.title("Asistente judicial de violencia de genero")
st.caption("UI simple para probar OCR, analisis del caso y generacion de borradores.")

try:
    llm_runtime_options = _get_json("/options/llm")
    ocr_runtime_options = _get_json("/options/ocr")
except Exception:
    llm_runtime_options = {
        "providers": [
            {
                "provider": "openai",
                "label": "OpenAI",
                "configured": False,
                "default_model": "",
                "models": [],
            },
            {
                "provider": "deepseek",
                "label": "DeepSeek",
                "configured": False,
                "default_model": "",
                "models": [],
            },
            {
                "provider": "huggingface",
                "label": "Hugging Face Router",
                "configured": False,
                "default_model": "",
                "models": [],
            },
        ]
    }
    ocr_runtime_options = {
        "backends": [
            {"value": "tesseract", "label": "Tesseract"},
            {"value": "ollama", "label": "Ollama"},
        ],
        "tesseract_lang": "spa",
        "ollama_models": ["llama3.2-vision", "qwen2.5-vl:7b"],
        "default_backend": "tesseract",
        "default_ollama_model": "llama3.2-vision",
    }

provider_specs = {item["provider"]: item for item in llm_runtime_options["providers"]}
ocr_backend_values = [item["value"] for item in ocr_runtime_options["backends"]]

with st.sidebar:
    st.subheader("Configuracion")
    api_url = st.text_input("API base URL", value=API_BASE_URL)
    default_ocr_backend = ocr_runtime_options.get("default_backend", "tesseract")
    default_ocr_index = ocr_backend_values.index(default_ocr_backend) if default_ocr_backend in ocr_backend_values else 0
    ocr_backend = st.selectbox("OCR backend", ocr_backend_values, index=default_ocr_index)
    tesseract_lang = st.text_input("Tesseract lang", value=str(ocr_runtime_options.get("tesseract_lang", "spa")))

    ollama_models = list(ocr_runtime_options.get("ollama_models", []))
    default_ollama_model = str(ocr_runtime_options.get("default_ollama_model", "")) or (ollama_models[0] if ollama_models else "")
    if ollama_models:
        ollama_index = ollama_models.index(default_ollama_model) if default_ollama_model in ollama_models else 0
        ollama_model = st.selectbox("Ollama model", ollama_models, index=ollama_index)
    else:
        ollama_model = st.text_input("Ollama model", value=default_ollama_model)

    provider_values = [item["provider"] for item in llm_runtime_options["providers"]]
    llm_provider = st.selectbox("LLM provider", provider_values, format_func=lambda x: provider_specs[x]["label"])
    selected_provider = provider_specs[llm_provider]
    model_options = list(selected_provider.get("models", []))
    default_model = str(selected_provider.get("default_model", ""))
    if model_options:
        model_index = model_options.index(default_model) if default_model in model_options else 0
        llm_model = st.selectbox("LLM model", model_options, index=model_index)
    else:
        llm_model = st.text_input("LLM model", value=default_model)
    st.caption(
        f"Proveedor configurado: {'si' if selected_provider.get('configured') else 'no'}"
    )
    st.session_state["api_url"] = api_url

tab_text, tab_pdf, tab_classify, tab_draft = st.tabs(["Texto", "PDF", "Clasificacion", "Borrador"])

with tab_text:
    st.subheader("Procesar texto")
    text = st.text_area("Denuncia anonimizada o texto de prueba", height=220)
    if st.button("Procesar texto", use_container_width=True):
        if not text.strip():
            st.warning("Ingresá un texto.")
        else:
            API_BASE_URL = st.session_state["api_url"]
            try:
                response = _post_json("/process-text", {"text": text})
                st.json(response["result"])
            except Exception as exc:
                st.error(f"Error al procesar texto: {exc}")

with tab_pdf:
    st.subheader("Procesar PDF")
    uploaded_pdf = st.file_uploader("Subir denuncia PDF", type=["pdf"], key="process_pdf")
    if st.button("Procesar PDF", use_container_width=True):
        if not uploaded_pdf:
            st.warning("Subí un PDF primero.")
        else:
            API_BASE_URL = st.session_state["api_url"]
            try:
                response = _post_file(
                    "/process-pdf",
                    uploaded_pdf.name,
                    uploaded_pdf.getvalue(),
                    {
                        "ocr_backend": ocr_backend,
                        "tesseract_lang": tesseract_lang,
                        "ollama_model": ollama_model,
                    },
                )
                st.json(response["result"])
            except Exception as exc:
                st.error(f"Error al procesar PDF: {exc}")

with tab_draft:
    st.subheader("Generar borrador")
    draft_pdf = st.file_uploader("Subir denuncia PDF", type=["pdf"], key="draft_pdf")
    if st.button("Generar borrador", use_container_width=True):
        if not draft_pdf:
            st.warning("Subí un PDF primero.")
        else:
            API_BASE_URL = st.session_state["api_url"]
            try:
                response = _post_file(
                    "/generate-draft",
                    draft_pdf.name,
                    draft_pdf.getvalue(),
                    {
                        "ocr_backend": ocr_backend,
                        "tesseract_lang": tesseract_lang,
                        "ollama_model": ollama_model,
                        "measures_dir": "data/measures",
                    },
                )
                st.write(f"Template seleccionado: `{response['template_id']}`")
                st.json(response["fields"])
                st.text_area("Borrador TXT", value=response["txt_content"], height=360)
            except Exception as exc:
                st.error(f"Error al generar borrador: {exc}")

with tab_classify:
    st.subheader("Clasificar medida con LLM")
    classify_pdf = st.file_uploader("Subir denuncia PDF", type=["pdf"], key="classify_pdf")
    include_draft = st.checkbox("Pedir draft al modelo", value=False)
    if st.button("Clasificar medida", use_container_width=True):
        if not classify_pdf:
            st.warning("Subí un PDF primero.")
        else:
            API_BASE_URL = st.session_state["api_url"]
            try:
                response = _post_file(
                    "/classify-measure-pdf",
                    classify_pdf.name,
                    classify_pdf.getvalue(),
                    {
                        "provider": llm_provider,
                        "model": llm_model,
                        "include_draft": str(include_draft).lower(),
                        "ocr_backend": ocr_backend,
                        "tesseract_lang": tesseract_lang,
                        "ollama_model": ollama_model,
                    },
                )
                st.json(response)
            except Exception as exc:
                st.error(f"Error al clasificar medida: {exc}")
