from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

from src.agents.ingestion_agent import IngestionAgent
from src.orchestration.orchestrator import JudicialOrchestrator
import subprocess


TEMPLATE_FILE_MAP = {
    "medida_perimetro": "medida_perimetro.pdf",
    "medida_exclusion": "medida_exclusion.pdf",
    "medida_impedimento_contacto": "medida_impedimento_contacto.pdf",
    "medida_abstencion_violencia": "medida_abstencion_violencia.txt",
}


def load_template_text(template_id: str, measures_dir: Path) -> str:
    filename = TEMPLATE_FILE_MAP.get(template_id)
    if not filename:
        raise ValueError(f"No template file configured for {template_id}")
    fpath = measures_dir / filename
    if not fpath.exists():
        raise FileNotFoundError(f"Template file not found: {fpath}")

    if fpath.suffix.lower() == ".txt":
        return fpath.read_text(encoding="utf-8", errors="ignore")

    cmd = ["pdftotext", str(fpath), "-"]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return proc.stdout


def extract_case_fields(text: str) -> dict[str, str]:
    src = text or ""

    def _find(pattern: str) -> str:
        m = re.search(pattern, src, flags=re.IGNORECASE)
        return m.group(1).strip() if m else ""

    expediente = _find(r"N['*]?\s*de\s*VG[:\s]*([A-Z0-9\-/]+)")
    if not expediente:
        expediente = _find(r"EXPEDIENTE\s*N[°º]?\s*[:\-]?\s*([A-Z0-9\-/\.]+)")

    fecha_denuncia = _find(r"Fecha y hora de denuncia[:\s]*([0-9/\-:\s]+)")
    if not fecha_denuncia:
        fecha_denuncia = datetime.now().strftime("%d/%m/%Y")

    return {
        "caratula_causa": "[COMPLETAR_CARATULA]",
        "expediente_numero": expediente or "[COMPLETAR_EXPEDIENTE]",
        "registro_numero": "[COMPLETAR_REGISTRO]",
        "fecha_resolucion": fecha_denuncia,
        "nombre_denunciante": "[COMPLETAR_DENUNCIANTE]",
        "nombre_denunciado": "[COMPLETAR_DENUNCIADO]",
        "domicilio_protegido": "[COMPLETAR_DOMICILIO]",
        "domicilio_hogar_familiar": "[COMPLETAR_DOMICILIO_HOGAR]",
        "nombres_nna": "[COMPLETAR_NNA]",
        "radio_metros": "300",
        "medios_contacto_alcanzados": "telefonica, mensajeria, redes sociales y correo electronico",
    }


def fill_template_text(template_text: str, fields: dict[str, str]) -> str:
    out = template_text
    replacements = {
        "EXPEDIENTE N° Identificación de la Causa": f"EXPEDIENTE N° {fields['expediente_numero']}",
        "EXPEDIENTE N°Identificación de la Causa": f"EXPEDIENTE N° {fields['expediente_numero']}",
        "REGISTRO N°:.......": f"REGISTRO N°: {fields['registro_numero']}",
        "Fecha del Sistema (de Mes de Año)": fields["fecha_resolucion"],
        "Fecha del Sistema (Día de Mes de Año)": fields["fecha_resolucion"],
        "APELLIDO y NOMBRE del Denunciado": fields["nombre_denunciado"],
        "NOMBRE y APELLIDO del Denunciado": fields["nombre_denunciado"],
        "APELLIDO y NOMBRE del Denunciante": fields["nombre_denunciante"],
        "NOMBRE y APELLIDO del Denunciante": fields["nombre_denunciante"],
        "Domicilio del Actor (Tipo y Nro)": fields["domicilio_protegido"],
        "Domicilio de la Parte (Tipo y Nro)": fields["domicilio_hogar_familiar"],
    }
    for src, dst in replacements.items():
        out = out.replace(src, dst)

    for k, v in fields.items():
        out = out.replace(f"{{{{{k}}}}}", v)

    return out


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def text_to_simple_pdf(text: str, pdf_path: Path) -> None:
    """
    Minimal dependency-free PDF writer for monospaced text.
    Generates A4 pages and wraps lines by character count.
    """
    max_chars = 95
    max_lines = 60
    wrapped_lines: list[str] = []
    for raw in (text or "").splitlines():
        line = raw.rstrip()
        if not line:
            wrapped_lines.append("")
            continue
        while len(line) > max_chars:
            wrapped_lines.append(line[:max_chars])
            line = line[max_chars:]
        wrapped_lines.append(line)

    if not wrapped_lines:
        wrapped_lines = [" "]

    pages = [wrapped_lines[i : i + max_lines] for i in range(0, len(wrapped_lines), max_lines)]

    objects: list[bytes] = []
    page_obj_ids: list[int] = []
    content_obj_ids: list[int] = []

    # Object numbering plan:
    # 1: Catalog, 2: Pages root, 3: Font
    next_id = 4
    for _ in pages:
        page_obj_ids.append(next_id)
        next_id += 1
        content_obj_ids.append(next_id)
        next_id += 1

    for lines in pages:
        content_lines = ["BT", "/F1 10 Tf", "40 800 Td", "12 TL"]
        for ln in lines:
            content_lines.append(f"({_pdf_escape(ln)}) Tj")
            content_lines.append("T*")
        content_lines.append("ET")
        stream = "\n".join(content_lines).encode("latin-1", errors="replace")
        obj = b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
        objects.append(obj)

    # Build page objects referencing corresponding content streams.
    # Since content objects were appended first, map ids carefully.
    content_by_page = dict(zip(page_obj_ids, content_obj_ids))

    # Rebuild objects list in real object-id order.
    obj_map: dict[int, bytes] = {}
    obj_map[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    kids = " ".join(f"{pid} 0 R" for pid in page_obj_ids).encode()
    obj_map[2] = b"<< /Type /Pages /Kids [ " + kids + b" ] /Count " + str(len(page_obj_ids)).encode() + b" >>"
    obj_map[3] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    for idx, page_id in enumerate(page_obj_ids):
        content_id = content_obj_ids[idx]
        obj_map[page_id] = (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            b"/Resources << /Font << /F1 3 0 R >> >> "
            b"/Contents "
            + str(content_id).encode()
            + b" 0 R >>"
        )
        obj_map[content_id] = objects[idx]

    max_obj = max(obj_map.keys())
    parts = [b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"]
    xref_positions = [0] * (max_obj + 1)
    offset = len(parts[0])

    for obj_id in range(1, max_obj + 1):
        payload = obj_map[obj_id]
        header = f"{obj_id} 0 obj\n".encode()
        footer = b"\nendobj\n"
        xref_positions[obj_id] = offset
        parts.extend([header, payload, footer])
        offset += len(header) + len(payload) + len(footer)

    xref_start = offset
    xref = [f"xref\n0 {max_obj + 1}\n".encode(), b"0000000000 65535 f \n"]
    for obj_id in range(1, max_obj + 1):
        xref.append(f"{xref_positions[obj_id]:010d} 00000 n \n".encode())

    trailer = (
        b"trailer\n<< /Size "
        + str(max_obj + 1).encode()
        + b" /Root 1 0 R >>\nstartxref\n"
        + str(xref_start).encode()
        + b"\n%%EOF\n"
    )
    parts.extend(xref)
    parts.append(trailer)

    pdf_path.write_bytes(b"".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--denuncia-pdf", required=True)
    parser.add_argument("--measures-dir", default="data/measures")
    parser.add_argument("--out-dir", default="data/generated_measures")
    parser.add_argument("--ocr-backend", default="tesseract", choices=["tesseract", "ollama"])
    parser.add_argument("--tesseract-lang", default="spa")
    parser.add_argument("--ollama-model", default="llama3.2-vision")
    parser.add_argument("--no-pdf", action="store_true", help="Solo genera TXT.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ingestion = IngestionAgent()
    ingest = ingestion.invoke_pdf(
        pdf_path=args.denuncia_pdf,
        ocr_backend=args.ocr_backend,
        tesseract_lang=args.tesseract_lang,
        ollama_model=args.ollama_model,
    )

    orchestrator = JudicialOrchestrator()
    result = orchestrator.run_from_ingest_payload(ingest)

    template_id = result.draft.selected_template_id
    if not template_id:
        raise RuntimeError("No template selected by the pipeline.")

    template_text = load_template_text(template_id, Path(args.measures_dir).resolve())
    fields = extract_case_fields(ingest["anonymized_text"])
    final_text = fill_template_text(template_text, fields)

    stem = Path(args.denuncia_pdf).stem
    txt_out = out_dir / f"{stem}_{template_id}.txt"
    txt_out.write_text(final_text, encoding="utf-8")
    print(f"Generated TXT: {txt_out}")

    if not args.no_pdf:
        pdf_out = out_dir / f"{stem}_{template_id}.pdf"
        text_to_simple_pdf(final_text, pdf_out)
        print(f"Generated PDF: {pdf_out}")

    print(f"Selected template: {template_id}")
    print("Fields used:")
    for k, v in fields.items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
