from __future__ import annotations

import operator
from pathlib import Path
from typing import Annotated, Any, TypedDict

try:
    from langgraph.constants import Send
    from langgraph.graph import END, START, StateGraph
except ImportError as exc:
    Send = None
    END = START = StateGraph = None
    _LANGGRAPH_IMPORT_ERROR = exc
else:
    _LANGGRAPH_IMPORT_ERROR = None

from src.orchestration.batch_runner import BatchConfig, list_pdfs, process_one_pdf_safe, write_manifest


class BatchGraphState(TypedDict, total=False):
    cfg: BatchConfig
    pdf_paths: list[str]
    results: Annotated[list[dict[str, Any]], operator.add]
    manifest_path: str
    processed: int
    failed: int


class PDFTaskState(TypedDict):
    cfg: BatchConfig
    pdf_path: str


def discover_pdfs(state: BatchGraphState) -> dict[str, Any]:
    cfg = state["cfg"]
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    pdf_paths = [str(path) for path in list_pdfs(cfg.input_dir, cfg.glob_pattern)]
    return {"pdf_paths": pdf_paths}


def route_pdfs(state: BatchGraphState) -> list[Send] | str:
    pdf_paths = state.get("pdf_paths", [])
    if not pdf_paths:
        return "finalize_batch"
    return [Send("process_pdf", {"cfg": state["cfg"], "pdf_path": pdf_path}) for pdf_path in pdf_paths]


def process_pdf(state: PDFTaskState) -> dict[str, Any]:
    row = process_one_pdf_safe(Path(state["pdf_path"]), state["cfg"])
    return {"results": [row]}


def finalize_batch(state: BatchGraphState) -> dict[str, Any]:
    cfg = state["cfg"]
    manifest_path, summary = write_manifest(cfg.out_dir, state.get("results", []))
    return {
        "manifest_path": str(manifest_path),
        "processed": summary["processed"],
        "failed": summary["failed"],
    }


def build_batch_graph():
    if _LANGGRAPH_IMPORT_ERROR is not None:
        raise RuntimeError(
            "LangGraph is not installed. Install project dependencies again to enable the batch graph."
        ) from _LANGGRAPH_IMPORT_ERROR

    graph = StateGraph(BatchGraphState)
    graph.add_node("discover_pdfs", discover_pdfs)
    graph.add_node("process_pdf", process_pdf)
    graph.add_node("finalize_batch", finalize_batch)
    graph.add_edge(START, "discover_pdfs")
    graph.add_conditional_edges("discover_pdfs", route_pdfs)
    graph.add_edge("process_pdf", "finalize_batch")
    graph.add_edge("finalize_batch", END)
    return graph.compile()


def run_batch_graph(cfg: BatchConfig) -> BatchGraphState:
    graph = build_batch_graph()
    return graph.invoke({"cfg": cfg, "results": []}, config={"max_concurrency": cfg.workers})
