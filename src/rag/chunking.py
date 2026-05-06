from __future__ import annotations

import re

from src.rag.tokenization import (
    default_chunk_max_tokens,
    default_chunk_overlap_tokens,
    estimate_tokens,
)


def _normalize_whitespace(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def _split_sentences(paragraph: str) -> list[str]:
    parts = re.split(r"(?<=[.!?;:])\s+|\n", paragraph.strip())
    return [part.strip() for part in parts if part and part.strip()]


def _split_words(text: str) -> list[str]:
    return re.findall(r"\S+", text or "", flags=re.UNICODE)


def _break_long_unit(
    unit: str,
    max_tokens: int,
    embedding_model: str | None = None,
    tokenizer_name: str | None = None,
) -> list[str]:
    words = _split_words(unit)
    if not words:
        return []
    pieces: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join([*current, word]).strip()
        if current and estimate_tokens(
            candidate,
            embedding_model=embedding_model,
            tokenizer_name=tokenizer_name,
        ) > max_tokens:
            pieces.append(" ".join(current).strip())
            current = [word]
        else:
            current.append(word)
    if current:
        pieces.append(" ".join(current).strip())
    return pieces


def _split_units(
    text: str,
    max_tokens: int,
    embedding_model: str | None = None,
    tokenizer_name: str | None = None,
) -> list[str]:
    units: list[str] = []
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p and p.strip()]
    for paragraph in paragraphs:
        if estimate_tokens(
            paragraph,
            embedding_model=embedding_model,
            tokenizer_name=tokenizer_name,
        ) <= max_tokens:
            units.append(paragraph)
            continue
        for sentence in _split_sentences(paragraph):
            if estimate_tokens(
                sentence,
                embedding_model=embedding_model,
                tokenizer_name=tokenizer_name,
            ) <= max_tokens:
                units.append(sentence)
            else:
                units.extend(
                    _break_long_unit(
                        sentence,
                        max_tokens,
                        embedding_model=embedding_model,
                        tokenizer_name=tokenizer_name,
                    )
                )
    return units


def _tail_overlap_units(
    units: list[str],
    target_tokens: int,
    embedding_model: str | None = None,
    tokenizer_name: str | None = None,
) -> list[str]:
    if target_tokens <= 0 or not units:
        return []
    tail: list[str] = []
    total = 0
    for unit in reversed(units):
        tail.insert(0, unit)
        total += estimate_tokens(
            unit,
            embedding_model=embedding_model,
            tokenizer_name=tokenizer_name,
        )
        if total >= target_tokens:
            break
    return tail


def chunk_text(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 200,
    *,
    max_tokens: int | None = None,
    overlap_tokens: int | None = None,
    embedding_model: str | None = None,
    tokenizer_name: str | None = None,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    clean = _normalize_whitespace(text)
    if not clean:
        return []

    effective_max_tokens = max_tokens or default_chunk_max_tokens(chunk_size)
    effective_overlap_tokens = overlap_tokens if overlap_tokens is not None else default_chunk_overlap_tokens(overlap)
    units = _split_units(
        clean,
        effective_max_tokens,
        embedding_model=embedding_model,
        tokenizer_name=tokenizer_name,
    )
    if not units:
        return []

    chunks: list[str] = []
    current_units: list[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = estimate_tokens(
            unit,
            embedding_model=embedding_model,
            tokenizer_name=tokenizer_name,
        )
        if current_units and current_tokens + unit_tokens > effective_max_tokens:
            chunk = "\n".join(current_units).strip()
            if chunk:
                chunks.append(chunk)
            current_units = _tail_overlap_units(
                current_units,
                effective_overlap_tokens,
                embedding_model=embedding_model,
                tokenizer_name=tokenizer_name,
            )
            current_tokens = sum(
                estimate_tokens(
                    item,
                    embedding_model=embedding_model,
                    tokenizer_name=tokenizer_name,
                )
                for item in current_units
            )
            if current_units and current_tokens + unit_tokens > effective_max_tokens:
                # If overlap leaves too little room, reset to prioritize the new unit.
                current_units = []
                current_tokens = 0
        current_units.append(unit)
        current_tokens += unit_tokens

    if current_units:
        chunk = "\n".join(current_units).strip()
        if chunk:
            chunks.append(chunk)
    return chunks
