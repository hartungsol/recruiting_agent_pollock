"""Utilities for turning agent output into short, user-facing speakable text.

Voice mode must never feed Piper:
- internal reasoning / OSCAR traces
- JSON blobs / code
- long summaries / job description text

This module enforces that separation.
"""

from __future__ import annotations

import json
import re
from typing import Any


_TAG_REASONING_RE = re.compile(r"<reasoning>.*?</reasoning>", re.IGNORECASE | re.DOTALL)
_TAG_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_CODE_FENCE_RE = re.compile(r"```", re.DOTALL)


def _looks_like_json(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False

    if t.startswith("{") or t.startswith("["):
        try:
            json.loads(t)
            return True
        except Exception:
            # Heuristic fallback: starts like JSON but invalid.
            return True

    # JSON-y inline blobs
    if '"' in t and ":" in t and ("{" in t or "[" in t):
        return True

    return False


def _split_sentences(text: str) -> list[str]:
    t = (text or "").strip()
    if not t:
        return []

    # Simple, robust sentence splitting for English-ish text.
    parts = re.split(r"(?<=[.!?])\s+", t)
    return [p.strip() for p in parts if p and p.strip()]


def _normalize_question(q: str) -> str:
    s = (q or "").strip()
    if not s:
        return s

    # If the question is embedded after a label, prefer the portion after the last colon.
    # Examples: "Next question: ...?", "First question: ...?"
    if ":" in s and "?" in s:
        tail = s.rsplit(":", 1)[-1].strip()
        if tail.endswith("?") and len(tail) >= 3:
            s = tail

    # Drop common boilerplate prefixes when they remain on the same line.
    s = re.sub(r"^(thanks\b.*?\.)\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^(welcome\b.*?\.)\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^(next\s+question\s*[:\-]?\s*)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^(first\s+question\s*[:\-]?\s*)", "", s, flags=re.IGNORECASE)

    # Speak one question only.
    if "?" in s:
        s = s.split("?", 1)[0].strip() + "?"

    return s.strip()


def _first_question(text: str) -> str | None:
    if not text:
        return None

    # Prefer explicit question lines.
    for line in [ln.strip() for ln in text.splitlines() if ln.strip()]:
        if line.endswith("?"):
            return _normalize_question(line)

    # Fall back to first sentence that looks like a question.
    for s in _split_sentences(text):
        if "?" in s:
            # Trim to the first question mark.
            q = s.split("?", 1)[0].strip() + "?"
            return _normalize_question(q)

    return None


def to_speakable_question(
    text: str,
    *,
    max_chars: int = 300,
    max_sentences: int = 2,
    speak_reasoning: bool = False,
) -> tuple[str | None, dict[str, Any]]:
    """Return (speakable_text_or_None, debug_info).

    Rules:
    - If <reasoning> exists, it is stripped.
    - If <answer> exists, prefer it.
    - If content is JSON or contains code fences, do not speak.
    - Enforce max chars / sentences.
    - Speak one question only (first detected question). If none, speak a short first 1–2 sentences.
    """

    debug: dict[str, Any] = {
        "input_chars": len(text or ""),
        "stripped_reasoning": False,
        "used_answer_tag": False,
        "skipped": False,
        "skip_reason": None,
        "truncated": False,
        "selected_question": False,
        "output_chars": 0,
        "sentences": 0,
    }

    raw = (text or "").strip()
    if not raw:
        debug.update({"skipped": True, "skip_reason": "empty"})
        return None, debug

    if not speak_reasoning:
        if _TAG_REASONING_RE.search(raw):
            debug["stripped_reasoning"] = True
            raw = _TAG_REASONING_RE.sub("", raw).strip()

    m = _TAG_ANSWER_RE.search(raw)
    if m:
        debug["used_answer_tag"] = True
        raw = (m.group(1) or "").strip()

    if not raw:
        debug.update({"skipped": True, "skip_reason": "empty_after_strip"})
        return None, debug

    if _CODE_FENCE_RE.search(raw):
        debug.update({"skipped": True, "skip_reason": "contained_code_fence"})
        return None, debug

    if _looks_like_json(raw):
        debug.update({"skipped": True, "skip_reason": "contained_json"})
        return None, debug

    # Remove common markup-ish tags other than answer/reasoning.
    raw = re.sub(r"</?\w+?>", "", raw).strip()

    q = _first_question(raw)
    if q:
        speak = q.strip()
        debug["selected_question"] = True
    else:
        # Speak the first 1–2 sentences as a short prompt.
        sentences = _split_sentences(raw)
        speak = " ".join(sentences[:max(1, max_sentences)]).strip() if sentences else raw

    # Enforce max sentences.
    speak_sentences = _split_sentences(speak)
    if len(speak_sentences) > max_sentences:
        speak = " ".join(speak_sentences[:max_sentences]).strip()
        debug["truncated"] = True

    # Enforce max chars.
    if len(speak) > max_chars:
        speak = speak[: max(0, max_chars - 1)].rstrip() + "…"
        debug["truncated"] = True

    speak = speak.strip()
    if not speak:
        debug.update({"skipped": True, "skip_reason": "empty_after_filter"})
        return None, debug

    debug["output_chars"] = len(speak)
    debug["sentences"] = len(_split_sentences(speak))
    return speak, debug
