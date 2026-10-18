"""Tolerant extraction of a JSON object from small-VLM output."""
from __future__ import annotations

import ast
import json
import re
from typing import Any


def extract_json(text: str) -> dict[str, Any] | None:
    """Return the first JSON object in `text`, repairing common VLM mistakes
    (code fences, trailing commas, single quotes, Python literals, truncation)."""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.IGNORECASE | re.MULTILINE).strip()
    start = t.find("{")
    if start < 0:
        return None
    body = _balanced(t[start:])
    for cand in (body, _repair(body), _close(_repair(body))):
        if cand is None:
            continue
        try:
            v = json.loads(cand)
            if isinstance(v, dict):
                return v
        except Exception:  # noqa: BLE001
            pass
        try:
            v = ast.literal_eval(_pythonize(cand))
            if isinstance(v, dict):
                return v
        except Exception:  # noqa: BLE001
            pass
    return None


def _balanced(s: str) -> str:
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[: i + 1]
    return s  # truncated - caller will try to close it


def _repair(s: str) -> str:
    s = re.sub(r",\s*([}\]])", r"\1", s)
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)
    s = re.sub(r"(?<=[{,\s])'([^'\n]+?)'\s*:", r'"\1":', s)
    s = re.sub(r":\s*'([^'\n]*?)'(?=\s*[,}\]])", r': "\1"', s)
    return s


def _close(s: str) -> str:
    """Close a truncated object: drop a dangling key/partial value, then balance."""
    s = s.rstrip()
    if s.count('"') % 2:
        s += '"'
    s = re.sub(r',\s*"[^"]*"\s*:?\s*$', "", s)
    s = re.sub(r",\s*$", "", s)
    opens = []
    in_str = esc = False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            opens.append("}" if ch == "{" else "]")
        elif ch in "}]" and opens:
            opens.pop()
    return s + "".join(reversed(opens))


def _pythonize(s: str) -> str:
    return re.sub(r"\btrue\b", "True", re.sub(r"\bfalse\b", "False", re.sub(r"\bnull\b", "None", s)))


def to_bool(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("true", "yes", "y", "1", "present", "detected"):
        return True
    if s in ("false", "no", "n", "0", "absent", "none", "not present"):
        return False
    return None
