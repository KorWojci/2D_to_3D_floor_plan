"""Parsing of dimension texts ("3450", "3,45", "345 cm", "11'-6\"", ...)."""
from __future__ import annotations

import re
from typing import Optional

UNIT_MM = {"mm": 1.0, "cm": 10.0, "dm": 100.0, "m": 1000.0, "in": 25.4, '"': 25.4, "ft": 304.8, "'": 304.8}

_MTEXT_CODES = re.compile(r"\\[A-Za-z][^;\\]*;|\\[PpNn~]|[{}]")
_NUM = r"\d+(?:[.,]\d+)?"
_METRIC = re.compile(rf"^\s*(?:[A-Za-z]{{1,3}}\s*[=:]\s*)?({_NUM})\s*(mm|cm|dm|m)?\s*$", re.I)
_FT_IN = re.compile(r"^\s*(\d+)\s*'\s*-?\s*(\d+(?:\.\d+)?)?\s*(?:(\d+)/(\d+))?\s*\"?\s*$")
_IN = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(?:\"|in)\s*$")


def clean_text(text: str) -> str:
    """Strip MTEXT formatting codes and normalise whitespace."""
    t = _MTEXT_CODES.sub(" ", text or "")
    t = t.replace(" ", " ").replace("%%c", "Ø").replace("%%d", "°")
    return " ".join(t.split())


def parse_dimension_text(text: str) -> tuple[Optional[float], Optional[float]]:
    """Return ``(value, value_mm)``.

    ``value`` is the number in the drawing's *display* units (unknown unit);
    ``value_mm`` is set only if the text carried an explicit unit.
    Area labels (m², m2) and non-numeric strings return ``(None, None)``.
    """
    t = clean_text(text)
    if not t or re.search(r"m\s*[²2]|sq|%|°|ø|Ø|\+|x\s*\d|\d\s*x", t, re.I):
        return None, None
    m = _FT_IN.match(t)
    if m:
        ft = float(m.group(1))
        inch = float(m.group(2) or 0)
        if m.group(3):
            inch += float(m.group(3)) / float(m.group(4))
        mm = ft * 304.8 + inch * 25.4
        return mm, mm
    m = _IN.match(t)
    if m:
        mm = float(m.group(1)) * 25.4
        return mm, mm
    m = _METRIC.match(t)
    if not m:
        return None, None
    num = m.group(1)
    # "3,45" -> 3.45 ; "3,450" (thousands separator) -> 3450
    if "," in num:
        head, tail = num.split(",")
        num = head + tail if len(tail) == 3 and len(head) <= 3 and not m.group(2) and int(head) > 0 else head + "." + tail
    try:
        v = float(num)
    except ValueError:
        return None, None
    if v <= 0:
        return None, None
    unit = (m.group(2) or "").lower()
    if unit:
        return v, v * UNIT_MM[unit]
    return v, None
