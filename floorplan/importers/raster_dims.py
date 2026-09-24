"""Dimensions on bitmaps: OCR the numbers, then measure their dimension lines in pixels.

OCR engine: RapidOCR (ONNX, pip-installable, handles rotated text) or, as a fallback,
Tesseract through ``pytesseract``.  Without either, bitmaps need a paper scale or a
known overall size.

For each recognised number the dimension line is searched on the rows (or columns for
vertical text) around the text.  The line may run under the text or be interrupted by it.
Its measured extent ends at tick marks, at arrow tips touching a wall, or at the line end.
"""
from __future__ import annotations


import cv2
import numpy as np

from ..analysis.numbers import clean_text, parse_dimension_text
from ..log import Log
from ..model import DimEntity

_ENGINE = None


def _rapid():
    global _ENGINE
    if _ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR

        _ENGINE = RapidOCR()
    return _ENGINE


def ocr(img_bgr_or_gray: np.ndarray, log: Log) -> list[dict]:
    """Return texts as dicts: text, cx, cy, w, h (along/across the reading direction), vertical."""
    img = img_bgr_or_gray
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    out: list[dict] = []
    try:
        res, _ = _rapid()(img)
        for box, txt, conf in res or []:
            b = np.asarray(box, dtype=float)
            xs, ys = b[:, 0], b[:, 1]
            w, h = xs.max() - xs.min(), ys.max() - ys.min()
            vertical = h > 1.3 * w
            out.append({"text": clean_text(txt), "cx": float(xs.mean()), "cy": float(ys.mean()),
                        "x0": float(xs.min()), "x1": float(xs.max()), "y0": float(ys.min()), "y1": float(ys.max()),
                        "len": float(h if vertical else w), "h": float(w if vertical else h),
                        "vertical": vertical, "conf": float(conf)})
        log.info(f"OCR (RapidOCR): {len(out)} text(s) recognised")
        return out
    except ImportError:
        pass
    except Exception as e:  # noqa: BLE001
        log.warn(f"RapidOCR failed: {e}")
    try:
        import pytesseract

        d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config="--psm 11")
        for i, t in enumerate(d["text"]):
            t = clean_text(t)
            if not t:
                continue
            x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
            out.append({"text": t, "cx": x + w / 2, "cy": y + h / 2, "x0": x, "x1": x + w, "y0": y, "y1": y + h,
                        "len": w, "h": h, "vertical": False, "conf": float(d["conf"][i]) / 100})
        log.info(f"OCR (Tesseract): {len(out)} word(s) recognised (horizontal text only)")
    except Exception:  # noqa: BLE001 - no OCR available
        log.info("No OCR engine installed (pip install rapidocr-onnxruntime) - dimensions on bitmaps are not read")
    return out


def _merge_numbers(texts: list[dict]) -> list[dict]:
    """OCR sometimes splits '3,35 m' into '3,35' + 'm' - join a number with a following unit."""
    out = []
    used = set()
    for i, t in enumerate(texts):
        if i in used:
            continue
        for j, u in enumerate(texts):
            if j == i or j in used or t["vertical"] != u["vertical"]:
                continue
            if u["text"].lower() in ("m", "cm", "mm") and abs((u["cy"] if not t["vertical"] else u["cx"]) -
                                                              (t["cy"] if not t["vertical"] else t["cx"])) < t["h"] * 0.6:
                gap = (u["x0"] - t["x1"]) if not t["vertical"] else (t["y0"] - u["y1"])
                if -2 <= gap <= t["h"]:
                    t = {**t, "text": t["text"] + " " + u["text"], "x1": max(t["x1"], u["x1"]), "y0": min(t["y0"], u["y0"]),
                         "len": t["len"] + gap + u["len"]}
                    used.add(j)
                    break
        out.append(t)
    return out


def _run(line: np.ndarray, walls: np.ndarray, start: int, step: int, near: int, max_gap: int) -> tuple[int, str]:
    """Walk along a 1-D ink profile from ``start``; returns (last ink index, how it ended).

    Within ``near`` pixels of the start only 2-pixel gaps are bridged (so text strokes are
    never chained into a line); further out, gaps up to ``max_gap`` are bridged (dimension
    lines drawn under furniture fills are interrupted)."""
    n = len(line)
    i = start
    last = start
    gap = 0
    while 0 <= i < n:
        if walls[i]:
            return last, "wall"
        if line[i]:
            last = i
            gap = 0
        else:
            gap += 1
            if gap > (2 if abs(i - start) <= near else max_gap):
                return last, "end"
        i += step
    return last, "edge"


def _area_label(t: dict, texts: list[dict]) -> bool:
    """A number right below a word (room name) is the room's area, not a dimension."""
    for u in texts:
        if u is t or u["vertical"] != t["vertical"] or not any(ch.isalpha() for ch in u["text"].replace("m", "")):
            continue
        if t["vertical"]:
            overlap = min(t["y1"], u["y1"]) - max(t["y0"], u["y0"])
            gap = t["x0"] - u["x1"]
        else:
            overlap = min(t["x1"], u["x1"]) - max(t["x0"], u["x0"])
            gap = t["y0"] - u["y1"]
        if overlap > 0.3 * min(t["len"], u["len"]) and -2 <= gap <= 1.2 * t["h"]:
            return True
    return False


def find_dimensions(texts: list[dict], ink: np.ndarray, walls: np.ndarray, log: Log) -> list[DimEntity]:
    """``ink`` / ``walls``: boolean masks (image rows = y down).  Returns dims in y-up pixels."""
    H, W = ink.shape
    dims: list[DimEntity] = []
    walls_near = cv2.dilate(walls.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    ink = ink & ~walls_near  # the anti-aliased rim of a wall is not a line
    merged = _merge_numbers(texts)
    for t in merged:
        value, value_mm = parse_dimension_text(t["text"])
        if value is None or _area_label(t, merged):
            continue
        # work in a frame where the text reads horizontally: transpose for vertical text
        if t["vertical"]:
            m_ink, m_wall = ink.T, walls_near.T
            a0, a1, c0, c1 = t["y0"], t["y1"], t["x0"], t["x1"]
        else:
            m_ink, m_wall = ink, walls_near
            a0, a1, c0, c1 = t["x0"], t["x1"], t["y0"], t["y1"]
        h = max(4.0, c1 - c0)
        amid = (a0 + a1) / 2
        cmid = (c0 + c1) / 2
        half = int((a1 - a0) / 2) + 3
        best = None  # (distance from text, -length, row, lo, how_lo, hi, how_hi)
        for row in range(int(c0 - 0.7 * h), int(c1 + 0.7 * h) + 1):
            if not 0 <= row < m_ink.shape[0]:
                continue
            prof = m_ink[row]
            wal = m_wall[row]
            starts = []
            if prof[max(0, int(amid) - 1):int(amid) + 2].any():
                starts.append((int(amid), int(amid), half))  # line under / beside the text
            if c0 + 2 <= row <= c1 - 2:
                # a line interrupted by the text: ink must continue right next to both text ends
                ls, rs = int(a0) - 2, int(a1) + 2
                if 0 < ls and rs < len(prof) - 1 and prof[max(ls - 4, 0):ls + 1].any() and prof[rs:rs + 5].any():
                    starts.append((ls, rs, 4))
            for ls, rs, near in starts:
                lo, how_lo = _run(prof, wal, ls, -1, near, int(2.5 * h))
                hi, how_hi = _run(prof, wal, rs, 1, near, int(2.5 * h))
                if hi - lo < 1.1 * (a1 - a0):
                    continue
                cand = (abs(row - cmid) > h / 2 + 2, -(hi - lo), row, lo, how_lo, hi, how_hi)
                if best is None or cand < best:
                    best = cand
        if best is None:
            continue
        _, _, row, lo, how_lo, hi, how_hi = best
        # tick marks: short strokes crossing the line; the dimension spans between the ticks
        ticks = []
        tmin = max(3, int(0.25 * h))
        for col in range(lo, hi + 1):
            up = down = 0
            while row - up - 1 >= 0 and m_ink[row - up - 1, col] and up < 6 * h:
                up += 1
            while row + down + 1 < m_ink.shape[0] and m_ink[row + down + 1, col] and down < 6 * h:
                down += 1
            # a tick is short, or an extension line ending just past the dimension line;
            # long strokes on both sides (furniture edges crossing the line) are not ticks
            if max(up, down) >= tmin and min(up, down) >= 1 and (up + down <= 1.3 * h or (min(up, down) <= 0.3 * h and max(up, down) >= 1.5 * h)):
                ticks.append(col)
        left = [c for c in ticks if c < a0 - 1]
        right = [c for c in ticks if c > a1 + 1]
        # pixel index -> coordinate along the line: a tick is measured at its centre; a line
        # stopped by a wall ends at the wall face (1 px beyond the wall's anti-alias rim);
        # a free line end at the outer edge of its last pixel
        s1 = max(left) + 0.5 if left else float(lo - 1 if how_lo == "wall" else lo)
        s2 = min(right) + 0.5 if right else float(hi + 2 if how_hi == "wall" else hi + 1)
        if s2 - s1 < 0.8 * (a1 - a0):
            continue
        if t["vertical"]:  # along = image row (y down), across = column
            p1, p2 = (row + 0.5, H - s2), (row + 0.5, H - s1)
            ang = 90.0
            lp = (row + 0.5, H - t["cy"])
        else:
            p1, p2 = (s1, H - row - 0.5), (s2, H - row - 0.5)
            ang = 0.0
            lp = (t["cx"], H - row - 0.5)
        dims.append(DimEntity(p1, p2, ang, t["text"], value, value_mm, lp, "text"))
    log.info(f"Measured {len(dims)} dimension line(s) for the recognised numbers")
    return dims

