"""Multilingual layer / block name keywords used to classify CAD content."""
from __future__ import annotations

import re

WALL_WORDS = [
    "wall", "walls", "a-wall", "s-wall", "mur", "murs", "sciana", "ściana", "sciany", "ściany",
    "wand", "wände", "waende", "muro", "muros", "pared", "paredes", "parete", "pareti", "zed", "stena",
]
DOOR_WORDS = [
    "door", "doors", "a-door", "drzwi", "tur", "tür", "tuer", "turen", "türen", "porte", "portes",
    "puerta", "puertas", "porta", "porte", "deur", "dvere", "dveře", "ajto", "ovi", "dør", "dorr", "dörr",
]
WINDOW_WORDS = [
    "window", "windows", "win", "wind", "a-glaz", "glaz", "glazing", "okno", "okna", "fenster",
    "fenetre", "fenêtre", "fenetres", "ventana", "ventanas", "finestra", "finestre", "raam", "okna",
    "okno", "ablak", "ikkuna", "vindu", "fönster",
]
IGNORE_WORDS = [
    "dim", "dims", "dimension", "dimensions", "wymiar", "wymiary", "cota", "cotas", "bemassung", "cotation",
    "text", "texts", "txt", "tekst", "anno", "annotation", "furn", "furniture", "mebl", "meble", "mobel",
    "möbel", "equip", "grid", "axis", "osie", "hatch", "kresk", "title", "frame", "defpoints", "viewport",
    "vport", "plumb", "elec", "lighting", "sanit", "stairs",
]


def _tokens(name: str) -> list[str]:
    return [t for t in re.split(r"[^0-9a-ząćęłńóśźżäöüßéèêàçñ]+", (name or "").lower()) if t]


def _match(name: str, words: list[str], extra: str = "") -> bool:
    low = (name or "").lower()
    if not low:
        return False
    toks = _tokens(low)
    for w in words:
        if "-" in w:
            if w in low:
                return True
        elif w in toks or any(t.startswith(w) and len(w) >= 4 for t in toks):
            return True
    for pat in [p.strip().lower() for p in extra.split(",") if p.strip()]:
        if pat in low:
            return True
    return False


def is_wall(name: str, extra: str = "") -> bool:
    return _match(name, WALL_WORDS, extra)


def is_door(name: str, extra: str = "") -> bool:
    return _match(name, DOOR_WORDS, extra)


def is_window(name: str, extra: str = "") -> bool:
    return _match(name, WINDOW_WORDS, extra) and not is_door(name)


def is_ignored(name: str) -> bool:
    return _match(name, IGNORE_WORDS)


def classify(name: str, door_extra: str = "", window_extra: str = "") -> str:
    if is_door(name, door_extra):
        return "door"
    if is_window(name, window_extra):
        return "window"
    return ""
