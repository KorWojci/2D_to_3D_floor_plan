"""Unit tests: dimension text parsing and dimension-driven calibration."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from floorplan.analysis.calibration import calibrate, finalize_derived  # noqa: E402
from floorplan.analysis.numbers import parse_dimension_text  # noqa: E402
from floorplan.log import Log  # noqa: E402
from floorplan.model import DimEntity, Drawing, Seg, Settings  # noqa: E402


@pytest.mark.parametrize("text,value,mm", [
    ("3450", 3450, None),
    ("3,45", 3.45, None),
    ("3.45 m", 3.45, 3450),
    ("345cm", 345, 3450),
    ("12'-6\"", 3810, 3810),
    ("11' 6 1/2\"", 11 * 304.8 + 6.5 * 25.4, 11 * 304.8 + 6.5 * 25.4),
    ("12.5 m²", None, None),
    ("Kitchen", None, None),
    ("\\A1;{\\H0.7x;3450}", 3450, None),
])
def test_parse(text, value, mm):
    v, vmm = parse_dimension_text(text)
    if value is None:
        assert v is None
    else:
        assert v == pytest.approx(value)
        assert (vmm is None) if mm is None else vmm == pytest.approx(mm)


def _hdim(x1, x2, value, y=-100.0):
    return DimEntity((x1, 0.0), (x2, 0.0), 0.0, f"{value:g}", value, None, (x1, y))


def test_chain_missing_member_is_derived_and_geometry_follows_dimensions():
    # drawn in cm with the middle part drawn 5 % too long; dims say 300 + ? + 400 = 1000
    d = Drawing(unit_mm=None)
    d.segments = [Seg(0, 0, 0, 300), Seg(1015, 0, 1015, 300)]
    d.dims = [_hdim(0, 300, 300), _hdim(615, 1015, 400), _hdim(0, 1015, 1000, y=-200)]
    cal = calibrate(d, Settings(), Log())
    assert cal.unit_label == "cm"
    # the drawn 1015 cm is mapped to the stated 10 000 mm
    assert cal.fx(1015) - cal.fx(0) == pytest.approx(10000)
    assert cal.fx(300) - cal.fx(0) == pytest.approx(3000)
    derived = finalize_derived(cal)
    assert [round(x.value) for x in derived] == [3000]


def test_header_units_without_dimensions():
    d = Drawing(unit_mm=10.0, unit_name="cm")
    d.segments = [Seg(0, 0, 1000, 0), Seg(0, 0, 0, 700)]
    cal = calibrate(d, Settings(), Log())
    assert cal.fx(1000) == pytest.approx(10000)
    assert cal.confidence == "header"
