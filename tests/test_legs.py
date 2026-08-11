import math

import pytest

from snd.legs import place_legs_radially, clamp_edge_width


def test_place_legs_radially_even_angular_spacing():
    placements = place_legs_radially(["a", "b", "c", "d"])

    angles = [p.angle for p in placements]
    assert angles == pytest.approx([0.0, math.pi / 2, math.pi, 3 * math.pi / 2])


def test_place_legs_radially_empty_input():
    assert place_legs_radially([]) == []


def test_place_legs_radially_shrinks_font_for_many_legs():
    few = place_legs_radially(["a", "b"])
    many = place_legs_radially([f"leg{i}" for i in range(12)])

    assert many[0].font_size < few[0].font_size
    assert many[0].font_size >= 6.0  # never shrinks below the legibility floor


def test_clamp_edge_width_respects_bounds():
    assert clamp_edge_width(bond_dim=2, min_width=0.5, max_width=4.0) >= 0.5
    assert clamp_edge_width(bond_dim=10_000, min_width=0.5, max_width=4.0) <= 4.0


def test_clamp_edge_width_grows_with_bond_dim_within_bounds():
    small = clamp_edge_width(bond_dim=2, min_width=0.1, max_width=10.0)
    large = clamp_edge_width(bond_dim=64, min_width=0.1, max_width=10.0)

    assert large > small
