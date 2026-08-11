"""Leg placement geometry for many-leg tensors: even radial spacing plus
font-size/offset shrink as leg count grows, so labels stay legible
instead of overlapping; and a clamped edge-width function so many-leg
tensors aren't visually dominated by a few thick bonds."""

import math
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class LegPlacement:
    ind: str
    angle: float
    label_offset: float
    font_size: float


def place_legs_radially(
    inds: List[str], base_font_size: float = 10.0, base_offset: float = 0.15
) -> List[LegPlacement]:
    n = len(inds)
    if n == 0:
        return []
    scale = min(1.0, 6 / n) if n > 6 else 1.0
    font_size = max(base_font_size * scale, 6.0)
    offset = base_offset * (1.0 + 0.5 * (1 - scale))
    return [
        LegPlacement(
            ind=ind,
            angle=2 * math.pi * i / n,
            label_offset=offset,
            font_size=font_size,
        )
        for i, ind in enumerate(inds)
    ]


def clamp_edge_width(
    bond_dim: int,
    min_width: float = 0.5,
    max_width: float = 4.0,
    scale: float = 1.0,
) -> float:
    raw = scale * math.log2(max(bond_dim, 2))
    return max(min_width, min(max_width, raw))
