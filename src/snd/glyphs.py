"""Node glyph renderers. Three kinds:

- plain: existing simple shapes (circle/triangle), for non-symmetric
  tensors and backward compatibility with the old figures.
- wavefunction: a circle, for tensors with many incoming charge legs
  fusing to one outgoing leg (legs handle radial placement well).
- block-diagonal: a square subdivided into diagonal blocks, one per
  allowed charge sector, colored and labeled by charge — the glyph's
  shape *is* the tensor's sparsity pattern.
"""

from typing import Dict, Sequence

from quimb.schematic import Drawing

from .charges import SectorSpec
from .palette import color_for_charge


def draw_plain(d: Drawing, coo, radius: float, style: dict, shape: str = "circle"):
    if shape == "circle":
        d.circle(coo, radius=radius, **style)
    elif shape == "triangle":
        d.regular_polygon(coo, n=3, radius=radius, **style)
    else:
        raise ValueError(f"unknown plain glyph shape: {shape}")


def draw_wavefunction(d: Drawing, coo, radius: float, style: dict):
    d.circle(coo, radius=radius, **style)


def draw_block_diagonal(
    d: Drawing,
    coo,
    size: float,
    sector_spec: SectorSpec,
    row_charges: Sequence[int],
    col_charges: Sequence[int],
    style: Dict,
):
    nrow, ncol = len(row_charges), len(col_charges)
    cell_w, cell_h = size / ncol, size / nrow
    x0, y0 = coo[0] - size / 2, coo[1] - size / 2
    row_index = {c: i for i, c in enumerate(row_charges)}
    col_index = {c: i for i, c in enumerate(col_charges)}
    charge_fontsize = style.get("charge_fontsize", 8)
    extra_style = {
        k: v for k, v in style.items() if k not in ("facecolor", "charge_fontsize")
    }

    for row_charge, col_charge in sector_spec.sectors:
        i, j = row_index[row_charge], col_index[col_charge]
        cell_x0 = x0 + j * cell_w
        cell_y0 = y0 + (nrow - 1 - i) * cell_h
        corners = [
            (cell_x0, cell_y0),
            (cell_x0, cell_y0 + cell_h),
            (cell_x0 + cell_w, cell_y0 + cell_h),
            (cell_x0 + cell_w, cell_y0),
        ]
        d.shape(corners, facecolor=color_for_charge(row_charge), **extra_style)
        d.text(
            (cell_x0 + cell_w / 2, cell_y0 + cell_h / 2),
            str(row_charge),
            fontsize=charge_fontsize,
        )

    border = [
        (x0, y0),
        (x0, y0 + size),
        (x0 + size, y0 + size),
        (x0 + size, y0),
    ]
    d.shape(border, facecolor="none", edgecolor=style.get("edgecolor", "black"))
