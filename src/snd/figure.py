"""TNFigure: the orchestration object. Owns a quimb.schematic.Drawing
end-to-end (unlike quimb's own draw_tn, which builds one internally and
discards it), so vector export always works."""

import math
from typing import Dict, Optional, Sequence

from quimb.schematic import Drawing

from .charges import (
    ChargeLeg,
    charges_from_spec,
    charges_from_symmray,
    default_sector_spec,
)
from .glyphs import draw_block_diagonal, draw_plain, draw_wavefunction
from .layout import LayoutKind, compute_positions
from .legs import clamp_edge_width, place_legs_radially
from .palette import color_for_charge
from .style import StyleSheet


class TNFigure:
    def __init__(self, tn, figsize=(6, 4)):
        self.tn = tn
        self.figsize = figsize
        self.styles = StyleSheet()
        self._layout_kind: LayoutKind = "auto"
        self._fix = None
        self._node_types: Dict[int, str] = {}
        self._leg_labels: Dict[str, Dict[str, str]] = {}
        self._explicit_charges: Dict[str, ChargeLeg] = {}
        self._drawing: Optional[Drawing] = None
        self._positions = None

    def layout(self, kind: LayoutKind = "auto", fix=None) -> "TNFigure":
        self._layout_kind = kind
        self._fix = fix
        return self

    def style(self, target: str, node: Optional[str] = None, **kwargs) -> "TNFigure":
        if node is not None:
            for tid in self.tn._get_tids_from_tags(target):
                self._node_types[tid] = node
        if kwargs:
            self.styles.add(target, **kwargs)
        return self

    def label_legs(self, selector: str, labels: Dict[str, str]) -> "TNFigure":
        self._leg_labels[selector] = labels
        return self

    def charges(
        self,
        ind: str,
        sectors: Optional[Sequence[int]] = None,
        charges: Optional[Sequence[int]] = None,
        dual: bool = False,
    ) -> "TNFigure":
        source = sectors if sectors is not None else charges
        self._explicit_charges[ind] = charges_from_spec(ind, source, dual=dual)
        return self

    def draw(self) -> "TNFigure":
        self._positions = compute_positions(self.tn, kind=self._layout_kind, fix=self._fix)
        self._drawing = Drawing(figsize=self.figsize)
        for tid, t in self.tn.tensor_map.items():
            coo = self._positions[tid]
            node_type = self._node_types.get(tid, "plain")
            style = self.styles.resolve_tensor(set(t.tags), node_type)
            if node_type == "wavefunction":
                draw_wavefunction(self._drawing, coo, style.get("radius", 0.3), style)
            elif node_type == "block-diagonal":
                row_charges, col_charges, sector_spec = self._resolve_sectors(t)
                draw_block_diagonal(
                    self._drawing,
                    coo,
                    style.get("size", 0.6),
                    sector_spec,
                    row_charges,
                    col_charges,
                    style,
                )
            else:
                draw_plain(self._drawing, coo, style.get("radius", 0.25), style, shape=style.get("shape", "circle"))
            self._draw_legs(tid, t, coo)
        self._draw_bonds()
        return self

    def savefig(self, path: str, dpi: int = 300, **kwargs) -> None:
        if self._drawing is None:
            self.draw()
        self._drawing.savefig(path, dpi=dpi, **kwargs)

    def _resolve_sectors(self, t):
        inds = t.inds
        left_inds = t.left_inds or inds[: len(inds) // 2]
        right_inds = tuple(ix for ix in inds if ix not in left_inds)
        if not left_inds:
            # single-leg (e.g. boundary) tensor: no natural row/col split,
            # so treat its one group of legs as both row and column,
            # giving a plain diagonal square rather than an empty one.
            left_inds, right_inds = right_inds, left_inds
        row_charges = self._charges_for_group(t, left_inds)
        col_charges = self._charges_for_group(t, right_inds) or row_charges
        total = getattr(t.data, "charge", 0)
        return row_charges, col_charges, default_sector_spec(row_charges, col_charges, total=total)

    def _charges_for_group(self, t, inds):
        charges = set()
        for ix in inds:
            leg = charges_from_symmray(t, ix) or self._explicit_charges.get(ix)
            if leg is not None:
                charges.update(leg.charges)
        return sorted(charges)

    def _labels_for_tensor(self, tid, t):
        labels: Dict[str, str] = {}
        for selector, mapping in self._leg_labels.items():
            if selector in t.tags or selector == tid:
                labels.update(mapping)
        return labels

    def _draw_legs(self, tid, t, coo):
        labels = self._labels_for_tensor(tid, t)
        for placement in place_legs_radially(list(t.inds)):
            label = labels.get(placement.ind, placement.ind)
            lx = coo[0] + placement.label_offset * math.cos(placement.angle)
            ly = coo[1] + placement.label_offset * math.sin(placement.angle)
            self._drawing.text((lx, ly), label, fontsize=placement.font_size)

    def _leg_charge_leg(self, ix) -> Optional[ChargeLeg]:
        for t in self.tn.tensor_map.values():
            if ix in t.inds:
                leg = charges_from_symmray(t, ix)
                if leg is not None:
                    return leg
        return self._explicit_charges.get(ix)

    def _draw_bonds(self):
        for ix, tids in self.tn.ind_map.items():
            tids = tuple(tids)
            if len(tids) != 2:
                continue
            cooa, coob = self._positions[tids[0]], self._positions[tids[1]]
            leg = self._leg_charge_leg(ix)
            if leg is None:
                width = clamp_edge_width(self.tn.ind_size(ix))
                self._drawing.line(cooa, coob, linewidth=width)
                continue
            n = len(leg.charges)
            offsets = [(-0.15 + 0.3 * i / max(n - 1, 1)) for i in range(n)] if n > 1 else [0.0]
            for charge, offset in zip(leg.charges, offsets):
                self._drawing.line_offset(cooa, coob, offset, color=color_for_charge(charge))
