"""Position computation for a TensorNetwork's tensors. Three lattice-
aware modes (`chain`, `grid`, `manual`) cover the small, hand-tunable
networks this library targets; `auto` defers to quimb's own graph
layout (via `draw_tn(..., get="pos")`) for anything else."""

from typing import Dict, Literal, Optional, Tuple

from quimb.tensor.drawing import draw_tn

LayoutKind = Literal["chain", "grid", "auto", "manual"]

Positions = Dict[int, Tuple[float, float]]


def compute_positions(
    tn, kind: LayoutKind = "auto", fix: Optional[Positions] = None
) -> Positions:
    if kind == "manual":
        if not fix:
            raise ValueError(
                "layout kind 'manual' requires a `fix` mapping of "
                "tid -> (x, y)"
            )
        return dict(fix)
    if kind == "chain":
        return _chain_positions(tn)
    if kind == "grid":
        return _grid_positions(tn)
    return _auto_positions(tn, fix)


def _chain_positions(tn) -> Positions:
    return {tid: (float(i), 0.0) for i, tid in enumerate(tn.tensor_map)}


def _grid_positions(tn) -> Positions:
    if not (
        hasattr(tn, "gen_site_coos")
        and hasattr(tn, "site_tag")
        and hasattr(tn, "Lx")
        and hasattr(tn, "Ly")
    ):
        raise ValueError(
            "layout kind 'grid' requires a lattice-shaped TensorNetwork "
            "(e.g. a quimb PEPS) exposing gen_site_coos()/site_tag()"
        )
    positions: Positions = {}
    for i, j in tn.gen_site_coos():
        for tid in tn._get_tids_from_tags(tn.site_tag(i, j)):
            positions[tid] = (float(j), float(-i))
    return positions


def _auto_positions(tn, fix: Optional[Positions]) -> Positions:
    pos = draw_tn(tn, fix=fix, dim=2, get="pos")
    return {
        tid: (float(pos[tid][0]), float(pos[tid][1])) for tid in tn.tensor_map
    }
