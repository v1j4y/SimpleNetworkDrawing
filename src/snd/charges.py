"""Charge-sector data model.

Reads real charge/block metadata from symmray-backed quimb Tensor.data
when present (duck-typed, mirroring quimb's own `isblocksparse` check),
and otherwise accepts an explicit, illustrative charge spec so the
library never requires real symmetric tensor data to draw a
block-diagonal glyph.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

Charge = int


@dataclass(frozen=True)
class ChargeLeg:
    """Charge-sector info for a single tensor leg (index)."""

    ind: str
    charges: Tuple[Charge, ...]
    dual: bool = False


@dataclass(frozen=True)
class SectorSpec:
    """Allowed (row_charge, col_charge) sector combinations for a
    block-diagonal tensor glyph."""

    sectors: Tuple[Tuple[Charge, Charge], ...]


def charges_from_symmray(tensor, ind: str) -> Optional[ChargeLeg]:
    """Read charge-sector info for `ind` from a symmray-backed
    Tensor.data, if present. Returns None otherwise."""
    data = tensor.data
    if not hasattr(data, "indices") or not hasattr(data, "sectors"):
        return None
    axis = tensor.inds.index(ind)
    block_index = data.indices[axis]
    charges = tuple(sorted(block_index.chargemap.keys()))
    return ChargeLeg(ind=ind, charges=charges, dual=bool(block_index.dual))


def charges_from_spec(
    ind: str, charges: Sequence[Charge], dual: bool = False
) -> ChargeLeg:
    """Build charge-sector info for `ind` from an explicit charge list,
    with no real tensor data required."""
    return ChargeLeg(ind=ind, charges=tuple(charges), dual=dual)


def default_sector_spec(
    row_charges: Sequence[Charge],
    col_charges: Sequence[Charge],
    total: Charge = 0,
) -> SectorSpec:
    """Diagonal-in-total-charge default: pairs (r, c) with r + c == total,
    e.g. electron-number conservation for a 4-electron total across a
    5-dimensional leg on each side gives (0,4),(1,3),(2,2),(3,1),(4,0)."""
    sectors = tuple(
        (r, c) for r in row_charges for c in col_charges if r + c == total
    )
    return SectorSpec(sectors=sectors)
