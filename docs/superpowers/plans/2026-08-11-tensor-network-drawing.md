# SimpleNetworkDrawing Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hand-coordinate org-mode scripts in `SimpleNetworkDrawing` with a reusable Python package (`snd`) that draws publication-quality SVG/PDF figures directly from quimb `Tensor`/`TensorNetwork` objects, with first-class support for block-diagonal abelian-symmetric tensors.

**Architecture:** A small stack of focused modules — charge-sector reading (`charges.py`), palette (`palette.py`), tag/leg/node-type styling (`style.py`), leg geometry (`legs.py`), layout (`layout.py`), and glyph rendering (`glyphs.py`) — composed by one orchestration class, `TNFigure` (`figure.py`), which owns a `quimb.schematic.Drawing` instance end-to-end so vector export (`savefig`) works. See `docs/superpowers/specs/2026-08-11-tensor-network-drawing-design.md` for the full design rationale.

**Tech Stack:** Python, quimb (the user's fork, `v1j4y/quimb`), matplotlib (via `quimb.schematic.Drawing`), pytest. No dependency on `symmray` for core functionality — it is used only opportunistically when real tensor data carries it (duck-typed).

## Global Constraints

- Input type is always a quimb `Tensor` or `TensorNetwork` — never a parallel/decoupled data model (spec §1).
- No dependency on `symmray` being installed for core functionality or examples/tests (spec §13). Charge-sector data may come from real symmray-backed `tensor.data` (read opportunistically, duck-typed) or from an explicit illustrative spec.
- Color palette: reuse quimb's existing Okabe-Ito colorblind-safe palette (`quimb.schematic.get_color`/`hash_to_color`/`auto_colors`) for both tag colors and charge-value colors (spec §3, §9).
- Tag-based styling is central, extended to also target leg-name patterns and node type (spec §3, §9).
- Leg/charge labels render at the node (near the leg's attachment point), not in a side legend, with spacing/font-size that adapts so many-leg tensors stay legible (spec §3, §7).
- Two symmetric-tensor glyphs: block-diagonal grid (generic symmetric tensor) and circle (wavefunction/fusion tensor) (spec §3, §6).
- Typical networks are small (<20 tensors); no large-lattice auto-layout engine is in scope (spec §13).
- Export must produce real SVG/PDF at print resolution (`dpi=300` default) via the retained `Drawing.savefig` (spec §10).

---

## File Structure

```
pyproject.toml                     # package metadata + deps (quimb from the fork, matplotlib, networkx, pytest)
src/snd/__init__.py                # public exports: TNFigure, quick_draw
src/snd/charges.py                 # ChargeLeg, charges_from_symmray, charges_from_spec, SectorSpec, default_sector_spec
src/snd/palette.py                 # color_for_tag, color_for_charge (thin wrappers over quimb.schematic color fns)
src/snd/style.py                   # StyleRule, StyleSheet (tag / leg-pattern / node-type style dispatch)
src/snd/legs.py                    # LegPlacement, place_legs_radially, clamp_edge_width
src/snd/layout.py                  # compute_positions: chain / grid / auto / manual
src/snd/glyphs.py                  # draw_plain, draw_wavefunction, draw_block_diagonal
src/snd/figure.py                  # TNFigure: orchestrates everything, owns Drawing, draw()/savefig()
src/snd/compat.py                  # quick_draw(tn, **kwargs) backward-compatible entry point
tests/conftest.py                  # shared quimb TensorNetwork fixtures
tests/test_charges.py
tests/test_palette.py
tests/test_style.py
tests/test_legs.py
tests/test_layout.py
tests/test_glyphs.py
tests/test_figure.py
tests/test_compat.py
examples/mps_chain.py              # plain-glyph MPS chain, migrated from legacy/figures.org
examples/small_circuit.py          # small circuit-style diagram
examples/peps_patch.py             # small PEPS patch, grid layout
examples/charge_sector_mps.py      # abelian-symmetric example: electron-number conservation
examples/figs/                     # generated output (gitignored except .gitkeep)
legacy/figures.org                 # moved from src/figures.org (git mv, preserved as reference)
legacy/figs/                       # moved from figs/ (git mv)
README.md                          # rewritten: install, quickstart, API reference, gallery
```

Rationale: each module has one responsibility and is unit-testable without a real symmetric backend (charges/palette/style/legs/layout are pure-ish functions on plain data); `glyphs.py` and `figure.py` are the only modules that touch the `Drawing` canvas, kept separate so glyph logic can be tested by inspecting the matplotlib `Axes` patch count without going through full `TNFigure` orchestration.

---

## Task 1: Project scaffolding and legacy content move

**Files:**
- Create: `pyproject.toml`
- Create: `src/snd/__init__.py` (empty placeholder, filled in Task 9)
- Create: `tests/conftest.py`
- Move: `src/figures.org` → `legacy/figures.org` (git mv)
- Move: `figs/` → `legacy/figs/` (git mv)
- Create: `examples/figs/.gitkeep`
- Modify: `.gitignore`

**Interfaces:**
- Produces: an installable empty `snd` package, a working `pytest` invocation, and a `qtn_chain` fixture (3-site plain MPS-like `TensorNetwork`) used by later test tasks.

- [ ] **Step 1: Move legacy content out of the way**

```bash
mkdir -p legacy
git mv src/figures.org legacy/figures.org
git mv figs legacy/figs
mkdir -p src/snd examples/figs
touch examples/figs/.gitkeep
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "snd"
version = "0.1.0"
description = "Publication-quality tensor network diagrams from quimb TensorNetwork objects."
requires-python = ">=3.10"
dependencies = [
    "quimb @ git+https://github.com/v1j4y/quimb.git",
    "matplotlib>=3.7",
    "networkx>=3.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.0"]

[tool.hatch.build.targets.wheel]
packages = ["src/snd"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Create empty package init**

```python
# src/snd/__init__.py
"""Publication-quality tensor network diagrams from quimb TensorNetwork objects."""
```

- [ ] **Step 4: Write the shared test fixture**

```python
# tests/conftest.py
import pytest
import quimb.tensor as qtn


@pytest.fixture
def qtn_chain():
    """A plain 3-tensor MPS-like TensorNetwork with no symmetry, for layout/style tests."""
    return qtn.MPS_rand_state(L=3, bond_dim=4)
```

- [ ] **Step 5: Update `.gitignore` and install the package editable**

Add to `.gitignore`:
```
examples/figs/*
!examples/figs/.gitkeep
*.egg-info/
__pycache__/
```

```bash
pip install -e ".[dev]"
```

- [ ] **Step 6: Verify the environment is wired up**

Run: `pytest --collect-only`
Expected: exits 0, reports "no tests collected" (no test files exist yet — that's expected at this step).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/snd/__init__.py tests/conftest.py .gitignore \
        legacy/figures.org legacy/figs examples/figs/.gitkeep
git commit -m "Scaffold snd package, move legacy figures.org content to legacy/"
```

---

## Task 2: Charge-sector data model (`charges.py`)

**Files:**
- Create: `src/snd/charges.py`
- Test: `tests/test_charges.py`

**Interfaces:**
- Produces:
  - `ChargeLeg` (frozen dataclass): fields `ind: str`, `charges: tuple[int, ...]`, `dual: bool`.
  - `charges_from_symmray(tensor, ind: str) -> ChargeLeg | None`
  - `charges_from_spec(ind: str, charges: Sequence[int], dual: bool = False) -> ChargeLeg`
  - `SectorSpec` (frozen dataclass): field `sectors: tuple[tuple[int, int], ...]`.
  - `default_sector_spec(row_charges: Sequence[int], col_charges: Sequence[int], total: int = 0) -> SectorSpec`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_charges.py
from types import SimpleNamespace
from snd.charges import (
    ChargeLeg,
    charges_from_symmray,
    charges_from_spec,
    default_sector_spec,
)


class FakeBlockIndex:
    def __init__(self, chargemap, dual):
        self.chargemap = chargemap
        self.dual = dual


class FakeSymmrayData:
    """Duck-typed stand-in for a symmray-backed array, matching the
    attributes quimb itself reads (indices, sectors, duals, charge)."""

    def __init__(self, indices):
        self.indices = indices
        self.sectors = ((0, 0),)
        self.duals = tuple(ix.dual for ix in indices)
        self.charge = 0


class FakeTensor:
    def __init__(self, inds, data):
        self.inds = inds
        self.data = data


def test_charges_from_symmray_reads_chargemap_keys_sorted():
    data = FakeSymmrayData(
        indices=[
            FakeBlockIndex({1: 2, 0: 3}, dual=False),
            FakeBlockIndex({0: 2}, dual=True),
        ]
    )
    tensor = FakeTensor(inds=("a", "b"), data=data)

    leg = charges_from_symmray(tensor, "a")

    assert leg == ChargeLeg(ind="a", charges=(0, 1), dual=False)


def test_charges_from_symmray_returns_none_for_plain_array():
    tensor = FakeTensor(inds=("a", "b"), data=object())

    assert charges_from_symmray(tensor, "a") is None


def test_charges_from_spec_builds_charge_leg():
    leg = charges_from_spec("b01", charges=[0, 1, 2, 3, 4], dual=True)

    assert leg == ChargeLeg(ind="b01", charges=(0, 1, 2, 3, 4), dual=True)


def test_default_sector_spec_electron_number_conservation():
    spec = default_sector_spec(row_charges=range(5), col_charges=range(5), total=4)

    assert spec.sectors == ((0, 4), (1, 3), (2, 2), (3, 1), (4, 0))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_charges.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'snd.charges'`

- [ ] **Step 3: Implement `charges.py`**

```python
# src/snd/charges.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_charges.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/snd/charges.py tests/test_charges.py
git commit -m "Add charge-sector data model (charges.py)"
```

---

## Task 3: Palette (`palette.py`)

**Files:**
- Create: `src/snd/palette.py`
- Test: `tests/test_palette.py`

**Interfaces:**
- Consumes: `quimb.schematic.hash_to_color`.
- Produces: `color_for_tag(tag: str) -> tuple`, `color_for_charge(charge: int) -> tuple`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_palette.py
from snd.palette import color_for_tag, color_for_charge


def test_color_for_charge_is_deterministic():
    assert color_for_charge(2) == color_for_charge(2)


def test_color_for_charge_differs_across_values():
    colors = {color_for_charge(c) for c in range(5)}
    assert len(colors) == 5


def test_color_for_tag_is_deterministic_and_differs_from_charge_namespace():
    assert color_for_tag("A") == color_for_tag("A")
    # same raw value in the tag vs. charge namespace must not collide,
    # since they are hashed with different prefixes
    assert color_for_tag("2") != color_for_charge(2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_palette.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'snd.palette'`

- [ ] **Step 3: Implement `palette.py`**

```python
# src/snd/palette.py
"""Thin wrapper over quimb's existing colorblind-safe (Okabe-Ito) color
machinery, reused for both tag coloring and charge-value coloring so a
given charge value is visually consistent across a whole figure."""

from quimb.schematic import hash_to_color


def color_for_tag(tag: str) -> tuple:
    return hash_to_color(f"tag:{tag}")


def color_for_charge(charge: int) -> tuple:
    return hash_to_color(f"charge:{charge}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_palette.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/snd/palette.py tests/test_palette.py
git commit -m "Add palette module reusing quimb's colorblind-safe colors"
```

---

## Task 4: Style dispatch (`style.py`)

**Files:**
- Create: `src/snd/style.py`
- Test: `tests/test_style.py`

**Interfaces:**
- Produces:
  - `StyleRule` (dataclass): fields `target: str`, `style: dict`.
  - `StyleSheet`: methods `add(target: str, **style) -> StyleSheet`, `resolve_tensor(tags: set[str], node_type: str) -> dict`, `resolve_leg(ind: str) -> dict`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_style.py
from snd.style import StyleSheet


def test_resolve_tensor_merges_matching_tag_rules_in_order():
    sheet = StyleSheet()
    sheet.add("A", color="blue")
    sheet.add("A", shape="triangle")  # later rule overrides/extends

    resolved = sheet.resolve_tensor(tags={"A"}, node_type="plain")

    assert resolved == {"color": "blue", "shape": "triangle"}


def test_resolve_tensor_matches_node_type_rule():
    sheet = StyleSheet()
    sheet.add("type:wavefunction", radius=0.4)

    resolved = sheet.resolve_tensor(tags={"psi"}, node_type="wavefunction")

    assert resolved == {"radius": 0.4}


def test_resolve_tensor_ignores_non_matching_rules():
    sheet = StyleSheet()
    sheet.add("B", color="red")

    resolved = sheet.resolve_tensor(tags={"A"}, node_type="plain")

    assert resolved == {}


def test_resolve_leg_matches_glob_pattern():
    sheet = StyleSheet()
    sheet.add("leg:b*", linewidth=2)
    sheet.add("leg:k0", linewidth=5)

    assert sheet.resolve_leg("b01") == {"linewidth": 2}
    assert sheet.resolve_leg("k0") == {"linewidth": 5}
    assert sheet.resolve_leg("other") == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_style.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'snd.style'`

- [ ] **Step 3: Implement `style.py`**

```python
# src/snd/style.py
"""Tag-centric style dispatch, extended to also target leg-name glob
patterns (`leg:<pattern>`) and node type (`type:<node_type>`), not just
tensor tags. Later-added rules override earlier ones on conflicting
keys, mirroring quimb's own tag-styling precedence."""

import fnmatch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set


@dataclass
class StyleRule:
    target: str
    style: Dict[str, Any] = field(default_factory=dict)


class StyleSheet:
    def __init__(self) -> None:
        self._rules: List[StyleRule] = []

    def add(self, target: str, **style: Any) -> "StyleSheet":
        self._rules.append(StyleRule(target=target, style=style))
        return self

    def resolve_tensor(self, tags: Set[str], node_type: str) -> Dict[str, Any]:
        resolved: Dict[str, Any] = {}
        for rule in self._rules:
            if rule.target.startswith("leg:"):
                continue
            if rule.target in tags or rule.target == f"type:{node_type}":
                resolved.update(rule.style)
        return resolved

    def resolve_leg(self, ind: str) -> Dict[str, Any]:
        resolved: Dict[str, Any] = {}
        for rule in self._rules:
            if not rule.target.startswith("leg:"):
                continue
            pattern = rule.target[len("leg:") :]
            if fnmatch.fnmatch(ind, pattern):
                resolved.update(rule.style)
        return resolved
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_style.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/snd/style.py tests/test_style.py
git commit -m "Add tag/leg-pattern/node-type style dispatch (style.py)"
```

---

## Task 5: Leg geometry (`legs.py`)

**Files:**
- Create: `src/snd/legs.py`
- Test: `tests/test_legs.py`

**Interfaces:**
- Produces:
  - `LegPlacement` (frozen dataclass): fields `ind: str`, `angle: float`, `label_offset: float`, `font_size: float`.
  - `place_legs_radially(inds: list[str], base_font_size: float = 10.0, base_offset: float = 0.15) -> list[LegPlacement]`
  - `clamp_edge_width(bond_dim: int, min_width: float = 0.5, max_width: float = 4.0, scale: float = 1.0) -> float`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_legs.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_legs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'snd.legs'`

- [ ] **Step 3: Implement `legs.py`**

```python
# src/snd/legs.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_legs.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/snd/legs.py tests/test_legs.py
git commit -m "Add leg placement geometry and edge-width clamping (legs.py)"
```

---

## Task 6: Layout (`layout.py`)

**Files:**
- Create: `src/snd/layout.py`
- Test: `tests/test_layout.py`

**Interfaces:**
- Consumes: `quimb.tensor.drawing.get_positions`; a quimb `TensorNetwork`'s `.tensor_map`, `._get_tids_from_tags`, `.gen_site_coos`/`.site_tag` (when present, e.g. PEPS).
- Produces: `compute_positions(tn, kind="auto", fix=None) -> dict[int, tuple[float, float]]` (maps tid → (x, y)).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_layout.py
import pytest
import quimb.tensor as qtn

from snd.layout import compute_positions


def test_chain_layout_orders_tensors_along_x(qtn_chain):
    positions = compute_positions(qtn_chain, kind="chain")

    tids = list(qtn_chain.tensor_map.keys())
    xs = [positions[tid][0] for tid in tids]
    assert xs == sorted(xs)
    assert all(positions[tid][1] == 0.0 for tid in tids)


def test_manual_layout_requires_fix(qtn_chain):
    with pytest.raises(ValueError, match="requires a `fix`"):
        compute_positions(qtn_chain, kind="manual")


def test_manual_layout_returns_fix_dict(qtn_chain):
    fix = {tid: (float(i), 1.0) for i, tid in enumerate(qtn_chain.tensor_map)}

    assert compute_positions(qtn_chain, kind="manual", fix=fix) == fix


def test_auto_layout_returns_a_position_per_tensor(qtn_chain):
    positions = compute_positions(qtn_chain, kind="auto")

    assert set(positions.keys()) == set(qtn_chain.tensor_map.keys())


def test_grid_layout_uses_peps_site_coordinates():
    psi = qtn.PEPS.rand(2, 2, bond_dim=2)

    positions = compute_positions(psi, kind="grid")

    for i, j in psi.gen_site_coos():
        for tid in psi._get_tids_from_tags(psi.site_tag(i, j)):
            assert positions[tid] == (float(j), float(-i))


def test_grid_layout_rejects_non_lattice_network(qtn_chain):
    with pytest.raises(ValueError, match="lattice-shaped"):
        compute_positions(qtn_chain, kind="grid")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_layout.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'snd.layout'`

- [ ] **Step 3: Implement `layout.py`**

```python
# src/snd/layout.py
"""Position computation for a TensorNetwork's tensors. Three lattice-
aware modes (`chain`, `grid`, `manual`) cover the small, hand-tunable
networks this library targets; `auto` defers to quimb's own graph
layout (`get_positions`) for anything else."""

from typing import Dict, Literal, Optional, Tuple

from quimb.tensor.drawing import get_positions

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
    return get_positions(tn, fix=fix, dim=2)


def _chain_positions(tn) -> Positions:
    return {tid: (float(i), 0.0) for i, tid in enumerate(tn.tensor_map)}


def _grid_positions(tn) -> Positions:
    if not (hasattr(tn, "gen_site_coos") and hasattr(tn, "site_tag")):
        raise ValueError(
            "layout kind 'grid' requires a lattice-shaped TensorNetwork "
            "(e.g. a quimb PEPS) exposing gen_site_coos()/site_tag()"
        )
    positions: Positions = {}
    for i, j in tn.gen_site_coos():
        for tid in tn._get_tids_from_tags(tn.site_tag(i, j)):
            positions[tid] = (float(j), float(-i))
    return positions
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_layout.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/snd/layout.py tests/test_layout.py
git commit -m "Add chain/grid/manual/auto layout modes (layout.py)"
```

---

## Task 7: Glyph rendering (`glyphs.py`)

**Files:**
- Create: `src/snd/glyphs.py`
- Test: `tests/test_glyphs.py`

**Interfaces:**
- Consumes: `quimb.schematic.Drawing`; `snd.charges.SectorSpec`; `snd.palette.color_for_charge`.
- Produces: `draw_plain(d, coo, radius, style, shape="circle")`, `draw_wavefunction(d, coo, radius, style)`, `draw_block_diagonal(d, coo, size, sector_spec, row_charges, col_charges, style)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_glyphs.py
import pytest
from quimb.schematic import Drawing

from snd.charges import default_sector_spec
from snd.glyphs import draw_plain, draw_wavefunction, draw_block_diagonal


def test_draw_plain_circle_adds_one_patch():
    d = Drawing()
    n_before = len(d.ax.patches)

    draw_plain(d, (0, 0), radius=0.25, style={})

    assert len(d.ax.patches) == n_before + 1


def test_draw_plain_unknown_shape_raises():
    d = Drawing()

    with pytest.raises(ValueError, match="unknown plain glyph shape"):
        draw_plain(d, (0, 0), radius=0.25, style={}, shape="hexagon")


def test_draw_wavefunction_adds_one_patch():
    d = Drawing()
    n_before = len(d.ax.patches)

    draw_wavefunction(d, (0, 0), radius=0.3, style={})

    assert len(d.ax.patches) == n_before + 1


def test_draw_block_diagonal_adds_one_patch_per_sector_plus_border():
    d = Drawing()
    n_before = len(d.ax.patches)
    spec = default_sector_spec(range(5), range(5), total=4)

    draw_block_diagonal(
        d,
        (0, 0),
        size=1.0,
        sector_spec=spec,
        row_charges=list(range(5)),
        col_charges=list(range(5)),
        style={},
    )

    # 5 sector blocks + 1 outer border shape
    assert len(d.ax.patches) == n_before + 6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_glyphs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'snd.glyphs'`

- [ ] **Step 3: Implement `glyphs.py`**

```python
# src/snd/glyphs.py
"""Node glyph renderers. Three kinds:

- plain: existing simple shapes (circle/triangle), for non-symmetric
  tensors and backward compatibility with the old figures.
- wavefunction: a circle, for tensors with many incoming charge legs
  fusing to one outgoing leg (legs handle radial placement well).
- block-diagonal: a square subdivided into diagonal blocks, one per
  allowed charge sector, colored and labeled by charge — the glyph's
  shape *is* the tensor's sparsity pattern.
"""

from typing import Dict, List, Sequence

from quimb.schematic import Drawing

from .charges import SectorSpec
from .palette import color_for_charge


def draw_plain(d: Drawing, coo, radius: float, style: dict, shape: str = "circle"):
    if shape == "circle":
        d.circle(coo, radius=radius, **style)
    elif shape == "triangle":
        d.regular_polygon(coo, n=3, **style)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_glyphs.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/snd/glyphs.py tests/test_glyphs.py
git commit -m "Add plain/wavefunction/block-diagonal glyph renderers (glyphs.py)"
```

---

## Task 8: `TNFigure` orchestration (`figure.py`)

**Files:**
- Create: `src/snd/figure.py`
- Test: `tests/test_figure.py`

**Interfaces:**
- Consumes: `snd.layout.compute_positions`, `snd.style.StyleSheet`, `snd.legs.place_legs_radially`, `snd.legs.clamp_edge_width`, `snd.charges.{charges_from_symmray, charges_from_spec, ChargeLeg, SectorSpec, default_sector_spec}`, `snd.palette.color_for_charge`, `snd.glyphs.{draw_plain, draw_wavefunction, draw_block_diagonal}`, `quimb.schematic.Drawing`.
- Produces: `TNFigure` class with chainable `.layout(kind, fix=None)`, `.style(target, node=None, **kwargs)`, `.label_legs(selector, labels: dict)`, `.charges(ind, sectors=None, charges=None, dual=False)`, `.draw() -> TNFigure`, `.savefig(path, dpi=300, **kwargs)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_figure.py
import numpy as np
import quimb.tensor as qtn

from snd.charges import ChargeLeg
from snd.figure import TNFigure


def test_draw_plain_chain_produces_one_patch_per_tensor_plus_bonds(qtn_chain):
    fig = TNFigure(qtn_chain).layout(kind="chain")

    fig.draw()

    n_tensors = qtn_chain.num_tensors
    # each plain tensor draws one circle patch; bonds are lines, not patches
    assert len(fig._drawing.ax.patches) == n_tensors


def test_savefig_writes_a_real_svg(qtn_chain, tmp_path):
    fig = TNFigure(qtn_chain).layout(kind="chain")
    out = tmp_path / "out.svg"

    fig.savefig(str(out))

    assert out.exists()
    assert out.read_text(encoding="utf-8").startswith("<?xml")


def test_style_with_node_type_selects_block_diagonal_glyph():
    tn = qtn.TensorNetwork(
        [qtn.Tensor(data=np.random.rand(2, 2), inds=("a", "b"), tags={"T"})]
    )
    # bypass real array data: attach explicit charges instead
    fig = (
        TNFigure(tn)
        .layout(kind="manual", fix={next(iter(tn.tensor_map)): (0.0, 0.0)})
        .style("T", node="block-diagonal")
        .charges("a", charges=range(5))
        .charges("b", charges=range(5))
    )

    fig.draw()

    tid = next(iter(tn.tensor_map))
    assert fig._node_types[tid] == "block-diagonal"


def test_label_legs_stores_labels_by_selector(qtn_chain):
    fig = TNFigure(qtn_chain)

    fig.label_legs("I0", {"k0": "sigma"})

    assert fig._leg_labels["I0"] == {"k0": "sigma"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_figure.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'snd.figure'`

- [ ] **Step 3: Implement `figure.py`**

```python
# src/snd/figure.py
"""TNFigure: the orchestration object. Owns a quimb.schematic.Drawing
end-to-end (unlike quimb's own draw_tn, which builds one internally and
discards it), so vector export always works."""

import math
from typing import Dict, Optional, Sequence

from quimb.schematic import Drawing

from .charges import ChargeLeg, SectorSpec, charges_from_spec, charges_from_symmray, default_sector_spec
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_figure.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/snd/figure.py tests/test_figure.py
git commit -m "Add TNFigure orchestration class (figure.py)"
```

---

## Task 9: Backward-compatible entry point and public exports (`compat.py`, `__init__.py`)

**Files:**
- Create: `src/snd/compat.py`
- Modify: `src/snd/__init__.py`
- Test: `tests/test_compat.py`

**Interfaces:**
- Consumes: `snd.figure.TNFigure`.
- Produces: `quick_draw(tn, layout="auto", figsize=(6, 4), **style_kwargs) -> TNFigure`; public exports `snd.TNFigure`, `snd.quick_draw`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_compat.py
from snd import TNFigure, quick_draw


def test_quick_draw_returns_a_drawn_tnfigure(qtn_chain):
    fig = quick_draw(qtn_chain)

    assert isinstance(fig, TNFigure)
    assert fig._drawing is not None  # already drawn
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_compat.py -v`
Expected: FAIL with `ImportError: cannot import name 'quick_draw' from 'snd'`

- [ ] **Step 3: Implement `compat.py` and update `__init__.py`**

```python
# src/snd/compat.py
"""Backward-compatible, single-call entry point mirroring quimb's own
tn.draw(tn, **kwargs) shape, for quick figures and easy migration from
the old figures.org scripts."""

from .figure import TNFigure


def quick_draw(tn, layout: str = "auto", figsize=(6, 4), **style_kwargs) -> TNFigure:
    fig = TNFigure(tn, figsize=figsize).layout(kind=layout)
    for tag, kwargs in style_kwargs.items():
        fig.style(tag, **kwargs)
    return fig.draw()
```

```python
# src/snd/__init__.py
"""Publication-quality tensor network diagrams from quimb TensorNetwork objects."""

from .figure import TNFigure
from .compat import quick_draw

__all__ = ["TNFigure", "quick_draw"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_compat.py -v`
Expected: PASS

- [ ] **Step 5: Run the full test suite**

Run: `pytest -v`
Expected: PASS (all tests from Tasks 2-9)

- [ ] **Step 6: Commit**

```bash
git add src/snd/compat.py src/snd/__init__.py tests/test_compat.py
git commit -m "Add quick_draw backward-compatible entry point and public exports"
```

---

## Task 10: Examples — plain-glyph networks

**Files:**
- Create: `examples/mps_chain.py`
- Create: `examples/small_circuit.py`
- Create: `examples/peps_patch.py`

**Interfaces:**
- Consumes: `snd.TNFigure`, `quimb.tensor`.
- Produces: `examples/figs/mps_chain.svg`, `examples/figs/small_circuit.svg`, `examples/figs/peps_patch.svg` when run.

- [ ] **Step 1: Write `examples/mps_chain.py`**

```python
# examples/mps_chain.py
"""Small MPS chain, migrated from legacy/figures.org's 'MPS' section."""

import quimb.tensor as qtn

from snd import TNFigure

psi = qtn.MPS_rand_state(L=6, bond_dim=4)

fig = (
    TNFigure(psi, figsize=(8, 2))
    .layout(kind="chain")
    .style("I0", color="bluedark", shape="triangle")
)
for i in range(1, psi.L):
    fig.style(f"I{i}", color="blue", shape="triangle")

fig.draw()
fig.savefig("examples/figs/mps_chain.svg")
fig.savefig("examples/figs/mps_chain.pdf")
```

- [ ] **Step 2: Write `examples/small_circuit.py`**

```python
# examples/small_circuit.py
"""A small 1D chain standing in for a circuit-style diagram: alternating
tagged layers with distinct colors, to exercise tag-based styling on a
non-lattice-specific network."""

import quimb.tensor as qtn

from snd import TNFigure

tn = qtn.MPS_rand_state(L=5, bond_dim=2)
for i, t in enumerate(tn):
    t.add_tag("even" if i % 2 == 0 else "odd")

fig = (
    TNFigure(tn, figsize=(6, 2))
    .layout(kind="chain")
    .style("even", color="orange")
    .style("odd", color="green")
)
fig.draw()
fig.savefig("examples/figs/small_circuit.svg")
```

- [ ] **Step 3: Write `examples/peps_patch.py`**

```python
# examples/peps_patch.py
"""Small PEPS patch using the grid layout mode."""

import quimb.tensor as qtn

from snd import TNFigure

psi = qtn.PEPS.rand(3, 3, bond_dim=3)

fig = TNFigure(psi, figsize=(6, 6)).layout(kind="grid")
for i, j in psi.gen_site_coos():
    fig.style(psi.site_tag(i, j), color="blue")

fig.draw()
fig.savefig("examples/figs/peps_patch.svg")
```

- [ ] **Step 4: Run all three and confirm output files exist**

```bash
python examples/mps_chain.py
python examples/small_circuit.py
python examples/peps_patch.py
ls -la examples/figs/
```

Expected: `mps_chain.svg`, `mps_chain.pdf`, `small_circuit.svg`, `peps_patch.svg` all present and non-empty.

- [ ] **Step 5: Visually inspect each SVG**

Open each file (e.g. `xdg-open examples/figs/mps_chain.svg`) and confirm: tensors are visible circles/triangles positioned as expected (chain left-to-right, grid in a 3x3 layout), bonds connect neighboring tensors, colors match the `.style(...)` calls, no overlapping/garbled text.

- [ ] **Step 6: Commit**

```bash
git add examples/mps_chain.py examples/small_circuit.py examples/peps_patch.py examples/figs/*.svg examples/figs/*.pdf
git commit -m "Add plain-glyph examples: MPS chain, small circuit, PEPS patch"
```

---

## Task 11: Example — abelian-symmetric charge-sector network

**Files:**
- Create: `examples/charge_sector_mps.py`

**Interfaces:**
- Consumes: `snd.TNFigure`, `quimb.tensor`.
- Produces: `examples/figs/charge_sector_mps.svg` when run.

- [ ] **Step 1: Write `examples/charge_sector_mps.py`**

```python
# examples/charge_sector_mps.py
"""Illustrative abelian-symmetric example: a 3-site chain under electron-
number conservation (total charge 4), rendered with the block-diagonal
glyph, plus one wavefunction/fusion tensor combining two incoming charge
legs into the final state. No real symmray data is used -- charge
sectors are specified directly, per the design spec's decision that the
library must not require real block-sparse tensor data to draw one."""

import numpy as np
import quimb.tensor as qtn

from snd import TNFigure

# three "generic" symmetric tensors in a chain, each block-diagonal
# under total charge 4, plus one wavefunction tensor that fuses the
# two boundary bonds into the final state. The dense arrays here are
# arbitrary placeholders -- charge sectors are attached explicitly via
# .charges(...) below, not read from this data.
tn = qtn.TensorNetwork(
    [
        qtn.Tensor(data=np.random.rand(5), inds=("b01",), tags={"T0"}),
        qtn.Tensor(data=np.random.rand(5, 5), inds=("b01", "b12"), tags={"T1"}),
        qtn.Tensor(data=np.random.rand(5), inds=("b12",), tags={"T2"}),
        qtn.Tensor(data=np.random.rand(5, 5), inds=("b01", "b12"), tags={"WF"}),
    ]
)

fig = (
    TNFigure(tn, figsize=(6, 4))
    .layout(
        kind="manual",
        fix={
            tid: coo
            for tid, coo in zip(
                tn.tensor_map,
                [(0.0, 1.0), (1.0, 1.0), (2.0, 1.0), (1.0, -1.0)],
            )
        },
    )
    .style("T0", node="block-diagonal")
    .style("T1", node="block-diagonal")
    .style("T2", node="block-diagonal")
    .style("WF", node="wavefunction", color="orange")
    .charges("b01", sectors=range(5))
    .charges("b12", sectors=range(5))
)

fig.draw()
fig.savefig("examples/figs/charge_sector_mps.svg")
fig.savefig("examples/figs/charge_sector_mps.pdf")
```

- [ ] **Step 2: Run it and confirm output**

```bash
python examples/charge_sector_mps.py
ls -la examples/figs/charge_sector_mps.svg examples/figs/charge_sector_mps.pdf
```

Expected: both files present and non-empty.

- [ ] **Step 3: Visually inspect the SVG**

Open `examples/figs/charge_sector_mps.svg` and confirm: each block-diagonal tensor shows 5 colored diagonal blocks labeled 0-4; the `b01`/`b12` bonds show 5 parallel colored lines (one per charge, matching the block colors on each end) rather than a single line, making the block correspondence across the bond visually obvious; the wavefunction tensor renders as a plain circle.

- [ ] **Step 4: Commit**

```bash
git add examples/charge_sector_mps.py examples/figs/charge_sector_mps.svg examples/figs/charge_sector_mps.pdf
git commit -m "Add abelian-symmetric charge-sector example (electron-number conservation)"
```

---

## Task 12: README rewrite

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: nothing (documentation only).
- Produces: updated `README.md`.

- [ ] **Step 1: Rewrite `README.md`**

```markdown
# SimpleNetworkDrawing (`snd`)

Publication-quality tensor network diagrams (SVG/PDF) drawn directly
from [quimb](https://github.com/v1j4y/quimb) `Tensor`/`TensorNetwork`
objects, with first-class support for block-diagonal abelian-symmetric
tensors and their charge sectors.

## Install

```bash
git clone <this-repo-url>
cd SimpleNetworkDrawing
pip install -e ".[dev]"
```

This installs `quimb` from the `v1j4y/quimb` fork automatically (see
`pyproject.toml`).

## Quickstart

```python
import quimb.tensor as qtn
from snd import TNFigure

psi = qtn.MPS_rand_state(L=6, bond_dim=4)

fig = TNFigure(psi, figsize=(8, 2)).layout(kind="chain")
fig.draw()
fig.savefig("mps.svg")
```

Or, for a one-liner mirroring quimb's own `tn.draw()`:

```python
from snd import quick_draw
quick_draw(psi, layout="chain").savefig("mps.svg")
```

## API reference

- `TNFigure(tn, figsize=(6, 4))` — main entry point, wraps a quimb
  `Tensor`/`TensorNetwork`.
  - `.layout(kind="auto"|"chain"|"grid"|"manual", fix=None)` — position
    the tensors; `"manual"` requires `fix={tid: (x, y), ...}`.
  - `.style(target, node=None, **kwargs)` — style by tag, by
    `"leg:<glob>"` pattern, or by `"type:<node_type>"`; pass
    `node="block-diagonal"` or `node="wavefunction"` to select a
    symmetric-tensor glyph for tensors matching `target`.
  - `.label_legs(selector, {ind: label, ...})` — batch-assign leg
    labels for all tensors matching a tag or tid.
  - `.charges(ind, sectors=[...])` — attach an illustrative charge
    spec to a leg (no real symmray data required); if the tensor's
    `.data` is symmray-backed, real charge data is read automatically
    instead.
  - `.draw()` — render; `.savefig(path, dpi=300)` — export SVG/PDF/PNG.
- `quick_draw(tn, layout="auto", figsize=(6, 4), **style_kwargs)` —
  one-call convenience wrapper.

## Gallery

| MPS chain | PEPS patch | Charge-sector example |
|---|---|---|
| ![](examples/figs/mps_chain.svg) | ![](examples/figs/peps_patch.svg) | ![](examples/figs/charge_sector_mps.svg) |

See `examples/` for the scripts that generate these figures, including
`examples/charge_sector_mps.py` for the abelian-symmetric,
block-diagonal glyph example.

## Design

See `docs/superpowers/specs/2026-08-11-tensor-network-drawing-design.md`
for the full design rationale, including how this library relates to
quimb's own `TensorNetwork.draw()` and `quimb.schematic.Drawing`.

## Legacy

The original hand-coordinate org-mode figure scripts live in `legacy/`
for reference.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "Rewrite README: install, quickstart, API reference, gallery"
```

---

## Plan self-review notes

- **Spec coverage:** §4 architecture → Tasks 1, 8; §5 charge model → Task 2; §6 glyphs → Task 7; §7 leg labeling → Tasks 5, 8; §8 API → Tasks 8, 9; §9 layout/styling reuse → Tasks 4, 6; §10 export/QoL → Tasks 8 (savefig), 12 (README gallery; multi-panel `TNFigureGrid` from spec §10 is deferred — see below); §11 examples → Tasks 10, 11; §12 testing approach → every task's Step 1-4; §13 non-goals → Global Constraints.
- **Deferred from spec:** `TNFigureGrid` multi-panel helper (spec §10) is not included as a task — it's additive and independent of the core rendering path; add as a follow-up task once the core library is in use, rather than blocking this plan on it.
- **Type consistency check:** `TNFigure._node_types`/`_leg_labels`/`_explicit_charges` types match across Tasks 8-11; `ChargeLeg`/`SectorSpec`/`default_sector_spec` signatures from Task 2 are used unchanged in Tasks 7, 8, 11; `compute_positions` signature from Task 6 matches its use in Task 8.
