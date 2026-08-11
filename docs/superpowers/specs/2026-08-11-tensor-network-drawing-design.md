# SimpleNetworkDrawing redesign — design spec

Date: 2026-08-11
Status: approved by user, pending implementation plan

## 1. Goal

Turn `SimpleNetworkDrawing` from a collection of hand-coordinate org-mode
scripts into a reusable Python library that draws publication-quality
static figures (SVG/PDF) directly from **quimb `Tensor`/`TensorNetwork`
objects** (the user's quimb fork, `v1j4y/quimb`). The centerpiece
capability is elegant rendering of abelian-symmetric, block-diagonal
tensors and the networks built from them — showing charge sectors as
distinct blocks on the tensor glyph, labeling legs with charge
assignments, and making block-to-block connections across bonds visually
obvious.

The library stays coupled to quimb's data model by design — it is not a
decoupled generic graph-drawing tool. It takes `Tensor`/`TensorNetwork`
instances as input and reads whatever charge/symmetry metadata they
carry (via quimb's existing duck-typed `symmray` integration), falling
back to a lightweight illustrative charge-sector spec when no real
symmetric backend is present.

## 2. Findings from Step 1 (context, not requirements)

- `SimpleNetworkDrawing` currently never calls `TensorNetwork.draw()`.
  It hand-places coordinates using `quimb.schematic.Drawing` primitives
  (`circle`, `line`, `bezier`, `arrowhead`, `text`) in `src/figures.org`.
  There is no reusable function that takes a `TensorNetwork` and
  produces a figure.
- quimb's `draw_tn` (`quimb/tensor/drawing.py`) provides automatic graph
  layout (`get_positions`, spring/kamada-kawai/graphviz), tag-based
  styling (`parse_dict_to_tids_or_inds`), and multi-edge handling, but:
  - has no lattice-aware layouts (MPS chain, PEPS grid) — users must
    manually `fix=` every tensor **and** every dangling index position;
  - has no clean export path — it builds a `schematic.Drawing`
    internally and discards it instead of returning/keeping it, so
    `savefig(dpi=...)` is unreachable through the public API;
  - dangling legs are literal zero-size nodes in the spring simulation,
    not controllable-length stubs;
  - its only symmetry awareness is drawing the `+`/`-`
    `tensor.data.signature` character at each bond endpoint — it never
    reads block/sector data.
- Real symmetric tensor data model (confirmed by installing `symmray`
  0.2.1 and inspecting a live `U1Array`), reachable today only via
  duck-typing (`hasattr(data, "align_axes")` etc., see
  `quimb/tensor/array_ops.py:70-91`):
  - `tensor.data.indices` — tuple of `BlockIndex`, one per axis; each
    has `.chargemap` (`{charge: dim}`) and `.dual` (bool, arrow
    direction).
  - `tensor.data.duals` — parallel tuple of the same bools.
  - `tensor.data.blocks` — `{sector_tuple: dense_ndarray}`.
  - `tensor.data.sectors` — tuple of allowed sector tuples (the keys of
    `.blocks`).
  - `tensor.data.charge` — overall conserved charge (fusion target).
  - There is **no per-index name → charge map** on `Tensor` itself;
    everything is positional (`t.inds.index(ix)` into the arrays above).
- Per user direction, the library must **not require real symmray data**
  for its examples/tests: an illustrative charge spec (e.g. total
  electron number 4, sectors `(0,4),(1,3),(2,2),(3,1),(4,0)`) is
  sufficient and is the primary path to design against. Real
  symmray-backed tensors are read opportunistically when present, using
  the same rendering code.

## 3. User decisions (from Step 2 dialogue)

| Topic | Decision |
|---|---|
| Color palette | Keep quimb's existing Okabe-Ito colorblind-safe palette, for both tag coloring and charge-value coloring. |
| Tag-based styling | Remains central to the API, as in quimb's `draw_tn`. |
| Many-leg tensors | First-class concern: must be easy to name/tag many legs at once, and legs must render without clutter. |
| Leg/charge labels | Placed at the node (near leg attachment point), styled/spaced to stay uncluttered — not moved to a side legend. |
| Typical network size | Mostly small, hand-tunable networks (< ~20 tensors). No requirement for automatic large-lattice (100+ tensor) layout. |
| Wavefunction/fusion tensors | Tensors with multiple incoming charge legs that fuse to a single outgoing wavefunction leg get a **distinct glyph: a circle** (chosen specifically because circles handle many radially-arranged legs better than a grid shape). |
| Generic symmetric tensors | Rendered with a **block-diagonal grid glyph** — a square divided into diagonal blocks by charge sector, i.e. literally drawn as a block-diagonal matrix. This is the current biggest gap in both SimpleNetworkDrawing and quimb. |
| Pain points to fix | (a) tagging/naming many-index tensors is currently tedious; (b) edge/line-thickness scaling makes many-leg tensors look bad; (c) no tool exists at all for the block-diagonal glyph. |
| Charge sector count | Typically 5-10 sectors per leg. |
| Example data | No real data needed; use an illustrative/synthetic charge spec (not `symmray`'s random generators, not real project data) — e.g. the electron-number-conservation example with sectors `(0,4),(1,3),(2,2),(3,1),(4,0)`. |

## 4. Architecture

`SimpleNetworkDrawing` sits between quimb's two existing layers and
reuses rather than reimplements where it can:

- **Reused from `draw_tn`**: layout math (`get_positions`,
  `layout_networkx`, `fix=` pinning) as an optional starting point;
  the tag→style dispatch pattern (`parse_dict_to_tids_or_inds`).
- **Reused from `quimb.schematic`**: the `Drawing` primitive canvas
  (`circle`, `shape`, `line`, `bezier`, `arrowhead`, `text_between`,
  presets, pseudo-3D projection, `savefig`). Critically, the library
  owns and keeps its `Drawing` instance (unlike `draw_tn`, which
  discards it), so vector export works end-to-end.
- **New in this library**:
  - a charge-sector reader/spec layer (Section 5),
  - two new node glyphs — block-diagonal grid and wavefunction circle
    (Section 6),
  - a leg-labeling system designed for high-degree tensors
    (Section 7),
  - an object-oriented `TNFigure` API (Section 8).

## 5. Charge-sector data model

A small internal abstraction, `ChargeLeg`, decouples the renderer from
requiring real symmray data:

- If `tensor.data` looks symmray-backed (duck-typed, mirroring quimb's
  own `isblocksparse()` check), read `.indices[i].chargemap`,
  `.sectors`, `.duals`, `.charge` directly from the real object — no
  parallel data model, no duplication.
- Otherwise, sectors can be attached explicitly and only for drawing
  purposes: `fig.charges(ind_name, sectors=[(0, 4), (1, 3), (2, 2),
  (3, 1), (4, 0)])`, or per-leg charge lists for tensors not tied to any
  particular fusion pairing. This is the primary path used in examples
  and tests, per Section 3.
- Both paths feed the identical rendering code — real and illustrative
  tensors are visually indistinguishable in kind, only in data source.

## 6. Node glyphs

1. **Block-diagonal grid glyph** (generic symmetric tensor): an outer
   square subdivided into diagonal blocks, one per allowed charge
   sector. Each block is colored by its charge value (consistent hash
   → color across the whole figure, using quimb's Okabe-Ito /
   `hash_to_color` machinery) and annotated with a small numeric charge
   label. Disallowed (off-diagonal) combinations are simply absent —
   the glyph's shape *is* the sparsity pattern. Legs attach at the
   block matching their charge, so when two such tensors are connected
   by a bond, matching charge blocks visually line up across the
   connection. For tensors with more than two legs, legs are grouped
   into a "row" side and a "column" side (default: quimb's `left_inds`
   vs remaining inds, overridable) — the standard row/column fusion
   used to express any symmetric tensor as a block-diagonal matrix.
2. **Wavefunction/fusion glyph** (tensors with multiple incoming charge
   legs fusing to one outgoing leg): a circle, with legs placed
   radially at even angular spacing (spacing/font-size adapt to leg
   count so many-leg tensors stay legible).
3. **Plain glyphs** (non-symmetric tensors): existing simple shapes
   (circle/triangle), kept for continuity with the current figures and
   for backward compatibility.

## 7. Leg / tag labeling

Directly targets the two ergonomic pain points called out in Step 2:

- **Naming many legs at once**: a single call sets labels for a batch
  of legs (`fig.label_legs(tensor, {ind: label, ...})`, or positional
  shorthand with an auto-numbered prefix) instead of one manual text
  call per leg.
- **Uncluttered placement**: labels sit just outside the node at the
  leg's attachment angle; font size and angular spacing auto-adjust
  once leg count crosses a threshold, replacing quimb's fixed
  bond-midpoint text placement.
- **Line thickness**: default edge width is log-scaled but **clamped**
  to a legible range regardless of bond dimension, instead of raw
  `edge_scale * log2(bond_dim)`; always overridable per tag/per leg.

## 8. Public API sketch

```python
from snd import TNFigure

fig = TNFigure(tn, figsize=(6, 4))            # tn: quimb Tensor or TensorNetwork
fig.layout(kind="chain")                       # "chain" | "grid" | "auto" | "manual"
fig.style(tag="A", color="bluedark", shape="triangle")
fig.style(tag="wf", node="wavefunction")       # -> circle glyph
fig.label_legs("T0", {"k": "sigma", "b01": "chi"})
fig.charges("b01", sectors=[(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)])
fig.draw()
fig.savefig("out.svg")                         # also .pdf/.png, dpi=300 default
```

Backward-compatible entry point: a `quick_draw(tn, **kwargs)` function
mirroring `tn.draw()`'s call signature, for easy migration from
existing scripts and for very simple one-off figures.

## 9. Layout & styling reuse

- Manual coordinate control (primary use case, given <20-tensor typical
  networks) via a `fix=`-style API identical in spirit to `draw_tn`'s,
  plus direct primitive access through the underlying `Drawing` for
  full hand-tuning where wanted.
- Tag-based styling stays central, extended so style rules can also
  target leg-name patterns and node type (`plain` / `block-diagonal` /
  `wavefunction`), not just tensor tags.
- Palette: quimb's Okabe-Ito colorblind-safe set throughout, reused for
  both tag colors and charge-value colors.

## 10. Export & QoL

- `fig.savefig(path, dpi=300)` for SVG/PDF/PNG via the retained
  `Drawing.savefig`, print resolution by default.
- Optional auto-legend for tag colors and/or charge-value colors.
- `TNFigureGrid` thin helper for side-by-side multi-panel comparisons,
  built on matplotlib subplots with one `Drawing(ax=...)` per panel
  (already supported by `schematic.Drawing`).

## 11. Examples to build (Step 3/4)

- Small MPS chain (plain glyphs, migrated from existing `figures.org`
  content).
- Small PEPS patch.
- A small circuit diagram.
- At least one abelian-symmetric example using the illustrative
  electron-number-conservation charge spec (sectors `(0,4) .. (4,0)`),
  exercising both the block-diagonal glyph and, if applicable, the
  wavefunction/fusion glyph, composed into a small network so
  block-to-block bond connections are visible.

## 12. Testing approach

Basic tests on layout/API logic (charge-spec parsing, tag→style
dispatch, leg-label batch assignment, glyph selection rules) — not
pixel-perfect image comparison. Example scripts are rendered and
visually inspected as the correctness check for figure quality.

## 13. Non-goals

- No dependency on `symmray` being installed for core functionality or
  examples — it is read opportunistically when present on real tensor
  data.
- No automatic large-lattice (100+ tensor) layout engine — out of scope
  given the stated typical network size.
- Not a general-purpose graph-drawing library — stays coupled to quimb
  `Tensor`/`TensorNetwork` as the input type.
