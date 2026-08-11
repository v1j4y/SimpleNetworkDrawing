"""Plot the sparsity pattern of a square block-sparse matrix using quimb's
`schematic` drawing tools, optionally combined with projector tensors that
merge the per-block indices down to a single index, dropping ("projecting
out") whichever blocks are not kept.

Follows the same pattern as `src/figures.org`: build a `schematic.Drawing`
with named style presets, add shapes/text with those presets, then save
the figure.
"""

import matplotlib.colors as mcolors
import matplotlib.patheffects as patheffects
import numpy as np
from matplotlib.collections import LineCollection

_TEXT_HALO = [patheffects.withStroke(linewidth=3, foreground="white")]

from quimb import schematic

_block_color = schematic.get_color("orange")
_dropped_color = (0.6, 0.6, 0.6, 0.9)

presets = {
    "filled": {"color": _block_color},
    "border": {
        "color": schematic.get_color("orange", alpha=0.15),
        "edgecolor": schematic.darken_color(_block_color),
        "linewidth": 3,
    },
    "leg": {"linewidth": 1.5},
    "leg_hidden": {"linewidth": 1.5, "linestyle": "dashed"},
    "trunk": {"linewidth": 3},
    "projector": {"color": schematic.get_color("orange"), "radius": 0.6},
}


def save_figure(d, fname, dpi=600):
    """Save `d` as both a raster PNG and a vector PDF of equivalent
    quality, regardless of which extension `fname` was given - e.g.
    `save_figure(d, "figs/foo.png")` writes both `figs/foo.png` (at `dpi`)
    and `figs/foo.pdf`. Both go through `Drawing.savefig`, so they share
    the same `bbox_inches="tight"` cropping.
    """
    stem = fname[:-4] if fname.lower().endswith((".png", ".pdf")) else fname
    for ext in ("png", "pdf"):
        d.savefig(f"{stem}.{ext}", dpi=dpi)


def _bezier_xy(p0, p3, t):
    """Point at parameter `t` along the same smooth S-curve `merge_curve`
    draws from `p0` to `p3` (local, unprojected coordinates - `t=0` is
    `p0`, `t=1` is `p3`).
    """
    x0, y0 = p0
    x3, y3 = p3
    xmid = (x0 + x3) / 2
    p1, p2 = (xmid, y0), (xmid, y3)
    x = (1 - t) ** 3 * x0 + 3 * (1 - t) ** 2 * t * p1[0] + 3 * (1 - t) * t**2 * p2[0] + t**3 * x3
    y = (1 - t) ** 3 * y0 + 3 * (1 - t) ** 2 * t * p1[1] + 3 * (1 - t) * t**2 * p2[1] + t**3 * y3
    return x, y


def merge_curve(d, p0, p3, width0, width1, color, n=80, t_max=1.0):
    """Draw a smooth S-shaped curve from `p0` towards `p3` (as used to feed
    a single index leg into a merge point or a projector), stopping early
    at `t_max < 1` to leave it dangling part way there. Its stroke width
    tapers continuously from `width0` at `p0` towards `width1` at `p3`
    (reached only if `t_max == 1`).

    A plain `Line2D`/`PathPatch` can only take one linewidth for its whole
    length, so the taper is built by hand: sample the bezier finely and
    draw it as a `LineCollection` with a per-segment linewidth.

    Returns the curve's final (x, y) point (including `d`'s current
    `translate()` offset, if any).
    """
    ox, oy, _ = d._offset
    x0, y0 = p0[0] + ox, p0[1] + oy
    x3, y3 = p3[0] + ox, p3[1] + oy
    xmid = (x0 + x3) / 2
    p1, p2 = (xmid, y0), (xmid, y3)

    t = np.linspace(0, t_max, n)
    x = (
        (1 - t) ** 3 * x0
        + 3 * (1 - t) ** 2 * t * p1[0]
        + 3 * (1 - t) * t**2 * p2[0]
        + t**3 * x3
    )
    y = (
        (1 - t) ** 3 * y0
        + 3 * (1 - t) ** 2 * t * p1[1]
        + 3 * (1 - t) * t**2 * p2[1]
        + t**3 * y3
    )
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # ease-in-out (smoothstep) the width, not just the position, so the
    # thickening itself feels smooth rather than linear/abrupt
    tmid = (t[:-1] + t[1:]) / 2
    smooth_t = tmid**2 * (3 - 2 * tmid)
    widths = width0 + (width1 - width0) * smooth_t

    lc = LineCollection(
        segments,
        linewidths=widths,
        colors=[color] * len(segments),
        capstyle="round",
        joinstyle="round",
        zorder=0,
    )
    d.ax.add_collection(lc)
    d._adjust_lims(x0, y0)
    d._adjust_lims(x3, y3)
    return x[-1], y[-1]


def _new_drawing(block_sizes, d=None):
    """Set up (or reuse) a `Drawing` with the matrix outline and the
    diagonal blocks already drawn on it. Returns `(d, offsets, total,
    y_center)`.
    """
    total = sum(block_sizes)
    offsets = [0]
    for size in block_sizes:
        offsets.append(offsets[-1] + size)

    if d is None:
        d = schematic.Drawing(presets=presets, background=(1, 1, 1, 1))
    y_center = total / 2

    # matrix outline + translucent fill (same color/edge as the blocks),
    # drawn first so the fully opaque diagonal blocks sit on top of it
    d.shape([(0, 0), (total, 0), (total, total), (0, total)], preset="border")

    for i in range(len(block_sizes)):
        # flip the row so block 0 is drawn top-left, matching standard
        # matrix layout rather than plot-axis layout
        x0, x1 = offsets[i], offsets[i + 1]
        y0, y1 = total - x1, total - x0
        d.shape([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], preset="filled")

    return d, offsets, total, y_center


def _draw_uncombined_legs(d, block_sizes, offsets, total):
    """For every block, draw the dashed leg running from the matrix edge to
    the block (since it passes "behind" the other, empty blocks in its
    row). Returns the list of each block's row mid-height, in block order.
    """
    y_mids = []
    for i, size in enumerate(block_sizes):
        x_left, x_right = offsets[i], offsets[i + 1]
        y_mid = total - offsets[i] - size / 2
        y_mids.append(y_mid)

        if x_left > 0:
            d.line((0, y_mid), (x_left, y_mid), preset="leg_hidden")
        if x_right < total:
            d.line((x_right, y_mid), (total, y_mid), preset="leg_hidden")
    return y_mids


def _tag_leg(d, text, p0, p3, side, t=0.22, clearance=0.13):
    """Tag an index leg with `text`, anchored to a point a short way along
    its actual merge curve (rather than a fixed offset from the matrix
    edge) so it sits close to the line at any curve scale, nudged
    `clearance` further from the curve so it doesn't sit on the stroke.
    `side` is "left" or "right" (which way the tag reads outward).
    """
    x, y = _bezier_xy(p0, p3, t)
    d.text(
        (x, y + clearance),
        text,
        fontsize=13,
        va="bottom",
        ha="right" if side == "left" else "left",
        zorder=10,
        path_effects=_TEXT_HALO,
    )


def _default_index_labels(n_blocks):
    """The `i_0, i_1, ..., i_{2 * n_blocks - 1}` per-block left/right index
    labels used unless a caller supplies their own via `index_labels`.
    Block `k`'s left tag is `index_labels[2 * k]`, its right tag is
    `index_labels[2 * k + 1]` - every block gets its own pair, not shared
    with its neighbours.
    """
    return [f"$i_{{{k}}}$" for k in range(2 * n_blocks)]


def _draw_trunk(d, tip, outward_x, y_center, label, pointing="out"):
    end = (outward_x, y_center)
    # `d.line(..., arrowhead=...)` internally projects cooa/coob and then
    # passes those *already-projected* points on to `d.arrowhead()`, which
    # projects them again - under a nonzero `d.translate()` offset (as used
    # when embedding this into a larger figure) that double-applies the
    # offset and puts the arrowhead in the wrong place. Draw the line and
    # arrowhead as two separate calls, both in local coordinates, so each
    # is projected (and offset) exactly once.
    d.line(tip, end, preset="trunk")
    ah_from, ah_to = (tip, end) if pointing == "out" else (end, tip)
    d.arrowhead(
        ah_from, ah_to, preset="trunk", center=0.5, linewidth=1.5, length=0.2,
        width=0.08,
    )
    d.text((outward_x, y_center + 0.25), label, fontsize=16, va="bottom")


def draw_block_sparse_matrix(
    block_sizes,
    fname=None,
    merge_length=6.0,
    trunk_length=1.5,
    d=None,
    origin=(0, 0),
    show_title=True,
    trunk_pointing="out",
    index_labels=None,
    combined_labels=("$I$", "$J$"),
):
    """Plot the block-sparse matrix alone: each block's index fans out and
    smoothly merges into a single combined index `I` (left) / `J` (right).

    Pass an existing `d` (a `schematic.Drawing`) and `origin` to embed this
    diagram into a larger figure at that offset instead of creating (and
    saving) a standalone one - in that case `fname` is ignored and nothing
    is saved; the caller does that itself once the whole figure is built.

    `trunk_pointing` sets the `I`/`J` arrowheads' direction: "out" points
    them away from the matrix (the default), "in" points them into it.

    `index_labels` overrides the per-block tags: a sequence of
    `2 * len(block_sizes)` strings, one pair per block rather than one
    shared per boundary - block `k`'s left/right tags are
    `index_labels[2 * k]`/`index_labels[2 * k + 1]`, all distinct even
    between neighbouring blocks. Defaults to that same `$i_k$` naming.
    `combined_labels` overrides the merged `(I, J)` tags.
    """
    owns_d = d is None
    if owns_d:
        d = schematic.Drawing(presets=presets, background=(1, 1, 1, 1))

    if index_labels is None:
        index_labels = _default_index_labels(len(block_sizes))
    label_left, label_right = combined_labels

    with d.translate(*origin):
        d, offsets, total, y_center = _new_drawing(block_sizes, d=d)
        y_mids = _draw_uncombined_legs(d, block_sizes, offsets, total)

        apex_left = -merge_length
        apex_right = total + merge_length
        leg_width = presets["leg"]["linewidth"]
        trunk_width = presets["trunk"]["linewidth"]

        for i, y_mid in enumerate(y_mids):
            p0_left, p3_left = (0, y_mid), (apex_left, y_center)
            merge_curve(d, p0_left, p3_left, leg_width, trunk_width, d.drawcolor)
            _tag_leg(d, index_labels[2 * i], p0_left, p3_left, side="left")

            p0_right, p3_right = (total, y_mid), (apex_right, y_center)
            merge_curve(d, p0_right, p3_right, leg_width, trunk_width, d.drawcolor)
            _tag_leg(d, index_labels[2 * i + 1], p0_right, p3_right, side="right")

        _draw_trunk(
            d, (apex_left, y_center), apex_left - trunk_length, y_center, label_left,
            pointing=trunk_pointing,
        )
        _draw_trunk(
            d, (apex_right, y_center), apex_right + trunk_length, y_center,
            label_right, pointing=trunk_pointing,
        )

        if show_title:
            d.text(
                (total / 2, total + 1.0),
                f"block-sparse matrix ({len(block_sizes)} diagonal blocks)",
                fontsize=16,
            )

    if owns_d:
        # fix physical inches-per-data-unit so text/line sizes (specified
        # in points) stay proportioned correctly regardless of how wide the
        # merge tails make the drawing - otherwise matplotlib's default
        # figsize would squeeze everything to fit and the title would
        # collide with the border
        d.scale_figsize(scale=0.5)
        if fname:
            save_figure(d, fname)
    return d


def draw_block_sparse_matrix_with_projectors(
    block_sizes,
    fname,
    keep_mask=None,
    merge_length=6.0,
    trunk_length=1.5,
    index_labels=None,
    combined_labels=("$I$", "$J$"),
):
    """Plot the block-sparse matrix with a projector tensor (a triangle,
    its main vertex pointing outwards) on each side, combining the kept
    blocks' indices into a single index and dropping the rest. Dropped
    indices are drawn as short, faded, dead-end legs marked with a red
    cross.

    `index_labels` and `combined_labels` override the tag text - see
    `draw_block_sparse_matrix`.
    """
    n_blocks = len(block_sizes)
    if keep_mask is None:
        keep_mask = [True] * n_blocks
    if index_labels is None:
        index_labels = _default_index_labels(n_blocks)
    label_left, label_right = combined_labels

    d, offsets, total, y_center = _new_drawing(block_sizes)
    y_mids = _draw_uncombined_legs(d, block_sizes, offsets, total)

    # projector tensors: triangles that combine the per-block indices into
    # a single index, keeping only the blocks in `keep_mask`. The main
    # vertex points outwards (left on the left side, right on the right),
    # matching the "isometry" triangles used elsewhere in this repo
    # (src/figures.org's "Tensor n-gon left/right" figures).
    proj_r = presets["projector"]["radius"]
    apex_left = -merge_length
    apex_right = total + merge_length

    left_tip = (apex_left - proj_r, y_center)
    left_base_x = apex_left + 0.5 * proj_r
    right_tip = (apex_right + proj_r, y_center)
    right_base_x = apex_right - 0.5 * proj_r
    base_half_height = 0.8 * (0.866 * proj_r)

    kept = [i for i in range(n_blocks) if keep_mask[i]]
    if len(kept) > 1:
        base_ys = np.linspace(
            y_center + base_half_height, y_center - base_half_height, len(kept)
        )
    else:
        base_ys = [y_center] * len(kept)
    base_y_of = dict(zip(kept, base_ys))

    leg_width = presets["leg"]["linewidth"]
    trunk_width = presets["trunk"]["linewidth"]

    for i, y_mid in enumerate(y_mids):
        for x_edge, base_x, apex_pt, side, i_label in (
            (0, left_base_x, apex_left, "left", index_labels[2 * i]),
            (total, right_base_x, apex_right, "right", index_labels[2 * i + 1]),
        ):
            p0 = (x_edge, y_mid)
            if keep_mask[i]:
                p3 = (base_x, base_y_of[i])
                merge_curve(d, p0, p3, leg_width, trunk_width, d.drawcolor)
            else:
                p3 = (apex_pt, y_center)
                end = merge_curve(
                    d, p0, p3, leg_width * 0.7,
                    leg_width * 0.7, _dropped_color, n=40, t_max=0.45,
                )
                d.cross(end, radius=0.09, color="red", linewidth=1.5)
            _tag_leg(d, i_label, p0, p3, side=side)

    d.regular_polygon(
        (apex_left, y_center), n=3, orientation=np.pi / 2, preset="projector"
    )
    d.regular_polygon(
        (apex_right, y_center), n=3, orientation=-np.pi / 2, preset="projector"
    )

    _draw_trunk(d, left_tip, left_tip[0] - trunk_length, y_center, label_left)
    _draw_trunk(d, right_tip, right_tip[0] + trunk_length, y_center, label_right)

    d.text(
        (total / 2, total + 1.0),
        f"block-sparse matrix ({n_blocks} diagonal blocks, "
        f"{n_blocks - len(kept)} projected out)",
        fontsize=16,
    )
    d.scale_figsize(scale=0.5)
    save_figure(d, fname)


def draw_simple_matrix(
    d,
    coo,
    size=0.6,
    leg_length=0.6,
    pointing="out",
    color=None,
    leg_color=None,
    fill_alpha=0.15,
    border_linewidth=2.0,
    leg_linewidth=1.5,
    labels=None,
):
    """Draw the simplest possible block-sparse matrix icon: a plain square
    (no inner blocks) with two straight horizontal index legs. Meant for
    embedding a bare matrix/operator symbol into a larger diagram, so it
    doesn't depend on any named presets registered on `d` - just plain
    style kwargs, self-contained.

    `pointing` controls the legs' arrowheads: "out" points them away from
    the square (matches this module's I/J trunk convention), "in" points
    them into the square (matches the bond-flows-into-center convention
    used for the mixed-canonical MPS center in `figures.org`), or `None`
    for plain unarrowed lines.

    `color` sets the square's base color - any matplotlib color spec. The
    fill is this color at `fill_alpha`; the edge uses a darkened version
    of it. Defaults to the same blue used elsewhere in this module.

    `leg_color` sets the two index legs' color independently (e.g. to
    match a diagram's other bond lines instead of the square's own edge).
    Defaults to that same darkened edge color.

    `labels` optionally tags the two legs: a `(left, right)` pair of
    strings placed above each leg's outer tip. Defaults to `None` (no
    tags), since callers often label this from outside (e.g. a `$\\Lambda$`
    above the whole thing).
    """
    cx, cy = coo
    hs = size / 2
    base = mcolors.to_rgb(color) if color is not None else _block_color
    fill = (*base, fill_alpha)
    edge = schematic.darken_color(base)
    leg_color = edge if leg_color is None else leg_color

    d.shape(
        [
            (cx - hs, cy - hs),
            (cx + hs, cy - hs),
            (cx + hs, cy + hs),
            (cx - hs, cy + hs),
        ],
        color=fill,
        edgecolor=edge,
        linewidth=border_linewidth,
    )

    left = ((cx - hs - leg_length, cy), (cx - hs, cy))
    right = ((cx + hs, cy), (cx + hs + leg_length, cy))
    if pointing == "out":
        left_ah, right_ah = dict(center=0.5, reverse=True), dict(center=0.5)
    elif pointing == "in":
        left_ah, right_ah = dict(center=0.5), dict(center=0.5, reverse=True)
    else:
        left_ah = right_ah = None

    d.line(*left, color=leg_color, linewidth=leg_linewidth, arrowhead=left_ah)
    d.line(*right, color=leg_color, linewidth=leg_linewidth, arrowhead=right_ah)

    if labels is not None:
        label_left, label_right = labels
        d.text(
            (left[0][0], cy + 0.15), label_left, fontsize=13, va="bottom", ha="center"
        )
        d.text(
            (right[1][0], cy + 0.15), label_right, fontsize=13, va="bottom",
            ha="center",
        )


def draw_block_sparse_matrix_simple(fname, size=1.5, leg_length=1.0, pointing="out"):
    """Standalone rendering of `draw_simple_matrix` alone: a bare matrix
    with just two (uncombined) indices and no inner block structure.
    """
    d = schematic.Drawing(background=(1, 1, 1, 1))
    draw_simple_matrix(d, (0, 0), size=size, leg_length=leg_length, pointing=pointing)
    d.scale_figsize(scale=1.0)
    save_figure(d, fname)


_mps_presets = {
    "bond": {"linewidth": 3},
    "phys": {"linewidth": 1.5},
    "left": {"color": schematic.get_color("bluedark")},
    "right": {"color": schematic.get_color("blue")},
}


def draw_mps_mixed_canonical(n_left, n_right, fname="figs/paper1.1.png", matrix_size=0.6):
    """Draw a mixed-canonical MPS: `n_left` left-orthogonal ("A") tensors,
    a central bond matrix (drawn with `draw_simple_matrix`, replacing the
    old diamond marker), and `n_right` right-orthogonal ("B") tensors.

    This generalizes the hardcoded `** MPS` figure in `figures.org` (which
    used `n_left = n_right = 2`) to any chain length.
    """
    d = schematic.Drawing(presets=_mps_presets)
    center = n_left

    for i in range(n_left):
        d.regular_polygon((i, 0), n=3, orientation=-np.pi / 2, preset="left")
        d.text((i, -0.8), f"$A^{{\\sigma_{i + 1}}}$\n", fontsize=22)
        cooa = (i, 0.1)
        anca = (i - 1, 0.1)
        ancb = (i - 1, 0.1)
        coob = (i - 1, 1)
        d.text(coob, f"$\\sigma_{i + 1}$\n", fontsize=22)
        d.bezier([cooa, anca, ancb, coob], linewidth=3)
        if i != 0:
            d.line((i, 0.0), (i - 1, 0.0), arrowhead=dict(center=0.7, reverse=True))
        d.arrowhead(ancb, cooa, center=0.7)

    for i in range(center + 1, center + 1 + n_right):
        d.regular_polygon((i, 0), n=3, orientation=np.pi / 2, preset="right")
        d.text((i, -0.8), f"$B^{{\\sigma_{i}}}$\n", fontsize=22)
        cooa = (i, 0.1)
        anca = (i + 1, 0.1)
        ancb = (i + 1, 0.1)
        coob = (i + 1, 1)
        d.bezier([cooa, anca, ancb, coob], linewidth=3)
        d.text(coob, f"$\\sigma_{i}$\n", fontsize=22)
        d.arrowhead(ancb, cooa, center=0.7)
        if i != center + n_right:
            d.line((i, 0.0), (i + 1, 0.0), arrowhead=dict(center=0.5, reverse=True))

    d.text((center, -0.6), "$\\Lambda$", fontsize=20)
    draw_simple_matrix(
        d,
        (center, 0),
        size=matrix_size,
        leg_length=1 - matrix_size / 2,
        pointing="in",
        color=schematic.get_color("orange"),
        leg_color=d.drawcolor,
    )

    if fname:
        save_figure(d, fname)
    return d


def draw_symmetry_equivalence(
    fname="figs/paper1.3.0.png",
    block_sizes=(1, 1, 1),
    merge_length=1.0,
    trunk_length=2.5,
    matrix_size=0.5,
):
    """Draw the "block-sparse matrix = simple matrix" equivalence: the full,
    per-block-resolved `draw_block_sparse_matrix` on the left and the bare
    two-index `draw_simple_matrix` on the right, joined by a "<=>". This
    replaces the old `** Symmetry` figure in `figures.org`, which drew the
    same equivalence using a diamond marker with hand-drawn multi-edges on
    the left and single edges on the right.

    `trunk_length` needs to be generous here (much more so than a plain
    `draw_block_sparse_matrix` call) because with few blocks - as used in
    this small equivalence diagram - and long sector-style `index_labels`,
    a short combined-index line leaves the `(N, M)`-style combined tag
    sitting right on top of the individual block tags near the merge
    point; a longer trunk pushes it clear.
    """
    total = sum(block_sizes)
    y_center = total / 2

    d = schematic.Drawing(presets=presets, background=(1, 1, 1, 1))

    left_origin = (-total - merge_length - trunk_length - 1.4, -y_center)
    draw_block_sparse_matrix(
        block_sizes,
        d=d,
        origin=left_origin,
        merge_length=merge_length,
        trunk_length=trunk_length,
        show_title=False,
        trunk_pointing="in",
        index_labels=[
            f"$(N_{1}^A,M_{1}^A)$",
             f"$(N_{1}^B,M_{1}^B)$",
            f"$(N_{2}^A,M_{2}^A)$",
             f"$(N_{2}^B,M_{2}^B)$",
            f"$(N_{3}^A,M_{3}^A)$",
             f"$(N_{3}^B,M_{3}^B)$",
        ],
        combined_labels=("$(N_A,M_A)$", "$(N_B,M_B)$"),
    )
    d.text(
        (left_origin[0] + total / 2, -y_center + total + 0.5),
        "$\\Lambda$",
        fontsize=20,
    )

    d.text((0, 0), "$\\Leftrightarrow$", fontsize=24)

    right_x = 2.2
    draw_simple_matrix(
        d,
        (right_x, 0),
        size=matrix_size,
        leg_length=1.0,
        pointing="in",
        leg_color=d.drawcolor,
    )
    d.text((right_x, matrix_size / 2 + 0.5), "$\\Lambda$", fontsize=20)

    d.scale_figsize(scale=0.7)
    if fname:
        save_figure(d, fname)
    return d


if __name__ == "__main__":
    draw_block_sparse_matrix(
        block_sizes=[1, 2, 1, 3, 2], fname="figs/block_sparse_matrix.png"
    )
    draw_block_sparse_matrix_with_projectors(
        block_sizes=[1, 2, 1, 3, 2],
        keep_mask=[True, True, False, True, True],
        fname="figs/block_sparse_matrix_projectors.png",
    )
    draw_block_sparse_matrix_simple(fname="figs/block_sparse_matrix_simple.png")
    draw_mps_mixed_canonical(2, 2, fname="figs/paper1.1.png")
    draw_symmetry_equivalence(fname="figs/paper1.3.0.png")

    # example: custom index_labels/combined_labels, e.g. tagging each
    # block's own left/right index by its (N, M) symmetry sector instead
    # of a bare i_k - each block gets a unique pair, not shared with its
    # neighbours, so this needs 2 * len(example_block_sizes) labels
    example_block_sizes = [1, 2, 1, 3, 2]
    draw_block_sparse_matrix(
        example_block_sizes,
        fname="figs/block_sparse_matrix_symmetry_labels.png",
        index_labels=[
            f"$(N_{{{k}}},M_{{{k}}})$" for k in range(2 * len(example_block_sizes))
        ],
        combined_labels=("$(N,M)$", "$(N',M')$"),
    )
