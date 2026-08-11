"""Plot the sparsity pattern of a square block-sparse matrix using quimb's
`schematic` drawing tools.

Follows the same pattern as `src/figures.org`: build a `schematic.Drawing`
with named style presets, add shapes/text with those presets, then save
the figure.
"""

import numpy as np
from matplotlib.collections import LineCollection

from quimb import schematic

_block_color = schematic.get_color("blue")

presets = {
    "filled": {"color": _block_color},
    "border": {
        "color": schematic.get_color("blue", alpha=0.15),
        "edgecolor": schematic.darken_color(_block_color),
        "linewidth": 3,
    },
    "leg": {"linewidth": 1.5},
    "leg_hidden": {"linewidth": 1.5, "linestyle": "dashed"},
    "trunk": {"linewidth": 3},
}


def merge_curve(d, p0, p3, width0, width1, color, n=80):
    """Draw a smooth S-shaped curve from `p0` to `p3` (as used to merge a
    single index leg into the combined trunk), with its stroke width
    tapering continuously from `width0` at `p0` to `width1` at `p3`.

    A plain `Line2D`/`PathPatch` can only take one linewidth for its whole
    length, so the taper is built by hand: sample the bezier finely and
    draw it as a `LineCollection` with a per-segment linewidth.
    """
    x0, y0 = p0
    x3, y3 = p3
    xmid = (x0 + x3) / 2
    p1, p2 = (xmid, y0), (xmid, y3)

    t = np.linspace(0, 1, n)
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


def draw_block_sparse_matrix(
    block_sizes, fname, merge_length=6.0, trunk_length=1.5
):
    n_blocks = len(block_sizes)
    total = sum(block_sizes)
    offsets = [0]
    for size in block_sizes:
        offsets.append(offsets[-1] + size)

    d = schematic.Drawing(presets=presets, background=(1, 1, 1, 1))
    y_center = total / 2

    # matrix outline + translucent fill (same color/edge as the blocks),
    # drawn first so the fully opaque diagonal blocks sit on top of it
    d.shape(
        [(0, 0), (total, 0), (total, total), (0, total)],
        preset="border",
    )

    for i in range(n_blocks):
        # flip the row so block 0 is drawn top-left, matching standard
        # matrix layout rather than plot-axis layout
        x0, x1 = offsets[i], offsets[i + 1]
        y0, y1 = total - x1, total - x0
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        d.shape(corners, preset="filled")

    # bond-index legs: index i_k enters block k from the left, index i_{k+1}
    # leaves to the right. Each one stays separate right where it meets the
    # matrix (dashed, running "behind" the other empty blocks in its row to
    # reach the actual diagonal block), and is tagged there before it bends
    # via a smooth curve and merges into a single combined trunk index well
    # outside the matrix. The curve's stroke thickens continuously from the
    # leg width up to the trunk width, so it blends into the trunk with no
    # abrupt jump at the join.
    apex_left = -merge_length
    apex_right = total + merge_length
    leg_width = presets["leg"]["linewidth"]
    trunk_width = presets["trunk"]["linewidth"]

    for i in range(n_blocks):
        size = block_sizes[i]
        x_left, x_right = offsets[i], offsets[i + 1]
        y_mid = total - offsets[i] - size / 2

        merge_curve(
            d,
            (0, y_mid),
            (apex_left, y_center),
            leg_width,
            trunk_width,
            d.drawcolor,
        )
        d.text((-0.3, y_mid + 0.18), f"$i_{{{i}}}$", fontsize=13, va="bottom", ha="right")
        if x_left > 0:
            d.line((0, y_mid), (x_left, y_mid), preset="leg_hidden")

        if x_right < total:
            d.line((x_right, y_mid), (total, y_mid), preset="leg_hidden")
        merge_curve(
            d,
            (total, y_mid),
            (apex_right, y_center),
            leg_width,
            trunk_width,
            d.drawcolor,
        )
        d.text(
            (total + 0.3, y_mid + 0.18),
            f"$i_{{{i + 1}}}$",
            fontsize=13,
            va="bottom",
            ha="left",
        )

    # single combined trunk index on each side
    d.line(
        (apex_left - trunk_length, y_center),
        (apex_left, y_center),
        preset="trunk",
        arrowhead=dict(center=0.5, linewidth=1.5, length=0.2, width=0.08),
    )
    d.text(
        (apex_left - trunk_length, y_center + 0.25), "$I$", fontsize=16, va="bottom"
    )
    d.line(
        (apex_right, y_center),
        (apex_right + trunk_length, y_center),
        preset="trunk",
        arrowhead=dict(center=0.5, linewidth=1.5, length=0.2, width=0.08),
    )
    d.text(
        (apex_right + trunk_length, y_center + 0.25),
        "$J$",
        fontsize=16,
        va="bottom",
    )

    d.text(
        (total / 2, total + 1.0),
        f"block-sparse matrix ({n_blocks} diagonal blocks)",
        fontsize=16,
    )
    # fix physical inches-per-data-unit so text/line sizes (specified in
    # points) stay proportioned correctly regardless of how wide the merge
    # tails make the drawing - otherwise matplotlib's default figsize would
    # squeeze everything to fit and the title would collide with the border
    d.scale_figsize(scale=0.5)
    d.savefig(fname, dpi=600)


if __name__ == "__main__":
    draw_block_sparse_matrix(
        block_sizes=[1, 2, 1, 3, 2], fname="figs/block_sparse_matrix.png"
    )
