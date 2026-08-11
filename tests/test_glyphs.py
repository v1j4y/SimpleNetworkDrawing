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
