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
