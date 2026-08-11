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
