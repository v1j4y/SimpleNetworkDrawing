import pytest
import quimb.tensor as qtn


@pytest.fixture
def qtn_chain():
    """A plain 3-tensor MPS-like TensorNetwork with no symmetry, for layout/style tests."""
    return qtn.MPS_rand_state(L=3, bond_dim=4)
