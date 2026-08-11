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
