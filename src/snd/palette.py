"""Thin wrapper over quimb's existing colorblind-safe (Okabe-Ito) color
machinery, reused for both tag coloring and charge-value coloring so a
given charge value is visually consistent across a whole figure."""

import hashlib

from quimb.schematic import get_color

_PALETTE = ("blue", "orange", "green", "red", "yellow", "pink", "bluedark")


def _hash_index(key: str, n: int) -> int:
    digest = hashlib.sha256(key.encode()).hexdigest()
    return int(digest, 16) % n


def color_for_tag(tag: str) -> tuple:
    return get_color(_PALETTE[_hash_index(f"tag:{tag}", len(_PALETTE))])


def color_for_charge(charge: int) -> tuple:
    return get_color(_PALETTE[_hash_index(f"charge:{charge}", len(_PALETTE))])
