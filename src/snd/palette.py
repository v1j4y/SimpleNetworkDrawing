"""Thin wrapper over quimb's existing colorblind-safe (Okabe-Ito) color
machinery, reused for both tag coloring and charge-value coloring so a
given charge value is visually consistent across a whole figure."""

from quimb.schematic import hash_to_color


def color_for_tag(tag: str) -> tuple:
    return hash_to_color(f"tag:{tag}")


def color_for_charge(charge: int) -> tuple:
    return hash_to_color(f"charge:{charge}")
