from snd.palette import color_for_tag, color_for_charge


def test_color_for_charge_is_deterministic():
    assert color_for_charge(2) == color_for_charge(2)


def test_color_for_charge_differs_across_values():
    colors = {color_for_charge(c) for c in range(5)}
    assert len(colors) == 5


def test_color_for_tag_is_deterministic_and_differs_from_charge_namespace():
    assert color_for_tag("A") == color_for_tag("A")
    # same raw value in the tag vs. charge namespace must not collide,
    # since they are hashed with different prefixes
    assert color_for_tag("2") != color_for_charge(2)
