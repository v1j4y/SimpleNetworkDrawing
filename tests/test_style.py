from snd.style import StyleSheet


def test_resolve_tensor_merges_matching_tag_rules_in_order():
    sheet = StyleSheet()
    sheet.add("A", color="blue")
    sheet.add("A", shape="triangle")  # later rule overrides/extends

    resolved = sheet.resolve_tensor(tags={"A"}, node_type="plain")

    assert resolved == {"color": "blue", "shape": "triangle"}


def test_resolve_tensor_matches_node_type_rule():
    sheet = StyleSheet()
    sheet.add("type:wavefunction", radius=0.4)

    resolved = sheet.resolve_tensor(tags={"psi"}, node_type="wavefunction")

    assert resolved == {"radius": 0.4}


def test_resolve_tensor_ignores_non_matching_rules():
    sheet = StyleSheet()
    sheet.add("B", color="red")

    resolved = sheet.resolve_tensor(tags={"A"}, node_type="plain")

    assert resolved == {}


def test_resolve_leg_matches_glob_pattern():
    sheet = StyleSheet()
    sheet.add("leg:b*", linewidth=2)
    sheet.add("leg:k0", linewidth=5)

    assert sheet.resolve_leg("b01") == {"linewidth": 2}
    assert sheet.resolve_leg("k0") == {"linewidth": 5}
    assert sheet.resolve_leg("other") == {}
