"""Tag-centric style dispatch, extended to also target leg-name glob
patterns (`leg:<pattern>`) and node type (`type:<node_type>`), not just
tensor tags. Later-added rules override earlier ones on conflicting
keys, mirroring quimb's own tag-styling precedence."""

import fnmatch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set


@dataclass
class StyleRule:
    target: str
    style: Dict[str, Any] = field(default_factory=dict)


class StyleSheet:
    def __init__(self) -> None:
        self._rules: List[StyleRule] = []

    def add(self, target: str, **style: Any) -> "StyleSheet":
        self._rules.append(StyleRule(target=target, style=style))
        return self

    def resolve_tensor(self, tags: Set[str], node_type: str) -> Dict[str, Any]:
        resolved: Dict[str, Any] = {}
        for rule in self._rules:
            if rule.target.startswith("leg:"):
                continue
            if rule.target in tags or rule.target == f"type:{node_type}":
                resolved.update(rule.style)
        return resolved

    def resolve_leg(self, ind: str) -> Dict[str, Any]:
        resolved: Dict[str, Any] = {}
        for rule in self._rules:
            if not rule.target.startswith("leg:"):
                continue
            pattern = rule.target[len("leg:") :]
            if fnmatch.fnmatch(ind, pattern):
                resolved.update(rule.style)
        return resolved
