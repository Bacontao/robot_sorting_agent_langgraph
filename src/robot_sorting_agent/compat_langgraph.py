from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

try:  # pragma: no cover
    from langgraph.graph import END, START, StateGraph  # type: ignore
except Exception:
    START = "__start__"
    END = "__end__"

    ConditionFn = Callable[[dict[str, Any]], str]
    NodeFn = Callable[[dict[str, Any]], dict[str, Any]]

    @dataclass
    class _FallbackCompiledGraph:
        nodes: dict[str, NodeFn]
        edges: dict[str, list[str]]
        conditionals: dict[str, tuple[ConditionFn, dict[str, str]]]

        def invoke(self, initial_state: dict[str, Any]) -> dict[str, Any]:
            state = dict(initial_state)
            current = START
            steps = 0
            while steps < 100:
                steps += 1
                next_nodes = self.edges.get(current, [])
                if current in self.conditionals:
                    condition, mapping = self.conditionals[current]
                    target = mapping[condition(state)]
                elif next_nodes:
                    target = next_nodes[0]
                else:
                    break
                if target == END:
                    break
                updates = self.nodes[target](state) or {}
                for key, value in updates.items():
                    state[key] = value
                current = target
            return state

    @dataclass
    class StateGraph:
        _state_type: Any
        nodes: dict[str, NodeFn] = field(default_factory=dict)
        edges: dict[str, list[str]] = field(default_factory=dict)
        conditionals: dict[str, tuple[ConditionFn, dict[str, str]]] = field(default_factory=dict)

        def add_node(self, name: str, fn: NodeFn) -> None:
            self.nodes[name] = fn

        def add_edge(self, source: str, target: str) -> None:
            self.edges.setdefault(source, []).append(target)

        def add_conditional_edges(
            self,
            source: str,
            condition: ConditionFn,
            path_map: dict[str, str],
        ) -> None:
            self.conditionals[source] = (condition, path_map)

        def compile(self) -> _FallbackCompiledGraph:
            return _FallbackCompiledGraph(self.nodes, self.edges, self.conditionals)
