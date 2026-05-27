from __future__ import annotations

from typing import Any


def compact_for_llm(value: Any) -> Any:
    if isinstance(value, list):
        return [compact_for_llm(item) for item in value]
    if not isinstance(value, dict):
        return value

    compact: dict[str, Any] = {}
    for key, item in value.items():
        if key == "data" and "uri" in value and "encoding" in value:
            continue
        compact[key] = compact_for_llm(item)
    return compact
