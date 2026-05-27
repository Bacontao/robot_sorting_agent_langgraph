from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format="%(levelname)s %(name)s %(message)s")


class ArtifactStore:
    def __init__(self, root: Path):
        self.root = root

    def request_dir(self, request_id: str) -> Path:
        path = self.root / request_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, request_id: str, filename: str, data: Any) -> str:
        path = self.request_dir(request_id) / filename
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)

    def dump_failure(self, request_id: str, error: Exception, state: dict[str, Any]) -> str:
        return self.write_json(
            request_id,
            "failure.json",
            {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "state": state,
            },
        )
