from __future__ import annotations

import argparse
import json
from pathlib import Path

from .graph import WorkflowRuntime
from .schemas import PipelineRequest
from .settings import Settings


def _compact_object_table(response) -> str:
    if not response.object_table:
        return "null"
    payload = response.object_table.model_dump()
    for obj in payload.get("objects", []):
        mask = obj.get("mask")
        if isinstance(mask, dict):
            mask.pop("data", None)
    return json.dumps(payload, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, help="Path to request JSON")
    parser.add_argument("--print", default="response", choices=["response", "objects", "plan", "commands", "trace"])
    args = parser.parse_args()

    request = PipelineRequest.model_validate_json(Path(args.request).read_text(encoding="utf-8"))
    runtime = WorkflowRuntime(Settings.from_env())
    response = runtime.invoke(request)

    if args.print == "response":
        print(response.model_dump_json(indent=2))
    elif args.print == "objects":
        print(_compact_object_table(response))
    elif args.print == "plan":
        print(response.plan.model_dump_json(indent=2) if response.plan else "null")
    elif args.print == "commands":
        print(json.dumps([c.model_dump() if hasattr(c, "model_dump") else c for c in response.execution_commands], ensure_ascii=False, indent=2))
    elif args.print == "trace":
        print(json.dumps([t.model_dump() for t in response.agent_trace], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
