from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robot_sorting_agent.graph import WorkflowRuntime  # noqa: E402
from robot_sorting_agent.schemas import ObjectInstance, ObjectTable, PipelineRequest, PipelineResponse  # noqa: E402
from robot_sorting_agent.settings import Settings  # noqa: E402


def _load_cases(path: Path, limit: int | None, case_id: str | None) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if case_id:
        rows = [row for row in rows if row.get("case_id") == case_id]
    return rows[:limit] if limit else rows


def _terms(obj: ObjectInstance) -> set[str]:
    values = [
        obj.object_id,
        obj.label,
        obj.canonical_label,
        obj.attributes.color,
        obj.attributes.material,
        obj.attributes.shape,
        obj.attributes.state,
    ]
    return {str(value).strip().lower() for value in values if value}


def _matches(obj: ObjectInstance, spec: dict[str, Any]) -> bool:
    terms = _terms(obj)
    structured_values = {
        "color": obj.attributes.color,
        "shape": obj.attributes.shape,
        "material": obj.attributes.material,
        "state": obj.attributes.state,
    }
    for field in ["object_id", "label", "canonical_label", "color", "shape", "material", "state"]:
        expected = spec.get(field)
        if expected is None:
            continue
        expected_text = str(expected).strip().lower()
        if field in structured_values and structured_values[field]:
            if str(structured_values[field]).strip().lower() != expected_text:
                return False
            continue
        if expected_text not in terms and not any(expected_text in term or term in expected_text for term in terms):
            return False
    return True


def _find_object_id(table: ObjectTable | None, spec: dict[str, Any]) -> str | None:
    if table is None:
        return None
    matches = [obj for obj in table.objects if _matches(obj, spec)]
    if not matches:
        return None
    matches.sort(key=lambda obj: (obj.task_relevance, obj.confidence), reverse=True)
    return matches[0].object_id


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 1.0


def _evaluate_response(case: dict[str, Any], response: PipelineResponse, latency_ms: float) -> dict[str, Any]:
    expected = case.get("expected", {})
    table = response.object_table
    plan = response.plan
    expected_objects = expected.get("objects", [])
    expected_assignments = expected.get("assignments", [])
    expected_relations = expected.get("relations", [])
    min_commands = int(expected.get("min_commands", 1))

    found_objects = sum(1 for spec in expected_objects if _find_object_id(table, spec))
    assignment_hits = 0
    relation_hits = 0
    command_hits = 0
    resolved_assignments = []
    resolved_relations = []

    for item in expected_assignments:
        obj_id = _find_object_id(table, item.get("object", item))
        target = str(item.get("target", "")).lower()
        hit = False
        if obj_id and plan:
            hit = any(a.object_id == obj_id and str(a.target).lower() == target for a in plan.assignments)
            command_hits += int(any(c.object_id == obj_id and c.action == "place" and str(c.target).lower() == target for c in response.execution_commands))
        assignment_hits += int(hit)
        resolved_assignments.append({"object_id": obj_id, "target": target, "hit": hit})

    for item in expected_relations:
        subject_id = _find_object_id(table, item.get("subject", {}))
        reference_id = _find_object_id(table, item.get("reference", {}))
        relation = item.get("relation")
        hit = False
        if subject_id and reference_id and plan:
            hit = any(
                step.action == "place"
                and step.object_id == subject_id
                and step.reference_object_id == reference_id
                and step.relation == relation
                for step in plan.steps
            )
            command_hits += int(
                any(
                    command.action == "place"
                    and command.object_id == subject_id
                    and command.reference_object_id == reference_id
                    and command.relation == relation
                    for command in response.execution_commands
                )
            )
        relation_hits += int(hit)
        resolved_relations.append(
            {
                "subject_id": subject_id,
                "reference_id": reference_id,
                "relation": relation,
                "hit": hit,
            }
        )

    command_expectations = len(expected_assignments) + len(expected_relations)
    perception_recall = _ratio(found_objects, len(expected_objects))
    assignment_accuracy = _ratio(assignment_hits, len(expected_assignments))
    relation_accuracy = _ratio(relation_hits, len(expected_relations))
    command_accuracy = _ratio(command_hits, command_expectations)
    dry_run_ready_ratio = response.dry_run.ready_ratio if response.dry_run else 0.0
    plan_valid = bool(plan and plan.validation.status in {"pass", "warning"})
    command_valid = len(response.execution_commands) >= min_commands and dry_run_ready_ratio > 0.0
    semantic_success = bool(
        plan_valid
        and command_valid
        and perception_recall >= float(expected.get("min_perception_recall", 1.0))
        and assignment_accuracy >= float(expected.get("min_assignment_accuracy", 1.0))
        and relation_accuracy >= float(expected.get("min_relation_accuracy", 1.0))
        and command_accuracy >= float(expected.get("min_command_accuracy", 1.0))
    )
    metadata = table.metadata if table else {}
    return {
        "case_id": case.get("case_id"),
        "status": "ok",
        "semantic_success": semantic_success,
        "perception_recall": perception_recall,
        "assignment_accuracy": assignment_accuracy,
        "relation_accuracy": relation_accuracy,
        "command_accuracy": command_accuracy,
        "plan_valid": plan_valid,
        "plan_status": plan.validation.status if plan else "missing",
        "command_valid": command_valid,
        "dry_run_ready_ratio": dry_run_ready_ratio,
        "num_objects": len(table.objects) if table else 0,
        "num_commands": len(response.execution_commands),
        "backend_used": metadata.get("segmentation_backend"),
        "segmentation_attempts": metadata.get("segmentation_attempts", []),
        "diagnostics": [d.model_dump() for d in response.diagnostics],
        "resolved_assignments": resolved_assignments,
        "resolved_relations": resolved_relations,
        "latency_ms": latency_ms,
    }


def evaluate_case(runtime: WorkflowRuntime, case: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        request_payload = {
            key: case[key]
            for key in ["image", "instruction", "request_id", "execution_feedback"]
            if key in case
        }
        response = runtime.invoke(PipelineRequest.model_validate(request_payload))
    except Exception as exc:
        return {
            "case_id": case.get("case_id"),
            "status": "exception",
            "semantic_success": False,
            "error": str(exc),
            "latency_ms": (time.perf_counter() - start) * 1000,
        }
    return _evaluate_response(case, response, (time.perf_counter() - start) * 1000)


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok_results = [row for row in results if row["status"] == "ok"]
    backend_usage = Counter(str(row.get("backend_used") or "unknown") for row in ok_results)
    return {
        "num_cases": len(results),
        "num_ok": len(ok_results),
        "exception_rate": _ratio(len(results) - len(ok_results), len(results)),
        "semantic_success_rate": _ratio(sum(row.get("semantic_success", False) for row in results), len(results)),
        "plan_valid_rate": _ratio(sum(row.get("plan_valid", False) for row in ok_results), len(ok_results)),
        "command_valid_rate": _ratio(sum(row.get("command_valid", False) for row in ok_results), len(ok_results)),
        "avg_perception_recall": round(statistics.mean(row.get("perception_recall", 0.0) for row in ok_results), 4) if ok_results else 0.0,
        "avg_assignment_accuracy": round(statistics.mean(row.get("assignment_accuracy", 0.0) for row in ok_results), 4) if ok_results else 0.0,
        "avg_relation_accuracy": round(statistics.mean(row.get("relation_accuracy", 0.0) for row in ok_results), 4) if ok_results else 0.0,
        "avg_command_accuracy": round(statistics.mean(row.get("command_accuracy", 0.0) for row in ok_results), 4) if ok_results else 0.0,
        "avg_dry_run_ready_ratio": round(statistics.mean(row.get("dry_run_ready_ratio", 0.0) for row in ok_results), 4) if ok_results else 0.0,
        "avg_latency_ms": round(statistics.mean(row.get("latency_ms", 0.0) for row in results), 3) if results else 0.0,
        "backend_usage": dict(backend_usage),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run semantic evaluation cases for the robot sorting agent.")
    parser.add_argument("--cases", default="samples/eval_cases.jsonl")
    parser.add_argument("--output", default="artifacts/eval_report.json")
    parser.add_argument("--artifact-dir", default="artifacts/eval_runs")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-id")
    parser.add_argument("--print-cases", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("ARTIFACT_DIR", args.artifact_dir)
    cases = _load_cases(Path(args.cases), args.limit, args.case_id)
    runtime = WorkflowRuntime(Settings.from_env())
    results = [evaluate_case(runtime, case) for case in cases]
    report = {"summary": summarize(results), "cases": results}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.print_cases:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
