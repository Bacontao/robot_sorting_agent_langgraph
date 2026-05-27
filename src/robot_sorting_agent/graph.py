from __future__ import annotations

import time
import uuid
from typing import Any, TypedDict

from .compat_langgraph import END, START, StateGraph
from .execution import ExecutionAdapter
from .llm import ModelRouter
from .observability import ArtifactStore
from .perception import PerceptionService
from .planning import PlanningService
from .prompts import TOOL_POLICY_SYSTEM_PROMPT
from .schemas import (
    AgentTraceStep,
    Assignment,
    BenchmarkSummary,
    DryRunReport,
    ExecutionCommand,
    ExecutionResult,
    FailureDiagnosis,
    ImageInput,
    ObjectTable,
    PipelineRequest,
    PipelineResponse,
    Plan,
    PlanReview,
    Rule,
    TaskIntent,
    ToolPolicyDecision,
)
from .segmentation import build_segmentation_backend
from .settings import Settings


class WorkflowState(TypedDict, total=False):
    request_id: str
    request: dict[str, Any]
    instruction: str
    segments: list[dict[str, Any]]
    segmentation_backend_used: str | None
    segmentation_attempts: list[dict[str, Any]]
    object_table: dict[str, Any]
    task_intent: dict[str, Any]
    rules: list[dict[str, Any]]
    assignments: list[dict[str, Any]]
    plan: dict[str, Any]
    plan_review: dict[str, Any]
    repair_decision: dict[str, Any]
    diagnosis: dict[str, Any]
    diagnostics: list[dict[str, Any]]
    restart_from: str | None
    warning_history: list[str]
    execution_commands: list[dict[str, Any]]
    dry_run: dict[str, Any]
    execution_results: list[dict[str, Any]]
    needs_replan: bool
    failed_reason: str | None
    failed_object_id: str | None
    benchmark: dict[str, Any]
    trace: list[dict[str, Any]]
    warnings: list[str]
    replan_count: int
    recovery_count: int
    repair_applied: bool
    needs_repair: bool


class WorkflowRuntime:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = ModelRouter(settings)
        self.segmentation = build_segmentation_backend(settings)
        self.perception = PerceptionService(settings, self.llm)
        self.planning = PlanningService(self.llm)
        self.execution = ExecutionAdapter()
        self.artifacts = ArtifactStore(settings.artifact_dir)

    def _append_trace(self, state: WorkflowState, tool_name: str, status: str, decision: str, input_summary: str, output_summary: str, latency_ms: float, error: str | None = None) -> None:
        trace = state.setdefault("trace", [])
        trace.append(
            AgentTraceStep(
                order=len(trace) + 1,
                tool_name=tool_name,
                status=status,  # type: ignore[arg-type]
                decision=decision,
                input_summary=input_summary,
                output_summary=output_summary,
                latency_ms=latency_ms,
                error=error,
            ).model_dump()
        )

    def _tool_policy(self, state: WorkflowState, tool_name: str) -> ToolPolicyDecision:
        if not self.settings.enable_tool_policy:
            return ToolPolicyDecision(action="run", rationale="tool policy disabled", confidence=1.0)
        if tool_name != "repair_tool":
            return ToolPolicyDecision(action="run", rationale="mandatory tool", confidence=1.0)
        if state.get("restart_from") == "repair" or state.get("needs_repair"):
            return ToolPolicyDecision(action="run", rationale="repair requested by graph diagnosis", confidence=1.0)
        return self.llm.call(
            role="tool_policy",
            task_name="tool_policy",
            system_prompt=TOOL_POLICY_SYSTEM_PROMPT,
            user_payload={
                "tool_name": tool_name,
                "warnings": state.get("warnings", []),
                "needs_replan": state.get("needs_replan", False),
                "replan_count": state.get("replan_count", 0),
            },
            schema_model=ToolPolicyDecision,
        )

    def _run_tool(self, state: WorkflowState, tool_name: str, fn, input_summary: str) -> dict[str, Any]:
        start = time.perf_counter()
        policy = self._tool_policy(state, tool_name)
        if policy.action == "skip":
            self._append_trace(state, tool_name, "skipped", f"tool policy: {policy.rationale}", input_summary, "skipped", (time.perf_counter() - start) * 1000)
            return {}
        try:
            output = fn(state)
            self._append_trace(
                state,
                tool_name,
                "success",
                f"ran: {policy.rationale}",
                input_summary,
                ", ".join(output.keys()) if output else "no updates",
                (time.perf_counter() - start) * 1000,
            )
            return output
        except Exception as exc:
            self._append_trace(state, tool_name, "failed", "exception", input_summary, "failed", (time.perf_counter() - start) * 1000, error=str(exc))
            raise

    @staticmethod
    def _merge_warnings(*groups: list[str]) -> list[str]:
        seen = set()
        merged = []
        for group in groups:
            for warning in group:
                if warning not in seen:
                    merged.append(warning)
                    seen.add(warning)
        return merged

    @staticmethod
    def _cleanup_for_restart(restart_from: str) -> dict[str, Any]:
        updates: dict[str, Any] = {
            "restart_from": restart_from,
            "warnings": [],
            "needs_repair": False,
        }
        if restart_from in {"segmentation", "perception", "parse_intent", "assignment", "step_generation", "repair"}:
            updates.update(
                {
                    "execution_commands": [],
                    "dry_run": {},
                    "execution_results": [],
                    "needs_replan": False,
                    "failed_reason": None,
                    "failed_object_id": None,
                }
            )
        if restart_from == "segmentation":
            updates.update(
                {
                    "segments": [],
                    "segmentation_backend_used": None,
                    "segmentation_attempts": [],
                    "object_table": {},
                    "task_intent": {},
                    "rules": [],
                    "assignments": [],
                    "plan": {},
                    "plan_review": {},
                }
            )
        elif restart_from == "perception":
            updates.update(
                {
                    "object_table": {},
                    "task_intent": {},
                    "rules": [],
                    "assignments": [],
                    "plan": {},
                    "plan_review": {},
                }
            )
        elif restart_from == "parse_intent":
            updates.update({"task_intent": {}, "rules": [], "assignments": [], "plan": {}, "plan_review": {}})
        elif restart_from == "assignment":
            updates.update({"rules": [], "assignments": [], "plan": {}, "plan_review": {}})
        elif restart_from == "step_generation":
            updates.update({"plan": {}, "plan_review": {}})
        elif restart_from == "execution_adapter":
            updates.update({"execution_commands": [], "dry_run": {}, "execution_results": [], "needs_replan": False})
        return updates

    def _record_diagnosis(self, state: WorkflowState, diagnosis: FailureDiagnosis) -> dict[str, Any]:
        diagnostics = list(state.get("diagnostics", []))
        dumped = diagnosis.model_dump()
        diagnostics.append(dumped)
        count = state.get("recovery_count", 0) + 1
        self.artifacts.write_json(state["request_id"], f"diagnosis_{count}.json", dumped)
        updates = self._cleanup_for_restart(diagnosis.restart_from)
        updates.update(
            {
                "diagnosis": dumped,
                "diagnostics": diagnostics,
                "restart_from": diagnosis.restart_from,
                "recovery_count": count,
            }
        )
        return updates

    def segmentation_node(self, state: WorkflowState) -> dict[str, Any]:
        image = ImageInput.model_validate(state["request"]["image"])
        segments = self.segmentation.segment(image, state["instruction"])
        dumped = [s.model_dump() for s in segments]
        backend_used = getattr(self.segmentation, "last_backend_name", None) or self.settings.segmentation_backend
        attempts = getattr(self.segmentation, "last_attempts", [])
        self.artifacts.write_json(state["request_id"], "segmentation.json", dumped)
        self.artifacts.write_json(
            state["request_id"],
            "segmentation_meta.json",
            {"backend_used": backend_used, "attempts": attempts, "num_segments": len(dumped)},
        )
        return {"segments": dumped, "segmentation_backend_used": backend_used, "segmentation_attempts": attempts}

    def perception_node(self, state: WorkflowState) -> dict[str, Any]:
        from .schemas import SegmentationCandidate
        image = ImageInput.model_validate(state["request"]["image"])
        segments = [SegmentationCandidate.model_validate(s) for s in state["segments"]]
        object_table = self.perception.build_object_table(image, state["instruction"], segments)
        object_table.metadata["segmentation_backend"] = state.get("segmentation_backend_used")
        object_table.metadata["segmentation_attempts"] = state.get("segmentation_attempts", [])
        self.artifacts.write_json(state["request_id"], "object_table.json", object_table.model_dump())
        return {"object_table": object_table.model_dump()}

    def validate_objects_node(self, state: WorkflowState) -> dict[str, Any]:
        object_table = ObjectTable.model_validate(state["object_table"])
        warnings = []
        if not object_table.objects:
            warnings.append("object_table is empty")
        if any(not o.execution_ref for o in object_table.objects):
            warnings.append("some objects miss execution_ref")
        return {"warnings": warnings, "warning_history": self._merge_warnings(state.get("warning_history", []), warnings)}

    def parse_intent_node(self, state: WorkflowState) -> dict[str, Any]:
        object_table = ObjectTable.model_validate(state["object_table"])
        intent = self.planning.parse_task_intent(state["instruction"], object_table)
        intent = self.planning.prepare_intent(state["instruction"], object_table, intent)
        self.artifacts.write_json(state["request_id"], "task_intent.json", intent.model_dump())
        return {"task_intent": intent.model_dump(), "restart_from": None}

    def assignment_node(self, state: WorkflowState) -> dict[str, Any]:
        object_table = ObjectTable.model_validate(state["object_table"])
        intent = TaskIntent.model_validate(state["task_intent"])
        intent = self.planning.prepare_intent(state["instruction"], object_table, intent)
        rules = self.planning.intent_to_rules(intent)
        assignments = self.planning.generate_assignments(object_table, rules)
        self.artifacts.write_json(state["request_id"], "rules.json", [r.model_dump() for r in rules])
        self.artifacts.write_json(state["request_id"], "assignments.json", [a.model_dump() for a in assignments])
        return {
            "task_intent": intent.model_dump(),
            "rules": [r.model_dump() for r in rules],
            "assignments": [a.model_dump() for a in assignments],
            "restart_from": None,
        }

    def step_generation_node(self, state: WorkflowState) -> dict[str, Any]:
        object_table = ObjectTable.model_validate(state["object_table"])
        intent = TaskIntent.model_validate(state["task_intent"])
        rules = [Rule.model_validate(r) for r in state["rules"]]
        assignments = [Assignment.model_validate(a) for a in state["assignments"]]
        plan = self.planning.build_plan(object_table, intent, rules, assignments)
        self.artifacts.write_json(state["request_id"], "plan.json", plan.model_dump())
        return {"plan": plan.model_dump(), "restart_from": None}

    def plan_review_node(self, state: WorkflowState) -> dict[str, Any]:
        object_table = ObjectTable.model_validate(state["object_table"])
        plan = Plan.model_validate(state["plan"])
        review = self.planning.review_plan(state["instruction"], object_table, plan)
        plan = self.planning.apply_review_to_plan(plan, review)
        self.artifacts.write_json(state["request_id"], "plan.json", plan.model_dump())
        self.artifacts.write_json(state["request_id"], "plan_review.json", review.model_dump())
        warnings = self._merge_warnings(review.warnings, plan.validation.warnings)
        return {
            "plan": plan.model_dump(),
            "plan_review": review.model_dump(),
            "warnings": warnings,
            "warning_history": self._merge_warnings(state.get("warning_history", []), warnings),
        }

    def planning_node(self, state: WorkflowState) -> dict[str, Any]:
        object_table = ObjectTable.model_validate(state["object_table"])
        intent, plan, review = self.planning.create_plan(state["instruction"], object_table)
        self.artifacts.write_json(state["request_id"], "task_intent.json", intent.model_dump())
        self.artifacts.write_json(state["request_id"], "plan.json", plan.model_dump())
        self.artifacts.write_json(state["request_id"], "plan_review.json", review.model_dump())
        warnings = list(state.get("warnings", [])) + review.warnings + plan.validation.warnings
        return {"task_intent": intent.model_dump(), "plan": plan.model_dump(), "plan_review": review.model_dump(), "warnings": warnings}

    def validate_plan_node(self, state: WorkflowState) -> dict[str, Any]:
        review = PlanReview.model_validate(state["plan_review"])
        plan = Plan.model_validate(state["plan"])
        return {"needs_repair": review.verdict in {"repair", "replan"} or plan.validation.status == "fail"}

    def repair_node(self, state: WorkflowState) -> dict[str, Any]:
        object_table = ObjectTable.model_validate(state["object_table"])
        plan = Plan.model_validate(state["plan"])
        review = PlanReview.model_validate(state["plan_review"])
        if review.verdict == "pass":
            return {"repair_applied": False}
        repaired, decision = self.planning.repair_plan(state["instruction"], object_table, plan, review)
        self.artifacts.write_json(state["request_id"], "repair_decision.json", decision.model_dump())
        self.artifacts.write_json(state["request_id"], "plan_repaired.json", repaired.model_dump())
        restart_from = None
        if any(action.kind == "reperceive" for action in decision.repairs):
            restart_from = "perception"
        elif any(action.kind == "clarify" for action in decision.repairs):
            restart_from = "finish"
        return {
            "plan": repaired.model_dump(),
            "repair_decision": decision.model_dump(),
            "repair_applied": True,
            "warnings": repaired.validation.warnings,
            "warning_history": self._merge_warnings(state.get("warning_history", []), repaired.validation.warnings),
            "restart_from": restart_from,
        }

    def execution_adapter_node(self, state: WorkflowState) -> dict[str, Any]:
        object_table = ObjectTable.model_validate(state["object_table"])
        plan = Plan.model_validate(state["plan"])
        commands = self.execution.build_commands(plan, object_table)
        dry_run = self.execution.dry_run(commands, object_table)
        self.artifacts.write_json(state["request_id"], "execution_commands.json", [c.model_dump() for c in commands])
        self.artifacts.write_json(state["request_id"], "dry_run.json", dry_run.model_dump())
        return {"execution_commands": [c.model_dump() for c in commands], "dry_run": dry_run.model_dump()}

    def execution_feedback_node(self, state: WorkflowState) -> dict[str, Any]:
        from .schemas import ExecutionFeedback
        commands = [ExecutionCommand.model_validate(c) for c in state["execution_commands"]]
        dry_run = DryRunReport.model_validate(state["dry_run"])
        feedback = None
        if self.settings.enable_execution_feedback_loop and state["request"].get("execution_feedback"):
            feedback = [ExecutionFeedback.model_validate(x) for x in state["request"]["execution_feedback"]]
        results, needs_replan, failed_reason, failed_object_id = self.execution.apply_feedback(commands, dry_run, feedback)
        self.artifacts.write_json(state["request_id"], "execution_results.json", [r.model_dump() for r in results])
        return {
            "execution_results": [r.model_dump() for r in results],
            "needs_replan": needs_replan,
            "failed_reason": failed_reason,
            "failed_object_id": failed_object_id,
        }

    def diagnose_plan_issue_node(self, state: WorkflowState) -> dict[str, Any]:
        object_table = ObjectTable.model_validate(state["object_table"]) if state.get("object_table") else None
        plan = Plan.model_validate(state["plan"]) if state.get("plan") else None
        review = PlanReview.model_validate(state["plan_review"]) if state.get("plan_review") else None
        diagnosis = self.planning.diagnose_failure(
            source="plan_review",
            instruction=state["instruction"],
            object_table=object_table,
            plan=plan,
            plan_review=review,
            warnings=state.get("warnings", []),
        )
        return self._record_diagnosis(state, diagnosis)

    def diagnose_execution_issue_node(self, state: WorkflowState) -> dict[str, Any]:
        object_table = ObjectTable.model_validate(state["object_table"]) if state.get("object_table") else None
        plan = Plan.model_validate(state["plan"]) if state.get("plan") else None
        diagnosis = self.planning.diagnose_failure(
            source="execution_feedback",
            instruction=state["instruction"],
            object_table=object_table,
            plan=plan,
            execution_results=state.get("execution_results", []),
            dry_run=state.get("dry_run"),
            failed_reason=state.get("failed_reason"),
            failed_object_id=state.get("failed_object_id"),
            warnings=state.get("warnings", []),
        )
        return self._record_diagnosis(state, diagnosis)

    def replan_node(self, state: WorkflowState) -> dict[str, Any]:
        object_table = ObjectTable.model_validate(state["object_table"])
        plan = Plan.model_validate(state["plan"])
        replanned, decision = self.planning.replan_from_feedback(
            state["instruction"],
            object_table,
            plan,
            state.get("failed_reason") or "execution issue",
            state.get("failed_object_id"),
        )
        count = state.get("replan_count", 0) + 1
        self.artifacts.write_json(state["request_id"], f"replan_{count}.json", replanned.model_dump())
        return {
            "plan": replanned.model_dump(),
            "repair_decision": decision.model_dump(),
            "replan_count": count,
            "needs_replan": False,
            "warnings": replanned.validation.warnings,
            "warning_history": self._merge_warnings(state.get("warning_history", []), replanned.validation.warnings),
        }

    def benchmark_node(self, state: WorkflowState) -> dict[str, Any]:
        commands = state.get("execution_commands", [])
        results = state.get("execution_results", [])
        plan = Plan.model_validate(state["plan"]) if state.get("plan") else None
        benchmark = BenchmarkSummary(
            task_intent_valid=bool(state.get("task_intent")),
            command_valid=bool(commands) and bool(results) and all(r["status"] in {"ready", "succeeded", "skipped", "failed", "blocked"} for r in results),
            repair_success=bool(plan and plan.validation.status != "fail"),
            feedback_loop_ready=bool(results),
            end_to_end_latency_ms=0.0,
        )
        return {"benchmark": benchmark.model_dump()}

    def output_node(self, state: WorkflowState) -> dict[str, Any]:
        self.artifacts.write_json(state["request_id"], "agent_trace.json", state.get("trace", []))
        self.artifacts.write_json(state["request_id"], "workflow_state.json", state)
        return {}

    def build_graph(self):
        graph = StateGraph(WorkflowState)
        graph.add_node("segmentation", lambda s: self._run_tool(s, "segmentation_tool", self.segmentation_node, "instruction"))
        graph.add_node("perception", lambda s: self._run_tool(s, "perception_tool", self.perception_node, "segments"))
        graph.add_node("validate_objects", lambda s: self._run_tool(s, "validate_objects_tool", self.validate_objects_node, "object_table"))
        graph.add_node("parse_intent", lambda s: self._run_tool(s, "parse_intent_tool", self.parse_intent_node, "instruction + object_table"))
        graph.add_node("assignment", lambda s: self._run_tool(s, "assignment_tool", self.assignment_node, "task_intent + object_table"))
        graph.add_node("step_generation", lambda s: self._run_tool(s, "step_generation_tool", self.step_generation_node, "assignments + rules"))
        graph.add_node("plan_review", lambda s: self._run_tool(s, "plan_review_tool", self.plan_review_node, "plan + object_table"))
        graph.add_node("validate_plan", lambda s: self._run_tool(s, "validate_plan_tool", self.validate_plan_node, "plan"))
        graph.add_node("diagnose_plan_issue", lambda s: self._run_tool(s, "diagnose_plan_issue_tool", self.diagnose_plan_issue_node, "plan_review + plan"))
        graph.add_node("repair", lambda s: self._run_tool(s, "repair_tool", self.repair_node, "plan_review + plan"))
        graph.add_node("execution_adapter", lambda s: self._run_tool(s, "execution_adapter_tool", self.execution_adapter_node, "plan + object_table"))
        graph.add_node("execution_feedback", lambda s: self._run_tool(s, "execution_feedback_tool", self.execution_feedback_node, "commands + feedback"))
        graph.add_node("diagnose_execution_issue", lambda s: self._run_tool(s, "diagnose_execution_issue_tool", self.diagnose_execution_issue_node, "execution_results + failed_reason"))
        graph.add_node("replan", lambda s: self._run_tool(s, "replan_tool", self.replan_node, "execution_results + failed_reason"))
        graph.add_node("benchmark", lambda s: self._run_tool(s, "benchmark_tool", self.benchmark_node, "final state"))
        graph.add_node("output", lambda s: self._run_tool(s, "output_tool", self.output_node, "trace + artifacts"))

        graph.add_edge(START, "segmentation")
        graph.add_edge("segmentation", "perception")
        graph.add_edge("perception", "validate_objects")
        graph.add_edge("validate_objects", "parse_intent")
        graph.add_edge("parse_intent", "assignment")
        graph.add_edge("assignment", "step_generation")
        graph.add_edge("step_generation", "plan_review")
        graph.add_edge("plan_review", "validate_plan")
        graph.add_conditional_edges("validate_plan", self._route_after_validate, {"diagnose": "diagnose_plan_issue", "execute": "execution_adapter"})
        graph.add_conditional_edges(
            "diagnose_plan_issue",
            self._route_from_diagnosis,
            {
                "segmentation": "segmentation",
                "perception": "perception",
                "parse_intent": "parse_intent",
                "assignment": "assignment",
                "step_generation": "step_generation",
                "repair": "repair",
                "replan": "replan",
                "execution_adapter": "execution_adapter",
                "finish": "benchmark",
            },
        )
        graph.add_conditional_edges(
            "repair",
            self._route_after_repair,
            {
                "segmentation": "segmentation",
                "perception": "perception",
                "plan_review": "plan_review",
                "finish": "benchmark",
            },
        )
        graph.add_edge("execution_adapter", "execution_feedback")
        graph.add_conditional_edges("execution_feedback", self._route_after_feedback, {"diagnose": "diagnose_execution_issue", "finish": "benchmark"})
        graph.add_conditional_edges(
            "diagnose_execution_issue",
            self._route_from_diagnosis,
            {
                "segmentation": "segmentation",
                "perception": "perception",
                "parse_intent": "parse_intent",
                "assignment": "assignment",
                "step_generation": "step_generation",
                "repair": "repair",
                "replan": "replan",
                "execution_adapter": "execution_adapter",
                "finish": "benchmark",
            },
        )
        graph.add_edge("replan", "execution_adapter")
        graph.add_edge("benchmark", "output")
        graph.add_edge("output", END)
        return graph.compile()

    @staticmethod
    def _route_after_validate(state: WorkflowState) -> str:
        return "diagnose" if state.get("needs_repair") else "execute"

    def _route_from_diagnosis(self, state: WorkflowState) -> str:
        if state.get("recovery_count", 0) > self.settings.max_replans:
            return "finish"
        restart_from = state.get("restart_from") or "repair"
        if restart_from in {
            "segmentation",
            "perception",
            "parse_intent",
            "assignment",
            "step_generation",
            "repair",
            "replan",
            "execution_adapter",
            "finish",
        }:
            return restart_from
        return "repair"

    @staticmethod
    def _route_after_repair(state: WorkflowState) -> str:
        restart_from = state.get("restart_from")
        if restart_from in {"segmentation", "perception", "finish"}:
            return restart_from
        return "plan_review"

    def _route_after_feedback(self, state: WorkflowState) -> str:
        if state.get("needs_replan") and state.get("recovery_count", 0) <= self.settings.max_replans:
            return "diagnose"
        return "finish"

    def invoke(self, request: PipelineRequest) -> PipelineResponse:
        request_id = request.request_id or f"req_{uuid.uuid4().hex[:10]}"
        state: WorkflowState = {
            "request_id": request_id,
            "request": request.model_dump(),
            "instruction": request.instruction,
            "trace": [],
            "warnings": [],
            "warning_history": [],
            "diagnostics": [],
            "needs_replan": False,
            "replan_count": 0,
            "recovery_count": 0,
            "repair_applied": False,
        }
        graph = self.build_graph()
        start = time.perf_counter()
        final_state = graph.invoke(state)
        latency_ms = (time.perf_counter() - start) * 1000
        benchmark = BenchmarkSummary.model_validate(final_state.get("benchmark", {}))
        benchmark.end_to_end_latency_ms = latency_ms
        return PipelineResponse(
            request_id=request_id,
            object_table=ObjectTable.model_validate(final_state["object_table"]) if final_state.get("object_table") else None,
            task_intent=TaskIntent.model_validate(final_state["task_intent"]) if final_state.get("task_intent") else None,
            plan=Plan.model_validate(final_state["plan"]) if final_state.get("plan") else None,
            diagnostics=[FailureDiagnosis.model_validate(d) for d in final_state.get("diagnostics", [])],
            execution_commands=[ExecutionCommand.model_validate(c) for c in final_state.get("execution_commands", [])],
            dry_run=DryRunReport.model_validate(final_state["dry_run"]) if final_state.get("dry_run") else None,
            execution_results=[ExecutionResult.model_validate(r) for r in final_state.get("execution_results", [])],
            needs_replan=final_state.get("needs_replan", False),
            replan_count=final_state.get("replan_count", 0),
            benchmark=benchmark,
            agent_trace=[AgentTraceStep.model_validate(t) for t in final_state.get("trace", [])],
        )
