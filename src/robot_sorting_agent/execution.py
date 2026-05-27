from __future__ import annotations

from .schemas import DryRunEvent, DryRunReport, ExecutionCommand, ExecutionFeedback, ExecutionResult, ObjectTable, Plan


class ExecutionAdapter:
    def build_commands(self, plan: Plan, object_table: ObjectTable) -> list[ExecutionCommand]:
        object_map = {o.object_id: o for o in object_table.objects}
        commands: list[ExecutionCommand] = []
        for idx, step in enumerate(plan.steps, start=1):
            obj = object_map.get(step.object_id) if step.object_id else None
            reference_obj = object_map.get(step.reference_object_id) if step.reference_object_id else None
            commands.append(
                ExecutionCommand(
                    command_id=f"cmd_{idx:03d}",
                    step_id=step.step_id,
                    action=step.action if step.action != "replan" else "inspect",
                    object_id=step.object_id,
                    target=step.target,
                    relation=step.relation,
                    reference_object_id=step.reference_object_id,
                    object_ref=obj.execution_ref if obj else None,
                    reference_object_ref=reference_obj.execution_ref if reference_obj else None,
                    bbox=obj.bbox if obj else None,
                    reference_bbox=reference_obj.bbox if reference_obj else None,
                    mask=obj.mask if obj else None,
                    grasp_planner="AnyGrasp" if step.action == "pick" else None,
                    parameters=step.arguments,
                )
            )
        return commands

    def dry_run(self, commands: list[ExecutionCommand], object_table: ObjectTable) -> DryRunReport:
        object_ids = {o.object_id for o in object_table.objects}
        events: list[DryRunEvent] = []
        ready = 0
        for cmd in commands:
            if cmd.action in {"inspect", "pick"}:
                if not cmd.object_id:
                    events.append(DryRunEvent(command_id=cmd.command_id, status="blocked", reason="missing object_id"))
                    continue
                if cmd.object_id not in object_ids:
                    events.append(DryRunEvent(command_id=cmd.command_id, status="blocked", reason="unknown object_id"))
                    continue
            if cmd.reference_object_id and cmd.reference_object_id not in object_ids:
                events.append(DryRunEvent(command_id=cmd.command_id, status="blocked", reason="unknown reference_object_id"))
                continue
            if cmd.action == "place" and cmd.relation and not cmd.reference_object_id:
                events.append(DryRunEvent(command_id=cmd.command_id, status="blocked", reason="missing reference_object_id"))
                continue
            if cmd.action == "place" and not cmd.target and not cmd.relation:
                events.append(DryRunEvent(command_id=cmd.command_id, status="blocked", reason="missing target"))
                continue
            if cmd.action == "skip":
                events.append(DryRunEvent(command_id=cmd.command_id, status="skipped", reason="skip action"))
                continue
            events.append(DryRunEvent(command_id=cmd.command_id, status="ready"))
            ready += 1
        ratio = ready / max(len(commands), 1)
        return DryRunReport(events=events, ready_ratio=ratio)

    def apply_feedback(self, commands: list[ExecutionCommand], dry_run: DryRunReport, feedback: list[ExecutionFeedback] | None) -> tuple[list[ExecutionResult], bool, str | None, str | None]:
        feedback_map = {f.command_id: f for f in feedback or []}
        results: list[ExecutionResult] = []
        needs_replan = False
        failed_reason = None
        failed_object_id = None
        dry_map = {e.command_id: e for e in dry_run.events}
        for cmd in commands:
            if cmd.command_id in feedback_map:
                fb = feedback_map[cmd.command_id]
                result = ExecutionResult(
                    command_id=cmd.command_id,
                    status=fb.status if fb.status != "uncertain" else "failed",
                    requires_replan=fb.status in {"failed", "blocked", "uncertain"},
                    reason=fb.reason,
                )
            else:
                dry = dry_map[cmd.command_id]
                mapped = "ready" if dry.status == "ready" else ("skipped" if dry.status == "skipped" else "blocked")
                result = ExecutionResult(
                    command_id=cmd.command_id,
                    status=mapped,
                    requires_replan=(mapped == "blocked"),
                    reason=dry.reason,
                )
            if result.requires_replan and not needs_replan:
                needs_replan = True
                failed_reason = result.reason or "execution issue"
                failed_object_id = cmd.object_id
            results.append(result)
        return results, needs_replan, failed_reason, failed_object_id
