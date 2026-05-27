from __future__ import annotations

import re
from typing import Any

from .llm import ModelRouter
from .payloads import compact_for_llm
from .prompts import FAILURE_DIAGNOSIS_SYSTEM_PROMPT, PLAN_REVIEW_SYSTEM_PROMPT, REPAIR_SYSTEM_PROMPT, REPLAN_SYSTEM_PROMPT, TASK_INTENT_SYSTEM_PROMPT
from .schemas import (
    Assignment,
    FailureDiagnosis,
    ObjectTable,
    Plan,
    PlanReview,
    PlanStep,
    RepairDecision,
    Rule,
    SpatialRelationBlueprint,
    TaskIntent,
    ValidationReport,
)

RELATION_ALIASES = {
    "left_of": ["左边", "左侧", "左面", "left of", "to the left of", "left"],
    "right_of": ["右边", "右侧", "右面", "right of", "to the right of", "right"],
    "above": ["上方", "上面", "above", "on top of"],
    "below": ["下方", "下面", "below", "under"],
    "near": ["旁边", "附近", "near", "next to", "beside"],
}

OBJECT_ALIASES = {
    "草莓": "strawberry",
    "strawberry": "strawberry",
    "桃子": "peach",
    "水蜜桃": "peach",
    "peach": "peach",
    "牛油果": "avocado",
    "鳄梨": "avocado",
    "avocado": "avocado",
    "香蕉": "banana",
    "banana": "banana",
    "石榴": "pomegranate",
    "pomegranate": "pomegranate",
    "椰子": "coconut",
    "coconut": "coconut",
    "苹果": "apple",
    "apple": "apple",
    "橙子": "orange",
    "orange": "orange",
    "杯子": "cup",
    "cup": "cup",
    "瓶子": "bottle",
    "bottle": "bottle",
    "盒子": "box",
    "box": "box",
}


class PlanningService:
    def __init__(self, llm: ModelRouter):
        self.llm = llm

    def parse_task_intent(self, instruction: str, object_table: ObjectTable) -> TaskIntent:
        return self.llm.call(
            role="planner",
            task_name="task_intent",
            system_prompt=TASK_INTENT_SYSTEM_PROMPT,
            user_payload={"instruction": instruction, "object_table": compact_for_llm(object_table.model_dump())},
            schema_model=TaskIntent,
        )

    def prepare_intent(self, instruction: str, object_table: ObjectTable, intent: TaskIntent) -> TaskIntent:
        prepared = intent.model_copy(deep=True)
        spatial_relations = self.resolve_spatial_relations(instruction, object_table, prepared)
        if spatial_relations:
            prepared.spatial_relations = spatial_relations
            if "spatial" not in prepared.grouping_axes:
                prepared.grouping_axes.append("spatial")
        return prepared

    def intent_to_rules(self, intent: TaskIntent) -> list[Rule]:
        rules: list[Rule] = []
        for relation in intent.spatial_relations:
            if relation.subject_object_id and relation.reference_object_id:
                rules.append(
                    Rule(
                        rule_id=relation.relation_id,
                        conditions={"object_id": relation.subject_object_id},
                        action={
                            "target": relation.reference_object_id,
                            "relation": relation.relation,
                            "reference_object_id": relation.reference_object_id,
                        },
                        verification_required=False,
                    )
                )
        if rules or intent.spatial_relations:
            return rules
        for bp in intent.rule_blueprints:
            conditions = {bp.condition_field: bp.condition_value}
            if bp.condition_value == "any":
                conditions = {"graspable": True}
            rules.append(
                Rule(
                    rule_id=bp.rule_id,
                    conditions=conditions,
                    action={"target": bp.target},
                    verification_required=bp.verification_required,
                )
            )
        return rules

    @staticmethod
    def _normalize_query(query: str | None) -> str:
        if not query:
            return ""
        text = query.strip().lower()
        text = re.sub(r"^(把|将)\s*", "", text)
        text = re.sub(r"^(the|a|an)\s+", "", text)
        text = re.sub(r"(这个|那个|物体|对象|水果|的|放在|放到|放置)", "", text)
        text = re.sub(r"\b(object|objects|item|items|thing|things)\b", "", text)
        text = re.sub(r"\b(to|the|a|an)\b", "", text)
        text = text.strip(" .,，。")
        return OBJECT_ALIASES.get(text, text)

    @staticmethod
    def _object_terms(obj) -> set[str]:
        terms = {
            obj.label.lower(),
            obj.canonical_label.lower(),
        }
        for value in [obj.attributes.color, obj.attributes.material, obj.attributes.shape, obj.attributes.state]:
            if value:
                terms.add(value.lower())
        return {OBJECT_ALIASES.get(term, term) for term in terms if term}

    def _find_object(self, object_table: ObjectTable, query: str) -> str | None:
        normalized = self._normalize_query(query)
        if not normalized:
            return None
        exact_candidates = []
        candidates = []
        for obj in object_table.objects:
            terms = self._object_terms(obj)
            exact_fields = {
                value.lower()
                for value in [obj.label, obj.canonical_label, obj.attributes.color, obj.attributes.material, obj.attributes.shape, obj.attributes.state]
                if value
            }
            if normalized in exact_fields:
                exact_candidates.append(obj)
            elif normalized in terms or any(normalized in term or term in normalized for term in terms):
                candidates.append(obj)
        candidates = exact_candidates or candidates
        if not candidates:
            return None
        candidates.sort(key=lambda obj: (obj.task_relevance, obj.confidence), reverse=True)
        return candidates[0].object_id

    def _extract_spatial_relation_from_text(self, instruction: str) -> tuple[str, str, str] | None:
        text = instruction.strip()
        cn_relation = r"(左边|左侧|左面|右边|右侧|右面|上方|上面|下方|下面|旁边|附近)"
        cn_match = re.search(rf"(?:把|将)?(.+?)放(?:在|到|置到)?(.+?)的{cn_relation}", text)
        if cn_match:
            subject, reference, relation_text = cn_match.groups()
            relation = next((key for key, aliases in RELATION_ALIASES.items() if relation_text in aliases), "near")
            return self._normalize_query(subject), relation, self._normalize_query(reference)

        en_match = re.search(
            r"(?:put|place|move)\s+(.+?)\s+(?:to\s+)?(?:the\s+)?(left|right|above|below|near|beside|next to)(?:\s+of)?\s+(.+)",
            text.lower(),
        )
        if en_match:
            subject, relation_text, reference = en_match.groups()
            relation = next((key for key, aliases in RELATION_ALIASES.items() if relation_text in aliases), "near")
            return self._normalize_query(subject), relation, self._normalize_query(reference)
        return None

    def resolve_spatial_relations(self, instruction: str, object_table: ObjectTable, intent: TaskIntent) -> list[SpatialRelationBlueprint]:
        relations = list(intent.spatial_relations)
        if not relations:
            extracted = self._extract_spatial_relation_from_text(instruction)
            if extracted:
                subject_query, relation, reference_query = extracted
                relations.append(
                    SpatialRelationBlueprint(
                        relation_id="relation_001",
                        subject_query=subject_query,
                        relation=relation,  # type: ignore[arg-type]
                        reference_query=reference_query,
                        confidence=0.85,
                    )
                )
        resolved: list[SpatialRelationBlueprint] = []
        for idx, relation in enumerate(relations, start=1):
            subject_object_id = self._find_object(object_table, relation.subject_query) or relation.subject_object_id
            reference_object_id = self._find_object(object_table, relation.reference_query) or relation.reference_object_id
            resolved.append(
                relation.model_copy(
                    update={
                        "relation_id": relation.relation_id or f"relation_{idx:03d}",
                        "subject_object_id": subject_object_id,
                        "reference_object_id": reference_object_id,
                    }
                )
            )
        return resolved

    def generate_assignments(self, object_table: ObjectTable, rules: list[Rule]) -> list[Assignment]:
        assignments: list[Assignment] = []
        for idx, obj in enumerate(object_table.objects, start=1):
            matched = None
            for rule in rules:
                if "object_id" in rule.conditions:
                    if obj.object_id == rule.conditions["object_id"]:
                        matched = rule
                        break
                elif "attributes.color" in rule.conditions:
                    if obj.attributes.color == rule.conditions["attributes.color"]:
                        matched = rule
                        break
                elif "attributes.shape" in rule.conditions or "canonical_label" in rule.conditions or "label" in rule.conditions:
                    expected = str(
                        rule.conditions.get("attributes.shape")
                        or rule.conditions.get("canonical_label")
                        or rule.conditions.get("label")
                    )
                    normalized = self._normalize_query(expected)
                    if normalized in self._object_terms(obj):
                        matched = rule
                        break
                elif rule.conditions.get("graspable") is True:
                    matched = rule
                    break
            if matched:
                assignments.append(
                    Assignment(
                        assignment_id=f"as_{idx:03d}",
                        object_id=obj.object_id,
                        rule_id=matched.rule_id,
                        target=matched.action["target"],
                        relation=matched.action.get("relation"),
                        reference_object_id=matched.action.get("reference_object_id"),
                        confidence=max(obj.confidence, 0.25),
                        rationale=f"matched {matched.rule_id}",
                    )
            )
        return assignments

    def build_plan(self, object_table: ObjectTable, intent: TaskIntent, rules: list[Rule], assignments: list[Assignment]) -> Plan:
        plan = Plan(
            task_intent=intent,
            rules=rules,
            assignments=assignments,
            steps=self.generate_steps(object_table, assignments, rules, intent.inspection_policy),
            validation=ValidationReport(status="pass"),
            metadata={},
        )
        plan.validation = self.local_validate(plan, object_table)
        return plan

    def generate_steps(self, object_table: ObjectTable, assignments: list[Assignment], rules: list[Rule], inspection_policy: str) -> list[PlanStep]:
        rule_map = {r.rule_id: r for r in rules}
        steps: list[PlanStep] = []
        order = 1
        uncertain = set(object_table.uncertain_objects)
        for assignment in assignments:
            rule = rule_map[assignment.rule_id]
            need_inspect = assignment.object_id in uncertain or rule.verification_required
            if need_inspect:
                steps.append(
                    PlanStep(
                        step_id=f"step_{order:03d}",
                        action="inspect",
                        object_id=assignment.object_id,
                        arguments={"inspection_policy": inspection_policy},
                        expected_result="object identity and pose verified",
                    )
                )
                order += 1
            steps.append(PlanStep(step_id=f"step_{order:03d}", action="pick", object_id=assignment.object_id))
            order += 1
            arguments = {"placement_mode": "upright"}
            if assignment.relation and assignment.reference_object_id:
                arguments.update(
                    {
                        "relation": assignment.relation,
                        "reference_object_id": assignment.reference_object_id,
                    }
                )
            steps.append(
                PlanStep(
                    step_id=f"step_{order:03d}",
                    action="place",
                    object_id=assignment.object_id,
                    target=assignment.target,
                    relation=assignment.relation,
                    reference_object_id=assignment.reference_object_id,
                    arguments=arguments,
                )
            )
            order += 1
        return steps

    def local_validate(self, plan: Plan, object_table: ObjectTable) -> ValidationReport:
        errors: list[str] = []
        warnings: list[str] = []
        object_ids = {o.object_id for o in object_table.objects}
        rule_ids = {r.rule_id for r in plan.rules}
        step_objects = {s.object_id for s in plan.steps if s.object_id}
        step_objects.update({s.reference_object_id for s in plan.steps if s.reference_object_id})
        if not object_ids:
            errors.append("object_table is empty")
        if not plan.rules:
            errors.append("plan has no rules")
        if not plan.assignments:
            errors.append("plan has no assignments")
        if not plan.steps:
            errors.append("plan has no executable steps")
        for relation in plan.task_intent.spatial_relations:
            if not relation.subject_object_id:
                errors.append(f"unresolved spatial subject: {relation.subject_query}")
            if not relation.reference_object_id:
                errors.append(f"unresolved spatial reference: {relation.reference_query}")
        for assignment in plan.assignments:
            if assignment.object_id not in object_ids:
                errors.append(f"assignment uses unknown object: {assignment.object_id}")
            if assignment.rule_id not in rule_ids:
                errors.append(f"assignment uses unknown rule: {assignment.rule_id}")
            if assignment.relation and not assignment.reference_object_id:
                errors.append(f"assignment missing reference object: {assignment.assignment_id}")
            if assignment.reference_object_id and assignment.reference_object_id not in object_ids:
                errors.append(f"assignment uses unknown reference object: {assignment.reference_object_id}")
        for step in plan.steps:
            if step.object_id and step.object_id not in object_ids:
                errors.append(f"unknown object reference: {step.object_id}")
            if step.reference_object_id and step.reference_object_id not in object_ids:
                errors.append(f"unknown reference object: {step.reference_object_id}")
            if step.action == "place" and not step.target and not step.relation:
                errors.append(f"place step missing target: {step.step_id}")
            if step.action == "place" and step.relation and not step.reference_object_id:
                errors.append(f"relative place step missing reference object: {step.step_id}")
        uncovered = sorted(object_ids - step_objects)
        if uncovered and not plan.task_intent.spatial_relations:
            warnings.extend([f"{oid} not covered by plan steps" for oid in uncovered])
        for oid in object_table.uncertain_objects:
            if not any(s.object_id == oid and s.action == "inspect" for s in plan.steps):
                warnings.append(f"{oid} should be inspected before pick")
        status = "fail" if errors else ("warning" if warnings else "pass")
        return ValidationReport(status=status, warnings=warnings, errors=errors)

    def review_plan(self, instruction: str, object_table: ObjectTable, plan: Plan) -> PlanReview:
        return self.llm.call(
            role="critic",
            task_name="plan_review",
            system_prompt=PLAN_REVIEW_SYSTEM_PROMPT,
            user_payload={
                "instruction": instruction,
                "object_table": compact_for_llm(object_table.model_dump()),
                "plan": plan.model_dump(),
            },
            schema_model=PlanReview,
        )

    @staticmethod
    def apply_review_to_plan(plan: Plan, review: PlanReview) -> Plan:
        if review.verdict == "repair" and not plan.validation.warnings and not plan.validation.errors:
            plan.validation.warnings = review.warnings
            plan.validation.status = "warning"
        if review.verdict == "replan" and not plan.validation.errors:
            plan.validation.warnings.extend(review.warnings or ["critic requests replan"])
            plan.validation.status = "warning"
        return plan

    def create_plan(self, instruction: str, object_table: ObjectTable) -> tuple[TaskIntent, Plan, PlanReview]:
        intent = self.parse_task_intent(instruction, object_table)
        intent = self.prepare_intent(instruction, object_table, intent)
        rules = self.intent_to_rules(intent)
        assignments = self.generate_assignments(object_table, rules)
        provisional = self.build_plan(object_table, intent, rules, assignments)
        review = self.review_plan(instruction, object_table, provisional)
        provisional = self.apply_review_to_plan(provisional, review)
        return intent, provisional, review

    def repair_plan(self, instruction: str, object_table: ObjectTable, plan: Plan, review: PlanReview) -> tuple[Plan, RepairDecision]:
        existing = {a.object_id for a in plan.assignments}
        missing = [o.object_id for o in object_table.objects if o.object_id not in existing]
        decision = self.llm.call(
            role="critic",
            task_name="repair_plan",
            system_prompt=REPAIR_SYSTEM_PROMPT,
            user_payload={
                "instruction": instruction,
                "object_table": compact_for_llm(object_table.model_dump()),
                "plan": plan.model_dump(),
                "review": review.model_dump(),
                "unassigned_object_ids": missing,
            },
            schema_model=RepairDecision,
        )
        assignments_changed = False
        extra_repairs: list[tuple[str, str, str]] = []
        for repair in decision.repairs:
            if repair.kind == "reassign" and repair.object_id and repair.target:
                assignment = next((a for a in plan.assignments if a.object_id == repair.object_id), None)
                if assignment:
                    assignment.target = repair.target
                    assignment.rationale = f"{assignment.rationale}; critic reassigned target"
                    assignments_changed = True
            elif repair.kind in {"inspect", "skip"} and repair.object_id:
                extra_repairs.append((repair.kind, repair.object_id, repair.reason))
        if assignments_changed:
            plan.steps = self.generate_steps(object_table, plan.assignments, plan.rules, plan.task_intent.inspection_policy)
        step_order = len(plan.steps) + 1
        for kind, object_id, reason in extra_repairs:
            if kind == "inspect":
                plan.steps.append(
                    PlanStep(
                        step_id=f"step_{step_order:03d}",
                        action="inspect",
                        object_id=object_id,
                        arguments={"reason": reason},
                        expected_result="object verified after critic repair",
                    )
                )
                step_order += 1
            elif kind == "skip":
                plan.steps.append(
                    PlanStep(
                        step_id=f"step_{step_order:03d}",
                        action="skip",
                        object_id=object_id,
                        arguments={"reason": reason},
                        expected_result="object skipped by critic repair",
                    )
                )
                step_order += 1
        plan.validation = self.local_validate(plan, object_table)
        plan.validation.repaired.append(decision.summary)
        return plan, decision

    def diagnose_failure(
        self,
        *,
        source: str,
        instruction: str,
        object_table: ObjectTable | None,
        plan: Plan | None,
        plan_review: PlanReview | None = None,
        execution_results: list[dict[str, Any]] | None = None,
        dry_run: dict[str, Any] | None = None,
        failed_reason: str | None = None,
        failed_object_id: str | None = None,
        warnings: list[str] | None = None,
    ) -> FailureDiagnosis:
        role = "replan" if source == "execution_feedback" else "critic"
        return self.llm.call(
            role=role,
            task_name="failure_diagnosis",
            system_prompt=FAILURE_DIAGNOSIS_SYSTEM_PROMPT,
            user_payload={
                "source": source,
                "instruction": instruction,
                "object_table": compact_for_llm(object_table.model_dump()) if object_table else None,
                "plan": plan.model_dump() if plan else None,
                "plan_review": plan_review.model_dump() if plan_review else None,
                "execution_results": execution_results or [],
                "dry_run": dry_run,
                "failed_reason": failed_reason,
                "failed_object_id": failed_object_id,
                "warnings": warnings or [],
            },
            schema_model=FailureDiagnosis,
        )

    def replan_from_feedback(self, instruction: str, object_table: ObjectTable, plan: Plan, failed_reason: str, failed_object_id: str | None) -> tuple[Plan, RepairDecision]:
        decision = self.llm.call(
            role="replan",
            task_name="replan",
            system_prompt=REPLAN_SYSTEM_PROMPT,
            user_payload={
                "instruction": instruction,
                "object_table": compact_for_llm(object_table.model_dump()),
                "plan": plan.model_dump(),
                "failed_reason": failed_reason,
                "failed_object_id": failed_object_id,
                "fallback_target": plan.task_intent.fallback_target,
            },
            schema_model=RepairDecision,
        )
        if failed_object_id:
            for assignment in plan.assignments:
                if assignment.object_id == failed_object_id:
                    fallback = next((r.target for r in decision.repairs if r.object_id == failed_object_id and r.target), None)
                    if fallback:
                        assignment.target = fallback
                        assignment.rationale = f"{assignment.rationale}; replanned because {failed_reason}"
        plan.steps = self.generate_steps(object_table, plan.assignments, plan.rules, plan.task_intent.inspection_policy)
        plan.validation = self.local_validate(plan, object_table)
        plan.validation.repaired.append(f"replanned: {decision.summary}")
        return plan, decision
