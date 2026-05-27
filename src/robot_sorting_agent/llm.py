from __future__ import annotations

import json
import re
from typing import Any
import httpx
from pydantic import BaseModel, ValidationError

from .image_utils import image_to_openai_url
from .schemas import ImageInput
from .settings import Settings


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _coerce_score(value: Any, default: float = 0.65) -> float:
    if isinstance(value, int | float):
        return max(0.0, min(float(value), 1.0))
    if isinstance(value, str):
        normalized = value.lower()
        if normalized in {"high", "relevant", "yes", "true"}:
            return 0.9
        if normalized in {"medium", "moderate", "partial"}:
            return 0.65
        if normalized in {"low", "irrelevant", "no", "false"}:
            return 0.25
    return default


def _coerce_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if x is not None]
    if isinstance(value, str):
        return [value]
    return [str(value)]


def _normalize_target(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_") or "default_bin"


def _extract_label_sort_rules(instruction: str) -> tuple[list[str], list[dict[str, Any]]]:
    if any(token in instruction for token in ["左边", "右边", "上方", "下方", "left of", "right of", "above", "below", "near"]):
        return [], []
    bins: list[str] = []
    rules: list[dict[str, Any]] = []
    clauses = [part.strip() for part in re.split(r"\band\b|,|;|，|。", instruction) if part.strip()]
    for clause in clauses:
        match = re.search(r"(?:sort|move|put|place)\s+(?:the\s+)?(.+?)\s+(?:to|into|in)\s+([a-z0-9_\- ]+)$", clause)
        if not match:
            match = re.search(r"(?:把|将)(.+?)(?:放到|放入|放进|移动到|放在)(.+?)(?:里|内)?$", clause)
        if not match:
            continue
        label, target = match.groups()
        label = re.sub(r"^(the|a|an)\s+", "", label.strip().lower())
        label = re.sub(r"(这个|那个|物体|对象|水果|的)", "", label).strip(" .，。")
        target = _normalize_target(target)
        if not label:
            continue
        rule_id = f"rule_{_slug(label) or len(rules) + 1}"
        bins.append(target)
        rules.append(
            {
                "rule_id": rule_id,
                "condition_field": "canonical_label",
                "condition_value": label,
                "target": target,
                "verification_required": "inspect" in instruction or "检查" in instruction,
            }
        )
    return bins, rules


def _repair_vision_extraction(payload: dict[str, Any], user_payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not user_payload:
        return None
    segments = user_payload.get("segments") or []
    if not isinstance(segments, list) or not segments:
        return None
    labels = payload.get("normalized_labels") or payload.get("labels") or []
    attrs = payload.get("semantic_attributes") or payload.get("attributes") or []
    relevance = payload.get("task_relevance") or []
    focus = payload.get("inspection_focus") or []
    objects: list[dict[str, Any]] = []
    for idx, seg in enumerate(segments):
        label = None
        if idx < len(labels):
            label = labels[idx]
        raw_attrs = attrs[idx] if isinstance(attrs, list) and idx < len(attrs) else {}
        if isinstance(raw_attrs, dict):
            label = label or raw_attrs.get("normalized_label") or raw_attrs.get("label")
            color = raw_attrs.get("color")
            shape = raw_attrs.get("shape") or label
            material = raw_attrs.get("material")
            state = raw_attrs.get("state")
            affordances = _coerce_list(raw_attrs.get("affordances")) or ["pick", "place"]
            risk_flags = _coerce_list(raw_attrs.get("risk_flags"))
        else:
            color = None
            shape = label
            material = None
            state = None
            affordances = ["pick", "place"]
            risk_flags = []
        label = str(label or seg.get("label_hint") or "object").strip()
        rel_value = relevance[idx] if isinstance(relevance, list) and idx < len(relevance) else None
        if isinstance(rel_value, dict):
            rel_value = rel_value.get("score") or rel_value.get("relevance")
        focus_value = focus[idx] if isinstance(focus, list) and idx < len(focus) else None
        if isinstance(focus_value, dict):
            focus_value = focus_value.get("focus") or focus_value.get("items")
        objects.append(
            {
                "candidate_id": seg["candidate_id"],
                "normalized_label": label.split()[-1] if " " in label else label,
                "attributes": {
                    "color": color,
                    "material": material,
                    "shape": shape,
                    "state": state,
                    "affordances": affordances,
                    "risk_flags": risk_flags,
                },
                "task_relevance": _coerce_score(rel_value),
                "inspection_focus": _coerce_list(focus_value),
                "confidence": max(_coerce_score(seg.get("confidence"), 0.0), 0.65),
            }
        )
    return {
        "scene_summary": payload.get("scene_summary") or payload.get("summary") or f"{len(objects)} segmented objects",
        "objects": objects,
    }


def _repair_task_intent(payload: dict[str, Any], user_payload: dict[str, Any] | None) -> dict[str, Any]:
    instruction = (user_payload or {}).get("instruction") or payload.get("summary") or "sort objects"
    target = (
        payload.get("fallback_target")
        or payload.get("target")
        or payload.get("destination")
        or payload.get("bin")
        or "default_bin"
    )
    if isinstance(target, list):
        target = target[0] if target else "default_bin"
    target = str(target).lower().replace(" ", "_")
    rule_blueprints = payload.get("rule_blueprints")
    if not isinstance(rule_blueprints, list) or not rule_blueprints:
        rule_blueprints = [
            {
                "rule_id": "rule_default",
                "condition_field": "attributes.color",
                "condition_value": "any",
                "target": target,
                "verification_required": False,
            }
        ]
    target_bins = payload.get("target_bins")
    if not isinstance(target_bins, list) or not target_bins:
        target_bins = [target]
    grouping_axes = payload.get("grouping_axes")
    if not isinstance(grouping_axes, list) or not grouping_axes:
        grouping_axes = ["default"]
    return {
        "goal": payload.get("goal") or "semantic sorting",
        "summary": payload.get("summary") or str(instruction),
        "grouping_axes": grouping_axes,
        "target_bins": target_bins,
        "rule_blueprints": rule_blueprints,
        "inspection_policy": payload.get("inspection_policy") or "inspect low-confidence objects first",
        "fallback_target": payload.get("fallback_target") or "default_bin",
        "confidence": _coerce_score(payload.get("confidence"), 0.8),
    }


def _repair_plan_review(payload: dict[str, Any], user_payload: dict[str, Any] | None) -> dict[str, Any]:
    verdict = payload.get("verdict")
    if verdict not in {"pass", "repair", "replan"}:
        if payload.get("needs_replan") is True:
            verdict = "replan"
        elif payload.get("needs_repair") is True or payload.get("executable") is False or payload.get("safe") is False:
            verdict = "repair"
        else:
            verdict = "pass"
    reason = payload.get("reason") or payload.get("summary") or "Plan review completed."
    warnings = payload.get("warnings")
    if not isinstance(warnings, list):
        warnings = [str(reason)] if verdict != "pass" else []
    errors = payload.get("errors")
    if not isinstance(errors, list):
        errors = []
    reasons = payload.get("reasons")
    if not isinstance(reasons, list):
        reasons = [str(reason)] if verdict != "pass" else []
    return {
        "verdict": verdict,
        "summary": payload.get("summary") or str(reason),
        "warnings": warnings,
        "errors": errors,
        "reasons": reasons,
        "confidence": _coerce_score(payload.get("confidence"), 0.75),
    }


def _repair_tool_policy(payload: dict[str, Any], user_payload: dict[str, Any] | None) -> dict[str, Any]:
    action = payload.get("action") or payload.get("decision") or "run"
    if action not in {"run", "skip", "retry"}:
        action = "run"
    return {
        "action": action,
        "rationale": payload.get("rationale") or payload.get("reason") or "tool policy repaired from model output",
        "confidence": _coerce_score(payload.get("confidence"), 0.75),
    }


def _repair_repair_decision(payload: dict[str, Any], user_payload: dict[str, Any] | None) -> dict[str, Any]:
    raw_repairs = payload.get("repairs") or payload.get("repair_actions") or []
    repairs: list[dict[str, Any]] = []
    if isinstance(raw_repairs, list):
        for item in raw_repairs:
            if not isinstance(item, dict):
                continue
            kind = item.get("kind") or item.get("action") or "inspect"
            if kind not in {"inspect", "reassign", "skip", "reperceive", "clarify"}:
                kind = "clarify" if "clarify" in str(kind).lower() else "inspect"
            repairs.append(
                {
                    "kind": kind,
                    "object_id": item.get("object_id"),
                    "target": item.get("target"),
                    "reason": item.get("reason") or item.get("description") or "repair action from model output",
                }
            )
    return {
        "summary": payload.get("summary") or payload.get("repair_summary") or "Repair decision repaired from model output.",
        "repairs": repairs,
        "confidence": _coerce_score(payload.get("confidence"), 0.75),
    }


def _repair_failure_diagnosis(payload: dict[str, Any], user_payload: dict[str, Any] | None) -> dict[str, Any]:
    source = payload.get("source") or (user_payload or {}).get("source") or "manual"
    if source not in {"plan_review", "execution_feedback", "manual"}:
        source = "manual"

    failed_stage = payload.get("failed_stage") or payload.get("stage") or "unknown"
    stage_aliases = {
        "intent": "intent_parsing",
        "intent_parse": "intent_parsing",
        "planning": "assignment",
        "rules": "rule_generation",
        "rule": "rule_generation",
        "commands": "execution_adapter",
        "dry_run": "execution_adapter",
        "feedback": "execution",
    }
    failed_stage = stage_aliases.get(str(failed_stage), str(failed_stage))
    allowed_stages = {
        "segmentation",
        "perception",
        "intent_parsing",
        "rule_generation",
        "assignment",
        "step_generation",
        "plan_review",
        "repair",
        "execution_adapter",
        "execution",
        "unknown",
    }
    if failed_stage not in allowed_stages:
        failed_stage = "unknown"

    restart_from = payload.get("restart_from") or payload.get("restart_node") or payload.get("retry_from")
    if not restart_from:
        restart_from = {
            "segmentation": "segmentation",
            "perception": "perception",
            "intent_parsing": "parse_intent",
            "rule_generation": "assignment",
            "assignment": "assignment",
            "step_generation": "step_generation",
            "execution_adapter": "execution_adapter",
            "execution": "replan",
        }.get(failed_stage, "repair")
    restart_aliases = {
        "intent": "parse_intent",
        "intent_parsing": "parse_intent",
        "planning": "assignment",
        "rules": "assignment",
        "rule_generation": "assignment",
        "steps": "step_generation",
        "commands": "execution_adapter",
        "dry_run": "execution_adapter",
        "stop": "finish",
    }
    restart_from = restart_aliases.get(str(restart_from), str(restart_from))
    allowed_restarts = {
        "segmentation",
        "perception",
        "parse_intent",
        "assignment",
        "step_generation",
        "repair",
        "replan",
        "execution_adapter",
        "finish",
    }
    if restart_from not in allowed_restarts:
        restart_from = "repair"

    return {
        "source": source,
        "failed_stage": failed_stage,
        "restart_from": restart_from,
        "summary": payload.get("summary") or payload.get("reason") or "Failure diagnosis repaired from model output.",
        "evidence": _coerce_list(payload.get("evidence") or payload.get("reasons")),
        "affected_object_ids": _coerce_list(payload.get("affected_object_ids") or payload.get("object_ids")),
        "confidence": _coerce_score(payload.get("confidence"), 0.70),
    }


def _repair_payload(payload: dict[str, Any], schema_model: type[BaseModel], user_payload: dict[str, Any] | None) -> dict[str, Any] | None:
    model_name = schema_model.__name__
    if model_name in payload and isinstance(payload[model_name], dict):
        payload = payload[model_name]
    if model_name == "VisionExtraction":
        return _repair_vision_extraction(payload, user_payload)
    if model_name == "TaskIntent":
        return _repair_task_intent(payload, user_payload)
    if model_name == "PlanReview":
        return _repair_plan_review(payload, user_payload)
    if model_name == "ToolPolicyDecision":
        return _repair_tool_policy(payload, user_payload)
    if model_name == "RepairDecision":
        return _repair_repair_decision(payload, user_payload)
    if model_name == "FailureDiagnosis":
        return _repair_failure_diagnosis(payload, user_payload)
    return payload


def _json_from_content(content: str, schema_model: type[BaseModel], user_payload: dict[str, Any] | None = None) -> BaseModel:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        return schema_model.model_validate_json(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            snippet = text[start : end + 1]
            try:
                return schema_model.model_validate_json(snippet)
            except ValidationError:
                repaired = _repair_payload(json.loads(snippet), schema_model, user_payload)
                if repaired:
                    return schema_model.model_validate(repaired)
                raise
        raise


def _payload_without_inline_image(payload: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(payload)
    image = cleaned.get("image")
    if isinstance(image, dict):
        image_copy = dict(image)
        if image_copy.get("image_b64"):
            image_copy["image_b64"] = "<attached as image_url data URL>"
        cleaned["image"] = image_copy
    return cleaned


class LLMClient:
    name = "base"

    def structured_invoke(
        self,
        *,
        task_name: str,
        system_prompt: str,
        user_payload: dict[str, Any],
        schema_model: type[BaseModel],
        model: str,
    ) -> BaseModel:
        raise NotImplementedError


class HeuristicLLMClient(LLMClient):
    name = "heuristic"

    def structured_invoke(
        self,
        *,
        task_name: str,
        system_prompt: str,
        user_payload: dict[str, Any],
        schema_model: type[BaseModel],
        model: str,
    ) -> BaseModel:
        instruction = (user_payload.get("instruction") or "").lower()
        if task_name == "vision_extraction":
            objects = []
            for seg in user_payload.get("segments", []):
                label_hint = seg["label_hint"].lower()
                normalized = "cube" if "cube" in label_hint or "block" in label_hint else (
                    "cylinder" if "cylinder" in label_hint else label_hint.split()[-1]
                )
                color = next((c for c in ["red", "blue", "green", "yellow"] if c in label_hint), None)
                task_relevance = 0.95 if color and color in instruction else 0.65
                inspection_focus = ["identity", "pose"] if seg.get("confidence", 0.0) < 0.55 else []
                objects.append(
                    {
                        "candidate_id": seg["candidate_id"],
                        "normalized_label": normalized,
                        "attributes": {
                            "color": color,
                            "shape": normalized,
                            "affordances": ["pick", "place"],
                            "risk_flags": ["fragile"] if "glass" in label_hint else [],
                        },
                        "task_relevance": task_relevance,
                        "inspection_focus": inspection_focus,
                        "confidence": max(seg.get("confidence", 0.0), 0.65),
                    }
                )
            payload = {"scene_summary": f"{len(objects)} segmented objects", "objects": objects}
        elif task_name == "task_intent":
            bins, rule_blueprints = _extract_label_sort_rules(instruction)
            for color, default_bin in [("red", "bin_a"), ("blue", "bin_b"), ("green", "bin_c"), ("yellow", "bin_d")]:
                if color in instruction:
                    bins.append(default_bin)
                    rule_blueprints.append(
                        {
                            "rule_id": f"rule_{color}",
                            "condition_field": "attributes.color",
                            "condition_value": color,
                            "target": default_bin,
                            "verification_required": "inspect" in instruction or "fragile" in instruction,
                        }
                    )
            if not rule_blueprints:
                bins = ["default_bin"]
                rule_blueprints = [
                    {
                        "rule_id": "rule_default",
                        "condition_field": "attributes.color",
                        "condition_value": "any",
                        "target": "default_bin",
                        "verification_required": False,
                    }
                ]
            payload = {
                "goal": "semantic sorting",
                "summary": instruction or "sort objects",
                "grouping_axes": ["color"] if len(rule_blueprints) > 1 else ["default"],
                "target_bins": bins,
                "rule_blueprints": rule_blueprints,
                "inspection_policy": "inspect low-confidence objects first",
                "fallback_target": "default_bin",
                "confidence": 0.90,
            }
        elif task_name == "plan_review":
            plan = user_payload.get("plan", {})
            validation = plan.get("validation") or {}
            errors = validation.get("errors") or []
            warnings = validation.get("warnings") or []
            if errors:
                payload = {
                    "verdict": "repair",
                    "summary": "Plan has structural validation errors.",
                    "warnings": warnings,
                    "errors": errors,
                    "reasons": errors,
                    "confidence": 0.87,
                }
            else:
                payload = {
                    "verdict": "pass",
                    "summary": "Plan is structurally acceptable.",
                    "warnings": warnings,
                    "errors": [],
                    "reasons": warnings,
                    "confidence": 0.90,
                }
        elif task_name == "repair_plan":
            missing = user_payload.get("unassigned_object_ids", [])
            repairs = [{"kind": "inspect", "object_id": oid, "reason": "unassigned_or_uncertain"} for oid in missing]
            payload = {"summary": "Add inspect for unresolved objects.", "repairs": repairs, "confidence": 0.86}
        elif task_name == "tool_policy":
            tool_name = user_payload.get("tool_name", "")
            has_warnings = bool(user_payload.get("warnings"))
            action = "run" if tool_name in {"repair_tool", "execution_feedback_tool"} and has_warnings else "skip"
            payload = {"action": action, "rationale": "heuristic decision", "confidence": 0.80}
        elif task_name == "replan":
            payload = {
                "summary": "Execution feedback indicates corrective replanning is needed.",
                "repairs": [
                    {
                        "kind": "reassign",
                        "object_id": user_payload.get("failed_object_id"),
                        "target": user_payload.get("fallback_target", "default_bin"),
                        "reason": "fallback after execution failure",
                    }
                ],
                "confidence": 0.84,
            }
        elif task_name == "failure_diagnosis":
            source = user_payload.get("source", "manual")
            plan_review = user_payload.get("plan_review") or {}
            failed_reason = str(user_payload.get("failed_reason") or "").lower()
            execution_results = user_payload.get("execution_results") or []
            if source == "execution_feedback":
                blocked_reason = next((str(r.get("reason") or "") for r in execution_results if r.get("requires_replan")), "")
                reason = f"{failed_reason} {blocked_reason}".lower()
                if "unknown object" in reason or "unknown reference" in reason:
                    failed_stage = "assignment"
                    restart_from = "assignment"
                elif "missing reference_object_id" in reason or "missing reference" in reason:
                    failed_stage = "assignment"
                    restart_from = "assignment"
                elif "missing target" in reason or "missing object_id" in reason:
                    failed_stage = "step_generation"
                    restart_from = "step_generation"
                elif "command" in reason and ("serialize" in reason or "schema" in reason or "adapter" in reason):
                    failed_stage = "execution_adapter"
                    restart_from = "execution_adapter"
                else:
                    failed_stage = "execution"
                    restart_from = "replan"
            else:
                text = " ".join(plan_review.get("warnings", []) + plan_review.get("errors", []) + plan_review.get("reasons", [])).lower()
                if "object_table is empty" in text or "missing task objects" in text:
                    failed_stage = "perception"
                    restart_from = "perception"
                elif "plan has no rules" in text:
                    failed_stage = "intent_parsing"
                    restart_from = "parse_intent"
                elif "plan has no assignments" in text or "unresolved spatial" in text:
                    failed_stage = "assignment"
                    restart_from = "assignment"
                elif "plan has no executable steps" in text:
                    failed_stage = "step_generation"
                    restart_from = "step_generation"
                elif "unknown object" in text or "unknown reference" in text:
                    failed_stage = "assignment"
                    restart_from = "assignment"
                elif "missing target" in text or "step" in text:
                    failed_stage = "step_generation"
                    restart_from = "step_generation"
                elif "coverage" in text or "not covered" in text:
                    failed_stage = "assignment"
                    restart_from = "assignment"
                else:
                    failed_stage = "repair"
                    restart_from = "repair"
            payload = {
                "source": source,
                "failed_stage": failed_stage,
                "restart_from": restart_from,
                "summary": f"Heuristic diagnosis routes recovery to {restart_from}.",
                "evidence": [failed_reason] if failed_reason else [],
                "affected_object_ids": _coerce_list(user_payload.get("failed_object_id")),
                "confidence": 0.76,
            }
        else:
            payload = {}
        return schema_model.model_validate(payload)


class OpenAICompatibleLLMClient(LLMClient):
    name = "openai_compatible"

    def __init__(self, *, base_url: str, api_key: str, response_format: str = "json_schema"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.response_format = response_format

    def structured_invoke(
        self,
        *,
        task_name: str,
        system_prompt: str,
        user_payload: dict[str, Any],
        schema_model: type[BaseModel],
        model: str,
    ) -> BaseModel:
        schema = schema_model.model_json_schema()
        user_content = self._build_user_content(user_payload)
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.1,
        }
        response_format = self.response_format.lower()
        if response_format == "json_schema":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": _slug(task_name) or "structured_output",
                    "schema": schema,
                },
            }
        elif response_format == "json_object":
            body["response_format"] = {"type": "json_object"}
        elif response_format == "text":
            body["response_format"] = {"type": "text"}
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=body,
            )
            response.raise_for_status()
            payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return _json_from_content(content, schema_model, user_payload)

    def _build_user_content(self, user_payload: dict[str, Any]) -> str | list[dict[str, Any]]:
        image_payload = user_payload.get("image")
        if not isinstance(image_payload, dict):
            return json.dumps(user_payload, ensure_ascii=False)

        image = ImageInput.model_validate(image_payload)
        text_payload = _payload_without_inline_image(user_payload)
        return [
            {
                "type": "text",
                "text": json.dumps(text_payload, ensure_ascii=False),
            },
            {
                "type": "image_url",
                "image_url": {"url": image_to_openai_url(image)},
            },
        ]


class ModelRouter:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.heuristic_client = HeuristicLLMClient()
        self.clients: dict[str, LLMClient] = {}

    def _endpoint_for_role(self, role: str) -> tuple[str | None, str | None]:
        role_overrides = {
            "vlm": (self.settings.vlm_openai_compat_base_url, self.settings.vlm_openai_compat_api_key),
            "planner": (self.settings.planner_openai_compat_base_url, self.settings.planner_openai_compat_api_key),
            "critic": (self.settings.critic_openai_compat_base_url, self.settings.critic_openai_compat_api_key),
            "tool_policy": (self.settings.tool_policy_openai_compat_base_url, self.settings.tool_policy_openai_compat_api_key),
            "replan": (self.settings.replan_openai_compat_base_url, self.settings.replan_openai_compat_api_key),
        }
        base_url, api_key = role_overrides.get(role, (None, None))
        return base_url or self.settings.openai_compat_base_url, api_key or self.settings.openai_compat_api_key

    def _response_format_for_role(self, role: str) -> str:
        role_overrides = {
            "vlm": self.settings.vlm_openai_compat_response_format,
            "planner": self.settings.planner_openai_compat_response_format,
            "critic": self.settings.critic_openai_compat_response_format,
            "tool_policy": self.settings.tool_policy_openai_compat_response_format,
            "replan": self.settings.replan_openai_compat_response_format,
        }
        return role_overrides.get(role) or self.settings.openai_compat_response_format

    def _client_for_role(self, role: str) -> LLMClient:
        if self.settings.llm_backend != "openai_compatible":
            return self.heuristic_client
        base_url, api_key = self._endpoint_for_role(role)
        if not base_url or not api_key:
            raise ValueError(
                "OPENAI_COMPAT_BASE_URL and OPENAI_COMPAT_API_KEY are required for "
                f"role '{role}', unless role-specific overrides are set."
            )
        response_format = self._response_format_for_role(role)
        cache_key = f"{role}:{base_url}:{response_format}"
        if cache_key not in self.clients:
            self.clients[cache_key] = OpenAICompatibleLLMClient(
                base_url=base_url,
                api_key=api_key,
                response_format=response_format,
            )
        return self.clients[cache_key]

    def call(self, *, role: str, task_name: str, system_prompt: str, user_payload: dict[str, Any], schema_model: type[BaseModel]) -> BaseModel:
        role_to_model = {
            "vlm": self.settings.vlm_model,
            "planner": self.settings.planner_model,
            "critic": self.settings.critic_model,
            "tool_policy": self.settings.tool_policy_model,
            "replan": self.settings.replan_model,
        }
        model = role_to_model.get(role, self.settings.openai_compat_model)
        return self._client_for_role(role).structured_invoke(
            task_name=task_name,
            system_prompt=system_prompt,
            user_payload=user_payload,
            schema_model=schema_model,
            model=model,
        )
