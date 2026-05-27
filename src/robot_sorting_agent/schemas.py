from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ImageInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_path: str | None = None
    image_url: str | None = None
    image_b64: str | None = None

    @model_validator(mode="after")
    def validate_one_source(self) -> "ImageInput":
        used = [x for x in [self.image_path, self.image_url, self.image_b64] if x]
        if len(used) != 1:
            raise ValueError("Exactly one of image_path, image_url, image_b64 must be provided.")
        return self


class PipelineRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image: ImageInput
    instruction: str
    request_id: str | None = None
    execution_feedback: list["ExecutionFeedback"] | None = None


class BoundingBox(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)
    x2: float = Field(ge=0.0, le=1.0)
    y2: float = Field(ge=0.0, le=1.0)


class MaskRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    uri: str
    encoding: Literal["rle", "polygon", "png", "npy", "none"] = "none"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    data: Any | None = None


class ObjectAttributes(BaseModel):
    model_config = ConfigDict(extra="forbid")
    color: str | None = None
    material: str | None = None
    shape: str | None = None
    state: str | None = None
    affordances: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)


class SegmentationCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_id: str
    label_hint: str
    bbox: BoundingBox
    mask: MaskRef | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class VisionObjectHint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_id: str
    normalized_label: str
    attributes: ObjectAttributes = Field(default_factory=ObjectAttributes)
    task_relevance: float = Field(default=0.5, ge=0.0, le=1.0)
    inspection_focus: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class VisionExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scene_summary: str
    objects: list[VisionObjectHint]


class ObjectInstance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    object_id: str
    label: str
    canonical_label: str
    attributes: ObjectAttributes = Field(default_factory=ObjectAttributes)
    bbox: BoundingBox
    mask: MaskRef | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    task_relevance: float = Field(default=0.5, ge=0.0, le=1.0)
    execution_ref: str
    notes: list[str] = Field(default_factory=list)


class ObjectTable(BaseModel):
    model_config = ConfigDict(extra="forbid")
    objects: list[ObjectInstance]
    uncertain_objects: list[str] = Field(default_factory=list)
    scene_summary: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskRuleBlueprint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rule_id: str
    condition_field: str
    condition_value: str
    target: str
    verification_required: bool = False


class SpatialRelationBlueprint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    relation_id: str
    subject_query: str
    relation: Literal["left_of", "right_of", "above", "below", "near"]
    reference_query: str
    subject_object_id: str | None = None
    reference_object_id: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class TaskIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    goal: str
    summary: str
    grouping_axes: list[str] = Field(default_factory=list)
    target_bins: list[str] = Field(default_factory=list)
    rule_blueprints: list[TaskRuleBlueprint] = Field(default_factory=list)
    spatial_relations: list[SpatialRelationBlueprint] = Field(default_factory=list)
    inspection_policy: str = "inspect low-confidence objects first"
    fallback_target: str = "default_bin"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class Rule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rule_id: str
    conditions: dict[str, Any]
    action: dict[str, Any]
    verification_required: bool = False


class Assignment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    assignment_id: str
    object_id: str
    rule_id: str
    target: str
    relation: Literal["left_of", "right_of", "above", "below", "near"] | None = None
    reference_object_id: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = ""


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step_id: str
    action: Literal["inspect", "pick", "place", "skip", "replan"]
    object_id: str | None = None
    target: str | None = None
    relation: Literal["left_of", "right_of", "above", "below", "near"] | None = None
    reference_object_id: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    expected_result: str | None = None


class ValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["pass", "warning", "fail"]
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    repaired: list[str] = Field(default_factory=list)


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_intent: TaskIntent
    rules: list[Rule]
    assignments: list[Assignment]
    steps: list[PlanStep]
    validation: ValidationReport
    metadata: dict[str, Any] = Field(default_factory=dict)


class PlanReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdict: Literal["pass", "repair", "replan"]
    summary: str
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class RepairAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["inspect", "reassign", "skip", "reperceive", "clarify"]
    object_id: str | None = None
    target: str | None = None
    reason: str = ""


class RepairDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str
    repairs: list[RepairAction] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class FailureDiagnosis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: Literal["plan_review", "execution_feedback", "manual"]
    failed_stage: Literal[
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
    ]
    restart_from: Literal[
        "segmentation",
        "perception",
        "parse_intent",
        "assignment",
        "step_generation",
        "repair",
        "replan",
        "execution_adapter",
        "finish",
    ]
    summary: str
    evidence: list[str] = Field(default_factory=list)
    affected_object_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ToolPolicyDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["run", "skip", "retry"]
    rationale: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ExecutionCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    command_id: str
    step_id: str
    action: Literal["inspect", "pick", "place", "skip"]
    object_id: str | None = None
    target: str | None = None
    relation: Literal["left_of", "right_of", "above", "below", "near"] | None = None
    reference_object_id: str | None = None
    object_ref: str | None = None
    reference_object_ref: str | None = None
    frame_id: str = "camera_frame"
    bbox: BoundingBox | None = None
    reference_bbox: BoundingBox | None = None
    mask: MaskRef | None = None
    grasp_planner: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class DryRunEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    command_id: str
    status: Literal["ready", "blocked", "skipped"]
    reason: str | None = None


class DryRunReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    events: list[DryRunEvent]
    ready_ratio: float = Field(default=0.0, ge=0.0, le=1.0)


class ExecutionFeedback(BaseModel):
    model_config = ConfigDict(extra="forbid")
    command_id: str
    status: Literal["succeeded", "failed", "blocked", "uncertain"]
    reason: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    command_id: str
    status: Literal["ready", "succeeded", "failed", "blocked", "skipped"]
    requires_replan: bool = False
    reason: str | None = None


class BenchmarkSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_intent_valid: bool = False
    command_valid: bool = False
    repair_success: bool = False
    feedback_loop_ready: bool = False
    end_to_end_latency_ms: float = 0.0


class AgentTraceStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order: int
    tool_name: str
    status: Literal["success", "skipped", "failed"]
    decision: str
    input_summary: str
    output_summary: str
    latency_ms: float = 0.0
    error: str | None = None


class PipelineResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str
    object_table: ObjectTable | None = None
    task_intent: TaskIntent | None = None
    plan: Plan | None = None
    diagnostics: list[FailureDiagnosis] = Field(default_factory=list)
    execution_commands: list[ExecutionCommand] = Field(default_factory=list)
    dry_run: DryRunReport | None = None
    execution_results: list[ExecutionResult] = Field(default_factory=list)
    needs_replan: bool = False
    replan_count: int = 0
    benchmark: BenchmarkSummary | None = None
    agent_trace: list[AgentTraceStep] = Field(default_factory=list)


PipelineRequest.model_rebuild()
