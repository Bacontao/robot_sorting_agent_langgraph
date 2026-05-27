import base64

from robot_sorting_agent.graph import WorkflowRuntime
from robot_sorting_agent.image_utils import image_to_local_path
from robot_sorting_agent.planning import PlanningService
from robot_sorting_agent.llm import ModelRouter
from robot_sorting_agent.schemas import BoundingBox, FailureDiagnosis, MaskRef, ObjectAttributes, ObjectInstance, ObjectTable, PipelineRequest, PlanReview, SegmentationCandidate
from robot_sorting_agent.segmentation import AutoFallbackSegmentationBackend
from robot_sorting_agent.settings import Settings


def test_smoke_pipeline(monkeypatch):
    monkeypatch.setenv("SEGMENTATION_BACKEND", "stub")
    monkeypatch.setenv("LLM_BACKEND", "heuristic")
    runtime = WorkflowRuntime(Settings.from_env())
    request = PipelineRequest.model_validate(
        {
            "image": {"image_path": "samples/demo.png"},
            "instruction": "Sort the red object to bin_a and the blue object to bin_b.",
        }
    )
    response = runtime.invoke(request)
    assert response.object_table is not None
    assert response.plan is not None
    assert response.execution_commands
    assert response.dry_run is not None


def test_spatial_relation_plan(monkeypatch):
    monkeypatch.setenv("SEGMENTATION_BACKEND", "stub")
    monkeypatch.setenv("LLM_BACKEND", "heuristic")
    settings = Settings.from_env()
    service = PlanningService(ModelRouter(settings))
    object_table = ObjectTable(
        scene_summary="peach and strawberry",
        objects=[
            ObjectInstance(
                object_id="obj_001",
                label="peach",
                canonical_label="peach",
                attributes=ObjectAttributes(shape="peach", affordances=["pick", "place"]),
                bbox=BoundingBox(x1=0.2, y1=0.2, x2=0.4, y2=0.5),
                confidence=0.9,
                task_relevance=0.9,
                execution_ref="exec_obj_001",
            ),
            ObjectInstance(
                object_id="obj_002",
                label="strawberry",
                canonical_label="strawberry",
                attributes=ObjectAttributes(shape="strawberry", affordances=["pick", "place"]),
                bbox=BoundingBox(x1=0.6, y1=0.2, x2=0.8, y2=0.5),
                confidence=0.9,
                task_relevance=0.95,
                execution_ref="exec_obj_002",
            ),
        ],
    )

    _intent, plan, _review = service.create_plan("把草莓放在桃子的左边", object_table)

    place_steps = [step for step in plan.steps if step.action == "place"]
    assert place_steps
    assert place_steps[0].object_id == "obj_002"
    assert place_steps[0].relation == "left_of"
    assert place_steps[0].reference_object_id == "obj_001"


def test_unresolved_spatial_relation_fails_validation(monkeypatch):
    monkeypatch.setenv("SEGMENTATION_BACKEND", "stub")
    monkeypatch.setenv("LLM_BACKEND", "heuristic")
    settings = Settings.from_env()
    service = PlanningService(ModelRouter(settings))
    object_table = ObjectTable(
        scene_summary="only peach is visible",
        objects=[
            ObjectInstance(
                object_id="obj_001",
                label="peach",
                canonical_label="peach",
                attributes=ObjectAttributes(shape="peach", affordances=["pick", "place"]),
                bbox=BoundingBox(x1=0.2, y1=0.2, x2=0.4, y2=0.5),
                confidence=0.9,
                task_relevance=0.9,
                execution_ref="exec_obj_001",
            )
        ],
    )

    _intent, plan, _review = service.create_plan("把牛油果放在桃子的左边", object_table)

    assert plan.validation.status == "fail"
    assert not plan.steps
    assert any("unresolved spatial subject" in error for error in plan.validation.errors)
    assert any("plan has no assignments" in error for error in plan.validation.errors)


def test_diagnosis_can_route_back_to_assignment(monkeypatch):
    monkeypatch.setenv("SEGMENTATION_BACKEND", "stub")
    monkeypatch.setenv("LLM_BACKEND", "heuristic")
    runtime = WorkflowRuntime(Settings.from_env())
    review_calls = {"count": 0}

    def fake_review_plan(instruction, object_table, plan):
        review_calls["count"] += 1
        if review_calls["count"] == 1:
            return PlanReview(
                verdict="repair",
                summary="Assignment does not match the requested rule.",
                warnings=["assignment mismatch"],
                errors=[],
                reasons=["assignment"],
                confidence=0.9,
            )
        return PlanReview(verdict="pass", summary="Recovered after assignment rerun.", confidence=0.9)

    def fake_diagnose_failure(**kwargs):
        return FailureDiagnosis(
            source="plan_review",
            failed_stage="assignment",
            restart_from="assignment",
            summary="The plan issue is in assignment, so rerun assignment.",
            evidence=["assignment mismatch"],
            affected_object_ids=[],
            confidence=0.92,
        )

    monkeypatch.setattr(runtime.planning, "review_plan", fake_review_plan)
    monkeypatch.setattr(runtime.planning, "diagnose_failure", fake_diagnose_failure)

    request = PipelineRequest.model_validate(
        {
            "image": {"image_path": "samples/demo.png"},
            "instruction": "Sort the red object to bin_a and the blue object to bin_b.",
        }
    )
    response = runtime.invoke(request)
    trace_tools = [step.tool_name for step in response.agent_trace]

    assert response.diagnostics
    assert response.diagnostics[0].restart_from == "assignment"
    assert trace_tools.count("assignment_tool") == 2
    assert "repair_tool" not in trace_tools
    assert response.plan is not None
    assert response.benchmark is not None
    assert response.benchmark.repair_success is True


def test_repair_is_reviewed_before_execution(monkeypatch):
    monkeypatch.setenv("SEGMENTATION_BACKEND", "stub")
    monkeypatch.setenv("LLM_BACKEND", "heuristic")
    runtime = WorkflowRuntime(Settings.from_env())
    review_calls = {"count": 0}

    def fake_review_plan(instruction, object_table, plan):
        review_calls["count"] += 1
        if review_calls["count"] == 1:
            return PlanReview(
                verdict="repair",
                summary="Needs a local repair.",
                warnings=["local repair needed"],
                reasons=["repair"],
                confidence=0.9,
            )
        return PlanReview(verdict="pass", summary="Repair was reviewed and accepted.", confidence=0.9)

    def fake_diagnose_failure(**kwargs):
        return FailureDiagnosis(
            source="plan_review",
            failed_stage="repair",
            restart_from="repair",
            summary="A local repair is enough.",
            evidence=["local repair needed"],
            confidence=0.9,
        )

    monkeypatch.setattr(runtime.planning, "review_plan", fake_review_plan)
    monkeypatch.setattr(runtime.planning, "diagnose_failure", fake_diagnose_failure)
    request = PipelineRequest.model_validate(
        {
            "image": {"image_path": "samples/demo.png"},
            "instruction": "Sort the red object to bin_a.",
        }
    )

    response = runtime.invoke(request)
    trace_tools = [step.tool_name for step in response.agent_trace]

    assert trace_tools.count("plan_review_tool") == 2
    repair_index = trace_tools.index("repair_tool")
    second_review_index = [idx for idx, tool in enumerate(trace_tools) if tool == "plan_review_tool"][1]
    execution_index = trace_tools.index("execution_adapter_tool")
    assert repair_index < second_review_index < execution_index


def test_execution_feedback_diagnosis_routes_reference_errors_to_assignment(monkeypatch):
    monkeypatch.setenv("SEGMENTATION_BACKEND", "stub")
    monkeypatch.setenv("LLM_BACKEND", "heuristic")
    settings = Settings.from_env()
    service = PlanningService(ModelRouter(settings))

    diagnosis = service.diagnose_failure(
        source="execution_feedback",
        instruction="把草莓放在桃子的左边",
        object_table=None,
        plan=None,
        execution_results=[
            {
                "command_id": "cmd_001",
                "status": "blocked",
                "requires_replan": True,
                "reason": "unknown reference_object_id",
            }
        ],
        failed_reason="unknown reference_object_id",
        failed_object_id="obj_002",
    )

    assert diagnosis.failed_stage == "assignment"
    assert diagnosis.restart_from == "assignment"


def test_grounded_sam_defaults_to_vit_b(monkeypatch):
    monkeypatch.setenv("SAM_MODEL_TYPE", "")
    monkeypatch.setenv("SAM_CHECKPOINT", "")
    settings = Settings.from_env()

    assert settings.sam_model_type == "vit_b"
    assert settings.sam_checkpoint == ".models/sam_vit_b_01ec64.pth"


def test_image_to_local_path_supports_base64_temp_file():
    payload = base64.b64encode(b"not-a-real-image-but-valid-bytes").decode("ascii")
    path, remove_after = image_to_local_path(PipelineRequest.model_validate({"image": {"image_b64": payload}, "instruction": "x"}).image)

    try:
        assert remove_after is True
        with open(path, "rb") as handle:
            assert handle.read() == b"not-a-real-image-but-valid-bytes"
    finally:
        import os

        os.unlink(path)


def test_auto_segmentation_falls_back_to_yolo(monkeypatch):
    monkeypatch.setenv("SEGMENTATION_BACKEND", "auto")
    settings = Settings.from_env()
    backend = AutoFallbackSegmentationBackend(settings)

    class FailingBackend:
        def segment(self, image, instruction):
            raise RuntimeError("missing checkpoint")

    class WorkingBackend:
        def segment(self, image, instruction):
            return [
                SegmentationCandidate(
                    candidate_id="cand_001",
                    label_hint="bus",
                    bbox=BoundingBox(x1=0.1, y1=0.1, x2=0.5, y2=0.5),
                    mask=MaskRef(uri="memory://yolo/mask/1", encoding="none", confidence=0.9),
                    confidence=0.9,
                )
            ]

    def fake_get_backend(name):
        return FailingBackend() if name == "grounded_sam" else WorkingBackend()

    monkeypatch.setattr(backend, "_get_backend", fake_get_backend)
    candidates = backend.segment(PipelineRequest.model_validate({"image": {"image_path": "samples/demo.png"}, "instruction": "x"}).image, "x")

    assert candidates[0].label_hint == "bus"
    assert backend.last_backend_name == "yolo"
    assert backend.last_attempts[0]["status"] == "failed"
    assert backend.last_attempts[1]["status"] == "success"
