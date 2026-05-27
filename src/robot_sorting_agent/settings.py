from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def _get_list(name: str, default: list[str]) -> list[str]:
    value = os.getenv(name)
    if value is None:
        return default
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or default


@dataclass(frozen=True)
class Settings:
    app_env: str
    log_level: str
    artifact_dir: Path

    segmentation_backend: str
    segmentation_endpoint: str | None
    segmentation_fallback_chain: list[str]
    segmentation_min_candidates: int
    segmentation_min_confidence: float
    segmentation_fallback_after_failures: int
    segmentation_allow_stub_fallback: bool
    yolo_model_path: str
    yolo_confidence: float
    yolo_image_size: int
    yolo_device: str | None
    grounding_dino_config: str | None
    grounding_dino_checkpoint: str | None
    grounding_dino_box_threshold: float
    grounding_dino_text_threshold: float
    grounding_text_prompt: str | None
    sam_checkpoint: str | None
    sam_model_type: str
    sam_device: str | None

    llm_backend: str
    openai_compat_base_url: str | None
    openai_compat_api_key: str | None
    openai_compat_model: str
    openai_compat_response_format: str

    vlm_openai_compat_base_url: str | None
    vlm_openai_compat_api_key: str | None
    vlm_openai_compat_response_format: str | None
    planner_openai_compat_base_url: str | None
    planner_openai_compat_api_key: str | None
    planner_openai_compat_response_format: str | None
    critic_openai_compat_base_url: str | None
    critic_openai_compat_api_key: str | None
    critic_openai_compat_response_format: str | None
    tool_policy_openai_compat_base_url: str | None
    tool_policy_openai_compat_api_key: str | None
    tool_policy_openai_compat_response_format: str | None
    replan_openai_compat_base_url: str | None
    replan_openai_compat_api_key: str | None
    replan_openai_compat_response_format: str | None

    vlm_model: str
    planner_model: str
    critic_model: str
    tool_policy_model: str
    replan_model: str

    inspect_threshold: float
    task_relevance_threshold: float
    enable_tool_policy: bool
    enable_execution_feedback_loop: bool
    max_replans: int

    @classmethod
    def from_env(cls) -> "Settings":
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except Exception:
            pass
        return cls(
            app_env=os.getenv("APP_ENV", "dev"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            artifact_dir=Path(os.getenv("ARTIFACT_DIR", "./artifacts")),
            segmentation_backend=os.getenv("SEGMENTATION_BACKEND", "auto"),
            segmentation_endpoint=os.getenv("SEGMENTATION_ENDPOINT") or None,
            segmentation_fallback_chain=_get_list("SEGMENTATION_FALLBACK_CHAIN", ["grounded_sam", "yolo"]),
            segmentation_min_candidates=_get_int("SEGMENTATION_MIN_CANDIDATES", 1),
            segmentation_min_confidence=_get_float("SEGMENTATION_MIN_CONFIDENCE", 0.20),
            segmentation_fallback_after_failures=_get_int("SEGMENTATION_FALLBACK_AFTER_FAILURES", 2),
            segmentation_allow_stub_fallback=_get_bool("SEGMENTATION_ALLOW_STUB_FALLBACK", False),
            yolo_model_path=os.getenv("YOLO_MODEL_PATH", "yolo11n-seg.pt"),
            yolo_confidence=_get_float("YOLO_CONFIDENCE", 0.25),
            yolo_image_size=_get_int("YOLO_IMAGE_SIZE", 640),
            yolo_device=os.getenv("YOLO_DEVICE") or None,
            grounding_dino_config=os.getenv("GROUNDING_DINO_CONFIG") or ".models/GroundingDINO_SwinT_OGC.py",
            grounding_dino_checkpoint=os.getenv("GROUNDING_DINO_CHECKPOINT") or ".models/groundingdino_swint_ogc.pth",
            grounding_dino_box_threshold=_get_float("GROUNDING_DINO_BOX_THRESHOLD", 0.30),
            grounding_dino_text_threshold=_get_float("GROUNDING_DINO_TEXT_THRESHOLD", 0.25),
            grounding_text_prompt=os.getenv("GROUNDING_TEXT_PROMPT") or None,
            sam_checkpoint=os.getenv("SAM_CHECKPOINT") or ".models/sam_vit_b_01ec64.pth",
            sam_model_type=os.getenv("SAM_MODEL_TYPE") or "vit_b",
            sam_device=os.getenv("SAM_DEVICE") or os.getenv("YOLO_DEVICE") or None,
            llm_backend=os.getenv("LLM_BACKEND", "heuristic"),
            openai_compat_base_url=os.getenv("OPENAI_COMPAT_BASE_URL") or None,
            openai_compat_api_key=os.getenv("OPENAI_COMPAT_API_KEY") or None,
            openai_compat_model=os.getenv("OPENAI_COMPAT_MODEL", "gpt-4.1-mini"),
            openai_compat_response_format=os.getenv("OPENAI_COMPAT_RESPONSE_FORMAT", "json_schema"),
            vlm_openai_compat_base_url=os.getenv("VLM_OPENAI_COMPAT_BASE_URL") or None,
            vlm_openai_compat_api_key=os.getenv("VLM_OPENAI_COMPAT_API_KEY") or None,
            vlm_openai_compat_response_format=os.getenv("VLM_OPENAI_COMPAT_RESPONSE_FORMAT") or None,
            planner_openai_compat_base_url=os.getenv("PLANNER_OPENAI_COMPAT_BASE_URL") or None,
            planner_openai_compat_api_key=os.getenv("PLANNER_OPENAI_COMPAT_API_KEY") or None,
            planner_openai_compat_response_format=os.getenv("PLANNER_OPENAI_COMPAT_RESPONSE_FORMAT") or None,
            critic_openai_compat_base_url=os.getenv("CRITIC_OPENAI_COMPAT_BASE_URL") or None,
            critic_openai_compat_api_key=os.getenv("CRITIC_OPENAI_COMPAT_API_KEY") or None,
            critic_openai_compat_response_format=os.getenv("CRITIC_OPENAI_COMPAT_RESPONSE_FORMAT") or None,
            tool_policy_openai_compat_base_url=os.getenv("TOOL_POLICY_OPENAI_COMPAT_BASE_URL") or None,
            tool_policy_openai_compat_api_key=os.getenv("TOOL_POLICY_OPENAI_COMPAT_API_KEY") or None,
            tool_policy_openai_compat_response_format=os.getenv("TOOL_POLICY_OPENAI_COMPAT_RESPONSE_FORMAT") or None,
            replan_openai_compat_base_url=os.getenv("REPLAN_OPENAI_COMPAT_BASE_URL") or None,
            replan_openai_compat_api_key=os.getenv("REPLAN_OPENAI_COMPAT_API_KEY") or None,
            replan_openai_compat_response_format=os.getenv("REPLAN_OPENAI_COMPAT_RESPONSE_FORMAT") or None,
            vlm_model=os.getenv("VLM_MODEL", "qwen2.5-vl-7b-instruct"),
            planner_model=os.getenv("PLANNER_MODEL", "gpt-4.1-mini"),
            critic_model=os.getenv("CRITIC_MODEL", "gpt-4.1-mini"),
            tool_policy_model=os.getenv("TOOL_POLICY_MODEL", "gpt-4.1-mini"),
            replan_model=os.getenv("REPLAN_MODEL", "gpt-4.1-mini"),
            inspect_threshold=_get_float("INSPECT_THRESHOLD", 0.55),
            task_relevance_threshold=_get_float("TASK_RELEVANCE_THRESHOLD", 0.40),
            enable_tool_policy=_get_bool("ENABLE_TOOL_POLICY", True),
            enable_execution_feedback_loop=_get_bool("ENABLE_EXECUTION_FEEDBACK_LOOP", True),
            max_replans=_get_int("MAX_REPLANS", 2),
        )
