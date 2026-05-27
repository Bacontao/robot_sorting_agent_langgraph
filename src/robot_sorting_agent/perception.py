from __future__ import annotations

from .llm import ModelRouter
from .payloads import compact_for_llm
from .prompts import VISION_SYSTEM_PROMPT
from .schemas import ImageInput, ObjectAttributes, ObjectInstance, ObjectTable, SegmentationCandidate, VisionExtraction
from .settings import Settings


def _canonicalize(label: str) -> str:
    label = label.lower().strip()
    aliases = {"block": "cube", "sphere": "ball", "can": "cylinder"}
    return aliases.get(label, label)


class PerceptionService:
    def __init__(self, settings: Settings, llm: ModelRouter):
        self.settings = settings
        self.llm = llm

    def build_object_table(self, image: ImageInput, instruction: str, segments: list[SegmentationCandidate]) -> ObjectTable:
        hints = self.llm.call(
            role="vlm",
            task_name="vision_extraction",
            system_prompt=VISION_SYSTEM_PROMPT,
            user_payload={
                "image": image.model_dump(),
                "instruction": instruction,
                "segments": compact_for_llm([seg.model_dump() for seg in segments]),
            },
            schema_model=VisionExtraction,
        )
        hint_map = {h.candidate_id: h for h in hints.objects}
        objects = []
        uncertain = []
        for idx, seg in enumerate(segments, start=1):
            hint = hint_map.get(seg.candidate_id)
            normalized = _canonicalize((hint.normalized_label if hint else seg.label_hint).split()[-1])
            attrs = hint.attributes if hint else ObjectAttributes()
            confidence = max(seg.confidence, hint.confidence if hint else 0.0)
            task_relevance = hint.task_relevance if hint else 0.60
            notes = []
            if confidence < self.settings.inspect_threshold:
                uncertain.append(f"obj_{idx:03d}")
                notes.append("inspect before pick because confidence is low")
            if task_relevance < self.settings.task_relevance_threshold:
                notes.append("likely peripheral to current task")
            objects.append(
                ObjectInstance(
                    object_id=f"obj_{idx:03d}",
                    label=seg.label_hint,
                    canonical_label=normalized,
                    attributes=attrs,
                    bbox=seg.bbox,
                    mask=seg.mask,
                    confidence=confidence,
                    task_relevance=task_relevance,
                    execution_ref=f"exec_obj_{idx:03d}",
                    notes=notes,
                )
            )
        return ObjectTable(
            objects=objects,
            uncertain_objects=uncertain,
            scene_summary=hints.scene_summary,
            metadata={"num_segments": len(segments)},
        )
