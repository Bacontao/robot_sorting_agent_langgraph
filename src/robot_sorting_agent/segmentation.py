from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol

from .image_utils import image_to_local_path, image_to_yolo_source
from .schemas import BoundingBox, ImageInput, MaskRef, SegmentationCandidate
from .settings import Settings


class SegmentationBackend(Protocol):
    def segment(self, image: ImageInput, instruction: str) -> list[SegmentationCandidate]:
        ...


class StubSegmentationBackend:
    name = "stub"

    def segment(self, image: ImageInput, instruction: str) -> list[SegmentationCandidate]:
        tokens = []
        text = instruction.lower()
        color_specs = [
            ("red", "cube", ["red", "红"]),
            ("blue", "cylinder", ["blue", "蓝"]),
            ("green", "bottle", ["green", "绿"]),
            ("yellow", "box", ["yellow", "黄"]),
        ]
        for color, shape, aliases in color_specs:
            if any(alias in text for alias in aliases):
                tokens.append(f"{color} {shape}")
        if not tokens:
            tokens = ["object candidate"]
        candidates = []
        for idx, label in enumerate(tokens, start=1):
            x1 = 0.08 * idx
            y1 = 0.12
            x2 = min(x1 + 0.18, 0.95)
            y2 = 0.45
            candidates.append(
                SegmentationCandidate(
                    candidate_id=f"cand_{idx:03d}",
                    label_hint=label,
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    mask=MaskRef(uri=f"memory://mask/{idx}", encoding="none", confidence=0.45),
                    confidence=0.45,
                )
            )
        return candidates


class YoloSegmentationBackend:
    name = "yolo"

    def __init__(self, settings: Settings):
        self.settings = settings
        cache_root = Path(".cache")
        yolo_config_dir = Path(os.environ.setdefault("YOLO_CONFIG_DIR", str(cache_root / "ultralytics")))
        matplotlib_config_dir = Path(os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib")))
        yolo_config_dir.mkdir(parents=True, exist_ok=True)
        matplotlib_config_dir.mkdir(parents=True, exist_ok=True)
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Ultralytics is required for YOLO backend.") from exc
        self.model = YOLO(settings.yolo_model_path)

    def segment(self, image: ImageInput, instruction: str) -> list[SegmentationCandidate]:
        source = image_to_yolo_source(image)
        predict_kwargs = {
            "source": source,
            "conf": self.settings.yolo_confidence,
            "imgsz": self.settings.yolo_image_size,
            "verbose": False,
        }
        if self.settings.yolo_device:
            predict_kwargs["device"] = self.settings.yolo_device
        results = self.model.predict(**predict_kwargs)
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        height, width = result.orig_shape[:2]
        names = getattr(self.model, "names", {}) or {}
        masks = getattr(result, "masks", None)
        polygons = getattr(masks, "xyn", None) if masks is not None else None

        candidates: list[SegmentationCandidate] = []
        for idx, box in enumerate(boxes, start=1):
            xyxy = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            if isinstance(names, dict):
                label = str(names.get(class_id, f"class_{class_id}"))
            else:
                label = str(names[class_id]) if class_id < len(names) else f"class_{class_id}"

            x1 = max(0.0, min(float(xyxy[0]) / width, 1.0))
            y1 = max(0.0, min(float(xyxy[1]) / height, 1.0))
            x2 = max(0.0, min(float(xyxy[2]) / width, 1.0))
            y2 = max(0.0, min(float(xyxy[3]) / height, 1.0))

            polygon = None
            if polygons is not None and idx - 1 < len(polygons):
                polygon = polygons[idx - 1].tolist()

            candidates.append(
                SegmentationCandidate(
                    candidate_id=f"cand_{idx:03d}",
                    label_hint=label,
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    mask=MaskRef(
                        uri=f"memory://yolo/mask/{idx}",
                        encoding="polygon" if polygon is not None else "none",
                        confidence=confidence,
                        data={"points": polygon} if polygon is not None else None,
                    ),
                    confidence=confidence,
                )
            )
        return candidates


class GroundedSamSegmentationBackend:
    """Open-vocabulary segmentation backend using GroundingDINO boxes and SAM masks."""

    name = "grounded_sam"

    def __init__(self, settings: Settings):
        self.settings = settings
        try:
            import cv2  # type: ignore
            import groundingdino  # type: ignore
            import torch  # type: ignore
            from groundingdino.util.inference import load_image, load_model, predict  # type: ignore
            from segment_anything import SamPredictor, sam_model_registry  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "GroundingDINO + SAM backend requires groundingdino, segment-anything, torch, "
                "opencv-python, and their model checkpoints."
            ) from exc

        self.cv2 = cv2
        self.torch = torch
        self.groundingdino = groundingdino
        self.load_image = load_image
        self.load_model = load_model
        self.predict = predict
        self.device = settings.sam_device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.grounding_dino_config = self._resolve_grounding_config(settings.grounding_dino_config)
        self.grounding_dino_checkpoint = self._require_file(settings.grounding_dino_checkpoint, "GROUNDING_DINO_CHECKPOINT")
        self.sam_checkpoint = self._require_file(settings.sam_checkpoint, "SAM_CHECKPOINT")
        self.dino_model = self._load_dino_model()
        sam = sam_model_registry[settings.sam_model_type](checkpoint=self.sam_checkpoint)
        sam.to(device=self.device)
        self.sam_predictor = SamPredictor(sam)

    def _resolve_grounding_config(self, configured: str | None) -> str:
        candidates = []
        if configured:
            candidates.append(Path(configured).expanduser())
        candidates.append(Path(self.groundingdino.__file__).resolve().parent / "config" / "GroundingDINO_SwinT_OGC.py")
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        raise FileNotFoundError(
            "GroundingDINO config file not found. Set GROUNDING_DINO_CONFIG or run "
            "`python scripts/download_grounded_sam_models.py`."
        )

    @staticmethod
    def _require_file(value: str | None, env_name: str) -> str:
        if not value:
            raise ValueError(f"{env_name} is required when SEGMENTATION_BACKEND=grounded_sam.")
        path = Path(value).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"{env_name} file not found: {path}")
        return str(path)

    def _load_dino_model(self) -> Any:
        try:
            return self.load_model(self.grounding_dino_config, self.grounding_dino_checkpoint, device=self.device)
        except TypeError:
            model = self.load_model(self.grounding_dino_config, self.grounding_dino_checkpoint)
            if hasattr(model, "to"):
                model.to(self.device)
            return model

    def _predict_grounding(self, image_tensor: Any, caption: str) -> tuple[Any, Any, Any]:
        try:
            return self.predict(
                model=self.dino_model,
                image=image_tensor,
                caption=caption,
                box_threshold=self.settings.grounding_dino_box_threshold,
                text_threshold=self.settings.grounding_dino_text_threshold,
                device=self.device,
            )
        except TypeError:
            return self.predict(
                model=self.dino_model,
                image=image_tensor,
                caption=caption,
                box_threshold=self.settings.grounding_dino_box_threshold,
                text_threshold=self.settings.grounding_dino_text_threshold,
            )

    @staticmethod
    def _caption(instruction: str, configured_prompt: str | None) -> str:
        caption = (configured_prompt or instruction or "objects").strip()
        if not caption.endswith("."):
            caption = f"{caption}."
        return caption

    def _boxes_to_xyxy_pixels(self, boxes: Any, width: int, height: int) -> Any:
        if not hasattr(boxes, "device"):
            boxes = self.torch.as_tensor(boxes, dtype=self.torch.float32, device=self.device)
        else:
            boxes = boxes.to(self.device)
        scale = self.torch.tensor([width, height, width, height], device=boxes.device)
        scaled = boxes * scale
        cx, cy, box_w, box_h = scaled.unbind(-1)
        return self.torch.stack(
            [
                cx - box_w / 2,
                cy - box_h / 2,
                cx + box_w / 2,
                cy + box_h / 2,
            ],
            dim=-1,
        )

    def _mask_polygon(self, mask: Any, width: int, height: int) -> list[list[float]] | None:
        mask_uint8 = mask.astype("uint8")
        contours, _ = self.cv2.findContours(mask_uint8, self.cv2.RETR_EXTERNAL, self.cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=self.cv2.contourArea).reshape(-1, 2)
        if len(contour) < 3:
            return None
        if len(contour) > 128:
            step = max(len(contour) // 128, 1)
            contour = contour[::step]
        return [[float(x) / width, float(y) / height] for x, y in contour]

    def segment(self, image: ImageInput, instruction: str) -> list[SegmentationCandidate]:
        image_path, remove_after = image_to_local_path(image)
        try:
            caption = self._caption(instruction, self.settings.grounding_text_prompt)
            image_source, image_tensor = self.load_image(image_path)
            height, width = image_source.shape[:2]
            boxes, logits, phrases = self._predict_grounding(image_tensor, caption)
            if boxes is None or len(boxes) == 0:
                return []

            boxes_xyxy = self._boxes_to_xyxy_pixels(boxes, width, height)
            self.sam_predictor.set_image(image_source)
            transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(self.device)
            masks, scores, _ = self.sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )

            candidates: list[SegmentationCandidate] = []
            for idx, box in enumerate(boxes_xyxy.detach().cpu().tolist(), start=1):
                x1 = max(0.0, min(float(box[0]) / width, 1.0))
                y1 = max(0.0, min(float(box[1]) / height, 1.0))
                x2 = max(0.0, min(float(box[2]) / width, 1.0))
                y2 = max(0.0, min(float(box[3]) / height, 1.0))
                mask = masks[idx - 1][0].detach().cpu().numpy()
                polygon = self._mask_polygon(mask, width, height)
                confidence = float(logits[idx - 1]) if idx - 1 < len(logits) else float(scores[idx - 1][0])
                confidence = max(0.0, min(confidence, 1.0))
                label = str(phrases[idx - 1]) if idx - 1 < len(phrases) else "grounded_object"
                candidates.append(
                    SegmentationCandidate(
                        candidate_id=f"cand_{idx:03d}",
                        label_hint=label,
                        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                        mask=MaskRef(
                            uri=f"memory://grounded_sam/mask/{idx}",
                            encoding="polygon" if polygon is not None else "none",
                            confidence=confidence,
                            data={"points": polygon} if polygon is not None else None,
                        ),
                        confidence=confidence,
                    )
                )
            return candidates
        finally:
            if remove_after:
                Path(image_path).unlink(missing_ok=True)


class AutoFallbackSegmentationBackend:
    """Prefer GroundingDINO + SAM, then fall back to YOLO when the primary backend is unavailable or weak."""

    name = "auto"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.backends: dict[str, SegmentationBackend] = {}
        self.failure_counts: dict[str, int] = {}
        self.last_backend_name: str | None = None
        self.last_attempts: list[dict[str, Any]] = []
        self.chain = self._normalize_chain(settings.segmentation_fallback_chain)
        if settings.segmentation_allow_stub_fallback and "stub" not in self.chain:
            self.chain.append("stub")

    @staticmethod
    def _normalize_backend_name(name: str) -> str:
        normalized = name.strip().lower().replace("-", "_")
        aliases = {
            "groundingdino_sam": "grounded_sam",
            "grounding_dino_sam": "grounded_sam",
            "sam": "grounded_sam",
        }
        return aliases.get(normalized, normalized)

    def _normalize_chain(self, chain: list[str]) -> list[str]:
        normalized: list[str] = []
        for backend in chain:
            name = self._normalize_backend_name(backend)
            if name == "auto":
                continue
            if name not in {"grounded_sam", "yolo", "stub"}:
                continue
            if name not in normalized:
                normalized.append(name)
        return normalized or ["grounded_sam", "yolo"]

    def _ordered_chain(self) -> list[str]:
        chain = list(self.chain)
        if not chain:
            return []
        primary = chain[0]
        if self.failure_counts.get(primary, 0) >= self.settings.segmentation_fallback_after_failures:
            chain = chain[1:] + [primary]
        return chain

    def _get_backend(self, name: str) -> SegmentationBackend:
        if name not in self.backends:
            if name == "grounded_sam":
                self.backends[name] = GroundedSamSegmentationBackend(self.settings)
            elif name == "yolo":
                self.backends[name] = YoloSegmentationBackend(self.settings)
            elif name == "stub":
                self.backends[name] = StubSegmentationBackend()
            else:
                raise ValueError(f"Unsupported segmentation backend in fallback chain: {name}")
        return self.backends[name]

    def _quality(self, candidates: list[SegmentationCandidate]) -> tuple[bool, float]:
        if len(candidates) < self.settings.segmentation_min_candidates:
            return False, 0.0
        max_confidence = max((candidate.confidence for candidate in candidates), default=0.0)
        return max_confidence >= self.settings.segmentation_min_confidence, max_confidence

    def _mark_failure(self, name: str) -> None:
        self.failure_counts[name] = self.failure_counts.get(name, 0) + 1

    def _mark_success(self, name: str) -> None:
        self.failure_counts[name] = 0

    def segment(self, image: ImageInput, instruction: str) -> list[SegmentationCandidate]:
        self.last_backend_name = None
        self.last_attempts = []
        best_name: str | None = None
        best_candidates: list[SegmentationCandidate] = []
        best_confidence = -1.0
        errors: list[str] = []

        for name in self._ordered_chain():
            try:
                backend = self._get_backend(name)
                candidates = backend.segment(image, instruction)
                usable, max_confidence = self._quality(candidates)
                self.last_attempts.append(
                    {
                        "backend": name,
                        "status": "success" if usable else "weak",
                        "num_candidates": len(candidates),
                        "max_confidence": round(max_confidence, 4),
                    }
                )
                if max_confidence > best_confidence or (max_confidence == best_confidence and len(candidates) > len(best_candidates)):
                    best_name = name
                    best_candidates = candidates
                    best_confidence = max_confidence
                if usable:
                    self._mark_success(name)
                    self.last_backend_name = name
                    return candidates
                self._mark_failure(name)
            except Exception as exc:
                self._mark_failure(name)
                message = f"{name}: {exc}"
                errors.append(message)
                self.last_attempts.append({"backend": name, "status": "failed", "error": str(exc)})

        if best_name is not None:
            self.last_backend_name = best_name
            return best_candidates
        if errors:
            raise RuntimeError("All segmentation backends failed: " + " | ".join(errors))
        self.last_backend_name = self.chain[-1] if self.chain else None
        return []


def build_segmentation_backend(settings: Settings) -> SegmentationBackend:
    backend = AutoFallbackSegmentationBackend._normalize_backend_name(settings.segmentation_backend)
    if backend == "auto":
        return AutoFallbackSegmentationBackend(settings)
    if backend == "yolo":
        return YoloSegmentationBackend(settings)
    if backend == "grounded_sam":
        return GroundedSamSegmentationBackend(settings)
    return StubSegmentationBackend()
