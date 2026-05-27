from __future__ import annotations

import base64
import mimetypes
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

from .schemas import ImageInput


def _strip_data_url(value: str) -> tuple[str | None, str]:
    if not value.startswith("data:"):
        return None, value
    header, _, payload = value.partition(",")
    mime = header.removeprefix("data:").split(";")[0] or None
    return mime, payload


def _mime_for_path(path: Path) -> str:
    return mimetypes.guess_type(str(path))[0] or "image/png"


def image_to_openai_url(image: ImageInput) -> str:
    if image.image_url:
        return image.image_url
    if image.image_b64:
        mime, payload = _strip_data_url(image.image_b64)
        return f"data:{mime or 'image/png'};base64,{payload}"
    if not image.image_path:
        raise ValueError("Image input is missing an image source.")
    path = Path(image.image_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{_mime_for_path(path)};base64,{payload}"


def image_to_yolo_source(image: ImageInput) -> str | Any:
    if image.image_path:
        path = Path(image.image_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        return str(path)
    if image.image_url:
        return image.image_url
    if not image.image_b64:
        raise ValueError("Image input is missing an image source.")

    _, payload = _strip_data_url(image.image_b64)
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("opencv-python and numpy are required for base64 image segmentation.") from exc

    raw = base64.b64decode(payload)
    array = np.frombuffer(raw, dtype=np.uint8)
    decoded = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("Unable to decode base64 image for segmentation.")
    return decoded


def image_to_local_path(image: ImageInput, *, suffix: str = ".png") -> tuple[str, bool]:
    if image.image_path:
        path = Path(image.image_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        return str(path), False

    if image.image_url:
        with urllib.request.urlopen(image.image_url, timeout=30) as response:  # noqa: S310
            payload = response.read()
        suffix = Path(image.image_url.split("?", 1)[0]).suffix or suffix
        handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            handle.write(payload)
            return handle.name, True
        finally:
            handle.close()

    if not image.image_b64:
        raise ValueError("Image input is missing an image source.")

    mime, payload = _strip_data_url(image.image_b64)
    if mime:
        guessed = mimetypes.guess_extension(mime)
        if guessed:
            suffix = guessed
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        handle.write(base64.b64decode(payload))
        return handle.name, True
    finally:
        handle.close()
