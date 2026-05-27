from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "red": (220, 50, 47),
    "blue": (38, 139, 210),
    "green": (80, 161, 79),
    "yellow": (230, 190, 60),
}

TARGETS = {
    "red": "bin_a",
    "blue": "bin_b",
    "green": "bin_c",
    "yellow": "bin_d",
}

RELATIONS = ["left_of", "right_of", "above", "below", "near"]
RELATION_TEXT = {
    "left_of": "to the left of",
    "right_of": "to the right of",
    "above": "above",
    "below": "below",
    "near": "near",
}


def _draw_object(draw: ImageDraw.ImageDraw, color: str, box: tuple[int, int, int, int], shape: str, font: ImageFont.ImageFont) -> None:
    fill = COLORS[color]
    if shape == "circle":
        draw.ellipse(box, fill=fill, outline=(30, 30, 30), width=4)
    else:
        draw.rounded_rectangle(box, radius=12, fill=fill, outline=(30, 30, 30), width=4)
    x1, y1, x2, y2 = box
    draw.text((x1 + 8, y2 + 6), f"{color} object", fill=(20, 20, 20), font=font)


def _make_image(path: Path, objects: list[dict[str, Any]]) -> None:
    image = Image.new("RGB", (640, 420), (246, 246, 238))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.rectangle((24, 24, 616, 356), outline=(190, 190, 180), width=2)
    draw.text((28, 362), "synthetic tabletop scene", fill=(100, 100, 100), font=font)
    for obj in objects:
        _draw_object(draw, obj["color"], obj["box"], obj["shape"], font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _layout(seed: int, colors: list[str]) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    slots = [
        (70, 70, 190, 190),
        (260, 70, 380, 190),
        (450, 70, 570, 190),
        (150, 220, 270, 340),
        (360, 220, 480, 340),
    ]
    rng.shuffle(slots)
    shapes = ["square", "circle"]
    return [
        {
            "color": color,
            "shape": shapes[(idx + seed) % len(shapes)],
            "box": slots[idx],
        }
        for idx, color in enumerate(colors)
    ]


def _color_case(idx: int, image_path: Path, colors: list[str]) -> dict[str, Any]:
    instruction = " and ".join(f"sort the {color} object to {TARGETS[color]}" for color in colors[:2])
    instruction = instruction[0].upper() + instruction[1:] + "."
    return {
        "case_id": f"eval_{idx:03d}",
        "image": {"image_path": str(image_path)},
        "instruction": instruction,
        "expected": {
            "objects": [{"color": color} for color in colors[:2]],
            "assignments": [{"object": {"color": color}, "target": TARGETS[color]} for color in colors[:2]],
            "relations": [],
            "min_commands": 4,
        },
    }


def _spatial_case(idx: int, image_path: Path, colors: list[str], relation: str) -> dict[str, Any]:
    subject, reference = colors[0], colors[1]
    return {
        "case_id": f"eval_{idx:03d}",
        "image": {"image_path": str(image_path)},
        "instruction": f"Place the {subject} object {RELATION_TEXT[relation]} the {reference} object.",
        "expected": {
            "objects": [{"color": subject}, {"color": reference}],
            "assignments": [],
            "relations": [
                {
                    "subject": {"color": subject},
                    "relation": relation,
                    "reference": {"color": reference},
                }
            ],
            "min_commands": 2,
        },
    }


def generate_cases(count: int, image_dir: Path) -> list[dict[str, Any]]:
    palette = list(COLORS)
    cases = []
    for idx in range(count):
        rng = random.Random(2026 + idx)
        colors = palette[:]
        rng.shuffle(colors)
        scene_colors = colors[: rng.choice([3, 4])]
        image_path = image_dir / f"eval_{idx:03d}.png"
        _make_image(image_path, _layout(idx, scene_colors))
        if idx < int(count * 0.8):
            cases.append(_color_case(idx, image_path, scene_colors))
        else:
            relation = RELATIONS[(idx - int(count * 0.8)) % len(RELATIONS)]
            cases.append(_spatial_case(idx, image_path, scene_colors, relation))
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 100 semantic evaluation cases with known expected answers.")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--image-dir", default="samples/eval_images")
    parser.add_argument("--output", default="samples/eval_cases.jsonl")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    cases = generate_cases(args.count, image_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(case, ensure_ascii=False) for case in cases) + "\n", encoding="utf-8")
    print(f"generated {len(cases)} cases -> {output}")


if __name__ == "__main__":
    main()
