from __future__ import annotations

import argparse
import shutil
import ssl
from pathlib import Path
from urllib.request import urlopen


ASSETS = {
    "GroundingDINO_SwinT_OGC.py": "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "groundingdino_swint_ogc.pth": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
    "sam_vit_b_01ec64.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}

BERT_ASSETS = {
    "config.json": "https://huggingface.co/bert-base-uncased/resolve/main/config.json",
    "model.safetensors": "https://huggingface.co/bert-base-uncased/resolve/main/model.safetensors",
    "tokenizer.json": "https://huggingface.co/bert-base-uncased/resolve/main/tokenizer.json",
    "tokenizer_config.json": "https://huggingface.co/bert-base-uncased/resolve/main/tokenizer_config.json",
    "vocab.txt": "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt",
}


def _ssl_context(*, insecure: bool) -> ssl.SSLContext | None:
    if insecure:
        return ssl._create_unverified_context()  # noqa: S323
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return None


def download(url: str, target: Path, *, force: bool, insecure: bool) -> None:
    if target.exists() and not force:
        print(f"exists: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading: {target}")
    with urlopen(url, context=_ssl_context(insecure=insecure), timeout=60) as response:  # noqa: S310
        with target.open("wb") as handle:
            shutil.copyfileobj(response, handle)


def patch_grounding_config(root: Path) -> None:
    config_path = root / "GroundingDINO_SwinT_OGC.py"
    bert_path = (root / "bert-base-uncased").resolve()
    text = config_path.read_text(encoding="utf-8")
    text = text.replace('text_encoder_type = "bert-base-uncased"', f'text_encoder_type = "{bert_path}"')
    config_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download GroundingDINO + SAM vit_b assets.")
    parser.add_argument("--output-dir", default=".models", help="Directory for model configs and checkpoints.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--insecure", action="store_true", help="Disable TLS verification if this machine has broken CA certs.")
    args = parser.parse_args()

    root = Path(args.output_dir).expanduser()
    for filename, url in ASSETS.items():
        download(url, root / filename, force=args.force, insecure=args.insecure)
    for filename, url in BERT_ASSETS.items():
        download(url, root / "bert-base-uncased" / filename, force=args.force, insecure=args.insecure)
    patch_grounding_config(root)

    print("\nUse these environment values:")
    print("SEGMENTATION_BACKEND=grounded_sam")
    print(f"GROUNDING_DINO_CONFIG={root / 'GroundingDINO_SwinT_OGC.py'}")
    print(f"GROUNDING_DINO_CHECKPOINT={root / 'groundingdino_swint_ogc.pth'}")
    print(f"SAM_CHECKPOINT={root / 'sam_vit_b_01ec64.pth'}")
    print("SAM_MODEL_TYPE=vit_b")


if __name__ == "__main__":
    main()
