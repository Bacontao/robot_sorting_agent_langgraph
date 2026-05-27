from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_GITIGNORE_PATTERNS = [
    ".env",
    ".env.*",
    "!.env.example",
    "!.env.siliconflow.example",
    ".venv/",
    ".models/",
    "artifacts/",
    "*.pt",
    "*.pth",
    "*.safetensors",
    "__pycache__/",
    ".DS_Store",
]

SKIP_DIRS = {
    ".git",
    ".venv",
    ".models",
    "artifacts",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".cache",
    "__pycache__",
    "build",
    "dist",
}

SKIP_FILES = {".env"}

SECRET_PATTERNS = [
    ("env secret", re.compile(r"\b[A-Z0-9_]*(?:API_KEY|TOKEN|SECRET)\s*=\s*['\"]?([^'\"\s#]*)")),
    ("openai-style key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
]

PLACEHOLDER_WORDS = {
    "your",
    "example",
    "placeholder",
    "changeme",
    "replace",
    "dummy",
    "test",
    "none",
    "null",
}

LARGE_FILE_SUFFIXES = {
    ".pt",
    ".pth",
    ".onnx",
    ".safetensors",
    ".bin",
    ".gguf",
    ".ckpt",
}


def _read_gitignore() -> set[str]:
    path = ROOT / ".gitignore"
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")}


def _is_skipped(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if path.name in SKIP_FILES:
        return True
    return any(part in SKIP_DIRS for part in rel.parts)


def _looks_like_placeholder(value: str) -> bool:
    lowered = value.lower()
    if value == "":
        return True
    return any(word in lowered for word in PLACEHOLDER_WORDS)


def _iter_text_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or _is_skipped(path):
            continue
        if path.suffix.lower() in LARGE_FILE_SUFFIXES:
            continue
        try:
            path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        files.append(path)
    return files


def check_gitignore() -> list[str]:
    current = _read_gitignore()
    missing = [pattern for pattern in REQUIRED_GITIGNORE_PATTERNS if pattern not in current]
    return [f"missing .gitignore pattern: {pattern}" for pattern in missing]


def check_secrets() -> list[str]:
    findings: list[str] = []
    for path in _iter_text_files():
        rel = path.relative_to(ROOT)
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for label, pattern in SECRET_PATTERNS:
                match = pattern.search(line)
                if not match:
                    continue
                value = match.group(match.lastindex or 0).strip()
                if _looks_like_placeholder(value):
                    continue
                findings.append(f"{rel}:{line_no}: possible {label}")
    return findings


def check_large_files() -> list[str]:
    findings: list[str] = []
    gitignore = _read_gitignore()
    for path in ROOT.rglob("*"):
        if not path.is_file() or _is_skipped(path):
            continue
        rel = path.relative_to(ROOT)
        size_mb = path.stat().st_size / (1024 * 1024)
        if path.suffix.lower() in LARGE_FILE_SUFFIXES or size_mb >= 10:
            if f"*{path.suffix.lower()}" in gitignore:
                continue
            findings.append(f"{rel}: {size_mb:.1f} MB; keep out of git unless intentionally tracked")
    return findings


def check_local_only_paths() -> list[str]:
    notices: list[str] = []
    for name in [".env", ".models", "artifacts", ".venv", ".pytest_cache", ".ruff_cache", ".DS_Store"]:
        path = ROOT / name
        if path.exists():
            notices.append(f"{name} exists locally; make sure it remains ignored")
    return notices


def main() -> int:
    blockers = []
    blockers.extend(check_gitignore())
    blockers.extend(check_secrets())
    blockers.extend(check_large_files())
    notices = check_local_only_paths()

    if notices:
        print("Local-only files detected:")
        for item in notices:
            print(f"  - {item}")

    if blockers:
        print("\nPrepublish check failed:")
        for item in blockers:
            print(f"  - {item}")
        return 1

    print("\nPrepublish check passed: no obvious secret or large-file blockers found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
