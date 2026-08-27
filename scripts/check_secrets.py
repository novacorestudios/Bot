#!/usr/bin/env python3
"""Fail the build if anything that looks like a credential is committed.

Runs in CI and is worth running locally before every push:

    python scripts/check_secrets.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Patterns that indicate a real secret, not a placeholder.
PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("Binance API key/secret assignment",
     re.compile(r"""(?i)\b(api[_-]?key|api[_-]?secret)\b\s*[:=]\s*["'][A-Za-z0-9]{30,}["']""")),
    ("Telegram bot token",
     re.compile(r"\b\d{8,10}:[A-Za-z0-9_-]{35}\b")),
    ("Generic 64-char hex secret",
     re.compile(r"""(?i)\bsecret\b\s*[:=]\s*["'][A-Fa-f0-9]{48,}["']""")),
    ("Private key block",
     re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("AWS access key id",
     re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
]

SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", "data",
             "logs", "reports", ".pytest_cache", ".ruff_cache", ".mypy_cache"}
SKIP_FILES = {"check_secrets.py"}
TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".toml", ".md", ".txt", ".sh",
                 ".env", ".ini", ".cfg", ".example", ".html", ".js", ".Dockerfile"}


def tracked_files() -> list[Path]:
    """Prefer git's index so untracked scratch files do not fail the build."""
    try:
        out = subprocess.run(  # noqa: S603
            ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
        )
        return [REPO / line for line in out.stdout.splitlines() if line]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return [p for p in REPO.rglob("*") if p.is_file()]


def should_scan(path: Path) -> bool:
    if not path.is_file():
        return False
    if any(part in SKIP_DIRS for part in path.parts):
        return False
    if path.name in SKIP_FILES:
        return False
    return path.suffix in TEXT_SUFFIXES or path.name.startswith(".env")


def main() -> int:
    findings: list[str] = []

    for path in tracked_files():
        if not should_scan(path):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for label, pattern in PATTERNS:
            for match in pattern.finditer(text):
                line_no = text[: match.start()].count("\n") + 1
                rel = path.relative_to(REPO)
                findings.append(f"{rel}:{line_no}: {label}")

    # A committed .env is a finding regardless of content.
    env_path = REPO / ".env"
    if env_path.exists():
        tracked = subprocess.run(  # noqa: S603
            ["git", "ls-files", "--error-unmatch", ".env"],
            cwd=REPO, capture_output=True, text=True, check=False
        )
        if tracked.returncode == 0:
            findings.append(".env: committed to git — remove it and rotate the keys")

    if findings:
        print("SECRET SCAN FAILED", file=sys.stderr)
        for finding in findings:
            print(f"  {finding}", file=sys.stderr)
        print(
            "\nIf a key reached git history, rotating it on Binance is the only "
            "real remedy — deleting the file is not enough.",
            file=sys.stderr,
        )
        return 1

    print(f"secret scan clean ({len([p for p in tracked_files() if should_scan(p)])} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
