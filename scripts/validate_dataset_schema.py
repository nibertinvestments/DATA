#!/usr/bin/env python3
"""Validate DATA repository datasets against the curated schema."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REQUIRED_TOP_LEVEL_KEYS = {
    "dataset_id",
    "name",
    "version",
    "description",
    "language_coverage",
    "task_types",
    "tasks",
}

REQUIRED_TASK_KEYS = {
    "id",
    "language",
    "task_type",
    "difficulty",
    "prompt",
    "response",
    "metadata",
}


def validate_dataset_file(path: Path) -> dict[str, Any]:
    """Validate a JSON dataset file and return summary metadata."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - surfaced in CLI
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Dataset root in {path} must be a JSON object")

    missing_root_keys = sorted(REQUIRED_TOP_LEVEL_KEYS - set(payload.keys()))
    if missing_root_keys:
        raise ValueError(f"{path} missing root keys: {missing_root_keys}")

    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"{path} must contain a non-empty tasks list")

    valid_languages = isinstance(payload.get("language_coverage"), list) and bool(payload["language_coverage"])
    valid_types = isinstance(payload.get("task_types"), list) and bool(payload["task_types"])
    if not valid_languages:
        raise ValueError(f"{path} language_coverage must be a non-empty list")
    if not valid_types:
        raise ValueError(f"{path} task_types must be a non-empty list")

    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError(f"{path} contains a non-object task entry")

        missing_task_keys = sorted(REQUIRED_TASK_KEYS - set(task.keys()))
        if missing_task_keys:
            raise ValueError(f"Task in {path} missing keys: {missing_task_keys}")

        for key in ("id", "language", "task_type", "difficulty", "prompt", "response"):
            if not isinstance(task[key], str) or not task[key].strip():
                raise ValueError(f"Task {task.get('id', '<unknown>')} in {path} has invalid '{key}'")

        if not isinstance(task["metadata"], dict):
            raise ValueError(f"Task {task.get('id', '<unknown>')} in {path} has invalid metadata")

    summary = {
        "dataset_id": payload["dataset_id"],
        "name": payload["name"],
        "version": payload["version"],
        "task_count": len(tasks),
        "language_count": len(payload["language_coverage"]),
        "task_type_count": len(payload["task_types"]),
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate DATA repository dataset JSON files.")
    parser.add_argument("paths", nargs="+", type=Path, help="Dataset JSON file(s) to validate")
    args = parser.parse_args()

    ok = True
    for dataset_path in args.paths:
        try:
            summary = validate_dataset_file(dataset_path)
            print(f"OK: {dataset_path} -> {summary['dataset_id']} ({summary['task_count']} tasks)")
        except Exception as exc:  # pragma: no cover - CLI user feedback
            print(f"ERROR: {dataset_path} -> {exc}", file=sys.stderr)
            ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
