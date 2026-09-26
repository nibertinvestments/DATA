from pathlib import Path

from scripts.validate_dataset_schema import validate_dataset_file


def test_curated_dataset_schema():
    dataset_path = Path(__file__).resolve().parents[1] / "datasets" / "processed" / "ai_coding_agent_training_suite.json"
    summary = validate_dataset_file(dataset_path)

    assert summary["dataset_id"] == "ai_coding_agent_training_suite_v1"
    assert summary["task_count"] >= 8
    assert summary["language_count"] >= 5
