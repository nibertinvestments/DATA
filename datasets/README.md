# DATA Repository

This repository is focused on building datasets, structures, and training resources for machine learning and AI systems that work with code. It is designed to support models that need high-quality examples in programming, software engineering, and algorithmic reasoning across multiple languages.

## Repository goals

- collect realistic and structured training examples for AI coding agents
- preserve machine-readable metadata alongside prompts and outputs
- support multilingual software engineering tasks
- create a reusable foundation for bug-fix, generation, translation, and evaluation data

## Dataset taxonomy

The repository is organized around a small set of practical data categories:

- raw data: exploratory or source-style samples
- processed data: curated, schema-valid datasets ready for training
- synthetic data: generated task variants and benchmark patterns
- code_samples: language-specific implementation corpus
- vector processing: embedding, similarity, and retrieval-related assets

## Curated dataset schema

The dataset contract used in this repository is intentionally simple and robust:

```json
{
  "dataset_id": "unique_dataset_name",
  "name": "Human readable dataset name",
  "version": "1.0.0",
  "description": "What the dataset is for",
  "language_coverage": ["python", "javascript"],
  "task_types": ["bug_fix", "code_generation"],
  "tasks": [
    {
      "id": "task-001",
      "language": "python",
      "task_type": "bug_fix",
      "difficulty": "medium",
      "prompt": "Describe the task for the model.",
      "response": "Ground truth answer or patch.",
      "metadata": {
        "tags": ["async", "exceptions"],
        "source": "synthetic"
      }
    }
  ]
}
```

This makes each dataset easy to validate, classify, and feed into training pipelines.

## Recommended usage

1. Start with the curated files in `datasets/processed/`.
2. Validate the schema with the repository script before training.
3. Extend by adding synthetic tasks under `datasets/synthetic/`.
4. Keep raw data separate and document provenance.
5. Use `code_samples/` for reference implementations and task grounding.

## Validation

Run the repository validation script before using a dataset in an ML workflow:

```bash
python3 scripts/validate_dataset_schema.py datasets/processed/ai_coding_agent_training_suite.json
```

## File structure

```text
datasets/
├── README.md
├── processed/
│   ├── README.md
│   └── ai_coding_agent_training_suite.json
├── raw/
├── synthetic/
│   └── README.md
└── DATASET_INDEX.json
```

## Notes

The repository is intentionally structured to stay useful for AI and ML workflows without becoming a large unmanaged corpus. Curated data should remain documented, schema-valid, and easy to consume programmatically.
