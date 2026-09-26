# Processed datasets

Processed datasets are the curated, schema-valid training assets intended for model ingestion, evaluation, and downstream experimentation.

## Current curated asset

- `ai_coding_agent_training_suite.json` — a compact benchmark dataset covering bug-fix, generation, translation, and secure coding tasks.

## Standards

Each processed dataset should include:

- dataset metadata (id, name, version, description)
- language coverage and task categories
- realistic task prompts and expected outputs
- consistent difficulty labels and tags
- validation-friendly JSON structure

## Recommended workflow

```bash
python3 scripts/validate_dataset_schema.py datasets/processed/ai_coding_agent_training_suite.json
```

## Use cases

- LLM fine-tuning on code tasks
- evaluation of patch generation quality
- multilingual reasoning benchmarks
- synthesis of code-review and bug-fix examples
