# DATA Repository

A curated repository of code samples, structured datasets, and validation tooling designed for ML and AI training pipelines focused on software engineering.

## High-level overview

This repository is built to support AI coding agents and ML workflows that need high-quality examples across languages, problem types, and software engineering tasks. It combines:

- production-ready code samples for common patterns and algorithms
- structured datasets for model training and evaluation
- validation scripts that check schema integrity and dataset consistency
- reusable task formats for code generation, repair, translation, and reasoning

The emphasis is on data quality, schema clarity, and practical usability rather than repository bloat.

## Repository purpose

DATA is designed for:

- code generation model training
- bug-fix and refactor dataset creation
- multilingual code understanding
- static analysis and reasoning benchmarks
- software engineering dataset research

## Top-level architecture

```text
DATA/
├── README.md
├── LICENSE
├── .gitignore
├── datasets/
│   ├── README.md
│   ├── processed/
│   │   ├── README.md
│   │   └── ai_coding_agent_training_suite.json
│   ├── raw/
│   ├── synthetic/
│   │   └── README.md
│   └── DATASET_INDEX.json
├── code_samples/
│   ├── README.md
│   ├── python/
│   ├── javascript/
│   ├── java/
│   ├── solidity/
│   └── ...
├── scripts/
│   ├── validate_dataset_schema.py
│   └── data_processing/
├── tests/
│   └── test_dataset_schema.py
├── VECTOR_PROCESSING/
│   ├── README.md
│   └── ...
├── documentation/
├── data-sources/
├── high_end_specialized/
└── public/
```

## What is included

- curated ML-ready dataset manifests with task metadata
- multilingual code examples for algorithmic and design-pattern work
- validation and quality checks for dataset integrity
- a clear path for expansion into synthetic, raw, and processed corpora

## Dataset quality standards

Every dataset in the curated collection should satisfy these minimum rules:

- explicit schema version metadata
- language coverage and task-type metadata
- deterministic validation checks
- realistic prompts and ground-truth answers
- consistent difficulty labels and tags

## Quick start

```bash
# install the basic Python tooling
python3 -m pip install --user numpy pandas matplotlib
python3 -m pip install --user pytest

# validate the curated dataset
python3 scripts/validate_dataset_schema.py datasets/processed/ai_coding_agent_training_suite.json

# run the repository validation test
python3 -m pytest tests/test_dataset_schema.py -q
```

## Use cases

- training code models to generate idiomatic implementations
- benchmarking bug-fix quality and reasoning traces
- measuring cross-language translation quality
- building evaluation sets for secure code generation
- dataset ingestion for LLM pipelines and academic experiments

## Contribution expectations

Contributions should stay high quality and focused:

- prefer realistic code tasks over toy examples
- keep metadata consistent and machine-readable
- validate dataset structure before merge
- document assumptions and version changes

## Documentation index

- README.md — repository overview
- datasets/README.md — dataset taxonomy and schema guide
- datasets/processed/README.md — processed dataset usage and conventions
- CODE_SAMPLES_SUMMARY.md — code sample inventory
- VECTOR_PROCESSING/README.md — vector operations and embedding utilities

## License

This repository is licensed under the MIT License. See LICENSE for details.

## Contact

Repository: https://github.com/nibertinvestments/DATA

---

Built for reliable ML dataset curation, AI coding agent research, and high-quality software engineering data.
