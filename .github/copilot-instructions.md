# DATA - ML Datasets and Structures Repository

Building Datasets, Structures and the like for ML and AI coding agents.

**ALWAYS reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the information here.**

## Repository Status and Architecture

**Current State**: This is a minimal repository with basic directory structure for ML datasets and AI training data. The repository is designed to develop comprehensive datasets specifically for AI coding agents across multiple programming languages.

**Purpose**: Create and maintain datasets, training data, and structures for machine learning models, particularly focused on AI coding agents that need to understand and generate code without errors.

## Working Effectively

### Initial Setup and Environment Bootstrap
**CRITICAL**: Always run these bootstrap steps in order before making any changes:

1. **Update system packages** (takes ~30 seconds):
   ```bash
   apt-get update
   ```

2. **Install Python ML environment** (takes ~15 seconds):
   ```bash
   pip3 install --user numpy pandas matplotlib
   ```

3. **Install development tools** (takes ~8 seconds):
   ```bash
   pip3 install --user black flake8 pylint pytest
   ```

4. **Install Jupyter ecosystem** (takes ~45 seconds, NEVER CANCEL):
   ```bash
   pip3 install --user jupyter notebook scikit-learn
   ```
   - Set timeout to 60+ minutes for safety
   - This step installs comprehensive ML and notebook environment

5. **Validate installation** (takes ~2 seconds):
   ```bash
   python3 -c "import numpy, pandas, matplotlib, sklearn; print('✅ All ML libraries working')"
   ```

### Repository Structure Creation
**Standard Directory Structure**: The repository follows this organization pattern:

```bash
mkdir -p datasets/{raw,processed,synthetic} \
         models/{training,inference,checkpoints} \
         code_samples/{python,javascript,java,cpp,rust,go,typescript} \
         scripts/{data_processing,model_training,evaluation} \
         documentation/{specifications,tutorials,api_docs} \
         tests/{unit,integration,performance}
```

**Expected completion time**: < 1 second

### Development Workflow Commands

1. **Format Python code** (takes ~200ms):
   ```bash
   python3 -m black <file_or_directory>
   ```

2. **Lint Python code** (takes ~200ms):
   ```bash
   python3 -m flake8 <file_or_directory>
   ```

3. **Run tests** (takes ~1-2 seconds per test file):
   ```bash
   python3 -m pytest <test_file> -v
   ```

4. **Git operations** (takes ~10ms):
   ```bash
   git status  # Check repository status
   git add .   # Stage all changes
   ```

## Build and Test Processes

**No Traditional Build Process**: This repository does not have a traditional build process like compilation. Instead, focus on:

1. **Data Validation Scripts**: Python scripts to validate dataset integrity
2. **ML Pipeline Scripts**: Training and evaluation scripts for models
3. **Code Generation Tests**: Tests for AI coding agent training data

### Running Tests
```bash
# Run all Python tests (NEVER CANCEL - may take 5-15 minutes for large datasets)
python3 -m pytest tests/ -v --timeout=1800

# Run specific test categories
python3 -m pytest tests/unit/ -v          # Unit tests (~1-2 minutes)
python3 -m pytest tests/integration/ -v   # Integration tests (~5-10 minutes)
python3 -m pytest tests/performance/ -v   # Performance tests (~10-30 minutes)
```

**CRITICAL**: Set timeout to 30+ minutes for test suites. NEVER CANCEL test runs.

## Validation and Quality Assurance

### Pre-commit Validation Steps
**ALWAYS run these before committing changes**:

1. **Format all Python code** (takes ~1-5 seconds):
   ```bash
   find . -name "*.py" -exec python3 -m black {} \;
   ```

2. **Lint all Python code** (takes ~2-10 seconds):
   ```bash
   find . -name "*.py" -exec python3 -m flake8 {} \;
   ```

3. **Run relevant tests** (takes ~1-15 minutes depending on scope):
   ```bash
   python3 -m pytest tests/ -v --timeout=1800
   ```

### Manual Validation Scenarios
**ALWAYS manually test these scenarios after making changes**:

1. **Dataset Integrity**: Verify datasets can be loaded and processed:
   ```python
   import pandas as pd
   import numpy as np
   
   # Test loading sample data
   df = pd.read_csv('datasets/raw/sample.csv')  # if exists
   print(f"Dataset shape: {df.shape}")
   print("✅ Dataset validation passed")
   ```

2. **Code Sample Validation**: Ensure code samples in different languages are syntactically correct:
   ```bash
   # Python samples
   python3 -m py_compile code_samples/python/*.py
   
   # JavaScript samples (if node.js available)
   node -c code_samples/javascript/*.js
   ```

3. **ML Pipeline Test**: Run a simple ML workflow:
   ```python
   from sklearn.datasets import make_classification
   from sklearn.model_selection import train_test_split
   from sklearn.ensemble import RandomForestClassifier
   
   # Create sample data and test ML pipeline
   X, y = make_classification(n_samples=100, n_features=4)
   X_train, X_test, y_train, y_test = train_test_split(X, y)
   model = RandomForestClassifier()
   model.fit(X_train, y_train)
   print("✅ ML pipeline validation passed")
   ```

## Common Development Tasks

### Adding New Datasets
1. **Place raw data** in `datasets/raw/`
2. **Create processing script** in `scripts/data_processing/`
3. **Save processed data** in `datasets/processed/`
4. **Add validation tests** in `tests/unit/test_datasets.py`

### Adding Code Samples for AI Training
1. **Organize by language** in `code_samples/{language}/`
2. **Include comprehensive examples**: basic syntax, advanced patterns, common algorithms
3. **Add metadata files** describing the code samples
4. **Test syntax validity** with language-specific linters

### Creating ML Models
1. **Training scripts** go in `scripts/model_training/`
2. **Model artifacts** go in `models/training/` or `models/inference/`
3. **Evaluation scripts** go in `scripts/evaluation/`
4. **Performance benchmarks** go in `tests/performance/`

## Expected Timing for Common Operations

| Operation | Expected Time | Timeout Setting |
|-----------|---------------|-----------------|
| Environment setup | 60-90 seconds | 300 seconds |
| Package installation (basic) | 15-20 seconds | 60 seconds |
| Package installation (full ML) | 40-50 seconds | 120 seconds |
| Code formatting | < 1 second | 30 seconds |
| Code linting | < 1 second | 30 seconds |
| Unit tests | 1-5 minutes | 600 seconds |
| Integration tests | 5-15 minutes | 1800 seconds |
| Performance tests | 10-60 minutes | 3600 seconds |
| Large dataset processing | 30-120 minutes | 7200 seconds |

**CRITICAL**: NEVER CANCEL long-running operations. ML and data processing tasks can take significant time.

## Repository Navigation

### Key Directories and Their Purpose

- **`datasets/`**: All data files organized by processing stage
  - `raw/`: Original, unprocessed datasets
  - `processed/`: Clean, processed datasets ready for ML
  - `synthetic/`: Generated or augmented datasets

- **`code_samples/`**: Training data for AI coding agents
  - `python/`, `javascript/`, `java/`, etc.: Language-specific code examples
  - Each directory should contain diverse, high-quality code samples

- **`models/`**: ML model storage and management
  - `training/`: Models during training process
  - `inference/`: Production-ready models
  - `checkpoints/`: Training checkpoints and intermediate states

- **`scripts/`**: Automation and processing scripts
  - `data_processing/`: Data cleaning and transformation
  - `model_training/`: ML training pipelines
  - `evaluation/`: Model evaluation and benchmarking

- **`documentation/`**: All documentation and specifications
  - `specifications/`: Technical specifications for datasets and models
  - `tutorials/`: How-to guides and examples
  - `api_docs/`: API documentation for any services

- **`tests/`**: All testing code
  - `unit/`: Fast, isolated tests
  - `integration/`: Cross-component tests
  - `performance/`: Benchmarking and performance tests

### Frequently Referenced Files

When adding new content, always check these locations first:

- **Dataset schemas**: `documentation/specifications/dataset_schemas.md`
- **Code sample guidelines**: `documentation/specifications/code_sample_standards.md`
- **ML pipeline configs**: `scripts/model_training/configs/`
- **Test utilities**: `tests/test_utils.py`

## Troubleshooting

### Common Issues and Solutions

1. **Python package conflicts**:
   ```bash
   pip3 install --user --force-reinstall <package_name>
   ```

2. **Permission issues with datasets**:
   ```bash
   chmod -R 755 datasets/
   ```

3. **Git tracking large files**:
   - Use `.gitignore` for large datasets
   - Consider Git LFS for essential large files
   - Keep datasets under 100MB when possible

4. **ML model training failures**:
   - Check available memory with `free -h`
   - Reduce batch sizes for large datasets
   - Use data sampling for initial testing

### Performance Optimization

- **Use data sampling** during development: Work with 10% samples for faster iteration
- **Parallel processing**: Utilize multiprocessing for data processing scripts
- **Memory management**: Monitor memory usage during large dataset operations
- **Incremental processing**: Process data in chunks rather than loading everything into memory

## Development Best Practices

1. **Always validate data integrity** before and after processing
2. **Use consistent naming conventions** across all directories
3. **Include comprehensive metadata** for all datasets and models
4. **Test with small samples first** before processing large datasets
5. **Document data sources and licenses** for all external datasets
6. **Version control model configurations** and hyperparameters
7. **Use reproducible random seeds** in all ML experiments
8. **Monitor resource usage** during long-running operations

## Current Repository Structure

```
/home/runner/work/DATA/DATA/
├── .github/
│   └── copilot-instructions.md
├── code_samples/
│   ├── cpp/
│   ├── go/
│   ├── java/
│   ├── javascript/
│   │   └── functional_async.js
│   ├── python/
│   │   └── algorithms_basic.py
│   ├── rust/
│   └── typescript/
├── datasets/
│   ├── processed/
│   ├── raw/
│   └── synthetic/
├── documentation/
│   ├── api_docs/
│   ├── specifications/
│   └── tutorials/
├── models/
│   ├── checkpoints/
│   ├── inference/
│   └── training/
├── scripts/
│   ├── data_processing/
│   ├── evaluation/
│   └── model_training/
├── tests/
│   ├── integration/
│   ├── performance/
│   └── unit/
│       └── test_ml_validation.py
└── README.md
```

**Remember**: This repository serves as training data for AI coding agents. Quality, consistency, and comprehensive coverage across programming languages are essential for effective AI training.