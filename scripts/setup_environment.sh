#!/bin/bash

# Enhanced Development Environment Setup for AI Coding Agents
# This script sets up the optimal development environment for ML dataset creation

set -e

echo "🚀 Setting up enhanced AI coding agent development environment..."

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running with proper permissions
check_permissions() {
    log_info "Checking permissions..."
    if [ "$EUID" -eq 0 ]; then
        log_warning "Running as root - some operations may require user context"
    fi
}

# Update system packages (if possible)
update_system() {
    log_info "Attempting system package update..."
    if command -v apt-get &> /dev/null; then
        if apt-get update 2>/dev/null; then
            log_success "System packages updated"
        else
            log_warning "Could not update system packages (permission denied)"
        fi
    elif command -v yum &> /dev/null; then
        if yum update -y 2>/dev/null; then
            log_success "System packages updated"
        else
            log_warning "Could not update system packages (permission denied)"
        fi
    fi
}

# Install Python ML ecosystem
setup_python_ml() {
    log_info "Setting up Python ML ecosystem..."
    
    # Core ML libraries
    pip3 install --user --upgrade \
        numpy>=1.24.0 \
        pandas>=2.0.0 \
        matplotlib>=3.6.0 \
        scikit-learn>=1.3.0 \
        scipy>=1.10.0 \
        seaborn>=0.12.0 \
        plotly>=5.15.0 \
        jupyter>=1.1.0 \
        notebook>=7.0.0 \
        jupyterlab>=4.0.0
    
    log_success "Core ML libraries installed"
    
    # Deep learning libraries (optional, for advanced datasets)
    pip3 install --user --upgrade \
        torch>=2.0.0 \
        tensorflow>=2.13.0 \
        transformers>=4.30.0 \
        datasets>=2.14.0 \
        huggingface-hub>=0.16.0 2>/dev/null || log_warning "Deep learning libraries skipped (optional)"
    
    # Data processing libraries
    pip3 install --user --upgrade \
        pyarrow>=12.0.0 \
        fastparquet>=2023.7.0 \
        h5py>=3.9.0 \
        openpyxl>=3.1.0 \
        xlsxwriter>=3.1.0
    
    log_success "Data processing libraries installed"
}

# Install development tools
setup_dev_tools() {
    log_info "Setting up development tools..."
    
    # Code formatting and linting
    pip3 install --user --upgrade \
        black>=23.7.0 \
        flake8>=6.0.0 \
        pylint>=2.17.0 \
        mypy>=1.5.0 \
        isort>=5.12.0 \
        autopep8>=2.0.0
    
    # Testing frameworks
    pip3 install --user --upgrade \
        pytest>=7.4.0 \
        pytest-cov>=4.1.0 \
        pytest-xdist>=3.3.0 \
        pytest-benchmark>=4.0.0 \
        hypothesis>=6.82.0
    
    # Documentation tools
    pip3 install --user --upgrade \
        sphinx>=7.1.0 \
        sphinx-rtd-theme>=1.3.0 \
        myst-parser>=2.0.0
    
    log_success "Development tools installed"
}

# Install language-specific tools
setup_language_tools() {
    log_info "Setting up multi-language support..."
    
    # Node.js and JavaScript tools
    if command -v npm &> /dev/null; then
        npm install -g --silent \
            typescript \
            eslint \
            prettier \
            jest \
            nodemon \
            @types/node 2>/dev/null || log_warning "Node.js tools skipped"
        log_success "JavaScript/TypeScript tools installed"
    else
        log_warning "Node.js not found - JavaScript tools skipped"
    fi
    
    # Rust tools
    if command -v cargo &> /dev/null; then
        cargo install --quiet \
            rustfmt \
            clippy \
            cargo-watch 2>/dev/null || log_warning "Rust tools skipped"
        log_success "Rust tools installed"
    else
        log_warning "Rust not found - Rust tools skipped"
    fi
    
    # Go tools
    if command -v go &> /dev/null; then
        go install golang.org/x/tools/cmd/gofmt@latest 2>/dev/null || log_warning "Go tools skipped"
        go install golang.org/x/tools/cmd/goimports@latest 2>/dev/null || log_warning "Go tools skipped"
        log_success "Go tools installed"
    else
        log_warning "Go not found - Go tools skipped"
    fi
}

# Create directory structure
create_directory_structure() {
    log_info "Creating enhanced directory structure..."
    
    mkdir -p \
        "datasets/raw" \
        "datasets/processed" \
        "datasets/synthetic" \
        "datasets/external" \
        "models/training" \
        "models/inference" \
        "models/checkpoints" \
        "models/exports" \
        "code_samples/python" \
        "code_samples/javascript" \
        "code_samples/typescript" \
        "code_samples/java" \
        "code_samples/cpp" \
        "code_samples/rust" \
        "code_samples/go" \
        "code_samples/csharp" \
        "code_samples/php" \
        "code_samples/ruby" \
        "code_samples/swift" \
        "code_samples/kotlin" \
        "scripts/data_processing" \
        "scripts/model_training" \
        "scripts/evaluation" \
        "scripts/automation" \
        "scripts/validation" \
        "documentation/specifications" \
        "documentation/tutorials" \
        "documentation/api_docs" \
        "documentation/research" \
        "tests/unit" \
        "tests/integration" \
        "tests/performance" \
        "tests/validation" \
        "configs/ml" \
        "configs/processing" \
        "configs/validation" \
        "configs/deployment" \
        "notebooks/exploration" \
        "notebooks/analysis" \
        "notebooks/training" \
        "notebooks/visualization" \
        "tools/parsers" \
        "tools/generators" \
        "tools/validators" \
        "tools/converters" \
        "benchmarks/performance" \
        "benchmarks/quality" \
        "benchmarks/accuracy" \
        "templates/code" \
        "templates/datasets" \
        "templates/models" \
        "templates/documentation"
    
    log_success "Directory structure created"
}

# Create configuration files
create_config_files() {
    log_info "Creating configuration files..."
    
    # Python configuration
    cat > .flake8 << 'EOF'
[flake8]
max-line-length = 88
extend-ignore = E203, W503
exclude = .git,__pycache__,build,dist,.venv,venv
max-complexity = 10
EOF

    cat > pyproject.toml << 'EOF'
[tool.black]
line-length = 88
target-version = ['py38', 'py39', 'py310', 'py311']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
EOF

    # Git configuration for datasets
    cat >> .gitignore << 'EOF'

# Enhanced ML and AI specific ignores
*.pkl
*.pickle
*.joblib
*.h5
*.hdf5
*.npz
*.parquet
datasets/raw/large/
models/training/checkpoints/
models/exports/
*.tfrecord
*.onnx
wandb/
mlruns/
.neptune/
.dvc/
*.dvc

# Language specific
__pycache__/
*.py[cod]
*$py.class
*.so
.pytest_cache/
.coverage
htmlcov/
.mypy_cache/
.dmypy.json
dmypy.json

# JavaScript/Node.js
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*
.env.local
.env.development.local
.env.test.local
.env.production.local

# Java
*.class
*.jar
*.war
target/
.gradle/
build/

# C/C++
*.o
*.exe
*.dll
*.so
*.dylib

# Rust
target/
Cargo.lock

# Go
*.exe
*.exe~
*.dll
*.so
*.dylib
*.test
*.out
go.work

# IDE specific
.vscode/settings.json
.idea/
*.swp
*.swo
*~

# OS specific
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db
EOF

    log_success "Configuration files created"
}

# Validate installation
validate_installation() {
    log_info "Validating installation..."
    
    # Test Python imports
    python3 -c "
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import jupyter
print(f'✅ Python {sys.version_info.major}.{sys.version_info.minor} with all ML libraries working')
print(f'✅ NumPy {np.__version__}')
print(f'✅ Pandas {pd.__version__}')
print(f'✅ Scikit-learn {sklearn.__version__}')
print(f'✅ Jupyter installed')
"
    
    # Test development tools
    if command -v black &> /dev/null; then
        echo "✅ Black formatter available"
    fi
    
    if command -v flake8 &> /dev/null; then
        echo "✅ Flake8 linter available"
    fi
    
    if command -v pytest &> /dev/null; then
        echo "✅ Pytest testing framework available"
    fi
    
    log_success "Installation validation completed"
}

# Create helpful scripts
create_helper_scripts() {
    log_info "Creating helper scripts..."
    
    # Dataset validation script
    cat > scripts/validation/validate_datasets.py << 'EOF'
#!/usr/bin/env python3
"""Dataset validation utility for ML-ready data."""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

def validate_dataset(dataset_path: str, schema_path: str = None) -> bool:
    """Validate dataset integrity and ML readiness."""
    try:
        if dataset_path.endswith('.csv'):
            df = pd.read_csv(dataset_path)
        elif dataset_path.endswith('.json'):
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        elif dataset_path.endswith('.parquet'):
            df = pd.read_parquet(dataset_path)
        else:
            print(f"Unsupported format: {dataset_path}")
            return False
        
        print(f"✅ Dataset loaded: {df.shape}")
        print(f"✅ Columns: {list(df.columns)}")
        print(f"✅ No missing values: {not df.isnull().any().any()}")
        return True
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python validate_datasets.py <dataset_path>")
        sys.exit(1)
    
    success = validate_dataset(sys.argv[1])
    sys.exit(0 if success else 1)
EOF

    chmod +x scripts/validation/validate_datasets.py
    
    # Code formatting script
    cat > scripts/automation/format_code.sh << 'EOF'
#!/bin/bash
# Automated code formatting for all supported languages

echo "🎨 Formatting code..."

# Python
echo "Formatting Python files..."
find . -name "*.py" -not -path "./.venv/*" -not -path "./venv/*" | xargs black
find . -name "*.py" -not -path "./.venv/*" -not -path "./venv/*" | xargs isort

# JavaScript/TypeScript
if command -v prettier &> /dev/null; then
    echo "Formatting JavaScript/TypeScript files..."
    prettier --write "**/*.{js,ts,jsx,tsx}"
fi

# Rust
if command -v rustfmt &> /dev/null; then
    echo "Formatting Rust files..."
    find . -name "*.rs" | xargs rustfmt
fi

# Go
if command -v gofmt &> /dev/null; then
    echo "Formatting Go files..."
    find . -name "*.go" | xargs gofmt -w
fi

echo "✅ Code formatting completed"
EOF

    chmod +x scripts/automation/format_code.sh
    
    log_success "Helper scripts created"
}

# Main execution
main() {
    echo "🚀 Enhanced AI Coding Agent Development Environment Setup"
    echo "========================================================"
    
    check_permissions
    update_system
    setup_python_ml
    setup_dev_tools
    setup_language_tools
    create_directory_structure
    create_config_files
    create_helper_scripts
    validate_installation
    
    echo ""
    echo "🎉 Setup completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Run 'python3 scripts/validation/validate_datasets.py' to test dataset validation"
    echo "2. Use 'scripts/automation/format_code.sh' to format all code"
    echo "3. Start Jupyter with 'jupyter lab' for interactive development"
    echo "4. Run tests with 'python3 -m pytest tests/ -v'"
    echo ""
    echo "Happy coding! 🚀"
}

# Run main function
main "$@"