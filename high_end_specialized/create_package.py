#!/usr/bin/env python3
"""
High-End Specialized Package Creator
===================================

Creates a comprehensive ZIP package of all high-end algorithms, functions,
and equations with proper organization and documentation.
"""

import os
import zipfile
import json
from datetime import datetime
import subprocess
import shutil


def get_file_count_and_size(directory):
    """Get file count and total size for a directory."""
    file_count = 0
    total_size = 0
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.py', '.js', '.java', '.cpp', '.go', '.rs', '.md', '.txt')):
                file_path = os.path.join(root, file)
                file_count += 1
                total_size += os.path.getsize(file_path)
    
    return file_count, total_size


def create_package_manifest():
    """Create a comprehensive manifest of all implementations."""
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    manifest = {
        "package_name": "High-End Specialized Algorithms, Functions & Equations",
        "version": "1.0.0",
        "created_date": datetime.now().isoformat(),
        "description": "Premium collection of advanced algorithms, functions, and equations for AI, Finance, DeFi, Coding, and Science",
        "author": "Nibert Investments LLC",
        "license": "MIT",
        "categories": {
            "algorithms": [],
            "functions": [],
            "equations": [],
            "implementations": [],
            "examples": [],
            "documentation": []
        },
        "statistics": {
            "total_files": 0,
            "total_size_bytes": 0,
            "algorithms_count": 0,
            "functions_count": 0,
            "equations_count": 0,
            "languages_supported": [],
            "total_lines_of_code": 0
        },
        "requirements": {
            "python": ">=3.8",
            "packages": ["numpy", "scipy", "pandas", "matplotlib"]
        },
        "usage_examples": [
            "Portfolio optimization with quantum annealing",
            "Neural network activation function analysis", 
            "Options pricing with Black-Scholes PDE",
            "DeFi AMM liquidity pool optimization",
            "Monte Carlo Tree Search for game AI"
        ]
    }
    
    # Scan algorithms
    algorithms_dir = os.path.join(base_dir, "algorithms")
    if os.path.exists(algorithms_dir):
        for file in os.listdir(algorithms_dir):
            if file.endswith('.py') and not file.startswith('__'):
                file_path = os.path.join(algorithms_dir, file)
                with open(file_path, 'r') as f:
                    first_lines = f.read(500)
                    title = ""
                    description = ""
                    
                    for line in first_lines.split('\\n'):
                        if '"""' in line or "'''" in line:
                            continue
                        if line.strip() and not title:
                            title = line.strip()
                        elif line.strip() and "=" in line:
                            break
                        elif line.strip() and title and not description:
                            description = line.strip()
                            break
                
                manifest["categories"]["algorithms"].append({
                    "file": file,
                    "title": title,
                    "description": description,
                    "size_bytes": os.path.getsize(file_path)
                })
                manifest["statistics"]["algorithms_count"] += 1
    
    # Scan functions
    functions_dir = os.path.join(base_dir, "functions")
    if os.path.exists(functions_dir):
        for file in os.listdir(functions_dir):
            if file.endswith('.py') and not file.startswith('__'):
                file_path = os.path.join(functions_dir, file)
                manifest["categories"]["functions"].append({
                    "file": file,
                    "size_bytes": os.path.getsize(file_path)
                })
                manifest["statistics"]["functions_count"] += 1
    
    # Scan equations
    equations_dir = os.path.join(base_dir, "equations")
    if os.path.exists(equations_dir):
        for file in os.listdir(equations_dir):
            if file.endswith('.py') and not file.startswith('__'):
                file_path = os.path.join(equations_dir, file)
                manifest["categories"]["equations"].append({
                    "file": file,
                    "size_bytes": os.path.getsize(file_path)
                })
                manifest["statistics"]["equations_count"] += 1
    
    # Scan implementations
    implementations_dir = os.path.join(base_dir, "implementations")
    if os.path.exists(implementations_dir):
        languages = []
        for lang_dir in os.listdir(implementations_dir):
            lang_path = os.path.join(implementations_dir, lang_dir)
            if os.path.isdir(lang_path):
                languages.append(lang_dir)
                for file in os.listdir(lang_path):
                    if not file.startswith('.'):
                        manifest["categories"]["implementations"].append({
                            "language": lang_dir,
                            "file": file,
                            "size_bytes": os.path.getsize(os.path.join(lang_path, file))
                        })
        
        manifest["statistics"]["languages_supported"] = languages
    
    # Calculate total statistics
    total_files, total_size = get_file_count_and_size(base_dir)
    manifest["statistics"]["total_files"] = total_files
    manifest["statistics"]["total_size_bytes"] = total_size
    
    # Estimate lines of code
    total_lines = 0
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(('.py', '.js', '.java')):
                try:
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        total_lines += len(f.readlines())
                except:
                    pass
    
    manifest["statistics"]["total_lines_of_code"] = total_lines
    
    return manifest


def create_premium_package():
    """Create the premium ZIP package."""
    
    print("Creating High-End Specialized Premium Package...")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    package_dir = os.path.join(base_dir, "high_end_specialized_premium")
    
    # Create temporary package directory
    if os.path.exists(package_dir):
        shutil.rmtree(package_dir)
    os.makedirs(package_dir)
    
    # Copy all source files
    source_dirs = ["algorithms", "functions", "equations", "implementations", "examples", "documentation"]
    
    for dir_name in source_dirs:
        source_path = os.path.join(base_dir, dir_name)
        if os.path.exists(source_path):
            dest_path = os.path.join(package_dir, dir_name)
            shutil.copytree(source_path, dest_path)
            print(f"  Copied {dir_name}/")
    
    # Copy main README
    readme_source = os.path.join(base_dir, "README.md")
    if os.path.exists(readme_source):
        shutil.copy2(readme_source, package_dir)
        print("  Copied README.md")
    
    # Create manifest
    manifest = create_package_manifest()
    manifest_path = os.path.join(package_dir, "package_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print("  Created package_manifest.json")
    
    # Create installation guide
    install_guide = """# Installation and Usage Guide

## Quick Start

1. Extract the ZIP file to your desired location
2. Install Python dependencies:
   ```bash
   pip install numpy scipy pandas matplotlib
   ```

3. Test the installation:
   ```python
   from algorithms.quantum_annealing_optimization import QuantumAnnealingOptimizer
   print("Installation successful!")
   ```

## Directory Structure

- `algorithms/` - Advanced optimization and ML algorithms
- `functions/` - Financial and mathematical functions  
- `equations/` - Mathematical models and PDEs
- `implementations/` - Cross-language implementations
- `examples/` - Comprehensive use cases
- `documentation/` - Technical documentation

## Key Features

### Algorithms ({algorithms_count})
- Quantum Annealing Optimization
- Advanced Monte Carlo Tree Search
- Transformer Attention Mechanism
- Variational Autoencoder
- Genetic Algorithm with Adaptive Mutation
- Fast Fourier Transform
- And more...

### Functions ({functions_count})  
- Advanced Options Pricing
- Portfolio Risk Management
- Neural Network Activations
- Cryptographic Hash Functions
- Yield Curve Construction
- And more...

### Equations ({equations_count})
- Black-Scholes-Merton PDE
- Capital Asset Pricing Model
- Modern Portfolio Theory
- And more...

## Usage Examples

### Portfolio Optimization
```python
from algorithms.quantum_annealing_optimization import QuantumAnnealingOptimizer
from functions.portfolio_risk_management import PortfolioRiskManager

# Optimize portfolio allocation
optimizer = QuantumAnnealingOptimizer(cost_function, schedule)
optimal_weights, optimal_cost = optimizer.optimize(initial_weights)

# Analyze risk
risk_manager = PortfolioRiskManager(returns)
risk_metrics = risk_manager.calculate_risk_metrics()
```

### Options Pricing
```python
from functions.advanced_options_pricing import AdvancedOptionsPricer
from equations.black_scholes_merton_pde import BlackScholesPDESolver

# Price options using multiple methods
pricer = AdvancedOptionsPricer()
bs_result = pricer.black_scholes_price(params)
monte_carlo_result = pricer.monte_carlo_price(params)

# Solve Black-Scholes PDE numerically
pde_solver = BlackScholesPDESolver(pde_params, payoff)
pde_result = pde_solver.solve_crank_nicolson()
```

### DeFi AMM Analysis
```python
from algorithms.automated_market_maker import AMMAlgorithm, PoolState

# Analyze AMM pools
amm = AMMAlgorithm(AMMType.CONSTANT_PRODUCT)
swap_output = amm.calculate_swap_output(pool, amount_in)
il_data = amm.calculate_impermanent_loss(pool, price_ratio)
```

## Cross-Language Support

Implementations available in:
- Python (primary)
- JavaScript 
- Java
- C++
- Rust
- Go
- TypeScript

## Performance Notes

- All algorithms are optimized for performance
- Vectorized operations using NumPy
- Efficient memory usage patterns
- Parallel processing where applicable

## License

MIT License - Free for commercial and educational use.

## Support

For questions or issues, please refer to the documentation or examples provided.
""".format(
    algorithms_count=manifest["statistics"]["algorithms_count"],
    functions_count=manifest["statistics"]["functions_count"], 
    equations_count=manifest["statistics"]["equations_count"]
)
    
    with open(os.path.join(package_dir, "INSTALLATION.md"), 'w') as f:
        f.write(install_guide)
    print("  Created INSTALLATION.md")
    
    # Create comprehensive test script
    test_script = '''#!/usr/bin/env python3
"""
Comprehensive Test Suite for High-End Specialized Package
"""

import sys
import importlib
import traceback

def test_imports():
    """Test that all major modules can be imported."""
    
    modules_to_test = [
        "algorithms.quantum_annealing_optimization",
        "algorithms.advanced_monte_carlo_tree_search", 
        "algorithms.transformer_attention_mechanism",
        "algorithms.automated_market_maker",
        "algorithms.genetic_algorithm_adaptive",
        "functions.advanced_options_pricing",
        "functions.portfolio_risk_management",
        "functions.neural_network_activations",
        "equations.black_scholes_merton_pde",
        "equations.capital_asset_pricing_model",
        "equations.modern_portfolio_theory"
    ]
    
    print("Testing imports...")
    passed = 0
    failed = 0
    
    for module_name in modules_to_test:
        try:
            importlib.import_module(module_name)
            print(f"  ✓ {module_name}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {module_name}: {str(e)}")
            failed += 1
    
    print(f"\\nImport tests: {passed} passed, {failed} failed")
    return failed == 0

def test_basic_functionality():
    """Test basic functionality of key components."""
    
    print("\\nTesting basic functionality...")
    
    try:
        # Test portfolio optimization
        from algorithms.quantum_annealing_optimization import QuantumAnnealingOptimizer
        print("  ✓ Quantum annealing optimizer loaded")
        
        # Test options pricing
        from functions.advanced_options_pricing import AdvancedOptionsPricer, OptionParameters
        pricer = AdvancedOptionsPricer()
        print("  ✓ Options pricer loaded")
        
        # Test activation functions
        from functions.neural_network_activations import ActivationAnalyzer
        analyzer = ActivationAnalyzer()
        print("  ✓ Activation analyzer loaded")
        
        # Test Modern Portfolio Theory
        from equations.modern_portfolio_theory import ModernPortfolioTheory
        import numpy as np
        
        # Quick functionality test
        returns = np.random.normal(0.001, 0.02, (100, 3))
        mpt = ModernPortfolioTheory(returns)
        min_var = mpt.minimum_variance_portfolio()
        print("  ✓ Portfolio optimization working")
        
        print("\\nAll basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ Functionality test failed: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    
    print("=" * 60)
    print("HIGH-END SPECIALIZED PACKAGE TEST SUITE")
    print("=" * 60)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test functionality  
    functionality_ok = test_basic_functionality()
    
    # Summary
    print("\\n" + "=" * 60)
    if imports_ok and functionality_ok:
        print("🎉 ALL TESTS PASSED - Package is ready to use!")
    else:
        print("❌ Some tests failed - Please check the error messages above")
    print("=" * 60)

if __name__ == "__main__":
    main()
'''
    
    with open(os.path.join(package_dir, "test_package.py"), 'w') as f:
        f.write(test_script)
    print("  Created test_package.py")
    
    # Create ZIP file
    zip_filename = "high_end_specialized_premium.zip"
    zip_path = os.path.join(os.path.dirname(base_dir), zip_filename)
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(package_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arc_name = os.path.relpath(file_path, package_dir)
                zipf.write(file_path, arc_name)
    
    # Clean up temporary directory
    shutil.rmtree(package_dir)
    
    # Get final package stats
    zip_size = os.path.getsize(zip_path)
    
    print(f"\\n✅ Premium package created successfully!")
    print(f"   📦 File: {zip_filename}")
    print(f"   📊 Size: {zip_size / 1024 / 1024:.2f} MB")
    print(f"   📁 Files: {manifest['statistics']['total_files']}")
    print(f"   🔢 Lines of code: {manifest['statistics']['total_lines_of_code']:,}")
    print(f"   🧮 Algorithms: {manifest['statistics']['algorithms_count']}")
    print(f"   🔧 Functions: {manifest['statistics']['functions_count']}")
    print(f"   📐 Equations: {manifest['statistics']['equations_count']}")
    print(f"   🌐 Languages: {len(manifest['statistics']['languages_supported'])}")
    
    return zip_path, manifest


if __name__ == "__main__":
    zip_path, manifest = create_premium_package()
    
    # Print summary for download page
    print("\\n" + "="*60)
    print("DOWNLOAD PAGE INFORMATION")
    print("="*60)
    print(f"Package Size: ~{os.path.getsize(zip_path) / 1024 / 1024:.1f}MB ZIP")
    print(f"Total Files: {manifest['statistics']['total_files']}")
    print(f"Code Examples: {len(manifest['categories']['algorithms']) + len(manifest['categories']['functions']) + len(manifest['categories']['equations'])}")
    print(f"Implementation Languages: {', '.join(manifest['statistics']['languages_supported'])}")
    print(f"Total Lines of Code: {manifest['statistics']['total_lines_of_code']:,}")