#!/usr/bin/env python3
"""
Dataset Usage Examples
Demonstrates how to load and use the external datasets for AI training.
"""

import json
from pathlib import Path
from typing import Dict, List, Any


class DatasetLoader:
    """Utility class for loading and working with datasets."""
    
    def __init__(self, base_dir: str = "datasets/raw/external"):
        """Initialize loader with base directory."""
        self.base_dir = Path(base_dir)
    
    def load_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Load a specific dataset by name."""
        filepath = self.base_dir / f"{dataset_name}_dataset.json"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def list_available_datasets(self) -> List[str]:
        """List all available datasets."""
        datasets = []
        for file_path in self.base_dir.glob("*_dataset.json"):
            datasets.append(file_path.stem.replace("_dataset", ""))
        return sorted(datasets)


def example_1_load_error_patterns():
    """Example 1: Load and analyze error patterns."""
    print("\n" + "=" * 70)
    print("Example 1: Working with Error Patterns")
    print("=" * 70 + "\n")
    
    loader = DatasetLoader()
    data = loader.load_dataset("common_programming_errors")
    
    print(f"📊 Dataset: {data['metadata']['description']}")
    print(f"📝 Total Patterns: {data['metadata']['total_patterns']}")
    print(f"🌐 Languages: {', '.join(data['metadata']['languages'])}\n")
    
    # Show Python errors
    python_errors = [e for e in data['error_patterns'] if e['language'] == 'python']
    
    print(f"🐍 Python Error Patterns ({len(python_errors)}):")
    for error in python_errors[:3]:  # Show first 3
        print(f"\n  • {error['error_type']}")
        print(f"    Severity: {error['severity']}")
        print(f"    Description: {error['description']}")
        print(f"    Common Cause: {error['common_cause']}")


def example_2_code_translation():
    """Example 2: Work with code translation examples."""
    print("\n" + "=" * 70)
    print("Example 2: Cross-Language Code Translation")
    print("=" * 70 + "\n")
    
    loader = DatasetLoader()
    data = loader.load_dataset("code_translation_examples")
    
    print(f"📊 Dataset: {data['metadata']['description']}")
    print(f"📝 Total Examples: {data['metadata']['total_examples']}\n")
    
    # Show first translation example
    example = data['translation_examples'][0]
    
    print(f"🔄 Concept: {example['concept']}")
    print(f"📚 Difficulty: {example['difficulty']}")
    print(f"🌐 Languages Available: {', '.join(example['implementations'].keys())}\n")
    
    # Show Python implementation
    if 'python' in example['implementations']:
        print("🐍 Python Implementation:")
        print(example['implementations']['python'][:200] + "...")
    
    # Show JavaScript implementation
    if 'javascript' in example['implementations']:
        print("\n📜 JavaScript Implementation:")
        print(example['implementations']['javascript'][:200] + "...")


def example_3_security_patterns():
    """Example 3: Analyze security vulnerabilities."""
    print("\n" + "=" * 70)
    print("Example 3: Security Vulnerability Analysis")
    print("=" * 70 + "\n")
    
    loader = DatasetLoader()
    data = loader.load_dataset("security_vulnerabilities")
    
    print(f"📊 Dataset: {data['metadata']['description']}")
    print(f"🔒 Total Vulnerabilities: {data['metadata']['total_vulnerabilities']}\n")
    
    # Group by severity
    by_severity = {}
    for vuln in data['vulnerabilities']:
        severity = vuln['severity']
        if severity not in by_severity:
            by_severity[severity] = []
        by_severity[severity].append(vuln['vulnerability_type'])
    
    print("📊 Vulnerabilities by Severity:")
    for severity in ['critical', 'high', 'medium', 'low']:
        if severity in by_severity:
            print(f"  • {severity.upper()}: {', '.join(by_severity[severity])}")
    
    # Show first vulnerability
    vuln = data['vulnerabilities'][0]
    print(f"\n🔍 Example: {vuln['vulnerability_type']}")
    print(f"   Severity: {vuln['severity']}")
    print(f"   Impact: {vuln['impact'][:100]}...")
    print(f"   Prevention Measures: {len(vuln['prevention'])} strategies")


def example_4_design_patterns():
    """Example 4: Explore design patterns."""
    print("\n" + "=" * 70)
    print("Example 4: Design Pattern Exploration")
    print("=" * 70 + "\n")
    
    loader = DatasetLoader()
    data = loader.load_dataset("design_patterns")
    
    print(f"📊 Dataset: {data['metadata']['description']}")
    print(f"🏗️  Total Patterns: {data['metadata']['total_patterns']}\n")
    
    # Group by category
    by_category = {}
    for pattern in data['patterns']:
        category = pattern['category']
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(pattern['pattern_name'])
    
    print("📊 Patterns by Category:")
    for category, patterns in by_category.items():
        print(f"  • {category.upper()}: {', '.join(patterns)}")
    
    # Show first pattern
    pattern = data['patterns'][0]
    print(f"\n🎯 Pattern: {pattern['pattern_name']}")
    print(f"   Category: {pattern['category']}")
    print(f"   Use Case: {pattern['use_case']}")
    print(f"   Languages Available: {', '.join(pattern['implementations'].keys())}")
    print(f"   Pros: {len(pattern['pros'])} | Cons: {len(pattern['cons'])}")


def example_5_performance_optimization():
    """Example 5: Performance optimization examples."""
    print("\n" + "=" * 70)
    print("Example 5: Performance Optimization")
    print("=" * 70 + "\n")
    
    loader = DatasetLoader()
    data = loader.load_dataset("performance_optimization")
    
    print(f"📊 Dataset: {data['metadata']['description']}")
    print(f"⚡ Total Optimizations: {data['metadata']['total_optimizations']}\n")
    
    print("⚡ Optimization Examples:")
    for opt in data['optimizations']:
        print(f"\n  • {opt['optimization_type']}")
        print(f"    Problem: {opt['problem']}")
        print(f"    Performance Gain: {opt['performance_gain']}")
        print(f"    Explanation: {opt['explanation']}")


def example_6_training_preparation():
    """Example 6: Prepare data for AI training."""
    print("\n" + "=" * 70)
    print("Example 6: Preparing Data for AI Training")
    print("=" * 70 + "\n")
    
    loader = DatasetLoader()
    
    # Load multiple datasets
    error_data = loader.load_dataset("common_programming_errors")
    security_data = loader.load_dataset("security_vulnerabilities")
    
    # Combine for comprehensive training
    training_samples = []
    
    # Add error patterns
    for error in error_data['error_patterns']:
        training_samples.append({
            'type': 'error_fix',
            'language': error['language'],
            'input': error['buggy_code'],
            'output': error['fixed_code'],
            'explanation': error['explanation'],
            'severity': error['severity']
        })
    
    # Add security fixes
    for vuln in security_data['vulnerabilities']:
        for lang in vuln['vulnerable_code'].keys():
            training_samples.append({
                'type': 'security_fix',
                'language': lang,
                'input': vuln['vulnerable_code'][lang],
                'output': vuln['fixed_code'][lang],
                'explanation': vuln['impact'],
                'severity': vuln['severity']
            })
    
    print(f"📊 Combined Training Dataset:")
    print(f"   Total Samples: {len(training_samples)}")
    print(f"   Error Fixes: {sum(1 for s in training_samples if s['type'] == 'error_fix')}")
    print(f"   Security Fixes: {sum(1 for s in training_samples if s['type'] == 'security_fix')}")
    
    # Show language distribution
    languages = {}
    for sample in training_samples:
        lang = sample['language']
        languages[lang] = languages.get(lang, 0) + 1
    
    print(f"\n   Language Distribution:")
    for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True):
        print(f"     • {lang}: {count} samples")


def list_all_datasets():
    """List all available datasets."""
    print("\n" + "=" * 70)
    print("Available Datasets")
    print("=" * 70 + "\n")
    
    loader = DatasetLoader()
    datasets = loader.list_available_datasets()
    
    print(f"📚 Total Datasets: {len(datasets)}\n")
    
    for i, dataset_name in enumerate(datasets, 1):
        print(f"{i:2d}. {dataset_name}")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("Dataset Usage Examples for AI Training")
    print("=" * 70)
    
    # List all datasets first
    list_all_datasets()
    
    # Run examples
    example_1_load_error_patterns()
    example_2_code_translation()
    example_3_security_patterns()
    example_4_design_patterns()
    example_5_performance_optimization()
    example_6_training_preparation()
    
    print("\n" + "=" * 70)
    print("✅ All examples completed successfully!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
