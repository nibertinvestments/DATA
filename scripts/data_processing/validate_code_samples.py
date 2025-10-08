#!/usr/bin/env python3
"""
Validate Code Samples for Syntax Correctness
Tests samples across multiple languages
"""

import os
import subprocess
from pathlib import Path
from collections import defaultdict


class CodeSampleValidator:
    """Validate code samples for syntax correctness."""
    
    def __init__(self, base_path: str = "/home/runner/work/DATA/DATA"):
        self.base_path = Path(base_path)
        self.code_samples_path = self.base_path / "code_samples"
        
    def validate_python(self):
        """Validate Python samples."""
        print("🐍 Validating Python samples...")
        python_path = self.code_samples_path / "python"
        passed = 0
        failed = 0
        
        for py_file in python_path.glob("*.py"):
            try:
                subprocess.run(
                    ['python3', '-m', 'py_compile', str(py_file)],
                    check=True,
                    capture_output=True,
                    timeout=5
                )
                passed += 1
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                failed += 1
                print(f"  ❌ {py_file.name}")
        
        print(f"  ✅ Passed: {passed}/{passed + failed}")
        return passed, failed
    
    def validate_javascript(self):
        """Validate JavaScript samples."""
        print("🟨 Validating JavaScript samples...")
        js_path = self.code_samples_path / "javascript"
        passed = 0
        failed = 0
        
        for js_file in js_path.glob("*.js"):
            try:
                subprocess.run(
                    ['node', '-c', str(js_file)],
                    check=True,
                    capture_output=True,
                    timeout=5
                )
                passed += 1
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
                # Node might not be available, skip
                failed += 1
        
        if passed + failed > 0:
            print(f"  ✅ Passed: {passed}/{passed + failed}")
        else:
            print(f"  ⚠️  Node.js not available, skipping")
        return passed, failed
    
    def count_samples_by_category(self):
        """Count samples by category."""
        print("\n📊 Counting samples by category...")
        categories = defaultdict(int)
        
        for lang_dir in self.code_samples_path.iterdir():
            if lang_dir.is_dir() and not lang_dir.name.startswith('.'):
                for file_path in lang_dir.iterdir():
                    if file_path.is_file() and not file_path.name.startswith('.'):
                        # Extract category
                        parts = file_path.stem.split('_')
                        if len(parts) >= 1:
                            category = parts[0]
                            categories[category] += 1
        
        print("\nTop categories:")
        for category, count in sorted(categories.items(), key=lambda x: -x[1])[:15]:
            print(f"  {category:25s}: {count:3d} files")
        
        return categories
    
    def generate_summary(self):
        """Generate validation summary."""
        print("\n" + "=" * 70)
        print("CODE SAMPLE VALIDATION SUMMARY")
        print("=" * 70)
        
        # Count total files
        total = 0
        by_language = {}
        for lang_dir in self.code_samples_path.iterdir():
            if lang_dir.is_dir() and not lang_dir.name.startswith('.'):
                count = len(list(lang_dir.glob("*")))
                by_language[lang_dir.name] = count
                total += count
        
        print(f"\n📁 Total Files: {total}")
        print(f"🌐 Languages: {len(by_language)}")
        print(f"📚 Average per Language: {total // len(by_language)}")
        
        print("\n📊 Files by Language:")
        for lang in sorted(by_language.keys()):
            print(f"  {lang:15s}: {by_language[lang]:3d} files")
        
        return total, by_language


def main():
    """Main validation function."""
    print("=" * 70)
    print("CODE SAMPLE VALIDATION")
    print("=" * 70)
    print()
    
    validator = CodeSampleValidator()
    
    # Validate Python
    py_passed, py_failed = validator.validate_python()
    
    # Validate JavaScript
    js_passed, js_failed = validator.validate_javascript()
    
    # Count by category
    categories = validator.count_samples_by_category()
    
    # Generate summary
    total, by_language = validator.generate_summary()
    
    print("\n" + "=" * 70)
    print("✅ VALIDATION COMPLETE")
    print("=" * 70)
    
    if py_failed == 0:
        print("🎉 All Python samples passed syntax validation!")
    if js_passed > 0 and js_failed == 0:
        print("🎉 All JavaScript samples passed syntax validation!")


if __name__ == "__main__":
    main()
