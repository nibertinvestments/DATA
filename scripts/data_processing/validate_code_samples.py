#!/usr/bin/env python3
"""
Validate Code Samples for Syntax Correctness
Tests samples across multiple languages and deletes files that fail validation
"""

import os
import subprocess
from pathlib import Path
from collections import defaultdict


class CodeSampleValidator:
    """Validate code samples for syntax correctness."""
    
    def __init__(self, base_path: str = "/home/runner/work/DATA/DATA", delete_invalid: bool = False):
        self.base_path = Path(base_path)
        self.code_samples_path = self.base_path / "code_samples"
        self.delete_invalid = delete_invalid
        self.failed_files = []
        
    def validate_python(self):
        """Validate Python samples."""
        print("🐍 Validating Python samples...")
        python_path = self.code_samples_path / "python"
        passed = 0
        failed = 0
        failed_files = []
        
        for py_file in python_path.glob("*.py"):
            try:
                subprocess.run(
                    ['python3', '-m', 'py_compile', str(py_file)],
                    check=True,
                    capture_output=True,
                    timeout=5
                )
                passed += 1
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                failed += 1
                failed_files.append(py_file)
                print(f"  ❌ {py_file.name}")
                if self.delete_invalid:
                    try:
                        py_file.unlink()
                        print(f"     🗑️  Deleted {py_file.name}")
                    except Exception as del_err:
                        print(f"     ⚠️  Could not delete: {del_err}")
        
        print(f"  ✅ Passed: {passed}/{passed + failed}")
        self.failed_files.extend(failed_files)
        return passed, failed, failed_files
    
    def validate_javascript(self):
        """Validate JavaScript samples."""
        print("🟨 Validating JavaScript samples...")
        js_path = self.code_samples_path / "javascript"
        passed = 0
        failed = 0
        failed_files = []
        
        # Check if node is available
        try:
            subprocess.run(['node', '--version'], check=True, capture_output=True, timeout=2)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            print(f"  ⚠️  Node.js not available, skipping JavaScript validation")
            return 0, 0, []
        
        for js_file in js_path.glob("*.js"):
            try:
                subprocess.run(
                    ['node', '-c', str(js_file)],
                    check=True,
                    capture_output=True,
                    timeout=5
                )
                passed += 1
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                failed += 1
                failed_files.append(js_file)
                print(f"  ❌ {js_file.name}")
                if self.delete_invalid:
                    try:
                        js_file.unlink()
                        print(f"     🗑️  Deleted {js_file.name}")
                    except Exception as del_err:
                        print(f"     ⚠️  Could not delete: {del_err}")
        
        print(f"  ✅ Passed: {passed}/{passed + failed}")
        self.failed_files.extend(failed_files)
        return passed, failed, failed_files
    
    def validate_typescript(self):
        """Validate TypeScript samples."""
        print("🔷 Validating TypeScript samples...")
        ts_path = self.code_samples_path / "typescript"
        passed = 0
        failed = 0
        failed_files = []
        
        # Check if tsc is available
        try:
            subprocess.run(['tsc', '--version'], check=True, capture_output=True, timeout=2)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            print(f"  ⚠️  TypeScript compiler not available, skipping")
            return 0, 0, []
        
        for ts_file in ts_path.glob("*.ts"):
            try:
                subprocess.run(
                    ['tsc', '--noEmit', '--skipLibCheck', str(ts_file)],
                    check=True,
                    capture_output=True,
                    timeout=10
                )
                passed += 1
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                failed += 1
                failed_files.append(ts_file)
                print(f"  ❌ {ts_file.name}")
                if self.delete_invalid:
                    try:
                        ts_file.unlink()
                        print(f"     🗑️  Deleted {ts_file.name}")
                    except Exception as del_err:
                        print(f"     ⚠️  Could not delete: {del_err}")
        
        print(f"  ✅ Passed: {passed}/{passed + failed}")
        self.failed_files.extend(failed_files)
        return passed, failed, failed_files
    
    def validate_go(self):
        """Validate Go samples."""
        print("🐹 Validating Go samples...")
        go_path = self.code_samples_path / "go"
        passed = 0
        failed = 0
        failed_files = []
        
        # Check if go is available
        try:
            subprocess.run(['go', 'version'], check=True, capture_output=True, timeout=2)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            print(f"  ⚠️  Go compiler not available, skipping")
            return 0, 0, []
        
        for go_file in go_path.glob("*.go"):
            try:
                # Use go vet for syntax checking
                subprocess.run(
                    ['go', 'vet', str(go_file)],
                    check=True,
                    capture_output=True,
                    timeout=10
                )
                passed += 1
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                # Try gofmt as fallback
                try:
                    result = subprocess.run(
                        ['gofmt', '-e', str(go_file)],
                        capture_output=True,
                        timeout=5
                    )
                    if result.returncode == 0 and not result.stderr:
                        passed += 1
                        continue
                except:
                    pass
                
                failed += 1
                failed_files.append(go_file)
                print(f"  ❌ {go_file.name}")
                if self.delete_invalid:
                    try:
                        go_file.unlink()
                        print(f"     🗑️  Deleted {go_file.name}")
                    except Exception as del_err:
                        print(f"     ⚠️  Could not delete: {del_err}")
        
        print(f"  ✅ Passed: {passed}/{passed + failed}")
        self.failed_files.extend(failed_files)
        return passed, failed, failed_files
    
    def validate_rust(self):
        """Validate Rust samples."""
        print("🦀 Validating Rust samples...")
        rust_path = self.code_samples_path / "rust"
        passed = 0
        failed = 0
        failed_files = []
        
        # Check if rustc is available
        try:
            subprocess.run(['rustc', '--version'], check=True, capture_output=True, timeout=2)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            print(f"  ⚠️  Rust compiler not available, skipping")
            return 0, 0, []
        
        for rs_file in rust_path.glob("*.rs"):
            try:
                # Use rustc with --crate-type lib to just check syntax
                subprocess.run(
                    ['rustc', '--crate-type', 'lib', '--emit', 'metadata', '-o', '/dev/null', str(rs_file)],
                    check=True,
                    capture_output=True,
                    timeout=15
                )
                passed += 1
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                failed += 1
                failed_files.append(rs_file)
                print(f"  ❌ {rs_file.name}")
                if self.delete_invalid:
                    try:
                        rs_file.unlink()
                        print(f"     🗑️  Deleted {rs_file.name}")
                    except Exception as del_err:
                        print(f"     ⚠️  Could not delete: {del_err}")
        
        print(f"  ✅ Passed: {passed}/{passed + failed}")
        self.failed_files.extend(failed_files)
        return passed, failed, failed_files
    
    def validate_java(self):
        """Validate Java samples."""
        print("☕ Validating Java samples...")
        java_path = self.code_samples_path / "java"
        passed = 0
        failed = 0
        failed_files = []
        
        # Check if javac is available
        try:
            subprocess.run(['javac', '-version'], check=True, capture_output=True, timeout=2)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            print(f"  ⚠️  Java compiler not available, skipping")
            return 0, 0, []
        
        for java_file in java_path.glob("*.java"):
            try:
                # Compile to check syntax
                subprocess.run(
                    ['javac', '-Xlint:all', str(java_file)],
                    check=True,
                    capture_output=True,
                    timeout=10,
                    cwd=java_path
                )
                passed += 1
                # Clean up .class files
                class_file = java_file.with_suffix('.class')
                if class_file.exists():
                    class_file.unlink()
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                failed += 1
                failed_files.append(java_file)
                print(f"  ❌ {java_file.name}")
                if self.delete_invalid:
                    try:
                        java_file.unlink()
                        print(f"     🗑️  Deleted {java_file.name}")
                    except Exception as del_err:
                        print(f"     ⚠️  Could not delete: {del_err}")
        
        print(f"  ✅ Passed: {passed}/{passed + failed}")
        self.failed_files.extend(failed_files)
        return passed, failed, failed_files
    
    def validate_php(self):
        """Validate PHP samples."""
        print("🐘 Validating PHP samples...")
        php_path = self.code_samples_path / "php"
        passed = 0
        failed = 0
        failed_files = []
        
        # Check if php is available
        try:
            subprocess.run(['php', '--version'], check=True, capture_output=True, timeout=2)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            print(f"  ⚠️  PHP not available, skipping")
            return 0, 0, []
        
        for php_file in php_path.glob("*.php"):
            try:
                subprocess.run(
                    ['php', '-l', str(php_file)],
                    check=True,
                    capture_output=True,
                    timeout=5
                )
                passed += 1
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                failed += 1
                failed_files.append(php_file)
                print(f"  ❌ {php_file.name}")
                if self.delete_invalid:
                    try:
                        php_file.unlink()
                        print(f"     🗑️  Deleted {php_file.name}")
                    except Exception as del_err:
                        print(f"     ⚠️  Could not delete: {del_err}")
        
        print(f"  ✅ Passed: {passed}/{passed + failed}")
        self.failed_files.extend(failed_files)
        return passed, failed, failed_files
    
    def validate_ruby(self):
        """Validate Ruby samples."""
        print("💎 Validating Ruby samples...")
        ruby_path = self.code_samples_path / "ruby"
        passed = 0
        failed = 0
        failed_files = []
        
        # Check if ruby is available
        try:
            subprocess.run(['ruby', '--version'], check=True, capture_output=True, timeout=2)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            print(f"  ⚠️  Ruby not available, skipping")
            return 0, 0, []
        
        for rb_file in ruby_path.glob("*.rb"):
            try:
                subprocess.run(
                    ['ruby', '-c', str(rb_file)],
                    check=True,
                    capture_output=True,
                    timeout=5
                )
                passed += 1
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                failed += 1
                failed_files.append(rb_file)
                print(f"  ❌ {rb_file.name}")
                if self.delete_invalid:
                    try:
                        rb_file.unlink()
                        print(f"     🗑️  Deleted {rb_file.name}")
                    except Exception as del_err:
                        print(f"     ⚠️  Could not delete: {del_err}")
        
        print(f"  ✅ Passed: {passed}/{passed + failed}")
        self.failed_files.extend(failed_files)
        return passed, failed, failed_files
    
    def validate_perl(self):
        """Validate Perl samples."""
        print("🐪 Validating Perl samples...")
        perl_path = self.code_samples_path / "perl"
        passed = 0
        failed = 0
        failed_files = []
        
        # Check if perl is available
        try:
            subprocess.run(['perl', '--version'], check=True, capture_output=True, timeout=2)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            print(f"  ⚠️  Perl not available, skipping")
            return 0, 0, []
        
        for pl_file in perl_path.glob("*.pl"):
            try:
                subprocess.run(
                    ['perl', '-c', str(pl_file)],
                    check=True,
                    capture_output=True,
                    timeout=5
                )
                passed += 1
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                failed += 1
                failed_files.append(pl_file)
                print(f"  ❌ {pl_file.name}")
                if self.delete_invalid:
                    try:
                        pl_file.unlink()
                        print(f"     🗑️  Deleted {pl_file.name}")
                    except Exception as del_err:
                        print(f"     ⚠️  Could not delete: {del_err}")
        
        print(f"  ✅ Passed: {passed}/{passed + failed}")
        self.failed_files.extend(failed_files)
        return passed, failed, failed_files
    
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
    import sys
    
    # Check for delete flag
    delete_invalid = '--delete' in sys.argv or '-d' in sys.argv
    
    print("=" * 70)
    print("CODE SAMPLE VALIDATION")
    if delete_invalid:
        print("⚠️  DELETE MODE: Invalid files will be removed!")
    print("=" * 70)
    print()
    
    validator = CodeSampleValidator(delete_invalid=delete_invalid)
    
    results = {}
    
    # Validate all languages
    print("🔍 Validating all language samples...\n")
    
    results['python'] = validator.validate_python()
    results['javascript'] = validator.validate_javascript()
    results['typescript'] = validator.validate_typescript()
    results['java'] = validator.validate_java()
    results['go'] = validator.validate_go()
    results['rust'] = validator.validate_rust()
    results['php'] = validator.validate_php()
    results['ruby'] = validator.validate_ruby()
    results['perl'] = validator.validate_perl()
    
    # Count by category
    categories = validator.count_samples_by_category()
    
    # Generate summary
    total, by_language = validator.generate_summary()
    
    # Summary of validation results
    print("\n" + "=" * 70)
    print("✅ VALIDATION COMPLETE")
    print("=" * 70)
    
    total_passed = sum(r[0] for r in results.values())
    total_failed = sum(r[1] for r in results.values())
    total_tested = total_passed + total_failed
    
    if total_tested > 0:
        print(f"\n📊 Overall Results:")
        print(f"   Tested: {total_tested} files")
        print(f"   ✅ Passed: {total_passed} ({100 * total_passed / total_tested:.1f}%)")
        print(f"   ❌ Failed: {total_failed} ({100 * total_failed / total_tested:.1f}%)")
        
        if delete_invalid and total_failed > 0:
            print(f"\n🗑️  Deleted {total_failed} invalid files")
            print(f"   Total failed files removed: {len(validator.failed_files)}")
    
    print("\n" + "=" * 70)
    
    # Print language-specific results
    print("\n📋 Validation by Language:")
    for lang, (passed, failed, _) in results.items():
        if passed + failed > 0:
            status = "✅" if failed == 0 else "⚠️"
            print(f"  {status} {lang:15s}: {passed}/{passed + failed} passed")
        else:
            print(f"  ⏭️  {lang:15s}: skipped (compiler not available)")


if __name__ == "__main__":
    main()
