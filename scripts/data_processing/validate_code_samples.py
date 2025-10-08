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
    
    def validate_cpp(self):
        """Validate C++ samples."""
        print("⚡ Validating C++ samples...")
        cpp_path = self.code_samples_path / "cpp"
        passed = 0
        failed = 0
        failed_files = []
        
        # Check if g++ is available
        try:
            subprocess.run(['g++', '--version'], check=True, capture_output=True, timeout=2)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            print(f"  ⚠️  C++ compiler not available, skipping")
            return 0, 0, []
        
        for cpp_file in cpp_path.glob("*.cpp"):
            try:
                # Compile with syntax check only
                subprocess.run(
                    ['g++', '-std=c++17', '-fsyntax-only', '-Wall', str(cpp_file)],
                    check=True,
                    capture_output=True,
                    timeout=10
                )
                passed += 1
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                failed += 1
                failed_files.append(cpp_file)
                print(f"  ❌ {cpp_file.name}")
                if self.delete_invalid:
                    try:
                        cpp_file.unlink()
                        print(f"     🗑️  Deleted {cpp_file.name}")
                    except Exception as del_err:
                        print(f"     ⚠️  Could not delete: {del_err}")
        
        print(f"  ✅ Passed: {passed}/{passed + failed}")
        self.failed_files.extend(failed_files)
        return passed, failed, failed_files
    
    def validate_csharp(self):
        """Validate C# samples."""
        print("🔷 Validating C# samples...")
        csharp_path = self.code_samples_path / "csharp"
        passed = 0
        failed = 0
        failed_files = []
        
        # Check if dotnet is available (use different check)
        try:
            result = subprocess.run(['dotnet', '--list-sdks'], capture_output=True, timeout=5)
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, 'dotnet')
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            print(f"  ⚠️  .NET compiler not available, skipping")
            return 0, 0, []
        
        # Create a temporary project for validation
        import tempfile
        import shutil
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Initialize a console project
            subprocess.run(
                ['dotnet', 'new', 'console', '-n', 'Validator', '-o', str(temp_dir)],
                check=True,
                capture_output=True,
                timeout=30
            )
            
            for cs_file in csharp_path.glob("*.cs"):
                try:
                    # Copy file to temp project
                    dest = temp_dir / cs_file.name
                    shutil.copy(cs_file, dest)
                    
                    # Try to build
                    result = subprocess.run(
                        ['dotnet', 'build', '--no-restore'],
                        cwd=temp_dir,
                        capture_output=True,
                        timeout=20
                    )
                    
                    # Remove the copied file
                    dest.unlink()
                    
                    if result.returncode == 0:
                        passed += 1
                    else:
                        # Some errors are expected (missing Program.Main, etc), check for syntax errors
                        error_output = result.stderr.decode('utf-8', errors='ignore') + result.stdout.decode('utf-8', errors='ignore')
                        # Real syntax errors contain "error CS" markers
                        if 'error CS' in error_output and not 'CS5001' in error_output:  # CS5001 is "no entry point"
                            failed += 1
                            failed_files.append(cs_file)
                            print(f"  ❌ {cs_file.name}")
                            if self.delete_invalid:
                                try:
                                    cs_file.unlink()
                                    print(f"     🗑️  Deleted {cs_file.name}")
                                except Exception as del_err:
                                    print(f"     ⚠️  Could not delete: {del_err}")
                        else:
                            # No critical syntax errors
                            passed += 1
                            
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                    failed += 1
                    failed_files.append(cs_file)
                    print(f"  ❌ {cs_file.name}")
                    if self.delete_invalid:
                        try:
                            cs_file.unlink()
                            print(f"     🗑️  Deleted {cs_file.name}")
                        except Exception as del_err:
                            print(f"     ⚠️  Could not delete: {del_err}")
        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        print(f"  ✅ Passed: {passed}/{passed + failed}")
        self.failed_files.extend(failed_files)
        return passed, failed, failed_files
    
    def validate_kotlin(self):
        """Validate Kotlin samples."""
        print("🟣 Validating Kotlin samples...")
        kotlin_path = self.code_samples_path / "kotlin"
        passed = 0
        failed = 0
        failed_files = []
        
        # Check if kotlinc is available
        try:
            result = subprocess.run(['kotlinc', '-version'], capture_output=True, timeout=5)
            # kotlinc -version writes to stderr, check both
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print(f"  ⚠️  Kotlin compiler not available, skipping")
            return 0, 0, []
        
        import tempfile
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            for kt_file in kotlin_path.glob("*.kt"):
                try:
                    # Compile to check syntax
                    result = subprocess.run(
                        ['kotlinc', str(kt_file), '-d', str(temp_dir)],
                        capture_output=True,
                        timeout=45
                    )
                    
                    if result.returncode == 0:
                        passed += 1
                    else:
                        # Check if it's a real syntax error vs dependency issue
                        error_output = result.stderr.decode('utf-8', errors='ignore') + result.stdout.decode('utf-8', errors='ignore')
                        # Ignore unresolved reference errors (dependencies) - focus on syntax
                        if 'error:' in error_output.lower():
                            # Check if it's only dependency errors
                            error_lines = [line for line in error_output.split('\n') if 'error:' in line.lower()]
                            dependency_errors = sum(1 for line in error_lines if 'unresolved reference' in line.lower() or 'unresolved import' in line.lower())
                            total_errors = len(error_lines)
                            
                            # If all errors are dependency-related, consider it passed
                            if dependency_errors == total_errors and total_errors > 0:
                                passed += 1
                            else:
                                failed += 1
                                failed_files.append(kt_file)
                                print(f"  ❌ {kt_file.name}")
                                if self.delete_invalid:
                                    try:
                                        kt_file.unlink()
                                        print(f"     🗑️  Deleted {kt_file.name}")
                                    except Exception as del_err:
                                        print(f"     ⚠️  Could not delete: {del_err}")
                        else:
                            # Warning or other non-critical issue
                            passed += 1
                            
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                    failed += 1
                    failed_files.append(kt_file)
                    print(f"  ❌ {kt_file.name}")
                    if self.delete_invalid:
                        try:
                            kt_file.unlink()
                            print(f"     🗑️  Deleted {kt_file.name}")
                        except Exception as del_err:
                            print(f"     ⚠️  Could not delete: {del_err}")
        finally:
            # Clean up temp directory
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        print(f"  ✅ Passed: {passed}/{passed + failed}")
        self.failed_files.extend(failed_files)
        return passed, failed, failed_files
    
    def validate_swift(self):
        """Validate Swift samples."""
        print("🍎 Validating Swift samples...")
        swift_path = self.code_samples_path / "swift"
        passed = 0
        failed = 0
        failed_files = []
        
        # Check if swift is available
        try:
            subprocess.run(['swift', '--version'], check=True, capture_output=True, timeout=5)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            print(f"  ⚠️  Swift compiler not available, skipping")
            return 0, 0, []
        
        for swift_file in swift_path.glob("*.swift"):
            try:
                # Use swiftc to check syntax
                result = subprocess.run(
                    ['swiftc', '-parse', str(swift_file)],
                    capture_output=True,
                    timeout=15
                )
                
                if result.returncode == 0:
                    passed += 1
                else:
                    # Check for real syntax errors
                    error_output = result.stderr.decode('utf-8', errors='ignore') + result.stdout.decode('utf-8', errors='ignore')
                    if 'error:' in error_output:
                        failed += 1
                        failed_files.append(swift_file)
                        print(f"  ❌ {swift_file.name}")
                        if self.delete_invalid:
                            try:
                                swift_file.unlink()
                                print(f"     🗑️  Deleted {swift_file.name}")
                            except Exception as del_err:
                                print(f"     ⚠️  Could not delete: {del_err}")
                    else:
                        # Warning only
                        passed += 1
                        
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                failed += 1
                failed_files.append(swift_file)
                print(f"  ❌ {swift_file.name}")
                if self.delete_invalid:
                    try:
                        swift_file.unlink()
                        print(f"     🗑️  Deleted {swift_file.name}")
                    except Exception as del_err:
                        print(f"     ⚠️  Could not delete: {del_err}")
        
        print(f"  ✅ Passed: {passed}/{passed + failed}")
        self.failed_files.extend(failed_files)
        return passed, failed, failed_files
    
    def validate_haskell(self):
        """Validate Haskell samples."""
        print("🔮 Validating Haskell samples...")
        haskell_path = self.code_samples_path / "haskell"
        passed = 0
        failed = 0
        failed_files = []
        
        # Check if ghc is available
        try:
            result = subprocess.run(['ghc', '--version'], capture_output=True, timeout=5)
            if result.returncode != 0 and not result.stdout:
                raise FileNotFoundError()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print(f"  ⚠️  Haskell compiler not available, skipping")
            return 0, 0, []
        
        import tempfile
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            for hs_file in haskell_path.glob("*.hs"):
                try:
                    # Use ghc to check syntax only
                    result = subprocess.run(
                        ['ghc', '-fno-code', '-outputdir', str(temp_dir), str(hs_file)],
                        capture_output=True,
                        timeout=20
                    )
                    
                    if result.returncode == 0:
                        passed += 1
                    else:
                        # Check for real syntax/type errors
                        error_output = result.stderr.decode('utf-8', errors='ignore') + result.stdout.decode('utf-8', errors='ignore')
                        if 'error:' in error_output.lower():
                            failed += 1
                            failed_files.append(hs_file)
                            print(f"  ❌ {hs_file.name}")
                            if self.delete_invalid:
                                try:
                                    hs_file.unlink()
                                    print(f"     🗑️  Deleted {hs_file.name}")
                                except Exception as del_err:
                                    print(f"     ⚠️  Could not delete: {del_err}")
                        else:
                            # Warning only
                            passed += 1
                            
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                    failed += 1
                    failed_files.append(hs_file)
                    print(f"  ❌ {hs_file.name}")
                    if self.delete_invalid:
                        try:
                            hs_file.unlink()
                            print(f"     🗑️  Deleted {hs_file.name}")
                        except Exception as del_err:
                            print(f"     ⚠️  Could not delete: {del_err}")
        finally:
            # Clean up temp directory
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        print(f"  ✅ Passed: {passed}/{passed + failed}")
        self.failed_files.extend(failed_files)
        return passed, failed, failed_files
    
    def validate_scala(self):
        """Validate Scala samples."""
        print("🔴 Validating Scala samples...")
        scala_path = self.code_samples_path / "scala"
        passed = 0
        failed = 0
        failed_files = []
        
        # Check if scalac is available
        try:
            subprocess.run(['scalac', '-version'], check=True, capture_output=True, timeout=2)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            print(f"  ⚠️  Scala compiler not available, skipping")
            return 0, 0, []
        
        for scala_file in scala_path.glob("*.scala"):
            try:
                # Compile to check syntax
                subprocess.run(
                    ['scalac', '-d', '/tmp', str(scala_file)],
                    check=True,
                    capture_output=True,
                    timeout=30
                )
                passed += 1
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                failed += 1
                failed_files.append(scala_file)
                print(f"  ❌ {scala_file.name}")
                if self.delete_invalid:
                    try:
                        scala_file.unlink()
                        print(f"     🗑️  Deleted {scala_file.name}")
                    except Exception as del_err:
                        print(f"     ⚠️  Could not delete: {del_err}")
        
        print(f"  ✅ Passed: {passed}/{passed + failed}")
        self.failed_files.extend(failed_files)
        return passed, failed, failed_files
    
    def validate_dart(self):
        """Validate Dart samples."""
        print("🎯 Validating Dart samples...")
        dart_path = self.code_samples_path / "dart"
        passed = 0
        failed = 0
        failed_files = []
        
        # Check if dart is available
        try:
            subprocess.run(['dart', '--version'], check=True, capture_output=True, timeout=2)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            print(f"  ⚠️  Dart not available, skipping")
            return 0, 0, []
        
        for dart_file in dart_path.glob("*.dart"):
            try:
                # Use dart analyze
                subprocess.run(
                    ['dart', 'analyze', str(dart_file)],
                    check=True,
                    capture_output=True,
                    timeout=10
                )
                passed += 1
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                failed += 1
                failed_files.append(dart_file)
                print(f"  ❌ {dart_file.name}")
                if self.delete_invalid:
                    try:
                        dart_file.unlink()
                        print(f"     🗑️  Deleted {dart_file.name}")
                    except Exception as del_err:
                        print(f"     ⚠️  Could not delete: {del_err}")
        
        print(f"  ✅ Passed: {passed}/{passed + failed}")
        self.failed_files.extend(failed_files)
        return passed, failed, failed_files
    
    def validate_r(self):
        """Validate R samples."""
        print("📊 Validating R samples...")
        r_path = self.code_samples_path / "r"
        passed = 0
        failed = 0
        failed_files = []
        
        # Check if Rscript is available
        try:
            subprocess.run(['Rscript', '--version'], check=True, capture_output=True, timeout=2)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            print(f"  ⚠️  R not available, skipping")
            return 0, 0, []
        
        for r_file in r_path.glob("*.R"):
            try:
                # Use Rscript to parse
                subprocess.run(
                    ['Rscript', '-e', f'parse("{r_file}")'],
                    check=True,
                    capture_output=True,
                    timeout=10
                )
                passed += 1
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                failed += 1
                failed_files.append(r_file)
                print(f"  ❌ {r_file.name}")
                if self.delete_invalid:
                    try:
                        r_file.unlink()
                        print(f"     🗑️  Deleted {r_file.name}")
                    except Exception as del_err:
                        print(f"     ⚠️  Could not delete: {del_err}")
        
        print(f"  ✅ Passed: {passed}/{passed + failed}")
        self.failed_files.extend(failed_files)
        return passed, failed, failed_files
    
    def validate_lua(self):
        """Validate Lua samples."""
        print("🌙 Validating Lua samples...")
        lua_path = self.code_samples_path / "lua"
        passed = 0
        failed = 0
        failed_files = []
        
        # Check if lua is available
        try:
            subprocess.run(['lua', '-v'], check=True, capture_output=True, timeout=2)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            print(f"  ⚠️  Lua not available, skipping")
            return 0, 0, []
        
        for lua_file in lua_path.glob("*.lua"):
            try:
                # Use lua to check syntax
                subprocess.run(
                    ['lua', '-p', str(lua_file)],
                    check=True,
                    capture_output=True,
                    timeout=5
                )
                passed += 1
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                failed += 1
                failed_files.append(lua_file)
                print(f"  ❌ {lua_file.name}")
                if self.delete_invalid:
                    try:
                        lua_file.unlink()
                        print(f"     🗑️  Deleted {lua_file.name}")
                    except Exception as del_err:
                        print(f"     ⚠️  Could not delete: {del_err}")
        
        print(f"  ✅ Passed: {passed}/{passed + failed}")
        self.failed_files.extend(failed_files)
        return passed, failed, failed_files
    
    def validate_elixir(self):
        """Validate Elixir samples."""
        print("💧 Validating Elixir samples...")
        elixir_path = self.code_samples_path / "elixir"
        passed = 0
        failed = 0
        failed_files = []
        
        # Check if elixir is available
        try:
            subprocess.run(['elixir', '--version'], check=True, capture_output=True, timeout=2)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            print(f"  ⚠️  Elixir not available, skipping")
            return 0, 0, []
        
        for ex_file in elixir_path.glob("*.ex"):
            try:
                # Use elixir to check syntax
                subprocess.run(
                    ['elixir', '-c', str(ex_file)],
                    check=True,
                    capture_output=True,
                    timeout=10
                )
                passed += 1
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                failed += 1
                failed_files.append(ex_file)
                print(f"  ❌ {ex_file.name}")
                if self.delete_invalid:
                    try:
                        ex_file.unlink()
                        print(f"     🗑️  Deleted {ex_file.name}")
                    except Exception as del_err:
                        print(f"     ⚠️  Could not delete: {del_err}")
        
        print(f"  ✅ Passed: {passed}/{passed + failed}")
        self.failed_files.extend(failed_files)
        return passed, failed, failed_files
    
    def validate_solidity(self):
        """Validate Solidity samples."""
        print("⛓️  Validating Solidity samples...")
        solidity_path = self.code_samples_path / "solidity"
        passed = 0
        failed = 0
        failed_files = []
        
        # Check if solc is available
        try:
            subprocess.run(['solc', '--version'], check=True, capture_output=True, timeout=2)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            print(f"  ⚠️  Solidity compiler not available, skipping")
            return 0, 0, []
        
        for sol_file in solidity_path.glob("*.sol"):
            try:
                # Compile to check syntax
                subprocess.run(
                    ['solc', '--optimize', str(sol_file)],
                    check=True,
                    capture_output=True,
                    timeout=10
                )
                passed += 1
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                failed += 1
                failed_files.append(sol_file)
                print(f"  ❌ {sol_file.name}")
                if self.delete_invalid:
                    try:
                        sol_file.unlink()
                        print(f"     🗑️  Deleted {sol_file.name}")
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
    results['cpp'] = validator.validate_cpp()
    results['csharp'] = validator.validate_csharp()
    results['kotlin'] = validator.validate_kotlin()
    results['swift'] = validator.validate_swift()
    results['haskell'] = validator.validate_haskell()
    results['scala'] = validator.validate_scala()
    results['dart'] = validator.validate_dart()
    results['r'] = validator.validate_r()
    results['lua'] = validator.validate_lua()
    results['elixir'] = validator.validate_elixir()
    results['solidity'] = validator.validate_solidity()
    
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
