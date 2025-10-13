#!/usr/bin/env python3
"""
AI/ML/LLM Training Dataset Generator
Creates specialized datasets for training AI coding agents.
"""

import json
import random
import uuid
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


class AITrainingDatasetGenerator:
    """Generate specialized datasets for AI/ML/LLM training."""
    
    def __init__(self, output_dir: str = "datasets/processed"):
        """Initialize generator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.languages = [
            "python", "javascript", "java", "cpp", "rust", "go", 
            "typescript", "ruby", "php", "swift", "kotlin", "csharp"
        ]
    
    def generate_code_completion_dataset(self) -> Dict[str, Any]:
        """Generate code completion training data."""
        print("📝 Generating code completion dataset...")
        
        samples = []
        
        # Function definitions
        functions = [
            {
                "prompt": "def binary_search(arr, target):",
                "completion": "\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
                "language": "python",
                "category": "algorithms",
                "complexity": "O(log n)"
            },
            {
                "prompt": "function quickSort(arr) {",
                "completion": "\n  if (arr.length <= 1) return arr;\n  const pivot = arr[Math.floor(arr.length / 2)];\n  const left = arr.filter(x => x < pivot);\n  const middle = arr.filter(x => x === pivot);\n  const right = arr.filter(x => x > pivot);\n  return [...quickSort(left), ...middle, ...quickSort(right)];\n}",
                "language": "javascript",
                "category": "algorithms",
                "complexity": "O(n log n)"
            },
            {
                "prompt": "public class LinkedList<T> {",
                "completion": "\n    private Node<T> head;\n    private int size;\n    \n    private class Node<T> {\n        T data;\n        Node<T> next;\n        \n        Node(T data) {\n            this.data = data;\n            this.next = null;\n        }\n    }\n    \n    public void add(T data) {\n        Node<T> newNode = new Node<>(data);\n        if (head == null) {\n            head = newNode;\n        } else {\n            Node<T> current = head;\n            while (current.next != null) {\n                current = current.next;\n            }\n            current.next = newNode;\n        }\n        size++;\n    }\n}",
                "language": "java",
                "category": "data_structures",
                "complexity": "O(1) insert at head"
            },
            {
                "prompt": "async function fetchUserData(userId) {",
                "completion": "\n  try {\n    const response = await fetch(`/api/users/${userId}`);\n    if (!response.ok) {\n      throw new Error(`HTTP error! status: ${response.status}`);\n    }\n    const data = await response.json();\n    return data;\n  } catch (error) {\n    console.error('Error fetching user data:', error);\n    throw error;\n  }\n}",
                "language": "javascript",
                "category": "async_programming",
                "complexity": "O(1)"
            },
            {
                "prompt": "def merge_sort(arr):",
                "completion": "\n    if len(arr) <= 1:\n        return arr\n    \n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    \n    return merge(left, right)\n\ndef merge(left, right):\n    result = []\n    i = j = 0\n    \n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            result.append(left[i])\n            i += 1\n        else:\n            result.append(right[j])\n            j += 1\n    \n    result.extend(left[i:])\n    result.extend(right[j:])\n    return result",
                "language": "python",
                "category": "algorithms",
                "complexity": "O(n log n)"
            }
        ]
        
        for i, func in enumerate(functions):
            samples.append({
                "id": f"code_completion_{i+1:04d}",
                "prompt": func["prompt"],
                "completion": func["completion"],
                "full_code": func["prompt"] + func["completion"],
                "language": func["language"],
                "category": func["category"],
                "complexity": func["complexity"],
                "training_objective": "code_completion"
            })
        
        # Add more variations
        for lang in ["python", "javascript", "java", "go", "rust"]:
            for i in range(20):
                samples.append({
                    "id": f"code_completion_{len(samples)+1:04d}",
                    "prompt": f"# {lang} function implementation\n",
                    "completion": f"def example_function_{i}():\n    # Implementation here\n    pass",
                    "language": lang,
                    "category": "general",
                    "training_objective": "code_completion"
                })
        
        return {
            "metadata": {
                "name": "code_completion_training_dataset",
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "total_samples": len(samples),
                "languages": list(set(s["language"] for s in samples)),
                "purpose": "AI code completion training"
            },
            "samples": samples
        }
    
    def generate_bug_detection_dataset(self) -> Dict[str, Any]:
        """Generate bug detection and fixing dataset."""
        print("🐛 Generating bug detection dataset...")
        
        samples = []
        
        bug_patterns = [
            {
                "buggy_code": "def divide(a, b):\n    return a / b",
                "fixed_code": "def divide(a, b):\n    if b == 0:\n        raise ValueError('Cannot divide by zero')\n    return a / b",
                "bug_type": "missing_validation",
                "language": "python",
                "severity": "high",
                "explanation": "Missing zero division check"
            },
            {
                "buggy_code": "for (let i = 0; i <= arr.length; i++) {\n  console.log(arr[i]);\n}",
                "fixed_code": "for (let i = 0; i < arr.length; i++) {\n  console.log(arr[i]);\n}",
                "bug_type": "off_by_one",
                "language": "javascript",
                "severity": "medium",
                "explanation": "Array index out of bounds due to <= instead of <"
            },
            {
                "buggy_code": "String s = null;\nint length = s.length();",
                "fixed_code": "String s = null;\nif (s != null) {\n    int length = s.length();\n} else {\n    // Handle null case\n}",
                "bug_type": "null_pointer",
                "language": "java",
                "severity": "high",
                "explanation": "NullPointerException when accessing null string"
            },
            {
                "buggy_code": "items = []\nfor i in range(10):\n    items.append(lambda: i)",
                "fixed_code": "items = []\nfor i in range(10):\n    items.append(lambda i=i: i)",
                "bug_type": "closure_binding",
                "language": "python",
                "severity": "medium",
                "explanation": "Lambda captures variable reference, not value"
            },
            {
                "buggy_code": "async function getData() {\n  fetch('/api/data')\n    .then(res => res.json())\n    .then(data => console.log(data));\n  console.log('Done');\n}",
                "fixed_code": "async function getData() {\n  const res = await fetch('/api/data');\n  const data = await res.json();\n  console.log(data);\n  console.log('Done');\n}",
                "bug_type": "async_handling",
                "language": "javascript",
                "severity": "medium",
                "explanation": "Not properly awaiting async operations"
            }
        ]
        
        for i, bug in enumerate(bug_patterns):
            samples.append({
                "id": f"bug_detection_{i+1:04d}",
                "buggy_code": bug["buggy_code"],
                "fixed_code": bug["fixed_code"],
                "bug_type": bug["bug_type"],
                "language": bug["language"],
                "severity": bug["severity"],
                "explanation": bug["explanation"],
                "training_objective": "bug_detection_and_fixing"
            })
        
        # Add more synthetic bugs
        common_bugs = [
            "IndexError", "KeyError", "TypeError", "ValueError",
            "NullPointerException", "ConcurrentModificationException",
            "MemoryLeak", "RaceCondition", "DeadLock"
        ]
        
        for bug_type in common_bugs:
            for i in range(10):
                samples.append({
                    "id": f"bug_detection_{len(samples)+1:04d}",
                    "bug_type": bug_type,
                    "language": random.choice(self.languages),
                    "severity": random.choice(["low", "medium", "high", "critical"]),
                    "training_objective": "bug_detection_and_fixing"
                })
        
        return {
            "metadata": {
                "name": "bug_detection_training_dataset",
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "total_samples": len(samples),
                "bug_types": list(set(s["bug_type"] for s in samples)),
                "purpose": "AI bug detection and fixing training"
            },
            "samples": samples
        }
    
    def generate_code_translation_dataset(self) -> Dict[str, Any]:
        """Generate cross-language code translation dataset."""
        print("🔄 Generating code translation dataset...")
        
        samples = []
        
        translations = [
            {
                "source_lang": "python",
                "target_lang": "javascript",
                "source_code": "def sum_array(arr):\n    return sum(arr)",
                "target_code": "function sumArray(arr) {\n  return arr.reduce((a, b) => a + b, 0);\n}",
                "concept": "array_sum"
            },
            {
                "source_lang": "javascript",
                "target_lang": "python",
                "source_code": "const double = x => x * 2;",
                "target_code": "double = lambda x: x * 2",
                "concept": "lambda_function"
            },
            {
                "source_lang": "python",
                "target_lang": "java",
                "source_code": "class Person:\n    def __init__(self, name):\n        self.name = name",
                "target_code": "public class Person {\n    private String name;\n    \n    public Person(String name) {\n        this.name = name;\n    }\n}",
                "concept": "class_definition"
            },
            {
                "source_lang": "java",
                "target_lang": "go",
                "source_code": "List<String> names = new ArrayList<>();\nnames.add(\"Alice\");",
                "target_code": "names := make([]string, 0)\nnames = append(names, \"Alice\")",
                "concept": "list_operations"
            },
            {
                "source_lang": "rust",
                "target_lang": "cpp",
                "source_code": "let mut vec = Vec::new();\nvec.push(42);",
                "target_code": "std::vector<int> vec;\nvec.push_back(42);",
                "concept": "vector_operations"
            }
        ]
        
        for i, trans in enumerate(translations):
            samples.append({
                "id": f"translation_{i+1:04d}",
                "source_language": trans["source_lang"],
                "target_language": trans["target_lang"],
                "source_code": trans["source_code"],
                "target_code": trans["target_code"],
                "concept": trans["concept"],
                "training_objective": "code_translation"
            })
        
        # Add more synthetic translations
        for i in range(50):
            source_lang = random.choice(self.languages)
            target_lang = random.choice([l for l in self.languages if l != source_lang])
            
            samples.append({
                "id": f"translation_{len(samples)+1:04d}",
                "source_language": source_lang,
                "target_language": target_lang,
                "concept": random.choice([
                    "loops", "conditionals", "functions", "classes",
                    "error_handling", "async_operations", "data_structures"
                ]),
                "training_objective": "code_translation"
            })
        
        return {
            "metadata": {
                "name": "code_translation_training_dataset",
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "total_samples": len(samples),
                "language_pairs": len(samples),
                "purpose": "AI code translation training"
            },
            "samples": samples
        }
    
    def generate_performance_optimization_dataset(self) -> Dict[str, Any]:
        """Generate performance optimization dataset."""
        print("⚡ Generating performance optimization dataset...")
        
        samples = []
        
        optimizations = [
            {
                "slow_code": "result = []\nfor i in range(len(arr)):\n    result.append(arr[i] * 2)",
                "fast_code": "result = [x * 2 for x in arr]",
                "language": "python",
                "optimization_type": "list_comprehension",
                "speedup": "2-3x faster",
                "explanation": "List comprehension is more efficient than loop with append"
            },
            {
                "slow_code": "let result = [];\nfor (let i = 0; i < arr.length; i++) {\n  if (arr[i] > 10) {\n    result.push(arr[i]);\n  }\n}",
                "fast_code": "const result = arr.filter(x => x > 10);",
                "language": "javascript",
                "optimization_type": "built_in_methods",
                "speedup": "1.5-2x faster",
                "explanation": "Built-in filter method is optimized"
            },
            {
                "slow_code": "for (int i = 0; i < n; i++) {\n    for (int j = 0; j < n; j++) {\n        sum += arr[i] * arr[j];\n    }\n}",
                "fast_code": "int sum_arr = 0;\nfor (int i = 0; i < n; i++) {\n    sum_arr += arr[i];\n}\nsum = sum_arr * sum_arr;",
                "language": "java",
                "optimization_type": "algorithm_improvement",
                "speedup": "O(n²) to O(n)",
                "explanation": "Reduce time complexity by mathematical transformation"
            },
            {
                "slow_code": "def find_duplicates(arr):\n    duplicates = []\n    for i in range(len(arr)):\n        for j in range(i+1, len(arr)):\n            if arr[i] == arr[j] and arr[i] not in duplicates:\n                duplicates.append(arr[i])\n    return duplicates",
                "fast_code": "def find_duplicates(arr):\n    seen = set()\n    duplicates = set()\n    for item in arr:\n        if item in seen:\n            duplicates.add(item)\n        seen.add(item)\n    return list(duplicates)",
                "language": "python",
                "optimization_type": "data_structure_choice",
                "speedup": "O(n²) to O(n)",
                "explanation": "Use hash set for O(1) lookups instead of list"
            },
            {
                "slow_code": "String result = \"\";\nfor (String s : strings) {\n    result += s;\n}",
                "fast_code": "StringBuilder result = new StringBuilder();\nfor (String s : strings) {\n    result.append(s);\n}\nreturn result.toString();",
                "language": "java",
                "optimization_type": "string_concatenation",
                "speedup": "10-100x faster for large inputs",
                "explanation": "StringBuilder avoids creating new string objects"
            }
        ]
        
        for i, opt in enumerate(optimizations):
            samples.append({
                "id": f"optimization_{i+1:04d}",
                "slow_code": opt["slow_code"],
                "optimized_code": opt["fast_code"],
                "language": opt["language"],
                "optimization_type": opt["optimization_type"],
                "speedup": opt["speedup"],
                "explanation": opt["explanation"],
                "training_objective": "performance_optimization"
            })
        
        # Add more optimization patterns
        optimization_types = [
            "caching", "memoization", "lazy_evaluation", "vectorization",
            "parallel_processing", "memory_pooling", "loop_unrolling"
        ]
        
        for opt_type in optimization_types:
            for i in range(8):
                samples.append({
                    "id": f"optimization_{len(samples)+1:04d}",
                    "optimization_type": opt_type,
                    "language": random.choice(self.languages),
                    "training_objective": "performance_optimization"
                })
        
        return {
            "metadata": {
                "name": "performance_optimization_training_dataset",
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "total_samples": len(samples),
                "optimization_types": list(set(s["optimization_type"] for s in samples)),
                "purpose": "AI performance optimization training"
            },
            "samples": samples
        }
    
    def generate_refactoring_patterns_dataset(self) -> Dict[str, Any]:
        """Generate code refactoring patterns dataset."""
        print("🔧 Generating refactoring patterns dataset...")
        
        samples = []
        
        refactorings = [
            {
                "before": "def process_user(name, email, age, address, phone):\n    # Too many parameters\n    pass",
                "after": "class UserInfo:\n    def __init__(self, name, email, age, address, phone):\n        self.name = name\n        self.email = email\n        self.age = age\n        self.address = address\n        self.phone = phone\n\ndef process_user(user_info: UserInfo):\n    pass",
                "pattern": "introduce_parameter_object",
                "language": "python",
                "smell": "long_parameter_list"
            },
            {
                "before": "if (user.type === 'admin') {\n    // admin logic\n} else if (user.type === 'moderator') {\n    // moderator logic\n} else if (user.type === 'user') {\n    // user logic\n}",
                "after": "const handlers = {\n    admin: () => { /* admin logic */ },\n    moderator: () => { /* moderator logic */ },\n    user: () => { /* user logic */ }\n};\n\nhandlers[user.type]();",
                "pattern": "replace_conditional_with_polymorphism",
                "language": "javascript",
                "smell": "switch_statements"
            },
            {
                "before": "public void calculateTotal() {\n    double total = 0;\n    for (Item item : items) {\n        total += item.price * item.quantity;\n        if (item.discount > 0) {\n            total -= item.discount;\n        }\n    }\n    this.total = total;\n}",
                "after": "public void calculateTotal() {\n    this.total = items.stream()\n        .mapToDouble(item -> item.getDiscountedPrice())\n        .sum();\n}\n\nclass Item {\n    double getDiscountedPrice() {\n        return price * quantity - discount;\n    }\n}",
                "pattern": "extract_method",
                "language": "java",
                "smell": "long_method"
            },
            {
                "before": "def get_user_data(id):\n    user = db.query('SELECT * FROM users WHERE id = ' + str(id))\n    return user",
                "after": "def get_user_data(id):\n    query = 'SELECT * FROM users WHERE id = ?'\n    user = db.query(query, (id,))\n    return user",
                "pattern": "security_fix",
                "language": "python",
                "smell": "sql_injection_vulnerability"
            },
            {
                "before": "class DataProcessor {\n    processData() { /* ... */ }\n    saveToDatabase() { /* ... */ }\n    sendEmail() { /* ... */ }\n    generateReport() { /* ... */ }\n}",
                "after": "class DataProcessor {\n    processData() { /* ... */ }\n}\n\nclass DatabaseService {\n    save(data) { /* ... */ }\n}\n\nclass EmailService {\n    send(message) { /* ... */ }\n}\n\nclass ReportGenerator {\n    generate(data) { /* ... */ }\n}",
                "pattern": "split_class",
                "language": "javascript",
                "smell": "large_class"
            }
        ]
        
        for i, ref in enumerate(refactorings):
            samples.append({
                "id": f"refactoring_{i+1:04d}",
                "before_code": ref["before"],
                "after_code": ref["after"],
                "refactoring_pattern": ref["pattern"],
                "language": ref["language"],
                "code_smell": ref["smell"],
                "training_objective": "code_refactoring"
            })
        
        # Add more refactoring patterns
        patterns = [
            "extract_method", "inline_method", "extract_variable",
            "inline_variable", "rename_method", "move_method",
            "pull_up_method", "push_down_method", "extract_interface",
            "introduce_null_object", "remove_dead_code"
        ]
        
        for pattern in patterns:
            for i in range(7):
                samples.append({
                    "id": f"refactoring_{len(samples)+1:04d}",
                    "refactoring_pattern": pattern,
                    "language": random.choice(self.languages),
                    "training_objective": "code_refactoring"
                })
        
        return {
            "metadata": {
                "name": "refactoring_patterns_training_dataset",
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "total_samples": len(samples),
                "patterns": list(set(s["refactoring_pattern"] for s in samples)),
                "purpose": "AI code refactoring training"
            },
            "samples": samples
        }
    
    def generate_security_patterns_dataset(self) -> Dict[str, Any]:
        """Generate security vulnerability detection dataset."""
        print("🔒 Generating security patterns dataset...")
        
        samples = []
        
        vulnerabilities = [
            {
                "vulnerable_code": "query = f\"SELECT * FROM users WHERE username = '{username}'\"",
                "secure_code": "query = \"SELECT * FROM users WHERE username = ?\"\ncursor.execute(query, (username,))",
                "vulnerability": "sql_injection",
                "language": "python",
                "severity": "critical",
                "cwe": "CWE-89"
            },
            {
                "vulnerable_code": "eval(user_input)",
                "secure_code": "# Avoid eval, use safe alternatives like json.loads() or ast.literal_eval()",
                "vulnerability": "code_injection",
                "language": "python",
                "severity": "critical",
                "cwe": "CWE-95"
            },
            {
                "vulnerable_code": "element.innerHTML = userInput;",
                "secure_code": "element.textContent = userInput;\n// or sanitize: DOMPurify.sanitize(userInput);",
                "vulnerability": "xss",
                "language": "javascript",
                "severity": "high",
                "cwe": "CWE-79"
            },
            {
                "vulnerable_code": "Runtime.getRuntime().exec(\"ls \" + userInput);",
                "secure_code": "ProcessBuilder pb = new ProcessBuilder(\"ls\", userInput);\npb.start();",
                "vulnerability": "command_injection",
                "language": "java",
                "severity": "critical",
                "cwe": "CWE-78"
            },
            {
                "vulnerable_code": "password = \"admin123\"  # Hardcoded password",
                "secure_code": "password = os.environ.get('DB_PASSWORD')\nif not password:\n    raise ValueError('DB_PASSWORD not set')",
                "vulnerability": "hardcoded_credentials",
                "language": "python",
                "severity": "high",
                "cwe": "CWE-798"
            }
        ]
        
        for i, vuln in enumerate(vulnerabilities):
            samples.append({
                "id": f"security_{i+1:04d}",
                "vulnerable_code": vuln["vulnerable_code"],
                "secure_code": vuln["secure_code"],
                "vulnerability_type": vuln["vulnerability"],
                "language": vuln["language"],
                "severity": vuln["severity"],
                "cwe_id": vuln["cwe"],
                "training_objective": "security_vulnerability_detection"
            })
        
        # Add more vulnerability types
        vuln_types = [
            "buffer_overflow", "path_traversal", "xxe", "ssrf",
            "insecure_deserialization", "broken_authentication",
            "sensitive_data_exposure", "broken_access_control",
            "security_misconfiguration", "insufficient_logging"
        ]
        
        for vuln_type in vuln_types:
            for i in range(8):
                samples.append({
                    "id": f"security_{len(samples)+1:04d}",
                    "vulnerability_type": vuln_type,
                    "language": random.choice(self.languages),
                    "severity": random.choice(["low", "medium", "high", "critical"]),
                    "training_objective": "security_vulnerability_detection"
                })
        
        return {
            "metadata": {
                "name": "security_patterns_training_dataset",
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "total_samples": len(samples),
                "vulnerability_types": list(set(s["vulnerability_type"] for s in samples)),
                "purpose": "AI security vulnerability detection training"
            },
            "samples": samples
        }
    
    def generate_all(self):
        """Generate all AI training datasets."""
        print("=" * 70)
        print("AI/ML/LLM Training Dataset Generator")
        print("=" * 70)
        print()
        
        datasets = [
            ("ai_code_completion_training.json", self.generate_code_completion_dataset()),
            ("ai_bug_detection_training.json", self.generate_bug_detection_dataset()),
            ("ai_code_translation_training.json", self.generate_code_translation_dataset()),
            ("ai_performance_optimization_training.json", self.generate_performance_optimization_dataset()),
            ("ai_refactoring_patterns_training.json", self.generate_refactoring_patterns_dataset()),
            ("ai_security_patterns_training.json", self.generate_security_patterns_dataset()),
        ]
        
        created_files = []
        
        for filename, dataset in datasets:
            filepath = self.output_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Created: {filename} ({len(dataset['samples'])} samples)")
            created_files.append(filename)
        
        print()
        print("=" * 70)
        print(f"✅ Successfully created {len(created_files)} AI training datasets")
        print(f"📁 Location: {self.output_dir}")
        print("=" * 70)
        
        return created_files


def main():
    """Main entry point."""
    generator = AITrainingDatasetGenerator()
    generator.generate_all()


if __name__ == "__main__":
    main()
