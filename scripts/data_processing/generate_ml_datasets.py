#!/usr/bin/env python3
"""
ML Training Data Generator for Code Samples
Processes code samples into various ML training formats.
"""

import os
import json
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CodeSample:
    """Represents a code sample with metadata."""
    language: str
    filename: str
    content: str
    file_path: str
    size_bytes: int
    line_count: int
    function_count: int
    class_count: int
    comment_lines: int
    complexity_score: float
    algorithms: List[str]
    data_structures: List[str]
    design_patterns: List[str]
    file_hash: str

@dataclass
class MLDataset:
    """Represents a processed ML dataset."""
    dataset_type: str
    language: str
    samples: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class CodeAnalyzer:
    """Analyzes code samples to extract features and metadata."""
    
    # Language-specific patterns for analysis
    LANGUAGE_PATTERNS = {
        'python': {
            'function': r'def\s+(\w+)\s*\(',
            'class': r'class\s+(\w+)[\s\(:]+',
            'comment': r'#.*|"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'',
            'import': r'import\s+(\w+)|from\s+(\w+)\s+import',
        },
        'javascript': {
            'function': r'function\s+(\w+)\s*\(|(\w+)\s*=\s*function|\w+\s*=>\s*{|(\w+)\s*\([^)]*\)\s*=>',
            'class': r'class\s+(\w+)[\s{]+',
            'comment': r'//.*|/\*[\s\S]*?\*/',
            'import': r'import\s+.*from\s+[\'"]([^\'"]+)[\'"]|require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)',
        },
        'typescript': {
            'function': r'function\s+(\w+)\s*\(|(\w+)\s*=\s*function|\w+\s*=>\s*{|(\w+)\s*\([^)]*\)\s*=>',
            'class': r'class\s+(\w+)[\s{<]+',
            'comment': r'//.*|/\*[\s\S]*?\*/',
            'import': r'import\s+.*from\s+[\'"]([^\'"]+)[\'"]',
        },
        'rust': {
            'function': r'fn\s+(\w+)\s*\(',
            'class': r'struct\s+(\w+)|enum\s+(\w+)',
            'comment': r'//.*|/\*[\s\S]*?\*/',
            'import': r'use\s+([^;]+);',
        },
        'go': {
            'function': r'func\s+(\w+)\s*\(',
            'class': r'type\s+(\w+)\s+struct',
            'comment': r'//.*|/\*[\s\S]*?\*/',
            'import': r'import\s+[\'"]([^\'"]+)[\'"]',
        },
        'csharp': {
            'function': r'(?:public|private|protected|internal)?\s*(?:static)?\s*\w+\s+(\w+)\s*\(',
            'class': r'(?:public|private|protected|internal)?\s*class\s+(\w+)',
            'comment': r'//.*|/\*[\s\S]*?\*/',
            'import': r'using\s+([^;]+);',
        },
        'php': {
            'function': r'function\s+(\w+)\s*\(',
            'class': r'class\s+(\w+)[\s{]+',
            'comment': r'//.*|/\*[\s\S]*?\*/|#.*',
            'import': r'require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)|include\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)',
        },
        'ruby': {
            'function': r'def\s+(\w+)[\s\(]+',
            'class': r'class\s+(\w+)[\s<]+',
            'comment': r'#.*',
            'import': r'require\s+[\'"]([^\'"]+)[\'"]',
        },
        'swift': {
            'function': r'func\s+(\w+)\s*\(',
            'class': r'class\s+(\w+)[\s{:<]+|struct\s+(\w+)[\s{:<]+',
            'comment': r'//.*|/\*[\s\S]*?\*/',
            'import': r'import\s+(\w+)',
        },
        'java': {
            'function': r'(?:public|private|protected)?\s*(?:static)?\s*\w+\s+(\w+)\s*\(',
            'class': r'(?:public|private|protected)?\s*class\s+(\w+)',
            'comment': r'//.*|/\*[\s\S]*?\*/',
            'import': r'import\s+([^;]+);',
        },
        'cpp': {
            'function': r'\w+\s+(\w+)\s*\(',
            'class': r'class\s+(\w+)[\s{:]+',
            'comment': r'//.*|/\*[\s\S]*?\*/',
            'import': r'#include\s*[<"]([^>"]+)[>"]',
        }
    }
    
    # Algorithm keywords for detection
    ALGORITHM_KEYWORDS = {
        'sorting': ['sort', 'bubble', 'quick', 'merge', 'heap', 'insertion', 'selection'],
        'searching': ['search', 'binary', 'linear', 'find', 'lookup'],
        'graph': ['dfs', 'bfs', 'dijkstra', 'graph', 'node', 'edge', 'vertex'],
        'dynamic_programming': ['memo', 'dp', 'fibonacci', 'knapsack', 'lcs'],
        'recursion': ['recursive', 'recursion', 'factorial'],
        'hashing': ['hash', 'dictionary', 'map'],
        'string': ['string', 'substring', 'palindrome', 'anagram'],
        'math': ['gcd', 'lcm', 'prime', 'factorial', 'fibonacci'],
        'concurrency': ['thread', 'async', 'await', 'parallel', 'concurrent', 'mutex', 'lock']
    }
    
    # Data structure keywords
    DATA_STRUCTURE_KEYWORDS = {
        'array': ['array', 'list', 'vector'],
        'stack': ['stack', 'push', 'pop'],
        'queue': ['queue', 'enqueue', 'dequeue'],
        'tree': ['tree', 'binary', 'node', 'leaf', 'root'],
        'graph': ['graph', 'vertex', 'edge', 'adjacency'],
        'hash_table': ['hash', 'dictionary', 'map', 'bucket'],
        'linked_list': ['linked', 'next', 'node'],
        'heap': ['heap', 'priority', 'min', 'max']
    }
    
    # Design pattern keywords
    DESIGN_PATTERN_KEYWORDS = {
        'singleton': ['singleton', 'instance'],
        'factory': ['factory', 'create'],
        'observer': ['observer', 'notify', 'update'],
        'strategy': ['strategy', 'algorithm'],
        'decorator': ['decorator', 'wrapper'],
        'builder': ['builder', 'build'],
        'adapter': ['adapter', 'interface'],
        'facade': ['facade', 'wrapper']
    }

        # Algorithm categories
        self.algorithm_categories = [
            "sorting",
            "searching",
            "graph",
            "dynamic_programming",
            "greedy",
            "divide_conquer",
            "string",
            "tree",
            "array",
            "math",
        ]

        # Code pattern categories
        self.pattern_categories = [
            "creational",
            "structural",
            "behavioral",
            "functional",
            "object_oriented",
            "concurrent",
            "error_handling",
            "testing",
        ]

    def generate_code_patterns_dataset(self) -> None:
        """Generate dataset of coding patterns and idioms."""
        print("📝 Generating code patterns dataset...")

        patterns_data = []

        # Python patterns
        python_patterns = [
            {
                "id": str(uuid.uuid4()),
                "language": "python",
                "pattern_type": "functional",
                "pattern_name": "list_comprehension",
                "description": "Concise way to create lists using functional programming",
                "code_before": """# Traditional approach
result = []
for i in range(10):
    if i % 2 == 0:
        result.append(i * 2)""",
                "code_after": """# List comprehension approach
result = [i * 2 for i in range(10) if i % 2 == 0]""",
                "complexity": "beginner",
                "tags": ["functional", "pythonic", "list", "comprehension"],
                "performance_impact": "positive",
                "readability_score": 9,
            },
            {
                "id": str(uuid.uuid4()),
                "language": "python",
                "pattern_type": "error_handling",
                "pattern_name": "eafp_pattern",
                "description": "Easier to Ask for Forgiveness than Permission",
                "code_before": """# Look Before You Leap (LBYL)
if key in dictionary:
    value = dictionary[key]
else:
    value = default_value""",
                "code_after": """# Easier to Ask for Forgiveness than Permission (EAFP)
try:
    value = dictionary[key]
except KeyError:
    value = default_value""",
                "complexity": "intermediate",
                "tags": ["error_handling", "pythonic", "exception"],
                "performance_impact": "positive",
                "readability_score": 8,
            },
            {
                "id": str(uuid.uuid4()),
                "language": "python",
                "pattern_type": "object_oriented",
                "pattern_name": "context_manager",
                "description": "Resource management using context managers",
                "code_before": """# Manual resource management
file = open('data.txt', 'r')
try:
    content = file.read()
    process(content)
finally:
    file.close()""",
                "code_after": """# Context manager approach
with open('data.txt', 'r') as file:
    content = file.read()
    process(content)
# File automatically closed""",
                "complexity": "intermediate",
                "tags": ["context_manager", "resource_management", "pythonic"],
                "performance_impact": "neutral",
                "readability_score": 9,
            },
            {
                "id": str(uuid.uuid4()),
                "language": "python",
                "pattern_type": "functional",
                "pattern_name": "generator_expression",
                "description": "Memory-efficient iteration using generators",
                "code_before": """# List approach (memory intensive)
def process_large_data():
    return [expensive_operation(x) for x in range(1000000)]

for item in process_large_data():
    consume(item)""",
                "code_after": """# Generator approach (memory efficient)
def process_large_data():
    for x in range(1000000):
        yield expensive_operation(x)

for item in process_large_data():
    consume(item)""",
                "complexity": "intermediate",
                "tags": ["generator", "memory_efficiency", "functional"],
                "performance_impact": "positive",
                "readability_score": 8,
            },
        ]

        # JavaScript patterns
        javascript_patterns = [
            {
                "id": str(uuid.uuid4()),
                "language": "javascript",
                "pattern_type": "functional",
                "pattern_name": "array_destructuring",
                "description": "Extract array elements into variables",
                "code_before": """// Traditional approach
const arr = [1, 2, 3];
const first = arr[0];
const second = arr[1];
const third = arr[2];""",
                "code_after": """// Destructuring approach
const arr = [1, 2, 3];
const [first, second, third] = arr;""",
                "complexity": "beginner",
                "tags": ["destructuring", "es6", "array"],
                "performance_impact": "neutral",
                "readability_score": 9,
            },
            {
                "id": str(uuid.uuid4()),
                "language": "javascript",
                "pattern_type": "functional",
                "pattern_name": "promise_chaining",
                "description": "Handle asynchronous operations with promises",
                "code_before": """// Callback hell
getData(function(a) {
    getMoreData(a, function(b) {
        getEvenMoreData(b, function(c) {
            // Finally do something with c
            console.log(c);
        });
    });
});""",
                "code_after": """// Promise chaining
getData()
    .then(a => getMoreData(a))
    .then(b => getEvenMoreData(b))
    .then(c => console.log(c))
    .catch(error => console.error(error));""",
                "complexity": "intermediate",
                "tags": ["promise", "async", "chaining"],
                "performance_impact": "positive",
                "readability_score": 9,
            },
            {
                "id": str(uuid.uuid4()),
                "language": "javascript",
                "pattern_type": "functional",
                "pattern_name": "async_await",
                "description": "Modern asynchronous code using async/await",
                "code_before": """// Promise chaining
function processData() {
    return getData()
        .then(a => getMoreData(a))
        .then(b => getEvenMoreData(b))
        .then(c => {
            console.log(c);
            return c;
        });
}""",
                "code_after": """// Async/await approach
async function processData() {
    try {
        const a = await getData();
        const b = await getMoreData(a);
        const c = await getEvenMoreData(b);
        console.log(c);
        return c;
    } catch (error) {
        console.error(error);
        throw error;
    }
}""",
                "complexity": "intermediate",
                "tags": ["async", "await", "modern_js"],
                "performance_impact": "neutral",
                "readability_score": 10,
            },
        ]

        # Java patterns
        java_patterns = [
            {
                "id": str(uuid.uuid4()),
                "language": "java",
                "pattern_type": "object_oriented",
                "pattern_name": "builder_pattern",
                "description": "Construct complex objects step by step",
                "code_before": """// Constructor with many parameters
public class Person {
    private String name;
    private int age;
    private String email;
    private String phone;
    private String address;
    
    public Person(String name, int age, String email, String phone, String address) {
        this.name = name;
        this.age = age;
        this.email = email;
        this.phone = phone;
        this.address = address;
    }
}

// Usage
Person person = new Person("John", 25, "john@email.com", "123-456-7890", "123 Main St");""",
                "code_after": """// Builder pattern
public class Person {
    private String name;
    private int age;
    private String email;
    private String phone;
    private String address;
    
    private Person(Builder builder) {
        this.name = builder.name;
        this.age = builder.age;
        this.email = builder.email;
        this.phone = builder.phone;
        this.address = builder.address;
    }
    
    public static class Builder {
        private String name;
        private int age;
        private String email;
        private String phone;
        private String address;
        
        public Builder setName(String name) { this.name = name; return this; }
        public Builder setAge(int age) { this.age = age; return this; }
        public Builder setEmail(String email) { this.email = email; return this; }
        public Builder setPhone(String phone) { this.phone = phone; return this; }
        public Builder setAddress(String address) { this.address = address; return this; }
        
        public Person build() { return new Person(this); }
    }
}

// Usage
Person person = new Person.Builder()
    .setName("John")
    .setAge(25)
    .setEmail("john@email.com")
    .setPhone("123-456-7890")
    .setAddress("123 Main St")
    .build();""",
                "complexity": "intermediate",
                "tags": ["builder", "design_pattern", "object_construction"],
                "performance_impact": "neutral",
                "readability_score": 9,
            },
            {
                "id": str(uuid.uuid4()),
                "language": "java",
                "pattern_type": "functional",
                "pattern_name": "stream_api",
                "description": "Functional programming with Java Streams",
                "code_before": """// Traditional imperative approach
List<String> names = Arrays.asList("Alice", "Bob", "Charlie", "David");
List<String> result = new ArrayList<>();
for (String name : names) {
    if (name.length() > 3) {
        result.add(name.toUpperCase());
    }
}
Collections.sort(result);""",
                "code_after": """// Stream API approach
List<String> names = Arrays.asList("Alice", "Bob", "Charlie", "David");
List<String> result = names.stream()
    .filter(name -> name.length() > 3)
    .map(String::toUpperCase)
    .sorted()
    .collect(Collectors.toList());""",
                "complexity": "intermediate",
                "tags": ["stream", "functional", "java8"],
                "performance_impact": "positive",
                "readability_score": 9,
            },
        ]

        # C++ patterns
        cpp_patterns = [
            {
                "id": str(uuid.uuid4()),
                "language": "cpp",
                "pattern_type": "object_oriented",
                "pattern_name": "raii_pattern",
                "description": "Resource Acquisition Is Initialization",
                "code_before": """// Manual resource management
void processFile(const std::string& filename) {
    FILE* file = fopen(filename.c_str(), "r");
    if (!file) {
        throw std::runtime_error("Cannot open file");
    }
    
    try {
        // Process file
        processData(file);
    } catch (...) {
        fclose(file);  // Must remember to close
        throw;
    }
    
    fclose(file);  // Must remember to close
}""",
                "code_after": """// RAII with smart pointers
void processFile(const std::string& filename) {
    auto file = std::unique_ptr<FILE, decltype(&fclose)>(
        fopen(filename.c_str(), "r"), 
        &fclose
    );
    
    if (!file) {
        throw std::runtime_error("Cannot open file");
    }
    
    // Process file - automatic cleanup on scope exit
    processData(file.get());
}""",
                "complexity": "advanced",
                "tags": ["raii", "smart_pointer", "resource_management"],
                "performance_impact": "neutral",
                "readability_score": 8,
            },
            {
                "id": str(uuid.uuid4()),
                "language": "cpp",
                "pattern_type": "functional",
                "pattern_name": "move_semantics",
                "description": "Efficient resource transfer using move semantics",
                "code_before": """// Copy semantics (expensive)
class LargeObject {
private:
    std::vector<int> data;
    
public:
    LargeObject(size_t size) : data(size, 42) {}
    
    // Copy constructor
    LargeObject(const LargeObject& other) : data(other.data) {
        // Expensive deep copy
    }
    
    // Copy assignment
    LargeObject& operator=(const LargeObject& other) {
        if (this != &other) {
            data = other.data;  // Expensive copy
        }
        return *this;
    }
};""",
                "code_after": """// Move semantics (efficient)
class LargeObject {
private:
    std::vector<int> data;
    
public:
    LargeObject(size_t size) : data(size, 42) {}
    
    // Copy constructor
    LargeObject(const LargeObject& other) : data(other.data) {}
    
    // Move constructor
    LargeObject(LargeObject&& other) noexcept : data(std::move(other.data)) {
        // Efficient move, no copying
    }
    
    // Copy assignment
    LargeObject& operator=(const LargeObject& other) {
        if (this != &other) {
            data = other.data;
        }
        return *this;
    }
    
    // Move assignment
    LargeObject& operator=(LargeObject&& other) noexcept {
        if (this != &other) {
            data = std::move(other.data);  // Efficient move
        }
        return *this;
    }
};""",
                "complexity": "advanced",
                "tags": ["move_semantics", "performance", "cpp11"],
                "performance_impact": "positive",
                "readability_score": 7,
            },
        ]

        # Combine all patterns
        patterns_data.extend(python_patterns)
        patterns_data.extend(javascript_patterns)
        patterns_data.extend(java_patterns)
        patterns_data.extend(cpp_patterns)

        # Add metadata
        dataset_metadata = {
            "dataset_name": "code_patterns",
            "version": "1.0.0",
            "description": "Code patterns and idioms across multiple programming languages",
            "created_date": datetime.now().isoformat(),
            "total_samples": len(patterns_data),
            "languages": list(set(pattern["language"] for pattern in patterns_data)),
            "pattern_types": list(
                set(pattern["pattern_type"] for pattern in patterns_data)
            ),
            "complexity_levels": list(
                set(pattern["complexity"] for pattern in patterns_data)
            ),
        }

        # Save as JSON
        output_file = self.output_dir / "code_patterns.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {"metadata": dataset_metadata, "patterns": patterns_data},
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"✅ Generated {len(patterns_data)} code patterns")
        print(f"📁 Saved to: {output_file}")

    def generate_algorithm_implementations_dataset(self) -> None:
        """Generate dataset of algorithm implementations across languages."""
        print("📝 Generating algorithm implementations dataset...")

        algorithms_data = []

        # Sorting algorithms
        sorting_algorithms = [
            {
                "id": str(uuid.uuid4()),
                "algorithm_name": "quicksort",
                "category": "sorting",
                "language": "python",
                "description": "Divide-and-conquer sorting algorithm with average O(n log n) complexity",
                "time_complexity": {
                    "best": "O(n log n)",
                    "average": "O(n log n)",
                    "worst": "O(n²)",
                },
                "space_complexity": "O(log n)",
                "code": """def quicksort(arr, low=0, high=None):
    \"\"\"
    Sorts an array using the quicksort algorithm.
    
    Args:
        arr: List of comparable elements
        low: Starting index (default: 0)
        high: Ending index (default: len(arr) - 1)
    
    Returns:
        None (sorts in-place)
    \"\"\"
    if high is None:
        high = len(arr) - 1
    
    if low < high:
        # Partition the array and get pivot index
        pivot_index = partition(arr, low, high)
        
        # Recursively sort elements before and after partition
        quicksort(arr, low, pivot_index - 1)
        quicksort(arr, pivot_index + 1, high)

def partition(arr, low, high):
    \"\"\"
    Partitions the array around a pivot element.
    
    Args:
        arr: Array to partition
        low: Starting index
        high: Ending index
    
    Returns:
        Index of the pivot element after partitioning
    \"\"\"
    # Choose the rightmost element as pivot
    pivot = arr[high]
    
    # Index of smaller element
    i = low - 1
    
    for j in range(low, high):
        # If current element is smaller than or equal to pivot
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    # Place pivot in correct position
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

# Example usage
if __name__ == "__main__":
    test_array = [64, 34, 25, 12, 22, 11, 90]
    print(f"Original array: {test_array}")
    quicksort(test_array)
    print(f"Sorted array: {test_array}")""",
                "test_cases": [
                    {
                        "input": "[64, 34, 25, 12, 22, 11, 90]",
                        "expected_output": "[11, 12, 22, 25, 34, 64, 90]",
                    },
                    {"input": "[5, 2, 8, 1, 9]", "expected_output": "[1, 2, 5, 8, 9]"},
                    {"input": "[1]", "expected_output": "[1]"},
                    {"input": "[]", "expected_output": "[]"},
                ],
                "tags": ["sorting", "divide_conquer", "recursive", "in_place"],
                "difficulty": "intermediate",
                "applications": [
                    "General sorting",
                    "Database indexing",
                    "Search optimization",
                ],
            },
            {
                "id": str(uuid.uuid4()),
                "algorithm_name": "mergesort",
                "category": "sorting",
                "language": "java",
                "description": "Stable divide-and-conquer sorting algorithm with guaranteed O(n log n) complexity",
                "time_complexity": {
                    "best": "O(n log n)",
                    "average": "O(n log n)",
                    "worst": "O(n log n)",
                },
                "space_complexity": "O(n)",
                "code": """import java.util.Arrays;

public class MergeSort {
    
    /**
     * Sorts an array using the merge sort algorithm.
     * 
     * @param arr The array to be sorted
     */
    public static void mergeSort(int[] arr) {
        if (arr.length <= 1) {
            return;
        }
        
        mergeSortHelper(arr, 0, arr.length - 1);
    }
    
    /**
     * Recursive helper method for merge sort.
     * 
     * @param arr The array to sort
     * @param left Starting index
     * @param right Ending index
     */
    private static void mergeSortHelper(int[] arr, int left, int right) {
        if (left < right) {
            // Find the middle point
            int mid = left + (right - left) / 2;
            
            // Sort first and second halves
            mergeSortHelper(arr, left, mid);
            mergeSortHelper(arr, mid + 1, right);
            
            // Merge the sorted halves
            merge(arr, left, mid, right);
        }
    }
    
    /**
     * Merges two sorted subarrays into one sorted array.
     * 
     * @param arr The array containing both subarrays
     * @param left Starting index of first subarray
     * @param mid Ending index of first subarray
     * @param right Ending index of second subarray
     */
    private static void merge(int[] arr, int left, int mid, int right) {
        // Calculate sizes of subarrays
        int n1 = mid - left + 1;
        int n2 = right - mid;
        
        // Create temporary arrays
        int[] leftArray = new int[n1];
        int[] rightArray = new int[n2];
        
        // Copy data to temporary arrays
        System.arraycopy(arr, left, leftArray, 0, n1);
        System.arraycopy(arr, mid + 1, rightArray, 0, n2);
        
        // Merge the temporary arrays back into arr[left..right]
        int i = 0, j = 0, k = left;
        
        while (i < n1 && j < n2) {
            if (leftArray[i] <= rightArray[j]) {
                arr[k] = leftArray[i];
                i++;
            } else {
                arr[k] = rightArray[j];
                j++;
            }
            k++;
        }
        
        // Copy remaining elements
        while (i < n1) {
            arr[k] = leftArray[i];
            i++;
            k++;
        }
        
        while (j < n2) {
            arr[k] = rightArray[j];
            j++;
            k++;
        }
    }
    
    /**
     * Main method for testing the merge sort implementation.
     */
    public static void main(String[] args) {
        int[] testArray = {64, 34, 25, 12, 22, 11, 90};
        System.out.println("Original array: " + Arrays.toString(testArray));
        
        mergeSort(testArray);
        System.out.println("Sorted array: " + Arrays.toString(testArray));
    }
}""",
                "test_cases": [
                    {
                        "input": "[64, 34, 25, 12, 22, 11, 90]",
                        "expected_output": "[11, 12, 22, 25, 34, 64, 90]",
                    },
                    {"input": "[5, 2, 8, 1, 9]", "expected_output": "[1, 2, 5, 8, 9]"},
                    {"input": "[1]", "expected_output": "[1]"},
                    {"input": "[]", "expected_output": "[]"},
                ],
                "tags": ["sorting", "divide_conquer", "stable", "recursive"],
                "difficulty": "intermediate",
                "applications": [
                    "External sorting",
                    "Stable sorting requirements",
                    "Parallel processing",
                ],
            },
        ]

        # Graph algorithms
        graph_algorithms = [
            {
                "id": str(uuid.uuid4()),
                "algorithm_name": "dijkstra",
                "category": "graph",
                "language": "python",
                "description": "Shortest path algorithm for weighted graphs with non-negative edge weights",
                "time_complexity": {
                    "best": "O((V + E) log V)",
                    "average": "O((V + E) log V)",
                    "worst": "O((V + E) log V)",
                },
                "space_complexity": "O(V)",
                "code": """import heapq
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

class Graph:
    \"\"\"Weighted graph implementation for Dijkstra's algorithm.\"\"\"
    
    def __init__(self):
        self.vertices = set()
        self.edges = defaultdict(list)
    
    def add_edge(self, u: int, v: int, weight: float) -> None:
        \"\"\"Add a weighted edge to the graph.\"\"\"
        self.vertices.add(u)
        self.vertices.add(v)
        self.edges[u].append((v, weight))
    
    def dijkstra(self, start: int) -> Tuple[Dict[int, float], Dict[int, Optional[int]]]:
        \"\"\"
        Find shortest paths from start vertex to all other vertices.
        
        Args:
            start: Starting vertex
            
        Returns:
            Tuple of (distances, predecessors) dictionaries
        \"\"\"
        # Initialize distances and predecessors
        distances = {vertex: float('infinity') for vertex in self.vertices}
        predecessors = {vertex: None for vertex in self.vertices}
        distances[start] = 0
        
        # Priority queue: (distance, vertex)
        pq = [(0, start)]
        visited = set()
        
        while pq:
            current_distance, current_vertex = heapq.heappop(pq)
            
            # Skip if already visited
            if current_vertex in visited:
                continue
                
            visited.add(current_vertex)
            
            # Check all neighbors
            for neighbor, weight in self.edges[current_vertex]:
                if neighbor not in visited:
                    new_distance = current_distance + weight
                    
                    # If shorter path found, update
                    if new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        predecessors[neighbor] = current_vertex
                        heapq.heappush(pq, (new_distance, neighbor))
        
        return distances, predecessors
    
    def get_shortest_path(self, start: int, end: int) -> Optional[List[int]]:
        \"\"\"
        Get the shortest path between two vertices.
        
        Args:
            start: Starting vertex
            end: Ending vertex
            
        Returns:
            List of vertices representing the shortest path, or None if no path exists
        \"\"\"
        distances, predecessors = self.dijkstra(start)
        
        if distances[end] == float('infinity'):
            return None  # No path exists
        
        # Reconstruct path
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = predecessors[current]
        
        return path[::-1]  # Reverse to get start -> end

# Example usage
if __name__ == "__main__":
    # Create a sample graph
    g = Graph()
    g.add_edge(0, 1, 4)
    g.add_edge(0, 2, 2)
    g.add_edge(1, 2, 1)
    g.add_edge(1, 3, 5)
    g.add_edge(2, 3, 8)
    g.add_edge(2, 4, 10)
    g.add_edge(3, 4, 2)
    
    # Find shortest paths from vertex 0
    distances, predecessors = g.dijkstra(0)
    print("Shortest distances from vertex 0:")
    for vertex, distance in distances.items():
        print(f"To vertex {vertex}: {distance}")
    
    # Find shortest path from 0 to 4
    path = g.get_shortest_path(0, 4)
    print(f"Shortest path from 0 to 4: {path}")""",
                "test_cases": [
                    {
                        "input": "Graph with edges: (0,1,4), (0,2,2), (1,2,1), (1,3,5), (2,3,8), (2,4,10), (3,4,2)",
                        "expected_output": "Shortest distances from 0: {0: 0, 1: 3, 2: 2, 3: 8, 4: 10}",
                    }
                ],
                "tags": ["graph", "shortest_path", "greedy", "priority_queue"],
                "difficulty": "advanced",
                "applications": [
                    "GPS navigation",
                    "Network routing",
                    "Social networks",
                ],
            }
        ]

        # Combine all algorithms
        algorithms_data.extend(sorting_algorithms)
        algorithms_data.extend(graph_algorithms)

        # Add metadata
        dataset_metadata = {
            "dataset_name": "algorithm_implementations",
            "version": "1.0.0",
            "description": "Comprehensive algorithm implementations across multiple programming languages",
            "created_date": datetime.now().isoformat(),
            "total_samples": len(algorithms_data),
            "languages": list(set(algo["language"] for algo in algorithms_data)),
            "categories": list(set(algo["category"] for algo in algorithms_data)),
            "difficulty_levels": list(
                set(algo["difficulty"] for algo in algorithms_data)
            ),
        }

        # Save as JSON
        output_file = self.output_dir / "algorithm_implementations.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {"metadata": dataset_metadata, "algorithms": algorithms_data},
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"✅ Generated {len(algorithms_data)} algorithm implementations")
        print(f"📁 Saved to: {output_file}")

    def generate_error_handling_dataset(self) -> None:
        """Generate dataset of error handling patterns and examples."""
        print("📝 Generating error handling dataset...")

        error_examples = []

        # Python error handling examples
        python_errors = [
            {
                "id": str(uuid.uuid4()),
                "language": "python",
                "error_type": "TypeError",
                "category": "type_errors",
                "description": "Attempting to use incompatible types in operations",
                "buggy_code": """def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)

# This will cause TypeError
result = calculate_average("123")  # Passing string instead of list""",
                "fixed_code": """def calculate_average(numbers):
    if not isinstance(numbers, (list, tuple)):
        raise TypeError("Expected list or tuple of numbers")
    
    if not numbers:
        raise ValueError("Cannot calculate average of empty sequence")
    
    total = 0
    for num in numbers:
        if not isinstance(num, (int, float)):
            raise TypeError(f"Expected number, got {type(num).__name__}")
        total += num
    
    return total / len(numbers)

# Correct usage
try:
    result = calculate_average([1, 2, 3, 4, 5])
    print(f"Average: {result}")
except (TypeError, ValueError) as e:
    print(f"Error: {e}")""",
                "error_message": "unsupported operand type(s) for +=: 'int' and 'str'",
                "fix_explanation": "Added type checking and proper error handling with descriptive error messages",
                "severity": "medium",
                "common_cause": "Not validating input types before processing",
                "prevention_tips": [
                    "Use type hints and type checking",
                    "Validate input parameters",
                    "Handle edge cases explicitly",
                    "Provide clear error messages",
                ],
            },
            {
                "id": str(uuid.uuid4()),
                "language": "python",
                "error_type": "IndexError",
                "category": "boundary_errors",
                "description": "Accessing list elements beyond valid indices",
                "buggy_code": """def get_nth_element(lst, n):
    return lst[n]

# This will cause IndexError
my_list = [1, 2, 3]
element = get_nth_element(my_list, 5)  # Index 5 doesn't exist""",
                "fixed_code": """def get_nth_element(lst, n, default=None):
    \"\"\"
    Safely get the nth element from a list.
    
    Args:
        lst: The list to access
        n: Index to retrieve
        default: Value to return if index is out of bounds
    
    Returns:
        Element at index n, or default if out of bounds
    \"\"\"
    if not isinstance(lst, (list, tuple)):
        raise TypeError("Expected list or tuple")
    
    if not isinstance(n, int):
        raise TypeError("Index must be an integer")
    
    try:
        return lst[n]
    except IndexError:
        if default is not None:
            return default
        else:
            raise IndexError(f"Index {n} is out of range for list of length {len(lst)}")

# Safe usage
my_list = [1, 2, 3]
try:
    element = get_nth_element(my_list, 5, default="Not found")
    print(f"Element: {element}")
except IndexError as e:
    print(f"Error: {e}")""",
                "error_message": "list index out of range",
                "fix_explanation": "Added bounds checking and optional default value for out-of-range access",
                "severity": "medium",
                "common_cause": "Not checking list bounds before accessing elements",
                "prevention_tips": [
                    "Always check list length before accessing by index",
                    "Use try-except for index access",
                    "Consider using get() method for dictionaries",
                    "Validate indices are within bounds",
                ],
            },
        ]

        # JavaScript error handling examples
        javascript_errors = [
            {
                "id": str(uuid.uuid4()),
                "language": "javascript",
                "error_type": "TypeError",
                "category": "null_undefined_errors",
                "description": "Attempting to call methods on null or undefined values",
                "buggy_code": """function processUser(user) {
    // This will throw TypeError if user is null/undefined
    return user.name.toUpperCase();
}

// This will cause TypeError
const result = processUser(null);""",
                "fixed_code": """function processUser(user) {
    // Validate input
    if (!user) {
        throw new Error('User object is required');
    }
    
    if (typeof user !== 'object') {
        throw new TypeError('User must be an object');
    }
    
    if (!user.name) {
        throw new Error('User name is required');
    }
    
    if (typeof user.name !== 'string') {
        throw new TypeError('User name must be a string');
    }
    
    return user.name.toUpperCase();
}

// Safe usage with error handling
try {
    const user = { name: 'John Doe' };
    const result = processUser(user);
    console.log('Processed user:', result);
} catch (error) {
    console.error('Error processing user:', error.message);
}

// Alternative: Using optional chaining (ES2020)
function processUserSafe(user) {
    return user?.name?.toUpperCase() ?? 'No name provided';
}""",
                "error_message": "Cannot read property 'toUpperCase' of undefined",
                "fix_explanation": "Added comprehensive null/undefined checks and used optional chaining for safer property access",
                "severity": "high",
                "common_cause": "Not handling null/undefined values before property access",
                "prevention_tips": [
                    "Always validate objects before accessing properties",
                    "Use optional chaining (?.) operator",
                    "Provide default values with nullish coalescing (??)",
                    "Use TypeScript for better type safety",
                ],
            }
        ]

        # Java error handling examples
        java_errors = [
            {
                "id": str(uuid.uuid4()),
                "language": "java",
                "error_type": "NullPointerException",
                "category": "null_reference_errors",
                "description": "Attempting to use a null reference",
                "buggy_code": """public class UserProcessor {
    public String processUserName(User user) {
        // This will throw NPE if user is null
        return user.getName().toUpperCase();
    }
    
    public static void main(String[] args) {
        UserProcessor processor = new UserProcessor();
        // This will cause NullPointerException
        String result = processor.processUserName(null);
    }
}""",
                "fixed_code": """import java.util.Objects;
import java.util.Optional;

public class UserProcessor {
    
    /**
     * Processes user name with proper null checking.
     * 
     * @param user The user object to process
     * @return Uppercase user name
     * @throws IllegalArgumentException if user or user name is null/empty
     */
    public String processUserName(User user) {
        // Validate user object
        Objects.requireNonNull(user, "User cannot be null");
        
        String name = user.getName();
        if (name == null || name.trim().isEmpty()) {
            throw new IllegalArgumentException("User name cannot be null or empty");
        }
        
        return name.toUpperCase();
    }
    
    /**
     * Safe version using Optional.
     * 
     * @param user The user object to process
     * @return Optional containing uppercase name, or empty if processing fails
     */
    public Optional<String> processUserNameSafe(User user) {
        return Optional.ofNullable(user)
                .map(User::getName)
                .filter(name -> !name.trim().isEmpty())
                .map(String::toUpperCase);
    }
    
    public static void main(String[] args) {
        UserProcessor processor = new UserProcessor();
        
        try {
            User user = new User("John Doe");
            String result = processor.processUserName(user);
            System.out.println("Processed user: " + result);
        } catch (IllegalArgumentException e) {
            System.err.println("Error: " + e.getMessage());
        }
        
        // Using Optional approach
        Optional<String> safeResult = processor.processUserNameSafe(new User("Jane Doe"));
        safeResult.ifPresentOrElse(
            name -> System.out.println("Safe processed user: " + name),
            () -> System.out.println("Could not process user safely")
        );
    }
}""",
                "error_message": "java.lang.NullPointerException",
                "fix_explanation": "Added null checking with Objects.requireNonNull() and provided an Optional-based safe alternative",
                "severity": "high",
                "common_cause": "Not checking for null references before method calls",
                "prevention_tips": [
                    "Use Objects.requireNonNull() for parameter validation",
                    "Consider using Optional for nullable return values",
                    "Use @Nullable and @NonNull annotations",
                    "Initialize fields appropriately",
                ],
            }
        ]

        # Combine all error examples
        error_examples.extend(python_errors)
        error_examples.extend(javascript_errors)
        error_examples.extend(java_errors)

        # Add metadata
        dataset_metadata = {
            "dataset_name": "error_handling_examples",
            "version": "1.0.0",
            "description": "Common programming errors and their fixes across multiple languages",
            "created_date": datetime.now().isoformat(),
            "total_samples": len(error_examples),
            "languages": list(set(error["language"] for error in error_examples)),
            "error_types": list(set(error["error_type"] for error in error_examples)),
            "categories": list(set(error["category"] for error in error_examples)),
            "severity_levels": list(set(error["severity"] for error in error_examples)),
        }

        # Save as JSON
        output_file = self.output_dir / "error_handling_examples.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {"metadata": dataset_metadata, "error_examples": error_examples},
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"✅ Generated {len(error_examples)} error handling examples")
        print(f"📁 Saved to: {output_file}")

    def generate_all_datasets(self) -> None:
        """Generate all raw datasets for ML training."""
        print("🚀 Starting comprehensive dataset generation...\n")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate all datasets
        self.generate_code_patterns_dataset()
        print()
        self.generate_algorithm_implementations_dataset()
        print()
        self.generate_error_handling_dataset()
        print()

        print("🎉 All raw datasets generated successfully!")
        print(f"📁 Output directory: {self.output_dir}")

        # Generate summary
        self.generate_dataset_summary()

    def generate_dataset_summary(self) -> None:
        """Generate a summary of all created datasets."""
        summary = {
            "generation_date": datetime.now().isoformat(),
            "total_datasets": 3,
            "datasets": [
                {
                    "name": "code_patterns",
                    "file": "code_patterns.json",
                    "description": "Code patterns and idioms across multiple programming languages",
                },
                {
                    "name": "algorithm_implementations",
                    "file": "algorithm_implementations.json",
                    "description": "Comprehensive algorithm implementations across multiple programming languages",
                },
                {
                    "name": "error_handling_examples",
                    "file": "error_handling_examples.json",
                    "description": "Common programming errors and their fixes across multiple languages",
                },
            ],
            "supported_languages": self.languages,
            "ready_for_processing": True,
        }

        summary_file = self.output_dir / "dataset_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"📋 Dataset summary saved to: {summary_file}")


if __name__ == "__main__":
    generator = MLDatasetGenerator()
    generator.generate_all_datasets()
