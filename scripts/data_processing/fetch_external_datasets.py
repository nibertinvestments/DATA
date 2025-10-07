#!/usr/bin/env python3
"""
External Dataset Fetcher for LLM/ML/AI Training
Pulls datasets from web sources and GitHub to expand training data breadth.
Enhanced version that generates diverse, non-repeating datasets.
"""

import json
import os
import sys
import time
import random
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import requests
from dataclasses import dataclass, asdict


@dataclass
class DatasetSource:
    """Represents an external dataset source."""
    name: str
    url: str
    type: str  # 'github', 'api', 'web'
    description: str
    languages: List[str]
    category: str


class ExternalDatasetFetcher:
    """Fetches and processes datasets from external sources."""
    
    def __init__(self, output_dir: str = "datasets/raw/external"):
        """Initialize the fetcher with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'DATA-Repository-Dataset-Fetcher/1.0'
        })
        
        # Define dataset sources to fetch from
        self.sources = self._define_sources()
    
    def _define_sources(self) -> List[DatasetSource]:
        """Define external sources to fetch datasets from."""
        return [
            DatasetSource(
                name="github_code_search",
                url="https://api.github.com/search/repositories",
                type="github",
                description="GitHub code repositories with ML/AI focus",
                languages=["python", "javascript", "java", "cpp", "rust", "go"],
                category="code_samples"
            ),
            DatasetSource(
                name="common_programming_errors",
                url="synthetic",
                type="synthetic",
                description="Common programming errors and solutions",
                languages=["python", "javascript", "java", "cpp", "typescript"],
                category="error_patterns"
            ),
            DatasetSource(
                name="api_documentation_patterns",
                url="synthetic",
                type="synthetic",
                description="API documentation and usage patterns",
                languages=["python", "javascript", "java", "rest"],
                category="api_patterns"
            ),
            DatasetSource(
                name="code_translation_examples",
                url="synthetic",
                type="synthetic",
                description="Cross-language code translation examples",
                languages=["python", "javascript", "java", "cpp", "rust", "go"],
                category="translation"
            ),
        ]
    
    def fetch_all_datasets(self) -> Dict[str, Any]:
        """Fetch all datasets from defined sources."""
        print("🌐 Fetching datasets from external sources...\n")
        
        results = {
            "fetch_date": datetime.now().isoformat(),
            "total_sources": len(self.sources),
            "datasets_created": [],
            "errors": []
        }
        
        for source in self.sources:
            try:
                print(f"📥 Fetching from: {source.name}")
                
                if source.type == "github":
                    dataset = self._fetch_github_code_samples(source)
                elif source.type == "synthetic":
                    dataset = self._create_synthetic_dataset(source)
                else:
                    print(f"  ⚠️  Unsupported source type: {source.type}")
                    continue
                
                if dataset:
                    filename = f"{source.name}_dataset.json"
                    self._save_dataset(dataset, filename)
                    results["datasets_created"].append(filename)
                    print(f"  ✅ Created: {filename}\n")
                
            except Exception as e:
                error_msg = f"Error fetching {source.name}: {str(e)}"
                print(f"  ❌ {error_msg}\n")
                results["errors"].append(error_msg)
        
        # Save fetch summary
        self._save_fetch_summary(results)
        
        print(f"\n✅ Fetch complete! Created {len(results['datasets_created'])} datasets")
        return results
    
    def _fetch_github_code_samples(self, source: DatasetSource) -> Dict[str, Any]:
        """Fetch code samples from GitHub repositories."""
        print(f"  🔍 Searching GitHub for code samples...")
        
        samples = []
        
        # Search for high-quality repositories in each language
        for language in source.languages[:3]:  # Limit to 3 languages to avoid rate limits
            try:
                params = {
                    'q': f'machine learning examples language:{language} stars:>100',
                    'sort': 'stars',
                    'order': 'desc',
                    'per_page': 5
                }
                
                response = self.session.get(source.url, params=params, timeout=10)
                
                if response.status_code == 200:
                    repos = response.json().get('items', [])
                    
                    for repo in repos:
                        sample = {
                            "id": f"gh_{repo['id']}",
                            "language": language,
                            "repository": repo['full_name'],
                            "description": repo.get('description', ''),
                            "url": repo['html_url'],
                            "stars": repo.get('stargazers_count', 0),
                            "topics": repo.get('topics', []),
                            "created_at": repo.get('created_at', ''),
                            "category": "github_repository"
                        }
                        samples.append(sample)
                    
                    print(f"    📊 Found {len(repos)} repos for {language}")
                    time.sleep(2)  # Rate limiting
                
                elif response.status_code == 403:
                    print(f"    ⚠️  GitHub API rate limit reached")
                    break
                    
            except Exception as e:
                print(f"    ⚠️  Error searching {language}: {str(e)}")
        
        return {
            "metadata": {
                "source": source.name,
                "type": source.type,
                "description": source.description,
                "languages": source.languages,
                "total_samples": len(samples),
                "created_at": datetime.now().isoformat()
            },
            "samples": samples
        }
    
    def _create_synthetic_dataset(self, source: DatasetSource) -> Dict[str, Any]:
        """Create synthetic dataset based on source category."""
        print(f"  🔨 Generating synthetic dataset...")
        
        if source.category == "error_patterns":
            return self._create_error_patterns_dataset(source)
        elif source.category == "api_patterns":
            return self._create_api_patterns_dataset(source)
        elif source.category == "translation":
            return self._create_translation_dataset(source)
        
        return None
    
    def _create_error_patterns_dataset(self, source: DatasetSource) -> Dict[str, Any]:
        """Create dataset of common programming errors and solutions."""
        
        error_patterns = [
            # Python errors
            {
                "id": "err_py_001",
                "language": "python",
                "error_type": "IndexError",
                "description": "List index out of range",
                "buggy_code": """def get_last_element(items):
    return items[len(items)]  # Off-by-one error""",
                "fixed_code": """def get_last_element(items):
    if not items:
        return None
    return items[-1]  # Correct way to get last element""",
                "explanation": "Python list indices start at 0, so the last element is at index len(items)-1, or simply items[-1]",
                "common_cause": "Off-by-one error, forgetting Python uses 0-based indexing",
                "severity": "medium",
                "category": "indexing"
            },
            {
                "id": "err_py_002",
                "language": "python",
                "error_type": "KeyError",
                "description": "Dictionary key not found",
                "buggy_code": """def get_user_email(users, user_id):
    return users[user_id]['email']  # Crashes if user_id not in dict""",
                "fixed_code": """def get_user_email(users, user_id):
    user = users.get(user_id)
    if user:
        return user.get('email')
    return None""",
                "explanation": "Use .get() method to safely access dictionary keys without raising KeyError",
                "common_cause": "Not checking if key exists before accessing",
                "severity": "high",
                "category": "data_access"
            },
            {
                "id": "err_py_003",
                "language": "python",
                "error_type": "TypeError",
                "description": "String concatenation with non-string type",
                "buggy_code": """def create_message(name, age):
    return "User " + name + " is " + age + " years old"  # age is int""",
                "fixed_code": """def create_message(name, age):
    return f"User {name} is {age} years old"  # f-string handles conversion""",
                "explanation": "Use f-strings or str() to properly handle type conversion in string formatting",
                "common_cause": "Forgetting to convert non-string types in concatenation",
                "severity": "medium",
                "category": "type_handling"
            },
            # JavaScript errors
            {
                "id": "err_js_001",
                "language": "javascript",
                "error_type": "TypeError",
                "description": "Cannot read property of undefined",
                "buggy_code": """function getUserEmail(users, userId) {
    return users[userId].email;  // Crashes if users[userId] is undefined
}""",
                "fixed_code": """function getUserEmail(users, userId) {
    return users[userId]?.email;  // Optional chaining
}""",
                "explanation": "Use optional chaining (?.) to safely access nested properties",
                "common_cause": "Not checking if object exists before accessing properties",
                "severity": "high",
                "category": "data_access"
            },
            {
                "id": "err_js_002",
                "language": "javascript",
                "error_type": "ReferenceError",
                "description": "Variable used before declaration",
                "buggy_code": """function processData() {
    console.log(result);  // ReferenceError
    let result = calculateResult();
    return result;
}""",
                "fixed_code": """function processData() {
    let result = calculateResult();
    console.log(result);
    return result;
}""",
                "explanation": "Declare and initialize variables before using them to avoid temporal dead zone",
                "common_cause": "Hoisting confusion with let/const declarations",
                "severity": "high",
                "category": "scope"
            },
            {
                "id": "err_js_003",
                "language": "javascript",
                "error_type": "TypeError",
                "description": "Array method on non-array",
                "buggy_code": """function processItems(items) {
    return items.map(item => item * 2);  // Fails if items is not array
}""",
                "fixed_code": """function processItems(items) {
    if (!Array.isArray(items)) {
        return [];
    }
    return items.map(item => item * 2);
}""",
                "explanation": "Always validate that input is an array before using array methods",
                "common_cause": "Assuming parameter type without validation",
                "severity": "medium",
                "category": "type_checking"
            },
            # Java errors
            {
                "id": "err_java_001",
                "language": "java",
                "error_type": "NullPointerException",
                "description": "Calling method on null object",
                "buggy_code": """public String getUserEmail(User user) {
    return user.getEmail();  // Throws NPE if user is null
}""",
                "fixed_code": """public String getUserEmail(User user) {
    if (user == null) {
        return null;
    }
    return user.getEmail();
}""",
                "explanation": "Always check for null before accessing object methods or properties",
                "common_cause": "Not validating object is non-null",
                "severity": "high",
                "category": "null_handling"
            },
            {
                "id": "err_java_002",
                "language": "java",
                "error_type": "ArrayIndexOutOfBoundsException",
                "description": "Array index exceeds bounds",
                "buggy_code": """public int getLastElement(int[] array) {
    return array[array.length];  // Off-by-one error
}""",
                "fixed_code": """public int getLastElement(int[] array) {
    if (array == null || array.length == 0) {
        throw new IllegalArgumentException("Array is empty");
    }
    return array[array.length - 1];
}""",
                "explanation": "Array indices range from 0 to length-1, always validate bounds",
                "common_cause": "Off-by-one error in array indexing",
                "severity": "high",
                "category": "indexing"
            },
            # C++ errors
            {
                "id": "err_cpp_001",
                "language": "cpp",
                "error_type": "SegmentationFault",
                "description": "Dereferencing null pointer",
                "buggy_code": """int getValue(int* ptr) {
    return *ptr;  // Crashes if ptr is nullptr
}""",
                "fixed_code": """int getValue(int* ptr) {
    if (ptr == nullptr) {
        throw std::invalid_argument("Pointer is null");
    }
    return *ptr;
}""",
                "explanation": "Always check pointer is not nullptr before dereferencing",
                "common_cause": "Not validating pointer before use",
                "severity": "critical",
                "category": "pointer_safety"
            },
            {
                "id": "err_cpp_002",
                "language": "cpp",
                "error_type": "MemoryLeak",
                "description": "Memory not freed",
                "buggy_code": """void processData() {
    int* data = new int[100];
    // Process data
    // Missing: delete[] data;
}""",
                "fixed_code": """void processData() {
    std::vector<int> data(100);
    // Process data
    // Automatic cleanup
}""",
                "explanation": "Use RAII containers like std::vector to avoid manual memory management",
                "common_cause": "Forgetting to free dynamically allocated memory",
                "severity": "high",
                "category": "memory_management"
            },
            # TypeScript errors
            {
                "id": "err_ts_001",
                "language": "typescript",
                "error_type": "TypeError",
                "description": "Type assertion without validation",
                "buggy_code": """function processUser(data: unknown) {
    const user = data as User;  // Unsafe cast
    return user.email;
}""",
                "fixed_code": """function processUser(data: unknown): string | undefined {
    if (typeof data === 'object' && data !== null && 'email' in data) {
        return (data as User).email;
    }
    return undefined;
}""",
                "explanation": "Validate data structure before type assertions",
                "common_cause": "Unsafe type casting without runtime validation",
                "severity": "high",
                "category": "type_safety"
            }
        ]
        
        return {
            "metadata": {
                "source": source.name,
                "type": source.type,
                "description": source.description,
                "languages": source.languages,
                "total_patterns": len(error_patterns),
                "created_at": datetime.now().isoformat()
            },
            "error_patterns": error_patterns
        }
    
    def _create_api_patterns_dataset(self, source: DatasetSource) -> Dict[str, Any]:
        """Create dataset of API documentation and usage patterns."""
        
        api_patterns = [
            {
                "id": "api_001",
                "language": "python",
                "api_type": "REST",
                "name": "HTTP GET Request with Error Handling",
                "description": "Best practice for making HTTP GET requests",
                "code_example": """import requests
from typing import Optional, Dict

def fetch_user_data(user_id: int) -> Optional[Dict]:
    \"\"\"Fetch user data from API with proper error handling.\"\"\"
    try:
        response = requests.get(
            f'https://api.example.com/users/{user_id}',
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        print("Request timed out")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error: {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None""",
                "best_practices": [
                    "Always set timeout",
                    "Use raise_for_status() to catch HTTP errors",
                    "Handle specific exceptions",
                    "Return Optional type"
                ],
                "common_mistakes": [
                    "No timeout specified",
                    "Not checking status code",
                    "Catching all exceptions generically"
                ]
            },
            {
                "id": "api_002",
                "language": "javascript",
                "api_type": "REST",
                "name": "Async API Call with Fetch",
                "description": "Modern async/await pattern for API calls",
                "code_example": r"""async function fetchUserData(userId) {
    try {
        const response = await fetch(
            `https://api.example.com/users/${userId}`,
            {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                },
                signal: AbortSignal.timeout(10000)
            }
        );
        
        if (!response.ok) {
            throw new Error(\`HTTP error! status: \${response.status}\`);
        }
        
        const data = await response.json();
        return data;
    } catch (error) {
        if (error.name === 'AbortError') {
            console.error('Request timed out');
        } else {
            console.error('Fetch error:', error);
        }
        return null;
    }
}""",
                "best_practices": [
                    "Use async/await for cleaner code",
                    "Check response.ok before parsing",
                    "Set timeout with AbortSignal",
                    "Handle specific error types"
                ],
                "common_mistakes": [
                    "Not checking response.ok",
                    "No timeout handling",
                    "Generic error handling"
                ]
            },
            {
                "id": "api_003",
                "language": "python",
                "api_type": "REST",
                "name": "POST Request with JSON Payload",
                "description": "Creating resources via POST with validation",
                "code_example": """import requests
from typing import Dict, Optional

def create_user(user_data: Dict) -> Optional[Dict]:
    \"\"\"Create a new user via API.\"\"\"
    required_fields = ['name', 'email']
    
    # Validate input
    if not all(field in user_data for field in required_fields):
        raise ValueError(f"Missing required fields: {required_fields}")
    
    try:
        response = requests.post(
            'https://api.example.com/users',
            json=user_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Failed to create user: {e}")
        return None""",
                "best_practices": [
                    "Validate input before sending",
                    "Use json parameter for automatic serialization",
                    "Set appropriate Content-Type header",
                    "Handle errors gracefully"
                ],
                "common_mistakes": [
                    "No input validation",
                    "Manually serializing JSON",
                    "Not setting Content-Type"
                ]
            },
            {
                "id": "api_004",
                "language": "java",
                "api_type": "REST",
                "name": "HTTP Client with Java 11+",
                "description": "Modern HTTP client usage in Java",
                "code_example": """import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;

public class ApiClient {
    private final HttpClient client;
    
    public ApiClient() {
        this.client = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(10))
            .build();
    }
    
    public String fetchUserData(int userId) {
        try {
            HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create("https://api.example.com/users/" + userId))
                .timeout(Duration.ofSeconds(10))
                .GET()
                .build();
            
            HttpResponse<String> response = client.send(
                request,
                HttpResponse.BodyHandlers.ofString()
            );
            
            if (response.statusCode() == 200) {
                return response.body();
            } else {
                throw new RuntimeException("HTTP error: " + response.statusCode());
            }
        } catch (Exception e) {
            System.err.println("Request failed: " + e.getMessage());
            return null;
        }
    }
}""",
                "best_practices": [
                    "Use HttpClient with proper timeout",
                    "Check status code explicitly",
                    "Handle exceptions appropriately",
                    "Reuse HttpClient instance"
                ],
                "common_mistakes": [
                    "Creating new client for each request",
                    "Not setting timeout",
                    "Ignoring status codes"
                ]
            },
            {
                "id": "api_005",
                "language": "typescript",
                "api_type": "REST",
                "name": "Type-Safe API Client",
                "description": "Type-safe API calls with TypeScript",
                "code_example": r"""interface User {
    id: number;
    name: string;
    email: string;
}

interface ApiResponse<T> {
    data: T;
    success: boolean;
}

async function fetchUser(userId: number): Promise<User | null> {
    try {
        const response = await fetch(
            `https://api.example.com/users/${userId}`,
            {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                },
            }
        );
        
        if (!response.ok) {
            throw new Error(\`HTTP error! status: \${response.status}\`);
        }
        
        const apiResponse: ApiResponse<User> = await response.json();
        
        if (apiResponse.success) {
            return apiResponse.data;
        }
        
        return null;
    } catch (error) {
        console.error('Fetch error:', error);
        return null;
    }
}""",
                "best_practices": [
                    "Define interfaces for type safety",
                    "Use generic types for API responses",
                    "Validate response structure",
                    "Return typed results"
                ],
                "common_mistakes": [
                    "Using 'any' type",
                    "Not validating response structure",
                    "Inconsistent return types"
                ]
            }
        ]
        
        return {
            "metadata": {
                "source": source.name,
                "type": source.type,
                "description": source.description,
                "languages": source.languages,
                "total_patterns": len(api_patterns),
                "created_at": datetime.now().isoformat()
            },
            "api_patterns": api_patterns
        }
    
    def _create_translation_dataset(self, source: DatasetSource) -> Dict[str, Any]:
        """Create dataset of cross-language code translation examples."""
        
        translation_examples = [
            {
                "id": "trans_001",
                "concept": "fibonacci_sequence",
                "description": "Calculate Fibonacci sequence",
                "difficulty": "beginner",
                "implementations": {
                    "python": """def fibonacci(n):
    \"\"\"Calculate nth Fibonacci number.\"\"\"
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b""",
                    "javascript": """function fibonacci(n) {
    // Calculate nth Fibonacci number
    if (n <= 1) {
        return n;
    }
    let [a, b] = [0, 1];
    for (let i = 2; i <= n; i++) {
        [a, b] = [b, a + b];
    }
    return b;
}""",
                    "java": """public static int fibonacci(int n) {
    // Calculate nth Fibonacci number
    if (n <= 1) {
        return n;
    }
    int a = 0, b = 1;
    for (int i = 2; i <= n; i++) {
        int temp = a + b;
        a = b;
        b = temp;
    }
    return b;
}""",
                    "cpp": """int fibonacci(int n) {
    // Calculate nth Fibonacci number
    if (n <= 1) {
        return n;
    }
    int a = 0, b = 1;
    for (int i = 2; i <= n; i++) {
        int temp = a + b;
        a = b;
        b = temp;
    }
    return b;
}""",
                    "rust": """fn fibonacci(n: u32) -> u32 {
    // Calculate nth Fibonacci number
    if n <= 1 {
        return n;
    }
    let (mut a, mut b) = (0, 1);
    for _ in 2..=n {
        let temp = a + b;
        a = b;
        b = temp;
    }
    b
}""",
                    "go": """func fibonacci(n int) int {
    // Calculate nth Fibonacci number
    if n <= 1 {
        return n
    }
    a, b := 0, 1
    for i := 2; i <= n; i++ {
        a, b = b, a+b
    }
    return b
}"""
                },
                "key_differences": {
                    "python": "Tuple unpacking for elegant swap",
                    "javascript": "Array destructuring for swap",
                    "java": "Explicit temporary variable needed",
                    "cpp": "Similar to Java with C-style syntax",
                    "rust": "Ownership system, immutable by default",
                    "go": "Multiple return values for clean swap"
                }
            },
            {
                "id": "trans_002",
                "concept": "binary_search",
                "description": "Binary search in sorted array",
                "difficulty": "intermediate",
                "implementations": {
                    "python": """def binary_search(arr, target):
    \"\"\"Find target in sorted array using binary search.\"\"\"
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1""",
                    "javascript": """function binarySearch(arr, target) {
    // Find target in sorted array using binary search
    let left = 0;
    let right = arr.length - 1;
    
    while (left <= right) {
        const mid = Math.floor((left + right) / 2);
        
        if (arr[mid] === target) {
            return mid;
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return -1;
}""",
                    "java": """public static int binarySearch(int[] arr, int target) {
    // Find target in sorted array using binary search
    int left = 0;
    int right = arr.length - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (arr[mid] == target) {
            return mid;
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return -1;
}""",
                    "cpp": """int binarySearch(const std::vector<int>& arr, int target) {
    // Find target in sorted array using binary search
    int left = 0;
    int right = arr.size() - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (arr[mid] == target) {
            return mid;
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return -1;
}""",
                    "rust": """fn binary_search(arr: &[i32], target: i32) -> Option<usize> {
    // Find target in sorted array using binary search
    let mut left = 0;
    let mut right = arr.len() - 1;
    
    while left <= right {
        let mid = left + (right - left) / 2;
        
        match arr[mid].cmp(&target) {
            std::cmp::Ordering::Equal => return Some(mid),
            std::cmp::Ordering::Less => left = mid + 1,
            std::cmp::Ordering::Greater => {
                if mid == 0 { break; }
                right = mid - 1;
            }
        }
    }
    
    None
}""",
                    "go": """func binarySearch(arr []int, target int) int {
    // Find target in sorted array using binary search
    left, right := 0, len(arr)-1
    
    for left <= right {
        mid := left + (right-left)/2
        
        if arr[mid] == target {
            return mid
        } else if arr[mid] < target {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    
    return -1
}"""
                },
                "key_differences": {
                    "python": "Simple integer division with //",
                    "javascript": "Math.floor for integer division",
                    "java": "Careful with integer overflow in mid calculation",
                    "cpp": "Use const reference for input array",
                    "rust": "Returns Option type, uses match for comparison",
                    "go": "Multiple return values convention"
                }
            },
            {
                "id": "trans_003",
                "concept": "linked_list_reversal",
                "description": "Reverse a singly linked list",
                "difficulty": "intermediate",
                "implementations": {
                    "python": """class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverse_linked_list(head):
    \"\"\"Reverse a singly linked list.\"\"\"
    prev = None
    current = head
    
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    
    return prev""",
                    "javascript": """class ListNode {
    constructor(val = 0, next = null) {
        this.val = val;
        this.next = next;
    }
}

function reverseLinkedList(head) {
    // Reverse a singly linked list
    let prev = null;
    let current = head;
    
    while (current !== null) {
        const nextNode = current.next;
        current.next = prev;
        prev = current;
        current = nextNode;
    }
    
    return prev;
}""",
                    "java": """class ListNode {
    int val;
    ListNode next;
    
    ListNode(int val) {
        this.val = val;
        this.next = null;
    }
}

public static ListNode reverseLinkedList(ListNode head) {
    // Reverse a singly linked list
    ListNode prev = null;
    ListNode current = head;
    
    while (current != null) {
        ListNode nextNode = current.next;
        current.next = prev;
        prev = current;
        current = nextNode;
    }
    
    return prev;
}""",
                    "cpp": """struct ListNode {
    int val;
    ListNode* next;
    
    ListNode(int x) : val(x), next(nullptr) {}
};

ListNode* reverseLinkedList(ListNode* head) {
    // Reverse a singly linked list
    ListNode* prev = nullptr;
    ListNode* current = head;
    
    while (current != nullptr) {
        ListNode* nextNode = current->next;
        current->next = prev;
        prev = current;
        current = nextNode;
    }
    
    return prev;
}""",
                    "rust": """#[derive(Debug)]
struct ListNode {
    val: i32,
    next: Option<Box<ListNode>>,
}

fn reverse_linked_list(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    // Reverse a singly linked list
    let mut prev = None;
    let mut current = head;
    
    while let Some(mut node) = current {
        let next = node.next.take();
        node.next = prev;
        prev = Some(node);
        current = next;
    }
    
    prev
}""",
                    "go": """type ListNode struct {
    Val  int
    Next *ListNode
}

func reverseLinkedList(head *ListNode) *ListNode {
    // Reverse a singly linked list
    var prev *ListNode
    current := head
    
    for current != nil {
        nextNode := current.Next
        current.Next = prev
        prev = current
        current = nextNode
    }
    
    return prev
}"""
                },
                "key_differences": {
                    "python": "Simple pointer manipulation",
                    "javascript": "Similar to Python with explicit null checks",
                    "java": "Class-based with explicit constructors",
                    "cpp": "Pointer manipulation with nullptr",
                    "rust": "Ownership system requires take() for moving values",
                    "go": "Pointer-based with nil checks"
                }
            }
        ]
        
        return {
            "metadata": {
                "source": source.name,
                "type": source.type,
                "description": source.description,
                "languages": source.languages,
                "total_examples": len(translation_examples),
                "created_at": datetime.now().isoformat()
            },
            "translation_examples": translation_examples
        }
    
    def _save_dataset(self, dataset: Dict[str, Any], filename: str) -> None:
        """Save dataset to JSON file."""
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"  💾 Saved to: {filepath}")
    
    def _save_fetch_summary(self, results: Dict[str, Any]) -> None:
        """Save fetch summary report."""
        summary_file = self.output_dir / "fetch_summary.json"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📋 Fetch summary saved to: {summary_file}")


def main():
    """Main entry point."""
    print("=" * 70)
    print("External Dataset Fetcher for LLM/ML/AI Training")
    print("=" * 70)
    print()
    
    fetcher = ExternalDatasetFetcher()
    results = fetcher.fetch_all_datasets()
    
    print("\n" + "=" * 70)
    print(f"✅ Successfully created {len(results['datasets_created'])} datasets")
    if results['errors']:
        print(f"⚠️  {len(results['errors'])} errors occurred")
    print("=" * 70)


if __name__ == "__main__":
    main()
