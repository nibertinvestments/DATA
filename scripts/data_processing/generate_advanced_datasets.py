#!/usr/bin/env python3
"""
Advanced Dataset Generator for LLM/ML/AI Training
Creates comprehensive training datasets for AI coding agents covering advanced topics.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


class AdvancedDatasetGenerator:
    """Generates advanced training datasets for AI agents."""
    
    def __init__(self, output_dir: str = "datasets/raw/external"):
        """Initialize generator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_all_datasets(self) -> None:
        """Generate all advanced datasets."""
        print("🚀 Generating advanced training datasets...\n")
        
        datasets = [
            ("design_patterns", self._create_design_patterns_dataset()),
            ("security_vulnerabilities", self._create_security_dataset()),
            ("performance_optimization", self._create_performance_dataset()),
            ("testing_strategies", self._create_testing_dataset()),
            ("database_patterns", self._create_database_patterns_dataset()),
            ("async_programming", self._create_async_patterns_dataset()),
        ]
        
        for name, dataset in datasets:
            filename = f"{name}_dataset.json"
            self._save_dataset(dataset, filename)
            print(f"✅ Created: {filename}")
        
        print(f"\n✅ Generated {len(datasets)} advanced datasets!")
    
    def _create_design_patterns_dataset(self) -> Dict[str, Any]:
        """Create comprehensive design patterns dataset."""
        
        patterns = [
            {
                "id": "pattern_singleton_001",
                "pattern_name": "Singleton Pattern",
                "category": "creational",
                "description": "Ensure a class has only one instance and provide global access",
                "use_case": "Database connections, logging, configuration managers",
                "implementations": {
                    "python": """class DatabaseConnection:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        self.connection = self._create_connection()
    
    def _create_connection(self):
        return "DB Connection"
    
    @classmethod
    def get_instance(cls):
        return cls()""",
                    "javascript": """class DatabaseConnection {
    static #instance = null;
    
    constructor() {
        if (DatabaseConnection.#instance) {
            return DatabaseConnection.#instance;
        }
        
        this.connection = this.createConnection();
        DatabaseConnection.#instance = this;
    }
    
    createConnection() {
        return 'DB Connection';
    }
    
    static getInstance() {
        if (!DatabaseConnection.#instance) {
            DatabaseConnection.#instance = new DatabaseConnection();
        }
        return DatabaseConnection.#instance;
    }
}""",
                    "java": """public class DatabaseConnection {
    private static volatile DatabaseConnection instance;
    private String connection;
    
    private DatabaseConnection() {
        this.connection = createConnection();
    }
    
    public static DatabaseConnection getInstance() {
        if (instance == null) {
            synchronized (DatabaseConnection.class) {
                if (instance == null) {
                    instance = new DatabaseConnection();
                }
            }
        }
        return instance;
    }
    
    private String createConnection() {
        return "DB Connection";
    }
}"""
                },
                "pros": [
                    "Controlled access to sole instance",
                    "Reduced namespace pollution",
                    "Lazy initialization possible"
                ],
                "cons": [
                    "Difficult to unit test",
                    "Global state can cause issues",
                    "Can hide dependencies"
                ]
            },
            {
                "id": "pattern_factory_001",
                "pattern_name": "Factory Pattern",
                "category": "creational",
                "description": "Create objects without specifying exact class",
                "use_case": "UI components, document parsers, database drivers",
                "implementations": {
                    "python": """from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return "Woof!"

class Cat(Animal):
    def speak(self):
        return "Meow!"

class AnimalFactory:
    @staticmethod
    def create_animal(animal_type):
        if animal_type == "dog":
            return Dog()
        elif animal_type == "cat":
            return Cat()
        else:
            raise ValueError(f"Unknown animal type: {animal_type}")

# Usage
factory = AnimalFactory()
dog = factory.create_animal("dog")
print(dog.speak())  # Woof!""",
                    "javascript": """class Animal {
    speak() {
        throw new Error('Method must be implemented');
    }
}

class Dog extends Animal {
    speak() {
        return 'Woof!';
    }
}

class Cat extends Animal {
    speak() {
        return 'Meow!';
    }
}

class AnimalFactory {
    static createAnimal(animalType) {
        switch(animalType) {
            case 'dog':
                return new Dog();
            case 'cat':
                return new Cat();
            default:
                throw new Error(`Unknown animal type: ${animalType}`);
        }
    }
}

// Usage
const dog = AnimalFactory.createAnimal('dog');
console.log(dog.speak()); // Woof!""",
                    "java": """interface Animal {
    String speak();
}

class Dog implements Animal {
    public String speak() {
        return "Woof!";
    }
}

class Cat implements Animal {
    public String speak() {
        return "Meow!";
    }
}

class AnimalFactory {
    public static Animal createAnimal(String animalType) {
        switch(animalType) {
            case "dog":
                return new Dog();
            case "cat":
                return new Cat();
            default:
                throw new IllegalArgumentException("Unknown animal type: " + animalType);
        }
    }
}

// Usage
Animal dog = AnimalFactory.createAnimal("dog");
System.out.println(dog.speak()); // Woof!"""
                },
                "pros": [
                    "Loose coupling between creator and products",
                    "Single responsibility principle",
                    "Open/closed principle"
                ],
                "cons": [
                    "Code can become more complex",
                    "Need many subclasses"
                ]
            },
            {
                "id": "pattern_observer_001",
                "pattern_name": "Observer Pattern",
                "category": "behavioral",
                "description": "Define subscription mechanism to notify multiple objects",
                "use_case": "Event handling, UI updates, pub/sub systems",
                "implementations": {
                    "python": """from typing import List, Protocol

class Observer(Protocol):
    def update(self, message: str) -> None:
        ...

class Subject:
    def __init__(self):
        self._observers: List[Observer] = []
    
    def attach(self, observer: Observer):
        if observer not in self._observers:
            self._observers.append(observer)
    
    def detach(self, observer: Observer):
        self._observers.remove(observer)
    
    def notify(self, message: str):
        for observer in self._observers:
            observer.update(message)

class ConcreteObserver:
    def __init__(self, name: str):
        self.name = name
    
    def update(self, message: str):
        print(f"{self.name} received: {message}")

# Usage
subject = Subject()
observer1 = ConcreteObserver("Observer 1")
observer2 = ConcreteObserver("Observer 2")

subject.attach(observer1)
subject.attach(observer2)
subject.notify("Hello Observers!")""",
                    "javascript": """class Subject {
    constructor() {
        this.observers = [];
    }
    
    attach(observer) {
        if (!this.observers.includes(observer)) {
            this.observers.push(observer);
        }
    }
    
    detach(observer) {
        const index = this.observers.indexOf(observer);
        if (index > -1) {
            this.observers.splice(index, 1);
        }
    }
    
    notify(message) {
        this.observers.forEach(observer => observer.update(message));
    }
}

class ConcreteObserver {
    constructor(name) {
        this.name = name;
    }
    
    update(message) {
        console.log(`${this.name} received: ${message}`);
    }
}

// Usage
const subject = new Subject();
const observer1 = new ConcreteObserver('Observer 1');
const observer2 = new ConcreteObserver('Observer 2');

subject.attach(observer1);
subject.attach(observer2);
subject.notify('Hello Observers!');""",
                    "java": """import java.util.ArrayList;
import java.util.List;

interface Observer {
    void update(String message);
}

class Subject {
    private List<Observer> observers = new ArrayList<>();
    
    public void attach(Observer observer) {
        if (!observers.contains(observer)) {
            observers.add(observer);
        }
    }
    
    public void detach(Observer observer) {
        observers.remove(observer);
    }
    
    public void notifyObservers(String message) {
        for (Observer observer : observers) {
            observer.update(message);
        }
    }
}

class ConcreteObserver implements Observer {
    private String name;
    
    public ConcreteObserver(String name) {
        this.name = name;
    }
    
    @Override
    public void update(String message) {
        System.out.println(name + " received: " + message);
    }
}

// Usage
Subject subject = new Subject();
Observer observer1 = new ConcreteObserver("Observer 1");
Observer observer2 = new ConcreteObserver("Observer 2");

subject.attach(observer1);
subject.attach(observer2);
subject.notifyObservers("Hello Observers!");"""
                },
                "pros": [
                    "Loose coupling between subject and observers",
                    "Open/closed principle",
                    "Dynamic relationships at runtime"
                ],
                "cons": [
                    "Observers notified in random order",
                    "Memory leaks if not properly detached"
                ]
            }
        ]
        
        return {
            "metadata": {
                "name": "design_patterns",
                "description": "Comprehensive design patterns for multiple languages",
                "total_patterns": len(patterns),
                "categories": ["creational", "structural", "behavioral"],
                "created_at": datetime.now().isoformat()
            },
            "patterns": patterns
        }
    
    def _create_security_dataset(self) -> Dict[str, Any]:
        """Create security vulnerabilities and fixes dataset."""
        
        vulnerabilities = [
            {
                "id": "sec_001",
                "vulnerability_type": "SQL Injection",
                "severity": "critical",
                "description": "User input directly concatenated into SQL queries",
                "vulnerable_code": {
                    "python": """def get_user(username):
    query = f"SELECT * FROM users WHERE username = '{username}'"
    cursor.execute(query)  # VULNERABLE!
    return cursor.fetchone()""",
                    "javascript": """function getUser(username) {
    const query = \`SELECT * FROM users WHERE username = '\${username}'\`;
    return db.query(query);  // VULNERABLE!
}""",
                    "java": """public User getUser(String username) {
    String query = "SELECT * FROM users WHERE username = '" + username + "'";
    Statement stmt = connection.createStatement();
    ResultSet rs = stmt.executeQuery(query);  // VULNERABLE!
    return parseUser(rs);
}"""
                },
                "fixed_code": {
                    "python": """def get_user(username):
    query = "SELECT * FROM users WHERE username = ?"
    cursor.execute(query, (username,))  # SAFE: Parameterized query
    return cursor.fetchone()""",
                    "javascript": """function getUser(username) {
    const query = 'SELECT * FROM users WHERE username = $1';
    return db.query(query, [username]);  // SAFE: Parameterized query
}""",
                    "java": """public User getUser(String username) {
    String query = "SELECT * FROM users WHERE username = ?";
    PreparedStatement stmt = connection.prepareStatement(query);
    stmt.setString(1, username);  // SAFE: Prepared statement
    ResultSet rs = stmt.executeQuery();
    return parseUser(rs);
}"""
                },
                "impact": "Attacker can execute arbitrary SQL commands, read/modify data, or compromise entire database",
                "prevention": [
                    "Use parameterized queries/prepared statements",
                    "Input validation and sanitization",
                    "Use ORM frameworks",
                    "Principle of least privilege for database accounts"
                ]
            },
            {
                "id": "sec_002",
                "vulnerability_type": "Cross-Site Scripting (XSS)",
                "severity": "high",
                "description": "Untrusted data included in HTML without escaping",
                "vulnerable_code": {
                    "python": """from flask import Flask, request

@app.route('/greet')
def greet():
    name = request.args.get('name', '')
    return f'<h1>Hello {name}!</h1>'  # VULNERABLE!""",
                    "javascript": """function displayUserName(name) {
    document.getElementById('greeting').innerHTML = 
        '<h1>Hello ' + name + '!</h1>';  // VULNERABLE!
}""",
                    "java": """@GetMapping("/greet")
public String greet(@RequestParam String name) {
    return "<h1>Hello " + name + "!</h1>";  // VULNERABLE!
}"""
                },
                "fixed_code": {
                    "python": """from flask import Flask, request, escape

@app.route('/greet')
def greet():
    name = request.args.get('name', '')
    return f'<h1>Hello {escape(name)}!</h1>'  # SAFE: Escaped output""",
                    "javascript": """function displayUserName(name) {
    const greeting = document.getElementById('greeting');
    greeting.textContent = 'Hello ' + name + '!';  // SAFE: textContent escapes
}""",
                    "java": """@GetMapping("/greet")
public String greet(@RequestParam String name) {
    return "<h1>Hello " + 
        StringEscapeUtils.escapeHtml4(name) + 
        "!</h1>";  // SAFE: Escaped output
}"""
                },
                "impact": "Attacker can inject malicious scripts, steal session cookies, deface website, or redirect users",
                "prevention": [
                    "Escape all user input before rendering in HTML",
                    "Use Content Security Policy (CSP)",
                    "Use frameworks that auto-escape by default",
                    "Validate and sanitize input"
                ]
            },
            {
                "id": "sec_003",
                "vulnerability_type": "Path Traversal",
                "severity": "high",
                "description": "User-controlled file paths without validation",
                "vulnerable_code": {
                    "python": """def read_file(filename):
    path = f'/var/data/{filename}'
    with open(path, 'r') as f:  # VULNERABLE!
        return f.read()
# User can pass '../../../etc/passwd' to access system files""",
                    "javascript": """const fs = require('fs');

function readFile(filename) {
    const path = \`/var/data/\${filename}\`;
    return fs.readFileSync(path, 'utf8');  // VULNERABLE!
}""",
                    "java": """public String readFile(String filename) {
    String path = "/var/data/" + filename;
    return Files.readString(Paths.get(path));  // VULNERABLE!
}"""
                },
                "fixed_code": {
                    "python": """import os
from pathlib import Path

def read_file(filename):
    # Validate filename - no path separators
    if '/' in filename or '\\\\' in filename or '..' in filename:
        raise ValueError("Invalid filename")
    
    base_dir = Path('/var/data')
    file_path = (base_dir / filename).resolve()
    
    # Ensure file is within base directory
    if not str(file_path).startswith(str(base_dir)):
        raise ValueError("Access denied")
    
    with open(file_path, 'r') as f:
        return f.read()""",
                    "javascript": """const fs = require('fs');
const path = require('path');

function readFile(filename) {
    // Validate filename - no path separators
    if (filename.includes('/') || filename.includes('\\\\') || filename.includes('..')) {
        throw new Error('Invalid filename');
    }
    
    const baseDir = '/var/data';
    const filePath = path.resolve(baseDir, filename);
    
    // Ensure file is within base directory
    if (!filePath.startsWith(baseDir)) {
        throw new Error('Access denied');
    }
    
    return fs.readFileSync(filePath, 'utf8');
}""",
                    "java": """import java.nio.file.*;

public String readFile(String filename) throws IOException {
    // Validate filename - no path separators
    if (filename.contains("/") || filename.contains("\\\\") || filename.contains("..")) {
        throw new IllegalArgumentException("Invalid filename");
    }
    
    Path baseDir = Paths.get("/var/data");
    Path filePath = baseDir.resolve(filename).normalize();
    
    // Ensure file is within base directory
    if (!filePath.startsWith(baseDir)) {
        throw new SecurityException("Access denied");
    }
    
    return Files.readString(filePath);
}"""
                },
                "impact": "Attacker can read sensitive files, access system files, or overwrite critical data",
                "prevention": [
                    "Never use user input directly in file paths",
                    "Whitelist allowed filenames",
                    "Use path normalization and validation",
                    "Implement proper access controls"
                ]
            }
        ]
        
        return {
            "metadata": {
                "name": "security_vulnerabilities",
                "description": "Common security vulnerabilities and their fixes",
                "total_vulnerabilities": len(vulnerabilities),
                "severity_levels": ["critical", "high", "medium", "low"],
                "created_at": datetime.now().isoformat()
            },
            "vulnerabilities": vulnerabilities
        }
    
    def _create_performance_dataset(self) -> Dict[str, Any]:
        """Create performance optimization patterns dataset."""
        
        optimizations = [
            {
                "id": "perf_001",
                "optimization_type": "Algorithmic Complexity",
                "problem": "Finding duplicates in array",
                "slow_implementation": {
                    "python": """def has_duplicates_slow(arr):
    \"\"\"O(n²) time complexity\"\"\"
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] == arr[j]:
                return True
    return False""",
                    "javascript": """function hasDuplicatesSlow(arr) {
    // O(n²) time complexity
    for (let i = 0; i < arr.length; i++) {
        for (let j = i + 1; j < arr.length; j++) {
            if (arr[i] === arr[j]) {
                return true;
            }
        }
    }
    return false;
}"""
                },
                "optimized_implementation": {
                    "python": """def has_duplicates_fast(arr):
    \"\"\"O(n) time complexity\"\"\"
    seen = set()
    for item in arr:
        if item in seen:
            return True
        seen.add(item)
    return False""",
                    "javascript": """function hasDuplicatesFast(arr) {
    // O(n) time complexity
    const seen = new Set();
    for (const item of arr) {
        if (seen.has(item)) {
            return true;
        }
        seen.add(item);
    }
    return false;
}"""
                },
                "performance_gain": "From O(n²) to O(n) - 100x faster for large arrays",
                "explanation": "Using a Set for O(1) lookup instead of nested loops"
            },
            {
                "id": "perf_002",
                "optimization_type": "Memory Efficiency",
                "problem": "Processing large files",
                "slow_implementation": {
                    "python": """def process_large_file_slow(filename):
    \"\"\"Loads entire file into memory\"\"\"
    with open(filename, 'r') as f:
        content = f.read()  # Loads everything at once
        lines = content.split('\\n')
        return [line.upper() for line in lines]"""
                },
                "optimized_implementation": {
                    "python": """def process_large_file_fast(filename):
    \"\"\"Streams file line by line\"\"\"
    result = []
    with open(filename, 'r') as f:
        for line in f:  # Reads one line at a time
            result.append(line.strip().upper())
    return result
    
# Even better: Use generator
def process_large_file_generator(filename):
    \"\"\"Memory-efficient generator\"\"\"
    with open(filename, 'r') as f:
        for line in f:
            yield line.strip().upper()"""
                },
                "performance_gain": "Memory usage reduced from O(file size) to O(1)",
                "explanation": "Processing data in chunks instead of loading everything"
            },
            {
                "id": "perf_003",
                "optimization_type": "Caching",
                "problem": "Repeated expensive computations",
                "slow_implementation": {
                    "python": """def fibonacci_slow(n):
    \"\"\"Exponential time complexity O(2^n)\"\"\"
    if n <= 1:
        return n
    return fibonacci_slow(n-1) + fibonacci_slow(n-2)"""
                },
                "optimized_implementation": {
                    "python": """from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci_fast(n):
    \"\"\"Linear time complexity O(n) with memoization\"\"\"
    if n <= 1:
        return n
    return fibonacci_fast(n-1) + fibonacci_fast(n-2)
    
# Or use dynamic programming
def fibonacci_dp(n):
    \"\"\"O(n) time, O(1) space\"\"\"
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b"""
                },
                "performance_gain": "From exponential O(2^n) to linear O(n)",
                "explanation": "Caching results to avoid redundant calculations"
            }
        ]
        
        return {
            "metadata": {
                "name": "performance_optimization",
                "description": "Performance optimization patterns and techniques",
                "total_optimizations": len(optimizations),
                "categories": ["algorithmic", "memory", "caching", "database"],
                "created_at": datetime.now().isoformat()
            },
            "optimizations": optimizations
        }
    
    def _create_testing_dataset(self) -> Dict[str, Any]:
        """Create testing strategies dataset."""
        
        testing_examples = [
            {
                "id": "test_001",
                "test_type": "Unit Testing",
                "description": "Testing individual functions in isolation",
                "code_example": {
                    "python": """import unittest

class Calculator:
    def add(self, a, b):
        return a + b
    
    def divide(self, a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

class TestCalculator(unittest.TestCase):
    def setUp(self):
        self.calc = Calculator()
    
    def test_add(self):
        self.assertEqual(self.calc.add(2, 3), 5)
        self.assertEqual(self.calc.add(-1, 1), 0)
    
    def test_divide(self):
        self.assertEqual(self.calc.divide(10, 2), 5)
        with self.assertRaises(ValueError):
            self.calc.divide(10, 0)

if __name__ == '__main__':
    unittest.main()""",
                    "javascript": """const assert = require('assert');

class Calculator {
    add(a, b) {
        return a + b;
    }
    
    divide(a, b) {
        if (b === 0) {
            throw new Error('Cannot divide by zero');
        }
        return a / b;
    }
}

describe('Calculator', () => {
    let calc;
    
    beforeEach(() => {
        calc = new Calculator();
    });
    
    it('should add two numbers', () => {
        assert.strictEqual(calc.add(2, 3), 5);
        assert.strictEqual(calc.add(-1, 1), 0);
    });
    
    it('should divide two numbers', () => {
        assert.strictEqual(calc.divide(10, 2), 5);
        assert.throws(() => calc.divide(10, 0), Error);
    });
});"""
                },
                "best_practices": [
                    "Test one thing at a time",
                    "Use descriptive test names",
                    "Follow AAA pattern: Arrange, Act, Assert",
                    "Keep tests independent"
                ]
            },
            {
                "id": "test_002",
                "test_type": "Integration Testing",
                "description": "Testing interaction between components",
                "code_example": {
                    "python": """import unittest
from unittest.mock import Mock, patch

class UserService:
    def __init__(self, db, email_service):
        self.db = db
        self.email_service = email_service
    
    def create_user(self, email, name):
        # Save to database
        user_id = self.db.save_user({'email': email, 'name': name})
        
        # Send welcome email
        self.email_service.send_welcome_email(email, name)
        
        return user_id

class TestUserServiceIntegration(unittest.TestCase):
    def test_create_user_integration(self):
        # Mock dependencies
        mock_db = Mock()
        mock_db.save_user.return_value = 123
        
        mock_email = Mock()
        
        # Test service with mocked dependencies
        service = UserService(mock_db, mock_email)
        user_id = service.create_user('test@example.com', 'Test User')
        
        # Verify interactions
        self.assertEqual(user_id, 123)
        mock_db.save_user.assert_called_once()
        mock_email.send_welcome_email.assert_called_once_with(
            'test@example.com', 'Test User'
        )"""
                },
                "best_practices": [
                    "Use mocks for external dependencies",
                    "Test component interactions",
                    "Verify method calls and data flow",
                    "Test error handling between components"
                ]
            }
        ]
        
        return {
            "metadata": {
                "name": "testing_strategies",
                "description": "Comprehensive testing strategies and patterns",
                "total_examples": len(testing_examples),
                "test_types": ["unit", "integration", "e2e", "performance"],
                "created_at": datetime.now().isoformat()
            },
            "testing_examples": testing_examples
        }
    
    def _create_database_patterns_dataset(self) -> Dict[str, Any]:
        """Create database patterns and best practices dataset."""
        
        patterns = [
            {
                "id": "db_001",
                "pattern_name": "Connection Pooling",
                "description": "Reuse database connections efficiently",
                "problem": "Creating new connection for each query is expensive",
                "solution": {
                    "python": """from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

# Create engine with connection pooling
engine = create_engine(
    'postgresql://user:pass@localhost/db',
    poolclass=QueuePool,
    pool_size=10,  # Keep 10 connections
    max_overflow=20,  # Allow 20 additional connections if needed
    pool_pre_ping=True,  # Verify connection before use
    pool_recycle=3600  # Recycle connections after 1 hour
)

# Use connection from pool
with engine.connect() as conn:
    result = conn.execute("SELECT * FROM users")
    # Connection automatically returned to pool"""
                },
                "benefits": [
                    "Reduced connection overhead",
                    "Better resource utilization",
                    "Improved application performance",
                    "Automatic connection management"
                ]
            },
            {
                "id": "db_002",
                "pattern_name": "Batch Operations",
                "description": "Process multiple records efficiently",
                "problem": "Individual inserts are slow for bulk data",
                "comparison": {
                    "slow": """# Slow: Individual inserts
for user in users:
    cursor.execute(
        "INSERT INTO users (name, email) VALUES (?, ?)",
        (user['name'], user['email'])
    )
    conn.commit()  # Commit each insert""",
                    "fast": """# Fast: Batch insert
cursor.executemany(
    "INSERT INTO users (name, email) VALUES (?, ?)",
    [(u['name'], u['email']) for u in users]
)
conn.commit()  # Single commit for all"""
                },
                "performance_gain": "10-100x faster for large datasets"
            }
        ]
        
        return {
            "metadata": {
                "name": "database_patterns",
                "description": "Database design patterns and optimization techniques",
                "total_patterns": len(patterns),
                "created_at": datetime.now().isoformat()
            },
            "patterns": patterns
        }
    
    def _create_async_patterns_dataset(self) -> Dict[str, Any]:
        """Create async programming patterns dataset."""
        
        patterns = [
            {
                "id": "async_001",
                "pattern_name": "Concurrent API Calls",
                "description": "Make multiple API calls concurrently",
                "sequential_approach": {
                    "python": """import requests

def fetch_users_sequential(user_ids):
    \"\"\"Slow: Sequential fetching\"\"\"
    users = []
    for user_id in user_ids:
        response = requests.get(f'https://api.example.com/users/{user_id}')
        users.append(response.json())
    return users
# Time: n * request_time""",
                    "javascript": """async function fetchUsersSequential(userIds) {
    // Slow: Sequential fetching
    const users = [];
    for (const userId of userIds) {
        const response = await fetch(\`https://api.example.com/users/\${userId}\`);
        const user = await response.json();
        users.push(user);
    }
    return users;
}
// Time: n * request_time"""
                },
                "concurrent_approach": {
                    "python": """import asyncio
import aiohttp

async def fetch_user(session, user_id):
    async with session.get(f'https://api.example.com/users/{user_id}') as response:
        return await response.json()

async def fetch_users_concurrent(user_ids):
    \"\"\"Fast: Concurrent fetching\"\"\"
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_user(session, user_id) for user_id in user_ids]
        users = await asyncio.gather(*tasks)
    return users
# Time: ~request_time (all parallel)""",
                    "javascript": """async function fetchUsersConcurrent(userIds) {
    // Fast: Concurrent fetching
    const promises = userIds.map(userId =>
        fetch(\`https://api.example.com/users/\${userId}\`)
            .then(response => response.json())
    );
    const users = await Promise.all(promises);
    return users;
}
// Time: ~request_time (all parallel)"""
                },
                "performance_gain": "n times faster for n requests"
            }
        ]
        
        return {
            "metadata": {
                "name": "async_programming",
                "description": "Asynchronous programming patterns and best practices",
                "total_patterns": len(patterns),
                "created_at": datetime.now().isoformat()
            },
            "patterns": patterns
        }
    
    def _save_dataset(self, dataset: Dict[str, Any], filename: str) -> None:
        """Save dataset to JSON file."""
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)


def main():
    """Main entry point."""
    print("=" * 70)
    print("Advanced Dataset Generator for LLM/ML/AI Training")
    print("=" * 70)
    print()
    
    generator = AdvancedDatasetGenerator()
    generator.generate_all_datasets()
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
