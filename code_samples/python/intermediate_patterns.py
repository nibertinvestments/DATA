"""
Intermediate Python Programming Examples
========================================

This module demonstrates intermediate Python concepts including:
- Advanced object-oriented programming with metaclasses
- Context managers and decorators
- Generators and iterators
- Async/await patterns
- Design patterns implementation
- Performance optimization techniques
- Testing strategies and mocking
"""

import asyncio
import functools
import threading
import time
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Protocol, TypeVar, Generic
from unittest.mock import Mock, patch
import json
import sqlite3


# Advanced Object-Oriented Programming
# ===================================

class SingletonMeta(type):
    """Metaclass for implementing Singleton pattern"""
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class DatabaseConnection(metaclass=SingletonMeta):
    """Singleton database connection manager"""
    
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._connection = None
        self._setup_database()
    
    def _setup_database(self):
        """Initialize database tables"""
        self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = self._connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._connection.commit()
    
    def execute(self, query: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute SQL query safely"""
        return self._connection.execute(query, params)
    
    def commit(self):
        """Commit transaction"""
        self._connection.commit()
    
    def close(self):
        """Close database connection"""
        if self._connection:
            self._connection.close()


# Design Patterns Implementation
# =============================

class Observer(ABC):
    """Observer pattern interface"""
    
    @abstractmethod
    def update(self, subject: 'Subject') -> None:
        pass


class Subject:
    """Subject class for Observer pattern"""
    
    def __init__(self):
        self._observers: List[Observer] = []
        self._state = None
    
    def attach(self, observer: Observer) -> None:
        """Attach an observer"""
        if observer not in self._observers:
            self._observers.append(observer)
    
    def detach(self, observer: Observer) -> None:
        """Detach an observer"""
        self._observers.remove(observer)
    
    def notify(self) -> None:
        """Notify all observers"""
        for observer in self._observers:
            observer.update(self)
    
    @property
    def state(self) -> Any:
        return self._state
    
    @state.setter
    def state(self, value: Any) -> None:
        self._state = value
        self.notify()


class EmailNotifier(Observer):
    """Concrete observer for email notifications"""
    
    def __init__(self, email: str):
        self.email = email
    
    def update(self, subject: Subject) -> None:
        print(f"Email to {self.email}: State changed to {subject.state}")


class SMSNotifier(Observer):
    """Concrete observer for SMS notifications"""
    
    def __init__(self, phone: str):
        self.phone = phone
    
    def update(self, subject: Subject) -> None:
        print(f"SMS to {self.phone}: State changed to {subject.state}")


# Strategy Pattern
class SortStrategy(ABC):
    """Abstract base class for sorting strategies"""
    
    @abstractmethod
    def sort(self, data: List[Any]) -> List[Any]:
        pass


class QuickSortStrategy(SortStrategy):
    """Quick sort implementation"""
    
    def sort(self, data: List[Any]) -> List[Any]:
        if len(data) <= 1:
            return data
        
        pivot = data[len(data) // 2]
        left = [x for x in data if x < pivot]
        middle = [x for x in data if x == pivot]
        right = [x for x in data if x > pivot]
        
        return self.sort(left) + middle + self.sort(right)


class MergeSortStrategy(SortStrategy):
    """Merge sort implementation"""
    
    def sort(self, data: List[Any]) -> List[Any]:
        if len(data) <= 1:
            return data
        
        mid = len(data) // 2
        left = self.sort(data[:mid])
        right = self.sort(data[mid:])
        
        return self._merge(left, right)
    
    def _merge(self, left: List[Any], right: List[Any]) -> List[Any]:
        result = []
        i = j = 0
        
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        
        result.extend(left[i:])
        result.extend(right[j:])
        return result


class SortContext:
    """Context class for sorting strategies"""
    
    def __init__(self, strategy: SortStrategy):
        self._strategy = strategy
    
    def set_strategy(self, strategy: SortStrategy) -> None:
        self._strategy = strategy
    
    def sort_data(self, data: List[Any]) -> List[Any]:
        return self._strategy.sort(data.copy())


# Advanced Decorators and Context Managers
# ========================================

def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator for retrying function calls with exponential backoff"""
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 1
            current_delay = delay
            
            while attempt <= max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts:
                        raise
                    
                    print(f"Attempt {attempt} failed: {e}. Retrying in {current_delay}s...")
                    time.sleep(current_delay)
                    current_delay *= backoff
                    attempt += 1
            
        return wrapper
    return decorator


def timing(func):
    """Decorator to measure function execution time"""
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.perf_counter()
            print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
    
    return wrapper


def memoize(func):
    """Decorator for memoization/caching"""
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create a key from args and kwargs
        key = str(args) + str(sorted(kwargs.items()))
        
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    
    return wrapper


@contextmanager
def transaction_manager(connection):
    """Context manager for database transactions"""
    cursor = connection.cursor()
    try:
        yield cursor
        connection.commit()
        print("Transaction committed successfully")
    except Exception as e:
        connection.rollback()
        print(f"Transaction rolled back due to error: {e}")
        raise
    finally:
        cursor.close()


@contextmanager
def performance_monitor(name: str):
    """Context manager for performance monitoring"""
    start_time = time.perf_counter()
    start_memory = 0  # Would use psutil in real implementation
    
    print(f"Starting {name}...")
    try:
        yield
    finally:
        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f"{name} completed in {duration:.4f} seconds")


# Advanced Generators and Iterators
# =================================

class FibonacciGenerator:
    """Iterator class for Fibonacci sequence"""
    
    def __init__(self, max_count: Optional[int] = None):
        self.max_count = max_count
        self.count = 0
        self.current = 0
        self.next_val = 1
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.max_count and self.count >= self.max_count:
            raise StopIteration
        
        result = self.current
        self.current, self.next_val = self.next_val, self.current + self.next_val
        self.count += 1
        return result


def sliding_window(iterable, window_size: int):
    """Generator for sliding window over an iterable"""
    iterator = iter(iterable)
    window = []
    
    # Fill initial window
    for _ in range(window_size):
        try:
            window.append(next(iterator))
        except StopIteration:
            return
    
    yield tuple(window)
    
    # Slide the window
    for item in iterator:
        window.pop(0)
        window.append(item)
        yield tuple(window)


def batch_processor(iterable, batch_size: int):
    """Generator that yields batches of items"""
    iterator = iter(iterable)
    while True:
        batch = []
        for _ in range(batch_size):
            try:
                batch.append(next(iterator))
            except StopIteration:
                if batch:
                    yield batch
                return
        yield batch


def prime_generator(limit: Optional[int] = None):
    """Generator for prime numbers using Sieve of Eratosthenes"""
    if limit is None:
        limit = float('inf')
    
    primes = []
    candidate = 2
    
    while candidate <= limit:
        is_prime = True
        for prime in primes:
            if prime * prime > candidate:
                break
            if candidate % prime == 0:
                is_prime = False
                break
        
        if is_prime:
            primes.append(candidate)
            yield candidate
        
        candidate += 1


# Async/Await Programming
# ======================

class AsyncHTTPClient:
    """Asynchronous HTTP client simulation"""
    
    def __init__(self):
        self.session_data = {}
    
    async def fetch(self, url: str, delay: float = 1.0) -> dict:
        """Simulate async HTTP request"""
        print(f"Fetching {url}...")
        await asyncio.sleep(delay)  # Simulate network delay
        
        return {
            'url': url,
            'status': 200,
            'data': f'Data from {url}',
            'timestamp': time.time()
        }
    
    async def fetch_multiple(self, urls: List[str]) -> List[dict]:
        """Fetch multiple URLs concurrently"""
        tasks = [self.fetch(url) for url in urls]
        return await asyncio.gather(*tasks)
    
    async def fetch_with_semaphore(self, urls: List[str], max_concurrent: int = 5) -> List[dict]:
        """Fetch URLs with concurrency limit"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def fetch_limited(url):
            async with semaphore:
                return await self.fetch(url)
        
        tasks = [fetch_limited(url) for url in urls]
        return await asyncio.gather(*tasks)


class AsyncTaskQueue:
    """Asynchronous task queue implementation"""
    
    def __init__(self, max_workers: int = 5):
        self.queue = asyncio.Queue()
        self.max_workers = max_workers
        self.workers = []
        self.results = {}
    
    async def add_task(self, task_id: str, coro):
        """Add a task to the queue"""
        await self.queue.put((task_id, coro))
    
    async def worker(self, worker_id: int):
        """Worker coroutine to process tasks"""
        while True:
            try:
                task_id, coro = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                print(f"Worker {worker_id} processing task {task_id}")
                
                try:
                    result = await coro
                    self.results[task_id] = {'status': 'success', 'result': result}
                except Exception as e:
                    self.results[task_id] = {'status': 'error', 'error': str(e)}
                finally:
                    self.queue.task_done()
                    
            except asyncio.TimeoutError:
                # No tasks available, continue waiting
                continue
    
    async def start_workers(self):
        """Start worker coroutines"""
        self.workers = [
            asyncio.create_task(self.worker(i)) 
            for i in range(self.max_workers)
        ]
    
    async def stop_workers(self):
        """Stop all workers"""
        for worker in self.workers:
            worker.cancel()
        
        await asyncio.gather(*self.workers, return_exceptions=True)
    
    async def wait_for_completion(self):
        """Wait for all tasks to complete"""
        await self.queue.join()


# Data Classes and Type Annotations
# =================================

@dataclass
class User:
    """User data class with validation"""
    id: int
    name: str
    email: str
    age: Optional[int] = None
    preferences: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Validate data after initialization"""
        if '@' not in self.email:
            raise ValueError("Invalid email format")
        
        if self.age is not None and self.age < 0:
            raise ValueError("Age cannot be negative")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'age': self.age,
            'preferences': self.preferences,
            'created_at': self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        """Create instance from dictionary"""
        return cls(**data)


class UserStatus(Enum):
    """User status enumeration"""
    ACTIVE = auto()
    INACTIVE = auto()
    SUSPENDED = auto()
    DELETED = auto()


T = TypeVar('T')

class Cache(Generic[T]):
    """Generic cache implementation"""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._cache: Dict[str, T] = {}
        self._access_order: List[str] = []
    
    def get(self, key: str) -> Optional[T]:
        """Get item from cache"""
        if key in self._cache:
            # Move to end (most recently used)
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None
    
    def put(self, key: str, value: T) -> None:
        """Put item in cache"""
        if key in self._cache:
            self._access_order.remove(key)
        elif len(self._cache) >= self.max_size:
            # Remove least recently used
            lru_key = self._access_order.pop(0)
            del self._cache[lru_key]
        
        self._cache[key] = value
        self._access_order.append(key)
    
    def size(self) -> int:
        """Get cache size"""
        return len(self._cache)
    
    def clear(self) -> None:
        """Clear cache"""
        self._cache.clear()
        self._access_order.clear()


# Testing and Mocking Utilities
# =============================

class UserService:
    """Service class for user management"""
    
    def __init__(self, database: DatabaseConnection):
        self.database = database
    
    def create_user(self, name: str, email: str) -> int:
        """Create a new user"""
        cursor = self.database.execute(
            "INSERT INTO users (name, email) VALUES (?, ?)",
            (name, email)
        )
        self.database.commit()
        return cursor.lastrowid
    
    def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        cursor = self.database.execute(
            "SELECT id, name, email, created_at FROM users WHERE id = ?",
            (user_id,)
        )
        row = cursor.fetchone()
        
        if row:
            return {
                'id': row[0],
                'name': row[1],
                'email': row[2],
                'created_at': row[3]
            }
        return None
    
    def update_user(self, user_id: int, **kwargs) -> bool:
        """Update user information"""
        if not kwargs:
            return False
        
        set_clause = ', '.join(f"{key} = ?" for key in kwargs.keys())
        values = list(kwargs.values()) + [user_id]
        
        cursor = self.database.execute(
            f"UPDATE users SET {set_clause} WHERE id = ?",
            values
        )
        self.database.commit()
        return cursor.rowcount > 0


def test_user_service():
    """Example test using mocking"""
    # Mock the database
    mock_db = Mock()
    mock_cursor = Mock()
    mock_db.execute.return_value = mock_cursor
    mock_cursor.lastrowid = 123
    
    # Test user creation
    service = UserService(mock_db)
    user_id = service.create_user("John Doe", "john@example.com")
    
    assert user_id == 123
    mock_db.execute.assert_called_with(
        "INSERT INTO users (name, email) VALUES (?, ?)",
        ("John Doe", "john@example.com")
    )
    mock_db.commit.assert_called_once()
    
    print("✅ User service test passed")


# Performance Optimization Examples
# =================================

class PerformanceOptimizer:
    """Collection of performance optimization techniques"""
    
    @staticmethod
    @timing
    def list_comprehension_vs_loop(n: int):
        """Compare list comprehension vs traditional loop"""
        
        # List comprehension
        start = time.perf_counter()
        squares_lc = [x**2 for x in range(n)]
        lc_time = time.perf_counter() - start
        
        # Traditional loop
        start = time.perf_counter()
        squares_loop = []
        for x in range(n):
            squares_loop.append(x**2)
        loop_time = time.perf_counter() - start
        
        print(f"List comprehension: {lc_time:.4f}s")
        print(f"Traditional loop: {loop_time:.4f}s")
        print(f"Speedup: {loop_time/lc_time:.2f}x")
        
        return squares_lc
    
    @staticmethod
    def memory_efficient_processing(data: List[Any]):
        """Example of memory-efficient data processing"""
        
        # Generator expression (memory efficient)
        def process_with_generator():
            return (x**2 for x in data if x % 2 == 0)
        
        # List comprehension (loads all into memory)
        def process_with_list():
            return [x**2 for x in data if x % 2 == 0]
        
        # Process in chunks
        def process_in_chunks(chunk_size=1000):
            for chunk in batch_processor(data, chunk_size):
                yield [x**2 for x in chunk if x % 2 == 0]
        
        return {
            'generator': process_with_generator(),
            'list': process_with_list(),
            'chunks': process_in_chunks()
        }


# Main Example Usage
# ==================

def main():
    """Demonstrate all intermediate Python concepts"""
    print("=== Intermediate Python Programming Examples ===\n")
    
    # 1. Singleton Pattern
    print("1. Singleton Database Connection:")
    db1 = DatabaseConnection()
    db2 = DatabaseConnection()
    print(f"   Same instance: {db1 is db2}")
    
    # 2. Observer Pattern
    print("\n2. Observer Pattern:")
    subject = Subject()
    email_observer = EmailNotifier("user@example.com")
    sms_observer = SMSNotifier("+1234567890")
    
    subject.attach(email_observer)
    subject.attach(sms_observer)
    subject.state = "Important Update"
    
    # 3. Strategy Pattern
    print("\n3. Strategy Pattern:")
    data = [64, 34, 25, 12, 22, 11, 90]
    
    context = SortContext(QuickSortStrategy())
    quick_sorted = context.sort_data(data)
    print(f"   Quick sort: {quick_sorted}")
    
    context.set_strategy(MergeSortStrategy())
    merge_sorted = context.sort_data(data)
    print(f"   Merge sort: {merge_sorted}")
    
    # 4. Decorators
    print("\n4. Decorators:")
    
    @retry(max_attempts=2, delay=0.1)
    @timing
    @memoize
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)
    
    result = fibonacci(10)
    print(f"   Fibonacci(10) = {result}")
    
    # 5. Generators
    print("\n5. Generators:")
    fib_gen = FibonacciGenerator(max_count=10)
    fib_sequence = list(fib_gen)
    print(f"   Fibonacci sequence: {fib_sequence}")
    
    window_gen = sliding_window([1, 2, 3, 4, 5], 3)
    windows = list(window_gen)
    print(f"   Sliding windows: {windows}")
    
    primes = list(prime_generator(30))
    print(f"   Primes up to 30: {primes}")
    
    # 6. Async Programming
    print("\n6. Async Programming:")
    
    async def async_demo():
        client = AsyncHTTPClient()
        urls = [f"https://api{i}.example.com" for i in range(1, 4)]
        
        results = await client.fetch_multiple(urls)
        print(f"   Fetched {len(results)} URLs concurrently")
        
        # Task queue demo
        queue = AsyncTaskQueue(max_workers=2)
        await queue.start_workers()
        
        # Add some tasks
        for i in range(5):
            await queue.add_task(f"task_{i}", asyncio.sleep(0.1))
        
        await queue.wait_for_completion()
        await queue.stop_workers()
        print(f"   Processed {len(queue.results)} tasks")
    
    # Run async demo
    asyncio.run(async_demo())
    
    # 7. Data Classes
    print("\n7. Data Classes:")
    user = User(
        id=1,
        name="John Doe",
        email="john@example.com",
        age=30,
        preferences={"theme": "dark", "notifications": True}
    )
    print(f"   User: {user.name} ({user.email})")
    print(f"   Created at: {user.created_at}")
    
    # 8. Generic Cache
    print("\n8. Generic Cache:")
    cache = Cache[str](max_size=3)
    cache.put("key1", "value1")
    cache.put("key2", "value2")
    cache.put("key3", "value3")
    cache.put("key4", "value4")  # Should evict key1
    
    print(f"   Cache size: {cache.size()}")
    print(f"   Get key1: {cache.get('key1')}")  # Should be None
    print(f"   Get key2: {cache.get('key2')}")  # Should return value2
    
    # 9. Testing
    print("\n9. Testing with Mocks:")
    test_user_service()
    
    # 10. Performance Optimization
    print("\n10. Performance Optimization:")
    optimizer = PerformanceOptimizer()
    optimizer.list_comprehension_vs_loop(100000)
    
    print("\n=== Intermediate Python Demo Complete ===")


if __name__ == "__main__":
    main()