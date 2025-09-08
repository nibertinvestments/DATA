"""
Python Design Patterns - Singleton Pattern

This module demonstrates the Singleton pattern implementation in Python
with thread safety and best practices.
"""

import threading
from typing import Optional, Any


class SingletonMeta(type):
    """
    Thread-safe Singleton metaclass implementation.
    
    This metaclass ensures that only one instance of a class exists
    across the entire application, even in multi-threaded environments.
    """
    _instances = {}
    _lock: threading.Lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


class DatabaseConnection(metaclass=SingletonMeta):
    """
    Singleton database connection class.
    
    This class ensures only one database connection exists throughout
    the application lifecycle.
    """
    
    def __init__(self, host: str = "localhost", port: int = 5432):
        """Initialize database connection parameters."""
        if hasattr(self, '_initialized'):
            return
        
        self.host = host
        self.port = port
        self._connection = None
        self._initialized = True
        print(f"Database connection initialized: {host}:{port}")
    
    def connect(self) -> str:
        """Establish database connection."""
        if self._connection is None:
            self._connection = f"Connected to {self.host}:{self.port}"
            print("Database connection established")
        return self._connection
    
    def disconnect(self) -> None:
        """Close database connection."""
        if self._connection:
            print("Database connection closed")
            self._connection = None
    
    def execute_query(self, query: str) -> str:
        """Execute a database query."""
        if not self._connection:
            self.connect()
        return f"Executing query: {query}"


class ConfigManager(metaclass=SingletonMeta):
    """
    Singleton configuration manager.
    
    Manages application configuration settings with thread-safe access.
    """
    
    def __init__(self):
        """Initialize configuration manager."""
        if hasattr(self, '_initialized'):
            return
        
        self._config = {
            "debug": False,
            "log_level": "INFO",
            "max_connections": 100
        }
        self._initialized = True
        print("Configuration manager initialized")
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)
    
    def set_config(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self._config[key] = value
        print(f"Config updated: {key} = {value}")
    
    def get_all_config(self) -> dict:
        """Get all configuration settings."""
        return self._config.copy()


# Alternative Singleton implementation using decorator
def singleton(cls):
    """
    Singleton decorator implementation.
    
    A simpler alternative to metaclass approach for creating singletons.
    """
    instances = {}
    lock = threading.Lock()
    
    def get_instance(*args, **kwargs):
        if cls not in instances:
            with lock:
                if cls not in instances:
                    instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance


@singleton
class Logger:
    """
    Singleton logger class using decorator pattern.
    """
    
    def __init__(self, log_file: str = "app.log"):
        """Initialize logger."""
        self.log_file = log_file
        self._logs = []
        print(f"Logger initialized with file: {log_file}")
    
    def log(self, message: str, level: str = "INFO") -> None:
        """Log a message."""
        log_entry = f"[{level}] {message}"
        self._logs.append(log_entry)
        print(f"Logged: {log_entry}")
    
    def get_logs(self) -> list:
        """Get all log entries."""
        return self._logs.copy()


# Enum-based Singleton (Python 3.4+)
from enum import Enum

class CacheManager(Enum):
    """
    Enum-based Singleton for cache management.
    
    Enums are naturally singletons in Python.
    """
    INSTANCE = 1
    
    def __init__(self, value):
        self._cache = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        return self._cache.get(key)
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        self._cache[key] = value
    
    def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()


def demonstrate_singleton_pattern():
    """Demonstrate different Singleton pattern implementations."""
    
    print("=== Singleton Pattern Demonstration ===\n")
    
    # Database Connection Singleton
    print("1. Database Connection Singleton:")
    db1 = DatabaseConnection("localhost", 5432)
    db2 = DatabaseConnection("remote", 3306)  # Parameters ignored
    
    print(f"db1 is db2: {db1 is db2}")  # True
    print(f"db1.host: {db1.host}")      # localhost
    print(f"db2.host: {db2.host}")      # localhost (same instance)
    
    db1.connect()
    print(f"Query result: {db2.execute_query('SELECT * FROM users')}")
    print()
    
    # Configuration Manager Singleton
    print("2. Configuration Manager Singleton:")
    config1 = ConfigManager()
    config2 = ConfigManager()
    
    print(f"config1 is config2: {config1 is config2}")  # True
    
    config1.set_config("debug", True)
    print(f"Debug from config2: {config2.get_config('debug')}")  # True
    print()
    
    # Logger Singleton (decorator)
    print("3. Logger Singleton (decorator):")
    logger1 = Logger("app.log")
    logger2 = Logger("error.log")  # File name ignored
    
    print(f"logger1 is logger2: {logger1 is logger2}")  # True
    
    logger1.log("Application started", "INFO")
    logger2.log("User logged in", "DEBUG")
    print(f"Total logs: {len(logger2.get_logs())}")
    print()
    
    # Cache Manager Singleton (enum)
    print("4. Cache Manager Singleton (enum):")
    cache1 = CacheManager.INSTANCE
    cache2 = CacheManager.INSTANCE
    
    print(f"cache1 is cache2: {cache1 is cache2}")  # True
    
    cache1.set("user_id", 12345)
    print(f"User ID from cache2: {cache2.get('user_id')}")  # 12345


def test_thread_safety():
    """Test thread safety of Singleton implementation."""
    import time
    import concurrent.futures
    
    print("5. Thread Safety Test:")
    instances = []
    
    def create_db_connection(worker_id):
        """Create database connection in thread."""
        time.sleep(0.01)  # Simulate some work
        db = DatabaseConnection()
        instances.append(db)
        return db
    
    # Create multiple threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(create_db_connection, i) for i in range(10)]
        concurrent.futures.wait(futures)
    
    # Check if all instances are the same
    all_same = all(instance is instances[0] for instance in instances)
    print(f"All instances are the same: {all_same}")
    print(f"Number of unique instances: {len(set(id(inst) for inst in instances))}")


if __name__ == "__main__":
    demonstrate_singleton_pattern()
    test_thread_safety()