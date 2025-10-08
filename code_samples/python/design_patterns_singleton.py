#!/usr/bin/env python3
"""
Singleton Design Pattern
Thread-safe and various implementations
"""

class SingletonMeta(type):
    """Metaclass for Singleton pattern."""
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Singleton(metaclass=SingletonMeta):
    """Singleton class using metaclass."""
    
    def __init__(self):
        self.value = None
    
    def set_value(self, value):
        self.value = value
    
    def get_value(self):
        return self.value

class DatabaseConnection:
    """Example: Database connection as singleton."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.connection = None
            cls._instance.is_connected = False
        return cls._instance
    
    def connect(self, connection_string):
        """Establish database connection."""
        if not self.is_connected:
            self.connection = connection_string
            self.is_connected = True
            print(f"Connected to: {connection_string}")
    
    def disconnect(self):
        """Close database connection."""
        if self.is_connected:
            self.connection = None
            self.is_connected = False
            print("Disconnected from database")
    
    def query(self, sql):
        """Execute SQL query."""
        if self.is_connected:
            return f"Executing: {sql}"
        return "Not connected"

class ThreadSafeSingleton:
    """Thread-safe singleton implementation."""
    _instance = None
    _lock = None
    
    @classmethod
    def __new__(cls):
        if cls._instance is None:
            import threading
            if cls._lock is None:
                cls._lock = threading.Lock()
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

if __name__ == "__main__":
    # Test singleton
    s1 = Singleton()
    s1.set_value(10)
    s2 = Singleton()
    print(f"s1 value: {s1.get_value()}")
    print(f"s2 value: {s2.get_value()}")
    print(f"s1 is s2: {s1 is s2}")
    
    # Test database singleton
    db1 = DatabaseConnection()
    db1.connect("postgresql://localhost:5432/mydb")
    db2 = DatabaseConnection()
    print(f"db1 is db2: {db1 is db2}")
    print(db2.query("SELECT * FROM users"))
