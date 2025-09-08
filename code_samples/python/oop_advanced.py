"""
Advanced Object-Oriented Programming Examples in Python
Demonstrates inheritance, polymorphism, decorators, and design patterns.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Protocol
import functools
import time


# Abstract Base Class and Inheritance
class Animal(ABC):
    """Abstract base class for all animals."""
    
    def __init__(self, name: str, species: str):
        self.name = name
        self.species = species
        self._energy = 100
    
    @abstractmethod
    def make_sound(self) -> str:
        """Abstract method that must be implemented by subclasses."""
        pass
    
    def eat(self, food_value: int = 10) -> None:
        """Increase energy by eating."""
        self._energy = min(100, self._energy + food_value)
        print(f"{self.name} ate and now has {self._energy} energy")
    
    def sleep(self, hours: int = 8) -> None:
        """Restore energy by sleeping."""
        self._energy = min(100, self._energy + hours * 5)
        print(f"{self.name} slept for {hours} hours")


class Dog(Animal):
    """Dog implementation of Animal."""
    
    def __init__(self, name: str, breed: str):
        super().__init__(name, "Canis familiaris")
        self.breed = breed
    
    def make_sound(self) -> str:
        return "Woof!"
    
    def fetch(self, item: str) -> str:
        """Dog-specific behavior."""
        return f"{self.name} fetched the {item}!"


class Cat(Animal):
    """Cat implementation of Animal."""
    
    def __init__(self, name: str, indoor: bool = True):
        super().__init__(name, "Felis catus")
        self.indoor = indoor
    
    def make_sound(self) -> str:
        return "Meow!"
    
    def hunt(self) -> str:
        """Cat-specific behavior."""
        if not self.indoor:
            return f"{self.name} caught a mouse!"
        return f"{self.name} stalked a dust bunny!"


# Protocol-based polymorphism (Python 3.8+)
class Flyable(Protocol):
    """Protocol for objects that can fly."""
    
    def fly(self) -> str:
        """Return description of flying behavior."""
        ...


class Bird(Animal):
    """Bird implementation with flying capability."""
    
    def __init__(self, name: str, species: str, can_fly: bool = True):
        super().__init__(name, species)
        self.can_fly = can_fly
    
    def make_sound(self) -> str:
        return "Tweet!"
    
    def fly(self) -> str:
        if self.can_fly:
            return f"{self.name} soars through the sky!"
        return f"{self.name} flaps wings but stays grounded."


class Airplane:
    """Non-animal object that can also fly."""
    
    def __init__(self, model: str):
        self.model = model
    
    def fly(self) -> str:
        return f"{self.model} flies at 30,000 feet!"


# Decorator examples
def timing_decorator(func):
    """Decorator to measure execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper


def validate_positive(func):
    """Decorator to validate that numeric arguments are positive."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for arg in args[1:]:  # Skip 'self'
            if isinstance(arg, (int, float)) and arg < 0:
                raise ValueError("All numeric arguments must be positive")
        return func(*args, **kwargs)
    return wrapper


# Singleton Design Pattern
class DatabaseConnection:
    """Singleton database connection class."""
    
    _instance: Optional['DatabaseConnection'] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseConnection, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.connection_string = "sqlite://memory"
            self.is_connected = False
            DatabaseConnection._initialized = True
    
    def connect(self) -> str:
        """Simulate database connection."""
        self.is_connected = True
        return f"Connected to {self.connection_string}"
    
    def disconnect(self) -> str:
        """Simulate database disconnection."""
        self.is_connected = False
        return "Disconnected from database"


# Factory Pattern
class VehicleFactory:
    """Factory for creating different types of vehicles."""
    
    @staticmethod
    def create_vehicle(vehicle_type: str, **kwargs):
        """Create a vehicle based on type."""
        if vehicle_type.lower() == "car":
            return Car(**kwargs)
        elif vehicle_type.lower() == "motorcycle":
            return Motorcycle(**kwargs)
        elif vehicle_type.lower() == "bicycle":
            return Bicycle(**kwargs)
        else:
            raise ValueError(f"Unknown vehicle type: {vehicle_type}")


class Vehicle(ABC):
    """Abstract vehicle class."""
    
    def __init__(self, brand: str, model: str):
        self.brand = brand
        self.model = model
    
    @abstractmethod
    def start_engine(self) -> str:
        pass
    
    @abstractmethod
    def get_max_speed(self) -> int:
        pass


class Car(Vehicle):
    """Car implementation."""
    
    def __init__(self, brand: str, model: str, doors: int = 4):
        super().__init__(brand, model)
        self.doors = doors
    
    def start_engine(self) -> str:
        return f"{self.brand} {self.model} engine started with a rumble"
    
    def get_max_speed(self) -> int:
        return 200  # km/h


class Motorcycle(Vehicle):
    """Motorcycle implementation."""
    
    def __init__(self, brand: str, model: str, engine_size: int):
        super().__init__(brand, model)
        self.engine_size = engine_size
    
    def start_engine(self) -> str:
        return f"{self.brand} {self.model} engine roars to life"
    
    def get_max_speed(self) -> int:
        return 250  # km/h


class Bicycle(Vehicle):
    """Bicycle implementation."""
    
    def __init__(self, brand: str, model: str, gears: int):
        super().__init__(brand, model)
        self.gears = gears
    
    def start_engine(self) -> str:
        return f"Ready to pedal the {self.brand} {self.model}"
    
    def get_max_speed(self) -> int:
        return 50  # km/h


# Context Manager example
class FileManager:
    """Context manager for file operations."""
    
    def __init__(self, filename: str, mode: str = 'r'):
        self.filename = filename
        self.mode = mode
        self.file = None
    
    def __enter__(self):
        print(f"Opening file: {self.filename}")
        self.file = open(self.filename, self.mode)
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"Closing file: {self.filename}")
        if self.file:
            self.file.close()
        if exc_type:
            print(f"Exception occurred: {exc_val}")
        return False


# Property decorators and getters/setters
class Temperature:
    """Temperature class with property validation."""
    
    def __init__(self, celsius: float = 0):
        self._celsius = celsius
    
    @property
    def celsius(self) -> float:
        return self._celsius
    
    @celsius.setter
    @validate_positive
    def celsius(self, value: float):
        if value < -273.15:
            raise ValueError("Temperature cannot be below absolute zero")
        self._celsius = value
    
    @property
    def fahrenheit(self) -> float:
        return (self._celsius * 9/5) + 32
    
    @fahrenheit.setter
    def fahrenheit(self, value: float):
        self.celsius = (value - 32) * 5/9
    
    @property
    def kelvin(self) -> float:
        return self._celsius + 273.15


def demonstrate_polymorphism(flying_objects: List[Flyable]):
    """Demonstrate polymorphism with Protocol."""
    for obj in flying_objects:
        print(obj.fly())


@timing_decorator
def demonstrate_inheritance():
    """Demonstrate inheritance and method overriding."""
    animals = [
        Dog("Buddy", "Golden Retriever"),
        Cat("Whiskers", indoor=False),
        Bird("Tweety", "Canary")
    ]
    
    for animal in animals:
        print(f"{animal.name} says: {animal.make_sound()}")
        animal.eat()
        
        # Demonstrate specific behaviors
        if isinstance(animal, Dog):
            print(animal.fetch("ball"))
        elif isinstance(animal, Cat):
            print(animal.hunt())
        elif isinstance(animal, Bird):
            print(animal.fly())


def demonstrate_design_patterns():
    """Demonstrate various design patterns."""
    
    # Singleton pattern
    db1 = DatabaseConnection()
    db2 = DatabaseConnection()
    print(f"Same instance: {db1 is db2}")  # True
    
    # Factory pattern
    vehicles = [
        VehicleFactory.create_vehicle("car", brand="Toyota", model="Camry"),
        VehicleFactory.create_vehicle("motorcycle", brand="Honda", model="CBR", engine_size=600),
        VehicleFactory.create_vehicle("bicycle", brand="Trek", model="FX", gears=21)
    ]
    
    for vehicle in vehicles:
        print(vehicle.start_engine())
        print(f"Max speed: {vehicle.get_max_speed()} km/h")


if __name__ == "__main__":
    print("=== Advanced OOP Demonstration ===\n")
    
    print("1. Inheritance and Polymorphism:")
    demonstrate_inheritance()
    print()
    
    print("2. Protocol-based Polymorphism:")
    flying_objects = [Bird("Eagle", "Bald Eagle"), Airplane("Boeing 747")]
    demonstrate_polymorphism(flying_objects)
    print()
    
    print("3. Design Patterns:")
    demonstrate_design_patterns()
    print()
    
    print("4. Property Decorators:")
    temp = Temperature(25)
    print(f"25°C = {temp.fahrenheit:.1f}°F = {temp.kelvin:.1f}K")
    
    temp.fahrenheit = 86
    print(f"86°F = {temp.celsius:.1f}°C = {temp.kelvin:.1f}K")