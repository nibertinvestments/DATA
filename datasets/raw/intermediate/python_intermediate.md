# Intermediate Python Dataset - Advanced Programming Concepts

## Dataset 1: Advanced Object-Oriented Programming
```python
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Protocol
import weakref

# Abstract base class
class Shape(ABC):
    def __init__(self, name: str):
        self._name = name
    
    @property
    def name(self) -> str:
        return self._name
    
    @abstractmethod
    def area(self) -> float:
        pass
    
    @abstractmethod
    def perimeter(self) -> float:
        pass
    
    def __str__(self) -> str:
        return f"{self._name}: Area={self.area():.2f}, Perimeter={self.perimeter():.2f}"

# Protocol for drawable objects
class Drawable(Protocol):
    def draw(self) -> str:
        ...

# Mixin class
class ColorMixin:
    def __init__(self, *args, color: str = "black", **kwargs):
        super().__init__(*args, **kwargs)
        self._color = color
    
    @property
    def color(self) -> str:
        return self._color
    
    @color.setter
    def color(self, value: str):
        self._color = value

# Concrete implementations
class Rectangle(ColorMixin, Shape):
    def __init__(self, width: float, height: float, **kwargs):
        super().__init__(name="Rectangle", **kwargs)
        self._width = width
        self._height = height
    
    @property
    def width(self) -> float:
        return self._width
    
    @property
    def height(self) -> float:
        return self._height
    
    def area(self) -> float:
        return self._width * self._height
    
    def perimeter(self) -> float:
        return 2 * (self._width + self._height)
    
    def draw(self) -> str:
        return f"Drawing a {self.color} rectangle {self.width}x{self.height}"

class Circle(ColorMixin, Shape):
    def __init__(self, radius: float, **kwargs):
        super().__init__(name="Circle", **kwargs)
        self._radius = radius
    
    @property
    def radius(self) -> float:
        return self._radius
    
    def area(self) -> float:
        return 3.14159 * self._radius ** 2
    
    def perimeter(self) -> float:
        return 2 * 3.14159 * self._radius
    
    def draw(self) -> str:
        return f"Drawing a {self.color} circle with radius {self.radius}"

# Descriptor for validation
class PositiveNumber:
    def __init__(self, name: str):
        self.name = name
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__.get(self.name, 0)
    
    def __set__(self, obj, value):
        if value <= 0:
            raise ValueError(f"{self.name} must be positive")
        obj.__dict__[self.name] = value

class Square(ColorMixin, Shape):
    side = PositiveNumber("side")
    
    def __init__(self, side: float, **kwargs):
        super().__init__(name="Square", **kwargs)
        self.side = side
    
    def area(self) -> float:
        return self.side ** 2
    
    def perimeter(self) -> float:
        return 4 * self.side
    
    def draw(self) -> str:
        return f"Drawing a {self.color} square with side {self.side}"

# Factory pattern
class ShapeFactory:
    @staticmethod
    def create_shape(shape_type: str, **kwargs) -> Shape:
        shapes = {
            "rectangle": Rectangle,
            "circle": Circle,
            "square": Square
        }
        
        if shape_type not in shapes:
            raise ValueError(f"Unknown shape type: {shape_type}")
        
        return shapes[shape_type](**kwargs)

# Observer pattern
class ShapeObserver:
    def update(self, shape: Shape, event: str):
        print(f"Observer: {shape.name} {event}")

class ObservableShape(Rectangle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._observers: List[ShapeObserver] = []
    
    def add_observer(self, observer: ShapeObserver):
        self._observers.append(observer)
    
    def remove_observer(self, observer: ShapeObserver):
        self._observers.remove(observer)
    
    def notify_observers(self, event: str):
        for observer in self._observers:
            observer.update(self, event)
    
    @Rectangle.color.setter
    def color(self, value: str):
        old_color = self._color
        self._color = value
        self.notify_observers(f"color changed from {old_color} to {value}")

# Example usage
def demonstrate_oop():
    # Factory usage
    shapes = [
        ShapeFactory.create_shape("rectangle", width=10, height=5, color="red"),
        ShapeFactory.create_shape("circle", radius=7, color="blue"),
        ShapeFactory.create_shape("square", side=4, color="green")
    ]
    
    # Polymorphism
    for shape in shapes:
        print(shape)
        if isinstance(shape, Drawable):
            print(f"  {shape.draw()}")
    
    # Observer pattern
    observable_rect = ObservableShape(8, 6, color="yellow")
    observer = ShapeObserver()
    observable_rect.add_observer(observer)
    observable_rect.color = "purple"  # Triggers observer notification
    
    # Descriptor validation
    try:
        square = Square(-5)  # Should raise ValueError
    except ValueError as e:
        print(f"Validation error: {e}")

if __name__ == "__main__":
    demonstrate_oop()
```

## Dataset 2: Decorators and Metaclasses
```python
import functools
import time
from typing import Any, Callable, TypeVar, cast
import inspect

F = TypeVar('F', bound=Callable[..., Any])

# Function decorators
def timer(func: F) -> F:
    """Decorator to measure function execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return cast(F, wrapper)

def retry(max_attempts: int = 3, delay: float = 1.0):
    """Decorator to retry function on failure"""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
        return cast(F, wrapper)
    return decorator

def cache_result(func: F) -> F:
    """Simple memoization decorator"""
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create cache key from arguments
        key = str(args) + str(sorted(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    
    wrapper.cache = cache  # Expose cache for inspection
    return cast(F, wrapper)

def validate_types(func: F) -> F:
    """Decorator to validate function arguments against type hints"""
    sig = inspect.signature(func)
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        for name, value in bound_args.arguments.items():
            if name in sig.parameters:
                param = sig.parameters[name]
                if param.annotation != inspect.Parameter.empty:
                    expected_type = param.annotation
                    if not isinstance(value, expected_type):
                        raise TypeError(
                            f"Argument '{name}' must be {expected_type.__name__}, "
                            f"got {type(value).__name__}"
                        )
        
        return func(*args, **kwargs)
    return cast(F, wrapper)

# Class decorators
def singleton(cls):
    """Decorator to make a class a singleton"""
    instances = {}
    
    @functools.wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance

def add_methods(methods_dict):
    """Decorator to dynamically add methods to a class"""
    def decorator(cls):
        for name, method in methods_dict.items():
            setattr(cls, name, method)
        return cls
    return decorator

# Metaclass example
class AutoPropertyMeta(type):
    """Metaclass that automatically creates properties for _private attributes"""
    
    def __new__(mcs, name, bases, namespace):
        # Find all _private attributes and create properties
        for attr_name in list(namespace.keys()):
            if attr_name.startswith('_') and not attr_name.startswith('__'):
                prop_name = attr_name[1:]  # Remove leading underscore
                
                def make_property(attr):
                    def getter(self):
                        return getattr(self, attr)
                    
                    def setter(self, value):
                        setattr(self, attr, value)
                    
                    return property(getter, setter)
                
                if prop_name not in namespace:
                    namespace[prop_name] = make_property(attr_name)
        
        return super().__new__(mcs, name, bases, namespace)

class RegistryMeta(type):
    """Metaclass that maintains a registry of all created classes"""
    registry = {}
    
    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        mcs.registry[name] = cls
        return cls
    
    @classmethod
    def get_class(mcs, name):
        return mcs.registry.get(name)
    
    @classmethod
    def list_classes(mcs):
        return list(mcs.registry.keys())

# Example classes using decorators and metaclasses
@singleton
class DatabaseConnection:
    def __init__(self):
        self.connection_id = id(self)
        print(f"Database connection created: {self.connection_id}")
    
    def execute(self, query):
        return f"Executing: {query}"

@add_methods({
    'bark': lambda self: f"{self.name} says Woof!",
    'fetch': lambda self: f"{self.name} fetches the ball"
})
class Dog:
    def __init__(self, name):
        self.name = name

class Person(metaclass=AutoPropertyMeta):
    def __init__(self, name, age):
        self._name = name
        self._age = age

class Animal(metaclass=RegistryMeta):
    def __init__(self, species):
        self.species = species

class Mammal(Animal):
    def __init__(self, species, warm_blooded=True):
        super().__init__(species)
        self.warm_blooded = warm_blooded

# Example functions using decorators
@timer
@retry(max_attempts=3, delay=0.5)
def unreliable_network_call():
    import random
    if random.random() < 0.7:  # 70% chance of failure
        raise ConnectionError("Network error")
    return "Success!"

@cache_result
@timer
def fibonacci(n: int) -> int:
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

@validate_types
def calculate_area(length: float, width: float) -> float:
    return length * width

# Demonstration
def demonstrate_decorators_metaclasses():
    # Singleton pattern
    db1 = DatabaseConnection()
    db2 = DatabaseConnection()
    print(f"Same instance: {db1 is db2}")  # Should be True
    
    # Dynamic methods
    dog = Dog("Buddy")
    print(dog.bark())
    print(dog.fetch())
    
    # Auto properties
    person = Person("Alice", 30)
    print(f"Name: {person.name}, Age: {person.age}")
    person.age = 31
    print(f"Updated age: {person.age}")
    
    # Registry metaclass
    cat = Animal("Cat")
    dog_animal = Mammal("Dog")
    print(f"Registered classes: {RegistryMeta.list_classes()}")
    
    # Function decorators
    try:
        result = unreliable_network_call()
        print(f"Network call result: {result}")
    except ConnectionError as e:
        print(f"Final error: {e}")
    
    # Memoization
    print(f"Fibonacci(10): {fibonacci(10)}")
    print(f"Fibonacci(10) again: {fibonacci(10)}")  # Should be faster
    print(f"Cache: {fibonacci.cache}")
    
    # Type validation
    try:
        area = calculate_area(10.5, 5.2)
        print(f"Area: {area}")
        
        # This should raise TypeError
        calculate_area("10", 5)
    except TypeError as e:
        print(f"Type error: {e}")

if __name__ == "__main__":
    demonstrate_decorators_metaclasses()
```

## Dataset 3: Asynchronous Programming and Concurrency
```python
import asyncio
import aiohttp
import concurrent.futures
import threading
import time
import queue
from typing import List, Any
from dataclasses import dataclass
import multiprocessing

# Async/await examples
@dataclass
class WebResponse:
    url: str
    status: int
    content: str
    duration: float

async def fetch_url(session: aiohttp.ClientSession, url: str) -> WebResponse:
    """Fetch a single URL asynchronously"""
    start_time = time.time()
    try:
        async with session.get(url) as response:
            content = await response.text()
            duration = time.time() - start_time
            return WebResponse(url, response.status, content[:100], duration)
    except Exception as e:
        duration = time.time() - start_time
        return WebResponse(url, 0, str(e), duration)

async def fetch_multiple_urls(urls: List[str]) -> List[WebResponse]:
    """Fetch multiple URLs concurrently"""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in responses if isinstance(r, WebResponse)]

# Async context manager
class AsyncDatabase:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.connected = False
    
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
    
    async def connect(self):
        print(f"Connecting to {self.connection_string}")
        await asyncio.sleep(0.1)  # Simulate connection time
        self.connected = True
    
    async def disconnect(self):
        print("Disconnecting from database")
        await asyncio.sleep(0.1)  # Simulate disconnection time
        self.connected = False
    
    async def execute_query(self, query: str):
        if not self.connected:
            raise RuntimeError("Database not connected")
        print(f"Executing query: {query}")
        await asyncio.sleep(0.2)  # Simulate query time
        return f"Result for: {query}"

# Async generator
async def async_counter(max_count: int):
    """Async generator that yields numbers with delays"""
    for i in range(max_count):
        await asyncio.sleep(0.1)
        yield i

async def process_async_stream():
    """Process data from async generator"""
    async for number in async_counter(5):
        print(f"Processed: {number}")

# Producer-Consumer pattern with asyncio
class AsyncProducerConsumer:
    def __init__(self, max_queue_size: int = 10):
        self.queue = asyncio.Queue(maxsize=max_queue_size)
        self.running = False
    
    async def producer(self, name: str, items: List[Any]):
        """Produce items and put them in the queue"""
        for item in items:
            await self.queue.put(f"{name}: {item}")
            print(f"Produced: {name}: {item}")
            await asyncio.sleep(0.1)
        
        # Signal completion
        await self.queue.put(None)
    
    async def consumer(self, name: str):
        """Consume items from the queue"""
        while True:
            item = await self.queue.get()
            if item is None:
                break
            
            print(f"Consumer {name} processing: {item}")
            await asyncio.sleep(0.2)  # Simulate processing time
            self.queue.task_done()

# Threading examples
def cpu_bound_task(n: int) -> int:
    """CPU-intensive task for threading/multiprocessing demonstration"""
    result = 0
    for i in range(n * 1000000):
        result += i % 10
    return result

def io_bound_task(delay: float) -> str:
    """I/O bound task simulation"""
    time.sleep(delay)
    return f"Completed after {delay}s"

class ThreadSafeCounter:
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()
    
    def increment(self):
        with self._lock:
            self._value += 1
    
    @property
    def value(self):
        return self._value

def thread_worker(counter: ThreadSafeCounter, iterations: int):
    """Worker function for threading example"""
    for _ in range(iterations):
        counter.increment()

# Multiprocessing examples
def multiprocessing_worker(start: int, end: int) -> int:
    """Worker function for multiprocessing"""
    return sum(range(start, end))

# Event loop management
class AsyncEventManager:
    def __init__(self):
        self.events = {}
        self.running = False
    
    async def emit_event(self, event_name: str, data: Any):
        """Emit an event to all registered handlers"""
        if event_name in self.events:
            tasks = []
            for handler in self.events[event_name]:
                tasks.append(asyncio.create_task(handler(data)))
            await asyncio.gather(*tasks)
    
    def on(self, event_name: str, handler):
        """Register an event handler"""
        if event_name not in self.events:
            self.events[event_name] = []
        self.events[event_name].append(handler)

async def event_handler_1(data):
    print(f"Handler 1 received: {data}")
    await asyncio.sleep(0.1)

async def event_handler_2(data):
    print(f"Handler 2 received: {data}")
    await asyncio.sleep(0.2)

# Demonstration functions
async def demonstrate_async_programming():
    print("=== Async URL Fetching ===")
    urls = [
        "https://httpbin.org/delay/1",
        "https://httpbin.org/delay/2",
        "https://httpbin.org/json"
    ]
    
    start_time = time.time()
    responses = await fetch_multiple_urls(urls)
    total_time = time.time() - start_time
    
    for response in responses:
        print(f"URL: {response.url}, Status: {response.status}, Duration: {response.duration:.2f}s")
    print(f"Total time: {total_time:.2f}s")
    
    print("\n=== Async Database ===")
    async with AsyncDatabase("postgresql://localhost/test") as db:
        result = await db.execute_query("SELECT * FROM users")
        print(result)
    
    print("\n=== Async Stream Processing ===")
    await process_async_stream()
    
    print("\n=== Producer-Consumer Pattern ===")
    pc = AsyncProducerConsumer()
    
    # Create tasks
    producer_task = asyncio.create_task(
        pc.producer("Producer1", ["item1", "item2", "item3"])
    )
    consumer_task = asyncio.create_task(
        pc.consumer("Consumer1")
    )
    
    await asyncio.gather(producer_task, consumer_task)
    
    print("\n=== Event Manager ===")
    event_manager = AsyncEventManager()
    event_manager.on("test_event", event_handler_1)
    event_manager.on("test_event", event_handler_2)
    
    await event_manager.emit_event("test_event", "Hello Events!")

def demonstrate_threading():
    print("\n=== Threading Examples ===")
    
    # I/O bound tasks with threading
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(io_bound_task, 1),
            executor.submit(io_bound_task, 1),
            executor.submit(io_bound_task, 1)
        ]
        
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    print(f"Threading I/O tasks completed in {time.time() - start_time:.2f}s")
    
    # Thread-safe counter
    counter = ThreadSafeCounter()
    threads = []
    
    for i in range(5):
        thread = threading.Thread(target=thread_worker, args=(counter, 1000))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    print(f"Thread-safe counter value: {counter.value}")

def demonstrate_multiprocessing():
    print("\n=== Multiprocessing Examples ===")
    
    # CPU-bound tasks with multiprocessing
    start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(multiprocessing_worker, 0, 1000000),
            executor.submit(multiprocessing_worker, 1000000, 2000000),
            executor.submit(multiprocessing_worker, 2000000, 3000000)
        ]
        
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    print(f"Multiprocessing completed in {time.time() - start_time:.2f}s")
    print(f"Results: {results}")

# Main execution
async def main():
    await demonstrate_async_programming()
    demonstrate_threading()
    demonstrate_multiprocessing()

if __name__ == "__main__":
    asyncio.run(main())
```