"""
Basic Python Data Structures Examples

This module demonstrates the usage of Python's built-in data structures
with proper error handling and best practices.
"""

from typing import List, Dict, Set, Tuple, Optional
import json


def list_operations() -> None:
    """Demonstrate common list operations."""
    # List creation and initialization
    numbers: List[int] = [1, 2, 3, 4, 5]
    words: List[str] = ["apple", "banana", "cherry"]
    
    # List comprehensions
    squares = [x**2 for x in numbers]
    even_numbers = [x for x in numbers if x % 2 == 0]
    
    # List methods
    words.append("date")
    words.extend(["elderberry", "fig"])
    words.insert(1, "apricot")
    
    # Safe list operations
    try:
        first_word = words[0]
        last_word = words[-1]
    except IndexError:
        print("List is empty")
    
    # List slicing
    first_three = words[:3]
    last_two = words[-2:]
    
    print(f"Numbers: {numbers}")
    print(f"Squares: {squares}")
    print(f"Even numbers: {even_numbers}")
    print(f"Words: {words}")


def dict_operations() -> None:
    """Demonstrate dictionary operations and best practices."""
    # Dictionary creation
    person: Dict[str, str] = {
        "name": "John Doe",
        "email": "john@example.com",
        "city": "New York"
    }
    
    # Safe dictionary access
    name = person.get("name", "Unknown")
    phone = person.get("phone", "Not provided")
    
    # Dictionary comprehensions
    uppercase_person = {k: v.upper() for k, v in person.items()}
    
    # Merging dictionaries (Python 3.9+)
    additional_info = {"age": "30", "occupation": "Engineer"}
    complete_person = person | additional_info
    
    # Dictionary methods
    keys = list(person.keys())
    values = list(person.values())
    items = list(person.items())
    
    print(f"Person: {person}")
    print(f"Name: {name}, Phone: {phone}")
    print(f"Uppercase: {uppercase_person}")


def set_operations() -> None:
    """Demonstrate set operations for unique collections."""
    # Set creation
    fruits: Set[str] = {"apple", "banana", "cherry"}
    colors: Set[str] = {"red", "green", "blue", "red"}  # Duplicates removed
    
    # Set operations
    tropical_fruits: Set[str] = {"banana", "mango", "pineapple"}
    
    # Union, intersection, difference
    all_fruits = fruits | tropical_fruits
    common_fruits = fruits & tropical_fruits
    unique_fruits = fruits - tropical_fruits
    
    # Set methods
    fruits.add("orange")
    fruits.discard("apple")  # Safe removal
    
    try:
        fruits.remove("grape")  # Raises KeyError if not found
    except KeyError:
        print("Grape not found in fruits set")
    
    print(f"Fruits: {fruits}")
    print(f"All fruits: {all_fruits}")
    print(f"Common fruits: {common_fruits}")


def tuple_operations() -> None:
    """Demonstrate tuple usage for immutable sequences."""
    # Tuple creation
    coordinates: Tuple[float, float] = (40.7128, -74.0060)  # NYC coordinates
    rgb_color: Tuple[int, int, int] = (255, 128, 0)
    
    # Tuple unpacking
    latitude, longitude = coordinates
    red, green, blue = rgb_color
    
    # Named tuples for better readability
    from collections import namedtuple
    Point = namedtuple('Point', ['x', 'y'])
    origin = Point(0, 0)
    
    # Tuple as dictionary key (immutable)
    locations: Dict[Tuple[float, float], str] = {
        (40.7128, -74.0060): "New York",
        (34.0522, -118.2437): "Los Angeles"
    }
    
    print(f"Coordinates: {latitude}, {longitude}")
    print(f"RGB: ({red}, {green}, {blue})")
    print(f"Origin: {origin.x}, {origin.y}")


def advanced_examples() -> None:
    """Advanced data structure patterns."""
    # Default dictionary
    from collections import defaultdict
    word_count: defaultdict = defaultdict(int)
    text = "hello world hello"
    
    for word in text.split():
        word_count[word] += 1
    
    # Counter for frequency counting
    from collections import Counter
    letter_freq = Counter("hello world")
    
    # Deque for efficient queue operations
    from collections import deque
    queue: deque = deque([1, 2, 3])
    queue.appendleft(0)  # Add to front
    queue.append(4)      # Add to back
    
    print(f"Word count: {dict(word_count)}")
    print(f"Letter frequency: {dict(letter_freq)}")
    print(f"Queue: {list(queue)}")


if __name__ == "__main__":
    print("=== Python Data Structures Examples ===\n")
    
    print("1. List Operations:")
    list_operations()
    print()
    
    print("2. Dictionary Operations:")
    dict_operations()
    print()
    
    print("3. Set Operations:")
    set_operations()
    print()
    
    print("4. Tuple Operations:")
    tuple_operations()
    print()
    
    print("5. Advanced Examples:")
    advanced_examples()