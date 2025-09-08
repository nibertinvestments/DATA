"""
Common Python Errors and Solutions

This module demonstrates frequent Python programming errors and their solutions,
helping AI coding agents learn to identify and fix common mistakes.
"""

import json
from typing import List, Dict, Optional


def example_1_index_error():
    """
    ERROR: IndexError - list index out of range
    Common when accessing array elements without checking bounds.
    """
    print("=== Example 1: IndexError ===")
    
    # ❌ BROKEN CODE
    def broken_get_last_item(items):
        """This will fail on empty lists."""
        return items[-1]  # IndexError if list is empty
    
    # ✅ FIXED CODE
    def safe_get_last_item(items: List) -> Optional:
        """Safely get the last item from a list."""
        if not items:  # Check if list is empty
            return None
        return items[-1]
    
    # Alternative with try-except
    def safe_get_last_item_v2(items: List) -> Optional:
        """Another approach using exception handling."""
        try:
            return items[-1]
        except IndexError:
            return None
    
    # Demonstration
    empty_list = []
    valid_list = [1, 2, 3, 4, 5]
    
    print("Testing with empty list:")
    # print(broken_get_last_item(empty_list))  # Would raise IndexError
    print(f"Safe version: {safe_get_last_item(empty_list)}")
    print(f"Try-except version: {safe_get_last_item_v2(empty_list)}")
    
    print("\nTesting with valid list:")
    print(f"Safe version: {safe_get_last_item(valid_list)}")
    print(f"Try-except version: {safe_get_last_item_v2(valid_list)}")


def example_2_key_error():
    """
    ERROR: KeyError - key not found in dictionary
    Common when accessing dictionary keys without checking existence.
    """
    print("\n=== Example 2: KeyError ===")
    
    # ❌ BROKEN CODE
    def broken_get_user_info(user_data, key):
        """This will fail if key doesn't exist."""
        return user_data[key]  # KeyError if key not found
    
    # ✅ FIXED CODE
    def safe_get_user_info(user_data: Dict, key: str, default=None):
        """Safely get user information with default value."""
        return user_data.get(key, default)
    
    # Alternative with explicit check
    def safe_get_user_info_v2(user_data: Dict, key: str, default=None):
        """Another approach with explicit key checking."""
        if key in user_data:
            return user_data[key]
        return default
    
    # Alternative with try-except
    def safe_get_user_info_v3(user_data: Dict, key: str, default=None):
        """Exception handling approach."""
        try:
            return user_data[key]
        except KeyError:
            return default
    
    # Demonstration
    user = {"name": "Alice", "age": 30, "email": "alice@example.com"}
    
    print("Testing with existing key:")
    print(f"Safe get: {safe_get_user_info(user, 'name')}")
    
    print("\nTesting with non-existing key:")
    # print(broken_get_user_info(user, 'phone'))  # Would raise KeyError
    print(f"Safe get: {safe_get_user_info(user, 'phone', 'Not provided')}")
    print(f"Explicit check: {safe_get_user_info_v2(user, 'phone', 'Not provided')}")
    print(f"Try-except: {safe_get_user_info_v3(user, 'phone', 'Not provided')}")


def example_3_type_error():
    """
    ERROR: TypeError - operation on incompatible types
    Common when mixing different data types in operations.
    """
    print("\n=== Example 3: TypeError ===")
    
    # ❌ BROKEN CODE
    def broken_calculate_total(prices):
        """This will fail if prices contains non-numeric values."""
        total = 0
        for price in prices:
            total += price  # TypeError if price is not a number
        return total
    
    # ✅ FIXED CODE
    def safe_calculate_total(prices: List) -> float:
        """Safely calculate total, filtering out non-numeric values."""
        total = 0.0
        for price in prices:
            if isinstance(price, (int, float)):
                total += price
            else:
                print(f"Warning: Skipping non-numeric value: {price}")
        return total
    
    # Alternative with type conversion
    def safe_calculate_total_v2(prices: List) -> float:
        """Convert values to float, handle conversion errors."""
        total = 0.0
        for price in prices:
            try:
                total += float(price)
            except (ValueError, TypeError):
                print(f"Warning: Cannot convert to number: {price}")
        return total
    
    # Demonstration
    mixed_prices = [10.99, 25.50, "invalid", 15.00, None, "12.99"]
    
    print("Testing with mixed data types:")
    # print(broken_calculate_total(mixed_prices))  # Would raise TypeError
    print(f"Safe version: ${safe_calculate_total(mixed_prices):.2f}")
    print(f"Conversion version: ${safe_calculate_total_v2(mixed_prices):.2f}")


def example_4_attribute_error():
    """
    ERROR: AttributeError - attribute not found on object
    Common when calling methods that don't exist on an object.
    """
    print("\n=== Example 4: AttributeError ===")
    
    # ❌ BROKEN CODE
    def broken_process_data(data):
        """This assumes data always has certain methods."""
        return data.strip().upper()  # AttributeError if data is not a string
    
    # ✅ FIXED CODE
    def safe_process_data(data) -> str:
        """Safely process data with type checking."""
        if isinstance(data, str):
            return data.strip().upper()
        elif data is None:
            return ""
        else:
            # Convert to string first
            return str(data).strip().upper()
    
    # Alternative with hasattr
    def safe_process_data_v2(data) -> str:
        """Use hasattr to check for method existence."""
        if hasattr(data, 'strip') and hasattr(data, 'upper'):
            return data.strip().upper()
        else:
            return str(data).upper() if data is not None else ""
    
    # Alternative with try-except
    def safe_process_data_v3(data) -> str:
        """Exception handling approach."""
        try:
            return data.strip().upper()
        except AttributeError:
            return str(data).upper() if data is not None else ""
    
    # Demonstration
    test_data = [
        "  hello world  ",  # Valid string
        123,                # Integer
        None,              # None value
        ["list", "data"]   # List
    ]
    
    for i, data in enumerate(test_data, 1):
        print(f"\nTest {i} with {type(data).__name__}: {data}")
        # print(broken_process_data(data))  # Would fail on non-strings
        print(f"Safe version: '{safe_process_data(data)}'")
        print(f"Hasattr version: '{safe_process_data_v2(data)}'")
        print(f"Try-except version: '{safe_process_data_v3(data)}'")


def example_5_infinite_loop():
    """
    ERROR: Infinite loop - loop that never terminates
    Common logic error in while loops and recursive functions.
    """
    print("\n=== Example 5: Infinite Loop Prevention ===")
    
    # ❌ BROKEN CODE
    def broken_countdown(n):
        """This creates an infinite loop if n is negative."""
        while n > 0:
            print(n)
            # Missing: n -= 1  # Infinite loop!
    
    # ✅ FIXED CODE
    def safe_countdown(n: int, max_iterations: int = 1000) -> None:
        """Countdown with safety checks."""
        if n <= 0:
            print("Invalid input: n must be positive")
            return
        
        iteration_count = 0
        while n > 0 and iteration_count < max_iterations:
            print(n)
            n -= 1
            iteration_count += 1
        
        if iteration_count >= max_iterations:
            print("Warning: Maximum iterations reached, stopping countdown")
    
    # Alternative with for loop (safer)
    def safe_countdown_v2(n: int) -> None:
        """Using for loop to avoid infinite loop possibility."""
        if n <= 0:
            print("Invalid input: n must be positive")
            return
        
        for i in range(n, 0, -1):
            print(i)
    
    # Demonstration
    print("Safe countdown from 5:")
    safe_countdown(5)
    
    print("\nSafe countdown using for loop:")
    safe_countdown_v2(3)
    
    print("\nTesting with invalid input:")
    safe_countdown(-1)


def example_6_json_decode_error():
    """
    ERROR: JSONDecodeError - invalid JSON format
    Common when parsing JSON from external sources.
    """
    print("\n=== Example 6: JSON Decode Error ===")
    
    # ❌ BROKEN CODE
    def broken_parse_json(json_string):
        """This will fail with invalid JSON."""
        return json.loads(json_string)  # JSONDecodeError on invalid JSON
    
    # ✅ FIXED CODE
    def safe_parse_json(json_string: str, default=None) -> Dict:
        """Safely parse JSON with error handling."""
        if not isinstance(json_string, str):
            print(f"Error: Expected string, got {type(json_string)}")
            return default or {}
        
        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return default or {}
        except Exception as e:
            print(f"Unexpected error: {e}")
            return default or {}
    
    # Demonstration
    test_cases = [
        '{"name": "Alice", "age": 30}',  # Valid JSON
        '{"name": "Bob", "age": }',      # Invalid JSON (missing value)
        'not json at all',               # Not JSON
        '',                              # Empty string
        None                             # None value
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case}")
        # result = broken_parse_json(test_case)  # Would fail on invalid JSON
        result = safe_parse_json(test_case)
        print(f"Result: {result}")


def example_7_division_by_zero():
    """
    ERROR: ZeroDivisionError - division by zero
    Common mathematical error that should be handled gracefully.
    """
    print("\n=== Example 7: Division by Zero ===")
    
    # ❌ BROKEN CODE
    def broken_calculate_average(total, count):
        """This will fail if count is zero."""
        return total / count  # ZeroDivisionError if count is 0
    
    # ✅ FIXED CODE
    def safe_calculate_average(total: float, count: int) -> Optional[float]:
        """Safely calculate average with zero check."""
        if count == 0:
            print("Warning: Cannot calculate average of zero items")
            return None
        return total / count
    
    # Alternative with try-except
    def safe_calculate_average_v2(total: float, count: int) -> Optional[float]:
        """Exception handling approach."""
        try:
            return total / count
        except ZeroDivisionError:
            print("Error: Division by zero")
            return None
    
    # Enhanced version with validation
    def robust_calculate_average(values: List[float]) -> Optional[float]:
        """Calculate average from list of values with comprehensive validation."""
        if not values:
            print("Warning: Empty list provided")
            return None
        
        # Filter out non-numeric values
        numeric_values = [v for v in values if isinstance(v, (int, float))]
        
        if not numeric_values:
            print("Warning: No numeric values found")
            return None
        
        return sum(numeric_values) / len(numeric_values)
    
    # Demonstration
    test_cases = [
        (100, 5),   # Valid case
        (50, 0),    # Division by zero
        (0, 0),     # Zero divided by zero
    ]
    
    for total, count in test_cases:
        print(f"\nCalculating average of {total} over {count} items:")
        # result = broken_calculate_average(total, count)  # Would fail on count=0
        result1 = safe_calculate_average(total, count)
        result2 = safe_calculate_average_v2(total, count)
        print(f"Safe version: {result1}")
        print(f"Try-except version: {result2}")
    
    # Test with list of values
    print(f"\nTesting with list of values:")
    test_lists = [
        [10, 20, 30, 40, 50],           # Valid numbers
        [],                              # Empty list
        [1, 2, "invalid", 4, None, 6],  # Mixed types
    ]
    
    for test_list in test_lists:
        print(f"Values: {test_list}")
        result = robust_calculate_average(test_list)
        print(f"Average: {result}")


# Best Practices Summary
def error_handling_best_practices():
    """
    Summary of Python error handling best practices.
    """
    print("\n" + "="*50)
    print("PYTHON ERROR HANDLING BEST PRACTICES")
    print("="*50)
    
    practices = [
        "1. Always validate input parameters before processing",
        "2. Use isinstance() for type checking instead of type()",
        "3. Prefer dict.get() over dict[key] for optional keys",
        "4. Use try-except for expected exceptions, if-checks for validation",
        "5. Provide meaningful default values for error cases",
        "6. Log errors with context information for debugging",
        "7. Fail fast: validate early and return/raise immediately",
        "8. Use specific exception types rather than bare except:",
        "9. Add bounds checking for array/list access",
        "10. Include timeout/iteration limits for loops"
    ]
    
    for practice in practices:
        print(practice)
    
    print("\nRemember: Good error handling makes code more robust and easier to debug!")


if __name__ == "__main__":
    print("Python Common Errors and Solutions")
    print("=" * 50)
    
    # Run all examples
    example_1_index_error()
    example_2_key_error()
    example_3_type_error()
    example_4_attribute_error()
    example_5_infinite_loop()
    example_6_json_decode_error()
    example_7_division_by_zero()
    
    # Show best practices
    error_handling_best_practices()