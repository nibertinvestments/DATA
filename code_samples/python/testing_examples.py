#!/usr/bin/env python3
"""
Testing-focused code examples for AI training dataset.
Demonstrates testing patterns, frameworks, and best practices.
"""

import unittest
import pytest
import doctest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Optional, Dict, Any
import asyncio
import time
import json
import tempfile
import os


# Example classes to test

class Calculator:
    """A simple calculator class for demonstration."""
    
    def add(self, a: float, b: float) -> float:
        """
        Add two numbers.
        
        >>> calc = Calculator()
        >>> calc.add(2, 3)
        5.0
        >>> calc.add(-1, 1)
        0.0
        """
        return a + b
    
    def subtract(self, a: float, b: float) -> float:
        """Subtract b from a."""
        return a - b
    
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b
    
    def divide(self, a: float, b: float) -> float:
        """
        Divide a by b.
        
        Raises:
            ValueError: If b is zero.
        """
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    
    def power(self, base: float, exponent: float) -> float:
        """Calculate base raised to exponent."""
        return base ** exponent


class UserService:
    """User service with external dependencies for mocking examples."""
    
    def __init__(self, database, email_service):
        self.database = database
        self.email_service = email_service
    
    def create_user(self, username: str, email: str) -> Dict[str, Any]:
        """Create a new user."""
        # Validate input
        if not username or not email:
            raise ValueError("Username and email are required")
        
        if self.database.user_exists(username):
            raise ValueError("User already exists")
        
        # Create user
        user_id = self.database.create_user(username, email)
        
        # Send welcome email
        self.email_service.send_welcome_email(email, username)
        
        return {"id": user_id, "username": username, "email": email}
    
    def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user by ID."""
        return self.database.get_user(user_id)
    
    def update_user_email(self, user_id: int, new_email: str) -> bool:
        """Update user's email address."""
        if not new_email:
            raise ValueError("Email cannot be empty")
        
        success = self.database.update_user_email(user_id, new_email)
        if success:
            self.email_service.send_email_change_notification(new_email)
        
        return success


class DataProcessor:
    """Data processor for async testing examples."""
    
    async def fetch_data(self, url: str) -> Dict[str, Any]:
        """Fetch data from external API."""
        # Simulate async network call
        await asyncio.sleep(0.1)
        
        if url == "http://api.example.com/invalid":
            raise ValueError("Invalid URL")
        
        return {"url": url, "data": "sample_data", "status": "success"}
    
    async def process_multiple_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Process multiple URLs concurrently."""
        tasks = [self.fetch_data(url) for url in urls]
        return await asyncio.gather(*tasks)


# Unit Tests using unittest

class TestCalculatorUnittest(unittest.TestCase):
    """Unit tests for Calculator class using unittest framework."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.calculator = Calculator()
    
    def tearDown(self):
        """Clean up after each test method."""
        # In this case, no cleanup needed
        pass
    
    def test_add_positive_numbers(self):
        """Test addition of positive numbers."""
        result = self.calculator.add(2, 3)
        self.assertEqual(result, 5.0)
    
    def test_add_negative_numbers(self):
        """Test addition of negative numbers."""
        result = self.calculator.add(-2, -3)
        self.assertEqual(result, -5.0)
    
    def test_add_mixed_numbers(self):
        """Test addition of positive and negative numbers."""
        result = self.calculator.add(5, -3)
        self.assertEqual(result, 2.0)
    
    def test_subtract(self):
        """Test subtraction."""
        result = self.calculator.subtract(10, 3)
        self.assertEqual(result, 7.0)
    
    def test_multiply(self):
        """Test multiplication."""
        result = self.calculator.multiply(4, 5)
        self.assertEqual(result, 20.0)
    
    def test_divide_normal(self):
        """Test normal division."""
        result = self.calculator.divide(10, 2)
        self.assertEqual(result, 5.0)
    
    def test_divide_by_zero(self):
        """Test division by zero raises ValueError."""
        with self.assertRaises(ValueError) as context:
            self.calculator.divide(10, 0)
        
        self.assertEqual(str(context.exception), "Cannot divide by zero")
    
    def test_power(self):
        """Test power calculation."""
        result = self.calculator.power(2, 3)
        self.assertEqual(result, 8.0)
    
    def test_power_with_float_exponent(self):
        """Test power with float exponent."""
        result = self.calculator.power(9, 0.5)
        self.assertAlmostEqual(result, 3.0, places=5)


# Tests using pytest

class TestCalculatorPytest:
    """Unit tests for Calculator class using pytest framework."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.calculator = Calculator()
    
    def test_add_integers(self):
        """Test addition with integers."""
        assert self.calculator.add(2, 3) == 5.0
    
    def test_add_floats(self):
        """Test addition with floats."""
        assert self.calculator.add(2.5, 3.7) == pytest.approx(6.2)
    
    @pytest.mark.parametrize("a,b,expected", [
        (1, 2, 3),
        (0, 0, 0),
        (-1, 1, 0),
        (10.5, 2.5, 13.0),
        (-5, -3, -8),
    ])
    def test_add_parametrized(self, a, b, expected):
        """Test addition with multiple parameter sets."""
        assert self.calculator.add(a, b) == pytest.approx(expected)
    
    def test_divide_by_zero_raises_exception(self):
        """Test that division by zero raises ValueError."""
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            self.calculator.divide(10, 0)
    
    @pytest.fixture
    def sample_numbers(self):
        """Fixture providing sample numbers for testing."""
        return [1, 2, 3, 4, 5]
    
    def test_with_fixture(self, sample_numbers):
        """Test using pytest fixture."""
        total = sum(sample_numbers)
        assert total == 15


# Mocking Examples

class TestUserServiceWithMocks(unittest.TestCase):
    """Test UserService using mocks to isolate dependencies."""
    
    def setUp(self):
        """Set up mocks for dependencies."""
        self.mock_database = Mock()
        self.mock_email_service = Mock()
        self.user_service = UserService(self.mock_database, self.mock_email_service)
    
    def test_create_user_success(self):
        """Test successful user creation."""
        # Configure mocks
        self.mock_database.user_exists.return_value = False
        self.mock_database.create_user.return_value = 123
        
        # Call method under test
        result = self.user_service.create_user("john_doe", "john@example.com")
        
        # Verify results
        expected = {"id": 123, "username": "john_doe", "email": "john@example.com"}
        self.assertEqual(result, expected)
        
        # Verify mock calls
        self.mock_database.user_exists.assert_called_once_with("john_doe")
        self.mock_database.create_user.assert_called_once_with("john_doe", "john@example.com")
        self.mock_email_service.send_welcome_email.assert_called_once_with("john@example.com", "john_doe")
    
    def test_create_user_already_exists(self):
        """Test creating user that already exists."""
        # Configure mock to return True for user_exists
        self.mock_database.user_exists.return_value = True
        
        # Verify exception is raised
        with self.assertRaises(ValueError) as context:
            self.user_service.create_user("existing_user", "user@example.com")
        
        self.assertEqual(str(context.exception), "User already exists")
        
        # Verify database was checked but user was not created
        self.mock_database.user_exists.assert_called_once_with("existing_user")
        self.mock_database.create_user.assert_not_called()
        self.mock_email_service.send_welcome_email.assert_not_called()
    
    def test_create_user_invalid_input(self):
        """Test creating user with invalid input."""
        # Test empty username
        with self.assertRaises(ValueError) as context:
            self.user_service.create_user("", "user@example.com")
        
        self.assertEqual(str(context.exception), "Username and email are required")
        
        # Test empty email
        with self.assertRaises(ValueError):
            self.user_service.create_user("username", "")
        
        # Verify no database calls were made
        self.mock_database.user_exists.assert_not_called()
        self.mock_database.create_user.assert_not_called()
    
    @patch('time.time')
    def test_with_patch_decorator(self, mock_time):
        """Test using patch decorator to mock built-in functions."""
        mock_time.return_value = 1234567890
        
        # Use time.time() in some method
        current_time = time.time()
        self.assertEqual(current_time, 1234567890)


# Async Testing

class TestDataProcessorAsync(unittest.TestCase):
    """Test async methods using unittest."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data_processor = DataProcessor()
    
    def test_fetch_data_success(self):
        """Test successful data fetching."""
        async def run_test():
            result = await self.data_processor.fetch_data("http://api.example.com/data")
            expected = {"url": "http://api.example.com/data", "data": "sample_data", "status": "success"}
            self.assertEqual(result, expected)
        
        asyncio.run(run_test())
    
    def test_fetch_data_invalid_url(self):
        """Test fetching data with invalid URL."""
        async def run_test():
            with self.assertRaises(ValueError):
                await self.data_processor.fetch_data("http://api.example.com/invalid")
        
        asyncio.run(run_test())
    
    def test_process_multiple_urls(self):
        """Test processing multiple URLs."""
        async def run_test():
            urls = [
                "http://api.example.com/data1",
                "http://api.example.com/data2",
                "http://api.example.com/data3"
            ]
            
            results = await self.data_processor.process_multiple_urls(urls)
            
            self.assertEqual(len(results), 3)
            for i, result in enumerate(results):
                self.assertEqual(result["url"], urls[i])
                self.assertEqual(result["status"], "success")
        
        asyncio.run(run_test())


# Pytest async tests

@pytest.mark.asyncio
async def test_async_fetch_data_pytest():
    """Test async method using pytest-asyncio."""
    processor = DataProcessor()
    result = await processor.fetch_data("http://api.example.com/test")
    
    assert result["url"] == "http://api.example.com/test"
    assert result["status"] == "success"


@pytest.mark.asyncio
async def test_async_fetch_data_failure_pytest():
    """Test async method failure using pytest-asyncio."""
    processor = DataProcessor()
    
    with pytest.raises(ValueError):
        await processor.fetch_data("http://api.example.com/invalid")


# Property-based testing example (using hypothesis)

# Note: hypothesis library would need to be installed
# from hypothesis import given, strategies as st

# @given(st.floats(), st.floats())
# def test_add_commutative_property(a, b):
#     """Test that addition is commutative using property-based testing."""
#     calculator = Calculator()
#     assert calculator.add(a, b) == calculator.add(b, a)


# Integration Testing

class TestCalculatorIntegration(unittest.TestCase):
    """Integration tests for Calculator."""
    
    def test_complex_calculation(self):
        """Test complex calculation involving multiple operations."""
        calculator = Calculator()
        
        # Test: (2 + 3) * 4 - 5 / 5 = 19
        step1 = calculator.add(2, 3)  # 5
        step2 = calculator.multiply(step1, 4)  # 20
        step3 = calculator.divide(5, 5)  # 1
        result = calculator.subtract(step2, step3)  # 19
        
        self.assertEqual(result, 19.0)
    
    def test_calculator_state_independence(self):
        """Test that calculator operations don't affect each other."""
        calculator = Calculator()
        
        # Perform multiple operations
        result1 = calculator.add(1, 2)
        result2 = calculator.multiply(3, 4)
        result3 = calculator.divide(10, 2)
        
        # Results should be independent
        self.assertEqual(result1, 3.0)
        self.assertEqual(result2, 12.0)
        self.assertEqual(result3, 5.0)


# Performance Testing

class TestCalculatorPerformance(unittest.TestCase):
    """Performance tests for Calculator."""
    
    def test_add_performance(self):
        """Test addition performance."""
        calculator = Calculator()
        
        start_time = time.time()
        
        # Perform many operations
        for i in range(100000):
            calculator.add(i, i + 1)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Assert it completes within reasonable time (adjust as needed)
        self.assertLess(execution_time, 1.0, "Addition should complete quickly")
    
    def test_complex_calculation_performance(self):
        """Test performance of complex calculations."""
        calculator = Calculator()
        
        start_time = time.time()
        
        # Perform complex calculations
        for i in range(10000):
            result = calculator.power(i, 2)
            result = calculator.divide(result, i + 1)
            result = calculator.multiply(result, 2)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"Complex calculation time: {execution_time:.4f} seconds")
        self.assertLess(execution_time, 5.0)


# File-based Testing

class TestFileOperations(unittest.TestCase):
    """Test file operations with temporary files."""
    
    def setUp(self):
        """Create temporary file for testing."""
        self.temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        self.temp_file_path = self.temp_file.name
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up temporary file."""
        if os.path.exists(self.temp_file_path):
            os.unlink(self.temp_file_path)
    
    def test_write_and_read_file(self):
        """Test writing and reading a file."""
        test_data = {"name": "John", "age": 30, "city": "New York"}
        
        # Write data to file
        with open(self.temp_file_path, 'w') as f:
            json.dump(test_data, f)
        
        # Read data from file
        with open(self.temp_file_path, 'r') as f:
            loaded_data = json.load(f)
        
        self.assertEqual(loaded_data, test_data)
    
    def test_file_not_found(self):
        """Test handling of missing files."""
        non_existent_file = "/path/to/non/existent/file.json"
        
        with self.assertRaises(FileNotFoundError):
            with open(non_existent_file, 'r') as f:
                f.read()


# Test Utilities and Helpers

class TestHelpers:
    """Helper methods for testing."""
    
    @staticmethod
    def assert_dict_contains(actual: Dict[str, Any], expected_subset: Dict[str, Any]):
        """Assert that actual dictionary contains all key-value pairs from expected."""
        for key, value in expected_subset.items():
            assert key in actual, f"Key '{key}' not found in actual dictionary"
            assert actual[key] == value, f"Value for key '{key}' doesn't match"
    
    @staticmethod
    def create_mock_user(user_id: int = 1, username: str = "testuser") -> Dict[str, Any]:
        """Create a mock user for testing."""
        return {
            "id": user_id,
            "username": username,
            "email": f"{username}@example.com",
            "created_at": "2023-01-01T00:00:00Z"
        }


# Doctest examples

def fibonacci(n):
    """
    Calculate nth Fibonacci number.
    
    >>> fibonacci(0)
    0
    >>> fibonacci(1)
    1
    >>> fibonacci(5)
    5
    >>> fibonacci(10)
    55
    """
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


def is_palindrome(s):
    """
    Check if string is a palindrome.
    
    >>> is_palindrome("racecar")
    True
    >>> is_palindrome("hello")
    False
    >>> is_palindrome("A man a plan a canal Panama")
    False
    >>> is_palindrome("amanaplanacanalpanama")
    True
    """
    return s == s[::-1]


# Test runners and suites

def create_test_suite():
    """Create a test suite with all test cases."""
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestCalculatorUnittest))
    suite.addTest(unittest.makeSuite(TestUserServiceWithMocks))
    suite.addTest(unittest.makeSuite(TestDataProcessorAsync))
    suite.addTest(unittest.makeSuite(TestCalculatorIntegration))
    suite.addTest(unittest.makeSuite(TestCalculatorPerformance))
    suite.addTest(unittest.makeSuite(TestFileOperations))
    
    return suite


def run_all_tests():
    """Run all tests and generate report."""
    print("Running all tests...")
    
    # Run unittest tests
    suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Run doctests
    print("\nRunning doctests...")
    doctest.testmod(verbose=True)
    
    # Summary
    print(f"\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")


if __name__ == "__main__":
    print("Testing Examples for AI Training Dataset")
    print("=" * 50)
    
    # Run specific test examples
    print("\n=== Running Calculator Tests ===")
    calculator_suite = unittest.TestSuite()
    calculator_suite.addTest(unittest.makeSuite(TestCalculatorUnittest))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(calculator_suite)
    
    print("\n=== Testing Best Practices Summary ===")
    print("1. Write clear, descriptive test names")
    print("2. Use setUp and tearDown for test fixtures")
    print("3. Test both success and failure cases")
    print("4. Use mocks to isolate units under test")
    print("5. Test edge cases and boundary conditions")
    print("6. Use parametrized tests for multiple inputs")
    print("7. Test async code properly")
    print("8. Include performance tests for critical paths")
    print("9. Use temporary files for file operation tests")
    print("10. Organize tests into logical suites")
    print("11. Use docstrings to document test purposes")
    print("12. Maintain test independence")