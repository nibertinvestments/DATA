"""
Python Testing Examples with pytest

This module demonstrates testing patterns and best practices for Python code,
showing how to write comprehensive tests for AI coding agents.
"""

import pytest
from typing import List, Dict
import json


# Sample functions to test (from examples directory)
def calculate_average(values: List[float]) -> float:
    """Calculate the average of a list of numbers."""
    if not values:
        raise ValueError("Cannot calculate average of empty list")
    
    return sum(values) / len(values)


def safe_dict_get(data: Dict, key: str, default=None):
    """Safely get a value from a dictionary."""
    return data.get(key, default)


def process_user_data(user_data: Dict) -> Dict:
    """Process user data with validation."""
    if not isinstance(user_data, dict):
        raise TypeError("User data must be a dictionary")
    
    processed = {
        'name': user_data.get('name', 'Unknown').strip().title(),
        'age': max(0, user_data.get('age', 0)),
        'email': user_data.get('email', '').lower().strip()
    }
    
    return processed


# Test Classes
class TestCalculateAverage:
    """Test cases for calculate_average function."""
    
    def test_valid_numbers(self):
        """Test with valid list of numbers."""
        assert calculate_average([1, 2, 3, 4, 5]) == 3.0
        assert calculate_average([10, 20]) == 15.0
        assert calculate_average([100]) == 100.0
    
    def test_negative_numbers(self):
        """Test with negative numbers."""
        assert calculate_average([-1, -2, -3]) == -2.0
        assert calculate_average([-5, 5]) == 0.0
    
    def test_decimal_numbers(self):
        """Test with decimal numbers."""
        result = calculate_average([1.5, 2.5, 3.5])
        assert abs(result - 2.5) < 0.001  # Use approximate equality for floats
    
    def test_empty_list(self):
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="Cannot calculate average of empty list"):
            calculate_average([])
    
    def test_single_value(self):
        """Test with single value."""
        assert calculate_average([42]) == 42


class TestSafeDictGet:
    """Test cases for safe_dict_get function."""
    
    def test_existing_key(self):
        """Test getting existing key."""
        data = {'name': 'Alice', 'age': 30}
        assert safe_dict_get(data, 'name') == 'Alice'
        assert safe_dict_get(data, 'age') == 30
    
    def test_missing_key_with_default(self):
        """Test getting missing key with default value."""
        data = {'name': 'Alice'}
        assert safe_dict_get(data, 'age', 0) == 0
        assert safe_dict_get(data, 'email', 'not provided') == 'not provided'
    
    def test_missing_key_no_default(self):
        """Test getting missing key without default (should return None)."""
        data = {'name': 'Alice'}
        assert safe_dict_get(data, 'missing_key') is None
    
    def test_empty_dict(self):
        """Test with empty dictionary."""
        assert safe_dict_get({}, 'any_key', 'default') == 'default'


class TestProcessUserData:
    """Test cases for process_user_data function."""
    
    def test_complete_user_data(self):
        """Test with complete user data."""
        input_data = {
            'name': '  alice smith  ',
            'age': 25,
            'email': '  ALICE@EXAMPLE.COM  '
        }
        
        expected = {
            'name': 'Alice Smith',
            'age': 25,
            'email': 'alice@example.com'
        }
        
        assert process_user_data(input_data) == expected
    
    def test_partial_user_data(self):
        """Test with partial user data."""
        input_data = {'name': 'bob'}
        
        expected = {
            'name': 'Bob',
            'age': 0,
            'email': ''
        }
        
        assert process_user_data(input_data) == expected
    
    def test_negative_age(self):
        """Test that negative age is converted to 0."""
        input_data = {'name': 'charlie', 'age': -5}
        result = process_user_data(input_data)
        assert result['age'] == 0
    
    def test_invalid_input_type(self):
        """Test that non-dict input raises TypeError."""
        with pytest.raises(TypeError, match="User data must be a dictionary"):
            process_user_data("not a dict")
        
        with pytest.raises(TypeError, match="User data must be a dictionary"):
            process_user_data([1, 2, 3])
    
    def test_empty_dict(self):
        """Test with empty dictionary."""
        expected = {
            'name': 'Unknown',
            'age': 0,
            'email': ''
        }
        
        assert process_user_data({}) == expected


# Parameterized Tests
class TestParameterizedExamples:
    """Examples of parameterized tests for testing multiple scenarios."""
    
    @pytest.mark.parametrize("values, expected", [
        ([1, 2, 3], 2.0),
        ([10, 20, 30, 40], 25.0),
        ([-1, 1], 0.0),
        ([100], 100.0),
        ([1.5, 2.5], 2.0),
    ])
    def test_calculate_average_multiple_cases(self, values, expected):
        """Test calculate_average with multiple parameter sets."""
        assert calculate_average(values) == expected
    
    @pytest.mark.parametrize("data, key, default, expected", [
        ({'a': 1}, 'a', None, 1),
        ({'a': 1}, 'b', 'default', 'default'),
        ({}, 'any', 0, 0),
        ({'name': 'test'}, 'name', None, 'test'),
    ])
    def test_safe_dict_get_multiple_cases(self, data, key, default, expected):
        """Test safe_dict_get with multiple parameter sets."""
        assert safe_dict_get(data, key, default) == expected


# Fixture Examples
@pytest.fixture
def sample_user_data():
    """Fixture providing sample user data for tests."""
    return {
        'name': 'John Doe',
        'age': 30,
        'email': 'john@example.com',
        'city': 'New York'
    }


@pytest.fixture
def empty_user_data():
    """Fixture providing empty user data."""
    return {}


class TestWithFixtures:
    """Tests demonstrating the use of fixtures."""
    
    def test_with_sample_data(self, sample_user_data):
        """Test using sample data fixture."""
        result = process_user_data(sample_user_data)
        assert result['name'] == 'John Doe'
        assert result['age'] == 30
        assert result['email'] == 'john@example.com'
    
    def test_with_empty_data(self, empty_user_data):
        """Test using empty data fixture."""
        result = process_user_data(empty_user_data)
        assert result['name'] == 'Unknown'
        assert result['age'] == 0
        assert result['email'] == ''


# Mock Examples (for external dependencies)
class APIClient:
    """Example class that makes external API calls."""
    
    def fetch_user(self, user_id: int) -> Dict:
        """Fetch user data from external API."""
        # In real code, this would make an HTTP request
        import requests
        response = requests.get(f"https://api.example.com/users/{user_id}")
        return response.json()


class UserService:
    """Service that uses APIClient."""
    
    def __init__(self, api_client: APIClient):
        self.api_client = api_client
    
    def get_user_summary(self, user_id: int) -> str:
        """Get a formatted user summary."""
        user_data = self.api_client.fetch_user(user_id)
        return f"{user_data['name']} ({user_data['age']} years old)"


class TestUserService:
    """Tests demonstrating mocking external dependencies."""
    
    def test_get_user_summary_with_mock(self, mocker):
        """Test user service with mocked API client."""
        # Create a mock API client
        mock_api_client = mocker.Mock(spec=APIClient)
        mock_api_client.fetch_user.return_value = {
            'name': 'Jane Doe',
            'age': 25
        }
        
        # Test the service
        service = UserService(mock_api_client)
        result = service.get_user_summary(123)
        
        # Verify the result
        assert result == "Jane Doe (25 years old)"
        
        # Verify the mock was called correctly
        mock_api_client.fetch_user.assert_called_once_with(123)


# Performance Testing Example
class TestPerformance:
    """Example of performance testing."""
    
    def test_calculate_average_performance(self):
        """Test that calculate_average performs well with large lists."""
        large_list = list(range(10000))
        
        import time
        start_time = time.time()
        result = calculate_average(large_list)
        end_time = time.time()
        
        # Should complete in reasonable time (less than 0.1 seconds)
        assert end_time - start_time < 0.1
        assert result == 4999.5  # Average of 0 to 9999


# Test Configuration Examples
def test_setup_and_teardown():
    """Example of setup and teardown in tests."""
    # Setup
    test_file = '/tmp/test_file.txt'
    
    try:
        # Create test file
        with open(test_file, 'w') as f:
            f.write('test content')
        
        # Test
        with open(test_file, 'r') as f:
            content = f.read()
        
        assert content == 'test content'
        
    finally:
        # Teardown - cleanup
        import os
        if os.path.exists(test_file):
            os.remove(test_file)


# Example test data
if __name__ == "__main__":
    # This shows how to run tests programmatically
    # In practice, use: pytest test_examples.py
    
    print("To run these tests, use:")
    print("pip install pytest pytest-mock")
    print("pytest test_examples.py -v")
    print()
    print("Test coverage:")
    print("pip install pytest-cov")
    print("pytest test_examples.py --cov=. --cov-report=html")
    print()
    print("Available test markers:")
    print("- @pytest.mark.slow - for slow tests")
    print("- @pytest.mark.integration - for integration tests")
    print("- @pytest.mark.unit - for unit tests")
    print()
    print("Run specific test types:")
    print("pytest -m unit  # Run only unit tests")
    print("pytest -m 'not slow'  # Skip slow tests")