"""
Comprehensive Testing Examples in Python
Demonstrates unittest, pytest patterns, mocking, fixtures, and test strategies.
"""

import unittest
import pytest
from unittest.mock import Mock, patch, MagicMock, call
from typing import List, Dict, Any, Optional
import tempfile
import os
import json
import requests
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import warnings

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)


# Classes to be tested
@dataclass
class User:
    """User model for testing."""
    id: int
    username: str
    email: str
    active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'active': self.active
        }


class UserRepository:
    """User repository for database operations."""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self._users: Dict[int, User] = {}
        self._next_id = 1
    
    def create_user(self, username: str, email: str) -> User:
        """Create a new user."""
        if self.find_by_username(username):
            raise ValueError(f"Username '{username}' already exists")
        
        if self.find_by_email(email):
            raise ValueError(f"Email '{email}' already exists")
        
        user = User(self._next_id, username, email)
        self._users[user.id] = user
        self._next_id += 1
        return user
    
    def find_by_id(self, user_id: int) -> Optional[User]:
        """Find user by ID."""
        return self._users.get(user_id)
    
    def find_by_username(self, username: str) -> Optional[User]:
        """Find user by username."""
        for user in self._users.values():
            if user.username == username:
                return user
        return None
    
    def find_by_email(self, email: str) -> Optional[User]:
        """Find user by email."""
        for user in self._users.values():
            if user.email == email:
                return user
        return None
    
    def get_all_users(self) -> List[User]:
        """Get all users."""
        return list(self._users.values())
    
    def update_user(self, user_id: int, **kwargs) -> Optional[User]:
        """Update user."""
        user = self.find_by_id(user_id)
        if not user:
            return None
        
        for key, value in kwargs.items():
            if hasattr(user, key):
                setattr(user, key, value)
        
        return user
    
    def delete_user(self, user_id: int) -> bool:
        """Delete user."""
        if user_id in self._users:
            del self._users[user_id]
            return True
        return False


class UserService:
    """User service with business logic."""
    
    def __init__(self, repository: UserRepository):
        self.repository = repository
    
    def register_user(self, username: str, email: str) -> User:
        """Register a new user with validation."""
        # Validate input
        if not username or len(username) < 3:
            raise ValueError("Username must be at least 3 characters")
        
        if not email or '@' not in email:
            raise ValueError("Invalid email address")
        
        return self.repository.create_user(username, email)
    
    def deactivate_user(self, user_id: int) -> bool:
        """Deactivate a user."""
        user = self.repository.find_by_id(user_id)
        if not user:
            return False
        
        self.repository.update_user(user_id, active=False)
        return True
    
    def get_active_users(self) -> List[User]:
        """Get all active users."""
        all_users = self.repository.get_all_users()
        return [user for user in all_users if user.active]


class ExternalAPIClient:
    """External API client for testing HTTP requests."""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({'Authorization': f'Bearer {api_key}'})
    
    def get_user_data(self, user_id: int) -> Dict[str, Any]:
        """Get user data from external API."""
        response = self.session.get(f'{self.base_url}/users/{user_id}')
        response.raise_for_status()
        return response.json()
    
    def create_external_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create user in external system."""
        response = self.session.post(f'{self.base_url}/users', json=user_data)
        response.raise_for_status()
        return response.json()


# Test Fixtures and Utilities
class TestDataFactory:
    """Factory for creating test data."""
    
    @staticmethod
    def create_user(id: int = 1, username: str = "testuser", 
                   email: str = "test@example.com", active: bool = True) -> User:
        """Create a test user."""
        return User(id, username, email, active)
    
    @staticmethod
    def create_users(count: int) -> List[User]:
        """Create multiple test users."""
        return [
            TestDataFactory.create_user(
                id=i,
                username=f"user{i}",
                email=f"user{i}@example.com"
            )
            for i in range(1, count + 1)
        ]


# Unittest Examples
class TestUserModel(unittest.TestCase):
    """Test cases for User model using unittest."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.user = TestDataFactory.create_user()
    
    def test_user_creation(self):
        """Test user creation."""
        user = User(1, "testuser", "test@example.com")
        self.assertEqual(user.id, 1)
        self.assertEqual(user.username, "testuser")
        self.assertEqual(user.email, "test@example.com")
        self.assertTrue(user.active)
    
    def test_user_to_dict(self):
        """Test user serialization."""
        expected = {
            'id': 1,
            'username': 'testuser',
            'email': 'test@example.com',
            'active': True
        }
        self.assertEqual(self.user.to_dict(), expected)
    
    def test_user_equality(self):
        """Test user equality."""
        user1 = User(1, "test", "test@example.com")
        user2 = User(1, "test", "test@example.com")
        self.assertEqual(user1, user2)
    
    def tearDown(self):
        """Clean up after tests."""
        self.user = None


class TestUserRepository(unittest.TestCase):
    """Test cases for UserRepository using unittest."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.repository = UserRepository("test://database")
    
    def test_create_user_success(self):
        """Test successful user creation."""
        user = self.repository.create_user("testuser", "test@example.com")
        
        self.assertIsInstance(user, User)
        self.assertEqual(user.username, "testuser")
        self.assertEqual(user.email, "test@example.com")
        self.assertTrue(user.active)
        self.assertEqual(user.id, 1)
    
    def test_create_user_duplicate_username(self):
        """Test creating user with duplicate username."""
        self.repository.create_user("testuser", "test1@example.com")
        
        with self.assertRaises(ValueError) as context:
            self.repository.create_user("testuser", "test2@example.com")
        
        self.assertIn("Username 'testuser' already exists", str(context.exception))
    
    def test_create_user_duplicate_email(self):
        """Test creating user with duplicate email."""
        self.repository.create_user("user1", "test@example.com")
        
        with self.assertRaises(ValueError):
            self.repository.create_user("user2", "test@example.com")
    
    def test_find_by_id(self):
        """Test finding user by ID."""
        created_user = self.repository.create_user("testuser", "test@example.com")
        found_user = self.repository.find_by_id(created_user.id)
        
        self.assertEqual(found_user, created_user)
    
    def test_find_by_id_not_found(self):
        """Test finding non-existent user by ID."""
        found_user = self.repository.find_by_id(999)
        self.assertIsNone(found_user)
    
    def test_update_user(self):
        """Test updating user."""
        user = self.repository.create_user("testuser", "test@example.com")
        updated_user = self.repository.update_user(user.id, username="newuser")
        
        self.assertEqual(updated_user.username, "newuser")
        self.assertEqual(updated_user.email, "test@example.com")
    
    def test_delete_user(self):
        """Test deleting user."""
        user = self.repository.create_user("testuser", "test@example.com")
        
        # Delete user
        result = self.repository.delete_user(user.id)
        self.assertTrue(result)
        
        # Verify user is deleted
        found_user = self.repository.find_by_id(user.id)
        self.assertIsNone(found_user)
    
    def test_get_all_users(self):
        """Test getting all users."""
        users = TestDataFactory.create_users(3)
        
        for user in users:
            self.repository.create_user(user.username, user.email)
        
        all_users = self.repository.get_all_users()
        self.assertEqual(len(all_users), 3)


# Pytest Examples with Fixtures
@pytest.fixture
def user_repository():
    """Pytest fixture for user repository."""
    return UserRepository("test://database")


@pytest.fixture
def user_service(user_repository):
    """Pytest fixture for user service."""
    return UserService(user_repository)


@pytest.fixture
def sample_users():
    """Pytest fixture for sample users."""
    return TestDataFactory.create_users(5)


class TestUserServicePytest:
    """Test cases for UserService using pytest."""
    
    def test_register_user_success(self, user_service):
        """Test successful user registration."""
        user = user_service.register_user("testuser", "test@example.com")
        
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.active is True
    
    def test_register_user_invalid_username(self, user_service):
        """Test user registration with invalid username."""
        with pytest.raises(ValueError, match="Username must be at least 3 characters"):
            user_service.register_user("ab", "test@example.com")
    
    def test_register_user_invalid_email(self, user_service):
        """Test user registration with invalid email."""
        with pytest.raises(ValueError, match="Invalid email address"):
            user_service.register_user("testuser", "invalid-email")
    
    def test_deactivate_user_success(self, user_service):
        """Test successful user deactivation."""
        user = user_service.register_user("testuser", "test@example.com")
        result = user_service.deactivate_user(user.id)
        
        assert result is True
        
        # Verify user is deactivated
        found_user = user_service.repository.find_by_id(user.id)
        assert found_user.active is False
    
    def test_deactivate_nonexistent_user(self, user_service):
        """Test deactivating non-existent user."""
        result = user_service.deactivate_user(999)
        assert result is False
    
    def test_get_active_users(self, user_service):
        """Test getting active users."""
        # Create users
        user1 = user_service.register_user("user1", "user1@example.com")
        user2 = user_service.register_user("user2", "user2@example.com")
        user3 = user_service.register_user("user3", "user3@example.com")
        
        # Deactivate one user
        user_service.deactivate_user(user2.id)
        
        # Get active users
        active_users = user_service.get_active_users()
        
        assert len(active_users) == 2
        assert all(user.active for user in active_users)
        assert user1 in active_users
        assert user3 in active_users
        assert user2 not in active_users


# Mocking Examples
class TestExternalAPIClientMocking(unittest.TestCase):
    """Test cases demonstrating mocking with unittest.mock."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.api_client = ExternalAPIClient("https://api.example.com", "test-api-key")
    
    @patch('requests.Session.get')
    def test_get_user_data_success(self, mock_get):
        """Test successful API call with mocking."""
        # Configure mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            'id': 1,
            'name': 'John Doe',
            'email': 'john@example.com'
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Call method
        result = self.api_client.get_user_data(1)
        
        # Assertions
        self.assertEqual(result['id'], 1)
        self.assertEqual(result['name'], 'John Doe')
        
        # Verify mock was called correctly
        mock_get.assert_called_once_with('https://api.example.com/users/1')
    
    @patch('requests.Session.get')
    def test_get_user_data_http_error(self, mock_get):
        """Test API call with HTTP error."""
        # Configure mock to raise exception
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_get.return_value = mock_response
        
        # Test exception is raised
        with self.assertRaises(requests.HTTPError):
            self.api_client.get_user_data(999)
    
    @patch('requests.Session.post')
    def test_create_external_user(self, mock_post):
        """Test creating external user with mocking."""
        # Setup
        user_data = {'username': 'testuser', 'email': 'test@example.com'}
        expected_response = {'id': 123, **user_data}
        
        mock_response = Mock()
        mock_response.json.return_value = expected_response
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        # Execute
        result = self.api_client.create_external_user(user_data)
        
        # Verify
        self.assertEqual(result, expected_response)
        mock_post.assert_called_once_with(
            'https://api.example.com/users',
            json=user_data
        )


# Parametrized Tests with Pytest
class TestParametrizedExamples:
    """Examples of parametrized tests with pytest."""
    
    @pytest.mark.parametrize("username,email,should_succeed", [
        ("validuser", "valid@example.com", True),
        ("ab", "valid@example.com", False),  # Username too short
        ("validuser", "invalid-email", False),  # Invalid email
        ("", "valid@example.com", False),  # Empty username
        ("validuser", "", False),  # Empty email
    ])
    def test_user_registration_validation(self, user_service, username, email, should_succeed):
        """Parametrized test for user registration validation."""
        if should_succeed:
            user = user_service.register_user(username, email)
            assert user.username == username
            assert user.email == email
        else:
            with pytest.raises(ValueError):
                user_service.register_user(username, email)
    
    @pytest.mark.parametrize("user_count", [0, 1, 5, 10])
    def test_get_all_users_count(self, user_repository, user_count):
        """Parametrized test for user count."""
        # Create users
        for i in range(user_count):
            user_repository.create_user(f"user{i}", f"user{i}@example.com")
        
        # Verify count
        all_users = user_repository.get_all_users()
        assert len(all_users) == user_count


# Integration Tests
class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_full_user_workflow(self, user_service):
        """Test complete user workflow."""
        # Register user
        user = user_service.register_user("integrationuser", "integration@example.com")
        assert user.id is not None
        
        # Verify user exists
        found_user = user_service.repository.find_by_id(user.id)
        assert found_user == user
        
        # Update user
        updated_user = user_service.repository.update_user(
            user.id, 
            username="updateduser"
        )
        assert updated_user.username == "updateduser"
        
        # Deactivate user
        result = user_service.deactivate_user(user.id)
        assert result is True
        
        # Verify user is not in active users list
        active_users = user_service.get_active_users()
        assert updated_user not in active_users
        
        # Delete user
        delete_result = user_service.repository.delete_user(user.id)
        assert delete_result is True
        
        # Verify user is deleted
        deleted_user = user_service.repository.find_by_id(user.id)
        assert deleted_user is None


# Test Utilities and Helpers
class TestUtilities:
    """Utility functions for testing."""
    
    @staticmethod
    def create_temp_file_with_content(content: str) -> str:
        """Create temporary file with content."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write(content)
            return f.name
    
    @staticmethod
    def cleanup_temp_file(file_path: str):
        """Clean up temporary file."""
        if os.path.exists(file_path):
            os.unlink(file_path)


# Performance Tests
class TestPerformance:
    """Performance-related tests."""
    
    def test_bulk_user_creation_performance(self, user_repository):
        """Test performance of bulk user creation."""
        import time
        
        start_time = time.time()
        
        # Create 1000 users
        for i in range(1000):
            user_repository.create_user(f"user{i}", f"user{i}@example.com")
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert duration < 5.0, f"Bulk creation took too long: {duration:.2f} seconds"
        
        # Verify all users were created
        all_users = user_repository.get_all_users()
        assert len(all_users) == 1000


# Test Markers and Categories
@pytest.mark.slow
class TestSlowOperations:
    """Tests marked as slow operations."""
    
    def test_slow_operation(self):
        """Slow test that might be skipped in quick test runs."""
        import time
        time.sleep(2)  # Simulate slow operation
        assert True


@pytest.mark.integration
class TestExternalDependencies:
    """Tests that require external dependencies."""
    
    @pytest.mark.skip(reason="Requires external service")
    def test_external_service_integration(self):
        """Test that requires external service."""
        # This would test integration with real external service
        pass


if __name__ == "__main__":
    print("=== Running Test Examples ===")
    
    # Run unittest tests
    print("\n1. Running unittest tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Note: pytest tests would typically be run with: pytest -v
    print("\n2. To run pytest tests, use: pytest -v")
    print("   - Run with markers: pytest -v -m 'not slow'")
    print("   - Run specific test: pytest -v TestUserServicePytest::test_register_user_success")
    
    print("\n=== Testing Features Demonstrated ===")
    print("- unittest.TestCase with setUp/tearDown")
    print("- pytest fixtures and dependency injection")
    print("- Mocking with unittest.mock")
    print("- Parametrized tests")
    print("- Integration tests")
    print("- Performance tests")
    print("- Test markers and categories")
    print("- Exception testing")
    print("- Test data factories")
    print("- Temporary file handling")