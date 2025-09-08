# Security-Focused Code Examples Dataset
# Demonstrates security patterns, vulnerabilities, and best practices

class SecurityExamples:
    """Collection of security-focused code examples for AI training"""
    
    # ========== Input Validation Examples ==========
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Secure email validation with regex"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email)) and len(email) <= 320
    
    @staticmethod
    def sanitize_input(user_input: str) -> str:
        """Sanitize user input to prevent XSS"""
        import html
        # Remove potentially dangerous characters
        sanitized = html.escape(user_input)
        # Remove script tags
        sanitized = re.sub(r'<script.*?</script>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        return sanitized.strip()
    
    @staticmethod
    def validate_file_upload(filename: str, allowed_extensions: set) -> bool:
        """Validate file uploads securely"""
        import os
        # Check file extension
        _, ext = os.path.splitext(filename.lower())
        ext = ext[1:]  # Remove the dot
        
        # Prevent directory traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            return False
        
        # Check against allowed extensions
        return ext in allowed_extensions and len(filename) <= 255
    
    # ========== Authentication Examples ==========
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Secure password hashing with bcrypt"""
        import bcrypt
        salt = bcrypt.gensalt(rounds=12)
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Verify password against hash"""
        import bcrypt
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def generate_secure_token(length: int = 32) -> str:
        """Generate cryptographically secure random token"""
        import secrets
        return secrets.token_urlsafe(length)
    
    # ========== SQL Injection Prevention ==========
    
    @staticmethod
    def safe_database_query(cursor, user_id: int):
        """Parameterized query to prevent SQL injection"""
        # GOOD: Using parameterized queries
        query = "SELECT * FROM users WHERE id = ? AND active = 1"
        cursor.execute(query, (user_id,))
        return cursor.fetchall()
    
    @staticmethod
    def unsafe_database_query(cursor, user_id):
        """VULNERABLE: SQL injection vulnerability example"""
        # BAD: String concatenation - vulnerable to SQL injection
        query = f"SELECT * FROM users WHERE id = {user_id}"
        cursor.execute(query)
        return cursor.fetchall()
    
    # ========== Encryption Examples ==========
    
    @staticmethod
    def encrypt_sensitive_data(data: str, key: bytes) -> str:
        """Encrypt sensitive data using Fernet"""
        from cryptography.fernet import Fernet
        f = Fernet(key)
        encrypted_data = f.encrypt(data.encode())
        return encrypted_data.decode()
    
    @staticmethod
    def decrypt_sensitive_data(encrypted_data: str, key: bytes) -> str:
        """Decrypt sensitive data using Fernet"""
        from cryptography.fernet import Fernet
        f = Fernet(key)
        decrypted_data = f.decrypt(encrypted_data.encode())
        return decrypted_data.decode()
    
    # ========== Session Security ==========
    
    @staticmethod
    def create_secure_session():
        """Create secure session configuration"""
        session_config = {
            'secure': True,          # HTTPS only
            'httponly': True,        # Prevent XSS access to cookies
            'samesite': 'Strict',    # CSRF protection
            'max_age': 3600,         # 1 hour expiration
            'domain': None,          # Don't set domain for security
            'path': '/'              # Limit to application path
        }
        return session_config
    
    # ========== CSRF Protection ==========
    
    @staticmethod
    def generate_csrf_token() -> str:
        """Generate CSRF token"""
        import secrets
        return secrets.token_hex(32)
    
    @staticmethod
    def validate_csrf_token(provided_token: str, session_token: str) -> bool:
        """Validate CSRF token with timing-safe comparison"""
        import hmac
        return hmac.compare_digest(provided_token, session_token)
    
    # ========== Rate Limiting ==========
    
    class RateLimiter:
        """Simple rate limiter for API endpoints"""
        
        def __init__(self, max_requests: int = 100, window_seconds: int = 3600):
            self.max_requests = max_requests
            self.window_seconds = window_seconds
            self.requests = {}
        
        def is_allowed(self, identifier: str) -> bool:
            """Check if request is allowed"""
            import time
            now = time.time()
            
            if identifier not in self.requests:
                self.requests[identifier] = []
            
            # Remove old requests outside the window
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier]
                if now - req_time < self.window_seconds
            ]
            
            # Check if under limit
            if len(self.requests[identifier]) < self.max_requests:
                self.requests[identifier].append(now)
                return True
            
            return False
    
    # ========== Secure Headers ==========
    
    @staticmethod
    def get_security_headers():
        """Get recommended security headers"""
        return {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'",
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': 'camera=(), microphone=(), geolocation=()',
        }
    
    # ========== Security Validation ==========
    
    @staticmethod
    def validate_jwt_token(token: str, secret: str) -> dict:
        """Validate JWT token securely"""
        import jwt
        try:
            payload = jwt.decode(token, secret, algorithms=['HS256'])
            # Additional validation
            required_fields = ['user_id', 'exp', 'iat']
            if not all(field in payload for field in required_fields):
                raise ValueError("Missing required fields")
            return payload
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError:
            raise ValueError("Invalid token")


# ========== Performance Optimization Examples ==========

class PerformanceExamples:
    """Performance optimization patterns and examples"""
    
    # ========== Caching Strategies ==========
    
    class MemoryCache:
        """Simple in-memory cache with TTL"""
        
        def __init__(self):
            self.cache = {}
            self.expiry = {}
        
        def get(self, key: str):
            """Get item from cache"""
            import time
            if key in self.cache:
                if key not in self.expiry or time.time() < self.expiry[key]:
                    return self.cache[key]
                else:
                    # Expired
                    del self.cache[key]
                    del self.expiry[key]
            return None
        
        def set(self, key: str, value, ttl_seconds: int = 300):
            """Set item in cache with TTL"""
            import time
            self.cache[key] = value
            self.expiry[key] = time.time() + ttl_seconds
    
    # ========== Database Optimization ==========
    
    @staticmethod
    def optimized_batch_insert(cursor, records: list):
        """Batch insert for better performance"""
        if not records:
            return
        
        # Use executemany for batch operations
        query = "INSERT INTO users (name, email, age) VALUES (?, ?, ?)"
        cursor.executemany(query, records)
    
    @staticmethod
    def optimized_query_with_index(cursor, email_domain: str):
        """Optimized query using database indexes"""
        # Assumes index on email column
        query = """
        SELECT id, name, email 
        FROM users 
        WHERE email LIKE ? 
        ORDER BY created_at DESC 
        LIMIT 100
        """
        cursor.execute(query, (f'%{email_domain}',))
        return cursor.fetchall()
    
    # ========== Algorithm Optimization ==========
    
    @staticmethod
    def efficient_fibonacci(n: int) -> int:
        """Efficient fibonacci using memoization"""
        cache = {}
        
        def fib_helper(num):
            if num in cache:
                return cache[num]
            if num <= 1:
                return num
            cache[num] = fib_helper(num - 1) + fib_helper(num - 2)
            return cache[num]
        
        return fib_helper(n)
    
    @staticmethod
    def optimized_search(sorted_list: list, target) -> int:
        """Binary search for O(log n) performance"""
        left, right = 0, len(sorted_list) - 1
        
        while left <= right:
            mid = (left + right) // 2
            if sorted_list[mid] == target:
                return mid
            elif sorted_list[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return -1
    
    # ========== Async/Concurrency Patterns ==========
    
    @staticmethod
    async def parallel_api_calls(urls: list):
        """Make multiple API calls concurrently"""
        import asyncio
        import aiohttp
        
        async def fetch_url(session, url):
            try:
                async with session.get(url) as response:
                    return await response.text()
            except Exception as e:
                return f"Error: {e}"
        
        async with aiohttp.ClientSession() as session:
            tasks = [fetch_url(session, url) for url in urls]
            results = await asyncio.gather(*tasks)
            return results
    
    # ========== Memory Optimization ==========
    
    @staticmethod
    def memory_efficient_file_processing(filename: str):
        """Process large files efficiently using generators"""
        def read_file_chunks(file_path, chunk_size=8192):
            with open(file_path, 'r') as file:
                while True:
                    chunk = file.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
        
        # Process file in chunks to avoid memory issues
        total_chars = 0
        for chunk in read_file_chunks(filename):
            total_chars += len(chunk)
        
        return total_chars


# ========== Testing Best Practices Examples ==========

class TestingExamples:
    """Comprehensive testing patterns and examples"""
    
    # ========== Unit Testing Patterns ==========
    
    import unittest
    from unittest.mock import Mock, patch, MagicMock
    
    class UserServiceTest(unittest.TestCase):
        """Example unit test class with comprehensive patterns"""
        
        def setUp(self):
            """Set up test fixtures"""
            self.user_service = UserService()
            self.mock_db = Mock()
            self.user_service.db = self.mock_db
        
        def test_create_user_success(self):
            """Test successful user creation"""
            # Arrange
            user_data = {'name': 'Test User', 'email': 'test@example.com'}
            self.mock_db.save.return_value = {'id': 1, **user_data}
            
            # Act
            result = self.user_service.create_user(user_data)
            
            # Assert
            self.assertEqual(result['id'], 1)
            self.assertEqual(result['name'], 'Test User')
            self.mock_db.save.assert_called_once_with(user_data)
        
        def test_create_user_validation_error(self):
            """Test user creation with invalid data"""
            # Arrange
            invalid_data = {'name': '', 'email': 'invalid-email'}
            
            # Act & Assert
            with self.assertRaises(ValidationError):
                self.user_service.create_user(invalid_data)
        
        @patch('requests.get')
        def test_external_api_call(self, mock_get):
            """Test external API integration with mocking"""
            # Arrange
            mock_response = Mock()
            mock_response.json.return_value = {'status': 'success'}
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            
            # Act
            result = self.user_service.fetch_external_data()
            
            # Assert
            self.assertEqual(result['status'], 'success')
            mock_get.assert_called_once()
        
        def tearDown(self):
            """Clean up after tests"""
            self.user_service = None
            self.mock_db = None
    
    # ========== Integration Testing ==========
    
    class DatabaseIntegrationTest(unittest.TestCase):
        """Integration test example"""
        
        @classmethod
        def setUpClass(cls):
            """Set up test database"""
            cls.test_db = create_test_database()
        
        @classmethod
        def tearDownClass(cls):
            """Clean up test database"""
            cls.test_db.drop_all_tables()
        
        def setUp(self):
            """Reset database state for each test"""
            self.test_db.clear_all_data()
        
        def test_user_crud_operations(self):
            """Test complete CRUD operations"""
            # Create
            user_id = self.test_db.create_user('Test User', 'test@example.com')
            self.assertIsNotNone(user_id)
            
            # Read
            user = self.test_db.get_user(user_id)
            self.assertEqual(user['name'], 'Test User')
            
            # Update
            self.test_db.update_user(user_id, {'name': 'Updated User'})
            updated_user = self.test_db.get_user(user_id)
            self.assertEqual(updated_user['name'], 'Updated User')
            
            # Delete
            result = self.test_db.delete_user(user_id)
            self.assertTrue(result)
            self.assertIsNone(self.test_db.get_user(user_id))
    
    # ========== Performance Testing ==========
    
    @staticmethod
    def benchmark_function(func, *args, iterations=1000):
        """Benchmark function performance"""
        import time
        
        start_time = time.time()
        for _ in range(iterations):
            func(*args)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / iterations
        
        return {
            'total_time': total_time,
            'average_time': avg_time,
            'iterations': iterations,
            'ops_per_second': iterations / total_time
        }
    
    # ========== Test Data Factories ==========
    
    class TestDataFactory:
        """Factory for creating test data"""
        
        @staticmethod
        def create_user(**kwargs):
            """Create test user with default values"""
            defaults = {
                'name': 'Test User',
                'email': 'test@example.com',
                'age': 25,
                'active': True
            }
            defaults.update(kwargs)
            return defaults
        
        @staticmethod
        def create_users(count=5):
            """Create multiple test users"""
            return [
                TestDataFactory.create_user(
                    name=f'User {i}',
                    email=f'user{i}@example.com',
                    age=20 + i
                )
                for i in range(1, count + 1)
            ]
    
    # ========== Property-Based Testing ==========
    
    @staticmethod
    def test_sorting_property(sort_function):
        """Property-based test for sorting functions"""
        import random
        
        # Generate random test data
        for _ in range(100):
            data = [random.randint(1, 1000) for _ in range(random.randint(1, 100))]
            original_data = data.copy()
            
            # Test the property: sorted list should be in order
            sorted_data = sort_function(data)
            
            # Properties that should hold
            assert len(sorted_data) == len(original_data)
            assert sorted(original_data) == sorted_data
            assert all(sorted_data[i] <= sorted_data[i+1] for i in range(len(sorted_data)-1))


# Example usage and demonstration
if __name__ == "__main__":
    print("=== Security, Performance, and Testing Examples ===")
    
    # Security examples
    security = SecurityExamples()
    print(f"Valid email: {security.validate_email('test@example.com')}")
    print(f"CSRF token: {security.generate_csrf_token()}")
    
    # Performance examples
    perf = PerformanceExamples()
    cache = perf.MemoryCache()
    cache.set("key1", "value1", 60)
    print(f"Cached value: {cache.get('key1')}")
    
    # Testing examples
    testing = TestingExamples()
    factory = testing.TestDataFactory()
    test_users = factory.create_users(3)
    print(f"Created {len(test_users)} test users")
    
    print("Examples demonstrate security, performance, and testing best practices!")