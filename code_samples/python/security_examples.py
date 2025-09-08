#!/usr/bin/env python3
"""
Security-focused code examples for AI training dataset.
Demonstrates secure coding practices, vulnerability prevention, and security patterns.
"""

import hashlib
import secrets
import hmac
import base64
import os
import re
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import bcrypt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class SecurePasswordManager:
    """Demonstrates secure password handling and storage."""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """
        Securely hash a password using bcrypt.
        
        Best practices:
        - Use a strong hashing algorithm (bcrypt, scrypt, or Argon2)
        - Salt is automatically handled by bcrypt
        - Cost factor of 12 provides good security vs performance balance
        """
        # Generate salt and hash password
        salt = bcrypt.gensalt(rounds=12)
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """
        Verify a password against its hash.
        
        Security considerations:
        - Always use constant-time comparison
        - bcrypt.checkpw is designed to be timing-attack resistant
        """
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        except ValueError:
            # Invalid hash format
            return False
    
    @staticmethod
    def generate_secure_password(length: int = 16) -> str:
        """
        Generate a cryptographically secure random password.
        
        Uses secrets module for cryptographically strong random generation.
        """
        if length < 8:
            raise ValueError("Password length should be at least 8 characters")
        
        # Character sets for password generation
        lowercase = 'abcdefghijklmnopqrstuvwxyz'
        uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        digits = '0123456789'
        special = '!@#$%^&*()_+-=[]{}|;:,.<>?'
        
        # Ensure at least one character from each set
        password = [
            secrets.choice(lowercase),
            secrets.choice(uppercase),
            secrets.choice(digits),
            secrets.choice(special)
        ]
        
        # Fill the rest randomly
        all_chars = lowercase + uppercase + digits + special
        for _ in range(length - 4):
            password.append(secrets.choice(all_chars))
        
        # Shuffle the password
        secrets.SystemRandom().shuffle(password)
        return ''.join(password)


class SecureDataEncryption:
    """Demonstrates secure data encryption and key management."""
    
    def __init__(self):
        self.key = None
    
    def generate_key(self) -> bytes:
        """Generate a new encryption key."""
        self.key = Fernet.generate_key()
        return self.key
    
    def derive_key_from_password(self, password: str, salt: Optional[bytes] = None) -> bytes:
        """
        Derive an encryption key from a password using PBKDF2.
        
        Security best practices:
        - Use a strong key derivation function
        - Use a random salt
        - Use sufficient iterations (100,000+)
        """
        if salt is None:
            salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        self.key = key
        return key, salt
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt data using Fernet (AES 128 in CBC mode with HMAC)."""
        if self.key is None:
            raise ValueError("No encryption key available. Generate or derive a key first.")
        
        f = Fernet(self.key)
        encrypted_data = f.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted_data).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt data."""
        if self.key is None:
            raise ValueError("No encryption key available.")
        
        try:
            f = Fernet(self.key)
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = f.decrypt(encrypted_bytes)
            return decrypted_data.decode()
        except Exception:
            raise ValueError("Failed to decrypt data. Invalid key or corrupted data.")


class InputValidator:
    """Demonstrates secure input validation and sanitization."""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """
        Validate email format using regex.
        
        Note: This is a basic validation. For production, use a proper
        email validation library or service.
        """
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @staticmethod
    def sanitize_sql_input(user_input: str) -> str:
        """
        Basic SQL injection prevention through input sanitization.
        
        WARNING: This is NOT sufficient for SQL injection prevention.
        Always use parameterized queries/prepared statements.
        This is for educational purposes only.
        """
        # Remove or escape dangerous SQL characters
        dangerous_chars = ["'", '"', ';', '--', '/*', '*/', 'xp_', 'sp_']
        sanitized = user_input
        
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        
        return sanitized
    
    @staticmethod
    def validate_file_upload(filename: str, max_size: int = 5 * 1024 * 1024) -> bool:
        """
        Validate file upload security.
        
        Security checks:
        - File extension whitelist
        - File size limit
        - Filename sanitization
        """
        # Allowed file extensions
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.pdf', '.txt', '.doc', '.docx'}
        
        # Check file extension
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in allowed_extensions:
            return False
        
        # Check for path traversal attempts
        if '..' in filename or '/' in filename or '\\' in filename:
            return False
        
        # Additional checks would include file size validation
        # (this would be done when processing the actual file)
        
        return True
    
    @staticmethod
    def sanitize_html_input(html_input: str) -> str:
        """
        Basic HTML sanitization to prevent XSS attacks.
        
        WARNING: This is a basic example. For production, use a proper
        HTML sanitization library like bleach.
        """
        # Remove script tags
        html_input = re.sub(r'<script[^>]*>.*?</script>', '', html_input, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove dangerous attributes
        html_input = re.sub(r'on\w+\s*=\s*["\'][^"\']*["\']', '', html_input, flags=re.IGNORECASE)
        
        # Remove javascript: links
        html_input = re.sub(r'javascript:', '', html_input, flags=re.IGNORECASE)
        
        return html_input


class SecureSessionManager:
    """Demonstrates secure session management."""
    
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.session_timeout = timedelta(hours=2)
    
    def create_session(self, user_id: str) -> str:
        """
        Create a new secure session.
        
        Security practices:
        - Use cryptographically secure random session IDs
        - Set reasonable session timeouts
        - Store minimal data in sessions
        """
        session_id = self._generate_session_id()
        
        session_data = {
            'user_id': user_id,
            'created_at': datetime.now(),
            'last_accessed': datetime.now(),
            'csrf_token': self._generate_csrf_token()
        }
        
        self.sessions[session_id] = session_data
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Validate and refresh a session.
        
        Security checks:
        - Session exists
        - Session not expired
        - Update last accessed time
        """
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        now = datetime.now()
        
        # Check if session is expired
        if now - session['last_accessed'] > self.session_timeout:
            self.destroy_session(session_id)
            return None
        
        # Update last accessed time
        session['last_accessed'] = now
        return session
    
    def destroy_session(self, session_id: str) -> bool:
        """Destroy a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def _generate_session_id(self) -> str:
        """Generate a cryptographically secure session ID."""
        return secrets.token_urlsafe(32)
    
    def _generate_csrf_token(self) -> str:
        """Generate a CSRF token for the session."""
        return secrets.token_urlsafe(32)


class SecureAPIClient:
    """Demonstrates secure API communication."""
    
    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
    
    def create_request_signature(self, method: str, url: str, body: str, timestamp: str) -> str:
        """
        Create HMAC signature for API request authentication.
        
        Security practices:
        - Use HMAC for request signing
        - Include timestamp to prevent replay attacks
        - Sign all relevant request data
        """
        # Create string to sign
        string_to_sign = f"{method}\n{url}\n{body}\n{timestamp}"
        
        # Create HMAC signature
        signature = hmac.new(
            self.secret_key.encode(),
            string_to_sign.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def validate_webhook_signature(self, payload: str, signature: str) -> bool:
        """
        Validate webhook signature to ensure it's from trusted source.
        
        Prevents webhook spoofing attacks.
        """
        expected_signature = hmac.new(
            self.secret_key.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Use constant-time comparison to prevent timing attacks
        return hmac.compare_digest(signature, expected_signature)


class RateLimiter:
    """
    Demonstrates rate limiting for API security.
    Prevents abuse and DoS attacks.
    """
    
    def __init__(self, max_requests: int = 100, time_window: int = 3600):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests allowed in time window
            time_window: Time window in seconds (default: 1 hour)
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: Dict[str, List[datetime]] = {}
    
    def is_allowed(self, identifier: str) -> bool:
        """
        Check if request is allowed based on rate limit.
        
        Args:
            identifier: Usually IP address or user ID
        """
        now = datetime.now()
        
        # Clean old requests
        if identifier in self.requests:
            cutoff_time = now - timedelta(seconds=self.time_window)
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier]
                if req_time > cutoff_time
            ]
        else:
            self.requests[identifier] = []
        
        # Check if under limit
        if len(self.requests[identifier]) < self.max_requests:
            self.requests[identifier].append(now)
            return True
        
        return False
    
    def get_reset_time(self, identifier: str) -> Optional[datetime]:
        """Get when the rate limit resets for an identifier."""
        if identifier not in self.requests or not self.requests[identifier]:
            return None
        
        oldest_request = min(self.requests[identifier])
        return oldest_request + timedelta(seconds=self.time_window)


# Security testing utilities

def test_password_security():
    """Test password security functions."""
    print("=== Password Security Tests ===")
    
    # Test password generation
    password = SecurePasswordManager.generate_secure_password(16)
    print(f"Generated secure password: {password}")
    
    # Test password hashing
    test_password = "MySecurePassword123!"
    hashed = SecurePasswordManager.hash_password(test_password)
    print(f"Password hash: {hashed[:50]}...")
    
    # Test password verification
    is_valid = SecurePasswordManager.verify_password(test_password, hashed)
    print(f"Password verification: {is_valid}")
    
    # Test with wrong password
    is_invalid = SecurePasswordManager.verify_password("WrongPassword", hashed)
    print(f"Wrong password verification: {is_invalid}")


def test_encryption():
    """Test data encryption."""
    print("\n=== Encryption Tests ===")
    
    encryptor = SecureDataEncryption()
    
    # Test key generation
    key = encryptor.generate_key()
    print(f"Generated key: {key[:20]}...")
    
    # Test encryption/decryption
    sensitive_data = "This is sensitive information that needs encryption"
    encrypted = encryptor.encrypt_data(sensitive_data)
    print(f"Encrypted data: {encrypted[:50]}...")
    
    decrypted = encryptor.decrypt_data(encrypted)
    print(f"Decrypted data: {decrypted}")
    print(f"Data integrity: {sensitive_data == decrypted}")


def test_input_validation():
    """Test input validation functions."""
    print("\n=== Input Validation Tests ===")
    
    # Test email validation
    valid_email = "user@example.com"
    invalid_email = "invalid.email"
    print(f"'{valid_email}' is valid email: {InputValidator.validate_email(valid_email)}")
    print(f"'{invalid_email}' is valid email: {InputValidator.validate_email(invalid_email)}")
    
    # Test file upload validation
    safe_file = "document.pdf"
    unsafe_file = "../../../etc/passwd"
    print(f"'{safe_file}' is safe upload: {InputValidator.validate_file_upload(safe_file)}")
    print(f"'{unsafe_file}' is safe upload: {InputValidator.validate_file_upload(unsafe_file)}")
    
    # Test HTML sanitization
    malicious_html = '<script>alert("XSS")</script><p onclick="alert(\'clicked\')">Click me</p>'
    sanitized = InputValidator.sanitize_html_input(malicious_html)
    print(f"Sanitized HTML: {sanitized}")


def test_session_management():
    """Test session management."""
    print("\n=== Session Management Tests ===")
    
    session_mgr = SecureSessionManager()
    
    # Create session
    session_id = session_mgr.create_session("user123")
    print(f"Created session: {session_id[:20]}...")
    
    # Validate session
    session_data = session_mgr.validate_session(session_id)
    print(f"Session valid: {session_data is not None}")
    if session_data:
        print(f"User ID: {session_data['user_id']}")
        print(f"CSRF token: {session_data['csrf_token'][:20]}...")
    
    # Destroy session
    destroyed = session_mgr.destroy_session(session_id)
    print(f"Session destroyed: {destroyed}")


def test_rate_limiting():
    """Test rate limiting."""
    print("\n=== Rate Limiting Tests ===")
    
    # Create rate limiter: 5 requests per 10 seconds
    rate_limiter = RateLimiter(max_requests=5, time_window=10)
    
    client_ip = "192.168.1.100"
    
    # Test multiple requests
    for i in range(7):
        allowed = rate_limiter.is_allowed(client_ip)
        print(f"Request {i+1}: {'Allowed' if allowed else 'Rate limited'}")


if __name__ == "__main__":
    print("Security-focused Code Examples for AI Training")
    print("=" * 50)
    
    test_password_security()
    test_encryption()
    test_input_validation()
    test_session_management()
    test_rate_limiting()
    
    print("\n=== Security Best Practices Summary ===")
    print("1. Always hash passwords with strong algorithms (bcrypt, scrypt, Argon2)")
    print("2. Use cryptographically secure random number generation")
    print("3. Validate and sanitize all user inputs")
    print("4. Implement proper session management with timeouts")
    print("5. Use HMAC for request signing and validation")
    print("6. Implement rate limiting to prevent abuse")
    print("7. Use parameterized queries to prevent SQL injection")
    print("8. Sanitize HTML output to prevent XSS")
    print("9. Use HTTPS for all communication")
    print("10. Implement proper error handling without information leakage")