"""
Security and Cryptography Examples in Python
Demonstrates password hashing, encryption, JWT tokens, and security best practices.
"""

import hashlib
import hmac
import secrets
import base64
import os
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import logging
from dataclasses import dataclass
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import bcrypt
import jwt
from urllib.parse import quote_plus, unquote_plus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TokenPayload:
    """JWT token payload structure."""
    user_id: int
    username: str
    email: str
    roles: list
    issued_at: datetime
    expires_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'user_id': self.user_id,
            'username': self.username,
            'email': self.email,
            'roles': self.roles,
            'iat': int(self.issued_at.timestamp()),
            'exp': int(self.expires_at.timestamp())
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TokenPayload':
        return cls(
            user_id=data['user_id'],
            username=data['username'],
            email=data['email'],
            roles=data['roles'],
            issued_at=datetime.fromtimestamp(data['iat']),
            expires_at=datetime.fromtimestamp(data['exp'])
        )


class PasswordManager:
    """Secure password hashing and verification using bcrypt."""
    
    @staticmethod
    def hash_password(password: str, rounds: int = 12) -> str:
        """Hash a password using bcrypt with salt."""
        # Generate salt and hash password
        salt = bcrypt.gensalt(rounds=rounds)
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    @staticmethod
    def verify_password(password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False
    
    @staticmethod
    def generate_secure_password(length: int = 16) -> str:
        """Generate a cryptographically secure random password."""
        # Use a mix of characters for password generation
        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    @staticmethod
    def check_password_strength(password: str) -> Dict[str, Any]:
        """Check password strength and return analysis."""
        analysis = {
            'length': len(password),
            'has_uppercase': any(c.isupper() for c in password),
            'has_lowercase': any(c.islower() for c in password),
            'has_digits': any(c.isdigit() for c in password),
            'has_special': any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password),
            'score': 0,
            'strength': 'Very Weak'
        }
        
        # Calculate score
        if analysis['length'] >= 8:
            analysis['score'] += 1
        if analysis['length'] >= 12:
            analysis['score'] += 1
        if analysis['has_uppercase']:
            analysis['score'] += 1
        if analysis['has_lowercase']:
            analysis['score'] += 1
        if analysis['has_digits']:
            analysis['score'] += 1
        if analysis['has_special']:
            analysis['score'] += 1
        
        # Determine strength
        if analysis['score'] >= 5:
            analysis['strength'] = 'Very Strong'
        elif analysis['score'] >= 4:
            analysis['strength'] = 'Strong'
        elif analysis['score'] >= 3:
            analysis['strength'] = 'Medium'
        elif analysis['score'] >= 2:
            analysis['strength'] = 'Weak'
        
        return analysis


class SymmetricEncryption:
    """Symmetric encryption using Fernet (AES 128 with HMAC)."""
    
    def __init__(self, key: bytes = None):
        """Initialize with key or generate new one."""
        if key is None:
            key = Fernet.generate_key()
        self.fernet = Fernet(key)
        self.key = key
    
    @classmethod
    def from_password(cls, password: str, salt: bytes = None) -> 'SymmetricEncryption':
        """Create encryption instance from password using PBKDF2."""
        if salt is None:
            salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        
        instance = cls(key)
        instance.salt = salt
        return instance
    
    def encrypt(self, data: str) -> str:
        """Encrypt string data."""
        encrypted = self.fernet.encrypt(data.encode('utf-8'))
        return base64.urlsafe_b64encode(encrypted).decode('utf-8')
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt string data."""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode('utf-8'))
            decrypted = self.fernet.decrypt(encrypted_bytes)
            return decrypted.decode('utf-8')
        except Exception as e:
            logger.error(f"Decryption error: {e}")
            raise ValueError("Invalid encrypted data or key")
    
    def encrypt_file(self, input_path: str, output_path: str):
        """Encrypt a file."""
        try:
            with open(input_path, 'rb') as file:
                data = file.read()
            
            encrypted_data = self.fernet.encrypt(data)
            
            with open(output_path, 'wb') as file:
                file.write(encrypted_data)
            
            logger.info(f"File encrypted: {input_path} -> {output_path}")
        except Exception as e:
            logger.error(f"File encryption error: {e}")
            raise
    
    def decrypt_file(self, input_path: str, output_path: str):
        """Decrypt a file."""
        try:
            with open(input_path, 'rb') as file:
                encrypted_data = file.read()
            
            decrypted_data = self.fernet.decrypt(encrypted_data)
            
            with open(output_path, 'wb') as file:
                file.write(decrypted_data)
            
            logger.info(f"File decrypted: {input_path} -> {output_path}")
        except Exception as e:
            logger.error(f"File decryption error: {e}")
            raise


class AsymmetricEncryption:
    """Asymmetric encryption using RSA."""
    
    def __init__(self, key_size: int = 2048):
        """Generate RSA key pair."""
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
        )
        self.public_key = self.private_key.public_key()
    
    def get_public_key_pem(self) -> str:
        """Get public key in PEM format."""
        pem = self.public_key.serialize(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return pem.decode('utf-8')
    
    def get_private_key_pem(self, password: bytes = None) -> str:
        """Get private key in PEM format."""
        encryption_algorithm = serialization.NoEncryption()
        if password:
            encryption_algorithm = serialization.BestAvailableEncryption(password)
        
        pem = self.private_key.serialize(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=encryption_algorithm
        )
        return pem.decode('utf-8')
    
    @classmethod
    def from_pem(cls, private_key_pem: str, password: bytes = None) -> 'AsymmetricEncryption':
        """Load from PEM-encoded private key."""
        instance = cls.__new__(cls)
        instance.private_key = serialization.load_pem_private_key(
            private_key_pem.encode('utf-8'),
            password=password,
        )
        instance.public_key = instance.private_key.public_key()
        return instance
    
    def encrypt(self, data: str) -> str:
        """Encrypt data with public key."""
        try:
            encrypted = self.public_key.encrypt(
                data.encode('utf-8'),
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            return base64.b64encode(encrypted).decode('utf-8')
        except Exception as e:
            logger.error(f"RSA encryption error: {e}")
            raise
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt data with private key."""
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
            decrypted = self.private_key.decrypt(
                encrypted_bytes,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            return decrypted.decode('utf-8')
        except Exception as e:
            logger.error(f"RSA decryption error: {e}")
            raise
    
    def sign(self, data: str) -> str:
        """Sign data with private key."""
        signature = self.private_key.sign(
            data.encode('utf-8'),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return base64.b64encode(signature).decode('utf-8')
    
    def verify_signature(self, data: str, signature: str) -> bool:
        """Verify signature with public key."""
        try:
            signature_bytes = base64.b64decode(signature.encode('utf-8'))
            self.public_key.verify(
                signature_bytes,
                data.encode('utf-8'),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception as e:
            logger.error(f"Signature verification failed: {e}")
            return False


class JWTManager:
    """JSON Web Token management for authentication."""
    
    def __init__(self, secret_key: str, algorithm: str = 'HS256'):
        self.secret_key = secret_key
        self.algorithm = algorithm
    
    def create_token(self, payload: TokenPayload, expires_hours: int = 24) -> str:
        """Create a JWT token."""
        try:
            token_data = payload.to_dict()
            return jwt.encode(token_data, self.secret_key, algorithm=self.algorithm)
        except Exception as e:
            logger.error(f"Token creation error: {e}")
            raise
    
    def verify_token(self, token: str) -> Optional[TokenPayload]:
        """Verify and decode JWT token."""
        try:
            decoded = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return TokenPayload.from_dict(decoded)
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
    
    def refresh_token(self, token: str, expires_hours: int = 24) -> Optional[str]:
        """Refresh an existing token."""
        payload = self.verify_token(token)
        if payload:
            # Create new token with extended expiration
            new_expires = datetime.utcnow() + timedelta(hours=expires_hours)
            payload.expires_at = new_expires
            payload.issued_at = datetime.utcnow()
            return self.create_token(payload, expires_hours)
        return None
    
    @staticmethod
    def generate_secret_key() -> str:
        """Generate a secure secret key for JWT."""
        return secrets.token_urlsafe(32)


class HashingUtilities:
    """Various hashing and integrity verification utilities."""
    
    @staticmethod
    def sha256_hash(data: str) -> str:
        """Generate SHA-256 hash of string."""
        return hashlib.sha256(data.encode('utf-8')).hexdigest()
    
    @staticmethod
    def md5_hash(data: str) -> str:
        """Generate MD5 hash of string (for non-security purposes)."""
        return hashlib.md5(data.encode('utf-8')).hexdigest()
    
    @staticmethod
    def file_hash(file_path: str, algorithm: str = 'sha256') -> str:
        """Calculate hash of a file."""
        hash_func = hashlib.new(algorithm)
        
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_func.update(chunk)
            return hash_func.hexdigest()
        except Exception as e:
            logger.error(f"File hashing error: {e}")
            raise
    
    @staticmethod
    def hmac_signature(message: str, secret_key: str, algorithm: str = 'sha256') -> str:
        """Generate HMAC signature."""
        key_bytes = secret_key.encode('utf-8')
        message_bytes = message.encode('utf-8')
        
        if algorithm == 'sha256':
            signature = hmac.new(key_bytes, message_bytes, hashlib.sha256)
        elif algorithm == 'sha1':
            signature = hmac.new(key_bytes, message_bytes, hashlib.sha1)
        else:
            raise ValueError(f"Unsupported HMAC algorithm: {algorithm}")
        
        return signature.hexdigest()
    
    @staticmethod
    def verify_hmac_signature(message: str, signature: str, secret_key: str, algorithm: str = 'sha256') -> bool:
        """Verify HMAC signature."""
        expected_signature = HashingUtilities.hmac_signature(message, secret_key, algorithm)
        return hmac.compare_digest(signature, expected_signature)


class SecureSession:
    """Secure session management with encryption and signing."""
    
    def __init__(self, secret_key: str):
        self.encryption = SymmetricEncryption.from_password(secret_key)
        self.secret_key = secret_key
    
    def create_session(self, user_data: Dict[str, Any], expires_minutes: int = 30) -> str:
        """Create encrypted and signed session data."""
        session_data = {
            'user_data': user_data,
            'created_at': datetime.utcnow().isoformat(),
            'expires_at': (datetime.utcnow() + timedelta(minutes=expires_minutes)).isoformat()
        }
        
        # Serialize and encrypt
        serialized = json.dumps(session_data)
        encrypted = self.encryption.encrypt(serialized)
        
        # Create HMAC signature
        signature = HashingUtilities.hmac_signature(encrypted, self.secret_key)
        
        # Combine encrypted data and signature
        session_token = f"{encrypted}.{signature}"
        return base64.urlsafe_b64encode(session_token.encode()).decode()
    
    def verify_session(self, session_token: str) -> Optional[Dict[str, Any]]:
        """Verify and decrypt session data."""
        try:
            # Decode and split
            decoded = base64.urlsafe_b64decode(session_token.encode()).decode()
            encrypted_data, signature = decoded.rsplit('.', 1)
            
            # Verify signature
            if not HashingUtilities.verify_hmac_signature(encrypted_data, signature, self.secret_key):
                logger.warning("Session signature verification failed")
                return None
            
            # Decrypt data
            decrypted = self.encryption.decrypt(encrypted_data)
            session_data = json.loads(decrypted)
            
            # Check expiration
            expires_at = datetime.fromisoformat(session_data['expires_at'])
            if datetime.utcnow() > expires_at:
                logger.warning("Session has expired")
                return None
            
            return session_data['user_data']
            
        except Exception as e:
            logger.error(f"Session verification error: {e}")
            return None


class SecurityUtils:
    """General security utilities and best practices."""
    
    @staticmethod
    def generate_csrf_token() -> str:
        """Generate CSRF token."""
        return secrets.token_urlsafe(32)
    
    @staticmethod
    def generate_api_key() -> str:
        """Generate API key."""
        return secrets.token_hex(32)
    
    @staticmethod
    def secure_compare(a: str, b: str) -> bool:
        """Timing-safe string comparison."""
        return hmac.compare_digest(a, b)
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for security."""
        # Remove potentially dangerous characters
        import re
        sanitized = re.sub(r'[^\w\-_\.]', '', filename)
        # Prevent directory traversal
        sanitized = sanitized.replace('..', '')
        return sanitized[:255]  # Limit length
    
    @staticmethod
    def escape_html(text: str) -> str:
        """Escape HTML to prevent XSS."""
        escape_dict = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#x27;',
            '/': '&#x2F;',
        }
        return ''.join(escape_dict.get(c, c) for c in text)
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Basic email validation."""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @staticmethod
    def rate_limit_key(identifier: str, action: str) -> str:
        """Generate rate limiting key."""
        return f"rate_limit:{action}:{HashingUtilities.sha256_hash(identifier)}"


def demonstrate_password_security():
    """Demonstrate password hashing and strength checking."""
    print("=== Password Security Demo ===")
    
    # Password strength checking
    passwords = ["123456", "password123", "StrongP@ssw0rd!", "VerySecureP@ssw0rd123!"]
    
    for password in passwords:
        strength = PasswordManager.check_password_strength(password)
        print(f"Password: {password}")
        print(f"  Strength: {strength['strength']} (Score: {strength['score']}/6)")
        print(f"  Analysis: {strength}")
        print()
    
    # Password hashing and verification
    password = "my_secure_password"
    hashed = PasswordManager.hash_password(password)
    print(f"Original: {password}")
    print(f"Hashed: {hashed}")
    print(f"Verification: {PasswordManager.verify_password(password, hashed)}")
    print(f"Wrong password: {PasswordManager.verify_password('wrong', hashed)}")
    
    # Generate secure password
    secure_password = PasswordManager.generate_secure_password(16)
    print(f"Generated secure password: {secure_password}")


def demonstrate_encryption():
    """Demonstrate symmetric and asymmetric encryption."""
    print("\n=== Encryption Demo ===")
    
    # Symmetric encryption
    print("Symmetric Encryption (Fernet):")
    symmetric = SymmetricEncryption()
    
    original_text = "This is a secret message that needs to be encrypted!"
    encrypted = symmetric.encrypt(original_text)
    decrypted = symmetric.decrypt(encrypted)
    
    print(f"Original: {original_text}")
    print(f"Encrypted: {encrypted[:50]}...")
    print(f"Decrypted: {decrypted}")
    print(f"Match: {original_text == decrypted}")
    
    # Password-based encryption
    print("\nPassword-based Encryption:")
    password_encryption = SymmetricEncryption.from_password("my_password")
    encrypted_with_password = password_encryption.encrypt(original_text)
    decrypted_with_password = password_encryption.decrypt(encrypted_with_password)
    print(f"Password-based encryption successful: {original_text == decrypted_with_password}")
    
    # Asymmetric encryption
    print("\nAsymmetric Encryption (RSA):")
    asymmetric = AsymmetricEncryption()
    
    short_message = "Secret message"  # RSA has length limitations
    rsa_encrypted = asymmetric.encrypt(short_message)
    rsa_decrypted = asymmetric.decrypt(rsa_encrypted)
    
    print(f"RSA Original: {short_message}")
    print(f"RSA Decrypted: {rsa_decrypted}")
    print(f"RSA Match: {short_message == rsa_decrypted}")
    
    # Digital signature
    signature = asymmetric.sign(short_message)
    signature_valid = asymmetric.verify_signature(short_message, signature)
    print(f"Digital signature valid: {signature_valid}")


def demonstrate_jwt_tokens():
    """Demonstrate JWT token management."""
    print("\n=== JWT Token Demo ===")
    
    # Create JWT manager
    secret_key = JWTManager.generate_secret_key()
    jwt_manager = JWTManager(secret_key)
    
    # Create token payload
    payload = TokenPayload(
        user_id=123,
        username="john_doe",
        email="john@example.com",
        roles=["user", "admin"],
        issued_at=datetime.utcnow(),
        expires_at=datetime.utcnow() + timedelta(hours=1)
    )
    
    # Create and verify token
    token = jwt_manager.create_token(payload)
    print(f"JWT Token: {token[:50]}...")
    
    # Verify token
    verified_payload = jwt_manager.verify_token(token)
    if verified_payload:
        print(f"Token verified successfully!")
        print(f"User: {verified_payload.username} ({verified_payload.email})")
        print(f"Roles: {verified_payload.roles}")
        print(f"Expires: {verified_payload.expires_at}")
    
    # Refresh token
    refreshed_token = jwt_manager.refresh_token(token)
    print(f"Token refreshed: {refreshed_token is not None}")


def demonstrate_hashing():
    """Demonstrate various hashing techniques."""
    print("\n=== Hashing Demo ===")
    
    message = "Hello, World!"
    
    # Basic hashing
    sha256_hash = HashingUtilities.sha256_hash(message)
    md5_hash = HashingUtilities.md5_hash(message)
    
    print(f"Message: {message}")
    print(f"SHA-256: {sha256_hash}")
    print(f"MD5: {md5_hash}")
    
    # HMAC signature
    secret_key = "my_secret_key"
    hmac_sig = HashingUtilities.hmac_signature(message, secret_key)
    signature_valid = HashingUtilities.verify_hmac_signature(message, hmac_sig, secret_key)
    
    print(f"HMAC Signature: {hmac_sig}")
    print(f"HMAC Valid: {signature_valid}")
    
    # File hashing example
    test_file = "/tmp/test_security.txt"
    with open(test_file, 'w') as f:
        f.write("This is a test file for hashing.")
    
    file_hash = HashingUtilities.file_hash(test_file)
    print(f"File hash: {file_hash}")


def demonstrate_secure_sessions():
    """Demonstrate secure session management."""
    print("\n=== Secure Session Demo ===")
    
    # Create secure session manager
    session_manager = SecureSession("session_secret_key")
    
    # Create session
    user_data = {
        "user_id": 456,
        "username": "jane_doe",
        "role": "admin",
        "last_login": datetime.utcnow().isoformat()
    }
    
    session_token = session_manager.create_session(user_data, expires_minutes=60)
    print(f"Session token: {session_token[:50]}...")
    
    # Verify session
    verified_data = session_manager.verify_session(session_token)
    if verified_data:
        print(f"Session verified!")
        print(f"User data: {verified_data}")
    else:
        print("Session verification failed!")


def demonstrate_security_utilities():
    """Demonstrate general security utilities."""
    print("\n=== Security Utilities Demo ===")
    
    # Generate tokens
    csrf_token = SecurityUtils.generate_csrf_token()
    api_key = SecurityUtils.generate_api_key()
    
    print(f"CSRF Token: {csrf_token}")
    print(f"API Key: {api_key}")
    
    # Secure comparison
    token1 = "secret_token_123"
    token2 = "secret_token_123"
    token3 = "different_token"
    
    print(f"Secure compare (same): {SecurityUtils.secure_compare(token1, token2)}")
    print(f"Secure compare (different): {SecurityUtils.secure_compare(token1, token3)}")
    
    # Filename sanitization
    dangerous_filename = "../../../etc/passwd"
    safe_filename = SecurityUtils.sanitize_filename(dangerous_filename)
    print(f"Original filename: {dangerous_filename}")
    print(f"Sanitized filename: {safe_filename}")
    
    # HTML escaping
    dangerous_html = "<script>alert('XSS')</script>"
    escaped_html = SecurityUtils.escape_html(dangerous_html)
    print(f"Original HTML: {dangerous_html}")
    print(f"Escaped HTML: {escaped_html}")
    
    # Email validation
    emails = ["valid@example.com", "invalid.email", "test@domain.co.uk"]
    for email in emails:
        print(f"Email '{email}' valid: {SecurityUtils.validate_email(email)}")


if __name__ == "__main__":
    print("=== Security and Cryptography Examples ===\n")
    
    # Run all demonstrations
    demonstrate_password_security()
    demonstrate_encryption()
    demonstrate_jwt_tokens()
    demonstrate_hashing()
    demonstrate_secure_sessions()
    demonstrate_security_utilities()
    
    print("\n=== Security Features Demonstrated ===")
    print("- Secure password hashing with bcrypt")
    print("- Password strength validation")
    print("- Symmetric encryption (Fernet/AES)")
    print("- Asymmetric encryption (RSA)")
    print("- Digital signatures")
    print("- JWT token management")
    print("- HMAC signatures")
    print("- Secure session management")
    print("- File encryption/decryption")
    print("- Cryptographic hashing")
    print("- Security utilities (CSRF, sanitization, etc.)")
    print("- Timing-safe comparisons")
    print("- Input validation and sanitization")