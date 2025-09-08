"""
Advanced Cryptographic Algorithm Implementations

This module contains production-ready implementations of cryptographic algorithms
for AI training purposes. All implementations follow security best practices
and include comprehensive documentation for educational use.

Time Complexities and Security Levels are documented for each algorithm.

Author: AI Training Dataset
Version: 1.0
"""

import hashlib
import secrets
import math
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class ECPoint:
    """Represents a point on an elliptic curve"""

    x: Optional[int]
    y: Optional[int]

    def __post_init__(self):
        # Point at infinity representation
        if self.x is None or self.y is None:
            self.x = None
            self.y = None

    def is_infinity(self) -> bool:
        return self.x is None and self.y is None


class EllipticCurve:
    """
    Elliptic Curve implementation for cryptographic operations

    Supports curves of the form y² = x³ + ax + b (mod p)

    Security Level: Provides equivalent security to RSA with much smaller key sizes
    Example: 256-bit EC key ≈ 3072-bit RSA key

    Time Complexity:
    - Point addition: O(1) - constant time for field operations
    - Scalar multiplication: O(log n) where n is the scalar
    """

    def __init__(self, a: int, b: int, p: int, g: ECPoint, n: int):
        """
        Initialize elliptic curve parameters

        Args:
            a, b: Curve parameters for y² = x³ + ax + b
            p: Prime modulus for the finite field
            g: Generator point (base point)
            n: Order of the generator point
        """
        self.a = a
        self.b = b
        self.p = p
        self.g = g  # Generator point
        self.n = n  # Order of generator

        # Verify curve equation for generator point
        if not self.is_on_curve(g):
            raise ValueError("Generator point is not on the curve")

    def is_on_curve(self, point: ECPoint) -> bool:
        """Verify if a point lies on the elliptic curve"""
        if point.is_infinity():
            return True

        left = (point.y * point.y) % self.p
        right = (point.x**3 + self.a * point.x + self.b) % self.p
        return left == right

    def point_add(self, p1: ECPoint, p2: ECPoint) -> ECPoint:
        """
        Add two points on the elliptic curve

        Uses the group law for elliptic curves with proper handling
        of special cases (point at infinity, point doubling)
        """
        if p1.is_infinity():
            return p2
        if p2.is_infinity():
            return p1

        if p1.x == p2.x:
            if p1.y == p2.y:
                return self.point_double(p1)
            else:
                # Points are inverses, return point at infinity
                return ECPoint(None, None)

        # Standard point addition formula
        try:
            slope = ((p2.y - p1.y) * self.mod_inverse(p2.x - p1.x, self.p)) % self.p
            x3 = (slope * slope - p1.x - p2.x) % self.p
            y3 = (slope * (p1.x - x3) - p1.y) % self.p
            return ECPoint(x3, y3)
        except ValueError as e:
            raise ValueError(f"Point addition failed: {e}")

    def point_double(self, point: ECPoint) -> ECPoint:
        """Double a point on the elliptic curve"""
        if point.is_infinity():
            return point

        if point.y == 0:
            return ECPoint(None, None)

        try:
            slope = (
                (3 * point.x * point.x + self.a) * self.mod_inverse(2 * point.y, self.p)
            ) % self.p
            x3 = (slope * slope - 2 * point.x) % self.p
            y3 = (slope * (point.x - x3) - point.y) % self.p
            return ECPoint(x3, y3)
        except ValueError as e:
            raise ValueError(f"Point doubling failed: {e}")

    def scalar_mult(self, k: int, point: ECPoint) -> ECPoint:
        """
        Scalar multiplication using double-and-add algorithm

        Computes k * point efficiently in O(log k) time
        Uses binary representation of k for optimization
        """
        if k == 0:
            return ECPoint(None, None)
        if k == 1:
            return point

        # Handle negative scalars
        if k < 0:
            k = -k
            point = ECPoint(point.x, (-point.y) % self.p)

        result = ECPoint(None, None)  # Start with point at infinity
        addend = point

        while k:
            if k & 1:  # If current bit is 1
                result = self.point_add(result, addend)
            addend = self.point_double(addend)
            k >>= 1

        return result

    @staticmethod
    def mod_inverse(a: int, m: int) -> int:
        """
        Compute modular multiplicative inverse using Extended Euclidean Algorithm

        Time Complexity: O(log min(a, m))
        """
        if math.gcd(a, m) != 1:
            raise ValueError("Modular inverse does not exist")

        def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
            if a == 0:
                return b, 0, 1
            gcd, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1
            return gcd, x, y

        _, x, _ = extended_gcd(a % m, m)
        return (x % m + m) % m


class ECDSA:
    """
    Elliptic Curve Digital Signature Algorithm implementation

    Provides digital signatures with strong security guarantees
    and smaller signature sizes compared to RSA

    Security: Based on the discrete logarithm problem in elliptic curve groups
    """

    def __init__(self, curve: EllipticCurve):
        self.curve = curve

    def generate_keypair(self) -> Tuple[int, ECPoint]:
        """
        Generate a public/private key pair

        Returns:
            Tuple of (private_key, public_key)
            private_key: Random integer in [1, n-1]
            public_key: Point on curve = private_key * generator
        """
        # Generate random private key
        private_key = secrets.randbelow(self.curve.n - 1) + 1

        # Compute public key
        public_key = self.curve.scalar_mult(private_key, self.curve.g)

        return private_key, public_key

    def sign(self, message_hash: bytes, private_key: int) -> Tuple[int, int]:
        """
        Sign a message hash using ECDSA

        Args:
            message_hash: SHA-256 hash of the message
            private_key: Signer's private key

        Returns:
            Signature tuple (r, s)
        """
        if len(message_hash) != 32:
            raise ValueError("Message hash must be 32 bytes (SHA-256)")

        # Convert hash to integer
        z = int.from_bytes(message_hash, "big")

        while True:
            # Generate random nonce k
            k = secrets.randbelow(self.curve.n - 1) + 1

            # Compute signature components
            point = self.curve.scalar_mult(k, self.curve.g)
            r = point.x % self.curve.n

            if r == 0:
                continue  # Try again with different k

            # Compute s = k⁻¹(z + r·private_key) mod n
            k_inv = self.curve.mod_inverse(k, self.curve.n)
            s = (k_inv * (z + r * private_key)) % self.curve.n

            if s == 0:
                continue  # Try again with different k

            return r, s

    def verify(
        self, message_hash: bytes, signature: Tuple[int, int], public_key: ECPoint
    ) -> bool:
        """
        Verify an ECDSA signature

        Args:
            message_hash: SHA-256 hash of the original message
            signature: Signature tuple (r, s)
            public_key: Signer's public key

        Returns:
            True if signature is valid, False otherwise
        """
        r, s = signature

        # Validate signature components
        if not (1 <= r < self.curve.n and 1 <= s < self.curve.n):
            return False

        # Convert hash to integer
        z = int.from_bytes(message_hash, "big")

        # Compute verification values
        s_inv = self.curve.mod_inverse(s, self.curve.n)
        u1 = (z * s_inv) % self.curve.n
        u2 = (r * s_inv) % self.curve.n

        # Compute verification point
        point1 = self.curve.scalar_mult(u1, self.curve.g)
        point2 = self.curve.scalar_mult(u2, public_key)
        verification_point = self.curve.point_add(point1, point2)

        if verification_point.is_infinity():
            return False

        return verification_point.x % self.curve.n == r


class ChaCha20:
    """
    ChaCha20 stream cipher implementation

    A modern, fast, and secure stream cipher designed by Daniel J. Bernstein

    Security: 256-bit key provides 2^256 security level
    Performance: Highly optimized for both software and hardware
    """

    def __init__(self, key: bytes, nonce: bytes):
        """
        Initialize ChaCha20 with key and nonce

        Args:
            key: 32-byte encryption key
            nonce: 12-byte nonce (number used once)
        """
        if len(key) != 32:
            raise ValueError("Key must be 32 bytes")
        if len(nonce) != 12:
            raise ValueError("Nonce must be 12 bytes")

        self.key = key
        self.nonce = nonce
        self.counter = 0

    @staticmethod
    def _quarter_round(state: List[int], a: int, b: int, c: int, d: int):
        """ChaCha20 quarter round function"""
        state[a] = (state[a] + state[b]) & 0xFFFFFFFF
        state[d] ^= state[a]
        state[d] = ((state[d] << 16) | (state[d] >> 16)) & 0xFFFFFFFF

        state[c] = (state[c] + state[d]) & 0xFFFFFFFF
        state[b] ^= state[c]
        state[b] = ((state[b] << 12) | (state[b] >> 20)) & 0xFFFFFFFF

        state[a] = (state[a] + state[b]) & 0xFFFFFFFF
        state[d] ^= state[a]
        state[d] = ((state[d] << 8) | (state[d] >> 24)) & 0xFFFFFFFF

        state[c] = (state[c] + state[d]) & 0xFFFFFFFF
        state[b] ^= state[c]
        state[b] = ((state[b] << 7) | (state[b] >> 25)) & 0xFFFFFFFF

    def _chacha20_block(self, counter: int) -> bytes:
        """Generate a single ChaCha20 block"""
        # Initialize state
        state = [
            0x61707865,
            0x3320646E,
            0x79622D32,
            0x6B206574,  # Constants
            *[
                int.from_bytes(self.key[i : i + 4], "little") for i in range(0, 32, 4)
            ],  # Key
            counter,  # Counter
            *[
                int.from_bytes(self.nonce[i : i + 4], "little") for i in range(0, 12, 4)
            ],  # Nonce
        ]

        working_state = state.copy()

        # Perform 20 rounds (10 double rounds)
        for _ in range(10):
            # Column rounds
            self._quarter_round(working_state, 0, 4, 8, 12)
            self._quarter_round(working_state, 1, 5, 9, 13)
            self._quarter_round(working_state, 2, 6, 10, 14)
            self._quarter_round(working_state, 3, 7, 11, 15)

            # Diagonal rounds
            self._quarter_round(working_state, 0, 5, 10, 15)
            self._quarter_round(working_state, 1, 6, 11, 12)
            self._quarter_round(working_state, 2, 7, 8, 13)
            self._quarter_round(working_state, 3, 4, 9, 14)

        # Add initial state to working state
        for i in range(16):
            working_state[i] = (working_state[i] + state[i]) & 0xFFFFFFFF

        # Convert to bytes
        return b"".join(x.to_bytes(4, "little") for x in working_state)

    def encrypt(self, plaintext: bytes) -> bytes:
        """Encrypt plaintext using ChaCha20"""
        ciphertext = bytearray()

        for i in range(0, len(plaintext), 64):
            keystream = self._chacha20_block(self.counter)
            self.counter += 1

            chunk = plaintext[i : i + 64]
            encrypted_chunk = bytes(a ^ b for a, b in zip(chunk, keystream))
            ciphertext.extend(encrypted_chunk)

        return bytes(ciphertext)

    def decrypt(self, ciphertext: bytes) -> bytes:
        """Decrypt ciphertext using ChaCha20 (same as encrypt for stream cipher)"""
        return self.encrypt(ciphertext)


def demonstrate_cryptography():
    """Demonstration of cryptographic algorithms"""
    print("=== Cryptographic Algorithm Demonstrations ===\n")

    # ECDSA Example using secp256k1-like parameters (simplified)
    print("1. Elliptic Curve Digital Signature Algorithm (ECDSA)")
    print("-" * 50)

    # Define a simple curve for demonstration (not secp256k1)
    p = 2**256 - 2**32 - 977  # Prime modulus
    a = 0
    b = 7
    g = ECPoint(
        0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
        0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8,
    )
    n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

    curve = EllipticCurve(a, b, p, g, n)
    ecdsa = ECDSA(curve)

    # Generate keypair
    private_key, public_key = ecdsa.generate_keypair()
    print(f"Generated keypair:")
    print(f"Private key: {private_key:064x}")
    print(f"Public key: ({public_key.x:064x}, {public_key.y:064x})")

    # Sign a message
    message = b"Hello, ECDSA!"
    message_hash = hashlib.sha256(message).digest()
    signature = ecdsa.sign(message_hash, private_key)
    print(f"\nMessage: {message.decode()}")
    print(f"Signature: (r={signature[0]:064x}, s={signature[1]:064x})")

    # Verify signature
    is_valid = ecdsa.verify(message_hash, signature, public_key)
    print(f"Signature valid: {is_valid}")

    # ChaCha20 Example
    print("\n\n2. ChaCha20 Stream Cipher")
    print("-" * 30)

    # Generate random key and nonce
    key = secrets.token_bytes(32)
    nonce = secrets.token_bytes(12)

    cipher = ChaCha20(key, nonce)

    plaintext = b"This is a secret message that will be encrypted with ChaCha20!"
    print(f"Plaintext: {plaintext.decode()}")
    print(f"Key: {key.hex()}")
    print(f"Nonce: {nonce.hex()}")

    # Encrypt
    ciphertext = cipher.encrypt(plaintext)
    print(f"Ciphertext: {ciphertext.hex()}")

    # Decrypt
    cipher2 = ChaCha20(key, nonce)  # Reset counter
    decrypted = cipher2.decrypt(ciphertext)
    print(f"Decrypted: {decrypted.decode()}")
    print(f"Decryption successful: {plaintext == decrypted}")


if __name__ == "__main__":
    demonstrate_cryptography()
