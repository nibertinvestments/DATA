"""
Advanced Cryptographic Hash Functions
====================================

Implementation of cryptographically secure hash functions including SHA-256,
Blake2, and custom hash constructions with security analysis and performance
optimization for blockchain and security applications.

Mathematical Foundation:
Hash Function Properties:
1. Deterministic: h(x) always produces same output
2. Fixed Output: |h(x)| = constant for any input x
3. Efficient: Fast computation
4. Pre-image Resistant: Hard to find x given h(x)
5. Second Pre-image Resistant: Hard to find x' ≠ x with h(x') = h(x)
6. Collision Resistant: Hard to find x, x' with h(x) = h(x')

Applications:
- Blockchain mining
- Digital signatures
- Password storage
- Data integrity
- Merkle trees
"""

import numpy as np
import hashlib
import struct
from typing import List, Tuple, Dict, Union, Optional
from dataclasses import dataclass
import time


@dataclass
class HashMetrics:
    """Metrics for hash function analysis."""
    avalanche_effect: float
    bit_independence: float
    uniform_distribution: float
    collision_resistance: float
    computation_time: float


class SHA256:
    """
    Pure Python implementation of SHA-256 for educational purposes.
    
    Based on FIPS PUB 180-4 specification.
    """
    
    def __init__(self):
        # SHA-256 constants
        self.K = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
            0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
            0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
            0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
            0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
            0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
            0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
            0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
        ]
        
        # Initial hash values
        self.H = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
        ]
    
    def _rotr(self, n: int, x: int) -> int:
        """Right rotate."""
        return ((x >> n) | (x << (32 - n))) & 0xffffffff
    
    def _ch(self, x: int, y: int, z: int) -> int:
        """Choice function."""
        return (x & y) ^ (~x & z)
    
    def _maj(self, x: int, y: int, z: int) -> int:
        """Majority function."""
        return (x & y) ^ (x & z) ^ (y & z)
    
    def _sigma0(self, x: int) -> int:
        """Sigma0 function."""
        return self._rotr(2, x) ^ self._rotr(13, x) ^ self._rotr(22, x)
    
    def _sigma1(self, x: int) -> int:
        """Sigma1 function."""
        return self._rotr(6, x) ^ self._rotr(11, x) ^ self._rotr(25, x)
    
    def _gamma0(self, x: int) -> int:
        """Gamma0 function."""
        return self._rotr(7, x) ^ self._rotr(18, x) ^ (x >> 3)
    
    def _gamma1(self, x: int) -> int:
        """Gamma1 function."""
        return self._rotr(17, x) ^ self._rotr(19, x) ^ (x >> 10)
    
    def _pad_message(self, message: bytes) -> bytes:
        """Pad message according to SHA-256 specification."""
        msg_len = len(message)
        message += b'\x80'
        
        # Pad with zeros
        while (len(message) % 64) != 56:
            message += b'\x00'
        
        # Append original length as 64-bit big-endian
        message += struct.pack('>Q', msg_len * 8)
        
        return message
    
    def hash(self, message: Union[str, bytes]) -> str:
        """
        Compute SHA-256 hash of message.
        
        Args:
            message: Input message
            
        Returns:
            Hexadecimal hash string
        """
        if isinstance(message, str):
            message = message.encode('utf-8')
        
        # Pad message
        padded = self._pad_message(message)
        
        # Initialize working variables
        h = self.H.copy()
        
        # Process message in 512-bit chunks
        for chunk_start in range(0, len(padded), 64):
            chunk = padded[chunk_start:chunk_start + 64]
            
            # Break chunk into sixteen 32-bit words
            w = list(struct.unpack('>16I', chunk))
            
            # Extend to 64 words
            for i in range(16, 64):
                s0 = self._gamma0(w[i-15])
                s1 = self._gamma1(w[i-2])
                w.append((w[i-16] + s0 + w[i-7] + s1) & 0xffffffff)
            
            # Initialize working variables
            a, b, c, d, e, f, g, h_val = h
            
            # Main loop
            for i in range(64):
                s1 = self._sigma1(e)
                ch = self._ch(e, f, g)
                temp1 = (h_val + s1 + ch + self.K[i] + w[i]) & 0xffffffff
                s0 = self._sigma0(a)
                maj = self._maj(a, b, c)
                temp2 = (s0 + maj) & 0xffffffff
                
                h_val = g
                g = f
                f = e
                e = (d + temp1) & 0xffffffff
                d = c
                c = b
                b = a
                a = (temp1 + temp2) & 0xffffffff
            
            # Add to hash values
            h[0] = (h[0] + a) & 0xffffffff
            h[1] = (h[1] + b) & 0xffffffff
            h[2] = (h[2] + c) & 0xffffffff
            h[3] = (h[3] + d) & 0xffffffff
            h[4] = (h[4] + e) & 0xffffffff
            h[5] = (h[5] + f) & 0xffffffff
            h[6] = (h[6] + g) & 0xffffffff
            h[7] = (h[7] + h_val) & 0xffffffff
        
        # Produce final hash value
        return ''.join(f'{x:08x}' for x in h)


class Blake2b:
    """
    Simplified Blake2b implementation.
    
    Blake2b is a cryptographic hash function optimized for speed.
    """
    
    def __init__(self, digest_size: int = 64):
        self.digest_size = digest_size
        
        # Blake2b constants
        self.IV = [
            0x6a09e667f3bcc908, 0xbb67ae8584caa73b,
            0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
            0x510e527fade682d1, 0x9b05688c2b3e6c1f,
            0x1f83d9abfb41bd6b, 0x5be0cd19137e2179
        ]
        
        # Initialize state
        self.h = self.IV.copy()
        self.h[0] ^= 0x01010000 ^ digest_size
        
        self.t = [0, 0]  # Offset counters
        self.buf = bytearray()
        self.buflen = 0
    
    def _rotr64(self, x: int, n: int) -> int:
        """64-bit right rotation."""
        return ((x >> n) | (x << (64 - n))) & 0xffffffffffffffff
    
    def _mix(self, v: List[int], a: int, b: int, c: int, d: int, x: int, y: int):
        """Blake2b mixing function."""
        v[a] = (v[a] + v[b] + x) & 0xffffffffffffffff
        v[d] = self._rotr64(v[d] ^ v[a], 32)
        v[c] = (v[c] + v[d]) & 0xffffffffffffffff
        v[b] = self._rotr64(v[b] ^ v[c], 24)
        v[a] = (v[a] + v[b] + y) & 0xffffffffffffffff
        v[d] = self._rotr64(v[d] ^ v[a], 16)
        v[c] = (v[c] + v[d]) & 0xffffffffffffffff
        v[b] = self._rotr64(v[b] ^ v[c], 63)
    
    def _compress(self, block: bytes, last: bool = False):
        """Compress a 128-byte block."""
        # Convert block to 16 64-bit words
        m = list(struct.unpack('<16Q', block))
        
        # Initialize working vector
        v = self.h.copy() + self.IV.copy()
        v[12] ^= self.t[0]
        v[13] ^= self.t[1]
        if last:
            v[14] ^= 0xffffffffffffffff
        
        # 12 rounds of mixing
        sigma = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3]
        ]
        
        for round_num in range(12):
            s = sigma[round_num % 2]
            
            # Column step
            self._mix(v, 0, 4, 8, 12, m[s[0]], m[s[1]])
            self._mix(v, 1, 5, 9, 13, m[s[2]], m[s[3]])
            self._mix(v, 2, 6, 10, 14, m[s[4]], m[s[5]])
            self._mix(v, 3, 7, 11, 15, m[s[6]], m[s[7]])
            
            # Diagonal step
            self._mix(v, 0, 5, 10, 15, m[s[8]], m[s[9]])
            self._mix(v, 1, 6, 11, 12, m[s[10]], m[s[11]])
            self._mix(v, 2, 7, 8, 13, m[s[12]], m[s[13]])
            self._mix(v, 3, 4, 9, 14, m[s[14]], m[s[15]])
        
        # Update hash state
        for i in range(8):
            self.h[i] ^= v[i] ^ v[i + 8]
    
    def update(self, data: bytes):
        """Update hash with new data."""
        datalen = len(data)
        data_pos = 0
        
        while data_pos < datalen:
            if self.buflen == 128:
                # Increment counter
                self.t[0] += 128
                if self.t[0] < 128:
                    self.t[1] += 1
                
                # Compress block
                self._compress(bytes(self.buf))
                self.buflen = 0
            
            # Fill buffer
            fill = min(128 - self.buflen, datalen - data_pos)
            self.buf[self.buflen:self.buflen + fill] = data[data_pos:data_pos + fill]
            self.buflen += fill
            data_pos += fill
    
    def digest(self) -> bytes:
        """Finalize and return digest."""
        # Increment counter for final block
        self.t[0] += self.buflen
        if self.t[0] < self.buflen:
            self.t[1] += 1
        
        # Pad final block with zeros
        while self.buflen < 128:
            self.buf.append(0)
            self.buflen += 1
        
        # Final compression
        self._compress(bytes(self.buf), last=True)
        
        # Return digest
        return struct.pack('<8Q', *self.h)[:self.digest_size]
    
    def hexdigest(self) -> str:
        """Return hexadecimal digest."""
        return self.digest().hex()


class HashAnalyzer:
    """
    Comprehensive hash function analysis tools.
    
    Analyzes cryptographic properties and performance characteristics.
    """
    
    def __init__(self, hash_function):
        self.hash_function = hash_function
    
    def avalanche_effect(self, num_tests: int = 1000) -> float:
        """
        Measure avalanche effect: small input change -> large output change.
        
        Args:
            num_tests: Number of test cases
            
        Returns:
            Average bit flip percentage
        """
        total_flips = 0
        
        for _ in range(num_tests):
            # Generate random input
            original = np.random.bytes(32)
            
            # Flip one random bit
            modified = bytearray(original)
            byte_pos = np.random.randint(0, len(modified))
            bit_pos = np.random.randint(0, 8)
            modified[byte_pos] ^= (1 << bit_pos)
            
            # Hash both versions
            hash1 = self.hash_function(original)
            hash2 = self.hash_function(bytes(modified))
            
            # Count bit differences
            if isinstance(hash1, str):
                hash1_bytes = bytes.fromhex(hash1)
                hash2_bytes = bytes.fromhex(hash2)
            else:
                hash1_bytes = hash1
                hash2_bytes = hash2
            
            diff_bits = sum(bin(b1 ^ b2).count('1') 
                          for b1, b2 in zip(hash1_bytes, hash2_bytes))
            
            total_flips += diff_bits
        
        total_bits = len(hash1_bytes) * 8 * num_tests
        return total_flips / total_bits
    
    def bit_independence(self, num_tests: int = 1000) -> float:
        """
        Test bit independence in hash outputs.
        
        Args:
            num_tests: Number of test cases
            
        Returns:
            Independence metric (closer to 0.5 is better)
        """
        correlations = []
        
        # Generate test data
        hashes = []
        for _ in range(num_tests):
            data = np.random.bytes(32)
            hash_val = self.hash_function(data)
            
            if isinstance(hash_val, str):
                hash_bytes = bytes.fromhex(hash_val)
            else:
                hash_bytes = hash_val
            
            # Convert to bit array
            bits = []
            for byte in hash_bytes:
                bits.extend([(byte >> i) & 1 for i in range(8)])
            hashes.append(bits)
        
        hashes = np.array(hashes)
        
        # Calculate correlations between bit positions
        num_bits = hashes.shape[1]
        for i in range(min(num_bits, 64)):  # Test first 64 bits
            for j in range(i + 1, min(num_bits, 64)):
                corr = np.corrcoef(hashes[:, i], hashes[:, j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0.0
    
    def distribution_uniformity(self, num_tests: int = 10000) -> float:
        """
        Test uniformity of hash output distribution.
        
        Args:
            num_tests: Number of test cases
            
        Returns:
            Chi-square test statistic (lower is more uniform)
        """
        # Count occurrences of each byte value
        byte_counts = np.zeros(256)
        
        for _ in range(num_tests):
            data = np.random.bytes(32)
            hash_val = self.hash_function(data)
            
            if isinstance(hash_val, str):
                hash_bytes = bytes.fromhex(hash_val)
            else:
                hash_bytes = hash_val
            
            for byte in hash_bytes:
                byte_counts[byte] += 1
        
        # Expected count for uniform distribution
        total_bytes = sum(byte_counts)
        expected_count = total_bytes / 256
        
        # Chi-square statistic
        chi_square = sum((observed - expected_count) ** 2 / expected_count 
                        for observed in byte_counts if expected_count > 0)
        
        return chi_square
    
    def collision_resistance_estimate(self, num_tests: int = 100000) -> float:
        """
        Estimate collision resistance using birthday paradox.
        
        Args:
            num_tests: Number of random inputs to test
            
        Returns:
            Estimated security level (in bits)
        """
        hashes_seen = set()
        
        for i in range(num_tests):
            data = np.random.bytes(32)
            hash_val = self.hash_function(data)
            
            if isinstance(hash_val, str):
                hash_key = hash_val
            else:
                hash_key = hash_val.hex()
            
            if hash_key in hashes_seen:
                # Found collision
                # Security level ≈ log2(sqrt(π/2) * sqrt(n))
                return np.log2(np.sqrt(np.pi / 2) * np.sqrt(i))
            
            hashes_seen.add(hash_key)
        
        # No collision found - estimate based on birthday paradox
        hash_length_bits = len(list(hashes_seen)[0]) * 4  # 4 bits per hex char
        return hash_length_bits / 2  # Theoretical collision resistance
    
    def performance_benchmark(self, data_sizes: List[int] = None) -> Dict[int, float]:
        """
        Benchmark hash function performance.
        
        Args:
            data_sizes: List of data sizes to test (in bytes)
            
        Returns:
            Dictionary mapping data size to throughput (MB/s)
        """
        if data_sizes is None:
            data_sizes = [1024, 4096, 16384, 65536, 262144]  # 1KB to 256KB
        
        results = {}
        
        for size in data_sizes:
            data = np.random.bytes(size)
            
            # Warm up
            for _ in range(10):
                self.hash_function(data)
            
            # Benchmark
            start_time = time.time()
            num_iterations = max(10, 10000000 // size)  # Adjust iterations based on size
            
            for _ in range(num_iterations):
                self.hash_function(data)
            
            end_time = time.time()
            
            total_bytes = size * num_iterations
            total_time = end_time - start_time
            throughput_mb_s = (total_bytes / (1024 * 1024)) / total_time
            
            results[size] = throughput_mb_s
        
        return results
    
    def comprehensive_analysis(self) -> HashMetrics:
        """
        Perform comprehensive hash function analysis.
        
        Returns:
            HashMetrics with all analysis results
        """
        print("Analyzing hash function properties...")
        
        # Avalanche effect
        print("  Testing avalanche effect...")
        avalanche = self.avalanche_effect(1000)
        
        # Bit independence
        print("  Testing bit independence...")
        independence = self.bit_independence(1000)
        
        # Distribution uniformity
        print("  Testing distribution uniformity...")
        uniformity = self.distribution_uniformity(5000)
        
        # Collision resistance
        print("  Estimating collision resistance...")
        collision_resistance = self.collision_resistance_estimate(50000)
        
        # Performance
        print("  Benchmarking performance...")
        perf_results = self.performance_benchmark([4096])
        avg_performance = sum(perf_results.values()) / len(perf_results)
        
        return HashMetrics(
            avalanche_effect=avalanche,
            bit_independence=independence,
            uniform_distribution=uniformity,
            collision_resistance=collision_resistance,
            computation_time=avg_performance
        )


def comprehensive_hash_example():
    """Comprehensive example demonstrating hash function analysis."""
    print("=== Advanced Cryptographic Hash Functions Example ===")
    
    # Test data
    test_message = "The quick brown fox jumps over the lazy dog"
    
    print(f"Test message: '{test_message}'")
    print()
    
    # SHA-256 implementation
    print("=== SHA-256 Analysis ===")
    sha256_impl = SHA256()
    sha256_hash = sha256_impl.hash(test_message)
    
    # Compare with standard library
    stdlib_hash = hashlib.sha256(test_message.encode()).hexdigest()
    
    print(f"Custom SHA-256:   {sha256_hash}")
    print(f"Standard SHA-256: {stdlib_hash}")
    print(f"Match: {sha256_hash == stdlib_hash}")
    print()
    
    # Blake2b implementation
    print("=== Blake2b Analysis ===")
    blake2b_impl = Blake2b()
    blake2b_impl.update(test_message.encode())
    blake2b_hash = blake2b_impl.hexdigest()
    
    # Compare with standard library
    stdlib_blake2b = hashlib.blake2b(test_message.encode()).hexdigest()
    
    print(f"Custom Blake2b:   {blake2b_hash}")
    print(f"Standard Blake2b: {stdlib_blake2b}")
    print()
    
    # Analyze SHA-256
    print("=== SHA-256 Cryptographic Analysis ===")
    sha256_analyzer = HashAnalyzer(lambda x: SHA256().hash(x))
    sha256_metrics = sha256_analyzer.comprehensive_analysis()
    
    print(f"Avalanche Effect:        {sha256_metrics.avalanche_effect:.6f} (ideal: ~0.5)")
    print(f"Bit Independence:        {sha256_metrics.bit_independence:.6f} (ideal: ~0.0)")
    print(f"Distribution Uniformity: {sha256_metrics.uniform_distribution:.2f} (lower is better)")
    print(f"Collision Resistance:    {sha256_metrics.collision_resistance:.1f} bits")
    print(f"Performance:             {sha256_metrics.computation_time:.2f} MB/s")
    print()
    
    # Analyze Blake2b
    print("=== Blake2b Cryptographic Analysis ===")
    
    def blake2b_wrapper(data):
        b2b = Blake2b()
        b2b.update(data)
        return b2b.digest()
    
    blake2b_analyzer = HashAnalyzer(blake2b_wrapper)
    blake2b_metrics = blake2b_analyzer.comprehensive_analysis()
    
    print(f"Avalanche Effect:        {blake2b_metrics.avalanche_effect:.6f} (ideal: ~0.5)")
    print(f"Bit Independence:        {blake2b_metrics.bit_independence:.6f} (ideal: ~0.0)")
    print(f"Distribution Uniformity: {blake2b_metrics.uniform_distribution:.2f} (lower is better)")
    print(f"Collision Resistance:    {blake2b_metrics.collision_resistance:.1f} bits")
    print(f"Performance:             {blake2b_metrics.computation_time:.2f} MB/s")
    print()
    
    # Performance comparison
    print("=== Performance Comparison ===")
    
    data_sizes = [1024, 4096, 16384, 65536]
    
    print(f"{'Size (KB)':<12} {'SHA-256 (MB/s)':<15} {'Blake2b (MB/s)':<15}")
    print("-" * 45)
    
    sha256_perf = sha256_analyzer.performance_benchmark(data_sizes)
    blake2b_perf = blake2b_analyzer.performance_benchmark(data_sizes)
    
    for size in data_sizes:
        size_kb = size // 1024
        sha256_speed = sha256_perf[size]
        blake2b_speed = blake2b_perf[size]
        
        print(f"{size_kb:<12} {sha256_speed:<15.2f} {blake2b_speed:<15.2f}")
    
    print()
    
    # Security analysis
    print("=== Security Analysis ===")
    
    # Test with various input patterns
    patterns = {
        "Empty": "",
        "Single char": "a",
        "Repeated": "a" * 100,
        "Sequential": "".join(chr(i) for i in range(32, 127)),
        "Binary": bytes(range(256)).decode('latin1')
    }
    
    print(f"{'Pattern':<12} {'SHA-256 (first 16 chars)':<25} {'Blake2b (first 16 chars)':<25}")
    print("-" * 65)
    
    for pattern_name, pattern_data in patterns.items():
        sha256_result = SHA256().hash(pattern_data)[:16]
        
        b2b = Blake2b()
        b2b.update(pattern_data.encode('latin1'))
        blake2b_result = b2b.hexdigest()[:16]
        
        print(f"{pattern_name:<12} {sha256_result:<25} {blake2b_result:<25}")
    
    print()
    
    # Merkle tree demonstration
    print("=== Merkle Tree Application ===")
    
    def build_merkle_tree(data_blocks: List[bytes]) -> str:
        """Build simple Merkle tree using SHA-256."""
        if not data_blocks:
            return ""
        
        sha256 = SHA256()
        
        # Hash all data blocks
        hashes = [sha256.hash(block) for block in data_blocks]
        
        # Build tree bottom-up
        while len(hashes) > 1:
            next_level = []
            
            for i in range(0, len(hashes), 2):
                if i + 1 < len(hashes):
                    combined = hashes[i] + hashes[i + 1]
                else:
                    combined = hashes[i] + hashes[i]  # Duplicate if odd number
                
                next_level.append(sha256.hash(combined.encode()))
            
            hashes = next_level
        
        return hashes[0]
    
    # Create sample data blocks
    data_blocks = [f"Block {i}: {np.random.bytes(32).hex()}".encode() for i in range(8)]
    
    merkle_root = build_merkle_tree(data_blocks)
    print(f"Merkle root (8 blocks): {merkle_root}")
    
    # Modify one block and recalculate
    data_blocks[3] = b"Modified block"
    modified_root = build_merkle_tree(data_blocks)
    print(f"Modified Merkle root:   {modified_root}")
    print(f"Roots match: {merkle_root == modified_root}")


if __name__ == "__main__":
    comprehensive_hash_example()