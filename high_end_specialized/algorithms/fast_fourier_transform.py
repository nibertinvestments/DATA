"""
Fast Fourier Transform (FFT) Algorithm
=====================================

Optimized implementation of the Fast Fourier Transform algorithm with
applications in signal processing, time series analysis, and scientific
computing. Includes multiple variants and performance optimizations.

Mathematical Foundation:
DFT: X(k) = Σ(n=0 to N-1) x(n) * e^(-j*2π*k*n/N)
FFT reduces complexity from O(N²) to O(N log N)

Cooley-Tukey Algorithm:
X(k) = Σ(even) x(2m) * e^(-j*2π*k*2m/N) + e^(-j*2π*k/N) * Σ(odd) x(2m+1) * e^(-j*2π*k*(2m+1)/N)

Applications:
- Digital signal processing
- Audio/video compression
- Image processing
- Numerical analysis
- Convolution operations
"""

import numpy as np
import cmath
from typing import Union, Tuple, List, Optional
from dataclasses import dataclass
import time


@dataclass
class FFTResult:
    """Result of FFT computation."""
    frequencies: np.ndarray
    magnitudes: np.ndarray
    phases: np.ndarray
    power_spectrum: np.ndarray
    computation_time: float


class AdvancedFFT:
    """
    Advanced Fast Fourier Transform implementation with multiple algorithms.
    
    Includes:
    - Radix-2 Cooley-Tukey FFT
    - Radix-4 FFT
    - Bluestein's algorithm for arbitrary lengths
    - Real FFT optimization
    - 2D FFT for image processing
    """
    
    def __init__(self):
        self.twiddle_factors = {}  # Cache for twiddle factors
    
    def _bit_reverse(self, x: np.ndarray) -> np.ndarray:
        """Bit-reverse permutation for FFT."""
        N = len(x)
        if N <= 1:
            return x
        
        # Calculate bit-reversed indices
        j = 0
        result = x.copy()
        
        for i in range(1, N):
            bit = N >> 1
            while j & bit:
                j ^= bit
                bit >>= 1
            j ^= bit
            
            if i < j:
                result[i], result[j] = result[j], result[i]
        
        return result
    
    def _get_twiddle_factors(self, N: int) -> np.ndarray:
        """Get cached twiddle factors for given size."""
        if N not in self.twiddle_factors:
            self.twiddle_factors[N] = np.exp(-2j * np.pi * np.arange(N // 2) / N)
        return self.twiddle_factors[N]
    
    def radix2_fft(self, x: np.ndarray) -> np.ndarray:
        """
        Radix-2 Cooley-Tukey FFT algorithm.
        
        Args:
            x: Input signal (length must be power of 2)
            
        Returns:
            FFT of input signal
        """
        x = np.asarray(x, dtype=complex)
        N = len(x)
        
        if N <= 1:
            return x
        
        # Check if N is power of 2
        if N & (N - 1) != 0:
            raise ValueError("Input length must be power of 2 for radix-2 FFT")
        
        # Bit-reverse permutation
        x = self._bit_reverse(x)
        
        # Compute FFT
        length = 2
        while length <= N:
            wlen = np.exp(-2j * np.pi / length)
            
            for i in range(0, N, length):
                w = 1.0
                
                for j in range(length // 2):
                    u = x[i + j]
                    v = x[i + j + length // 2] * w
                    
                    x[i + j] = u + v
                    x[i + j + length // 2] = u - v
                    
                    w *= wlen
            
            length <<= 1
        
        return x
    
    def radix4_fft(self, x: np.ndarray) -> np.ndarray:
        """
        Radix-4 FFT algorithm for improved efficiency.
        
        Args:
            x: Input signal (length must be power of 4)
            
        Returns:
            FFT of input signal
        """
        x = np.asarray(x, dtype=complex)
        N = len(x)
        
        if N <= 1:
            return x
        
        # Check if N is power of 4
        log4_N = int(np.log(N) / np.log(4))
        if 4 ** log4_N != N:
            raise ValueError("Input length must be power of 4 for radix-4 FFT")
        
        # Digit-reverse permutation for radix-4
        def digit_reverse_4(x):
            N = len(x)
            result = x.copy()
            
            for i in range(N):
                reversed_i = 0
                temp_i = i
                temp_N = N
                
                while temp_N > 1:
                    reversed_i = reversed_i * 4 + temp_i % 4
                    temp_i //= 4
                    temp_N //= 4
                
                if i < reversed_i:
                    result[i], result[reversed_i] = result[reversed_i], result[i]
            
            return result
        
        x = digit_reverse_4(x)
        
        # Radix-4 butterfly operations
        length = 4
        while length <= N:
            wlen = np.exp(-2j * np.pi / length)
            
            for i in range(0, N, length):
                w1 = 1.0
                w2 = 1.0
                w3 = 1.0
                
                for j in range(length // 4):
                    # Radix-4 butterfly
                    a = x[i + j]
                    b = x[i + j + length // 4] * w1
                    c = x[i + j + length // 2] * w2
                    d = x[i + j + 3 * length // 4] * w3
                    
                    # Butterfly computation
                    t1 = a + c
                    t2 = a - c
                    t3 = b + d
                    t4 = (b - d) * 1j
                    
                    x[i + j] = t1 + t3
                    x[i + j + length // 4] = t2 + t4
                    x[i + j + length // 2] = t1 - t3
                    x[i + j + 3 * length // 4] = t2 - t4
                    
                    w1 *= wlen
                    w2 *= wlen * wlen
                    w3 *= wlen * wlen * wlen
            
            length <<= 2
        
        return x
    
    def bluestein_fft(self, x: np.ndarray) -> np.ndarray:
        """
        Bluestein's algorithm for FFT of arbitrary length.
        
        Args:
            x: Input signal of any length
            
        Returns:
            FFT of input signal
        """
        x = np.asarray(x, dtype=complex)
        N = len(x)
        
        if N <= 1:
            return x
        
        # Find suitable size for convolution (next power of 2 >= 2*N-1)
        M = 1
        while M < 2 * N - 1:
            M <<= 1
        
        # Chirp sequence
        chirp = np.exp(-1j * np.pi * np.arange(N) ** 2 / N)
        
        # Multiply input by chirp
        a = x * chirp
        
        # Prepare convolution sequence
        b = np.zeros(M, dtype=complex)
        b[0] = 1
        for k in range(1, N):
            b[k] = b[M - k] = np.exp(1j * np.pi * k ** 2 / N)
        
        # Extend a to size M
        a_extended = np.zeros(M, dtype=complex)
        a_extended[:N] = a
        
        # Convolution via FFT
        A = self.radix2_fft(a_extended)
        B = self.radix2_fft(b)
        C = A * B
        c = self.radix2_fft(C[::-1])[::-1] / M  # IFFT
        
        # Extract result and multiply by chirp
        result = c[:N] * chirp
        
        return result
    
    def real_fft(self, x: np.ndarray) -> np.ndarray:
        """
        Optimized FFT for real-valued signals.
        
        Args:
            x: Real-valued input signal
            
        Returns:
            FFT of input signal (only positive frequencies)
        """
        x = np.asarray(x, dtype=float)
        N = len(x)
        
        if N == 1:
            return np.array([x[0]], dtype=complex)
        
        # Use complex FFT on half the data
        if N % 2 == 0:
            # Even length: pack two real sequences into one complex
            x_complex = x[::2] + 1j * x[1::2]
            X_complex = self.fft(x_complex)
            
            # Unpack the result
            N_half = N // 2
            X = np.zeros(N_half + 1, dtype=complex)
            
            # DC component
            X[0] = X_complex[0].real + X_complex[0].imag
            
            # Positive frequencies
            for k in range(1, N_half):
                X[k] = 0.5 * (X_complex[k] + np.conj(X_complex[N_half - k])) - \
                       0.5j * (X_complex[k] - np.conj(X_complex[N_half - k])) * \
                       np.exp(-2j * np.pi * k / N)
            
            # Nyquist frequency
            if N_half < len(X_complex):
                X[N_half] = X_complex[0].real - X_complex[0].imag
        else:
            # Odd length: use standard FFT
            X = self.fft(x.astype(complex))
            X = X[:(N + 1) // 2]
        
        return X
    
    def fft_2d(self, x: np.ndarray) -> np.ndarray:
        """
        2D FFT for image processing applications.
        
        Args:
            x: 2D input array
            
        Returns:
            2D FFT of input
        """
        x = np.asarray(x, dtype=complex)
        
        # Apply 1D FFT to rows
        result = np.apply_along_axis(self.fft, axis=1, arr=x)
        
        # Apply 1D FFT to columns
        result = np.apply_along_axis(self.fft, axis=0, arr=result)
        
        return result
    
    def fft(self, x: np.ndarray) -> np.ndarray:
        """
        Main FFT function that automatically selects best algorithm.
        
        Args:
            x: Input signal
            
        Returns:
            FFT of input signal
        """
        x = np.asarray(x, dtype=complex)
        N = len(x)
        
        if N <= 1:
            return x
        
        # Choose algorithm based on length
        if N & (N - 1) == 0:  # Power of 2
            if N >= 16 and (N & (N - 1)) == 0:  # Check if also power of 4
                log4_N = int(np.log(N) / np.log(4))
                if 4 ** log4_N == N:
                    return self.radix4_fft(x)
            return self.radix2_fft(x)
        else:
            return self.bluestein_fft(x)
    
    def ifft(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse FFT.
        
        Args:
            X: Frequency domain signal
            
        Returns:
            Time domain signal
        """
        X = np.asarray(X, dtype=complex)
        N = len(X)
        
        # IFFT = conj(FFT(conj(X))) / N
        return np.conj(self.fft(np.conj(X))) / N
    
    def power_spectrum(self, x: np.ndarray, sampling_rate: float = 1.0) -> FFTResult:
        """
        Compute power spectrum with detailed analysis.
        
        Args:
            x: Input signal
            sampling_rate: Sampling rate in Hz
            
        Returns:
            FFTResult with comprehensive spectral analysis
        """
        start_time = time.time()
        
        x = np.asarray(x, dtype=complex)
        N = len(x)
        
        # Compute FFT
        X = self.fft(x)
        
        # Frequency bins
        frequencies = np.fft.fftfreq(N, 1.0 / sampling_rate)
        
        # Only positive frequencies for real signals
        if np.all(np.isreal(x)):
            half_N = N // 2 + 1
            X = X[:half_N]
            frequencies = frequencies[:half_N]
        
        # Compute magnitudes and phases
        magnitudes = np.abs(X)
        phases = np.angle(X)
        
        # Power spectrum (magnitude squared)
        power_spectrum = magnitudes ** 2
        
        # Normalize by length and sampling rate
        if np.all(np.isreal(x)) and N > 1:
            power_spectrum[1:-1] *= 2  # Account for negative frequencies
        
        power_spectrum /= (N * sampling_rate)
        
        computation_time = time.time() - start_time
        
        return FFTResult(
            frequencies=frequencies,
            magnitudes=magnitudes,
            phases=phases,
            power_spectrum=power_spectrum,
            computation_time=computation_time
        )
    
    def convolution_fft(self, x: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Fast convolution using FFT.
        
        Args:
            x: First signal
            h: Second signal (kernel)
            
        Returns:
            Convolution of x and h
        """
        # Zero-pad to prevent circular convolution
        N = len(x) + len(h) - 1
        
        # Pad to next power of 2 for efficiency
        N_fft = 1
        while N_fft < N:
            N_fft <<= 1
        
        # Zero-pad signals
        x_padded = np.zeros(N_fft, dtype=complex)
        h_padded = np.zeros(N_fft, dtype=complex)
        
        x_padded[:len(x)] = x
        h_padded[:len(h)] = h
        
        # Convolution via FFT
        X = self.fft(x_padded)
        H = self.fft(h_padded)
        Y = X * H
        y = self.ifft(Y)
        
        # Return only valid part
        return y[:N].real if np.all(np.isreal(x)) and np.all(np.isreal(h)) else y[:N]


def comprehensive_fft_example():
    """Comprehensive example demonstrating FFT capabilities."""
    print("=== Fast Fourier Transform Algorithm Example ===")
    
    # Initialize FFT processor
    fft_processor = AdvancedFFT()
    
    # Test 1: Basic FFT on sinusoidal signal
    print("\n=== Test 1: Sinusoidal Signal Analysis ===")
    
    # Generate test signal
    fs = 1000  # Sampling frequency
    t = np.linspace(0, 1, fs, endpoint=False)
    
    # Composite signal: 50Hz + 120Hz + noise
    signal = (2 * np.sin(2 * np.pi * 50 * t) + 
             1.5 * np.sin(2 * np.pi * 120 * t) + 
             0.5 * np.random.randn(len(t)))
    
    # Analyze signal
    result = fft_processor.power_spectrum(signal, fs)
    
    print(f"Signal length: {len(signal)}")
    print(f"Sampling rate: {fs} Hz")
    print(f"FFT computation time: {result.computation_time:.6f} seconds")
    
    # Find peaks in spectrum
    peak_indices = []
    for i in range(1, len(result.power_spectrum) - 1):
        if (result.power_spectrum[i] > result.power_spectrum[i-1] and 
            result.power_spectrum[i] > result.power_spectrum[i+1] and
            result.power_spectrum[i] > 0.1 * np.max(result.power_spectrum)):
            peak_indices.append(i)
    
    print(f"Detected frequency peaks:")
    for idx in peak_indices:
        freq = result.frequencies[idx]
        power = result.power_spectrum[idx]
        print(f"  {freq:.1f} Hz: {power:.4f}")
    
    # Test 2: Algorithm comparison
    print("\n=== Test 2: Algorithm Performance Comparison ===")
    
    # Test different sizes
    test_sizes = [256, 512, 1024, 2048, 4096]
    
    print(f"{'Size':<8} {'Radix-2 (ms)':<15} {'Radix-4 (ms)':<15} {'Bluestein (ms)':<15}")
    print("-" * 60)
    
    for size in test_sizes:
        test_signal = np.random.randn(size) + 1j * np.random.randn(size)
        
        # Radix-2 FFT
        start_time = time.time()
        for _ in range(10):
            _ = fft_processor.radix2_fft(test_signal)
        radix2_time = (time.time() - start_time) * 100  # Convert to ms
        
        # Radix-4 FFT (if size is power of 4)
        radix4_time = "N/A"
        log4_size = int(np.log(size) / np.log(4))
        if 4 ** log4_size == size:
            start_time = time.time()
            for _ in range(10):
                _ = fft_processor.radix4_fft(test_signal)
            radix4_time = f"{(time.time() - start_time) * 100:.2f}"
        
        # Bluestein FFT
        start_time = time.time()
        for _ in range(10):
            _ = fft_processor.bluestein_fft(test_signal)
        bluestein_time = (time.time() - start_time) * 100
        
        print(f"{size:<8} {radix2_time:<15.2f} {radix4_time:<15} {bluestein_time:<15.2f}")
    
    # Test 3: Real signal optimization
    print("\n=== Test 3: Real Signal FFT Optimization ===")
    
    # Real signal
    real_signal = np.random.randn(1024)
    
    # Compare real FFT vs complex FFT
    start_time = time.time()
    for _ in range(100):
        real_fft_result = fft_processor.real_fft(real_signal)
    real_fft_time = time.time() - start_time
    
    start_time = time.time()
    for _ in range(100):
        complex_fft_result = fft_processor.fft(real_signal.astype(complex))
    complex_fft_time = time.time() - start_time
    
    print(f"Real FFT time: {real_fft_time:.6f} seconds")
    print(f"Complex FFT time: {complex_fft_time:.6f} seconds")
    print(f"Speed improvement: {complex_fft_time / real_fft_time:.2f}x")
    print(f"Real FFT output size: {len(real_fft_result)}")
    print(f"Complex FFT output size: {len(complex_fft_result)}")
    
    # Test 4: 2D FFT for image processing
    print("\n=== Test 4: 2D FFT for Image Processing ===")
    
    # Create test "image"
    image_size = 64
    x = np.linspace(-2, 2, image_size)
    y = np.linspace(-2, 2, image_size)
    X, Y = np.meshgrid(x, y)
    
    # Gaussian function
    image = np.exp(-(X**2 + Y**2))
    
    # Apply 2D FFT
    start_time = time.time()
    image_fft = fft_processor.fft_2d(image)
    fft_2d_time = time.time() - start_time
    
    print(f"2D FFT computation time: {fft_2d_time:.6f} seconds")
    print(f"Image size: {image.shape}")
    print(f"FFT magnitude range: {np.min(np.abs(image_fft)):.6f} to {np.max(np.abs(image_fft)):.6f}")
    
    # Test 5: Fast convolution
    print("\n=== Test 5: Fast Convolution ===")
    
    # Create signals for convolution
    signal_length = 1000
    kernel_length = 50
    
    test_signal = np.random.randn(signal_length)
    kernel = np.exp(-np.linspace(0, 5, kernel_length))  # Exponential decay kernel
    
    # FFT-based convolution
    start_time = time.time()
    fft_conv_result = fft_processor.convolution_fft(test_signal, kernel)
    fft_conv_time = time.time() - start_time
    
    # Direct convolution for comparison
    start_time = time.time()
    direct_conv_result = np.convolve(test_signal, kernel, mode='full')
    direct_conv_time = time.time() - start_time
    
    # Check accuracy
    max_error = np.max(np.abs(fft_conv_result.real - direct_conv_result))
    
    print(f"FFT convolution time: {fft_conv_time:.6f} seconds")
    print(f"Direct convolution time: {direct_conv_time:.6f} seconds")
    print(f"Speed improvement: {direct_conv_time / fft_conv_time:.2f}x")
    print(f"Maximum error: {max_error:.2e}")
    
    # Test 6: Arbitrary length FFT
    print("\n=== Test 6: Arbitrary Length FFT ===")
    
    # Test non-power-of-2 lengths
    arbitrary_lengths = [100, 250, 500, 750, 1000]
    
    print(f"{'Length':<8} {'Bluestein (ms)':<15} {'NumPy (ms)':<15} {'Max Error':<15}")
    print("-" * 60)
    
    for length in arbitrary_lengths:
        test_signal = np.random.randn(length) + 1j * np.random.randn(length)
        
        # Bluestein FFT
        start_time = time.time()
        bluestein_result = fft_processor.bluestein_fft(test_signal)
        bluestein_time = (time.time() - start_time) * 1000
        
        # NumPy FFT for comparison
        start_time = time.time()
        numpy_result = np.fft.fft(test_signal)
        numpy_time = (time.time() - start_time) * 1000
        
        # Check accuracy
        max_error = np.max(np.abs(bluestein_result - numpy_result))
        
        print(f"{length:<8} {bluestein_time:<15.2f} {numpy_time:<15.2f} {max_error:<15.2e}")
    
    # Test 7: Windowing effects
    print("\n=== Test 7: Windowing Effects Analysis ===")
    
    # Generate signal with spectral leakage
    fs = 1000
    t = np.linspace(0, 1, fs, endpoint=False)
    freq = 50.5  # Non-integer frequency to show leakage
    signal = np.sin(2 * np.pi * freq * t)
    
    # Different window functions
    windows = {
        'Rectangular': np.ones(len(signal)),
        'Hanning': np.hanning(len(signal)),
        'Hamming': np.hamming(len(signal)),
        'Blackman': np.blackman(len(signal))
    }
    
    print(f"Analyzing {freq} Hz signal with different windows:")
    print(f"{'Window':<12} {'Peak Freq (Hz)':<15} {'Peak Power':<12} {'Leakage':<10}")
    print("-" * 55)
    
    for window_name, window in windows.items():
        windowed_signal = signal * window
        result = fft_processor.power_spectrum(windowed_signal, fs)
        
        # Find peak
        peak_idx = np.argmax(result.power_spectrum)
        peak_freq = result.frequencies[peak_idx]
        peak_power = result.power_spectrum[peak_idx]
        
        # Measure leakage (power in adjacent bins)
        adjacent_power = (result.power_spectrum[max(0, peak_idx-1)] + 
                         result.power_spectrum[min(len(result.power_spectrum)-1, peak_idx+1)])
        leakage_ratio = adjacent_power / peak_power
        
        print(f"{window_name:<12} {peak_freq:<15.1f} {peak_power:<12.4f} {leakage_ratio:<10.4f}")


if __name__ == "__main__":
    comprehensive_fft_example()