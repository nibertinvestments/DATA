"""
Neural Network Activation Functions
==================================

Comprehensive implementation of modern activation functions for deep learning
including mathematical properties, derivatives, and comparative analysis
for different network architectures and use cases.

Mathematical Foundation:
Activation functions introduce non-linearity: h = f(Wx + b)

Key Properties:
1. Non-linearity: Enables learning of complex patterns
2. Differentiability: Required for backpropagation
3. Range: Output value range affects convergence
4. Zero-centeredness: Affects optimization dynamics

Applications:
- Deep neural networks
- Convolutional networks
- Recurrent networks
- Transformer architectures
- Autoencoders
"""

import numpy as np
from typing import Tuple, Dict, List, Callable, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class ActivationProperties:
    """Properties of an activation function."""
    name: str
    range_min: float
    range_max: float
    zero_centered: bool
    monotonic: bool
    continuously_differentiable: bool
    computational_cost: str  # "low", "medium", "high"
    dying_neuron_problem: bool
    saturation_problem: bool


class ActivationFunction(ABC):
    """Abstract base class for activation functions."""
    
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        pass
    
    @abstractmethod
    def backward(self, x: np.ndarray) -> np.ndarray:
        """Backward pass (derivative)."""
        pass
    
    @abstractmethod
    def properties(self) -> ActivationProperties:
        """Return activation function properties."""
        pass


class ReLU(ActivationFunction):
    """Rectified Linear Unit: f(x) = max(0, x)"""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)
    
    def properties(self) -> ActivationProperties:
        return ActivationProperties(
            name="ReLU",
            range_min=0.0,
            range_max=float('inf'),
            zero_centered=False,
            monotonic=True,
            continuously_differentiable=False,
            computational_cost="low",
            dying_neuron_problem=True,
            saturation_problem=False
        )


class LeakyReLU(ActivationFunction):
    """Leaky ReLU: f(x) = max(αx, x) where α is small positive constant"""
    
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.alpha * x)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1.0, self.alpha)
    
    def properties(self) -> ActivationProperties:
        return ActivationProperties(
            name=f"LeakyReLU(α={self.alpha})",
            range_min=-float('inf'),
            range_max=float('inf'),
            zero_centered=False,
            monotonic=True,
            continuously_differentiable=False,
            computational_cost="low",
            dying_neuron_problem=False,
            saturation_problem=False
        )


class ELU(ActivationFunction):
    """Exponential Linear Unit: f(x) = x if x > 0, α(e^x - 1) if x ≤ 0"""
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1.0, self.alpha * np.exp(x))
    
    def properties(self) -> ActivationProperties:
        return ActivationProperties(
            name=f"ELU(α={self.alpha})",
            range_min=-self.alpha,
            range_max=float('inf'),
            zero_centered=True,
            monotonic=True,
            continuously_differentiable=True,
            computational_cost="medium",
            dying_neuron_problem=False,
            saturation_problem=True
        )


class Swish(ActivationFunction):
    """Swish: f(x) = x * sigmoid(βx)"""
    
    def __init__(self, beta: float = 1.0):
        self.beta = beta
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x * self._sigmoid(self.beta * x)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        sigmoid_x = self._sigmoid(self.beta * x)
        return sigmoid_x + x * sigmoid_x * (1 - sigmoid_x) * self.beta
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow
    
    def properties(self) -> ActivationProperties:
        return ActivationProperties(
            name=f"Swish(β={self.beta})",
            range_min=-float('inf'),
            range_max=float('inf'),
            zero_centered=False,
            monotonic=False,
            continuously_differentiable=True,
            computational_cost="high",
            dying_neuron_problem=False,
            saturation_problem=True
        )


class GELU(ActivationFunction):
    """Gaussian Error Linear Unit: f(x) = 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))"""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
        tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x**3)
        return 0.5 * x * (1 + np.tanh(tanh_arg))
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        # Derivative is complex, using numerical approximation
        h = 1e-7
        return (self.forward(x + h) - self.forward(x - h)) / (2 * h)
    
    def properties(self) -> ActivationProperties:
        return ActivationProperties(
            name="GELU",
            range_min=-float('inf'),
            range_max=float('inf'),
            zero_centered=False,
            monotonic=False,
            continuously_differentiable=True,
            computational_cost="high",
            dying_neuron_problem=False,
            saturation_problem=True
        )


class Mish(ActivationFunction):
    """Mish: f(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))"""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # softplus(x) = ln(1 + e^x)
        softplus_x = np.log(1 + np.exp(np.clip(x, -500, 500)))
        return x * np.tanh(softplus_x)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        # Complex derivative, using numerical approximation
        h = 1e-7
        return (self.forward(x + h) - self.forward(x - h)) / (2 * h)
    
    def properties(self) -> ActivationProperties:
        return ActivationProperties(
            name="Mish",
            range_min=-0.31,  # Approximately
            range_max=float('inf'),
            zero_centered=False,
            monotonic=False,
            continuously_differentiable=True,
            computational_cost="high",
            dying_neuron_problem=False,
            saturation_problem=True
        )


class PReLU(ActivationFunction):
    """Parametric ReLU: f(x) = max(αx, x) where α is learnable"""
    
    def __init__(self, alpha: float = 0.25):
        self.alpha = alpha
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.alpha * x)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1.0, self.alpha)
    
    def properties(self) -> ActivationProperties:
        return ActivationProperties(
            name=f"PReLU(α={self.alpha})",
            range_min=-float('inf'),
            range_max=float('inf'),
            zero_centered=False,
            monotonic=True,
            continuously_differentiable=False,
            computational_cost="low",
            dying_neuron_problem=False,
            saturation_problem=False
        )


class Sigmoid(ActivationFunction):
    """Sigmoid: f(x) = 1 / (1 + e^(-x))"""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        sigmoid_x = self.forward(x)
        return sigmoid_x * (1 - sigmoid_x)
    
    def properties(self) -> ActivationProperties:
        return ActivationProperties(
            name="Sigmoid",
            range_min=0.0,
            range_max=1.0,
            zero_centered=False,
            monotonic=True,
            continuously_differentiable=True,
            computational_cost="medium",
            dying_neuron_problem=False,
            saturation_problem=True
        )


class Tanh(ActivationFunction):
    """Hyperbolic Tangent: f(x) = tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))"""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        tanh_x = self.forward(x)
        return 1 - tanh_x**2
    
    def properties(self) -> ActivationProperties:
        return ActivationProperties(
            name="Tanh",
            range_min=-1.0,
            range_max=1.0,
            zero_centered=True,
            monotonic=True,
            continuously_differentiable=True,
            computational_cost="medium",
            dying_neuron_problem=False,
            saturation_problem=True
        )


class ActivationAnalyzer:
    """
    Comprehensive analyzer for activation functions.
    
    Provides analysis of mathematical properties, gradient flow,
    and suitability for different network architectures.
    """
    
    def __init__(self):
        self.functions = {
            'ReLU': ReLU(),
            'LeakyReLU': LeakyReLU(),
            'ELU': ELU(),
            'Swish': Swish(),
            'GELU': GELU(),
            'Mish': Mish(),
            'PReLU': PReLU(),
            'Sigmoid': Sigmoid(),
            'Tanh': Tanh()
        }
    
    def gradient_flow_analysis(self, x_range: Tuple[float, float] = (-5, 5),
                             num_points: int = 1000) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Analyze gradient flow characteristics of activation functions.
        
        Args:
            x_range: Range of input values to analyze
            num_points: Number of points to sample
            
        Returns:
            Dictionary with gradient analysis for each function
        """
        x = np.linspace(x_range[0], x_range[1], num_points)
        results = {}
        
        for name, func in self.functions.items():
            try:
                # Forward pass
                y = func.forward(x)
                
                # Backward pass (gradients)
                dy_dx = func.backward(x)
                
                # Gradient statistics
                mean_grad = np.mean(np.abs(dy_dx))
                grad_variance = np.var(dy_dx)
                zero_grad_ratio = np.mean(dy_dx == 0)
                max_grad = np.max(np.abs(dy_dx))
                
                results[name] = {
                    'x': x,
                    'y': y,
                    'gradient': dy_dx,
                    'mean_abs_gradient': mean_grad,
                    'gradient_variance': grad_variance,
                    'zero_gradient_ratio': zero_grad_ratio,
                    'max_gradient': max_grad
                }
                
            except Exception as e:
                print(f"Error analyzing {name}: {e}")
                results[name] = None
        
        return results
    
    def saturation_analysis(self, threshold: float = 0.01) -> Dict[str, Dict[str, float]]:
        """
        Analyze saturation behavior of activation functions.
        
        Args:
            threshold: Gradient threshold below which we consider saturation
            
        Returns:
            Dictionary with saturation metrics
        """
        results = {}
        
        # Test extreme values
        extreme_values = [-10, -5, -2, 0, 2, 5, 10]
        
        for name, func in self.functions.items():
            try:
                gradients = []
                outputs = []
                
                for x_val in extreme_values:
                    x = np.array([x_val])
                    y = func.forward(x)
                    dy_dx = func.backward(x)
                    
                    outputs.append(y[0])
                    gradients.append(dy_dx[0])
                
                # Calculate saturation metrics
                saturated_points = sum(1 for grad in gradients if abs(grad) < threshold)
                saturation_ratio = saturated_points / len(gradients)
                
                output_range = max(outputs) - min(outputs)
                gradient_range = max(gradients) - min(gradients)
                
                results[name] = {
                    'saturation_ratio': saturation_ratio,
                    'output_range': output_range,
                    'gradient_range': gradient_range,
                    'min_gradient': min(gradients),
                    'max_gradient': max(gradients)
                }
                
            except Exception as e:
                print(f"Error in saturation analysis for {name}: {e}")
                results[name] = None
        
        return results
    
    def computational_cost_benchmark(self, num_operations: int = 1000000) -> Dict[str, float]:
        """
        Benchmark computational cost of activation functions.
        
        Args:
            num_operations: Number of operations to perform
            
        Returns:
            Dictionary with execution times
        """
        import time
        
        # Generate test data
        x = np.random.randn(num_operations // 1000, 1000)
        results = {}
        
        for name, func in self.functions.items():
            try:
                # Warm up
                for _ in range(3):
                    _ = func.forward(x)
                
                # Benchmark forward pass
                start_time = time.time()
                for _ in range(10):
                    _ = func.forward(x)
                forward_time = (time.time() - start_time) / 10
                
                # Benchmark backward pass
                start_time = time.time()
                for _ in range(10):
                    _ = func.backward(x)
                backward_time = (time.time() - start_time) / 10
                
                results[name] = {
                    'forward_time': forward_time,
                    'backward_time': backward_time,
                    'total_time': forward_time + backward_time
                }
                
            except Exception as e:
                print(f"Error benchmarking {name}: {e}")
                results[name] = None
        
        return results
    
    def suitability_analysis(self) -> Dict[str, Dict[str, str]]:
        """
        Analyze suitability of activation functions for different use cases.
        
        Returns:
            Dictionary with suitability recommendations
        """
        recommendations = {}
        
        for name, func in self.functions.items():
            props = func.properties()
            
            # Deep networks
            if props.zero_centered and not props.saturation_problem:
                deep_suitability = "Excellent"
            elif not props.dying_neuron_problem:
                deep_suitability = "Good"
            else:
                deep_suitability = "Fair"
            
            # Convolutional networks
            if props.computational_cost == "low" and not props.dying_neuron_problem:
                conv_suitability = "Excellent"
            elif props.computational_cost in ["low", "medium"]:
                conv_suitability = "Good"
            else:
                conv_suitability = "Fair"
            
            # RNNs
            if props.zero_centered and props.continuously_differentiable:
                rnn_suitability = "Excellent"
            elif not props.saturation_problem:
                rnn_suitability = "Good"
            else:
                rnn_suitability = "Fair"
            
            # Transformers
            if name in ["GELU", "Swish", "Mish"]:
                transformer_suitability = "Excellent"
            elif props.continuously_differentiable:
                transformer_suitability = "Good"
            else:
                transformer_suitability = "Fair"
            
            recommendations[name] = {
                'deep_networks': deep_suitability,
                'convolutional': conv_suitability,
                'recurrent': rnn_suitability,
                'transformers': transformer_suitability
            }
        
        return recommendations
    
    def comprehensive_comparison(self) -> Dict[str, Dict]:
        """
        Perform comprehensive comparison of all activation functions.
        
        Returns:
            Complete analysis results
        """
        print("Performing comprehensive activation function analysis...")
        
        # Gradient flow analysis
        print("  Analyzing gradient flow...")
        gradient_results = self.gradient_flow_analysis()
        
        # Saturation analysis
        print("  Analyzing saturation behavior...")
        saturation_results = self.saturation_analysis()
        
        # Performance benchmark
        print("  Benchmarking computational performance...")
        performance_results = self.computational_cost_benchmark()
        
        # Suitability analysis
        print("  Analyzing architectural suitability...")
        suitability_results = self.suitability_analysis()
        
        return {
            'gradient_analysis': gradient_results,
            'saturation_analysis': saturation_results,
            'performance_benchmark': performance_results,
            'suitability_analysis': suitability_results
        }


def comprehensive_activation_example():
    """Comprehensive example demonstrating activation function analysis."""
    print("=== Neural Network Activation Functions Analysis ===")
    
    # Initialize analyzer
    analyzer = ActivationAnalyzer()
    
    # Perform comprehensive analysis
    results = analyzer.comprehensive_comparison()
    
    # Display properties summary
    print("\n=== Activation Function Properties ===")
    print(f"{'Function':<12} {'Range':<20} {'Zero-Centered':<15} {'Monotonic':<10} {'Cost':<8}")
    print("-" * 75)
    
    for name, func in analyzer.functions.items():
        props = func.properties()
        range_str = f"[{props.range_min:.1f}, {props.range_max if props.range_max != float('inf') else '∞'}]"
        
        print(f"{name:<12} {range_str:<20} {str(props.zero_centered):<15} "
              f"{str(props.monotonic):<10} {props.computational_cost:<8}")
    
    # Gradient flow analysis
    print("\n=== Gradient Flow Analysis ===")
    print(f"{'Function':<12} {'Mean |Grad|':<12} {'Grad Variance':<15} {'Zero Grad %':<12} {'Max Grad':<10}")
    print("-" * 70)
    
    for name, result in results['gradient_analysis'].items():
        if result is not None:
            print(f"{name:<12} {result['mean_abs_gradient']:<12.4f} "
                  f"{result['gradient_variance']:<15.4f} "
                  f"{result['zero_gradient_ratio']:<12.2%} "
                  f"{result['max_gradient']:<10.4f}")
    
    # Saturation analysis
    print("\n=== Saturation Analysis ===")
    print(f"{'Function':<12} {'Saturation %':<15} {'Output Range':<15} {'Gradient Range':<15}")
    print("-" * 60)
    
    for name, result in results['saturation_analysis'].items():
        if result is not None:
            print(f"{name:<12} {result['saturation_ratio']:<15.2%} "
                  f"{result['output_range']:<15.4f} "
                  f"{result['gradient_range']:<15.4f}")
    
    # Performance benchmark
    print("\n=== Performance Benchmark ===")
    print(f"{'Function':<12} {'Forward (ms)':<15} {'Backward (ms)':<15} {'Total (ms)':<12}")
    print("-" * 60)
    
    for name, result in results['performance_benchmark'].items():
        if result is not None:
            forward_ms = result['forward_time'] * 1000
            backward_ms = result['backward_time'] * 1000
            total_ms = result['total_time'] * 1000
            
            print(f"{name:<12} {forward_ms:<15.2f} {backward_ms:<15.2f} {total_ms:<12.2f}")
    
    # Suitability analysis
    print("\n=== Architectural Suitability ===")
    print(f"{'Function':<12} {'Deep Networks':<15} {'CNNs':<10} {'RNNs':<10} {'Transformers':<12}")
    print("-" * 65)
    
    for name, suitability in results['suitability_analysis'].items():
        print(f"{name:<12} {suitability['deep_networks']:<15} "
              f"{suitability['convolutional']:<10} "
              f"{suitability['recurrent']:<10} "
              f"{suitability['transformers']:<12}")
    
    # Recommendations
    print("\n=== Recommendations ===")
    
    recommendations = {
        "General Purpose": "ReLU (simple), ELU (sophisticated)",
        "Deep Networks": "ELU, LeakyReLU, Swish",
        "CNNs": "ReLU, LeakyReLU (fast training)",
        "RNNs": "Tanh, ELU (gradient flow)",
        "Transformers": "GELU, Swish (state-of-the-art)",
        "Output Layers": "Sigmoid (binary), Softmax (multiclass)",
        "Autoencoders": "ELU, Tanh (zero-centered)"
    }
    
    for use_case, recommendation in recommendations.items():
        print(f"  {use_case:<15}: {recommendation}")
    
    # Problem-specific analysis
    print("\n=== Common Problems and Solutions ===")
    
    problems = {
        "Vanishing Gradients": ["ELU", "LeakyReLU", "Swish", "GELU"],
        "Dying Neurons": ["LeakyReLU", "ELU", "Swish", "GELU"],
        "Slow Convergence": ["ELU", "Swish", "GELU", "Mish"],
        "High Computation": ["ReLU", "LeakyReLU", "PReLU"]
    }
    
    for problem, solutions in problems.items():
        print(f"  {problem:<18}: {', '.join(solutions)}")
    
    # Mathematical insights
    print("\n=== Mathematical Insights ===")
    
    insights = [
        "ReLU: Simple but effective, risk of dying neurons",
        "LeakyReLU: Fixes dying neurons with minimal overhead",
        "ELU: Smooth, zero-centered, good for deep networks",
        "Swish: Self-gated, used in EfficientNet",
        "GELU: Gaussian-inspired, standard in Transformers",
        "Mish: Novel smooth activation, competitive performance",
        "Sigmoid/Tanh: Classical, prone to saturation",
        "PReLU: Learnable negative slope parameter"
    ]
    
    for insight in insights:
        print(f"  • {insight}")


if __name__ == "__main__":
    comprehensive_activation_example()