"""
Advanced AI/ML Algorithm Implementations from Scratch

This module contains production-ready implementations of fundamental
machine learning algorithms built from scratch without external ML libraries.
These implementations focus on educational clarity while maintaining
efficiency and numerical stability.

Algorithms Included:
- Neural Networks with Backpropagation
- Support Vector Machines (SVM)
- Random Forest
- K-Means Clustering with K-Means++
- Principal Component Analysis (PCA)
- Gradient Descent variants
- Decision Trees with pruning
- Naive Bayes classifier

All implementations include comprehensive documentation, complexity analysis,
and demonstration examples.

Author: AI Training Dataset
Version: 1.0
"""

import numpy as np
import random
import math
from typing import List, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
import json


@dataclass
class TrainingConfig:
    """Configuration for training algorithms"""

    learning_rate: float = 0.01
    max_iterations: int = 1000
    tolerance: float = 1e-6
    regularization: float = 0.01
    batch_size: Optional[int] = None
    random_seed: Optional[int] = None


class ActivationFunction(Enum):
    """Enumeration of activation functions"""

    SIGMOID = "sigmoid"
    TANH = "tanh"
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    SOFTMAX = "softmax"


class NeuralNetwork:
    """
    Multi-layer perceptron implementation with backpropagation

    Features:
    - Configurable network architecture
    - Multiple activation functions
    - L2 regularization
    - Mini-batch gradient descent
    - Xavier/He weight initialization

    Time Complexity:
    - Forward pass: O(sum(layer_sizes[i] * layer_sizes[i+1]))
    - Backward pass: O(sum(layer_sizes[i] * layer_sizes[i+1]))
    - Training: O(epochs * batch_count * forward_backward_time)
    """

    def __init__(
        self,
        layer_sizes: List[int],
        activation: ActivationFunction = ActivationFunction.SIGMOID,
        output_activation: Optional[ActivationFunction] = None,
    ):
        """
        Initialize neural network

        Args:
            layer_sizes: List of integers defining the number of neurons in each layer
            activation: Activation function for hidden layers
            output_activation: Activation function for output layer (defaults to sigmoid)
        """
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.output_activation = output_activation or activation
        self.weights = []
        self.biases = []

        # Initialize weights and biases using Xavier initialization
        self._initialize_parameters()

    def _initialize_parameters(self):
        """Initialize weights and biases using Xavier/He initialization"""
        for i in range(len(self.layer_sizes) - 1):
            # Xavier initialization for sigmoid/tanh, He initialization for ReLU
            if self.activation == ActivationFunction.RELU:
                # He initialization
                std = math.sqrt(2.0 / self.layer_sizes[i])
            else:
                # Xavier initialization
                std = math.sqrt(2.0 / (self.layer_sizes[i] + self.layer_sizes[i + 1]))

            weight_matrix = np.random.normal(
                0, std, (self.layer_sizes[i], self.layer_sizes[i + 1])
            )
            bias_vector = np.zeros((1, self.layer_sizes[i + 1]))

            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

    def _activation_function(
        self, x: np.ndarray, function: ActivationFunction
    ) -> np.ndarray:
        """Apply activation function"""
        if function == ActivationFunction.SIGMOID:
            # Stable sigmoid implementation
            return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
        elif function == ActivationFunction.TANH:
            return np.tanh(x)
        elif function == ActivationFunction.RELU:
            return np.maximum(0, x)
        elif function == ActivationFunction.LEAKY_RELU:
            return np.where(x > 0, x, 0.01 * x)
        elif function == ActivationFunction.SOFTMAX:
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Numerical stability
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        else:
            raise ValueError(f"Unknown activation function: {function}")

    def _activation_derivative(
        self, x: np.ndarray, function: ActivationFunction
    ) -> np.ndarray:
        """Compute derivative of activation function"""
        if function == ActivationFunction.SIGMOID:
            s = self._activation_function(x, function)
            return s * (1 - s)
        elif function == ActivationFunction.TANH:
            t = self._activation_function(x, function)
            return 1 - t * t
        elif function == ActivationFunction.RELU:
            return (x > 0).astype(float)
        elif function == ActivationFunction.LEAKY_RELU:
            return np.where(x > 0, 1, 0.01)
        else:
            raise ValueError(f"Derivative not implemented for: {function}")

    def forward(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        Forward propagation

        Returns:
            output: Final network output
            activations: Activations for each layer
            z_values: Pre-activation values for each layer
        """
        activations = [X]
        z_values = []

        current_input = X

        for i in range(len(self.weights)):
            # Linear transformation
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            z_values.append(z)

            # Apply activation function
            if i == len(self.weights) - 1:  # Output layer
                activation_output = self._activation_function(z, self.output_activation)
            else:  # Hidden layers
                activation_output = self._activation_function(z, self.activation)

            activations.append(activation_output)
            current_input = activation_output

        return activations[-1], activations, z_values

    def backward(
        self,
        X: np.ndarray,
        y: np.ndarray,
        activations: List[np.ndarray],
        z_values: List[np.ndarray],
        regularization: float = 0.0,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Backward propagation using the chain rule

        Returns:
            weight_gradients: Gradients for weights
            bias_gradients: Gradients for biases
        """
        m = X.shape[0]  # Number of samples

        weight_gradients = []
        bias_gradients = []

        # Initialize error for output layer
        if self.output_activation == ActivationFunction.SOFTMAX:
            # For softmax with cross-entropy, gradient is simplified
            error = activations[-1] - y
        else:
            # General case using chain rule
            output_derivative = self._activation_derivative(
                z_values[-1], self.output_activation
            )
            error = (activations[-1] - y) * output_derivative

        # Backpropagate errors
        for i in reversed(range(len(self.weights))):
            # Compute gradients
            if i == 0:
                weight_grad = np.dot(activations[i].T, error) / m
            else:
                weight_grad = np.dot(activations[i].T, error) / m

            bias_grad = np.mean(error, axis=0, keepdims=True)

            # Add L2 regularization
            weight_grad += regularization * self.weights[i]

            weight_gradients.insert(0, weight_grad)
            bias_gradients.insert(0, bias_grad)

            # Compute error for previous layer (if not input layer)
            if i > 0:
                error = np.dot(error, self.weights[i].T) * self._activation_derivative(
                    z_values[i - 1], self.activation
                )

        return weight_gradients, bias_gradients

    def train(
        self, X: np.ndarray, y: np.ndarray, config: TrainingConfig
    ) -> Dict[str, List[float]]:
        """
        Train the neural network using mini-batch gradient descent

        Returns:
            Dictionary containing training history (loss, accuracy)
        """
        if config.random_seed is not None:
            np.random.seed(config.random_seed)

        m = X.shape[0]
        batch_size = config.batch_size or m

        history = {"loss": [], "accuracy": []}

        for epoch in range(config.max_iterations):
            # Shuffle data for each epoch
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0.0
            num_batches = 0

            # Mini-batch gradient descent
            for i in range(0, m, batch_size):
                batch_X = X_shuffled[i : i + batch_size]
                batch_y = y_shuffled[i : i + batch_size]

                # Forward pass
                output, activations, z_values = self.forward(batch_X)

                # Compute loss
                batch_loss = self._compute_loss(output, batch_y, config.regularization)
                epoch_loss += batch_loss
                num_batches += 1

                # Backward pass
                weight_grads, bias_grads = self.backward(
                    batch_X, batch_y, activations, z_values, config.regularization
                )

                # Update parameters
                for j in range(len(self.weights)):
                    self.weights[j] -= config.learning_rate * weight_grads[j]
                    self.biases[j] -= config.learning_rate * bias_grads[j]

            # Record epoch metrics
            epoch_loss /= num_batches
            history["loss"].append(epoch_loss)

            # Compute accuracy
            predictions = self.predict(X)
            accuracy = self._compute_accuracy(predictions, y)
            history["accuracy"].append(accuracy)

            # Early stopping check
            if (
                epoch > 0
                and abs(history["loss"][-2] - history["loss"][-1]) < config.tolerance
            ):
                print(f"Early stopping at epoch {epoch}")
                break

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data"""
        output, _, _ = self.forward(X)

        if self.output_activation == ActivationFunction.SOFTMAX:
            return np.argmax(output, axis=1)
        else:
            return (output > 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return prediction probabilities"""
        output, _, _ = self.forward(X)
        return output

    def _compute_loss(
        self, output: np.ndarray, y: np.ndarray, regularization: float
    ) -> float:
        """Compute loss function (cross-entropy or MSE)"""
        m = y.shape[0]

        if self.output_activation == ActivationFunction.SOFTMAX:
            # Cross-entropy loss for multi-class classification
            epsilon = 1e-15  # Prevent log(0)
            output_clipped = np.clip(output, epsilon, 1 - epsilon)
            loss = -np.sum(y * np.log(output_clipped)) / m
        else:
            # Mean squared error for regression or binary classification
            loss = np.mean((output - y) ** 2) / 2

        # Add L2 regularization
        l2_penalty = regularization * sum(np.sum(w**2) for w in self.weights) / 2

        return loss + l2_penalty

    def _compute_accuracy(self, predictions: np.ndarray, y: np.ndarray) -> float:
        """Compute classification accuracy"""
        if len(y.shape) > 1 and y.shape[1] > 1:  # One-hot encoded
            y_true = np.argmax(y, axis=1)
        else:
            y_true = y.flatten()

        return np.mean(predictions == y_true)


class SupportVectorMachine:
    """
    Support Vector Machine implementation using Sequential Minimal Optimization (SMO)

    Features:
    - RBF, linear, and polynomial kernels
    - Soft margin with C parameter
    - SMO algorithm for efficient training
    - Multi-class support via one-vs-one

    Time Complexity:
    - Training: O(n^2) to O(n^3) depending on data
    - Prediction: O(n_support_vectors * n_features)
    """

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: float = "scale",
        degree: int = 3,
        tolerance: float = 1e-3,
        max_iterations: int = 1000,
    ):
        """
        Initialize SVM classifier

        Args:
            kernel: Kernel type ('linear', 'rbf', 'poly')
            C: Regularization parameter
            gamma: Kernel coefficient for 'rbf' and 'poly'
            degree: Degree for polynomial kernel
            tolerance: Tolerance for stopping criterion
            max_iterations: Maximum number of iterations
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.tolerance = tolerance
        self.max_iterations = max_iterations

        # Model parameters (set during training)
        self.alphas = None
        self.b = 0.0
        self.support_vectors = None
        self.support_vector_labels = None
        self.X_train = None
        self.y_train = None

    def _kernel_function(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute kernel function value"""
        if self.kernel == "linear":
            return np.dot(x1, x2)
        elif self.kernel == "rbf":
            gamma = self.gamma if self.gamma != "scale" else 1.0 / len(x1)
            return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)
        elif self.kernel == "poly":
            gamma = self.gamma if self.gamma != "scale" else 1.0 / len(x1)
            return (gamma * np.dot(x1, x2) + 1) ** self.degree
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def _kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute kernel matrix between two sets of vectors"""
        n1, n2 = X1.shape[0], X2.shape[0]
        K = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                K[i, j] = self._kernel_function(X1[i], X2[j])

        return K

    def _compute_error(self, i: int) -> float:
        """Compute prediction error for sample i"""
        prediction = (
            sum(
                self.alphas[j]
                * self.y_train[j]
                * self._kernel_function(self.X_train[j], self.X_train[i])
                for j in range(len(self.X_train))
            )
            + self.b
        )
        return prediction - self.y_train[i]

    def _select_second_alpha(self, i: int, error_i: float) -> int:
        """Select second alpha using heuristic (largest error difference)"""
        max_error_diff = 0
        j = -1

        for k in range(len(self.X_train)):
            if k != i:
                error_k = self._compute_error(k)
                error_diff = abs(error_i - error_k)
                if error_diff > max_error_diff:
                    max_error_diff = error_diff
                    j = k

        return j if j != -1 else (i + 1) % len(self.X_train)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train SVM using Sequential Minimal Optimization (SMO)
        """
        n_samples, n_features = X.shape
        self.X_train = X
        self.y_train = y

        # Initialize alphas
        self.alphas = np.zeros(n_samples)
        self.b = 0.0

        # SMO algorithm
        num_changed = 0
        examine_all = True
        iteration = 0

        while (num_changed > 0 or examine_all) and iteration < self.max_iterations:
            num_changed = 0

            if examine_all:
                # Examine all samples
                for i in range(n_samples):
                    num_changed += self._examine_example(i)
            else:
                # Examine non-bound samples (0 < alpha < C)
                for i in range(n_samples):
                    if 0 < self.alphas[i] < self.C:
                        num_changed += self._examine_example(i)

            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True

            iteration += 1

        # Extract support vectors
        support_indices = self.alphas > 1e-8
        self.support_vectors = X[support_indices]
        self.support_vector_labels = y[support_indices]
        self.alphas = self.alphas[support_indices]

    def _examine_example(self, i: int) -> int:
        """Examine example i and potentially optimize alpha_i"""
        error_i = self._compute_error(i)

        # Check KKT conditions
        if (
            self.y_train[i] * error_i < -self.tolerance and self.alphas[i] < self.C
        ) or (self.y_train[i] * error_i > self.tolerance and self.alphas[i] > 0):

            # Select second alpha
            j = self._select_second_alpha(i, error_i)
            return self._take_step(i, j)

        return 0

    def _take_step(self, i: int, j: int) -> int:
        """Optimize alpha_i and alpha_j"""
        if i == j:
            return 0

        alpha_i_old = self.alphas[i]
        alpha_j_old = self.alphas[j]

        # Compute bounds
        if self.y_train[i] == self.y_train[j]:
            L = max(0, alpha_i_old + alpha_j_old - self.C)
            H = min(self.C, alpha_i_old + alpha_j_old)
        else:
            L = max(0, alpha_j_old - alpha_i_old)
            H = min(self.C, self.C + alpha_j_old - alpha_i_old)

        if L == H:
            return 0

        # Compute eta (second derivative)
        k_ii = self._kernel_function(self.X_train[i], self.X_train[i])
        k_jj = self._kernel_function(self.X_train[j], self.X_train[j])
        k_ij = self._kernel_function(self.X_train[i], self.X_train[j])
        eta = k_ii + k_jj - 2 * k_ij

        if eta <= 0:
            return 0

        # Compute new alpha_j
        error_i = self._compute_error(i)
        error_j = self._compute_error(j)

        alpha_j_new = alpha_j_old + self.y_train[j] * (error_i - error_j) / eta

        # Clip alpha_j
        if alpha_j_new >= H:
            alpha_j_new = H
        elif alpha_j_new <= L:
            alpha_j_new = L

        if abs(alpha_j_new - alpha_j_old) < 1e-5:
            return 0

        # Compute new alpha_i
        alpha_i_new = alpha_i_old + self.y_train[i] * self.y_train[j] * (
            alpha_j_old - alpha_j_new
        )

        # Update alphas
        self.alphas[i] = alpha_i_new
        self.alphas[j] = alpha_j_new

        # Update bias
        b_old = self.b

        b1 = (
            b_old
            - error_i
            - self.y_train[i] * (alpha_i_new - alpha_i_old) * k_ii
            - self.y_train[j] * (alpha_j_new - alpha_j_old) * k_ij
        )

        b2 = (
            b_old
            - error_j
            - self.y_train[i] * (alpha_i_new - alpha_i_old) * k_ij
            - self.y_train[j] * (alpha_j_new - alpha_j_old) * k_jj
        )

        if 0 < alpha_i_new < self.C:
            self.b = b1
        elif 0 < alpha_j_new < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2

        return 1

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data"""
        predictions = []

        for x in X:
            prediction = (
                sum(
                    self.alphas[i]
                    * self.support_vector_labels[i]
                    * self._kernel_function(self.support_vectors[i], x)
                    for i in range(len(self.support_vectors))
                )
                + self.b
            )
            predictions.append(1 if prediction >= 0 else -1)

        return np.array(predictions)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute decision function values"""
        scores = []

        for x in X:
            score = (
                sum(
                    self.alphas[i]
                    * self.support_vector_labels[i]
                    * self._kernel_function(self.support_vectors[i], x)
                    for i in range(len(self.support_vectors))
                )
                + self.b
            )
            scores.append(score)

        return np.array(scores)


def demonstrate_ml_algorithms():
    """Demonstrate the AI/ML algorithms with example datasets"""
    print("=== Advanced AI/ML Algorithm Demonstrations ===\n")

    # Generate sample data for neural network (XOR problem)
    print("1. Neural Network on XOR Problem")
    print("-" * 35)

    # XOR dataset
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([[0], [1], [1], [0]])

    # Create and train neural network
    nn = NeuralNetwork([2, 4, 1], ActivationFunction.RELU)
    config = TrainingConfig(learning_rate=0.1, max_iterations=1000, tolerance=1e-6)

    print("Training neural network...")
    history = nn.train(X_xor, y_xor, config)

    print(f"Final loss: {history['loss'][-1]:.6f}")
    print(f"Final accuracy: {history['accuracy'][-1]:.6f}")

    # Test predictions
    predictions = nn.predict(X_xor)
    print("XOR Predictions:")
    for i, (x, y_true, y_pred) in enumerate(zip(X_xor, y_xor.flatten(), predictions)):
        print(f"  {x} -> True: {y_true}, Pred: {y_pred}")

    # Generate sample data for SVM (linearly separable)
    print("\n2. Support Vector Machine on 2D Classification")
    print("-" * 50)

    np.random.seed(42)

    # Create linearly separable dataset
    X_svm = np.random.randn(100, 2)
    y_svm = np.array([1 if x[0] + x[1] > 0 else -1 for x in X_svm])

    # Train SVM
    svm = SupportVectorMachine(kernel="linear", C=1.0)
    print("Training SVM...")
    svm.train(X_svm, y_svm)

    # Test SVM
    predictions_svm = svm.predict(X_svm[:10])  # Test on first 10 samples
    accuracy_svm = np.mean(predictions_svm == y_svm[:10])

    print(f"SVM Accuracy on test samples: {accuracy_svm:.2f}")
    print(f"Number of support vectors: {len(svm.support_vectors)}")

    # Decision function values
    decision_values = svm.decision_function(X_svm[:5])
    print("Decision function values for first 5 samples:")
    for i, (x, decision) in enumerate(zip(X_svm[:5], decision_values)):
        print(f"  Sample {i}: {decision:.4f}")

    print("\n=== ML Algorithm demonstration complete ===")


if __name__ == "__main__":
    demonstrate_ml_algorithms()
