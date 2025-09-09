"""
Genetic Algorithm with Adaptive Mutation
========================================

Advanced genetic algorithm implementation with adaptive mutation rates,
multiple selection strategies, and advanced crossover operators for
optimization problems in finance, engineering, and machine learning.

Mathematical Foundation:
Fitness Selection: P(i) = f(i) / Σf(j)
Crossover: offspring = α*parent1 + (1-α)*parent2
Mutation: x' = x + N(0, σ²) where σ adapts based on population diversity

Applications:
- Portfolio optimization
- Neural network hyperparameter tuning
- Feature selection
- Scheduling problems
- Engineering design optimization
"""

import numpy as np
from typing import List, Tuple, Callable, Dict, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum
import random
import matplotlib.pyplot as plt


class SelectionMethod(Enum):
    """Selection methods for genetic algorithm."""
    TOURNAMENT = "tournament"
    ROULETTE_WHEEL = "roulette_wheel"
    RANK_BASED = "rank_based"
    STOCHASTIC_UNIVERSAL = "stochastic_universal"


class CrossoverMethod(Enum):
    """Crossover methods for genetic algorithm."""
    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"
    ARITHMETIC = "arithmetic"
    BLEND_ALPHA = "blend_alpha"


@dataclass
class GAConfig:
    """Configuration for genetic algorithm."""
    population_size: int = 100
    chromosome_length: int = 20
    num_generations: int = 500
    mutation_rate: float = 0.01
    crossover_rate: float = 0.8
    elitism_rate: float = 0.1
    selection_method: SelectionMethod = SelectionMethod.TOURNAMENT
    crossover_method: CrossoverMethod = CrossoverMethod.ARITHMETIC
    adaptive_mutation: bool = True
    diversity_threshold: float = 0.01
    bounds: Tuple[float, float] = (-10.0, 10.0)


class FitnessFunction(ABC):
    """Abstract base class for fitness functions."""
    
    @abstractmethod
    def evaluate(self, chromosome: np.ndarray) -> float:
        """Evaluate fitness of a chromosome."""
        pass
    
    @abstractmethod
    def is_maximization(self) -> bool:
        """Return True if this is a maximization problem."""
        pass


class RastriginFunction(FitnessFunction):
    """Rastrigin function - multimodal optimization test function."""
    
    def __init__(self, dimension: int = 10):
        self.dimension = dimension
        self.A = 10
    
    def evaluate(self, chromosome: np.ndarray) -> float:
        """Rastrigin function: f(x) = A*n + Σ[xi² - A*cos(2π*xi)]"""
        n = len(chromosome)
        return -(self.A * n + np.sum(chromosome**2 - self.A * np.cos(2 * np.pi * chromosome)))
    
    def is_maximization(self) -> bool:
        return True  # We negate the function, so maximize


class PortfolioOptimizationFunction(FitnessFunction):
    """Portfolio optimization using Sharpe ratio."""
    
    def __init__(self, returns: np.ndarray, risk_free_rate: float = 0.02):
        self.returns = returns  # Expected returns for each asset
        self.cov_matrix = np.cov(returns.T)
        self.risk_free_rate = risk_free_rate
    
    def evaluate(self, weights: np.ndarray) -> float:
        """Calculate negative Sharpe ratio (minimize for maximization)."""
        # Normalize weights to sum to 1
        weights = np.abs(weights)
        weights = weights / np.sum(weights)
        
        # Portfolio return and risk
        portfolio_return = np.dot(weights, self.returns)
        portfolio_variance = np.dot(weights.T, np.dot(self.cov_matrix, weights))
        portfolio_std = np.sqrt(portfolio_variance)
        
        # Sharpe ratio
        if portfolio_std == 0:
            return -1000  # Penalty for zero variance
        
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std
        return sharpe_ratio
    
    def is_maximization(self) -> bool:
        return True


class FeatureSelectionFunction(FitnessFunction):
    """Feature selection for machine learning."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, model_type: str = "linear"):
        self.X = X
        self.y = y
        self.model_type = model_type
        self.n_features = X.shape[1]
    
    def evaluate(self, chromosome: np.ndarray) -> float:
        """Evaluate feature subset using cross-validation."""
        # Convert to binary selection
        selected_features = chromosome > 0.5
        
        if not np.any(selected_features):
            return -1000  # Penalty for no features
        
        X_selected = self.X[:, selected_features]
        
        # Simple cross-validation score simulation
        # In practice, use sklearn cross_val_score
        try:
            if self.model_type == "linear":
                # Simulate linear regression performance
                n_selected = np.sum(selected_features)
                complexity_penalty = n_selected / self.n_features
                
                # Simulate R² score with complexity penalty
                simulated_r2 = 0.8 - complexity_penalty * 0.3 + np.random.normal(0, 0.1)
                return max(0, simulated_r2)
            
        except Exception:
            return -1000
        
        return 0.5  # Default score
    
    def is_maximization(self) -> bool:
        return True


class AdaptiveGeneticAlgorithm:
    """
    Advanced genetic algorithm with adaptive mutation and multiple operators.
    
    Features:
    - Adaptive mutation rates based on population diversity
    - Multiple selection and crossover strategies
    - Elitism with diversity preservation
    - Convergence analysis and early stopping
    """
    
    def __init__(self, fitness_function: FitnessFunction, config: GAConfig):
        self.fitness_function = fitness_function
        self.config = config
        
        # Initialize population
        self.population = self._initialize_population()
        self.fitness_scores = np.zeros(config.population_size)
        
        # Evolution tracking
        self.generation = 0
        self.best_fitness_history = []
        self.average_fitness_history = []
        self.diversity_history = []
        self.mutation_rate_history = []
        
        # Adaptive parameters
        self.current_mutation_rate = config.mutation_rate
        
    def _initialize_population(self) -> np.ndarray:
        """Initialize random population within bounds."""
        low, high = self.config.bounds
        return np.random.uniform(
            low, high, 
            (self.config.population_size, self.config.chromosome_length)
        )
    
    def _evaluate_population(self):
        """Evaluate fitness for entire population."""
        for i, chromosome in enumerate(self.population):
            self.fitness_scores[i] = self.fitness_function.evaluate(chromosome)
    
    def _calculate_diversity(self) -> float:
        """Calculate population diversity using average pairwise distance."""
        distances = []
        
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                distance = np.linalg.norm(self.population[i] - self.population[j])
                distances.append(distance)
        
        return np.mean(distances) if distances else 0.0
    
    def _adapt_mutation_rate(self, diversity: float):
        """Adapt mutation rate based on population diversity."""
        if not self.config.adaptive_mutation:
            return
        
        # Increase mutation if diversity is low, decrease if high
        if diversity < self.config.diversity_threshold:
            self.current_mutation_rate = min(0.5, self.current_mutation_rate * 1.1)
        else:
            self.current_mutation_rate = max(0.001, self.current_mutation_rate * 0.9)
    
    def _tournament_selection(self, tournament_size: int = 3) -> int:
        """Tournament selection."""
        tournament_indices = np.random.choice(
            self.config.population_size, tournament_size, replace=False
        )
        tournament_fitness = self.fitness_scores[tournament_indices]
        
        if self.fitness_function.is_maximization():
            winner_idx = np.argmax(tournament_fitness)
        else:
            winner_idx = np.argmin(tournament_fitness)
        
        return tournament_indices[winner_idx]
    
    def _roulette_wheel_selection(self) -> int:
        """Roulette wheel selection."""
        fitness_scores = self.fitness_scores.copy()
        
        # Handle minimization problems
        if not self.fitness_function.is_maximization():
            fitness_scores = -fitness_scores
        
        # Shift to positive values if needed
        min_fitness = np.min(fitness_scores)
        if min_fitness < 0:
            fitness_scores = fitness_scores - min_fitness + 1e-8
        
        # Calculate selection probabilities
        total_fitness = np.sum(fitness_scores)
        if total_fitness == 0:
            return np.random.randint(self.config.population_size)
        
        probabilities = fitness_scores / total_fitness
        return np.random.choice(self.config.population_size, p=probabilities)
    
    def _rank_based_selection(self) -> int:
        """Rank-based selection."""
        if self.fitness_function.is_maximization():
            ranks = np.argsort(-self.fitness_scores)
        else:
            ranks = np.argsort(self.fitness_scores)
        
        # Linear ranking
        n = self.config.population_size
        rank_probabilities = np.zeros(n)
        for i, rank in enumerate(ranks):
            rank_probabilities[rank] = (2 - 1.2) / n + (2 * 1.2 * i) / (n * (n - 1))
        
        return np.random.choice(n, p=rank_probabilities)
    
    def _select_parent(self) -> int:
        """Select parent based on configured selection method."""
        if self.config.selection_method == SelectionMethod.TOURNAMENT:
            return self._tournament_selection()
        elif self.config.selection_method == SelectionMethod.ROULETTE_WHEEL:
            return self._roulette_wheel_selection()
        elif self.config.selection_method == SelectionMethod.RANK_BASED:
            return self._rank_based_selection()
        else:
            return self._tournament_selection()  # Default
    
    def _arithmetic_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Arithmetic crossover."""
        alpha = np.random.random()
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent1 + alpha * parent2
        return child1, child2
    
    def _uniform_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Uniform crossover."""
        mask = np.random.random(len(parent1)) < 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        return child1, child2
    
    def _single_point_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Single-point crossover."""
        crossover_point = np.random.randint(1, len(parent1))
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        return child1, child2
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform crossover based on configured method."""
        if self.config.crossover_method == CrossoverMethod.ARITHMETIC:
            return self._arithmetic_crossover(parent1, parent2)
        elif self.config.crossover_method == CrossoverMethod.UNIFORM:
            return self._uniform_crossover(parent1, parent2)
        elif self.config.crossover_method == CrossoverMethod.SINGLE_POINT:
            return self._single_point_crossover(parent1, parent2)
        else:
            return self._arithmetic_crossover(parent1, parent2)  # Default
    
    def _mutate(self, chromosome: np.ndarray) -> np.ndarray:
        """Mutate chromosome with adaptive Gaussian mutation."""
        mutated = chromosome.copy()
        
        for i in range(len(chromosome)):
            if np.random.random() < self.current_mutation_rate:
                # Adaptive Gaussian mutation
                mutation_strength = self.current_mutation_rate * (self.config.bounds[1] - self.config.bounds[0]) * 0.1
                mutated[i] += np.random.normal(0, mutation_strength)
                
                # Keep within bounds
                mutated[i] = np.clip(mutated[i], self.config.bounds[0], self.config.bounds[1])
        
        return mutated
    
    def _evolve_generation(self):
        """Evolve one generation."""
        new_population = []
        
        # Elitism: keep best individuals
        elite_count = int(self.config.elitism_rate * self.config.population_size)
        if elite_count > 0:
            if self.fitness_function.is_maximization():
                elite_indices = np.argsort(-self.fitness_scores)[:elite_count]
            else:
                elite_indices = np.argsort(self.fitness_scores)[:elite_count]
            
            for idx in elite_indices:
                new_population.append(self.population[idx].copy())
        
        # Generate offspring
        while len(new_population) < self.config.population_size:
            # Selection
            parent1_idx = self._select_parent()
            parent2_idx = self._select_parent()
            
            parent1 = self.population[parent1_idx]
            parent2 = self.population[parent2_idx]
            
            # Crossover
            if np.random.random() < self.config.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)
            
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        self.population = np.array(new_population[:self.config.population_size])
    
    def _check_convergence(self) -> bool:
        """Check if algorithm has converged."""
        if len(self.best_fitness_history) < 50:
            return False
        
        recent_best = self.best_fitness_history[-50:]
        improvement = abs(recent_best[-1] - recent_best[0])
        
        return improvement < 1e-6
    
    def evolve(self, verbose: bool = False) -> Dict[str, Union[np.ndarray, List[float]]]:
        """
        Run the genetic algorithm.
        
        Args:
            verbose: Whether to print progress
            
        Returns:
            Dictionary with evolution results
        """
        if verbose:
            print(f"Starting evolution with {self.config.population_size} individuals for {self.config.num_generations} generations")
        
        # Initial evaluation
        self._evaluate_population()
        
        for generation in range(self.config.num_generations):
            self.generation = generation
            
            # Calculate metrics
            diversity = self._calculate_diversity()
            best_fitness = np.max(self.fitness_scores) if self.fitness_function.is_maximization() else np.min(self.fitness_scores)
            avg_fitness = np.mean(self.fitness_scores)
            
            # Adapt mutation rate
            self._adapt_mutation_rate(diversity)
            
            # Record history
            self.best_fitness_history.append(best_fitness)
            self.average_fitness_history.append(avg_fitness)
            self.diversity_history.append(diversity)
            self.mutation_rate_history.append(self.current_mutation_rate)
            
            # Print progress
            if verbose and generation % 50 == 0:
                print(f"Generation {generation:3d}: Best={best_fitness:.6f}, Avg={avg_fitness:.6f}, "
                      f"Diversity={diversity:.6f}, MutRate={self.current_mutation_rate:.6f}")
            
            # Check convergence
            if self._check_convergence():
                if verbose:
                    print(f"Converged at generation {generation}")
                break
            
            # Evolve next generation
            self._evolve_generation()
            self._evaluate_population()
        
        # Find best solution
        if self.fitness_function.is_maximization():
            best_idx = np.argmax(self.fitness_scores)
        else:
            best_idx = np.argmin(self.fitness_scores)
        
        best_solution = self.population[best_idx]
        best_fitness = self.fitness_scores[best_idx]
        
        return {
            'best_solution': best_solution,
            'best_fitness': best_fitness,
            'best_fitness_history': self.best_fitness_history,
            'average_fitness_history': self.average_fitness_history,
            'diversity_history': self.diversity_history,
            'mutation_rate_history': self.mutation_rate_history,
            'generations_run': self.generation + 1
        }


def genetic_algorithm_comprehensive_example():
    """Comprehensive example demonstrating genetic algorithm capabilities."""
    print("=== Genetic Algorithm with Adaptive Mutation Example ===")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Test 1: Rastrigin function optimization
    print("\n=== Test 1: Rastrigin Function Optimization ===")
    
    rastrigin = RastriginFunction(dimension=10)
    config_rastrigin = GAConfig(
        population_size=100,
        chromosome_length=10,
        num_generations=300,
        mutation_rate=0.1,
        crossover_rate=0.8,
        elitism_rate=0.1,
        adaptive_mutation=True,
        bounds=(-5.12, 5.12)
    )
    
    ga_rastrigin = AdaptiveGeneticAlgorithm(rastrigin, config_rastrigin)
    results_rastrigin = ga_rastrigin.evolve(verbose=True)
    
    print(f"\nRastrigin Results:")
    print(f"  Best fitness: {results_rastrigin['best_fitness']:.6f}")
    print(f"  Best solution: {results_rastrigin['best_solution']}")
    print(f"  Generations run: {results_rastrigin['generations_run']}")
    print(f"  Solution norm: {np.linalg.norm(results_rastrigin['best_solution']):.6f} (ideal: 0)")
    
    # Test 2: Portfolio optimization
    print("\n=== Test 2: Portfolio Optimization ===")
    
    # Generate synthetic asset returns
    n_assets = 8
    np.random.seed(42)
    returns = np.random.normal(0.08, 0.15, n_assets)  # Annual returns
    returns[0] = 0.12  # High-return asset
    returns[1] = 0.05  # Low-risk asset
    
    portfolio_fitness = PortfolioOptimizationFunction(returns, risk_free_rate=0.03)
    config_portfolio = GAConfig(
        population_size=50,
        chromosome_length=n_assets,
        num_generations=200,
        mutation_rate=0.05,
        crossover_rate=0.9,
        elitism_rate=0.2,
        adaptive_mutation=True,
        bounds=(0.0, 1.0)
    )
    
    ga_portfolio = AdaptiveGeneticAlgorithm(portfolio_fitness, config_portfolio)
    results_portfolio = ga_portfolio.evolve(verbose=True)
    
    # Normalize weights
    best_weights = np.abs(results_portfolio['best_solution'])
    best_weights = best_weights / np.sum(best_weights)
    
    print(f"\nPortfolio Results:")
    print(f"  Best Sharpe ratio: {results_portfolio['best_fitness']:.6f}")
    print(f"  Optimal weights: {best_weights}")
    print(f"  Expected return: {np.dot(best_weights, returns):.4f}")
    print(f"  Weight concentration: {np.max(best_weights):.4f}")
    
    # Test 3: Feature selection
    print("\n=== Test 3: Feature Selection ===")
    
    # Generate synthetic dataset
    n_samples, n_features = 100, 20
    X = np.random.randn(n_samples, n_features)
    # Make first 5 features informative
    true_weights = np.zeros(n_features)
    true_weights[:5] = np.random.randn(5)
    y = X @ true_weights + np.random.randn(n_samples) * 0.1
    
    feature_fitness = FeatureSelectionFunction(X, y)
    config_features = GAConfig(
        population_size=30,
        chromosome_length=n_features,
        num_generations=100,
        mutation_rate=0.1,
        crossover_rate=0.7,
        elitism_rate=0.15,
        adaptive_mutation=True,
        bounds=(0.0, 1.0)
    )
    
    ga_features = AdaptiveGeneticAlgorithm(feature_fitness, config_features)
    results_features = ga_features.evolve(verbose=True)
    
    selected_features = results_features['best_solution'] > 0.5
    
    print(f"\nFeature Selection Results:")
    print(f"  Best fitness: {results_features['best_fitness']:.6f}")
    print(f"  Selected features: {np.where(selected_features)[0]}")
    print(f"  Number selected: {np.sum(selected_features)}/{n_features}")
    print(f"  True important features: [0, 1, 2, 3, 4]")
    
    # Analyze convergence patterns
    print("\n=== Convergence Analysis ===")
    
    print(f"{'Problem':<20} {'Final Diversity':<15} {'Final Mut Rate':<15} {'Convergence':<12}")
    print("-" * 65)
    
    problems = [
        ("Rastrigin", results_rastrigin),
        ("Portfolio", results_portfolio),
        ("Feature Selection", results_features)
    ]
    
    for name, results in problems:
        final_diversity = results['diversity_history'][-1] if results['diversity_history'] else 0
        final_mut_rate = results['mutation_rate_history'][-1] if results['mutation_rate_history'] else 0
        
        # Check if converged (improvement in last 20% of generations)
        history = results['best_fitness_history']
        if len(history) > 20:
            early_fitness = np.mean(history[:len(history)//5])
            late_fitness = np.mean(history[-len(history)//5:])
            improvement = abs(late_fitness - early_fitness)
            converged = "Yes" if improvement < 0.01 else "No"
        else:
            converged = "Unknown"
        
        print(f"{name:<20} {final_diversity:<15.6f} {final_mut_rate:<15.6f} {converged:<12}")
    
    # Parameter sensitivity analysis
    print("\n=== Parameter Sensitivity Analysis ===")
    
    # Test different mutation rates on Rastrigin
    mutation_rates = [0.01, 0.05, 0.1, 0.2]
    
    print(f"{'Mutation Rate':<15} {'Best Fitness':<15} {'Generations':<12}")
    print("-" * 45)
    
    for mut_rate in mutation_rates:
        config_test = GAConfig(
            population_size=50,
            chromosome_length=5,
            num_generations=100,
            mutation_rate=mut_rate,
            adaptive_mutation=False,
            bounds=(-5.12, 5.12)
        )
        
        ga_test = AdaptiveGeneticAlgorithm(RastriginFunction(5), config_test)
        results_test = ga_test.evolve(verbose=False)
        
        print(f"{mut_rate:<15.3f} {results_test['best_fitness']:<15.6f} {results_test['generations_run']:<12}")
    
    print("\n=== Selection Method Comparison ===")
    
    selection_methods = [
        SelectionMethod.TOURNAMENT,
        SelectionMethod.ROULETTE_WHEEL,
        SelectionMethod.RANK_BASED
    ]
    
    print(f"{'Selection Method':<20} {'Best Fitness':<15} {'Convergence':<12}")
    print("-" * 50)
    
    for method in selection_methods:
        config_test = GAConfig(
            population_size=50,
            chromosome_length=5,
            num_generations=100,
            selection_method=method,
            bounds=(-5.12, 5.12)
        )
        
        ga_test = AdaptiveGeneticAlgorithm(RastriginFunction(5), config_test)
        results_test = ga_test.evolve(verbose=False)
        
        # Check convergence
        history = results_test['best_fitness_history']
        final_improvement = abs(history[-1] - history[max(0, len(history)-20)])
        converged = "Fast" if final_improvement < 0.01 else "Slow"
        
        print(f"{method.value:<20} {results_test['best_fitness']:<15.6f} {converged:<12}")


if __name__ == "__main__":
    genetic_algorithm_comprehensive_example()