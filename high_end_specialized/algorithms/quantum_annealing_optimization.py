"""
Quantum Annealing Optimization Algorithm
=======================================

A sophisticated optimization algorithm inspired by quantum annealing processes.
Useful for solving combinatorial optimization problems, portfolio optimization,
and machine learning hyperparameter tuning.

Mathematical Foundation:
The quantum annealing process is modeled by the time-dependent Hamiltonian:
H(t) = A(t)H_0 + B(t)H_1

Where:
- H_0: Initial Hamiltonian (kinetic energy)
- H_1: Final Hamiltonian (potential energy)
- A(t), B(t): Time-dependent annealing schedules

Applications:
- Portfolio optimization
- Feature selection in ML
- QUBO (Quadratic Unconstrained Binary Optimization) problems
- Traveling salesman problem
- Graph coloring
"""

import numpy as np
import random
from typing import List, Tuple, Callable, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class AnnealingSchedule:
    """Defines the annealing schedule parameters."""
    initial_temp: float = 1000.0
    final_temp: float = 0.01
    cooling_rate: float = 0.95
    max_iterations: int = 10000


class CostFunction(ABC):
    """Abstract base class for cost functions."""
    
    @abstractmethod
    def evaluate(self, solution: np.ndarray) -> float:
        """Evaluate the cost of a given solution."""
        pass
    
    @abstractmethod
    def get_neighbor(self, solution: np.ndarray) -> np.ndarray:
        """Generate a neighboring solution."""
        pass


class PortfolioOptimizationCost(CostFunction):
    """Cost function for portfolio optimization using Modern Portfolio Theory."""
    
    def __init__(self, returns: np.ndarray, risk_aversion: float = 1.0):
        self.returns = returns  # Expected returns for each asset
        self.cov_matrix = np.cov(returns.T)  # Covariance matrix
        self.risk_aversion = risk_aversion
        self.n_assets = returns.shape[1]  # Number of assets
    
    def evaluate(self, weights: np.ndarray) -> float:
        """
        Evaluate portfolio using mean-variance optimization.
        Cost = -Expected Return + Risk Aversion * Variance
        """
        # Normalize weights to sum to 1
        weights = weights / np.sum(weights)
        
        # Calculate expected returns
        expected_returns = np.mean(self.returns, axis=0)  # Average return per asset
        expected_return = np.dot(weights, expected_returns)
        portfolio_variance = np.dot(weights.T, np.dot(self.cov_matrix, weights))
        
        # Minimize negative Sharpe ratio equivalent
        cost = -expected_return + self.risk_aversion * portfolio_variance
        
        # Penalty for constraint violations (weights must be non-negative and sum to 1)
        penalty = 0
        if np.any(weights < 0):
            penalty += 1000 * np.sum(np.abs(weights[weights < 0]))
        
        return cost + penalty
    
    def get_neighbor(self, weights: np.ndarray) -> np.ndarray:
        """Generate neighboring portfolio by small random perturbation."""
        new_weights = weights.copy()
        
        # Randomly select two assets to adjust
        i, j = random.sample(range(self.n_assets), 2)
        
        # Transfer small amount from asset i to asset j
        transfer_amount = random.uniform(-0.05, 0.05) * weights[i]
        new_weights[i] -= transfer_amount
        new_weights[j] += transfer_amount
        
        # Ensure non-negative weights
        new_weights = np.maximum(new_weights, 0.0)
        
        # Normalize to sum to 1
        if np.sum(new_weights) > 0:
            new_weights = new_weights / np.sum(new_weights)
        else:
            new_weights = np.ones(self.n_assets) / self.n_assets
            
        return new_weights


class TSPCostFunction(CostFunction):
    """Cost function for Traveling Salesman Problem."""
    
    def __init__(self, distance_matrix: np.ndarray):
        self.distance_matrix = distance_matrix
        self.n_cities = len(distance_matrix)
    
    def evaluate(self, tour: np.ndarray) -> float:
        """Calculate total tour distance."""
        total_distance = 0
        for i in range(len(tour)):
            from_city = int(tour[i])
            to_city = int(tour[(i + 1) % len(tour)])
            total_distance += self.distance_matrix[from_city][to_city]
        return total_distance
    
    def get_neighbor(self, tour: np.ndarray) -> np.ndarray:
        """Generate neighbor using 2-opt swap."""
        new_tour = tour.copy()
        i, j = sorted(random.sample(range(len(tour)), 2))
        new_tour[i:j+1] = new_tour[i:j+1][::-1]  # Reverse the segment
        return new_tour


class QuantumAnnealingOptimizer:
    """
    Quantum-inspired annealing optimizer for combinatorial optimization.
    
    Uses simulated annealing with quantum tunneling effects to escape local minima.
    """
    
    def __init__(self, cost_function: CostFunction, schedule: AnnealingSchedule):
        self.cost_function = cost_function
        self.schedule = schedule
        self.best_solution = None
        self.best_cost = float('inf')
        self.cost_history = []
        self.temperature_history = []
    
    def optimize(self, initial_solution: np.ndarray, verbose: bool = False) -> Tuple[np.ndarray, float]:
        """
        Perform quantum annealing optimization.
        
        Args:
            initial_solution: Starting solution
            verbose: Whether to print progress
            
        Returns:
            Tuple of (best_solution, best_cost)
        """
        current_solution = initial_solution.copy()
        current_cost = self.cost_function.evaluate(current_solution)
        
        self.best_solution = current_solution.copy()
        self.best_cost = current_cost
        
        temperature = self.schedule.initial_temp
        
        for iteration in range(self.schedule.max_iterations):
            # Generate neighboring solution
            candidate_solution = self.cost_function.get_neighbor(current_solution)
            candidate_cost = self.cost_function.evaluate(candidate_solution)
            
            # Calculate acceptance probability with quantum tunneling effect
            delta_cost = candidate_cost - current_cost
            
            # Quantum tunneling probability (allows escaping local minima)
            tunneling_prob = self._quantum_tunneling_probability(
                delta_cost, temperature, iteration
            )
            
            # Standard Boltzmann acceptance probability
            if delta_cost < 0:
                acceptance_prob = 1.0
            else:
                acceptance_prob = np.exp(-delta_cost / temperature)
            
            # Combined acceptance with quantum effects
            total_acceptance = min(1.0, acceptance_prob + tunneling_prob)
            
            # Accept or reject the candidate
            if random.random() < total_acceptance:
                current_solution = candidate_solution
                current_cost = candidate_cost
                
                # Update best solution if improved
                if current_cost < self.best_cost:
                    self.best_solution = current_solution.copy()
                    self.best_cost = current_cost
            
            # Cool down temperature
            temperature *= self.schedule.cooling_rate
            temperature = max(temperature, self.schedule.final_temp)
            
            # Record history
            self.cost_history.append(current_cost)
            self.temperature_history.append(temperature)
            
            if verbose and iteration % 1000 == 0:
                print(f"Iteration {iteration}: Cost = {current_cost:.6f}, "
                      f"Best = {self.best_cost:.6f}, Temp = {temperature:.6f}")
        
        return self.best_solution, self.best_cost
    
    def _quantum_tunneling_probability(self, delta_cost: float, temperature: float, 
                                     iteration: int) -> float:
        """
        Calculate quantum tunneling probability.
        
        Allows the system to tunnel through energy barriers,
        especially effective early in the optimization process.
        """
        if delta_cost <= 0:
            return 0.0
        
        # Tunneling strength decreases over time
        tunneling_strength = 0.1 * np.exp(-iteration / (self.schedule.max_iterations * 0.3))
        
        # Quantum tunneling probability based on barrier height and temperature
        tunneling_prob = tunneling_strength * np.exp(-delta_cost / (temperature + 1e-8))
        
        return min(tunneling_prob, 0.1)  # Cap at 10%


def portfolio_optimization_example():
    """Example: Portfolio optimization using quantum annealing."""
    print("=== Portfolio Optimization Example ===")
    
    # Generate synthetic asset returns data
    np.random.seed(42)
    n_assets = 5
    n_periods = 252  # One year of daily returns
    
    # Generate correlated returns
    base_returns = np.random.normal(0.0008, 0.02, (n_periods, n_assets))
    correlation_matrix = np.array([
        [1.0, 0.3, 0.1, 0.2, 0.0],
        [0.3, 1.0, 0.4, 0.1, 0.2],
        [0.1, 0.4, 1.0, 0.2, 0.3],
        [0.2, 0.1, 0.2, 1.0, 0.1],
        [0.0, 0.2, 0.3, 0.1, 1.0]
    ])
    
    # Apply correlation structure
    L = np.linalg.cholesky(correlation_matrix)
    correlated_returns = base_returns @ L.T
    
    # Calculate expected returns (annualized)
    expected_returns = np.mean(correlated_returns, axis=0) * 252
    
    print(f"Expected annual returns: {expected_returns}")
    print(f"Expected return range: {expected_returns.min():.4f} to {expected_returns.max():.4f}")
    
    # Create cost function and optimizer
    cost_function = PortfolioOptimizationCost(correlated_returns, risk_aversion=1.0)
    schedule = AnnealingSchedule(
        initial_temp=10.0,
        final_temp=0.001,
        cooling_rate=0.999,
        max_iterations=5000
    )
    
    optimizer = QuantumAnnealingOptimizer(cost_function, schedule)
    
    # Initial equal-weight portfolio
    initial_weights = np.ones(n_assets) / n_assets
    
    # Optimize
    optimal_weights, optimal_cost = optimizer.optimize(initial_weights, verbose=True)
    
    print(f"\nOptimal portfolio weights: {optimal_weights}")
    print(f"Optimal cost: {optimal_cost:.6f}")
    
    # Calculate portfolio metrics
    expected_returns = np.mean(correlated_returns, axis=0)  # Daily expected returns
    portfolio_return = np.dot(optimal_weights, expected_returns) * 252  # Annualized
    portfolio_variance = np.dot(optimal_weights.T, 
                               np.dot(cost_function.cov_matrix * 252, optimal_weights))
    portfolio_volatility = np.sqrt(portfolio_variance)
    sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
    
    print(f"Expected annual return: {portfolio_return:.4f}")
    print(f"Annual volatility: {portfolio_volatility:.4f}")
    print(f"Sharpe ratio: {sharpe_ratio:.4f}")


def tsp_optimization_example():
    """Example: Traveling Salesman Problem using quantum annealing."""
    print("\n=== Traveling Salesman Problem Example ===")
    
    # Generate random cities
    np.random.seed(42)
    n_cities = 10
    cities = np.random.rand(n_cities, 2) * 100  # Random cities in 100x100 grid
    
    # Calculate distance matrix
    distance_matrix = np.zeros((n_cities, n_cities))
    for i in range(n_cities):
        for j in range(n_cities):
            if i != j:
                distance_matrix[i][j] = np.sqrt(
                    (cities[i][0] - cities[j][0])**2 + 
                    (cities[i][1] - cities[j][1])**2
                )
    
    # Create cost function and optimizer
    cost_function = TSPCostFunction(distance_matrix)
    schedule = AnnealingSchedule(
        initial_temp=100.0,
        final_temp=0.01,
        cooling_rate=0.995,
        max_iterations=10000
    )
    
    optimizer = QuantumAnnealingOptimizer(cost_function, schedule)
    
    # Initial random tour
    initial_tour = np.arange(n_cities)
    np.random.shuffle(initial_tour)
    
    # Optimize
    optimal_tour, optimal_distance = optimizer.optimize(initial_tour, verbose=True)
    
    print(f"\nOptimal tour: {optimal_tour}")
    print(f"Optimal distance: {optimal_distance:.2f}")
    print(f"Initial distance: {cost_function.evaluate(initial_tour):.2f}")
    print(f"Improvement: {((cost_function.evaluate(initial_tour) - optimal_distance) / 
                         cost_function.evaluate(initial_tour) * 100):.1f}%")


if __name__ == "__main__":
    portfolio_optimization_example()
    tsp_optimization_example()