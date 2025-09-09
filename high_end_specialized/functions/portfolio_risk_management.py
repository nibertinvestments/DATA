"""
Portfolio Risk Management Functions
==================================

Advanced functions for portfolio risk calculation including Value at Risk (VaR),
Conditional Value at Risk (CVaR), Maximum Drawdown, and modern portfolio
optimization techniques with stress testing.

Mathematical Foundation:
VaR_α = F^(-1)(α) where F is the cumulative distribution function
CVaR_α = E[X | X ≤ VaR_α]
Maximum Drawdown = max(Peak - Trough) / Peak

Applications:
- Risk management
- Portfolio optimization
- Regulatory compliance
- Stress testing
- Capital allocation
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings


@dataclass
class RiskMetrics:
    """Container for portfolio risk metrics."""
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    max_drawdown: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    beta: Optional[float] = None
    tracking_error: Optional[float] = None


class PortfolioRiskManager:
    """
    Comprehensive portfolio risk management system.
    
    Includes:
    - Multiple VaR calculation methods
    - Expected Shortfall (CVaR)
    - Drawdown analysis
    - Risk decomposition
    - Stress testing scenarios
    """
    
    def __init__(self, returns: Union[np.ndarray, pd.DataFrame], 
                 benchmark_returns: Optional[np.ndarray] = None):
        """
        Initialize with portfolio returns data.
        
        Args:
            returns: Portfolio returns (daily or periodic)
            benchmark_returns: Optional benchmark returns for relative metrics
        """
        self.returns = np.array(returns).flatten() if isinstance(returns, (list, np.ndarray)) else returns
        self.benchmark_returns = np.array(benchmark_returns) if benchmark_returns is not None else None
        
        # Calculate basic statistics
        self.mean_return = np.mean(self.returns)
        self.volatility = np.std(self.returns)
        
    def calculate_var(self, confidence_level: float = 0.95, 
                     method: str = "historical") -> float:
        """
        Calculate Value at Risk using different methods.
        
        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
            method: "historical", "parametric", "monte_carlo"
            
        Returns:
            VaR value (positive number representing loss)
        """
        if method == "historical":
            return self._historical_var(confidence_level)
        elif method == "parametric":
            return self._parametric_var(confidence_level)
        elif method == "monte_carlo":
            return self._monte_carlo_var(confidence_level)
        else:
            raise ValueError(f"Unknown VaR method: {method}")
    
    def _historical_var(self, confidence_level: float) -> float:
        """Calculate historical VaR using empirical distribution."""
        alpha = 1 - confidence_level
        return -np.percentile(self.returns, alpha * 100)
    
    def _parametric_var(self, confidence_level: float) -> float:
        """Calculate parametric VaR assuming normal distribution."""
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(alpha)
        return -(self.mean_return + z_score * self.volatility)
    
    def _monte_carlo_var(self, confidence_level: float, 
                        num_simulations: int = 10000) -> float:
        """Calculate Monte Carlo VaR using simulated returns."""
        # Simulate returns using historical parameters
        simulated_returns = np.random.normal(
            self.mean_return, self.volatility, num_simulations
        )
        alpha = 1 - confidence_level
        return -np.percentile(simulated_returns, alpha * 100)
    
    def calculate_cvar(self, confidence_level: float = 0.95, 
                      method: str = "historical") -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).
        
        Args:
            confidence_level: Confidence level
            method: VaR calculation method to use
            
        Returns:
            CVaR value (positive number representing expected loss)
        """
        var = self.calculate_var(confidence_level, method)
        
        if method == "historical":
            # Calculate average of losses beyond VaR
            losses = -self.returns[self.returns < -var]
            return np.mean(losses) if len(losses) > 0 else var
        
        elif method == "parametric":
            # Analytical CVaR for normal distribution
            alpha = 1 - confidence_level
            z_alpha = stats.norm.ppf(alpha)
            phi_z = stats.norm.pdf(z_alpha)
            return -(self.mean_return - self.volatility * phi_z / alpha)
        
        else:
            # For Monte Carlo, use simulated returns
            simulated_returns = np.random.normal(
                self.mean_return, self.volatility, 10000
            )
            losses = -simulated_returns[simulated_returns < -var]
            return np.mean(losses) if len(losses) > 0 else var
    
    def calculate_maximum_drawdown(self) -> Dict[str, float]:
        """
        Calculate maximum drawdown and related metrics.
        
        Returns:
            Dictionary with drawdown metrics
        """
        cumulative_returns = np.cumprod(1 + self.returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        
        max_drawdown = np.min(drawdowns)
        max_dd_index = np.argmin(drawdowns)
        
        # Find peak before max drawdown
        peak_index = np.argmax(running_max[:max_dd_index+1])
        
        # Calculate recovery time
        recovery_index = None
        for i in range(max_dd_index, len(cumulative_returns)):
            if cumulative_returns[i] >= running_max[max_dd_index]:
                recovery_index = i
                break
        
        return {
            "max_drawdown": abs(max_drawdown),
            "drawdown_duration": max_dd_index - peak_index,
            "recovery_time": recovery_index - max_dd_index if recovery_index else None,
            "current_drawdown": abs(drawdowns[-1]),
            "average_drawdown": np.mean(np.abs(drawdowns))
        }
    
    def calculate_risk_metrics(self, risk_free_rate: float = 0.0) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics.
        
        Args:
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            
        Returns:
            RiskMetrics object with all calculated metrics
        """
        # VaR calculations
        var_95 = self.calculate_var(0.95)
        var_99 = self.calculate_var(0.99)
        cvar_95 = self.calculate_cvar(0.95)
        cvar_99 = self.calculate_cvar(0.99)
        
        # Drawdown
        dd_metrics = self.calculate_maximum_drawdown()
        max_drawdown = dd_metrics["max_drawdown"]
        
        # Performance ratios
        excess_return = self.mean_return - risk_free_rate
        sharpe_ratio = excess_return / self.volatility if self.volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = self.returns[self.returns < 0]
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
        sortino_ratio = excess_return / downside_deviation if downside_deviation > 0 else 0
        
        # Calmar ratio
        calmar_ratio = self.mean_return / max_drawdown if max_drawdown > 0 else 0
        
        # Beta and tracking error (if benchmark provided)
        beta = None
        tracking_error = None
        if self.benchmark_returns is not None:
            beta = self._calculate_beta()
            tracking_error = self._calculate_tracking_error()
        
        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            max_drawdown=max_drawdown,
            volatility=self.volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            beta=beta,
            tracking_error=tracking_error
        )
    
    def _calculate_beta(self) -> float:
        """Calculate portfolio beta relative to benchmark."""
        if self.benchmark_returns is None:
            return None
        
        covariance = np.cov(self.returns, self.benchmark_returns)[0, 1]
        benchmark_variance = np.var(self.benchmark_returns)
        
        return covariance / benchmark_variance if benchmark_variance > 0 else 0
    
    def _calculate_tracking_error(self) -> float:
        """Calculate tracking error relative to benchmark."""
        if self.benchmark_returns is None:
            return None
        
        active_returns = self.returns - self.benchmark_returns
        return np.std(active_returns)
    
    def stress_test(self, scenarios: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Perform stress testing under various scenarios.
        
        Args:
            scenarios: Dictionary of scenario name -> {"mean_shock": x, "vol_shock": y}
            
        Returns:
            Dictionary of scenario results
        """
        results = {}
        
        for scenario_name, scenario_params in scenarios.items():
            mean_shock = scenario_params.get("mean_shock", 0)
            vol_shock = scenario_params.get("vol_shock", 1)
            
            # Apply shocks to returns
            stressed_returns = (self.returns + mean_shock) * vol_shock
            
            # Create temporary risk manager for stressed scenario
            temp_manager = PortfolioRiskManager(stressed_returns)
            
            # Calculate metrics under stress
            var_95_stress = temp_manager.calculate_var(0.95)
            cvar_95_stress = temp_manager.calculate_cvar(0.95)
            max_dd_stress = temp_manager.calculate_maximum_drawdown()["max_drawdown"]
            
            results[scenario_name] = {
                "var_95": var_95_stress,
                "cvar_95": cvar_95_stress,
                "max_drawdown": max_dd_stress,
                "volatility": temp_manager.volatility,
                "mean_return": temp_manager.mean_return
            }
        
        return results
    
    def risk_decomposition(self, weights: np.ndarray, 
                          asset_returns: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Decompose portfolio risk into asset contributions.
        
        Args:
            weights: Portfolio weights
            asset_returns: Individual asset returns matrix [time x assets]
            
        Returns:
            Dictionary with risk decomposition
        """
        # Calculate covariance matrix
        cov_matrix = np.cov(asset_returns.T)
        
        # Portfolio variance
        portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_vol = np.sqrt(portfolio_var)
        
        # Marginal contribution to risk (MCTR)
        mctr = np.dot(cov_matrix, weights) / portfolio_vol
        
        # Component contribution to risk (CCTR)
        cctr = weights * mctr
        
        # Percentage contribution
        pctr = cctr / portfolio_vol
        
        return {
            "marginal_contribution": mctr,
            "component_contribution": cctr,
            "percentage_contribution": pctr,
            "portfolio_volatility": portfolio_vol
        }
    
    def optimize_risk_parity(self, asset_returns: np.ndarray, 
                           target_vol: float = 0.15) -> Dict[str, Union[np.ndarray, float]]:
        """
        Optimize portfolio for risk parity.
        
        Args:
            asset_returns: Individual asset returns matrix
            target_vol: Target portfolio volatility
            
        Returns:
            Optimal weights and metrics
        """
        n_assets = asset_returns.shape[1]
        cov_matrix = np.cov(asset_returns.T)
        
        def risk_parity_objective(weights):
            """Objective function for risk parity optimization."""
            weights = weights / np.sum(weights)  # Normalize weights
            
            # Calculate risk contributions
            portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))
            mctr = np.dot(cov_matrix, weights) / np.sqrt(portfolio_var)
            cctr = weights * mctr
            
            # Target is equal risk contribution
            target_risk = np.sqrt(portfolio_var) / n_assets
            risk_diff = cctr - target_risk
            
            return np.sum(risk_diff ** 2)
        
        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},  # Weights sum to 1
        ]
        
        # Bounds (no short selling)
        bounds = [(0.01, 0.4) for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            risk_parity_objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )
        
        optimal_weights = result.x / np.sum(result.x)
        
        # Scale to target volatility
        portfolio_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
        scale_factor = target_vol / portfolio_vol
        
        return {
            "weights": optimal_weights,
            "target_volatility": target_vol,
            "actual_volatility": portfolio_vol,
            "scale_factor": scale_factor,
            "optimization_success": result.success
        }


def comprehensive_risk_example():
    """Comprehensive example demonstrating portfolio risk management."""
    print("=== Portfolio Risk Management Example ===")
    
    # Generate synthetic portfolio returns
    np.random.seed(42)
    n_periods = 1000
    
    # Create correlated asset returns
    mean_returns = np.array([0.0008, 0.0006, 0.0010, 0.0004])  # Daily returns
    volatilities = np.array([0.015, 0.012, 0.020, 0.008])
    
    # Correlation matrix
    correlation = np.array([
        [1.0, 0.6, 0.3, 0.1],
        [0.6, 1.0, 0.4, 0.2],
        [0.3, 0.4, 1.0, 0.2],
        [0.1, 0.2, 0.2, 1.0]
    ])
    
    # Generate correlated returns
    cov_matrix = np.outer(volatilities, volatilities) * correlation
    asset_returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_periods)
    
    # Portfolio weights
    weights = np.array([0.4, 0.3, 0.2, 0.1])
    portfolio_returns = np.dot(asset_returns, weights)
    
    # Benchmark returns (market index)
    benchmark_returns = np.random.normal(0.0006, 0.012, n_periods)
    
    print(f"Portfolio Setup:")
    print(f"  Number of assets: {len(weights)}")
    print(f"  Portfolio weights: {weights}")
    print(f"  Time periods: {n_periods}")
    print(f"  Average daily return: {np.mean(portfolio_returns):.6f}")
    print(f"  Daily volatility: {np.std(portfolio_returns):.6f}")
    print()
    
    # Initialize risk manager
    risk_manager = PortfolioRiskManager(portfolio_returns, benchmark_returns)
    
    # Calculate comprehensive risk metrics
    risk_metrics = risk_manager.calculate_risk_metrics(risk_free_rate=0.0002)
    
    print("=== Risk Metrics ===")
    print(f"VaR (95%):           {risk_metrics.var_95:.4f}")
    print(f"VaR (99%):           {risk_metrics.var_99:.4f}")
    print(f"CVaR (95%):          {risk_metrics.cvar_95:.4f}")
    print(f"CVaR (99%):          {risk_metrics.cvar_99:.4f}")
    print(f"Maximum Drawdown:    {risk_metrics.max_drawdown:.4f}")
    print(f"Volatility:          {risk_metrics.volatility:.4f}")
    print(f"Sharpe Ratio:        {risk_metrics.sharpe_ratio:.4f}")
    print(f"Sortino Ratio:       {risk_metrics.sortino_ratio:.4f}")
    print(f"Calmar Ratio:        {risk_metrics.calmar_ratio:.4f}")
    print(f"Beta:                {risk_metrics.beta:.4f}")
    print(f"Tracking Error:      {risk_metrics.tracking_error:.4f}")
    print()
    
    # VaR method comparison
    print("=== VaR Method Comparison ===")
    methods = ["historical", "parametric", "monte_carlo"]
    confidence_levels = [0.95, 0.99]
    
    print(f"{'Method':<15} {'95% VaR':<12} {'99% VaR':<12}")
    print("-" * 40)
    
    for method in methods:
        var_95 = risk_manager.calculate_var(0.95, method)
        var_99 = risk_manager.calculate_var(0.99, method)
        print(f"{method:<15} {var_95:<12.6f} {var_99:<12.6f}")
    
    print()
    
    # Stress testing
    print("=== Stress Testing ===")
    
    stress_scenarios = {
        "Market Crash": {"mean_shock": -0.02, "vol_shock": 2.0},
        "Volatility Spike": {"mean_shock": 0.0, "vol_shock": 1.5},
        "Deflationary": {"mean_shock": -0.005, "vol_shock": 0.8},
        "High Inflation": {"mean_shock": 0.002, "vol_shock": 1.2}
    }
    
    stress_results = risk_manager.stress_test(stress_scenarios)
    
    print(f"{'Scenario':<15} {'VaR (95%)':<12} {'CVaR (95%)':<12} {'Max DD':<12}")
    print("-" * 60)
    
    for scenario, results in stress_results.items():
        print(f"{scenario:<15} {results['var_95']:<12.6f} {results['cvar_95']:<12.6f} {results['max_drawdown']:<12.6f}")
    
    print()
    
    # Risk decomposition
    print("=== Risk Decomposition ===")
    
    decomposition = risk_manager.risk_decomposition(weights, asset_returns)
    
    print(f"{'Asset':<8} {'Weight':<10} {'MCTR':<12} {'CCTR':<12} {'% Contrib':<12}")
    print("-" * 60)
    
    for i in range(len(weights)):
        print(f"Asset {i+1:<3} {weights[i]:<10.3f} {decomposition['marginal_contribution'][i]:<12.6f} "
              f"{decomposition['component_contribution'][i]:<12.6f} {decomposition['percentage_contribution'][i]:<12.6f}")
    
    print(f"\nPortfolio Volatility: {decomposition['portfolio_volatility']:.6f}")
    print()
    
    # Risk parity optimization
    print("=== Risk Parity Optimization ===")
    
    rp_result = risk_manager.optimize_risk_parity(asset_returns, target_vol=0.12)
    
    print(f"Optimization Success: {rp_result['optimization_success']}")
    print(f"Target Volatility: {rp_result['target_volatility']:.4f}")
    print(f"Actual Volatility: {rp_result['actual_volatility']:.4f}")
    print()
    
    print(f"{'Asset':<8} {'Original':<12} {'Risk Parity':<12} {'Change':<12}")
    print("-" * 50)
    
    for i in range(len(weights)):
        change = rp_result['weights'][i] - weights[i]
        print(f"Asset {i+1:<3} {weights[i]:<12.3f} {rp_result['weights'][i]:<12.3f} {change:+12.3f}")
    
    print()
    
    # Drawdown analysis
    print("=== Drawdown Analysis ===")
    
    dd_metrics = risk_manager.calculate_maximum_drawdown()
    
    print(f"Maximum Drawdown:    {dd_metrics['max_drawdown']:.4f}")
    print(f"Drawdown Duration:   {dd_metrics['drawdown_duration']} periods")
    print(f"Recovery Time:       {dd_metrics['recovery_time']} periods" if dd_metrics['recovery_time'] else "Not recovered")
    print(f"Current Drawdown:    {dd_metrics['current_drawdown']:.4f}")
    print(f"Average Drawdown:    {dd_metrics['average_drawdown']:.4f}")


if __name__ == "__main__":
    comprehensive_risk_example()