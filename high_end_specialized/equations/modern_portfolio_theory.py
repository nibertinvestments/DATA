"""
Modern Portfolio Theory Equations
=================================

Complete implementation of Harry Markowitz's Modern Portfolio Theory
including efficient frontier calculation, risk-return optimization,
and advanced portfolio construction techniques.

Mathematical Foundation:
Portfolio Return: E(Rp) = Σ wi * E(Ri)
Portfolio Variance: σp² = Σ Σ wi * wj * σij
Sharpe Ratio: SR = (E(Rp) - Rf) / σp
Efficient Frontier: Minimize σp² subject to E(Rp) = μ and Σ wi = 1

Applications:
- Asset allocation
- Risk management
- Investment strategy
- Wealth management
- Institutional investing
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, linprog
from scipy.linalg import inv, LinAlgError
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics."""
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    var_95: float
    max_drawdown: float
    beta: Optional[float] = None
    alpha: Optional[float] = None


@dataclass
class OptimizationConstraints:
    """Portfolio optimization constraints."""
    min_weights: Optional[np.ndarray] = None
    max_weights: Optional[np.ndarray] = None
    target_return: Optional[float] = None
    max_risk: Optional[float] = None
    sector_constraints: Optional[Dict[str, Tuple[float, float]]] = None
    turnover_limit: Optional[float] = None


class ModernPortfolioTheory:
    """
    Comprehensive implementation of Modern Portfolio Theory.
    
    Includes:
    - Mean-variance optimization
    - Efficient frontier computation
    - Risk budgeting
    - Black-Litterman model
    - Factor models integration
    """
    
    def __init__(self, returns: np.ndarray, asset_names: Optional[List[str]] = None):
        """
        Initialize with historical returns data.
        
        Args:
            returns: Asset returns matrix [time x assets]
            asset_names: Optional names for assets
        """
        self.returns = np.array(returns)
        self.n_periods, self.n_assets = self.returns.shape
        self.asset_names = asset_names or [f"Asset_{i}" for i in range(self.n_assets)]
        
        # Calculate statistics
        self.mean_returns = np.mean(self.returns, axis=0)
        self.cov_matrix = np.cov(self.returns.T)
        self.corr_matrix = np.corrcoef(self.returns.T)
        
        # Risk-free rate (can be updated)
        self.risk_free_rate = 0.02
    
    def set_risk_free_rate(self, rate: float):
        """Set risk-free rate for Sharpe ratio calculations."""
        self.risk_free_rate = rate
    
    def portfolio_metrics(self, weights: np.ndarray) -> PortfolioMetrics:
        """
        Calculate comprehensive portfolio metrics.
        
        Args:
            weights: Portfolio weights
            
        Returns:
            PortfolioMetrics object
        """
        weights = np.array(weights)
        
        # Expected return and risk
        expected_return = np.dot(weights, self.mean_returns)
        portfolio_variance = np.dot(weights.T, np.dot(self.cov_matrix, weights))
        volatility = np.sqrt(portfolio_variance)
        
        # Sharpe ratio
        excess_return = expected_return - self.risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0
        
        # VaR (95% confidence)
        portfolio_returns = self.returns @ weights
        var_95 = -np.percentile(portfolio_returns, 5)
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        return PortfolioMetrics(
            weights=weights,
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            var_95=var_95,
            max_drawdown=-max_drawdown
        )
    
    def minimum_variance_portfolio(self, constraints: Optional[OptimizationConstraints] = None) -> PortfolioMetrics:
        """
        Find the minimum variance portfolio.
        
        Args:
            constraints: Optional optimization constraints
            
        Returns:
            Minimum variance portfolio metrics
        """
        def objective(weights):
            return np.dot(weights.T, np.dot(self.cov_matrix, weights))
        
        # Setup constraints
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]  # Weights sum to 1
        
        if constraints and constraints.target_return is not None:
            cons.append({
                'type': 'eq',
                'fun': lambda w: np.dot(w, self.mean_returns) - constraints.target_return
            })
        
        # Bounds
        bounds = self._get_bounds(constraints)
        
        # Initial guess
        x0 = np.ones(self.n_assets) / self.n_assets
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
        
        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")
        
        return self.portfolio_metrics(result.x)
    
    def maximum_sharpe_portfolio(self, constraints: Optional[OptimizationConstraints] = None) -> PortfolioMetrics:
        """
        Find the maximum Sharpe ratio portfolio.
        
        Args:
            constraints: Optional optimization constraints
            
        Returns:
            Maximum Sharpe ratio portfolio metrics
        """
        def negative_sharpe(weights):
            expected_return = np.dot(weights, self.mean_returns)
            portfolio_variance = np.dot(weights.T, np.dot(self.cov_matrix, weights))
            volatility = np.sqrt(portfolio_variance)
            
            if volatility == 0:
                return -1000  # Penalty for zero volatility
            
            sharpe = (expected_return - self.risk_free_rate) / volatility
            return -sharpe  # Minimize negative Sharpe
        
        # Setup constraints
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        # Bounds
        bounds = self._get_bounds(constraints)
        
        # Initial guess
        x0 = np.ones(self.n_assets) / self.n_assets
        
        # Optimize
        result = minimize(negative_sharpe, x0, method='SLSQP', bounds=bounds, constraints=cons)
        
        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")
        
        return self.portfolio_metrics(result.x)
    
    def efficient_frontier(self, n_portfolios: int = 100, 
                          return_range: Optional[Tuple[float, float]] = None) -> List[PortfolioMetrics]:
        """
        Calculate the efficient frontier.
        
        Args:
            n_portfolios: Number of portfolios on the frontier
            return_range: Optional return range, defaults to min/max possible
            
        Returns:
            List of efficient portfolios
        """
        # Determine return range
        if return_range is None:
            min_return = np.min(self.mean_returns)
            max_return = np.max(self.mean_returns)
        else:
            min_return, max_return = return_range
        
        target_returns = np.linspace(min_return, max_return, n_portfolios)
        efficient_portfolios = []
        
        for target_return in target_returns:
            try:
                constraints = OptimizationConstraints(target_return=target_return)
                portfolio = self.minimum_variance_portfolio(constraints)
                efficient_portfolios.append(portfolio)
            except Exception:
                continue  # Skip if optimization fails
        
        return efficient_portfolios
    
    def _get_bounds(self, constraints: Optional[OptimizationConstraints] = None) -> List[Tuple[float, float]]:
        """Get optimization bounds from constraints."""
        if constraints is None:
            return [(0, 1) for _ in range(self.n_assets)]
        
        bounds = []
        for i in range(self.n_assets):
            min_weight = constraints.min_weights[i] if constraints.min_weights is not None else 0
            max_weight = constraints.max_weights[i] if constraints.max_weights is not None else 1
            bounds.append((min_weight, max_weight))
        
        return bounds
    
    def risk_parity_portfolio(self) -> PortfolioMetrics:
        """
        Calculate risk parity portfolio (equal risk contribution).
        
        Returns:
            Risk parity portfolio metrics
        """
        def risk_parity_objective(weights):
            """Objective function for risk parity optimization."""
            # Portfolio variance
            portfolio_var = np.dot(weights.T, np.dot(self.cov_matrix, weights))
            
            # Marginal contribution to risk
            mctr = np.dot(self.cov_matrix, weights) / np.sqrt(portfolio_var)
            
            # Component contribution to risk
            cctr = weights * mctr
            
            # Target is equal risk contribution
            target_risk = np.sqrt(portfolio_var) / self.n_assets
            
            # Minimize sum of squared deviations from target
            return np.sum((cctr - target_risk) ** 2)
        
        # Constraints
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0.01, 0.5) for _ in range(self.n_assets)]  # Prevent extreme weights
        
        # Initial guess
        x0 = np.ones(self.n_assets) / self.n_assets
        
        # Optimize
        result = minimize(risk_parity_objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
        
        if not result.success:
            raise ValueError(f"Risk parity optimization failed: {result.message}")
        
        return self.portfolio_metrics(result.x)
    
    def black_litterman_portfolio(self, views: Dict[int, float], 
                                view_uncertainties: Dict[int, float],
                                tau: float = 0.025) -> PortfolioMetrics:
        """
        Implement Black-Litterman model for portfolio optimization.
        
        Args:
            views: Dictionary of asset indices to expected returns
            view_uncertainties: Dictionary of asset indices to view uncertainties
            tau: Scalar indicating uncertainty of prior estimate
            
        Returns:
            Black-Litterman optimal portfolio
        """
        # Market capitalization weights (using equal weights as proxy)
        w_market = np.ones(self.n_assets) / self.n_assets
        
        # Implied equilibrium returns
        risk_aversion = 3.0  # Typical risk aversion parameter
        pi = risk_aversion * np.dot(self.cov_matrix, w_market)
        
        # Views matrix P and view vector Q
        n_views = len(views)
        P = np.zeros((n_views, self.n_assets))
        Q = np.zeros(n_views)
        
        for i, (asset_idx, view_return) in enumerate(views.items()):
            P[i, asset_idx] = 1
            Q[i] = view_return
        
        # View uncertainty matrix Omega
        Omega = np.diag([view_uncertainties[asset_idx] for asset_idx in views.keys()])
        
        # Black-Litterman calculation
        tau_cov = tau * self.cov_matrix
        
        try:
            # New expected returns
            M1 = inv(tau_cov)
            M2 = np.dot(P.T, np.dot(inv(Omega), P))
            M3 = np.dot(inv(tau_cov), pi) + np.dot(P.T, np.dot(inv(Omega), Q))
            
            mu_bl = np.dot(inv(M1 + M2), M3)
            
            # New covariance matrix
            cov_bl = inv(M1 + M2)
            
            # Optimal portfolio weights
            weights = np.dot(inv(risk_aversion * cov_bl), mu_bl)
            weights = weights / np.sum(weights)  # Normalize
            
            return self.portfolio_metrics(weights)
            
        except LinAlgError:
            raise ValueError("Black-Litterman calculation failed due to singular matrix")
    
    def factor_model_optimization(self, factor_loadings: np.ndarray, 
                                factor_returns: np.ndarray) -> PortfolioMetrics:
        """
        Optimize portfolio using factor model.
        
        Args:
            factor_loadings: Asset loadings on factors [assets x factors]
            factor_returns: Historical factor returns [time x factors]
            
        Returns:
            Factor model optimized portfolio
        """
        # Factor covariance matrix
        factor_cov = np.cov(factor_returns.T)
        
        # Idiosyncratic variances
        residuals = self.returns - np.dot(factor_returns, factor_loadings.T)
        idiosyncratic_var = np.var(residuals, axis=0)
        
        # Portfolio risk decomposition
        def portfolio_variance_factor_model(weights):
            # Factor risk
            factor_exposure = np.dot(factor_loadings.T, weights)
            factor_risk = np.dot(factor_exposure.T, np.dot(factor_cov, factor_exposure))
            
            # Idiosyncratic risk
            idiosyncratic_risk = np.dot(weights**2, idiosyncratic_var)
            
            return factor_risk + idiosyncratic_risk
        
        def negative_sharpe_factor(weights):
            expected_return = np.dot(weights, self.mean_returns)
            portfolio_var = portfolio_variance_factor_model(weights)
            volatility = np.sqrt(portfolio_var)
            
            if volatility == 0:
                return -1000
            
            sharpe = (expected_return - self.risk_free_rate) / volatility
            return -sharpe
        
        # Optimize
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0, 1) for _ in range(self.n_assets)]
        x0 = np.ones(self.n_assets) / self.n_assets
        
        result = minimize(negative_sharpe_factor, x0, method='SLSQP', bounds=bounds, constraints=cons)
        
        if not result.success:
            raise ValueError(f"Factor model optimization failed: {result.message}")
        
        return self.portfolio_metrics(result.x)
    
    def portfolio_analytics(self, weights: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
        """
        Comprehensive portfolio analytics.
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Dictionary with detailed analytics
        """
        metrics = self.portfolio_metrics(weights)
        
        # Risk decomposition
        portfolio_var = np.dot(weights.T, np.dot(self.cov_matrix, weights))
        mctr = np.dot(self.cov_matrix, weights) / np.sqrt(portfolio_var)  # Marginal contribution
        cctr = weights * mctr  # Component contribution
        pctr = cctr / np.sqrt(portfolio_var)  # Percentage contribution
        
        # Correlation analysis
        avg_correlation = np.mean(self.corr_matrix[np.triu_indices_from(self.corr_matrix, k=1)])
        
        # Diversification ratio
        weighted_vol = np.dot(weights, np.sqrt(np.diag(self.cov_matrix)))
        diversification_ratio = weighted_vol / metrics.volatility
        
        return {
            'expected_return': metrics.expected_return,
            'volatility': metrics.volatility,
            'sharpe_ratio': metrics.sharpe_ratio,
            'var_95': metrics.var_95,
            'max_drawdown': metrics.max_drawdown,
            'marginal_contribution': mctr,
            'component_contribution': cctr,
            'percentage_contribution': pctr,
            'average_correlation': avg_correlation,
            'diversification_ratio': diversification_ratio,
            'effective_number_assets': 1 / np.sum(weights**2)
        }


def comprehensive_mpt_example():
    """Comprehensive example demonstrating Modern Portfolio Theory."""
    print("=== Modern Portfolio Theory Example ===")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate synthetic returns data
    n_assets = 8
    n_periods = 252 * 3  # 3 years of daily data
    
    # Create correlation structure
    base_correlation = 0.3
    correlation_matrix = np.full((n_assets, n_assets), base_correlation)
    np.fill_diagonal(correlation_matrix, 1.0)
    
    # Add some structure (sectors)
    correlation_matrix[0:3, 0:3] = 0.7  # Tech sector
    correlation_matrix[3:6, 3:6] = 0.6  # Finance sector
    
    # Generate returns
    volatilities = np.array([0.15, 0.18, 0.20, 0.12, 0.14, 0.16, 0.22, 0.10])
    mean_returns = np.array([0.08, 0.10, 0.12, 0.06, 0.07, 0.09, 0.14, 0.04])
    
    # Convert to daily
    daily_vol = volatilities / np.sqrt(252)
    daily_mean = mean_returns / 252
    
    # Generate correlated returns
    L = np.linalg.cholesky(correlation_matrix)
    random_returns = np.random.normal(0, 1, (n_periods, n_assets))
    correlated_returns = random_returns @ L.T
    
    # Scale by volatilities and add means
    returns = correlated_returns * daily_vol + daily_mean
    
    asset_names = ['TECH_A', 'TECH_B', 'TECH_C', 'FIN_A', 'FIN_B', 'FIN_C', 'GROWTH', 'BOND']
    
    print(f"Generated {n_periods} periods of returns for {n_assets} assets")
    print(f"Annualized statistics:")
    print(f"  Mean returns: {mean_returns}")
    print(f"  Volatilities: {volatilities}")
    print()
    
    # Initialize MPT
    mpt = ModernPortfolioTheory(returns, asset_names)
    mpt.set_risk_free_rate(0.02)
    
    # Equal weight portfolio
    equal_weights = np.ones(n_assets) / n_assets
    equal_portfolio = mpt.portfolio_metrics(equal_weights)
    
    print("=== Equal Weight Portfolio ===")
    print(f"Expected Return: {equal_portfolio.expected_return:.4f}")
    print(f"Volatility: {equal_portfolio.volatility:.4f}")
    print(f"Sharpe Ratio: {equal_portfolio.sharpe_ratio:.4f}")
    print()
    
    # Minimum variance portfolio
    print("=== Minimum Variance Portfolio ===")
    min_var_portfolio = mpt.minimum_variance_portfolio()
    
    print(f"Expected Return: {min_var_portfolio.expected_return:.4f}")
    print(f"Volatility: {min_var_portfolio.volatility:.4f}")
    print(f"Sharpe Ratio: {min_var_portfolio.sharpe_ratio:.4f}")
    print(f"Weights: {min_var_portfolio.weights}")
    print()
    
    # Maximum Sharpe portfolio
    print("=== Maximum Sharpe Ratio Portfolio ===")
    max_sharpe_portfolio = mpt.maximum_sharpe_portfolio()
    
    print(f"Expected Return: {max_sharpe_portfolio.expected_return:.4f}")
    print(f"Volatility: {max_sharpe_portfolio.volatility:.4f}")
    print(f"Sharpe Ratio: {max_sharpe_portfolio.sharpe_ratio:.4f}")
    print(f"Weights: {max_sharpe_portfolio.weights}")
    print()
    
    # Risk parity portfolio
    print("=== Risk Parity Portfolio ===")
    risk_parity_portfolio = mpt.risk_parity_portfolio()
    
    print(f"Expected Return: {risk_parity_portfolio.expected_return:.4f}")
    print(f"Volatility: {risk_parity_portfolio.volatility:.4f}")
    print(f"Sharpe Ratio: {risk_parity_portfolio.sharpe_ratio:.4f}")
    print(f"Weights: {risk_parity_portfolio.weights}")
    print()
    
    # Efficient frontier
    print("=== Efficient Frontier ===")
    efficient_portfolios = mpt.efficient_frontier(n_portfolios=20)
    
    print(f"Generated {len(efficient_portfolios)} efficient portfolios")
    print(f"Return range: {efficient_portfolios[0].expected_return:.4f} to {efficient_portfolios[-1].expected_return:.4f}")
    print(f"Risk range: {efficient_portfolios[0].volatility:.4f} to {efficient_portfolios[-1].volatility:.4f}")
    print()
    
    # Portfolio comparison
    print("=== Portfolio Comparison ===")
    portfolios = {
        'Equal Weight': equal_portfolio,
        'Min Variance': min_var_portfolio,
        'Max Sharpe': max_sharpe_portfolio,
        'Risk Parity': risk_parity_portfolio
    }
    
    print(f"{'Portfolio':<15} {'Return':<10} {'Risk':<10} {'Sharpe':<10} {'VaR 95%':<10}")
    print("-" * 60)
    
    for name, portfolio in portfolios.items():
        print(f"{name:<15} {portfolio.expected_return:<10.4f} {portfolio.volatility:<10.4f} "
              f"{portfolio.sharpe_ratio:<10.4f} {portfolio.var_95:<10.4f}")
    
    print()
    
    # Black-Litterman example
    print("=== Black-Litterman Model ===")
    
    # Express views: expect TECH_A to outperform by 2%, FIN_A to underperform by 1%
    views = {0: 0.02, 3: -0.01}  # Asset indices and expected excess returns
    view_uncertainties = {0: 0.05, 3: 0.03}  # Uncertainty in views
    
    try:
        bl_portfolio = mpt.black_litterman_portfolio(views, view_uncertainties)
        
        print(f"Expected Return: {bl_portfolio.expected_return:.4f}")
        print(f"Volatility: {bl_portfolio.volatility:.4f}")
        print(f"Sharpe Ratio: {bl_portfolio.sharpe_ratio:.4f}")
        print(f"Weights: {bl_portfolio.weights}")
        
    except Exception as e:
        print(f"Black-Litterman optimization failed: {e}")
    
    print()
    
    # Detailed analytics for Max Sharpe portfolio
    print("=== Detailed Portfolio Analytics (Max Sharpe) ===")
    analytics = mpt.portfolio_analytics(max_sharpe_portfolio.weights)
    
    print(f"Expected Return: {analytics['expected_return']:.4f}")
    print(f"Volatility: {analytics['volatility']:.4f}")
    print(f"Sharpe Ratio: {analytics['sharpe_ratio']:.4f}")
    print(f"Average Correlation: {analytics['average_correlation']:.4f}")
    print(f"Diversification Ratio: {analytics['diversification_ratio']:.4f}")
    print(f"Effective # Assets: {analytics['effective_number_assets']:.2f}")
    print()
    
    print("Risk Decomposition:")
    print(f"{'Asset':<10} {'Weight':<10} {'Marg Contrib':<12} {'% Risk Contrib':<15}")
    print("-" * 50)
    
    for i, name in enumerate(asset_names):
        weight = max_sharpe_portfolio.weights[i]
        mctr = analytics['marginal_contribution'][i]
        pctr = analytics['percentage_contribution'][i]
        
        print(f"{name:<10} {weight:<10.4f} {mctr:<12.6f} {pctr:<15.4%}")
    
    print()
    
    # Constraint optimization example
    print("=== Constrained Optimization ===")
    
    # Example: No more than 30% in any single asset, at least 5% in bonds
    constraints = OptimizationConstraints(
        min_weights=np.array([0, 0, 0, 0, 0, 0, 0, 0.05]),  # Min 5% bonds
        max_weights=np.array([0.3] * n_assets)  # Max 30% each
    )
    
    try:
        constrained_portfolio = mpt.maximum_sharpe_portfolio(constraints)
        
        print(f"Constrained Max Sharpe Portfolio:")
        print(f"Expected Return: {constrained_portfolio.expected_return:.4f}")
        print(f"Volatility: {constrained_portfolio.volatility:.4f}")
        print(f"Sharpe Ratio: {constrained_portfolio.sharpe_ratio:.4f}")
        print(f"Max weight: {np.max(constrained_portfolio.weights):.4f}")
        print(f"Bond allocation: {constrained_portfolio.weights[-1]:.4f}")
        
    except Exception as e:
        print(f"Constrained optimization failed: {e}")
    
    print()
    
    # Factor model example
    print("=== Factor Model Optimization ===")
    
    # Create simple 3-factor model (market, size, value)
    n_factors = 3
    factor_returns = np.random.multivariate_normal(
        [0.0005, 0.0002, 0.0001],  # Factor means
        [[0.0004, 0.0001, 0.00005],
         [0.0001, 0.0002, 0.00003],
         [0.00005, 0.00003, 0.0001]],  # Factor covariance
        n_periods
    )
    
    # Random factor loadings
    factor_loadings = np.random.normal(0, 0.5, (n_assets, n_factors))
    factor_loadings[:, 0] = np.random.normal(1, 0.3, n_assets)  # Market beta around 1
    
    try:
        factor_portfolio = mpt.factor_model_optimization(factor_loadings, factor_returns)
        
        print(f"Factor Model Portfolio:")
        print(f"Expected Return: {factor_portfolio.expected_return:.4f}")
        print(f"Volatility: {factor_portfolio.volatility:.4f}")
        print(f"Sharpe Ratio: {factor_portfolio.sharpe_ratio:.4f}")
        
    except Exception as e:
        print(f"Factor model optimization failed: {e}")


if __name__ == "__main__":
    comprehensive_mpt_example()