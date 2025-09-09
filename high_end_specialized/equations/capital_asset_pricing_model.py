"""
Capital Asset Pricing Model (CAPM) Equations
===========================================

Complete implementation of the Capital Asset Pricing Model and related
equations including Security Market Line, Alpha/Beta calculations,
and multi-factor extensions (Fama-French, Arbitrage Pricing Theory).

Mathematical Foundation:
CAPM: E(Ri) = Rf + βi(E(Rm) - Rf)

Where:
- E(Ri): Expected return of security i
- Rf: Risk-free rate
- βi: Beta of security i
- E(Rm): Expected market return

Extensions:
Fama-French 3-Factor: E(Ri) - Rf = αi + βi(Rm - Rf) + si*SMB + hi*HML
APT: E(Ri) = Rf + βi1*F1 + βi2*F2 + ... + βik*Fk

Applications:
- Asset pricing
- Cost of capital calculation
- Performance attribution
- Risk-adjusted returns
- Portfolio optimization
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings


@dataclass
class CAPMResults:
    """Results from CAPM analysis."""
    alpha: float
    beta: float
    r_squared: float
    sharpe_ratio: float
    treynor_ratio: float
    information_ratio: float
    jensen_alpha: float
    expected_return: float
    systematic_risk: float
    idiosyncratic_risk: float
    total_risk: float


@dataclass
class FactorModelResults:
    """Results from multi-factor model analysis."""
    alpha: float
    factor_loadings: Dict[str, float]
    r_squared: float
    adjusted_r_squared: float
    factor_contributions: Dict[str, float]
    residual_risk: float
    factor_risk: float
    total_risk: float


class CAPMAnalyzer:
    """
    Comprehensive Capital Asset Pricing Model implementation.
    
    Includes:
    - Single-factor CAPM
    - Multi-factor models (Fama-French, APT)
    - Performance attribution
    - Risk decomposition
    - Rolling analysis
    """
    
    def __init__(self, asset_returns: Union[np.ndarray, pd.Series],
                 market_returns: Union[np.ndarray, pd.Series],
                 risk_free_rate: Union[float, np.ndarray, pd.Series] = 0.0):
        """
        Initialize CAPM analyzer.
        
        Args:
            asset_returns: Asset returns time series
            market_returns: Market returns time series
            risk_free_rate: Risk-free rate (constant or time series)
        """
        self.asset_returns = np.array(asset_returns)
        self.market_returns = np.array(market_returns)
        
        # Handle risk-free rate
        if isinstance(risk_free_rate, (int, float)):
            self.risk_free_rate = np.full_like(self.asset_returns, risk_free_rate)
        else:
            self.risk_free_rate = np.array(risk_free_rate)
        
        # Calculate excess returns
        self.asset_excess_returns = self.asset_returns - self.risk_free_rate
        self.market_excess_returns = self.market_returns - self.risk_free_rate
        
        # Validate data
        if len(self.asset_returns) != len(self.market_returns):
            raise ValueError("Asset and market returns must have same length")
    
    def calculate_capm(self) -> CAPMResults:
        """
        Calculate single-factor CAPM metrics.
        
        Returns:
            CAPMResults object with all CAPM metrics
        """
        # Linear regression: Ri - Rf = α + β(Rm - Rf) + ε
        X = self.market_excess_returns.reshape(-1, 1)
        y = self.asset_excess_returns
        
        # Perform regression
        model = LinearRegression()
        model.fit(X, y)
        
        alpha = model.intercept_
        beta = model.coef_[0]
        
        # Calculate R-squared
        y_pred = model.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Calculate performance metrics
        asset_mean = np.mean(self.asset_returns)
        market_mean = np.mean(self.market_returns)
        rf_mean = np.mean(self.risk_free_rate)
        
        asset_vol = np.std(self.asset_returns)
        market_vol = np.std(self.market_returns)
        
        # Sharpe ratio
        sharpe_ratio = (asset_mean - rf_mean) / asset_vol if asset_vol > 0 else 0
        
        # Treynor ratio
        treynor_ratio = (asset_mean - rf_mean) / beta if beta != 0 else 0
        
        # Information ratio (active return / tracking error)
        active_returns = self.asset_returns - self.market_returns
        tracking_error = np.std(active_returns)
        information_ratio = np.mean(active_returns) / tracking_error if tracking_error > 0 else 0
        
        # Jensen's Alpha (excess return beyond CAPM prediction)
        capm_expected_return = rf_mean + beta * (market_mean - rf_mean)
        jensen_alpha = asset_mean - capm_expected_return
        
        # Risk decomposition
        systematic_risk = beta * market_vol  # Risk due to market exposure
        total_risk = asset_vol
        idiosyncratic_risk = np.sqrt(max(0, total_risk**2 - systematic_risk**2))
        
        return CAPMResults(
            alpha=alpha,
            beta=beta,
            r_squared=r_squared,
            sharpe_ratio=sharpe_ratio,
            treynor_ratio=treynor_ratio,
            information_ratio=information_ratio,
            jensen_alpha=jensen_alpha,
            expected_return=capm_expected_return,
            systematic_risk=systematic_risk,
            idiosyncratic_risk=idiosyncratic_risk,
            total_risk=total_risk
        )
    
    def fama_french_three_factor(self, smb_returns: np.ndarray, 
                                hml_returns: np.ndarray) -> FactorModelResults:
        """
        Calculate Fama-French three-factor model.
        
        Model: E(Ri) - Rf = αi + βi(Rm - Rf) + si*SMB + hi*HML
        
        Args:
            smb_returns: Small Minus Big factor returns
            hml_returns: High Minus Low factor returns
            
        Returns:
            FactorModelResults object
        """
        # Prepare factors matrix
        factors = np.column_stack([
            self.market_excess_returns,
            smb_returns,
            hml_returns
        ])
        
        factor_names = ["Market", "SMB", "HML"]
        
        return self._fit_factor_model(factors, factor_names)
    
    def arbitrage_pricing_theory(self, factor_returns: np.ndarray, 
                               factor_names: List[str]) -> FactorModelResults:
        """
        Calculate Arbitrage Pricing Theory (APT) model.
        
        Model: E(Ri) = Rf + βi1*F1 + βi2*F2 + ... + βik*Fk
        
        Args:
            factor_returns: Matrix of factor returns [time x factors]
            factor_names: Names of the factors
            
        Returns:
            FactorModelResults object
        """
        return self._fit_factor_model(factor_returns, factor_names)
    
    def _fit_factor_model(self, factors: np.ndarray, 
                         factor_names: List[str]) -> FactorModelResults:
        """
        Fit multi-factor model using regression.
        
        Args:
            factors: Matrix of factor returns
            factor_names: Names of factors
            
        Returns:
            FactorModelResults object
        """
        # Perform multiple regression
        model = LinearRegression()
        model.fit(factors, self.asset_excess_returns)
        
        alpha = model.intercept_
        factor_loadings = dict(zip(factor_names, model.coef_))
        
        # Calculate R-squared and adjusted R-squared
        y_pred = model.predict(factors)
        ss_res = np.sum((self.asset_excess_returns - y_pred) ** 2)
        ss_tot = np.sum((self.asset_excess_returns - np.mean(self.asset_excess_returns)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        n = len(self.asset_excess_returns)
        p = factors.shape[1]
        adjusted_r_squared = 1 - ((1 - r_squared) * (n - 1) / (n - p - 1))
        
        # Calculate factor contributions to risk
        factor_covariance = np.cov(factors.T)
        factor_loadings_array = np.array(list(factor_loadings.values()))
        
        factor_risk_squared = np.dot(factor_loadings_array.T, 
                                   np.dot(factor_covariance, factor_loadings_array))
        factor_risk = np.sqrt(factor_risk_squared)
        
        residual_variance = np.var(self.asset_excess_returns - y_pred)
        residual_risk = np.sqrt(residual_variance)
        
        total_risk = np.std(self.asset_returns)
        
        # Calculate factor contributions to return
        factor_means = np.mean(factors, axis=0)
        factor_contributions = {
            name: loading * factor_means[i] 
            for i, (name, loading) in enumerate(factor_loadings.items())
        }
        
        return FactorModelResults(
            alpha=alpha,
            factor_loadings=factor_loadings,
            r_squared=r_squared,
            adjusted_r_squared=adjusted_r_squared,
            factor_contributions=factor_contributions,
            residual_risk=residual_risk,
            factor_risk=factor_risk,
            total_risk=total_risk
        )
    
    def rolling_capm_analysis(self, window_size: int = 252) -> pd.DataFrame:
        """
        Perform rolling CAPM analysis over time.
        
        Args:
            window_size: Rolling window size in periods
            
        Returns:
            DataFrame with rolling CAPM metrics
        """
        if len(self.asset_returns) < window_size:
            raise ValueError("Data length must be greater than window size")
        
        results = []
        
        for i in range(window_size, len(self.asset_returns) + 1):
            # Extract window data
            window_asset = self.asset_returns[i-window_size:i]
            window_market = self.market_returns[i-window_size:i]
            window_rf = self.risk_free_rate[i-window_size:i]
            
            # Create temporary analyzer
            temp_analyzer = CAPMAnalyzer(window_asset, window_market, window_rf)
            capm_result = temp_analyzer.calculate_capm()
            
            results.append({
                'period': i,
                'alpha': capm_result.alpha,
                'beta': capm_result.beta,
                'r_squared': capm_result.r_squared,
                'sharpe_ratio': capm_result.sharpe_ratio,
                'jensen_alpha': capm_result.jensen_alpha
            })
        
        return pd.DataFrame(results)
    
    def calculate_security_market_line(self, beta_range: Tuple[float, float] = (0, 2),
                                     num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Security Market Line for given beta range.
        
        Args:
            beta_range: Range of beta values to calculate
            num_points: Number of points on the line
            
        Returns:
            Tuple of (beta_values, expected_returns)
        """
        beta_values = np.linspace(beta_range[0], beta_range[1], num_points)
        
        rf_mean = np.mean(self.risk_free_rate)
        market_mean = np.mean(self.market_returns)
        market_premium = market_mean - rf_mean
        
        expected_returns = rf_mean + beta_values * market_premium
        
        return beta_values, expected_returns
    
    def test_capm_assumptions(self) -> Dict[str, Dict[str, float]]:
        """
        Test CAPM assumptions statistically.
        
        Returns:
            Dictionary with test results
        """
        results = {}
        
        # Test 1: Normality of returns
        asset_shapiro = stats.shapiro(self.asset_returns)
        market_shapiro = stats.shapiro(self.market_returns)
        
        results['normality'] = {
            'asset_statistic': asset_shapiro.statistic,
            'asset_p_value': asset_shapiro.pvalue,
            'market_statistic': market_shapiro.statistic,
            'market_p_value': market_shapiro.pvalue
        }
        
        # Test 2: Stationarity (Augmented Dickey-Fuller)
        try:
            from statsmodels.tsa.stattools import adfuller
            
            asset_adf = adfuller(self.asset_returns)
            market_adf = adfuller(self.market_returns)
            
            results['stationarity'] = {
                'asset_statistic': asset_adf[0],
                'asset_p_value': asset_adf[1],
                'market_statistic': market_adf[0],
                'market_p_value': market_adf[1]
            }
        except ImportError:
            results['stationarity'] = {'error': 'statsmodels not available'}
        
        # Test 3: Linear relationship
        correlation = np.corrcoef(self.asset_excess_returns, self.market_excess_returns)[0, 1]
        
        results['linearity'] = {
            'correlation': correlation,
            'correlation_squared': correlation ** 2
        }
        
        # Test 4: Homoscedasticity (Breusch-Pagan test approximation)
        # Fit CAPM regression
        X = self.market_excess_returns.reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, self.asset_excess_returns)
        residuals = self.asset_excess_returns - model.predict(X)
        
        # Test if residual variance depends on market returns
        residuals_squared = residuals ** 2
        het_model = LinearRegression()
        het_model.fit(X, residuals_squared)
        het_r_squared = het_model.score(X, residuals_squared)
        
        # Approximate Breusch-Pagan statistic
        n = len(residuals)
        bp_statistic = n * het_r_squared
        bp_p_value = 1 - stats.chi2.cdf(bp_statistic, df=1)
        
        results['homoscedasticity'] = {
            'bp_statistic': bp_statistic,
            'bp_p_value': bp_p_value
        }
        
        return results


def comprehensive_capm_example():
    """Comprehensive example demonstrating CAPM analysis."""
    print("=== Capital Asset Pricing Model Analysis ===")
    
    # Generate synthetic data
    np.random.seed(42)
    n_periods = 1000
    
    # Market returns (assume normally distributed)
    market_returns = np.random.normal(0.0008, 0.015, n_periods)  # Daily returns
    risk_free_rate = 0.0002  # Daily risk-free rate
    
    # Generate asset returns with specific beta
    true_beta = 1.2
    true_alpha = 0.0001
    
    # Asset returns = alpha + beta * market + idiosyncratic noise
    idiosyncratic_noise = np.random.normal(0, 0.008, n_periods)
    asset_returns = true_alpha + true_beta * market_returns + idiosyncratic_noise
    
    print(f"Data Generation:")
    print(f"  Periods: {n_periods}")
    print(f"  True Alpha: {true_alpha:.6f}")
    print(f"  True Beta: {true_beta:.3f}")
    print(f"  Market volatility: {np.std(market_returns):.6f}")
    print(f"  Asset volatility: {np.std(asset_returns):.6f}")
    print()
    
    # Initialize CAPM analyzer
    capm_analyzer = CAPMAnalyzer(asset_returns, market_returns, risk_free_rate)
    
    # Calculate CAPM metrics
    capm_results = capm_analyzer.calculate_capm()
    
    print("=== CAPM Results ===")
    print(f"Estimated Alpha:      {capm_results.alpha:.6f} (True: {true_alpha:.6f})")
    print(f"Estimated Beta:       {capm_results.beta:.6f} (True: {true_beta:.3f})")
    print(f"R-squared:            {capm_results.r_squared:.6f}")
    print(f"Sharpe Ratio:         {capm_results.sharpe_ratio:.6f}")
    print(f"Treynor Ratio:        {capm_results.treynor_ratio:.6f}")
    print(f"Information Ratio:    {capm_results.information_ratio:.6f}")
    print(f"Jensen's Alpha:       {capm_results.jensen_alpha:.6f}")
    print(f"Expected Return:      {capm_results.expected_return:.6f}")
    print()
    
    print("=== Risk Decomposition ===")
    print(f"Total Risk:           {capm_results.total_risk:.6f}")
    print(f"Systematic Risk:      {capm_results.systematic_risk:.6f}")
    print(f"Idiosyncratic Risk:   {capm_results.idiosyncratic_risk:.6f}")
    print(f"Systematic %:         {(capm_results.systematic_risk/capm_results.total_risk)**2:.2%}")
    print(f"Idiosyncratic %:      {(capm_results.idiosyncratic_risk/capm_results.total_risk)**2:.2%}")
    print()
    
    # Fama-French three-factor model
    print("=== Fama-French Three-Factor Model ===")
    
    # Generate synthetic factor data
    smb_returns = np.random.normal(0.0002, 0.005, n_periods)  # Size factor
    hml_returns = np.random.normal(0.0001, 0.004, n_periods)  # Value factor
    
    ff_results = capm_analyzer.fama_french_three_factor(smb_returns, hml_returns)
    
    print(f"Alpha:                {ff_results.alpha:.6f}")
    print(f"Factor Loadings:")
    for factor, loading in ff_results.factor_loadings.items():
        print(f"  {factor:>10}: {loading:.6f}")
    print(f"R-squared:            {ff_results.r_squared:.6f}")
    print(f"Adjusted R-squared:   {ff_results.adjusted_r_squared:.6f}")
    print()
    
    print(f"Risk Decomposition:")
    print(f"  Factor Risk:        {ff_results.factor_risk:.6f}")
    print(f"  Residual Risk:      {ff_results.residual_risk:.6f}")
    print(f"  Total Risk:         {ff_results.total_risk:.6f}")
    print()
    
    print(f"Factor Contributions to Return:")
    for factor, contribution in ff_results.factor_contributions.items():
        print(f"  {factor:>10}: {contribution:.6f}")
    print()
    
    # APT model with custom factors
    print("=== Arbitrage Pricing Theory Model ===")
    
    # Generate synthetic macroeconomic factors
    factor_data = np.column_stack([
        market_returns,  # Market factor
        np.random.normal(0.0001, 0.003, n_periods),  # Interest rate factor
        np.random.normal(0.0000, 0.004, n_periods),  # Inflation factor
        np.random.normal(0.0001, 0.005, n_periods),  # GDP growth factor
    ])
    
    factor_names = ["Market", "Interest_Rate", "Inflation", "GDP_Growth"]
    
    apt_results = capm_analyzer.arbitrage_pricing_theory(factor_data, factor_names)
    
    print(f"Alpha:                {apt_results.alpha:.6f}")
    print(f"Factor Loadings:")
    for factor, loading in apt_results.factor_loadings.items():
        print(f"  {factor:>12}: {loading:.6f}")
    print(f"R-squared:            {apt_results.r_squared:.6f}")
    print()
    
    # Security Market Line
    print("=== Security Market Line ===")
    
    beta_values, sml_returns = capm_analyzer.calculate_security_market_line()
    
    # Show selected points on SML
    selected_betas = [0.0, 0.5, 1.0, 1.5, 2.0]
    print(f"{'Beta':<8} {'Expected Return':<15}")
    print("-" * 25)
    
    for beta in selected_betas:
        idx = np.argmin(np.abs(beta_values - beta))
        expected_ret = sml_returns[idx]
        print(f"{beta:<8.1f} {expected_ret:<15.6f}")
    
    print()
    
    # Test CAPM assumptions
    print("=== CAPM Assumptions Testing ===")
    
    assumption_tests = capm_analyzer.test_capm_assumptions()
    
    print(f"Normality Tests:")
    print(f"  Asset Shapiro-Wilk:   p-value = {assumption_tests['normality']['asset_p_value']:.6f}")
    print(f"  Market Shapiro-Wilk:  p-value = {assumption_tests['normality']['market_p_value']:.6f}")
    print()
    
    if 'stationarity' in assumption_tests and 'error' not in assumption_tests['stationarity']:
        print(f"Stationarity Tests (ADF):")
        print(f"  Asset ADF:            p-value = {assumption_tests['stationarity']['asset_p_value']:.6f}")
        print(f"  Market ADF:           p-value = {assumption_tests['stationarity']['market_p_value']:.6f}")
        print()
    
    print(f"Linear Relationship:")
    print(f"  Correlation:          {assumption_tests['linearity']['correlation']:.6f}")
    print(f"  R-squared:            {assumption_tests['linearity']['correlation_squared']:.6f}")
    print()
    
    print(f"Homoscedasticity (Breusch-Pagan):")
    print(f"  BP Statistic:         {assumption_tests['homoscedasticity']['bp_statistic']:.6f}")
    print(f"  P-value:              {assumption_tests['homoscedasticity']['bp_p_value']:.6f}")
    print()
    
    # Performance evaluation
    print("=== Performance Evaluation ===")
    
    # Calculate various alpha measures
    market_mean = np.mean(market_returns)
    rf_mean = risk_free_rate
    asset_mean = np.mean(asset_returns)
    
    # Raw excess return
    raw_excess = asset_mean - rf_mean
    print(f"Raw Excess Return:    {raw_excess:.6f}")
    
    # Market-adjusted return
    market_adjusted = asset_mean - market_mean
    print(f"Market-Adjusted:      {market_adjusted:.6f}")
    
    # CAPM Alpha (Jensen's Alpha)
    print(f"Jensen's Alpha:       {capm_results.jensen_alpha:.6f}")
    
    # Three-factor Alpha
    print(f"Three-Factor Alpha:   {ff_results.alpha:.6f}")
    
    # APT Alpha
    print(f"APT Alpha:            {apt_results.alpha:.6f}")


if __name__ == "__main__":
    comprehensive_capm_example()