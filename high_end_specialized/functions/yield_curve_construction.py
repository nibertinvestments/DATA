"""
Yield Curve Construction and Analysis
====================================

Advanced mathematical models for yield curve construction including
Nelson-Siegel, Svensson, and spline-based methods with applications
in fixed income pricing and risk management.

Mathematical Foundation:
Nelson-Siegel: y(τ) = β₀ + β₁((1-e^(-τ/λ))/(τ/λ)) + β₂((1-e^(-τ/λ))/(τ/λ) - e^(-τ/λ))

Svensson Extension: 
y(τ) = β₀ + β₁((1-e^(-τ/λ₁))/(τ/λ₁)) + β₂((1-e^(-τ/λ₁))/(τ/λ₁) - e^(-τ/λ₁)) + β₃((1-e^(-τ/λ₂))/(τ/λ₂) - e^(-τ/λ₂))

Applications:
- Bond pricing
- Interest rate derivatives
- Risk management
- Central bank policy
- Portfolio management
"""

import numpy as np
from scipy.optimize import minimize, least_squares
from scipy.interpolate import CubicSpline, BSpline
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings


@dataclass
class YieldCurveData:
    """Market yield curve data."""
    maturities: np.ndarray  # Time to maturity in years
    yields: np.ndarray      # Yields (as decimals)
    weights: Optional[np.ndarray] = None  # Optional weights for fitting


@dataclass
class CurveParameters:
    """Parameters for yield curve models."""
    model_type: str
    parameters: Dict[str, float]
    r_squared: float
    rmse: float
    fitting_error: float


class YieldCurveBuilder:
    """
    Advanced yield curve construction and analysis.
    
    Includes multiple parametric and non-parametric methods:
    - Nelson-Siegel model
    - Nelson-Siegel-Svensson model
    - Cubic spline interpolation
    - B-spline smoothing
    - Bootstrapping method
    """
    
    def __init__(self, market_data: YieldCurveData):
        """
        Initialize with market data.
        
        Args:
            market_data: YieldCurveData object with market observations
        """
        self.market_data = market_data
        self.fitted_models = {}
    
    def nelson_siegel_yield(self, tau: np.ndarray, beta0: float, beta1: float, 
                           beta2: float, lambda_param: float) -> np.ndarray:
        """
        Calculate yield using Nelson-Siegel model.
        
        Args:
            tau: Time to maturity
            beta0: Long-term factor
            beta1: Short-term factor  
            beta2: Medium-term factor
            lambda_param: Decay parameter
            
        Returns:
            Yields for given maturities
        """
        tau = np.asarray(tau)
        
        # Avoid division by zero
        tau_lambda = tau / lambda_param
        tau_lambda = np.where(tau_lambda == 0, 1e-10, tau_lambda)
        
        # Nelson-Siegel formula
        factor1 = (1 - np.exp(-tau_lambda)) / tau_lambda
        factor2 = factor1 - np.exp(-tau_lambda)
        
        return beta0 + beta1 * factor1 + beta2 * factor2
    
    def fit_nelson_siegel(self, initial_guess: Optional[List[float]] = None) -> CurveParameters:
        """
        Fit Nelson-Siegel model to market data.
        
        Args:
            initial_guess: Optional initial parameter guess [beta0, beta1, beta2, lambda]
            
        Returns:
            CurveParameters with fitted model
        """
        if initial_guess is None:
            # Smart initial guess
            long_yield = self.market_data.yields[-1]  # Long-term yield
            short_yield = self.market_data.yields[0]   # Short-term yield
            
            initial_guess = [
                long_yield,              # beta0 (long-term)
                short_yield - long_yield, # beta1 (short-term factor)
                0.0,                     # beta2 (medium-term factor)
                2.0                      # lambda (decay parameter)
            ]
        
        def objective(params):
            beta0, beta1, beta2, lambda_param = params
            
            if lambda_param <= 0:
                return 1e10  # Penalty for invalid lambda
            
            fitted_yields = self.nelson_siegel_yield(
                self.market_data.maturities, beta0, beta1, beta2, lambda_param
            )
            
            # Weighted squared errors
            weights = self.market_data.weights if self.market_data.weights is not None else 1.0
            errors = (fitted_yields - self.market_data.yields) * np.sqrt(weights)
            
            return np.sum(errors ** 2)
        
        # Optimize
        bounds = [
            (-0.1, 0.2),    # beta0
            (-0.2, 0.2),    # beta1
            (-0.2, 0.2),    # beta2
            (0.1, 10.0)     # lambda
        ]
        
        result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
        
        if not result.success:
            warnings.warn(f"Nelson-Siegel optimization failed: {result.message}")
        
        # Calculate fit statistics
        beta0, beta1, beta2, lambda_param = result.x
        fitted_yields = self.nelson_siegel_yield(
            self.market_data.maturities, beta0, beta1, beta2, lambda_param
        )
        
        r_squared, rmse = self._calculate_fit_statistics(fitted_yields)
        
        parameters = CurveParameters(
            model_type="Nelson-Siegel",
            parameters={
                'beta0': beta0,
                'beta1': beta1,
                'beta2': beta2,
                'lambda': lambda_param
            },
            r_squared=r_squared,
            rmse=rmse,
            fitting_error=result.fun
        )
        
        self.fitted_models['nelson_siegel'] = parameters
        return parameters
    
    def svensson_yield(self, tau: np.ndarray, beta0: float, beta1: float, 
                      beta2: float, beta3: float, lambda1: float, lambda2: float) -> np.ndarray:
        """
        Calculate yield using Nelson-Siegel-Svensson model.
        
        Args:
            tau: Time to maturity
            beta0: Long-term factor
            beta1: Short-term factor
            beta2: Medium-term factor 1
            beta3: Medium-term factor 2
            lambda1: First decay parameter
            lambda2: Second decay parameter
            
        Returns:
            Yields for given maturities
        """
        tau = np.asarray(tau)
        
        # Nelson-Siegel components
        ns_yield = self.nelson_siegel_yield(tau, beta0, beta1, beta2, lambda1)
        
        # Additional Svensson component
        tau_lambda2 = tau / lambda2
        tau_lambda2 = np.where(tau_lambda2 == 0, 1e-10, tau_lambda2)
        
        svensson_factor = ((1 - np.exp(-tau_lambda2)) / tau_lambda2 - np.exp(-tau_lambda2))
        
        return ns_yield + beta3 * svensson_factor
    
    def fit_svensson(self, initial_guess: Optional[List[float]] = None) -> CurveParameters:
        """
        Fit Nelson-Siegel-Svensson model to market data.
        
        Args:
            initial_guess: Optional initial parameter guess
            
        Returns:
            CurveParameters with fitted model
        """
        if initial_guess is None:
            # Start with Nelson-Siegel fit
            ns_params = self.fit_nelson_siegel()
            
            initial_guess = [
                ns_params.parameters['beta0'],
                ns_params.parameters['beta1'],
                ns_params.parameters['beta2'],
                0.0,  # beta3
                ns_params.parameters['lambda'],
                5.0   # lambda2
            ]
        
        def objective(params):
            beta0, beta1, beta2, beta3, lambda1, lambda2 = params
            
            if lambda1 <= 0 or lambda2 <= 0:
                return 1e10
            
            fitted_yields = self.svensson_yield(
                self.market_data.maturities, beta0, beta1, beta2, beta3, lambda1, lambda2
            )
            
            weights = self.market_data.weights if self.market_data.weights is not None else 1.0
            errors = (fitted_yields - self.market_data.yields) * np.sqrt(weights)
            
            return np.sum(errors ** 2)
        
        # Optimize
        bounds = [
            (-0.1, 0.2),    # beta0
            (-0.2, 0.2),    # beta1
            (-0.2, 0.2),    # beta2
            (-0.2, 0.2),    # beta3
            (0.1, 10.0),    # lambda1
            (0.1, 10.0)     # lambda2
        ]
        
        result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
        
        if not result.success:
            warnings.warn(f"Svensson optimization failed: {result.message}")
        
        # Calculate fit statistics
        beta0, beta1, beta2, beta3, lambda1, lambda2 = result.x
        fitted_yields = self.svensson_yield(
            self.market_data.maturities, beta0, beta1, beta2, beta3, lambda1, lambda2
        )
        
        r_squared, rmse = self._calculate_fit_statistics(fitted_yields)
        
        parameters = CurveParameters(
            model_type="Svensson",
            parameters={
                'beta0': beta0,
                'beta1': beta1,
                'beta2': beta2,
                'beta3': beta3,
                'lambda1': lambda1,
                'lambda2': lambda2
            },
            r_squared=r_squared,
            rmse=rmse,
            fitting_error=result.fun
        )
        
        self.fitted_models['svensson'] = parameters
        return parameters
    
    def fit_cubic_spline(self, smoothing_factor: float = 0.0) -> CurveParameters:
        """
        Fit cubic spline to yield curve data.
        
        Args:
            smoothing_factor: Smoothing parameter (0 = interpolating spline)
            
        Returns:
            CurveParameters with spline model
        """
        if smoothing_factor == 0.0:
            # Interpolating spline
            spline = CubicSpline(self.market_data.maturities, self.market_data.yields)
        else:
            # Smoothing spline (would need scipy.interpolate.UnivariateSpline)
            from scipy.interpolate import UnivariateSpline
            weights = self.market_data.weights if self.market_data.weights is not None else None
            spline = UnivariateSpline(
                self.market_data.maturities, 
                self.market_data.yields,
                w=weights,
                s=smoothing_factor
            )
        
        # Evaluate fitted yields
        fitted_yields = spline(self.market_data.maturities)
        r_squared, rmse = self._calculate_fit_statistics(fitted_yields)
        
        parameters = CurveParameters(
            model_type="Cubic Spline",
            parameters={'smoothing_factor': smoothing_factor},
            r_squared=r_squared,
            rmse=rmse,
            fitting_error=rmse
        )
        
        self.fitted_models['cubic_spline'] = {
            'parameters': parameters,
            'spline': spline
        }
        
        return parameters
    
    def bootstrap_zero_curve(self, bond_prices: np.ndarray, 
                            coupon_rates: np.ndarray,
                            face_values: np.ndarray = None) -> np.ndarray:
        """
        Bootstrap zero-coupon curve from bond prices.
        
        Args:
            bond_prices: Market prices of bonds
            coupon_rates: Annual coupon rates
            face_values: Face values (default 100)
            
        Returns:
            Zero-coupon yields
        """
        if face_values is None:
            face_values = np.full_like(bond_prices, 100.0)
        
        n_bonds = len(bond_prices)
        zero_yields = np.zeros(n_bonds)
        maturities = self.market_data.maturities
        
        for i in range(n_bonds):
            maturity = maturities[i]
            coupon_rate = coupon_rates[i]
            bond_price = bond_prices[i]
            face_value = face_values[i]
            
            if i == 0:
                # First bond (shortest maturity)
                zero_yields[i] = (face_value * (1 + coupon_rate) / bond_price) ** (1/maturity) - 1
            else:
                # Discount previous coupon payments
                annual_coupon = face_value * coupon_rate
                present_value_coupons = 0
                
                for j in range(i):
                    # Approximate coupon timing
                    coupon_time = maturities[j]
                    if coupon_time < maturity:
                        discount_factor = (1 + zero_yields[j]) ** (-coupon_time)
                        present_value_coupons += annual_coupon * discount_factor
                
                # Solve for zero yield
                remaining_pv = bond_price - present_value_coupons
                final_payment = face_value * (1 + coupon_rate)
                
                if remaining_pv > 0:
                    zero_yields[i] = (final_payment / remaining_pv) ** (1/maturity) - 1
                else:
                    zero_yields[i] = zero_yields[i-1]  # Fallback
        
        return zero_yields
    
    def _calculate_fit_statistics(self, fitted_yields: np.ndarray) -> Tuple[float, float]:
        """Calculate R-squared and RMSE."""
        observed = self.market_data.yields
        
        # R-squared
        ss_res = np.sum((observed - fitted_yields) ** 2)
        ss_tot = np.sum((observed - np.mean(observed)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # RMSE
        rmse = np.sqrt(np.mean((observed - fitted_yields) ** 2))
        
        return r_squared, rmse
    
    def interpolate_yield(self, maturity: float, model: str = 'svensson') -> float:
        """
        Interpolate yield for given maturity using fitted model.
        
        Args:
            maturity: Time to maturity in years
            model: Model to use ('nelson_siegel', 'svensson', 'cubic_spline')
            
        Returns:
            Interpolated yield
        """
        if model not in self.fitted_models:
            raise ValueError(f"Model {model} not fitted. Available: {list(self.fitted_models.keys())}")
        
        if model == 'nelson_siegel':
            params = self.fitted_models[model].parameters
            return self.nelson_siegel_yield(
                np.array([maturity]), 
                params['beta0'], params['beta1'], params['beta2'], params['lambda']
            )[0]
        
        elif model == 'svensson':
            params = self.fitted_models[model].parameters
            return self.svensson_yield(
                np.array([maturity]),
                params['beta0'], params['beta1'], params['beta2'], params['beta3'],
                params['lambda1'], params['lambda2']
            )[0]
        
        elif model == 'cubic_spline':
            spline = self.fitted_models[model]['spline']
            return float(spline(maturity))
        
        else:
            raise ValueError(f"Unknown model: {model}")
    
    def forward_rate(self, t1: float, t2: float, model: str = 'svensson') -> float:
        """
        Calculate forward rate between two maturities.
        
        Forward rate: f(t1,t2) = [y(t2)*t2 - y(t1)*t1] / (t2 - t1)
        
        Args:
            t1: Start time
            t2: End time  
            model: Model to use
            
        Returns:
            Forward rate
        """
        if t2 <= t1:
            raise ValueError("t2 must be greater than t1")
        
        y1 = self.interpolate_yield(t1, model)
        y2 = self.interpolate_yield(t2, model)
        
        return (y2 * t2 - y1 * t1) / (t2 - t1)
    
    def duration_and_convexity(self, yield_level: float, coupon_rate: float, 
                              maturity: float, face_value: float = 100) -> Tuple[float, float]:
        """
        Calculate modified duration and convexity.
        
        Args:
            yield_level: Yield to maturity
            coupon_rate: Annual coupon rate
            maturity: Time to maturity
            face_value: Face value
            
        Returns:
            Tuple of (modified_duration, convexity)
        """
        # Assume annual payments for simplicity
        n_payments = int(maturity)
        payment_times = np.arange(1, n_payments + 1)
        
        # Cash flows
        annual_coupon = face_value * coupon_rate
        cash_flows = np.full(n_payments, annual_coupon)
        cash_flows[-1] += face_value  # Principal repayment
        
        # Present values
        discount_factors = (1 + yield_level) ** (-payment_times)
        present_values = cash_flows * discount_factors
        bond_price = np.sum(present_values)
        
        # Duration
        duration = np.sum(payment_times * present_values) / bond_price
        modified_duration = duration / (1 + yield_level)
        
        # Convexity
        convexity_numerator = np.sum(payment_times * (payment_times + 1) * present_values)
        convexity = convexity_numerator / (bond_price * (1 + yield_level) ** 2)
        
        return modified_duration, convexity


def comprehensive_yield_curve_example():
    """Comprehensive example demonstrating yield curve construction."""
    print("=== Yield Curve Construction and Analysis Example ===")
    
    # Create synthetic market data (US Treasury-like curve)
    maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])  # Years
    yields = np.array([0.015, 0.018, 0.020, 0.022, 0.024, 0.028, 0.032, 0.035, 0.038, 0.040])  # 4%
    
    # Add some market noise
    np.random.seed(42)
    yields += np.random.normal(0, 0.001, len(yields))
    
    # Create yield curve data
    market_data = YieldCurveData(maturities=maturities, yields=yields)
    
    print(f"Market Data:")
    print(f"  Maturities: {maturities}")
    print(f"  Yields: {(yields * 100).tolist()}")  # Convert to list for proper formatting
    print()
    
    # Initialize yield curve builder
    curve_builder = YieldCurveBuilder(market_data)
    
    # Fit Nelson-Siegel model
    print("=== Nelson-Siegel Model ===")
    ns_params = curve_builder.fit_nelson_siegel()
    
    print(f"Parameters:")
    for param, value in ns_params.parameters.items():
        print(f"  {param}: {value:.6f}")
    print(f"R-squared: {ns_params.r_squared:.6f}")
    print(f"RMSE: {ns_params.rmse * 10000:.2f} basis points")
    print()
    
    # Fit Svensson model
    print("=== Nelson-Siegel-Svensson Model ===")
    svensson_params = curve_builder.fit_svensson()
    
    print(f"Parameters:")
    for param, value in svensson_params.parameters.items():
        print(f"  {param}: {value:.6f}")
    print(f"R-squared: {svensson_params.r_squared:.6f}")
    print(f"RMSE: {svensson_params.rmse * 10000:.2f} basis points")
    print()
    
    # Fit cubic spline
    print("=== Cubic Spline Model ===")
    spline_params = curve_builder.fit_cubic_spline()
    
    print(f"R-squared: {spline_params.r_squared:.6f}")
    print(f"RMSE: {spline_params.rmse * 10000:.2f} basis points")
    print()
    
    # Model comparison
    print("=== Model Comparison ===")
    models = ['nelson_siegel', 'svensson', 'cubic_spline']
    
    print(f"{'Model':<20} {'R-squared':<12} {'RMSE (bps)':<12} {'Parameters':<12}")
    print("-" * 60)
    
    for model_name in models:
        if model_name in curve_builder.fitted_models:
            if model_name == 'cubic_spline':
                params_obj = curve_builder.fitted_models[model_name]['parameters']
                n_params = 1  # Simplified
            else:
                params_obj = curve_builder.fitted_models[model_name]
                n_params = len(params_obj.parameters)
            
            print(f"{model_name:<20} {params_obj.r_squared:<12.6f} "
                  f"{params_obj.rmse * 10000:<12.2f} {n_params:<12}")
    
    print()
    
    # Interpolation examples
    print("=== Yield Interpolation ===")
    
    test_maturities = [1.5, 4, 6, 8, 15, 25]
    
    print(f"{'Maturity':<10} {'N-S Yield':<12} {'Svensson':<12} {'Spline':<12}")
    print("-" * 50)
    
    for maturity in test_maturities:
        ns_yield = curve_builder.interpolate_yield(maturity, 'nelson_siegel')
        svensson_yield = curve_builder.interpolate_yield(maturity, 'svensson')
        spline_yield = curve_builder.interpolate_yield(maturity, 'cubic_spline')
        
        print(f"{maturity:<10.1f} {ns_yield*100:<12.4f} {svensson_yield*100:<12.4f} {spline_yield*100:<12.4f}")
    
    print()
    
    # Forward rate analysis
    print("=== Forward Rate Analysis ===")
    
    forward_periods = [(1, 2), (2, 3), (5, 7), (7, 10), (10, 20)]
    
    print(f"{'Period':<10} {'Forward Rate':<15} {'Current Spread':<15}")
    print("-" * 45)
    
    for t1, t2 in forward_periods:
        forward_rate = curve_builder.forward_rate(t1, t2, 'svensson')
        y1 = curve_builder.interpolate_yield(t1, 'svensson')
        y2 = curve_builder.interpolate_yield(t2, 'svensson')
        spread = y2 - y1
        
        print(f"{t1}Y-{t2}Y{'':<4} {forward_rate*100:<15.4f} {spread*10000:<15.2f}")
    
    print()
    
    # Duration and convexity analysis
    print("=== Duration and Convexity Analysis ===")
    
    # Example bonds
    bond_examples = [
        (0.02, 2),    # 2% coupon, 2-year
        (0.03, 5),    # 3% coupon, 5-year
        (0.04, 10),   # 4% coupon, 10-year
        (0.035, 30)   # 3.5% coupon, 30-year
    ]
    
    print(f"{'Bond':<15} {'Yield':<10} {'Duration':<12} {'Convexity':<12}")
    print("-" * 55)
    
    for coupon, maturity in bond_examples:
        yield_level = curve_builder.interpolate_yield(maturity, 'svensson')
        duration, convexity = curve_builder.duration_and_convexity(
            yield_level, coupon, maturity
        )
        
        bond_desc = f"{coupon*100:.1f}% {maturity}Y"
        print(f"{bond_desc:<15} {yield_level*100:<10.4f} {duration:<12.4f} {convexity:<12.4f}")
    
    print()
    
    # Curve shape analysis
    print("=== Curve Shape Analysis ===")
    
    # Calculate curve metrics using Svensson model
    short_rate = curve_builder.interpolate_yield(1, 'svensson')
    long_rate = curve_builder.interpolate_yield(30, 'svensson')
    ten_year_rate = curve_builder.interpolate_yield(10, 'svensson')
    
    # Level, slope, curvature
    level = (short_rate + long_rate) / 2
    slope = long_rate - short_rate
    curvature = 2 * ten_year_rate - short_rate - long_rate
    
    print(f"Curve Level: {level*100:.4f}%")
    print(f"Curve Slope: {slope*10000:.2f} basis points")
    print(f"Curve Curvature: {curvature*10000:.2f} basis points")
    print()
    
    # Stress testing
    print("=== Yield Curve Stress Testing ===")
    
    stress_scenarios = {
        'Parallel +100bp': np.ones_like(yields) * 0.01,
        'Steepening': np.linspace(0, 0.02, len(yields)),
        'Flattening': np.linspace(0.01, -0.01, len(yields)),
        'Twist': np.where(maturities <= 5, -0.005, 0.01)
    }
    
    print(f"{'Scenario':<15} {'2Y Change':<12} {'10Y Change':<12} {'30Y Change':<12}")
    print("-" * 55)
    
    for scenario_name, shock in stress_scenarios.items():
        stressed_yields = yields + shock
        stressed_data = YieldCurveData(maturities=maturities, yields=stressed_yields)
        stressed_builder = YieldCurveBuilder(stressed_data)
        stressed_builder.fit_svensson()
        
        change_2y = stressed_builder.interpolate_yield(2, 'svensson') - curve_builder.interpolate_yield(2, 'svensson')
        change_10y = stressed_builder.interpolate_yield(10, 'svensson') - curve_builder.interpolate_yield(10, 'svensson')
        change_30y = stressed_builder.interpolate_yield(30, 'svensson') - curve_builder.interpolate_yield(30, 'svensson')
        
        print(f"{scenario_name:<15} {change_2y*10000:<12.0f} {change_10y*10000:<12.0f} {change_30y*10000:<12.0f}")


if __name__ == "__main__":
    comprehensive_yield_curve_example()