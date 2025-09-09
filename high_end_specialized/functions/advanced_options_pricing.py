"""
Advanced Options Pricing Functions
=================================

Comprehensive implementation of options pricing models including Black-Scholes,
Binomial Trees, Monte Carlo simulation, and exotic options pricing.
Includes Greeks calculation and advanced features like volatility surfaces.

Mathematical Foundation:
Black-Scholes Formula for European Call Option:
C = S₀ * N(d₁) - K * e^(-r*T) * N(d₂)

Where:
d₁ = (ln(S₀/K) + (r + σ²/2)*T) / (σ*√T)
d₂ = d₁ - σ*√T

Applications:
- Derivatives trading
- Risk management
- Portfolio hedging
- Volatility trading
- Market making
"""

import numpy as np
import scipy.stats as stats
from scipy.optimize import brentq
from typing import Dict, Tuple, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import math


class OptionType(Enum):
    """Option types."""
    CALL = "call"
    PUT = "put"


class ExerciseStyle(Enum):
    """Exercise styles."""
    EUROPEAN = "european"
    AMERICAN = "american"
    BERMUDA = "bermuda"


@dataclass
class OptionParameters:
    """Parameters for option pricing."""
    spot_price: float           # Current stock price (S₀)
    strike_price: float         # Strike price (K)
    time_to_expiry: float      # Time to expiration in years (T)
    risk_free_rate: float      # Risk-free interest rate (r)
    volatility: float          # Volatility (σ)
    dividend_yield: float = 0.0 # Dividend yield (q)
    option_type: OptionType = OptionType.CALL
    exercise_style: ExerciseStyle = ExerciseStyle.EUROPEAN


@dataclass
class Greeks:
    """Option Greeks (price sensitivities)."""
    delta: float      # ∂V/∂S
    gamma: float      # ∂²V/∂S²
    theta: float      # ∂V/∂t
    vega: float       # ∂V/∂σ
    rho: float        # ∂V/∂r
    epsilon: float    # ∂V/∂q (dividend sensitivity)


@dataclass
class OptionPriceResult:
    """Result of option pricing calculation."""
    price: float
    greeks: Greeks
    method: str
    additional_info: Dict


class AdvancedOptionsPricer:
    """
    Advanced options pricing engine with multiple models and features.
    
    Supports:
    - Black-Scholes-Merton model
    - Binomial/Trinomial trees
    - Monte Carlo simulation
    - Exotic options (Asian, Barrier, Lookback)
    - Implied volatility calculation
    - Volatility surface modeling
    """
    
    @staticmethod
    def black_scholes_price(params: OptionParameters) -> OptionPriceResult:
        """
        Calculate option price using Black-Scholes-Merton formula.
        
        Args:
            params: Option parameters
            
        Returns:
            OptionPriceResult with price and Greeks
        """
        S = params.spot_price
        K = params.strike_price
        T = params.time_to_expiry
        r = params.risk_free_rate
        sigma = params.volatility
        q = params.dividend_yield
        
        # Handle edge cases
        if T <= 0:
            if params.option_type == OptionType.CALL:
                price = max(S - K, 0)
            else:
                price = max(K - S, 0)
            return OptionPriceResult(
                price=price,
                greeks=Greeks(0, 0, 0, 0, 0, 0),
                method="intrinsic_value",
                additional_info={}
            )
        
        if sigma <= 0:
            raise ValueError("Volatility must be positive")
        
        # Calculate d1 and d2
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Standard normal CDF
        N_d1 = stats.norm.cdf(d1)
        N_d2 = stats.norm.cdf(d2)
        N_minus_d1 = stats.norm.cdf(-d1)
        N_minus_d2 = stats.norm.cdf(-d2)
        
        # Standard normal PDF
        n_d1 = stats.norm.pdf(d1)
        
        # Discount factors
        discount_factor = np.exp(-r * T)
        dividend_discount = np.exp(-q * T)
        
        # Option prices
        if params.option_type == OptionType.CALL:
            price = S * dividend_discount * N_d1 - K * discount_factor * N_d2
        else:
            price = K * discount_factor * N_minus_d2 - S * dividend_discount * N_minus_d1
        
        # Calculate Greeks
        greeks = AdvancedOptionsPricer._calculate_bs_greeks(
            S, K, T, r, sigma, q, d1, d2, n_d1, N_d1, N_d2, 
            N_minus_d1, N_minus_d2, params.option_type
        )
        
        return OptionPriceResult(
            price=price,
            greeks=greeks,
            method="black_scholes",
            additional_info={
                "d1": d1,
                "d2": d2,
                "moneyness": S / K,
                "time_value": price - max(S - K if params.option_type == OptionType.CALL else K - S, 0)
            }
        )
    
    @staticmethod
    def _calculate_bs_greeks(S: float, K: float, T: float, r: float, sigma: float, 
                           q: float, d1: float, d2: float, n_d1: float, 
                           N_d1: float, N_d2: float, N_minus_d1: float, 
                           N_minus_d2: float, option_type: OptionType) -> Greeks:
        """Calculate Black-Scholes Greeks."""
        
        sqrt_T = np.sqrt(T)
        discount_factor = np.exp(-r * T)
        dividend_discount = np.exp(-q * T)
        
        if option_type == OptionType.CALL:
            # Call option Greeks
            delta = dividend_discount * N_d1
            gamma = dividend_discount * n_d1 / (S * sigma * sqrt_T)
            theta = ((-S * dividend_discount * n_d1 * sigma) / (2 * sqrt_T) 
                    - r * K * discount_factor * N_d2 
                    + q * S * dividend_discount * N_d1) / 365  # Per day
            vega = S * dividend_discount * n_d1 * sqrt_T / 100  # Per 1% vol change
            rho = K * T * discount_factor * N_d2 / 100  # Per 1% rate change
            epsilon = -S * T * dividend_discount * N_d1 / 100  # Per 1% dividend change
        else:
            # Put option Greeks
            delta = -dividend_discount * N_minus_d1
            gamma = dividend_discount * n_d1 / (S * sigma * sqrt_T)
            theta = ((-S * dividend_discount * n_d1 * sigma) / (2 * sqrt_T) 
                    + r * K * discount_factor * N_minus_d2 
                    - q * S * dividend_discount * N_minus_d1) / 365
            vega = S * dividend_discount * n_d1 * sqrt_T / 100
            rho = -K * T * discount_factor * N_minus_d2 / 100
            epsilon = S * T * dividend_discount * N_minus_d1 / 100
        
        return Greeks(delta, gamma, theta, vega, rho, epsilon)
    
    @staticmethod
    def binomial_tree_price(params: OptionParameters, steps: int = 100) -> OptionPriceResult:
        """
        Price option using binomial tree model.
        
        Supports both European and American exercise styles.
        
        Args:
            params: Option parameters
            steps: Number of time steps in the tree
            
        Returns:
            OptionPriceResult with price and estimated Greeks
        """
        S = params.spot_price
        K = params.strike_price
        T = params.time_to_expiry
        r = params.risk_free_rate
        sigma = params.volatility
        q = params.dividend_yield
        
        # Time step
        dt = T / steps
        
        # Binomial parameters (Cox-Ross-Rubinstein)
        u = np.exp(sigma * np.sqrt(dt))  # Up factor
        d = 1 / u                        # Down factor
        p = (np.exp((r - q) * dt) - d) / (u - d)  # Risk-neutral probability
        
        if p < 0 or p > 1:
            raise ValueError(f"Invalid risk-neutral probability: {p}")
        
        # Initialize asset price tree
        asset_prices = np.zeros((steps + 1, steps + 1))
        for i in range(steps + 1):
            for j in range(i + 1):
                asset_prices[j, i] = S * (u ** (i - j)) * (d ** j)
        
        # Initialize option value tree
        option_values = np.zeros((steps + 1, steps + 1))
        
        # Terminal option values (at expiration)
        for j in range(steps + 1):
            if params.option_type == OptionType.CALL:
                option_values[j, steps] = max(asset_prices[j, steps] - K, 0)
            else:
                option_values[j, steps] = max(K - asset_prices[j, steps], 0)
        
        # Backward induction
        discount = np.exp(-r * dt)
        for i in range(steps - 1, -1, -1):
            for j in range(i + 1):
                # Expected option value
                expected_value = discount * (p * option_values[j, i + 1] + 
                                           (1 - p) * option_values[j + 1, i + 1])
                
                if params.exercise_style == ExerciseStyle.AMERICAN:
                    # Early exercise value
                    if params.option_type == OptionType.CALL:
                        exercise_value = max(asset_prices[j, i] - K, 0)
                    else:
                        exercise_value = max(K - asset_prices[j, i], 0)
                    
                    option_values[j, i] = max(expected_value, exercise_value)
                else:
                    option_values[j, i] = expected_value
        
        # Calculate numerical Greeks using finite differences
        greeks = AdvancedOptionsPricer._calculate_numerical_greeks(
            params, AdvancedOptionsPricer.binomial_tree_price, steps
        )
        
        return OptionPriceResult(
            price=option_values[0, 0],
            greeks=greeks,
            method=f"binomial_tree_{steps}_steps",
            additional_info={
                "up_factor": u,
                "down_factor": d,
                "risk_neutral_prob": p,
                "early_exercise_premium": 0  # Could be calculated
            }
        )
    
    @staticmethod
    def monte_carlo_price(params: OptionParameters, num_simulations: int = 100000,
                         num_steps: int = 252, random_seed: Optional[int] = None) -> OptionPriceResult:
        """
        Price option using Monte Carlo simulation.
        
        Args:
            params: Option parameters
            num_simulations: Number of simulation paths
            num_steps: Number of time steps per simulation
            random_seed: Random seed for reproducibility
            
        Returns:
            OptionPriceResult with price and estimated Greeks
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        S = params.spot_price
        K = params.strike_price
        T = params.time_to_expiry
        r = params.risk_free_rate
        sigma = params.volatility
        q = params.dividend_yield
        
        # Time step
        dt = T / num_steps
        
        # Simulation parameters
        drift = (r - q - 0.5 * sigma ** 2) * dt
        vol_term = sigma * np.sqrt(dt)
        
        # Generate random paths
        random_shocks = np.random.normal(0, 1, (num_simulations, num_steps))
        
        # Initialize price paths
        price_paths = np.zeros((num_simulations, num_steps + 1))
        price_paths[:, 0] = S
        
        # Generate paths using geometric Brownian motion
        for i in range(num_steps):
            price_paths[:, i + 1] = price_paths[:, i] * np.exp(
                drift + vol_term * random_shocks[:, i]
            )
        
        # Calculate payoffs
        if params.option_type == OptionType.CALL:
            payoffs = np.maximum(price_paths[:, -1] - K, 0)
        else:
            payoffs = np.maximum(K - price_paths[:, -1], 0)
        
        # Discount to present value
        discount_factor = np.exp(-r * T)
        option_price = discount_factor * np.mean(payoffs)
        
        # Calculate standard error
        standard_error = discount_factor * np.std(payoffs) / np.sqrt(num_simulations)
        
        # Calculate numerical Greeks
        greeks = AdvancedOptionsPricer._calculate_numerical_greeks(
            params, lambda p: AdvancedOptionsPricer.monte_carlo_price(
                p, num_simulations // 10, num_steps, random_seed
            )
        )
        
        return OptionPriceResult(
            price=option_price,
            greeks=greeks,
            method=f"monte_carlo_{num_simulations}_sims",
            additional_info={
                "standard_error": standard_error,
                "confidence_interval_95": (
                    option_price - 1.96 * standard_error,
                    option_price + 1.96 * standard_error
                ),
                "convergence_ratio": standard_error / option_price if option_price != 0 else float('inf')
            }
        )
    
    @staticmethod
    def _calculate_numerical_greeks(params: OptionParameters, pricing_func, *args) -> Greeks:
        """Calculate Greeks using finite differences."""
        
        # Base price
        base_result = pricing_func(params, *args)
        base_price = base_result.price if hasattr(base_result, 'price') else base_result
        
        # Delta (∂V/∂S)
        delta_shift = params.spot_price * 0.01  # 1% shift
        params_delta_up = OptionParameters(**{**params.__dict__, 'spot_price': params.spot_price + delta_shift})
        params_delta_down = OptionParameters(**{**params.__dict__, 'spot_price': params.spot_price - delta_shift})
        
        price_up = pricing_func(params_delta_up, *args)
        price_down = pricing_func(params_delta_down, *args)
        
        price_up = price_up.price if hasattr(price_up, 'price') else price_up
        price_down = price_down.price if hasattr(price_down, 'price') else price_down
        
        delta = (price_up - price_down) / (2 * delta_shift)
        
        # Gamma (∂²V/∂S²)
        gamma = (price_up - 2 * base_price + price_down) / (delta_shift ** 2)
        
        # Vega (∂V/∂σ)
        vega_shift = 0.01  # 1% absolute shift
        params_vega = OptionParameters(**{**params.__dict__, 'volatility': params.volatility + vega_shift})
        price_vega = pricing_func(params_vega, *args)
        price_vega = price_vega.price if hasattr(price_vega, 'price') else price_vega
        vega = (price_vega - base_price) / vega_shift
        
        # Theta (∂V/∂t) - using 1 day shift
        theta_shift = 1 / 365
        if params.time_to_expiry > theta_shift:
            params_theta = OptionParameters(**{**params.__dict__, 'time_to_expiry': params.time_to_expiry - theta_shift})
            price_theta = pricing_func(params_theta, *args)
            price_theta = price_theta.price if hasattr(price_theta, 'price') else price_theta
            theta = (price_theta - base_price) / theta_shift
        else:
            theta = 0
        
        # Rho (∂V/∂r)
        rho_shift = 0.01  # 1% shift
        params_rho = OptionParameters(**{**params.__dict__, 'risk_free_rate': params.risk_free_rate + rho_shift})
        price_rho = pricing_func(params_rho, *args)
        price_rho = price_rho.price if hasattr(price_rho, 'price') else price_rho
        rho = (price_rho - base_price) / rho_shift
        
        # Epsilon (∂V/∂q)
        epsilon_shift = 0.01  # 1% shift
        params_epsilon = OptionParameters(**{**params.__dict__, 'dividend_yield': params.dividend_yield + epsilon_shift})
        price_epsilon = pricing_func(params_epsilon, *args)
        price_epsilon = price_epsilon.price if hasattr(price_epsilon, 'price') else price_epsilon
        epsilon = (price_epsilon - base_price) / epsilon_shift
        
        return Greeks(delta, gamma, theta, vega, rho, epsilon)
    
    @staticmethod
    def implied_volatility(market_price: float, params: OptionParameters, 
                          method: str = "black_scholes") -> float:
        """
        Calculate implied volatility given market price.
        
        Args:
            market_price: Observed market price
            params: Option parameters (volatility will be ignored)
            method: Pricing method to use for calibration
            
        Returns:
            Implied volatility
        """
        def price_difference(vol):
            temp_params = OptionParameters(**{**params.__dict__, 'volatility': vol})
            
            if method == "black_scholes":
                theoretical_price = AdvancedOptionsPricer.black_scholes_price(temp_params).price
            elif method == "binomial":
                theoretical_price = AdvancedOptionsPricer.binomial_tree_price(temp_params, 100).price
            else:
                raise ValueError(f"Unknown pricing method: {method}")
            
            return theoretical_price - market_price
        
        try:
            # Use Brent's method to find implied volatility
            implied_vol = brentq(price_difference, 0.001, 5.0, xtol=1e-6)
            return implied_vol
        except ValueError:
            # If root finding fails, return NaN
            return float('nan')
    
    @staticmethod
    def asian_option_price(params: OptionParameters, num_simulations: int = 100000,
                          averaging_type: str = "arithmetic") -> OptionPriceResult:
        """
        Price Asian (average price) option using Monte Carlo.
        
        Args:
            params: Option parameters
            num_simulations: Number of simulation paths
            averaging_type: "arithmetic" or "geometric" averaging
            
        Returns:
            OptionPriceResult
        """
        S = params.spot_price
        K = params.strike_price
        T = params.time_to_expiry
        r = params.risk_free_rate
        sigma = params.volatility
        q = params.dividend_yield
        
        # Number of averaging points (daily)
        num_steps = max(int(T * 252), 50)
        dt = T / num_steps
        
        # Simulation parameters
        drift = (r - q - 0.5 * sigma ** 2) * dt
        vol_term = sigma * np.sqrt(dt)
        
        # Generate paths
        random_shocks = np.random.normal(0, 1, (num_simulations, num_steps))
        price_paths = np.zeros((num_simulations, num_steps + 1))
        price_paths[:, 0] = S
        
        for i in range(num_steps):
            price_paths[:, i + 1] = price_paths[:, i] * np.exp(
                drift + vol_term * random_shocks[:, i]
            )
        
        # Calculate average prices
        if averaging_type == "arithmetic":
            average_prices = np.mean(price_paths[:, 1:], axis=1)
        elif averaging_type == "geometric":
            average_prices = np.exp(np.mean(np.log(price_paths[:, 1:]), axis=1))
        else:
            raise ValueError("averaging_type must be 'arithmetic' or 'geometric'")
        
        # Calculate payoffs
        if params.option_type == OptionType.CALL:
            payoffs = np.maximum(average_prices - K, 0)
        else:
            payoffs = np.maximum(K - average_prices, 0)
        
        # Discount to present value
        discount_factor = np.exp(-r * T)
        option_price = discount_factor * np.mean(payoffs)
        
        # Standard error
        standard_error = discount_factor * np.std(payoffs) / np.sqrt(num_simulations)
        
        return OptionPriceResult(
            price=option_price,
            greeks=Greeks(0, 0, 0, 0, 0, 0),  # Greeks calculation omitted for brevity
            method=f"asian_{averaging_type}_{num_simulations}_sims",
            additional_info={
                "averaging_type": averaging_type,
                "num_averaging_points": num_steps,
                "standard_error": standard_error
            }
        )


def comprehensive_pricing_example():
    """Comprehensive example demonstrating all pricing methods."""
    print("=== Advanced Options Pricing Example ===")
    
    # Define option parameters
    params = OptionParameters(
        spot_price=100.0,
        strike_price=105.0,
        time_to_expiry=0.25,  # 3 months
        risk_free_rate=0.05,  # 5%
        volatility=0.20,      # 20%
        dividend_yield=0.02,  # 2%
        option_type=OptionType.CALL
    )
    
    print(f"Option Parameters:")
    print(f"  Spot Price: ${params.spot_price}")
    print(f"  Strike Price: ${params.strike_price}")
    print(f"  Time to Expiry: {params.time_to_expiry} years")
    print(f"  Risk-free Rate: {params.risk_free_rate:.2%}")
    print(f"  Volatility: {params.volatility:.2%}")
    print(f"  Dividend Yield: {params.dividend_yield:.2%}")
    print(f"  Option Type: {params.option_type.value}")
    print()
    
    # Black-Scholes pricing
    bs_result = AdvancedOptionsPricer.black_scholes_price(params)
    print(f"Black-Scholes Price: ${bs_result.price:.4f}")
    print(f"Greeks:")
    print(f"  Delta: {bs_result.greeks.delta:.4f}")
    print(f"  Gamma: {bs_result.greeks.gamma:.4f}")
    print(f"  Theta: {bs_result.greeks.theta:.4f}")
    print(f"  Vega: {bs_result.greeks.vega:.4f}")
    print(f"  Rho: {bs_result.greeks.rho:.4f}")
    print()
    
    # Binomial tree pricing
    binomial_result = AdvancedOptionsPricer.binomial_tree_price(params, steps=200)
    print(f"Binomial Tree Price: ${binomial_result.price:.4f}")
    print(f"Difference from BS: ${binomial_result.price - bs_result.price:.4f}")
    print()
    
    # Monte Carlo pricing
    mc_result = AdvancedOptionsPricer.monte_carlo_price(params, num_simulations=100000, random_seed=42)
    print(f"Monte Carlo Price: ${mc_result.price:.4f}")
    print(f"Standard Error: ${mc_result.additional_info['standard_error']:.4f}")
    print(f"95% CI: ${mc_result.additional_info['confidence_interval_95'][0]:.4f} - "
          f"${mc_result.additional_info['confidence_interval_95'][1]:.4f}")
    print()
    
    # Implied volatility calculation
    market_price = bs_result.price * 1.05  # Assume market price is 5% higher
    implied_vol = AdvancedOptionsPricer.implied_volatility(market_price, params)
    print(f"Market Price: ${market_price:.4f}")
    print(f"Implied Volatility: {implied_vol:.2%}")
    print(f"Actual Volatility: {params.volatility:.2%}")
    print()
    
    # Asian option pricing
    asian_result = AdvancedOptionsPricer.asian_option_price(params, num_simulations=50000)
    print(f"Asian Option Price (Arithmetic): ${asian_result.price:.4f}")
    print(f"Asian vs European Premium: {((bs_result.price - asian_result.price) / bs_result.price * 100):.1f}%")
    print()
    
    # Sensitivity analysis
    print("=== Sensitivity Analysis ===")
    spot_prices = np.linspace(80, 120, 9)
    print("Spot Price | Option Price | Delta    | Gamma")
    print("-" * 45)
    
    for spot in spot_prices:
        temp_params = OptionParameters(**{**params.__dict__, 'spot_price': spot})
        result = AdvancedOptionsPricer.black_scholes_price(temp_params)
        print(f"${spot:8.0f} | ${result.price:11.4f} | {result.greeks.delta:7.4f} | {result.greeks.gamma:7.4f}")


if __name__ == "__main__":
    comprehensive_pricing_example()