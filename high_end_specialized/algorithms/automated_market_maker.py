"""
Automated Market Maker (AMM) Algorithm
=====================================

Implementation of advanced AMM algorithms including Constant Product (Uniswap),
Constant Sum, Constant Mean (Balancer), and StableSwap (Curve) formulas
with impermanent loss calculation and optimal liquidity strategies.

Mathematical Foundation:
Constant Product: x * y = k
Constant Sum: x + y = k  
Constant Mean: (x/w₁)^w₁ * (y/w₂)^w₂ = k
StableSwap: An² ∑xᵢ + D = ADn + D^(n+1)/(n^n ∏xᵢ)

Applications:
- Decentralized exchanges
- Liquidity pool optimization
- Arbitrage detection
- Yield farming strategies
- DeFi protocol design
"""

import numpy as np
import scipy.optimize as opt
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
from enum import Enum
import math


class AMMType(Enum):
    """Types of AMM algorithms."""
    CONSTANT_PRODUCT = "constant_product"      # Uniswap v2
    CONSTANT_SUM = "constant_sum"              # Simple sum
    CONSTANT_MEAN = "constant_mean"            # Balancer
    STABLESWAP = "stableswap"                  # Curve
    CONCENTRATED_LIQUIDITY = "concentrated"    # Uniswap v3


@dataclass
class PoolState:
    """Current state of a liquidity pool."""
    reserves_x: float
    reserves_y: float
    total_liquidity: float
    fee_rate: float = 0.003  # 0.3% default fee
    weights: Tuple[float, float] = (0.5, 0.5)  # For weighted pools
    
    @property
    def price(self) -> float:
        """Current price of token Y in terms of token X."""
        return self.reserves_y / self.reserves_x
    
    @property
    def k_value(self) -> float:
        """Invariant K value for constant product."""
        return self.reserves_x * self.reserves_y


class AMMAlgorithm:
    """
    Advanced Automated Market Maker with multiple algorithm implementations.
    """
    
    def __init__(self, amm_type: AMMType, amplification_factor: float = 100):
        self.amm_type = amm_type
        self.amplification_factor = amplification_factor  # For StableSwap
    
    def calculate_swap_output(self, pool: PoolState, amount_in: float, 
                            token_in: str = "x") -> Tuple[float, float]:
        """
        Calculate output amount for a given input amount.
        
        Args:
            pool: Current pool state
            amount_in: Input amount
            token_in: "x" or "y" indicating input token
            
        Returns:
            (amount_out, new_price) tuple
        """
        if self.amm_type == AMMType.CONSTANT_PRODUCT:
            return self._constant_product_swap(pool, amount_in, token_in)
        elif self.amm_type == AMMType.CONSTANT_SUM:
            return self._constant_sum_swap(pool, amount_in, token_in)
        elif self.amm_type == AMMType.CONSTANT_MEAN:
            return self._constant_mean_swap(pool, amount_in, token_in)
        elif self.amm_type == AMMType.STABLESWAP:
            return self._stableswap_swap(pool, amount_in, token_in)
        else:
            raise ValueError(f"Unsupported AMM type: {self.amm_type}")
    
    def _constant_product_swap(self, pool: PoolState, amount_in: float, 
                              token_in: str) -> Tuple[float, float]:
        """Constant Product Formula: x * y = k"""
        
        if token_in == "x":
            # Selling X for Y
            x_new = pool.reserves_x + amount_in * (1 - pool.fee_rate)
            y_new = pool.k_value / x_new
            amount_out = pool.reserves_y - y_new
            new_price = y_new / x_new
        else:
            # Selling Y for X
            y_new = pool.reserves_y + amount_in * (1 - pool.fee_rate)
            x_new = pool.k_value / y_new
            amount_out = pool.reserves_x - x_new
            new_price = y_new / x_new
        
        return amount_out, new_price
    
    def _constant_sum_swap(self, pool: PoolState, amount_in: float, 
                          token_in: str) -> Tuple[float, float]:
        """Constant Sum Formula: x + y = k"""
        
        # In constant sum, price is always 1:1 (ignoring fees)
        amount_out = amount_in * (1 - pool.fee_rate)
        
        if token_in == "x":
            new_price = (pool.reserves_y - amount_out) / (pool.reserves_x + amount_in)
        else:
            new_price = (pool.reserves_y + amount_in) / (pool.reserves_x - amount_out)
        
        return amount_out, new_price
    
    def _constant_mean_swap(self, pool: PoolState, amount_in: float, 
                           token_in: str) -> Tuple[float, float]:
        """Constant Mean Formula (Balancer): (x/w₁)^w₁ * (y/w₂)^w₂ = k"""
        
        w_x, w_y = pool.weights
        
        if token_in == "x":
            # Calculate new Y reserve
            x_in = amount_in * (1 - pool.fee_rate)
            x_new = pool.reserves_x + x_in
            
            # Solve for y_new using the invariant
            ratio = (pool.reserves_x / x_new) ** (w_x / w_y)
            y_new = pool.reserves_y * ratio
            amount_out = pool.reserves_y - y_new
            new_price = y_new / x_new
        else:
            # Calculate new X reserve
            y_in = amount_in * (1 - pool.fee_rate)
            y_new = pool.reserves_y + y_in
            
            ratio = (pool.reserves_y / y_new) ** (w_y / w_x)
            x_new = pool.reserves_x * ratio
            amount_out = pool.reserves_x - x_new
            new_price = y_new / x_new
        
        return amount_out, new_price
    
    def _stableswap_swap(self, pool: PoolState, amount_in: float, 
                        token_in: str) -> Tuple[float, float]:
        """StableSwap Formula (Curve): An² ∑xᵢ + D = ADn + D^(n+1)/(n^n ∏xᵢ)"""
        
        A = self.amplification_factor
        n = 2  # Number of tokens
        
        x = pool.reserves_x
        y = pool.reserves_y
        
        # Calculate D (total value)
        D = self._calculate_D([x, y], A)
        
        if token_in == "x":
            x_new = x + amount_in * (1 - pool.fee_rate)
            y_new = self._solve_y(x_new, D, A)
            amount_out = y - y_new
            new_price = y_new / x_new
        else:
            y_new = y + amount_in * (1 - pool.fee_rate)
            x_new = self._solve_y(y_new, D, A)
            amount_out = x - x_new
            new_price = y_new / x_new
        
        return amount_out, new_price
    
    def _calculate_D(self, balances: List[float], A: float) -> float:
        """Calculate D for StableSwap invariant."""
        n = len(balances)
        S = sum(balances)
        
        if S == 0:
            return 0
        
        D = S
        Ann = A * n
        
        for _ in range(255):  # Newton's method iterations
            D_P = D
            for balance in balances:
                D_P = D_P * D // (n * balance)
            
            D_prev = D
            D = (Ann * S + D_P * n) * D // ((Ann - 1) * D + (n + 1) * D_P)
            
            if abs(D - D_prev) <= 1:
                break
        
        return D
    
    def _solve_y(self, x: float, D: float, A: float) -> float:
        """Solve for y given x, D, and A in StableSwap."""
        n = 2
        Ann = A * n
        c = D * D // (2 * x) * D // (4 * Ann)
        b = x + D // Ann - D
        
        y = D
        for _ in range(255):
            y_prev = y
            y = (y * y + c) // (2 * y + b - D)
            
            if abs(y - y_prev) <= 1:
                break
        
        return y
    
    def calculate_impermanent_loss(self, initial_pool: PoolState, 
                                  current_price_ratio: float) -> Dict[str, float]:
        """
        Calculate impermanent loss for LP providers.
        
        Args:
            initial_pool: Pool state at liquidity provision
            current_price_ratio: Current price / Initial price
            
        Returns:
            Dictionary with impermanent loss metrics
        """
        initial_price = initial_pool.price
        current_price = initial_price * current_price_ratio
        
        # Calculate what LP would have if they held tokens vs provided liquidity
        if self.amm_type == AMMType.CONSTANT_PRODUCT:
            # For constant product, IL = 2√(ratio)/(1 + ratio) - 1
            sqrt_ratio = math.sqrt(current_price_ratio)
            il_percentage = (2 * sqrt_ratio) / (1 + current_price_ratio) - 1
            
            # Absolute IL value
            initial_value = initial_pool.reserves_x + initial_pool.reserves_y * initial_price
            current_lp_value = initial_value * (1 + il_percentage)
            hold_value = initial_pool.reserves_x + initial_pool.reserves_y * current_price
            
            absolute_il = hold_value - current_lp_value
        else:
            # Simplified calculation for other AMM types
            il_percentage = 0  # Other AMMs may have different IL characteristics
            absolute_il = 0
        
        return {
            "impermanent_loss_percentage": il_percentage,
            "absolute_impermanent_loss": absolute_il,
            "price_ratio": current_price_ratio,
            "initial_price": initial_price,
            "current_price": current_price
        }
    
    def find_optimal_arbitrage(self, pool1: PoolState, pool2: PoolState, 
                              max_amount: float = 1000) -> Dict[str, float]:
        """
        Find optimal arbitrage opportunity between two pools.
        
        Args:
            pool1: First pool state
            pool2: Second pool state  
            max_amount: Maximum arbitrage amount to consider
            
        Returns:
            Dictionary with arbitrage opportunity details
        """
        
        def arbitrage_profit(amount: float) -> float:
            """Calculate profit from arbitraging given amount."""
            if amount <= 0:
                return 0
            
            # Buy from cheaper pool
            if pool1.price < pool2.price:
                # Buy Y from pool1, sell to pool2
                amount_out1, _ = self.calculate_swap_output(pool1, amount, "x")
                amount_out2, _ = self.calculate_swap_output(pool2, amount_out1, "y")
                profit = amount_out2 - amount
            else:
                # Buy Y from pool2, sell to pool1
                amount_out2, _ = self.calculate_swap_output(pool2, amount, "x")
                amount_out1, _ = self.calculate_swap_output(pool1, amount_out2, "y")
                profit = amount_out1 - amount
            
            return profit
        
        # Find optimal arbitrage amount using optimization
        try:
            result = opt.minimize_scalar(
                lambda x: -arbitrage_profit(x),
                bounds=(0, max_amount),
                method='bounded'
            )
            
            optimal_amount = result.x
            max_profit = -result.fun
        except:
            optimal_amount = 0
            max_profit = 0
        
        return {
            "optimal_amount": optimal_amount,
            "max_profit": max_profit,
            "pool1_price": pool1.price,
            "pool2_price": pool2.price,
            "price_difference": abs(pool1.price - pool2.price),
            "profitable": max_profit > 0
        }
    
    def calculate_liquidity_mining_rewards(self, pool: PoolState, 
                                         user_liquidity: float, 
                                         reward_rate: float,
                                         time_period: float) -> Dict[str, float]:
        """
        Calculate liquidity mining rewards for a user.
        
        Args:
            pool: Pool state
            user_liquidity: User's liquidity tokens
            reward_rate: Reward tokens per second per unit liquidity
            time_period: Time period in seconds
            
        Returns:
            Dictionary with reward calculations
        """
        
        user_share = user_liquidity / pool.total_liquidity
        total_rewards = reward_rate * time_period
        user_rewards = total_rewards * user_share
        
        # Calculate APY
        user_value = user_liquidity  # Assuming 1:1 liquidity token value
        apy = (user_rewards / user_value) * (365 * 24 * 3600 / time_period)
        
        return {
            "user_rewards": user_rewards,
            "user_share": user_share,
            "apy": apy,
            "time_period_days": time_period / (24 * 3600)
        }


def comprehensive_amm_example():
    """Comprehensive example demonstrating AMM algorithms."""
    print("=== Advanced AMM Algorithm Example ===")
    
    # Create initial pool state
    initial_pool = PoolState(
        reserves_x=1000000,  # 1M token X
        reserves_y=2000000,  # 2M token Y
        total_liquidity=1414213,  # sqrt(x * y)
        fee_rate=0.003
    )
    
    print(f"Initial Pool State:")
    print(f"  Token X reserves: {initial_pool.reserves_x:,.0f}")
    print(f"  Token Y reserves: {initial_pool.reserves_y:,.0f}")
    print(f"  Initial price (Y/X): {initial_pool.price:.6f}")
    print(f"  K value: {initial_pool.k_value:,.0f}")
    print()
    
    # Test different AMM algorithms
    amm_types = [
        AMMType.CONSTANT_PRODUCT,
        AMMType.CONSTANT_SUM,
        AMMType.CONSTANT_MEAN,
        AMMType.STABLESWAP
    ]
    
    swap_amount = 10000  # Swap 10k tokens
    
    print(f"Swap Analysis: Selling {swap_amount:,} token X")
    print("-" * 60)
    print(f"{'AMM Type':<20} {'Output':<15} {'New Price':<15} {'Price Impact':<15}")
    print("-" * 60)
    
    results = {}
    for amm_type in amm_types:
        amm = AMMAlgorithm(amm_type)
        
        try:
            if amm_type == AMMType.CONSTANT_MEAN:
                # Set equal weights for Balancer
                test_pool = PoolState(
                    initial_pool.reserves_x, initial_pool.reserves_y,
                    initial_pool.total_liquidity, initial_pool.fee_rate,
                    weights=(0.5, 0.5)
                )
            else:
                test_pool = initial_pool
            
            amount_out, new_price = amm.calculate_swap_output(test_pool, swap_amount, "x")
            price_impact = abs(new_price - initial_pool.price) / initial_pool.price
            
            results[amm_type] = {
                'amount_out': amount_out,
                'new_price': new_price,
                'price_impact': price_impact
            }
            
            print(f"{amm_type.value:<20} {amount_out:<15,.2f} {new_price:<15.6f} {price_impact:<15.4%}")
        
        except Exception as e:
            print(f"{amm_type.value:<20} Error: {str(e)}")
    
    print()
    
    # Impermanent Loss Analysis
    print("=== Impermanent Loss Analysis ===")
    
    price_ratios = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 4.0]
    amm = AMMAlgorithm(AMMType.CONSTANT_PRODUCT)
    
    print(f"Price Ratio | IL Percentage | Absolute IL")
    print("-" * 40)
    
    for ratio in price_ratios:
        il_data = amm.calculate_impermanent_loss(initial_pool, ratio)
        print(f"{ratio:10.1f} | {il_data['impermanent_loss_percentage']:12.2%} | ${il_data['absolute_impermanent_loss']:10,.0f}")
    
    print()
    
    # Arbitrage Analysis
    print("=== Arbitrage Opportunity Analysis ===")
    
    # Create two pools with different prices
    pool1 = PoolState(1000000, 2000000, 1414213, 0.003)  # Price = 2.0
    pool2 = PoolState(1000000, 2200000, 1483239, 0.003)  # Price = 2.2
    
    arbitrage_amm = AMMAlgorithm(AMMType.CONSTANT_PRODUCT)
    arb_opportunity = arbitrage_amm.find_optimal_arbitrage(pool1, pool2, max_amount=50000)
    
    print(f"Pool 1 price: {arb_opportunity['pool1_price']:.6f}")
    print(f"Pool 2 price: {arb_opportunity['pool2_price']:.6f}")
    print(f"Price difference: {arb_opportunity['price_difference']:.6f}")
    print(f"Optimal arbitrage amount: {arb_opportunity['optimal_amount']:,.2f}")
    print(f"Maximum profit: ${arb_opportunity['max_profit']:,.2f}")
    print(f"Profitable: {arb_opportunity['profitable']}")
    print()
    
    # Liquidity Mining Analysis
    print("=== Liquidity Mining Rewards ===")
    
    user_liquidity = 100000  # User has 100k liquidity tokens
    reward_rate = 10  # 10 reward tokens per second per unit liquidity
    time_periods = [1, 7, 30, 365]  # Days
    
    print(f"Time Period | User Rewards | APY")
    print("-" * 35)
    
    for days in time_periods:
        seconds = days * 24 * 3600
        rewards_data = arbitrage_amm.calculate_liquidity_mining_rewards(
            initial_pool, user_liquidity, reward_rate, seconds
        )
        
        print(f"{days:10} days | {rewards_data['user_rewards']:11,.0f} | {rewards_data['apy']:6.1%}")
    
    print()
    
    # Concentration Analysis (Simplified Uniswap V3 concept)
    print("=== Concentrated Liquidity Analysis ===")
    
    # Analyze different price ranges for concentrated liquidity
    current_price = initial_pool.price
    ranges = [
        (current_price * 0.8, current_price * 1.2),   # ±20%
        (current_price * 0.9, current_price * 1.1),   # ±10%
        (current_price * 0.95, current_price * 1.05), # ±5%
    ]
    
    print(f"Price Range | Capital Efficiency | Fee Capture")
    print("-" * 45)
    
    for i, (min_price, max_price) in enumerate(ranges):
        range_pct = ((max_price - min_price) / current_price) * 100
        # Simplified calculation - in reality, this would be more complex
        capital_efficiency = 100 / range_pct  # Inversely related to range
        fee_capture = min(1.0, 50 / range_pct)  # Higher concentration = more fees
        
        print(f"±{range_pct/2:6.1f}%   | {capital_efficiency:16.1f}x | {fee_capture:11.1%}")


if __name__ == "__main__":
    comprehensive_amm_example()