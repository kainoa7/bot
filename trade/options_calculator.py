"""
Advanced options calculations including Greeks and probability analysis
"""
import numpy as np
from scipy import stats
from typing import Dict
import math


class OptionsCalculator:
    """Calculate options Greeks and probabilities"""
    
    def __init__(self, stock_price: float, strike: float, time_to_expiry: float, 
                 volatility: float, risk_free_rate: float = 0.05):
        """
        Initialize options calculator
        
        Args:
            stock_price: Current stock price
            strike: Strike price
            time_to_expiry: Time to expiration in years
            volatility: Implied volatility (as decimal, e.g., 0.30 for 30%)
            risk_free_rate: Risk-free interest rate (default 5%)
        """
        self.S = stock_price
        self.K = strike
        self.T = time_to_expiry
        self.sigma = volatility
        self.r = risk_free_rate
        
    def _d1(self):
        """Calculate d1 for Black-Scholes"""
        return (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / \
               (self.sigma * np.sqrt(self.T))
    
    def _d2(self):
        """Calculate d2 for Black-Scholes"""
        return self._d1() - self.sigma * np.sqrt(self.T)
    
    def call_price(self) -> float:
        """Calculate call option theoretical price"""
        if self.T <= 0:
            return max(self.S - self.K, 0)
        
        d1 = self._d1()
        d2 = self._d2()
        
        price = (self.S * stats.norm.cdf(d1)) - \
                (self.K * np.exp(-self.r * self.T) * stats.norm.cdf(d2))
        return price
    
    def put_price(self) -> float:
        """Calculate put option theoretical price"""
        if self.T <= 0:
            return max(self.K - self.S, 0)
        
        d1 = self._d1()
        d2 = self._d2()
        
        price = (self.K * np.exp(-self.r * self.T) * stats.norm.cdf(-d2)) - \
                (self.S * stats.norm.cdf(-d1))
        return price
    
    def delta(self, option_type: str = 'call') -> float:
        """
        Calculate Delta - rate of change of option price with respect to stock price
        
        Args:
            option_type: 'call' or 'put'
            
        Returns:
            Delta value
        """
        if self.T <= 0:
            return 1.0 if option_type == 'call' and self.S > self.K else 0.0
        
        d1 = self._d1()
        
        if option_type == 'call':
            return stats.norm.cdf(d1)
        else:
            return stats.norm.cdf(d1) - 1
    
    def gamma(self) -> float:
        """Calculate Gamma - rate of change of delta"""
        if self.T <= 0:
            return 0.0
        
        d1 = self._d1()
        return stats.norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))
    
    def theta(self, option_type: str = 'call') -> float:
        """
        Calculate Theta - time decay (per day)
        
        Args:
            option_type: 'call' or 'put'
            
        Returns:
            Theta value (daily)
        """
        if self.T <= 0:
            return 0.0
        
        d1 = self._d1()
        d2 = self._d2()
        
        term1 = -(self.S * stats.norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T))
        
        if option_type == 'call':
            term2 = self.r * self.K * np.exp(-self.r * self.T) * stats.norm.cdf(d2)
            theta_annual = term1 - term2
        else:
            term2 = self.r * self.K * np.exp(-self.r * self.T) * stats.norm.cdf(-d2)
            theta_annual = term1 + term2
        
        return theta_annual / 365  # Convert to daily
    
    def vega(self) -> float:
        """Calculate Vega - sensitivity to volatility (per 1% change)"""
        if self.T <= 0:
            return 0.0
        
        d1 = self._d1()
        return self.S * stats.norm.pdf(d1) * np.sqrt(self.T) / 100
    
    def probability_profit(self, option_type: str = 'call', premium: float = 0) -> float:
        """
        Calculate probability of profit at expiration
        
        Args:
            option_type: 'call' or 'put'
            premium: Option premium paid
            
        Returns:
            Probability of profit (0-1)
        """
        if self.T <= 0:
            if option_type == 'call':
                return 1.0 if self.S > self.K + premium else 0.0
            else:
                return 1.0 if self.S < self.K - premium else 0.0
        
        # Breakeven points
        if option_type == 'call':
            breakeven = self.K + premium
        else:
            breakeven = self.K - premium
        
        # Calculate probability stock will be past breakeven
        z_score = (np.log(breakeven / self.S) - (self.r - 0.5 * self.sigma**2) * self.T) / \
                  (self.sigma * np.sqrt(self.T))
        
        if option_type == 'call':
            prob = 1 - stats.norm.cdf(z_score)
        else:
            prob = stats.norm.cdf(z_score)
        
        return prob
    
    def max_profit(self, option_type: str = 'call', premium: float = 0, 
                   contracts: int = 1) -> float:
        """Calculate maximum profit"""
        if option_type == 'call':
            return float('inf')  # Unlimited for calls
        else:
            return (self.K - premium) * 100 * contracts
    
    def max_loss(self, premium: float = 0, contracts: int = 1) -> float:
        """Calculate maximum loss (buying options)"""
        return premium * 100 * contracts
    
    def breakeven_price(self, option_type: str = 'call', premium: float = 0) -> float:
        """Calculate breakeven stock price at expiration"""
        if option_type == 'call':
            return self.K + premium
        else:
            return self.K - premium
    
    def get_all_greeks(self, option_type: str = 'call') -> Dict:
        """Get all Greeks and metrics"""
        return {
            'delta': self.delta(option_type),
            'gamma': self.gamma(),
            'theta': self.theta(option_type),
            'vega': self.vega(),
            'theoretical_price': self.call_price() if option_type == 'call' else self.put_price(),
        }


def calculate_position_sizing(account_size: float, risk_percent: float, 
                              max_loss_per_contract: float) -> int:
    """
    Calculate optimal number of contracts based on risk management
    
    Args:
        account_size: Total account value
        risk_percent: Percentage of account to risk (e.g., 0.02 for 2%)
        max_loss_per_contract: Maximum loss per contract
        
    Returns:
        Number of contracts
    """
    max_risk_dollars = account_size * risk_percent
    contracts = int(max_risk_dollars / max_loss_per_contract)
    return max(1, contracts)  # At least 1 contract


def calculate_iv_percentile(current_iv: float, historical_ivs: list) -> float:
    """
    Calculate IV percentile (where current IV ranks historically)
    
    Args:
        current_iv: Current implied volatility
        historical_ivs: List of historical IV values
        
    Returns:
        Percentile (0-100)
    """
    if not historical_ivs:
        return 50.0
    
    below = sum(1 for iv in historical_ivs if iv < current_iv)
    percentile = (below / len(historical_ivs)) * 100
    return percentile


if __name__ == "__main__":
    # Example usage
    calc = OptionsCalculator(
        stock_price=150,
        strike=155,
        time_to_expiry=30/365,  # 30 days
        volatility=0.35  # 35% IV
    )
    
    print("Call Option Greeks:")
    greeks = calc.get_all_greeks('call')
    for greek, value in greeks.items():
        print(f"  {greek}: {value:.4f}")
    
    print(f"\nProbability of Profit: {calc.probability_profit('call', premium=3.50):.2%}")
    print(f"Breakeven Price: ${calc.breakeven_price('call', premium=3.50):.2f}")
