import numpy as np
from scipy.stats import norm

def calculate_option_price(
    spot_price: float | np.ndarray,
    strike_price: float,
    time_to_expiry_years: float,
    risk_free_rate: float,
    volatility: float,
    option_type: str
) -> float | np.ndarray:
    """
    Calculate the Black-Scholes price for a European option.
    Supports vectorized inputs for spot_price.
    
    :param spot_price: Current price of the underlying asset (scalar or array)
    :param strike_price: Strike price of the option
    :param time_to_expiry_years: Time to expiration in years
    :param risk_free_rate: Annualized risk-free interest rate (e.g., 0.05)
    :param volatility: Annualized volatility (sigma) (e.g., 0.2)
    :param option_type: 'call' or 'put'
    :return: Option price (scalar or array)
    """
    S = spot_price
    K = strike_price
    T = time_to_expiry_years
    r = risk_free_rate
    sigma = volatility
    
    # Handle expiration case
    if T <= 1e-8: # Epsilon for float comparison
        if option_type == 'call':
            return np.maximum(0.0, S - K)
        else:
            return np.maximum(0.0, K - S)
            
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
        
    return price
