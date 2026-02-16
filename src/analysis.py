import numpy as np
import pandas as pd
from scipy import stats

def calculate_returns(prices: pd.Series) -> pd.Series:
    """Calculate daily percent change."""
    return prices.pct_change().dropna()

def fit_normal_distribution(data: pd.Series):
    """Fit a Normal distribution to the data."""
    mu, std = stats.norm.fit(data)
    return mu, std

def fit_t_distribution(data: pd.Series):
    """Fit a Student's t-distribution to the data."""
    # Returns (df, loc, scale)
    params = stats.t.fit(data)
    return params

def calculate_pdf_normal(x: np.ndarray, mu: float, std: float):
    return stats.norm.pdf(x, mu, std)

def calculate_pdf_t(x: np.ndarray, df: float, loc: float, scale: float):
    return stats.t.pdf(x, df, loc, scale)

def calculate_metrics(data: pd.Series):
    return {
        "Mean": data.mean(),
        "Std Dev": data.std(),
        "Skewness": data.skew(),
        "Kurtosis": data.kurtosis()
    }

def simulate_paths_t_dist(
    n_paths: int,
    n_days: int,
    params: tuple,
    start_price: float
) -> np.ndarray:
    """
    Generate Monte Carlo price paths using Student's t-distribution.
    
    :param n_paths: Number of simulation paths (e.g., 10000)
    :param n_days: Number of days to simulate (e.g., 90)
    :param params: Fitted parameters (df, loc, scale) from stats.t.fit
    :param start_price: Starting price of the asset
    :return: Array of shape (n_days + 1, n_paths) containing prices
    """
    df, loc, scale = params
    
    # Generate random returns
    # Shape: (n_days, n_paths)
    random_returns = stats.t.rvs(df, loc=loc, scale=scale, size=(n_days, n_paths))
    
    # Calculate cumulative returns
    # Price_t = Price_0 * product(1 + r_i)
    
    price_paths = np.zeros((n_days + 1, n_paths))
    price_paths[0] = start_price
    
    # Calculate path
    # Using cumprod for efficiency
    # (1 + r)
    growth_factors = 1 + random_returns
    
    # Cumulative product along the days axis (axis 0)
    cum_growth = np.cumprod(growth_factors, axis=0)
    
    price_paths[1:] = start_price * cum_growth
    
    return price_paths

def calculate_rolling_volatility(
    price_paths: np.ndarray,
    window: int = 30
) -> np.ndarray:
    """
    Calculate 30-day rolling annualized volatility for price paths.
    
    :param price_paths: Array of shape (n_days, n_paths)
    :param window: Rolling window size (e.g., 30)
    :return: Array of shape (n_days, n_paths) containing annualized volatility
    """
    # Calculate Log Returns: ln(P_t / P_{t-1})
    # Shape: (n_days - 1, n_paths)
    log_returns = np.log(price_paths[1:] / price_paths[:-1])
    
    # We need to maintain shape. Prepend 0s or handle carefully?
    # The output needs to align with price_paths.
    # Volatility at time t is based on returns from t-window to t.
    
    n_days, n_paths = price_paths.shape
    vol_map = np.zeros_like(price_paths)
    
    # Standardize window
    # Efficient rolling std using pandas is easiest, but creating DF is slow for 10k cols.
    # Use simple strided loop for now or numpy convolution approximation.
    # Given window is small (30), loop is okay.
    
    # Pre-calculate squared returns for variance
    # Var = E[x^2] - (E[x])^2
    
    # To act like "Pandas Rolling Std", we iterate.
    # We will pad the log_returns to match size
    padded_returns = np.vstack([np.zeros((1, n_paths)), log_returns]) # Shape (n_days, n_paths)
    
    # We need a loop to be safe and correct.
    # Optimized numpy approach:
    # Use uniform filter? standard deviation filter?
    # Let's simple loop over the array size (vectorized over paths)
    
    # Vol array
    # First 'window' days will have incomplete data, assume constant or ramping?
    # Ideally, we PREPEND real history before calling this.
    
    for t in range(window, n_days):
        # Slice: look back 'window' days
        # returns from t-window to t
        slice_ret = padded_returns[t-window+1 : t+1]
        
        # Std Dev of this slice along axis 0
        std = np.std(slice_ret, axis=0) # Shape (n_paths,)
        
        # Annualize
        vol = std * np.sqrt(252)
        vol_map[t] = vol
        
    # Fill initial window with the first valid calculation?
    # Or assuming the caller prepended history, so we just slice off the valid part later.
    # Let's fill the start with the first valid value to avoid 0s.
    vol_map[:window] = vol_map[window]
        
    return vol_map
