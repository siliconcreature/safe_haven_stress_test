import numpy as np
import pandas as pd
from src.options import calculate_option_price

def run_monte_carlo_simulation(
    price_paths: np.ndarray, # Shape: (n_days + 1, n_paths)
    stock_qty: int,
    put_options: list, # List of {strike, qty, days_to_expiry}
    volatility: float | np.ndarray, # Scalar or Map of shape (n_days + 1, n_paths)
    cash_balance: float = 0.0,
    risk_free_rate: float = 0.05
):
    """
    Run simulation on generated price paths.
    Calculating daily portfolio value for all paths.
    """
    n_days_plus_1, n_paths = price_paths.shape
    n_days = n_days_plus_1 - 1
    
    # Initialize Portfolio Value Array with Cash
    # Shape: (n_days + 1, n_paths)
    portfolio_values = np.full_like(price_paths, cash_balance)
    
    # 1. Stock Component
    portfolio_values += price_paths * stock_qty
    
    # 2. Options Component
    for t in range(n_days + 1):
        current_prices = price_paths[t] # Vector of N paths
        
        # Determine Volatility for this step
        # If float, use constant. If array, use specific row.
        if isinstance(volatility, np.ndarray):
             # Ensure bounds
             idx = min(t, volatility.shape[0] - 1)
             current_vol = volatility[idx]
        else:
             current_vol = volatility
        
        for put in put_options:
            days_remaining = put['days_to_expiry'] - t
            time_remaining = max(0, days_remaining / 365.0)
            
            if time_remaining <= 1e-5:
                # Expired / Intrinsic
                # Vectorized numeric intrinsic
                intrinsic = np.maximum(0.0, put['strike'] - current_prices)
                portfolio_values[t] += intrinsic * put['qty']
            else:
                opt_prices = calculate_option_price(
                    spot_price=current_prices,
                    strike_price=put['strike'],
                    time_to_expiry_years=time_remaining,
                    risk_free_rate=risk_free_rate,
                    volatility=current_vol,
                    option_type='put'
                )
                portfolio_values[t] += opt_prices * put['qty']
            
    return portfolio_values

def calculate_simulation_stats(
    portfolio_values: np.ndarray, 
    price_paths: np.ndarray = None, 
    stock_qty: int = 0,
    volatility_map: np.ndarray = None # Shape: (n_days+1, n_paths) or None
):
    """
    Calculate statistics from simulation results.
    Includes Margin Call analysis if price_paths and stock_qty provided.
    Margin Requirement: Equity >= 30% of Stock Value.
    """
    initial_values = portfolio_values[0]
    final_values = portfolio_values[-1]
    
    pnl = final_values - initial_values
    
    # Standard Metrics
    metrics = {
        "Mean P&L": np.mean(pnl),
        "Median P&L": np.median(pnl),
        "5% VaR (P&L)": np.percentile(pnl, 5),
        "1% VaR (P&L)": np.percentile(pnl, 1),
        "5th %ile Value": np.percentile(final_values, 5),
        "Median Value": np.median(final_values),
        "Win Rate": np.mean(pnl > 0),
        "Initial Margin Req": 0.0
    }
    
    # Drawdown calc:
    peak_values = np.maximum.accumulate(portfolio_values, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        drawdowns = (portfolio_values - peak_values) / peak_values
        drawdowns = np.nan_to_num(drawdowns, nan=-1.0)
        
    max_drawdowns = np.min(drawdowns, axis=0)
    metrics["Avg Max Drawdown"] = np.mean(max_drawdowns)
    metrics["Worst Max Drawdown"] = np.min(max_drawdowns)
    
    # --- Contextual Metrics (Price/IV at X event) ---
    if price_paths is not None:
        # 1. At 5th Percentile Outcome
        # Find index of path closest to 5th percentile value
        target_val = metrics["5th %ile Value"]
        idx_5th = (np.abs(final_values - target_val)).argmin()
        metrics["Stock Price @ 5th %ile"] = price_paths[-1, idx_5th]
        
        if volatility_map is not None:
             metrics["IV @ 5th %ile"] = volatility_map[-1, idx_5th]
        
        # 2. At Max Drawdown (Worst Case Path)
        # Find index of path with worst max drawdown
        idx_worst_dd = np.argmin(max_drawdowns)
        # Find TIME of max drawdown for this path
        t_worst_dd = np.argmin(drawdowns[:, idx_worst_dd])
        
        metrics["Stock Price @ Max DD"] = price_paths[t_worst_dd, idx_worst_dd]
        if volatility_map is not None:
            metrics["IV @ Max DD"] = volatility_map[t_worst_dd, idx_worst_dd]
            
        if stock_qty > 0:
            # Initial Margin = 30% * (Stock Price[0] * Qty)
            # Assuming Price[0] is same for all paths
            initial_stock_val = price_paths[0, 0] * stock_qty
            metrics["Initial Margin Req"] = 0.30 * initial_stock_val
            
            # Margin Call Calc
            stock_values = price_paths * stock_qty
            margin_req = 0.30 * stock_values
            margin_calls = portfolio_values < margin_req
            any_margin_call = np.any(margin_calls, axis=0)
            metrics["Margin Call %"] = np.mean(any_margin_call)
            
            # Check margin call at 5th percentile path
            mc_at_5th = bool(any_margin_call[idx_5th])
            metrics["Margin Call @ 5th %ile"] = 1.0 if mc_at_5th else 0.0
            
            # Check margin call at median drawdown path
            median_dd_idx = np.argsort(max_drawdowns)[len(max_drawdowns) // 2]
            mc_at_median_dd = bool(any_margin_call[median_dd_idx])
            metrics["Margin Call @ Median DD"] = 1.0 if mc_at_median_dd else 0.0
            metrics["Median Drawdown"] = max_drawdowns[median_dd_idx]
        else:
            metrics["Margin Call %"] = 0.0
            metrics["Margin Call @ 5th %ile"] = 0.0
            metrics["Margin Call @ Median DD"] = 0.0
    else:
        metrics["Margin Call %"] = 0.0
    
    return metrics, pnl
