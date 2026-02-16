import pandas as pd
import numpy as np
from src.options import calculate_option_price

def run_backtest(
    market_data: pd.DataFrame,
    start_date: str,
    initial_cash: float,
    stock_qty: int,
    put_options: list, # List of dicts: {strike, qty, expiry_date_str}
    volatility: float,
    risk_free_rate: float = 0.05
):
    """
    Run a daily backtest simulation.
    """
    # Filter data to start date
    df = market_data[market_data.index >= start_date].copy()
    
    results = []
    
    # Establish Initial positions
    # Assume entry at Open of Start Day or Close? Using Close for simplicity.
    initial_price = df.iloc[0]['Close']
    
    # Calculate initial option cost
    initial_option_cost = 0
    start_dt = df.index[0]
    
    for put in put_options:
        expiry_dt = pd.to_datetime(put['expiry'])
        time_to_exp = (expiry_dt - start_dt).days / 365.0
        if time_to_exp < 0: time_to_exp = 0
        
        price = calculate_option_price(initial_price, put['strike'], time_to_exp, risk_free_rate, volatility, 'put')
        initial_option_cost += price * put['qty']
    
    # Cost Basis
    # We ignore cash drag on stock purchase for this P&L view, focusing on portfolio value.
    # But usually: Total Value = Cash + Stock Value + Option Value.
    
    for date, row in df.iterrows():
        current_price = row['Close']
        stock_value = stock_qty * current_price
        
        option_value = 0
        for put in put_options:
            expiry_dt = pd.to_datetime(put['expiry'])
            time_to_exp = (expiry_dt - date).days / 365.0
            
            if time_to_exp <= 0:
                # Intrinsic at expiration
                val = max(0, put['strike'] - current_price)
            else:
                val = calculate_option_price(current_price, put['strike'], time_to_exp, risk_free_rate, volatility, 'put')
            
            option_value += val * put['qty']
            
        total_value = stock_value + option_value
        
        results.append({
            'Date': date,
            'Price': current_price,
            'Stock Value': stock_value,
            'Option Value': option_value,
            'Total Value': total_value
        })
        
    return pd.DataFrame(results).set_index('Date')
