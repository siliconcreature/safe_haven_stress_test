import numpy as np
import pandas as pd
from src.simulation import run_monte_carlo_simulation, calculate_simulation_stats

def test_simulation():
    print("Testing Monte Carlo Simulation Logic...")
    
    # 1. Setup Parameters
    n_days = 30
    n_paths = 1000
    start_price = 100.0
    volatility = 0.50 # 50% Vol
    dt = 1/252
    
    # Generate simple Geometric Brownian Motion paths
    # S_t = S_0 * exp((mu - 0.5*sigma^2)*t + sigma*W_t)
    # Assume mu = 0 for simplicity
    
    np.random.seed(42)
    Z = np.random.normal(0, 1, size=(n_days, n_paths))
    
    # Construct paths
    log_returns = (0 - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * Z
    log_diffusion = np.cumsum(log_returns, axis=0)
    
    # Add start (day 0)
    log_paths = np.vstack([np.zeros((1, n_paths)), log_diffusion])
    price_paths = start_price * np.exp(log_paths)

    # 2. Define Portfolio
    stock_qty = 10000
    total_equity = 1700000 # 1.7M Total Equity as per user
    
    # Add a put option hedge
    puts = [
        {'strike': 90, 'qty': 10000, 'days_to_expiry': 30}
    ]
    
    # Calculate Cost Basis like the App does
    # Assume option price is ~1.5 (guess for 90 strike put when spot is 100)
    # App logic: effective_cash = cash_balance - cost_stock - cost_options
    # But user said "total equity is 1.7m".
    # If they are holding 10,000 shares @ 100 = 1,000,000
    # Then cash = 700,000 (ignoring option cost for a moment)
    
    # Let's say options cost 10000 * 1.0 = 10,000
    option_cost = 10000 * 1.5 
    stock_cost = stock_qty * start_price
    
    # Determine effective cash available after positions are entered
    # If 1.7m is the starting cash before buying
    effective_cash = total_equity - stock_cost - option_cost
    
    with open("results.txt", "w") as f:
        f.write("Testing Monte Carlo Simulation Logic...\n")
        f.write(f"Generated {n_paths} paths over {n_days} days.\n")
        f.write(f"Start Price: {start_price}\n")
        f.write(f"Final Mean Price: {price_paths[-1].mean():.2f}\n")
        f.write(f"Stock Qty: {stock_qty}\n")
        f.write(f"Total Initial Equity (Cash): {total_equity}\n")
        f.write(f"Effective Cash (after purchase): {effective_cash:.2f}\n")
        
        # 3. Run Simulation
        f.write("\nRunning Simulation...\n")
        portfolio_values = run_monte_carlo_simulation(
            price_paths=price_paths,
            stock_qty=stock_qty,
            put_options=puts,
            volatility=volatility,
            cash_balance=effective_cash
        )
        
        # 4. Calculate Stats
        f.write("Calculating Stats...\n")
        metrics, pnl = calculate_simulation_stats(
            portfolio_values=portfolio_values,
            price_paths=price_paths, # Important for Margin Calc
            stock_qty=stock_qty      # Important for Margin Calc
        )
        
        # 5. Output Results
        f.write("\nSimulation Results:\n")
        for k, v in metrics.items():
            if "Multiplier" not in k:
                f.write(f"{k}: {v:.2f}\n")
                
        # Verify Margin Logic specifically
        # specific check: if stock drops significantly, margin call should trigger
        # Margin Req = 30% * (Price * Qty)
        # Equity = Portfolio Value
        # Check a specific path where price drops
        
        # Find worst path
        final_prices = price_paths[-1]
        worst_idx = np.argmin(final_prices)
        worst_price = final_prices[worst_idx]
        
        f.write(f"\nWorst Case Path (Idx {worst_idx}):\n")
        f.write(f"Final Stock Price: {worst_price:.2f}\n")
        
        final_val = portfolio_values[-1, worst_idx]
        f.write(f"Final Portfolio Entry: {final_val:.2f}\n")
        
        # Manual Margin Check for worst case
        stock_val = worst_price * stock_qty
        margin_req = 0.30 * stock_val
        f.write(f"Margin Requirement (30%): {margin_req:.2f}\n")
        
        is_margin_call = final_val < margin_req
        f.write(f"Margin Call Triggered manually? {is_margin_call}\n")
        
        if metrics['Margin Call %'] > 0:
            f.write("\nSUCCESS: Margin Call Logic detected risks.\n")
        else:
            f.write("\nWARNING: No Margin Calls detected (might be expected depending on params).\n")


if __name__ == "__main__":
    test_simulation()
