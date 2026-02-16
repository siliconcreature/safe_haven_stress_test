import pandas as pd
import numpy as np
import datetime
import sys
# Mock streamlit
class MockSt:
    def spinner(self, text): return self
    def __enter__(self): pass
    def __exit__(self, exc_type, exc_val, exc_tb): pass
    def progress(self, val): pass
    def write(self, text): print(text)
    def error(self, text): print(f"ERROR: {text}")

from src.data import fetch_data, get_option_expirations, get_option_chain
from src.analysis import calculate_returns, fit_t_distribution, simulate_paths_t_dist, calculate_rolling_volatility
from src.simulation import run_monte_carlo_simulation, calculate_simulation_stats

def run_full_simulation():
    print("Starting Full Simulation Logic (Headless)...")
    
    # --- Configuration ---
    ticker = "NVDA"
    start_date = pd.to_datetime("2021-01-01")
    end_date = pd.to_datetime("today")
    
    n_paths = 1000
    n_days = 90 # Simulation Duration
    target_option_days = 120 # Option Maturity
    
    initial_iv_assumption = 0.50
    iv_stress_factor = 0.20 # 20%
    
    # Portfolio
    stock_qty = 10000
    total_equity = 1700000 
    
    # Hedge Settings
    put_qty_min = 150   # Contracts (x100) -> 15,000 shares
    put_qty_max = 500   # Contracts (x100) -> 50,000 shares
    put_qty_step = 50
    
    strike_pct_start = 0.70
    strike_pct_end = 0.90
    strike_pct_step = 0.05
    
    # --- 1. Data Fetching ---
    print(f"Fetching data for {ticker}...")
    df = fetch_data(ticker, start_date - pd.Timedelta(days=60), end_date).copy()
    if df.empty:
        print("No data found.")
        return

    df.index = df.index.tz_localize(None)
    fit_df = df[df.index >= start_date]
    current_price = fit_df['Close'].iloc[-1]
    print(f"Current Price: ${current_price:.2f}")
    
    # --- 2. Fit Distribution ---
    print("Fitting Distributions...")
    returns = calculate_returns(fit_df['Close'])
    t_params = fit_t_distribution(returns)
    
    # --- 3. Option Data ---
    expirations = get_option_expirations(ticker)
    
    chain = pd.DataFrame()
    days_to_exp = target_option_days # Default
    
    if expirations:
        today = datetime.date.today()
        target_date = today + datetime.timedelta(days=target_option_days)
        closest_exp = min(expirations, key=lambda x: abs((pd.to_datetime(x).date() - target_date).days))
        print(f"Selected Expiry: {closest_exp}")
        
        d_exp = (pd.to_datetime(closest_exp).date() - today).days
        days_to_exp = d_exp if d_exp > 0 else target_option_days
        
        # Validation
        if days_to_exp < n_days:
             print(f"Warning: Option expires in {days_to_exp} days, but simulation is {n_days} days!")
        
        chain = get_option_chain(ticker, closest_exp)
    else:
        print("Using synthetic options data.")

    # --- 4. Simulation Setup ---
    print(f"Generating {n_paths} paths over {days_to_exp} days...")
    sim_paths = simulate_paths_t_dist(n_paths, days_to_exp, t_params, current_price)
    
    # Volatility Map
    history_prices = fit_df['Close'].tail(30).values
    history_block = np.tile(history_prices.reshape(-1, 1), (1, n_paths))
    stitched_paths = np.vstack([history_block[:-1], sim_paths])
    full_vol_map = calculate_rolling_volatility(stitched_paths, window=30)
    sim_vol_map = full_vol_map[-(days_to_exp + 1):]
    
    start_vol = sim_vol_map[0, 0]
    # Estimate ATM IV
    anchor_iv_val = initial_iv_assumption
    if not chain.empty:
        atm_row = chain.iloc[(chain['strike'] - current_price).abs().argsort()[:1]]
        if not atm_row.empty:
            anchor_iv_val = atm_row['impliedVolatility'].values[0]
            
    scaler = anchor_iv_val / start_vol if start_vol > 0 else 1.0
    base_vol_map = sim_vol_map * scaler
    
    # --- 5. Portfolio Simulation Loop ---
    results_summary = []
    
    # Target Strikes
    target_pcts = np.arange(strike_pct_start, strike_pct_end + 0.001, strike_pct_step)
    target_strikes = current_price * target_pcts
    
    relevant_puts = pd.DataFrame()
    if not chain.empty:
        selected_indices = []
        for t_k in target_strikes:
            idx = (chain['strike'] - t_k).abs().argmin()
            if idx not in selected_indices:
                selected_indices.append(idx)
        relevant_puts = chain.iloc[selected_indices].sort_values('strike')
    else:
        # Synthetic
        relevant_puts = pd.DataFrame({'strike': target_strikes, 'lastPrice': 0.0, 'impliedVolatility': initial_iv_assumption})

    # SCENARIO: STOCK ONLY
    print("Running Scenario: Stock Only")
    stress_vol_map = base_vol_map * (1 + iv_stress_factor)
    cost_stock = current_price * stock_qty
    # Calculate effective cash based on Total Equity assumption
    # Total Equity = Cash + Stock + Options
    # Start: Cash_Balance(Input) is Total Liquidity? No, user said "Total Equity is 1.7m"
    # If they hold 10k shares, that's $1m+.
    # So Effective Cash = Total Equity - Cost of Position
    effective_cash_stock = total_equity - cost_stock
    
    pf_values_stock = run_monte_carlo_simulation(sim_paths, stock_qty, [], stress_vol_map, cash_balance=effective_cash_stock)
    metrics_stock, pnl_stock = calculate_simulation_stats(pf_values_stock, sim_paths, stock_qty, volatility_map=stress_vol_map)
    
    metrics_stock['Name'] = "Stock Only"
    metrics_stock['Put Qty'] = 0
    metrics_stock['Strike'] = 0
    results_summary.append(metrics_stock)
    
    # SCENARIOS: HEDGED
    put_qtys_contracts = list(range(put_qty_min, put_qty_max + 1, put_qty_step))
    
    print(f"Running Hedged Scenarios ({len(relevant_puts)} strikes x {len(put_qtys_contracts)} qtys)...")
    
    for idx, row in relevant_puts.iterrows():
        k = row['strike']
        strike_iv = row['impliedVolatility'] if 'impliedVolatility' in row else anchor_iv_val
        s_scaler = strike_iv / start_vol if start_vol > 0 else 1.0
        strike_vol_map = sim_vol_map * s_scaler * (1 + iv_stress_factor)
        
        opt_price = row['lastPrice'] if 'lastPrice' in row else 1.0 # Fallback
        
        for q_contracts in put_qtys_contracts:
            # Multiplier assumption: 100 shares per contract
            total_put_shares = q_contracts * 100 
            
            # NOTE: run_monte_carlo_simulation uses raw 'qty' for payoff calc.
            # If standard option, we pass total_put_shares as qty? 
            # Or does price model assume per-share?
            # Black-Scholes returns price PER SHARE.
            # So if we have 100 contracts, we have 10,000 shares exposure.
            # So 'qty' passed to simulation should be total_put_shares.
            
            puts = [{'strike': k, 'qty': total_put_shares, 'days_to_expiry': days_to_exp}]
            
            cost_hedge = (current_price * stock_qty) + (opt_price * total_put_shares)
            effective_cash_hedge = total_equity - cost_hedge
            
            # --- Portfolio Margin Proxy (Max Loss @ +/- 15%) ---
            price_stress_down = current_price * 0.85
            price_stress_up = current_price * 1.15
            
            # Downside Liquidation Value
            put_intrinsic_down = max(0, k - price_stress_down)
            val_down = (price_stress_down * stock_qty) + (put_intrinsic_down * total_put_shares)
            
            # Upside Liquidation Value (Put ~ 0)
            val_up = (price_stress_up * stock_qty)
            
            # Current Liquidation Value
            val_curr = (current_price * stock_qty) + (opt_price * total_put_shares)
            
            # Margin Req = Max Loss
            pm_margin_req = max(0, val_curr - val_down, val_curr - val_up)
            
            pf_values = run_monte_carlo_simulation(sim_paths, stock_qty, puts, strike_vol_map, cash_balance=effective_cash_hedge)
            metrics, pnl = calculate_simulation_stats(pf_values, sim_paths, stock_qty, volatility_map=strike_vol_map)
            
            # Override Initial Margin with PM Calc
            metrics["Initial Margin Req"] = pm_margin_req
            
            name = f"Put {int(k)} x{q_contracts}c"
            metrics['Name'] = name
            metrics['Put Qty'] = q_contracts # Contracts
            metrics['Strike'] = k
            results_summary.append(metrics)

    # --- Output ---
    df_res = pd.DataFrame(results_summary)
    
    # Custom Table as requested: Number of Puts | Strike | 5th %ile | Margin Breached?
    print("\n\n=== HEDGE ANALYSIS: Puts vs Risk ===")
    print(f"{'Qty (Contracts)':<15} {'Strike':<10} {'5th %ile Value':<20} {'Margin Call %':<15} {'Mean P&L':<15}")
    print("-" * 80)
    
    # Filter to only hedged
    hedged_df = df_res[df_res['Put Qty'] > 0].sort_values(['Strike', 'Put Qty'])
    
    for _, row in hedged_df.iterrows():
        qty = int(row['Put Qty'])
        strike = row['Strike']
        val_5th = row['5th %ile Value']
        margin_pct = row['Margin Call %']
        mean_pnl = row['Mean P&L']
        
        print(f"{qty:<15} {strike:<10.2f} ${val_5th:,.2f}          {margin_pct:.1%}          ${mean_pnl:,.0f}")
        
    print("\n--- Stock Only Reference ---")
    stock_row = df_res[df_res['Put Qty'] == 0].iloc[0]
    print(f"0               N/A        ${stock_row['5th %ile Value']:,.2f}          {stock_row['Margin Call %']:.1%}          ${stock_row['Mean P&L']:,.0f}")

    # Export to CSV
    df_res.to_csv("full_simulation_results_expanded.csv", index=False)
    print("\nResults saved to full_simulation_results_expanded.csv")

if __name__ == "__main__":
    run_full_simulation()
