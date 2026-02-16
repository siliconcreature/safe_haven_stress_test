import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from src.data import fetch_data, get_option_expirations, get_option_chain
from src.analysis import calculate_returns, fit_t_distribution, fit_normal_distribution, simulate_paths_t_dist, calculate_rolling_volatility
from src.simulation import run_monte_carlo_simulation, calculate_simulation_stats
import datetime

st.set_page_config(page_title="NVDA Portfolio Analyzer", layout="wide")

st.title("NVDA Portfolio Analyzer: Fat Tail & Monte Carlo")

# --- Load Data First ---
st.sidebar.header("Configuration")
ticker = "NVDA"
start_date = st.sidebar.date_input("Start Date (Fit)", pd.to_datetime("2021-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# Fetch Data
df = fetch_data(ticker, pd.to_datetime(start_date) - pd.Timedelta(days=60), end_date).copy()

if df.empty:
    st.error("No data found. Check dates.")
    st.stop()

# Fix Timezone
df.index = df.index.tz_localize(None)

# Filter for fitting
fit_df = df[df.index >= pd.to_datetime(start_date)]

# --- Sidebar Config ---
st.sidebar.subheader("Monte Carlo Settings")
n_paths = st.sidebar.number_input("Number of Paths", value=100, step=100)
n_days = st.sidebar.number_input("Simulation Days", value=90) # Sim Duration
target_option_days = st.sidebar.number_input("Target Option Expiry (Days)", value=120) # Option Maturity

# This vol assumption is for the initial anchor, user said 50%
initial_iv_assumption = st.sidebar.slider("Initial IV Assumption", 0.2, 1.0, 0.50, 0.01)

st.sidebar.subheader("Stress Testing")
iv_stress_factor = st.sidebar.slider("IV Stress Increase (%)", 0, 100, 20) / 100.0
st.sidebar.markdown(f"*Adjusts IV curve up by {iv_stress_factor:.0%} (Worst Case)*")

st.sidebar.subheader("Portfolio Settings")
total_equity = st.sidebar.number_input("Total Equity ($)", value=1700000, step=100000)
stock_qty = st.sidebar.number_input("NVDA Shares", value=10000)
# cash_balance derived for compatibility, but logic should use total_equity
# effective_cash = total_equity - stock_cost

st.sidebar.info("Hedge: Long Puts")
option_multiplier = st.sidebar.number_input("Option Multiplier (Shares/Contract)", value=100)
put_qty_min = st.sidebar.number_input("Min Put Qty (Contracts)", value=150)
put_qty_max = st.sidebar.number_input("Max Put Qty (Contracts)", value=500)
put_qty_step = st.sidebar.number_input("Put Qty Step", value=50)

st.sidebar.subheader("Strike Selection (vs Spot)")
strike_pct_start = st.sidebar.slider("Start %", 0.5, 1.0, 0.70, 0.05)
strike_pct_end = st.sidebar.slider("End %", 0.5, 1.0, 0.90, 0.05)
strike_pct_step = st.sidebar.slider("Step %", 0.01, 0.10, 0.05, 0.01)


# --- Real-Time Option Data ---
st.sidebar.subheader("Real-Time Option Data")

expirations = get_option_expirations(ticker)
selected_expiry = None
chain = pd.DataFrame()
days_to_exp = target_option_days # Default

if expirations:
    today = datetime.date.today()
    target_date = today + datetime.timedelta(days=target_option_days)
    # Find closest expiry
    closest_exp = min(expirations, key=lambda x: abs((pd.to_datetime(x).date() - target_date).days))
    selected_expiry = st.sidebar.selectbox("Select Expiry", expirations, index=expirations.index(closest_exp))
    
    if selected_expiry:
        d_exp = (pd.to_datetime(selected_expiry).date() - today).days
        days_to_exp = d_exp if d_exp > 0 else target_option_days
        st.sidebar.write(f"Days to Expiry: {days_to_exp}")
        
        # Validation: Expiry must be > Simulation Days
        if days_to_exp < n_days:
             st.sidebar.error(f"Error: Option expires in {days_to_exp} days, but simulation is {n_days} days!")
             st.stop()
        
        # Fetch Chain
        chain = get_option_chain(ticker, selected_expiry)
        if not chain.empty and not fit_df.empty:
            current_price = fit_df['Close'].iloc[-1]
            # Get ATM IV
            if 'impliedVolatility' in chain.columns:
                atm_row = chain.iloc[(chain['strike'] - current_price).abs().argsort()[:1]]
                if not atm_row.empty:
                    atm_iv = atm_row['impliedVolatility'].values[0]
                    st.sidebar.metric("ATM Implied Volatility", f"{atm_iv:.1%}")
else:
    st.sidebar.error("Could not fetch expirations.")

# --- Tab 1: Distribution Analysis ---
tab1, tab2 = st.tabs(["Distribution Analysis", "Monte Carlo Simulation"])

with tab1:
    st.subheader(f"Daily Returns Distribution ({start_date} to {end_date})")
    
    returns = calculate_returns(fit_df['Close'])
    
    # Fit Distributions
    mu, std = fit_normal_distribution(returns)
    t_params = fit_t_distribution(returns) # (df, loc, scale)
    
    st.write(f"**Normal Fit**: Mean={mu:.4f}, Std={std:.4f}")
    st.write(f"**Student's t Fit**: DF={t_params[0]:.2f}, Loc={t_params[1]:.4f}, Scale={t_params[2]:.4f}")
    
    # Visualization
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(x=returns, histnorm='probability density', name='Historical Returns', opacity=0.75))
    
    x_range = np.linspace(returns.min(), returns.max(), 1000)
    pdf_norm = stats.norm.pdf(x_range, mu, std)
    fig_dist.add_trace(go.Scatter(x=x_range, y=pdf_norm, mode='lines', name='Normal Dist', line=dict(color='red')))
    
    pdf_t = stats.t.pdf(x_range, *t_params)
    fig_dist.add_trace(go.Scatter(x=x_range, y=pdf_t, mode='lines', name='T-Dist (Fat Tail)', line=dict(color='green', width=3)))
    
    fig_dist.update_layout(title="Return Distribution Fit", xaxis_title="Daily Return", yaxis_title="Density")
    st.plotly_chart(fig_dist, use_container_width=True)

# --- Tab 2: Monte Carlo ---
with tab2:
    st.subheader(f"Monte Carlo Portfolio Simulation (Expiry: {selected_expiry if selected_expiry else 'N/A'})")
    
    if st.button("Run Simulation"):
        current_price = fit_df['Close'].iloc[-1]
        st.write(f"Current Price: ${current_price:.2f}")
        
        # 1. Generate Paths (Simulate n_days, not full expiry)
        with st.spinner(f"Generating {n_paths} Price Paths for {n_days} days..."):
            sim_paths = simulate_paths_t_dist(n_paths, n_days, t_params, current_price)
            
        # 2. Prepare Dynamic Volatility Map
        with st.spinner("Calculating Dynamic Rolling Volatility..."):
            history_prices = fit_df['Close'].tail(30).values
            history_block = np.tile(history_prices.reshape(-1, 1), (1, n_paths))
            
            stitched_paths = np.vstack([history_block[:-1], sim_paths])
            full_vol_map = calculate_rolling_volatility(stitched_paths, window=30)
            sim_vol_map = full_vol_map[-(n_days + 1):]
            
            # Anchor Logic
            start_vol = sim_vol_map[0, 0] 
            anchor_iv_val = atm_iv if 'atm_iv' in locals() else initial_iv_assumption
            
            scaler = anchor_iv_val / start_vol if start_vol > 0 else 1.0
            
            # Base Vol Map (Center)
            base_vol_map = sim_vol_map * scaler
            
            st.metric("Simulated Vol (Mean End)", f"{base_vol_map[-1].mean():.1%}")
            
        try:
            # Plot Volatility Cone with Price Paths
            from plotly.subplots import make_subplots
            
            fig_vol = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add IV paths (orange, solid)
            subset_vol = base_vol_map[:, :50]
            for i in range(50):
                fig_vol.add_trace(
                    go.Scatter(y=subset_vol[:, i], mode='lines', 
                              line=dict(width=1, color='orange'), 
                              opacity=0.1, showlegend=False),
                    secondary_y=False
                )
            
            # Add Price paths (blue, dashed)
            subset_price = sim_paths[:, :50]
            for i in range(50):
                fig_vol.add_trace(
                    go.Scatter(y=subset_price[:, i], mode='lines',
                              line=dict(width=1, color='blue', dash='dash'),
                              opacity=0.15, showlegend=False),
                    secondary_y=True
                )
            
            fig_vol.update_xaxes(title_text="Days")
            fig_vol.update_yaxes(title_text="Implied Volatility", secondary_y=False)
            fig_vol.update_yaxes(title_text="Stock Price ($)", secondary_y=True)
            fig_vol.update_layout(title="Projected Volatility & Price Paths")
            
            st.plotly_chart(fig_vol, use_container_width=True)
        except Exception as e:
            st.error(f"Error plotting vol: {e}")
            
        
        # 3. Simulate Portfolios
        st.subheader("Portfolio Performance Comparison")
        
        results_summary = []
        
        # Determine Strikes
        relevant_puts = pd.DataFrame()
        
        # Target Percentages (e.g., 0.70, 0.75...)
        target_pcts = np.arange(strike_pct_start, strike_pct_end + 0.001, strike_pct_step)
        target_strikes = current_price * target_pcts
        
        if not chain.empty:
            # Find closest real strikes to targets
            selected_indices = []
            for t_k in target_strikes:
                # Find index of closest strike in chain
                idx = (chain['strike'] - t_k).abs().argmin()
                if idx not in selected_indices:
                    selected_indices.append(idx)
            
            relevant_puts = chain.iloc[selected_indices].sort_values('strike')
        else:
            # Fallback to synthetic
            ks = target_strikes
            relevant_puts = pd.DataFrame({'strike': ks, 'lastPrice': 0, 'impliedVolatility': initial_iv_assumption})

        put_qtys_contracts = range(put_qty_min, put_qty_max + 1, put_qty_step)
        
        
        st.info("""
        **IBKR Portfolio Margin Calculation:**
        - **Initial Margin Req** = Max loss across stress scenarios (85%-115% price movement)
        - **Traditional Margin** = 30% of Stock Value + Full Put Premium (for comparison)
        - Long puts provide downside protection, significantly reducing margin requirements
        - All positions require Initial Margin ≤ Total Equity to be established
        """)
        
        progress_bar = st.progress(0)
        total_scenarios = len(relevant_puts) * len(put_qtys_contracts) + 1
        processed = 0
        
        # Scenario 1: Stock Only
        stress_vol_map = base_vol_map * (1 + iv_stress_factor)
        
        # Calculate Initial Cost for Stock Only
        cost_stock = current_price * stock_qty
        effective_cash_stock = total_equity - cost_stock
        
        pf_values_stock = run_monte_carlo_simulation(sim_paths, stock_qty, [], stress_vol_map, cash_balance=effective_cash_stock)
        
        # Pass price_paths and stock_qty for Margin Calc
        metrics_stock, pnl_stock = calculate_simulation_stats(pf_values_stock, sim_paths, stock_qty, volatility_map=stress_vol_map)
        
        metrics_stock['Name'] = "Stock Only"
        metrics_stock['Put Qty'] = 0
        metrics_stock['Current Value'] = pf_values_stock[0, 0]
        results_summary.append(metrics_stock)
        processed += 1
        progress_bar.progress(processed / total_scenarios)
        
        # Scenarios: Hedged
        for idx, row in relevant_puts.iterrows():
            k = row['strike']
            # Scale Map for this Strike
            strike_iv_val = row['impliedVolatility'] if 'impliedVolatility' in row else anchor_iv_val
            s_scaler = strike_iv_val / start_vol if start_vol > 0 else 1.0
            
            # Apply Stress
            strike_vol_map = sim_vol_map * s_scaler * (1 + iv_stress_factor)
            
            # Option Price (Ask/Last)
            opt_price = row['lastPrice'] if 'lastPrice' in row else 0.0
            
            for q_contracts in put_qtys_contracts:
                total_put_shares = q_contracts * option_multiplier
                
                puts = [{'strike': k, 'qty': total_put_shares, 'days_to_expiry': days_to_exp}]
                
                # Calculate Cost Basis
                cost_hedge = (current_price * stock_qty) + (opt_price * total_put_shares)
                effective_cash_hedge = total_equity - cost_hedge
                
                # --- IBKR Portfolio Margin Calculation ---
                # Long puts require NO margin (premium paid upfront)
                # Stock margin = max loss in stress scenarios
                
                # Test multiple stress scenarios
                stress_scenarios = [0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15]
                max_loss = 0
                
                for stress_factor in stress_scenarios:
                    stressed_price = current_price * stress_factor
                    
                    # Stock value at stressed price
                    stock_value_stressed = stressed_price * stock_qty
                    
                    # Put value at stressed price (intrinsic value approximation)
                    put_intrinsic = max(0, k - stressed_price)
                    put_value_stressed = put_intrinsic * total_put_shares
                    
                    # Total portfolio value at stressed price
                    portfolio_value_stressed = stock_value_stressed + put_value_stressed
                    
                    # Initial portfolio value
                    initial_portfolio_value = (current_price * stock_qty) + (opt_price * total_put_shares)
                    
                    # Loss in this scenario
                    loss = initial_portfolio_value - portfolio_value_stressed
                    max_loss = max(max_loss, loss)
                
                # IBKR Margin Requirement = Max Loss across scenarios
                ibkr_margin_req = max_loss
                
                # For comparison: Traditional 30% margin (stock only)
                traditional_margin = 0.30 * (current_price * stock_qty) + (opt_price * total_put_shares)
                
                pf_values = run_monte_carlo_simulation(sim_paths, stock_qty, puts, strike_vol_map, cash_balance=effective_cash_hedge)
                metrics, pnl = calculate_simulation_stats(pf_values, sim_paths, stock_qty, volatility_map=strike_vol_map)
                
                metrics["Initial Margin Req"] = ibkr_margin_req
                metrics["Traditional Margin"] = traditional_margin
                
                name = f"Put {int(k)} x{q_contracts}"
                metrics['Name'] = name
                metrics['Put Qty'] = q_contracts
                metrics['Strike'] = k
                metrics['Current Value'] = pf_values[0, 0]
                results_summary.append(metrics)
                
                processed += 1
                progress_bar.progress(processed / total_scenarios)
        
        # Display Results
        res_df = pd.DataFrame(results_summary)
        
        st.dataframe(res_df.style.format({
            "Current Value": "${:,.0f}",
            "5th %ile Value": "${:,.0f}",
            "Mean P&L": "${:,.0f}",
            "Median P&L": "${:,.0f}",
            "5% VaR (P&L)": "${:,.0f}",
            "Avg Max Drawdown": "{:.2%}",
            "Median Drawdown": "{:.2%}",
            "Win Rate": "{:.1%}",
            "Margin Call %": "{:.1%}",
            "Margin Call @ 5th %ile": lambda x: "Yes" if x > 0.5 else "No",
            "Margin Call @ Median DD": lambda x: "Yes" if x > 0.5 else "No",
            "Initial Margin Req": "${:,.0f}",
            "Traditional Margin": "${:,.0f}",
            "Stock Price @ 5th %ile": "${:.2f}",
            "IV @ 5th %ile": "{:.1%}",
            "Stock Price @ Max DD": "${:.2f}"
        }).highlight_max(axis=0, subset=['5th %ile Value'], color='lightgreen'))
        
        
        # Margin Call Summary Statistics
        if 'Margin Call @ 5th %ile' in res_df.columns and 'Margin Call @ Median DD' in res_df.columns:
            st.subheader("📊 Margin Call Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Overall Margin Call Risk", 
                         f"{res_df['Margin Call %'].mean():.1%}",
                         help="Average % of paths with margin calls across all scenarios")
            
            with col2:
                mc_at_5th = res_df['Margin Call @ 5th %ile'].mean()
                st.metric("Margin Call @ 5th Percentile", 
                         f"{int(mc_at_5th * 100)}% of scenarios",
                         help="How many hedge scenarios trigger margin calls in their worst 5% outcome")
            
            with col3:
                mc_at_median = res_df['Margin Call @ Median DD'].mean()
                st.metric("Margin Call @ Median Drawdown",
                         f"{int(mc_at_median * 100)}% of scenarios",
                         help="How many hedge scenarios trigger margin calls at median drawdown")
        
        # Scatter Plot
        fig_scatter = px.scatter(
            res_df, 
            x="Avg Max Drawdown", 
            y="Mean P&L", 
            hover_name="Name", 
            color="Win Rate",
            size="Win Rate",
            title=f"Risk vs Reward (Expiry: {selected_expiry})",
            height=600
        )
        stock_row = res_df[res_df['Name'] == "Stock Only"].iloc[0]
        fig_scatter.add_vline(x=stock_row['Avg Max Drawdown'], line_dash="dash", line_color="red", annotation_text="Stock Only Risk")
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Heatmap
        st.subheader("Hedge Optimization Heatmap")
        hedged_df = res_df[res_df['Put Qty'] > 0].copy()
        
        if not hedged_df.empty:
            # Pivot data for Heatmap
            # X = Put Qty, Y = Strike, Z = 5th %ile Value
            z_metric = "5th %ile Value"
            pivot_df = hedged_df.pivot(index='Strike', columns='Put Qty', values=z_metric)
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=pivot_df.values,
                x=pivot_df.columns,  # Put Qty
                y=pivot_df.index,    # Strike
                colorscale='RdYlGn',
                colorbar=dict(title=z_metric),
                hoverongaps=False,
                hovertemplate='Put Qty: %{x}<br>Strike: $%{y}<br>5th %ile: $%{z:,.0f}<extra></extra>'
            ))
            
            fig_heatmap.update_layout(
                title=f"Hedge Landscape: {z_metric}",
                xaxis_title='Put Quantity (Contracts)',
                yaxis_title='Strike Price ($)',
                width=1000,
                height=600,
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Median Value Heatmap
        st.subheader("Median Value Heatmap")
        if not hedged_df.empty and 'Median P&L' in hedged_df.columns:
            # Pivot data for Median Value
            median_metric = "Median P&L"
            pivot_median = hedged_df.pivot(index='Strike', columns='Put Qty', values=median_metric)
            
            fig_median = go.Figure(data=go.Heatmap(
                z=pivot_median.values,
                x=pivot_median.columns,  # Put Qty
                y=pivot_median.index,    # Strike
                colorscale='RdYlGn',
                colorbar=dict(title="Median Value"),
                hoverongaps=False,
                hovertemplate='Put Qty: %{x}<br>Strike: $%{y}<br>Median Value: $%{z:,.0f}<extra></extra>'
            ))
            
            fig_median.update_layout(
                title="Hedge Landscape: Median Value",
                xaxis_title='Put Quantity (Contracts)',
                yaxis_title='Strike Price ($)',
                width=1000,
                height=600,
            )
            st.plotly_chart(fig_median, use_container_width=True)
 
