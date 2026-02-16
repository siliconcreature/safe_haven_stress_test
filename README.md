# Safe Haven Stress Test

A Monte Carlo portfolio stress testing tool for analyzing put-protected stock positions using IBKR Portfolio Margin methodology.

## Features

- **Monte Carlo Simulation**: 10,000+ paths using fitted t-distribution for realistic fat-tail modeling
- **IBKR Portfolio Margin**: Accurate margin calculation for put-protected positions
- **Dynamic Volatility**: Rolling volatility with IV anchoring to real option prices
- **Comprehensive Risk Metrics**:
  - 5th Percentile Portfolio Value
  - Median Portfolio Value
  - Margin Call Analysis (overall, 5th percentile, median drawdown)
  - Stock Price & IV at key risk events
- **Interactive Visualizations**:
  - Dual-axis chart: Stock price paths + IV evolution
  - Risk/Reward scatter plot
  - 2D heatmaps: 5th percentile and median equity across strike/quantity combinations

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Streamlit App (Interactive)
```bash
streamlit run app.py
```

### Headless Simulation
```bash
python full_simulation.py
```

## Configuration

Key parameters in the sidebar:
- **Simulation Days**: Duration of simulation (default: 90 days)
- **Target Option Expiry**: Put option maturity (default: 120 days)
- **Number of Paths**: Monte Carlo paths (default: 10,000)
- **Put Quantity Range**: Contracts to test (150-500)
- **Strike Selection**: OTM put strikes (70%-90% of spot)

## Margin Calculation

Uses **IBKR Portfolio Margin** methodology:
- Tests multiple stress scenarios (±15% price movement)
- Margin = Maximum loss across scenarios
- Long puts provide downside protection, reducing margin significantly vs traditional 30% rule

## Output

- **Results Table**: All scenarios with risk metrics
- **Margin Call Summary**: Key statistics at 5th percentile and median drawdown
- **Heatmaps**: Visual optimization surface for strike/quantity selection
- **CSV Export**: Full results from `full_simulation.py`

## Project Structure

```
portfolio-analyzer-py/
├── app.py                    # Streamlit interface
├── full_simulation.py        # Headless batch simulation
├── src/
│   ├── data.py              # Market data fetching
│   ├── analysis.py          # Statistical modeling
│   └── simulation.py        # Monte Carlo engine
└── requirements.txt         # Dependencies
```

## License

MIT
