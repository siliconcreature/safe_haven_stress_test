import yfinance as yf
import pandas as pd

def test_options():
    ticker = "NVDA"
    try:
        nvda = yf.Ticker(ticker)
        expirations = nvda.options
        print(f"Expirations found: {len(expirations)}")
        if not expirations:
            print("No expirations found!")
            return

        # Pick one ~90 days out if possible, or just the 3rd one
        target_exp = expirations[min(len(expirations)-1, 4)]
        print(f"Fetching chain for: {target_exp}")
        
        chain = nvda.option_chain(target_exp)
        puts = chain.puts
        
        print(f"\nFound {len(puts)} Puts.")
        print(puts[['strike', 'lastPrice', 'impliedVolatility', 'volume']].head())
        
        # Check specific strike
        print("\nSample ATM Put:")
        # approximate ATM
        current_price = nvda.history(period='1d')['Close'].iloc[-1]
        print(f"Spot: {current_price}")
        
        atm_put = puts.iloc[(puts['strike'] - current_price).abs().argsort()[:1]]
        print(atm_put[['strike', 'lastPrice', 'impliedVolatility']])
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_options()
