import yfinance as yf
import pandas as pd
import streamlit as st

@st.cache_data
def fetch_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical data for a given symbol using yfinance.
    Cached by Streamlit to avoid re-fetching on every rerun.
    """
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date)
    return df

def get_option_expirations(ticker: str) -> list[str]:
    """Fetch available option expiration dates."""
    try:
        t = yf.Ticker(ticker)
        return list(t.options)
    except Exception as e:
        print(f"Error fetching expirations: {e}")
        return []

def get_option_chain(ticker: str, expiration_date: str) -> pd.DataFrame:
    """
    Fetch put option chain for a specific expiration.
    Returns DataFrame with columns: strike, lastPrice, impliedVolatility, inTheMoney
    """
    try:
        t = yf.Ticker(ticker)
        chain = t.option_chain(expiration_date)
        puts = chain.puts
        return puts[['strike', 'lastPrice', 'impliedVolatility', 'inTheMoney', 'volume', 'openInterest']]
    except Exception as e:
        print(f"Error fetching chain: {e}")
        return pd.DataFrame()
