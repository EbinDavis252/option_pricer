import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd
from datetime import date, timedelta

# --- Core Binomial Option Pricing Model ---
# (This is the same function from the previous response)
def binomial_option_pricer(S, K, T, r, v, N, option_type='call'):
    """
    Calculates European option price and Greeks using the Binomial Tree model.
    """
    # Helper function to avoid re-calculating greeks in vega/rho estimation
    def _pricer_no_greeks(S, K, T, r, v, N, option_type):
        dt = T / N
        u = np.exp(v * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(r * dt) - d) / (u - d)
        
        asset_prices = np.zeros((N + 1, N + 1))
        asset_prices[0, 0] = S
        for i in range(1, N + 1):
            asset_prices[i, 0] = asset_prices[i - 1, 0] * u
            for j in range(1, i + 1):
                asset_prices[i, j] = asset_prices[i - 1, j - 1] * d

        option_values = np.zeros((N + 1, N + 1))
        for j in range(N + 1):
            if option_type == 'call':
                option_values[N, j] = max(0, asset_prices[N, j] - K)
            else:
                option_values[N, j] = max(0, K - asset_prices[N, j])

        for i in range(N - 1, -1, -1):
            for j in range(i + 1):
                option_values[i, j] = np.exp(-r * dt) * (p * option_values[i + 1, j] + (1 - p) * option_values[i + 1, j + 1])
        
        return option_values[0, 0]

    # Main pricing logic
    dt = T / N
    u = np.exp(v * np.sqrt(dt))
    d = 1 / u
    # Ensure p is within a valid range
    if d >= np.exp(r * dt) or np.exp(r * dt) >= u:
        return {'price': np.nan, 'delta': np.nan, 'gamma': np.nan, 'theta': np.nan, 'vega': np.nan, 'rho': np.nan}
    p = (np.exp(r * dt) - d) / (u - d)

    asset_prices = np.zeros((N + 1, N + 1))
    option_values = np.zeros((N + 1, N + 1))
    asset_prices[0, 0] = S
    for i in range(1, N + 1):
        asset_prices[i, 0] = asset_prices[i - 1, 0] * u
        for j in range(1, i + 1):
            asset_prices[i, j] = asset_prices[i - 1, j - 1] * d

    for j in range(N + 1):
        if option_type == 'call':
            option_values[N, j] = max(0, asset_prices[N, j] - K)
        else:
            option_values[N, j] = max(0, K - asset_prices[N, j])

    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            option_values[i, j] = np.exp(-r * dt) * (p * option_values[i + 1, j] + (1 - p) * option_values[i + 1, j + 1])

    delta = (option_values[1, 0] - option_values[1, 1]) / (asset_prices[1, 0] - asset_prices[1, 1])
    delta_up = (option_values[2, 0] - option_values[2, 1]) / (asset_prices[2, 0] - asset_prices[2, 1])
    delta_down = (option_values[2, 1] - option_values[2, 2]) / (asset_prices[2, 1] - asset_prices[2, 2])
    gamma = (delta_up - delta_down) / ((asset_prices[0,0] * (u**2 - d**2))/2)
    theta_per_step = (option_values[2, 1] - option_values[0, 0]) / (2 * dt)
    theta = theta_per_step / 365

    # Use the helper for Vega/Rho to avoid recursion on greeks
    vega_price_up = _pricer_no_greeks(S, K, T, r, v + 0.01, N, option_type)
    vega = (vega_price_up - option_values[0, 0])

    rho_price_up = _pricer_no_greeks(S, K, T, r + 0.01, v, N, option_type)
    rho = (rho_price_up - option_values[0, 0])

    return {
        'price': option_values[0, 0], 'delta': delta, 'gamma': gamma,
        'theta': theta, 'vega': vega, 'rho': rho
    }

# --- Streamlit UI ---

st.set_page_config(layout="wide")
st.title("📈 Equity & Index Option Pricing Dashboard")
st.markdown("This application calculates European option prices and their Greeks using a Binomial Tree model. You can fetch live data for any stock or index from Yahoo Finance.")

# --- Sidebar for User Inputs ---
st.sidebar.header("⚙️ Option Parameters")

# Ticker input and data fetching
ticker_symbol = st.sidebar.text_input("Enter Ticker (e.g., ^NSEI for Nifty 50)", "^NSEI")
if st.sidebar.button("Fetch Live Data"):
    try:
        ticker_data = yf.Ticker(ticker_symbol)
        latest_price = ticker_data.history(period='1d')['Close'].iloc[0]
        st.session_state.S = latest_price # Use session state to store price
        st.sidebar.success(f"Fetched {ticker_symbol} price: ₹{latest_price:.2f}")
    except Exception as e:
        st.sidebar.error(f"Could not fetch data. Error: {e}")

# Use session state to pre-fill or default
S = st.sidebar.number_input("Underlying Asset Price (S)", value=st.session_state.get('S', 23500.0), step=100.0)
K = st.sidebar.number_input("Strike Price (K)", value=23600.0, step=100.0)

# User-friendly date input
today = date.today()
exp_date = st.sidebar.date_input("Expiration Date", today + timedelta(days=30))
T = (exp_date - today).days / 365.0
st.sidebar.write(f"Time to Expiration (T): {T*365:.0f} days ({T:.4f} years)")

# Sliders for r and v
r = st.sidebar.slider("Risk-Free Interest Rate (r)", 0.0, 0.2, 0.07, 0.005, format="%.3f")
v = st.sidebar.slider("Volatility (v)", 0.01, 1.00, 0.20, 0.01, format="%.2f")
N = st.sidebar.number_input("Number of Binomial Steps (N)", 50, 500, 100, step=10)


# --- Main Panel for Results ---
if T > 0:
    st.header("📊 Calculated Option Prices & Greeks")
    col1, col2 = st.columns(2)

    # Calculate Call Option
    call_option = binomial_option_pricer(S, K, T, r, v, N, 'call')
    with col1:
        st.subheader("Call Option")
        st.metric(label="Option Price", value=f"₹{call_option['price']:.2f}")
        st.markdown(f"""
        - **Delta:** `{call_option['delta']:.4f}`
        - **Gamma:** `{call_option['gamma']:.4f}`
        - **Theta (per day):** `₹{call_option['theta']:.2f}`
        - **Vega (per 1% vol):** `₹{call_option['vega']:.2f}`
        - **Rho (per 1% rate):** `₹{call_option['rho']:.2f}`
        """)

    # Calculate Put Option
    put_option = binomial_option_pricer(S, K, T, r, v, N, 'put')
    with col2:
        st.subheader("Put Option")
        st.metric(label="Option Price", value=f"₹{put_option['price']:.2f}")
        st.markdown(f"""
        - **Delta:** `{put_option['delta']:.4f}`
        - **Gamma:** `{put_option['gamma']:.4f}`
        - **Theta (per day):** `₹{put_option['theta']:.2f}`
        - **Vega (per 1% vol):** `₹{put_option['vega']:.2f}`
        - **Rho (per 1% rate):** `₹{put_option['rho']:.2f}`
        """)
else:
    st.warning("Please select a future expiration date.")

# --- Scenario Analysis Section ---
st.header("🔬 Scenario Analysis")
st.markdown("Analyze how the option price changes with different factors.")

# 1. Analysis vs. Underlying Price
st.subheader("Impact of Underlying Price")
price_range = np.linspace(S * 0.9, S * 1.1, 20)
call_prices_S = [binomial_option_pricer(p, K, T, r, v, N, 'call')['price'] for p in price_range]
put_prices_S = [binomial_option_pricer(p, K, T, r, v, N, 'put')['price'] for p in price_range]
price_df = pd.DataFrame({'Underlying Price': price_range, 'Call Price': call_prices_S, 'Put Price': put_prices_S}).set_index('Underlying Price')
st.line_chart(price_df)

# 2. Analysis vs. Volatility
st.subheader("Impact of Volatility")
vol_range = np.linspace(max(0.05, v * 0.5), v * 2, 20)
call_prices_v = [binomial_option_pricer(S, K, T, r, vol, N, 'call')['price'] for vol in vol_range]
put_prices_v = [binomial_option_pricer(S, K, T, r, vol, N, 'put')['price'] for vol in vol_range]
vol_df = pd.DataFrame({'Volatility': vol_range, 'Call Price': call_prices_v, 'Put Price': put_prices_v}).set_index('Volatility')
st.line_chart(vol_df)

# 3. Time Decay (Theta) Analysis
st.subheader("Impact of Time to Expiration (Theta Decay)")
days_range = np.arange(max(1, (exp_date - today).days), 0, -1)
time_range_T = days_range / 365.0
call_prices_t = [binomial_option_pricer(S, K, t, r, v, N, 'call')['price'] for t in time_range_T]
put_prices_t = [binomial_option_pricer(S, K, t, r, v, N, 'put')['price'] for t in time_range_T]
time_df = pd.DataFrame({'Days to Expiration': days_range, 'Call Price': call_prices_t, 'Put Price': put_prices_t}).set_index('Days to Expiration')
st.line_chart(time_df)
