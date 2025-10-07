import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd
from datetime import date, timedelta

# --- Model Implementations ---

# Binomial Option Pricing Model with Greeks
def binomial_option_pricer(S, K, T, r, v, N, option_type='call'):
    """
    Calculates European option price and Greeks using the Binomial Tree model.
    """
    # Base case for expired or invalid time
    if T <= 0:
        price = max(0, S - K) if option_type == 'call' else max(0, K - S)
        return {'price': price, 'delta': 1 if S > K else 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}

    # Helper function for Vega/Rho calculation
    def _pricer_just_price(S_h, K_h, T_h, r_h, v_h, N_h, option_type_h):
        if T_h <= 0: return max(0, S_h - K_h) if option_type_h == 'call' else max(0, K_h - S_h)
        dt_h = T_h / N_h
        u_h = np.exp(v_h * np.sqrt(dt_h))
        d_h = 1 / u_h
        p_h = (np.exp(r_h * dt_h) - d_h) / (u_h - d_h)
        if not (0 < p_h < 1): return np.nan
        asset_prices_h = S_h * d_h**np.arange(N_h, -1, -1) * u_h**np.arange(0, N_h + 1, 1)
        option_values_h = np.maximum(0, asset_prices_h - K_h) if option_type_h == 'call' else np.maximum(0, K_h - asset_prices_h)
        for i in range(N_h - 1, -1, -1):
            option_values_h = np.exp(-r_h * dt_h) * (p_h * option_values_h[1:] + (1 - p_h) * option_values_h[:-1])
        return option_values_h[0]

    # Main pricing logic
    dt = T / N
    u = np.exp(v * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    if not (0 < p < 1):
        return {'price': np.nan, 'delta': np.nan, 'gamma': np.nan, 'theta': np.nan, 'vega': np.nan, 'rho': np.nan}

    asset_tree, option_tree = generate_binomial_tree_data(S, K, T, r, v, N, option_type)
    
    # Greeks Calculation
    price = option_tree[0, 0]
    delta = (option_tree[0, 1] - option_tree[1, 1]) / (asset_tree[0, 1] - asset_tree[1, 1])
    delta_up = (option_tree[0, 2] - option_tree[1, 2]) / (asset_tree[0, 2] - asset_tree[1, 2])
    delta_down = (option_tree[1, 2] - option_tree[2, 2]) / (asset_tree[1, 2] - asset_tree[2, 2])
    gamma = (delta_up - delta_down) / (0.5 * (asset_tree[0, 2] - asset_tree[2, 2]))
    theta = (option_tree[1, 2] - option_tree[0, 0]) / (2 * dt)
    vega = (_pricer_just_price(S, K, T, r, v + 0.01, N, option_type) - price)
    rho = (_pricer_just_price(S, K, T, r + 0.01, v, N, option_type) - price)

    return {'price': price, 'delta': delta, 'gamma': gamma, 'theta': theta / 365, 'vega': vega, 'rho': rho}

# --- Data and Helper Functions ---
@st.cache_data
def get_nifty50_tickers():
    return {
        "NIFTY 50 Index": "^NSEI", "Reliance Industries": "RELIANCE.NS", "HDFC Bank": "HDFCBANK.NS", "ICICI Bank": "ICICIBANK.NS",
        "Infosys": "INFY.NS", "Tata Consultancy Services": "TCS.NS", "Hindustan Unilever": "HINDUNILVR.NS", "ITC": "ITC.NS",
        "Larsen & Toubro": "LT.NS", "Bajaj Finance": "BAJFINANCE.NS", "State Bank of India": "SBIN.NS", "Bharti Airtel": "BHARTIARTL.NS",
        "Kotak Mahindra Bank": "KOTAKBANK.NS", "Axis Bank": "AXISBANK.NS", "NTPC": "NTPC.NS", "Maruti Suzuki": "MARUTI.NS",
        "Sun Pharmaceutical": "SUNPHARMA.NS", "Tata Motors": "TATAMOTORS.NS", "Tata Steel": "TATASTEEL.NS", "Power Grid Corporation": "POWERGRID.NS",
        "Titan Company": "TITAN.NS", "Asian Paints": "ASIANPAINT.NS", "UltraTech Cement": "ULTRACEMCO.NS", "Wipro": "WIPRO.NS",
        "Adani Enterprises": "ADANIENT.NS", "Mahindra & Mahindra": "M&M.NS", "JSW Steel": "JSWSTEEL.NS", "Bajaj Finserv": "BAJAJFINSV.NS",
        "HCL Technologies": "HCLTECH.NS", "Nestle India": "NESTLEIND.NS", "Grasim Industries": "GRASIM.NS", "Cipla": "CIPLA.NS",
        "Dr. Reddy's Laboratories": "DRREDDY.NS", "Adani Ports": "ADANIPORTS.NS", "Britannia Industries": "BRITANNIA.NS",
        "Hindalco Industries": "HINDALCO.NS", "Eicher Motors": "EICHERMOT.NS", "Coal India": "COALINDIA.NS", "Hero MotoCorp": "HEROMOTOCO.NS",
        "Divi's Laboratories": "DIVISLAB.NS", "UPL": "UPL.NS", "SBI Life Insurance": "SBILIFE.NS", "HDFC Life Insurance": "HDFCLIFE.NS",
        "Tech Mahindra": "TECHM.NS", "Apollo Hospitals": "APOLLOHOSP.NS", "ONGC": "ONGC.NS", "LTIMindtree": "LTIM.NS",
        "Bajaj Auto": "BAJAJ-AUTO.NS", "Tata Consumer Products": "TATACONSUM.NS"
    }

@st.cache_data(ttl=900)
def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    hist = stock.history(period="1y")
    if hist.empty:
        return None, None, None
    latest_price = hist['Close'].iloc[-1]
    return latest_price, hist, info

def calculate_historical_volatility(hist_data):
    if hist_data is None or len(hist_data) < 2: return 0.0
    log_returns = np.log(hist_data['Close'] / hist_data['Close'].shift(1))
    return np.std(log_returns) * np.sqrt(252)

def generate_binomial_tree_data(S, K, T, r, v, N, option_type):
    dt = T / N
    u = np.exp(v * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    asset_tree = np.zeros((N + 1, N + 1))
    option_tree = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(i + 1):
            asset_tree[j, i] = S * (u ** (i - j)) * (d ** j)
    for j in range(N + 1):
        option_tree[j, N] = max(0, asset_tree[j, N] - K) if option_type == 'call' else max(0, K - asset_tree[j, N])
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            option_tree[j, i] = np.exp(-r * dt) * (p * option_tree[j, i + 1] + (1 - p) * option_tree[j + 1, i + 1])
    return asset_tree, option_tree

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Option Edge", page_icon="💡")

st.title("💡Option Edge")
st.markdown("##### A Binomial Model Dashboard for Advanced Option Analysis")

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("⚙️ Market & Model Inputs")
    nifty50_tickers = get_nifty50_tickers()
    selected_company_name = st.selectbox("Select Asset", list(nifty50_tickers.keys()))
    ticker_symbol = nifty50_tickers[selected_company_name]
    
    latest_price, hist_data, info = fetch_stock_data(ticker_symbol)
    
    if latest_price is None:
        st.error(f"Data unavailable for {ticker_symbol}. Please choose another asset.")
        st.stop()
        
    st.markdown(f"### {info.get('longName', selected_company_name)}")
    st.metric("Last Market Price", f"₹{latest_price:.2f}", f"{latest_price - info.get('previousClose', 0):.2f} (₹)")

    with st.expander("View Market Data"):
        col1, col2 = st.columns(2)
        col1.metric("Previous Close", f"₹{info.get('previousClose', 0):.2f}")
        col2.metric("Open", f"₹{info.get('open', 0):.2f}")
        col1.metric("Day High", f"₹{info.get('dayHigh', 0):.2f}")
        col2.metric("Day Low", f"₹{info.get('dayLow', 0):.2f}")

    st.header("🔧 Option Parameters")
    S = st.number_input("Underlying Price (S)", value=latest_price, format="%.2f")
    K = st.number_input("Strike Price (K)", value=round(latest_price, -2), step=100.0, format="%.2f")
    
    today = date.today()
    exp_date = st.date_input("Expiration Date", today + timedelta(days=30))
    T = (exp_date - today).days / 365.0
    st.write(f"Days to Expiry: {max(0, (exp_date - today).days)}")
    
    r = st.slider("Risk-Free Rate (r)", 0.0, 0.2, 0.071, 0.001, format="%.3f")
    hv = calculate_historical_volatility(hist_data)
    v = st.slider("Volatility (v)", 0.01, 2.00, value=hv, step=0.01, format="%.2f", 
                  help=f"Annualized historical volatility is {hv:.2f}. Adjust based on your market view.")

# --- Main Panel ---
st.subheader("🧮 Binomial Model Pricing & Risk Analysis")
binom_steps = 100
binom_call = binomial_option_pricer(S, K, T, r, v, binom_steps, 'call')
binom_put = binomial_option_pricer(S, K, T, r, v, binom_steps, 'put')

col1, col2 = st.columns(2)
with col1:
    with st.container(border=True):
        st.markdown("#### Call Option")
        st.metric(f"Option Price", f"₹{binom_call.get('price', 0):.2f}")
        st.markdown("---")
        st.markdown("**Risk Greeks (Binomial Approximation)**")
        g1, g2 = st.columns(2)
        g1.metric("Delta", f"{binom_call['delta']:.4f}")
        g2.metric("Gamma", f"{binom_call['gamma']:.4f}")
        g1.metric("Theta (per day)", f"₹{binom_call['theta']:.2f}")
        g2.metric("Vega (per 1% vol)", f"₹{binom_call['vega']:.2f}")
        g1.metric("Rho (per 1% rate)", f"₹{binom_call['rho']:.2f}")

with col2:
    with st.container(border=True):
        st.markdown("#### Put Option")
        st.metric(f"Option Price", f"₹{binom_put.get('price', 0):.2f}")
        st.markdown("---")
        st.markdown("**Risk Greeks (Binomial Approximation)**")
        g1, g2 = st.columns(2)
        g1.metric("Delta", f"{binom_put['delta']:.4f}")
        g2.metric("Gamma", f"{binom_put['gamma']:.4f}")
        g1.metric("Theta (per day)", f"₹{binom_put['theta']:.2f}")
        g2.metric("Vega (per 1% vol)", f"₹{binom_put['vega']:.2f}")
        g1.metric("Rho (per 1% rate)", f"₹{binom_put['rho']:.2f}")

st.divider()

st.subheader("📊 Visual Analysis Suite")
col1, col2 = st.columns(2)
with col1:
    with st.container(border=True):
        st.markdown("#### Strategy Payoff at Expiration")
        option_type_payoff = st.radio("Select Option for Payoff", ('Call', 'Put'), horizontal=True)
        premium = binom_call.get('price', 0) if option_type_payoff == 'Call' else binom_put.get('price', 0)
        price_at_exp = np.linspace(S * 0.8, S * 1.2, 100)
        payoff = np.maximum(price_at_exp - K, 0) - premium if option_type_payoff == 'Call' else np.maximum(K - price_at_exp, 0) - premium
        breakeven = K + premium if option_type_payoff == 'Call' else K - premium
        payoff_df = pd.DataFrame({'Profit / Loss (₹)': payoff}, index=price_at_exp)
        st.area_chart(payoff_df)
        p1, p2 = st.columns(2)
        p1.metric("Breakeven Price", f"₹{breakeven:.2f}")
        p2.metric("Max Loss (Premium)", f"₹{-premium:.2f}")

with col2:
    with st.container(border=True):
        st.markdown(f"#### Volatility Analysis")
        st.metric(f"1-Year Historical Volatility for {selected_company_name}", f"{hv:.2%}")
        st.line_chart(hist_data['Close'], use_container_width=True)
        st.caption("Historical volatility is based on the standard deviation of logarithmic returns over the last year.")

st.divider()

st.subheader("🌳 Binomial Tree Construction")
with st.container(border=True):
    st.markdown("This section visualizes the underlying asset price movements and the corresponding option values at each node of the binomial tree.")
    c1, c2 = st.columns([1,2])
    N_viz = c1.slider("Steps to visualize", 2, 8, 4, 1)
    option_type_viz = c2.radio("Option Type to Visualize", ('Call', 'Put'), horizontal=True, key="viz_choice")
    
    dt_viz = T / N_viz
    u_viz = np.exp(v * np.sqrt(dt_viz))
    d_viz = 1 / u_viz
    p_viz = (np.exp(r * dt_viz) - d_viz) / (u_viz - d_viz)
    
    st.markdown("---")
    
    if 0 < p_viz < 1:
        asset_tree_viz, option_tree_viz = generate_binomial_tree_data(S, K, T, r, v, N_viz, option_type_viz)
        for i in range(N_viz + 1):
            st.markdown(f"**Time Step {i}**")
            cols = st.columns(i + 1)
            for j in range(i + 1):
                with cols[j]:
                    st.metric(label=f"Asset Price", value=f"₹{asset_tree_viz[j, i]:.2f}")
                    st.info(f"Option Value: ₹{option_tree_viz[j, i]:.2f}")
    else:
        st.error("Arbitrage opportunity detected (p is not between 0 and 1). Please adjust parameters.")
        
    st.markdown("---")
    with st.expander("Learn about the Valuation Process"):
        st.markdown("""
        1.  **Asset Price Projection:** The tree projects asset prices from today (Step 0) to expiration. At each step, the price moves up by factor `u` or down by factor `d`.
        2.  **Terminal Valuation:** At expiration (the final step), the option's value is its intrinsic worth: `max(0, Asset Price - Strike)` for a call, or `max(0, Strike - Asset Price)` for a put.
        3.  **Backward Induction:** The model then works backward. The value at any earlier node is the discounted expected value of the two subsequent nodes, weighted by the risk-neutral probability `p`. This process continues until it reaches Step 0, giving the option's current theoretical price.
        """)

