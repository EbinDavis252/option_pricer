import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd
from scipy.stats import norm
from datetime import date, timedelta

# --- Model Implementations ---

# 1. Black-Scholes-Merton Model
def black_scholes_pricer(S, K, T, r, v, option_type='call'):
    """
    Calculates European option price and Greeks using the Black-Scholes model.
    """
    if T <= 0: # Handle expired options
        price = max(0, S - K) if option_type == 'call' else max(0, K - S)
        return {'price': price, 'delta': 1 if S > K else 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}

    d1 = (np.log(S / K) + (r + 0.5 * v ** 2) * T) / (v * np.sqrt(T))
    d2 = d1 - v * np.sqrt(T)
    
    if option_type == 'call':
        price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
        delta = norm.cdf(d1)
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else: # put
        price = (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
        delta = norm.cdf(d1) - 1
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)

    gamma = norm.pdf(d1) / (S * v * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    theta = -(S * norm.pdf(d1) * v) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * (norm.cdf(d2) if option_type == 'call' else -norm.cdf(-d2))
    
    return {
        'price': price, 'delta': delta, 'gamma': gamma,
        'theta': theta / 365, 'vega': vega / 100, 'rho': rho / 100
    }

# 2. Binomial Option Pricing Model
def binomial_option_pricer(S, K, T, r, v, N, option_type='call'):
    if T <= 0:
        price = max(0, S - K) if option_type == 'call' else max(0, K - S)
        return {'price': price}
        
    dt = T / N
    u = np.exp(v * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    if not (0 < p < 1):
        return {'price': np.nan}

    # Initialize asset prices at maturity
    asset_prices = S * d**np.arange(N, -1, -1) * u**np.arange(0, N + 1, 1)
    
    if option_type == 'call':
        option_values = np.maximum(0, asset_prices - K)
    else: # put
        option_values = np.maximum(0, K - asset_prices)

    # Step back through the tree
    for i in range(N - 1, -1, -1):
        option_values = np.exp(-r * dt) * (p * option_values[1:] + (1 - p) * option_values[:-1])
    
    return {'price': option_values[0]}


# --- Data and Helper Functions ---

@st.cache_data
def get_nifty50_tickers():
    # List of Nifty 50 stocks with their Yahoo Finance tickers
    return {
        "NIFTY 50 Index": "^NSEI",
        "Reliance Industries": "RELIANCE.NS", "HDFC Bank": "HDFCBANK.NS", "ICICI Bank": "ICICIBANK.NS",
        "Infosys": "INFY.NS", "Tata Consultancy Services": "TCS.NS", "Hindustan Unilever": "HINDUNILVR.NS",
        "ITC": "ITC.NS", "Larsen & Toubro": "LT.NS", "Bajaj Finance": "BAJFINANCE.NS",
        "State Bank of India": "SBIN.NS", "Bharti Airtel": "BHARTIARTL.NS", "Kotak Mahindra Bank": "KOTAKBANK.NS",
        "Axis Bank": "AXISBANK.NS", "NTPC": "NTPC.NS", "Maruti Suzuki": "MARUTI.NS",
        "Sun Pharmaceutical": "SUNPHARMA.NS", "Tata Motors": "TATAMOTORS.NS", "Tata Steel": "TATASTEEL.NS",
        "Power Grid Corporation": "POWERGRID.NS", "Titan Company": "TITAN.NS", "Asian Paints": "ASIANPAINT.NS",
        "UltraTech Cement": "ULTRACEMCO.NS", "Wipro": "WIPRO.NS", "Adani Enterprises": "ADANIENT.NS",
        "Mahindra & Mahindra": "M&M.NS", "JSW Steel": "JSWSTEEL.NS", "Bajaj Finserv": "BAJAJFINSV.NS",
        "HCL Technologies": "HCLTECH.NS", "Nestle India": "NESTLEIND.NS", "Grasim Industries": "GRASIM.NS",
        "Cipla": "CIPLA.NS", "Dr. Reddy's Laboratories": "DRREDDY.NS", "Adani Ports": "ADANIPORTS.NS",
        "Britannia Industries": "BRITANNIA.NS", "Hindalco Industries": "HINDALCO.NS", "Eicher Motors": "EICHERMOT.NS",
        "Coal India": "COALINDIA.NS", "Hero MotoCorp": "HEROMOTOCO.NS", "Divi's Laboratories": "DIVISLAB.NS",
        "UPL": "UPL.NS", "SBI Life Insurance": "SBILIFE.NS", "HDFC Life Insurance": "HDFCLIFE.NS",
        "Tech Mahindra": "TECHM.NS", "Apollo Hospitals": "APOLLOHOSP.NS", "ONGC": "ONGC.NS",
        "LTIMindtree": "LTIM.NS", "Bajaj Auto": "BAJAJ-AUTO.NS", "Tata Consumer Products": "TATACONSUM.NS"
    }

@st.cache_data(ttl=900) # Cache data for 15 minutes
def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")
    if hist.empty:
        return None, None
    latest_price = hist['Close'].iloc[-1]
    return latest_price, hist

def calculate_historical_volatility(hist_data):
    if hist_data is None or len(hist_data) < 2:
        return 0.0
    log_returns = np.log(hist_data['Close'] / hist_data['Close'].shift(1))
    return np.std(log_returns) * np.sqrt(252) # 252 trading days in a year

def generate_binomial_tree_data(S, K, T, r, v, N, option_type):
    """Generates the full data for asset and option price trees."""
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
        if option_type == 'call':
            option_tree[j, N] = max(0, asset_tree[j, N] - K)
        else:
            option_tree[j, N] = max(0, K - asset_tree[j, N])
    
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            option_tree[j, i] = np.exp(-r * dt) * (p * option_tree[j, i + 1] + (1 - p) * option_tree[j + 1, i + 1])
            
    return asset_tree, option_tree

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Financial Decision-Making Dashboard")
st.title("📈 Financial Decision-Making Dashboard")
st.markdown("An advanced tool for option pricing, volatility analysis, and strategy visualization.")

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("⚙️ Core Parameters")
    
    nifty50_tickers = get_nifty50_tickers()
    selected_company_name = st.selectbox("Select Nifty 50 Company or Index", list(nifty50_tickers.keys()))
    ticker_symbol = nifty50_tickers[selected_company_name]
    
    latest_price, hist_data = fetch_stock_data(ticker_symbol)
    
    if latest_price is None:
        st.error(f"Could not fetch data for {ticker_symbol}. Please try another ticker.")
        st.stop()
        
    st.success(f"**{selected_company_name}**\n\nLast Price: **₹{latest_price:.2f}**")
    
    S = st.number_input("Underlying Asset Price (S)", value=latest_price, format="%.2f")
    K = st.number_input("Strike Price (K)", value=round(latest_price, -2), step=100.0, format="%.2f")
    
    today = date.today()
    exp_date = st.date_input("Expiration Date", today + timedelta(days=30))
    T = (exp_date - today).days / 365.0
    st.write(f"Time to Expiration (T): {max(0, (exp_date - today).days)} days")
    
    r = st.slider("Risk-Free Interest Rate (r)", 0.0, 0.2, 0.071, 0.001, format="%.3f")
    
    hv = calculate_historical_volatility(hist_data)
    v = st.slider("Volatility (v)", 0.01, 2.00, value=hv, step=0.01, format="%.2f", 
                  help=f"Annualized historical volatility is {hv:.2f}. Adjust based on your market view.")

# --- Main Panel with Integrated Layout ---

# --- 1. Key Pricing Results ---
st.subheader("🧮 Option Price Comparison")
bsm_call = black_scholes_pricer(S, K, T, r, v, 'call')
bsm_put = black_scholes_pricer(S, K, T, r, v, 'put')
binom_call = binomial_option_pricer(S, K, T, r, v, 100, 'call')
binom_put = binomial_option_pricer(S, K, T, r, v, 100, 'put')

col1, col2 = st.columns(2)
with col1:
    with st.container(border=True):
        st.markdown("#### Call Option")
        st.metric("Black-Scholes Price", f"₹{bsm_call['price']:.2f}")
        st.metric("Binomial Tree Price (100 steps)", f"₹{binom_call.get('price', 0):.2f}", 
                  delta=f"{binom_call.get('price', 0) - bsm_call['price']:.2f} vs BSM", delta_color="off")
        
        with st.expander("View Greeks (Industry Standard: Black-Scholes)"):
            st.markdown(f"""
            - **Delta:** `{bsm_call['delta']:.4f}`
            - **Gamma:** `{bsm_call['gamma']:.4f}`
            - **Theta:** `₹{bsm_call['theta']:.2f}` (per day)
            - **Vega:** `₹{bsm_call['vega']:.2f}` (per 1% vol change)
            - **Rho:** `₹{bsm_call['rho']:.2f}` (per 1% rate change)
            """)

with col2:
    with st.container(border=True):
        st.markdown("#### Put Option")
        st.metric("Black-Scholes Price", f"₹{bsm_put['price']:.2f}")
        st.metric("Binomial Tree Price (100 steps)", f"₹{binom_put.get('price', 0):.2f}", 
                  delta=f"{binom_put.get('price', 0) - bsm_put['price']:.2f} vs BSM", delta_color="off")

        with st.expander("View Greeks (Industry Standard: Black-Scholes)"):
            st.markdown(f"""
            - **Delta:** `{bsm_put['delta']:.4f}`
            - **Gamma:** `{bsm_put['gamma']:.4f}`
            - **Theta:** `₹{bsm_put['theta']:.2f}`
            - **Vega:** `₹{bsm_put['vega']:.2f}`
            - **Rho:** `₹{bsm_put['rho']:.2f}`
            """)

st.divider()

# --- 2. Visual Analysis ---
st.subheader("📊 Visual Analysis")
col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.markdown("#### Strategy Payoff at Expiration")
        
        c1_payoff, c2_payoff = st.columns(2)
        option_type_payoff = c1_payoff.radio("Select Option", ('Call', 'Put'), horizontal=True, key="payoff_choice")
        model_for_payoff = c2_payoff.radio("Use premium from:", ('Black-Scholes', 'Binomial'), horizontal=True, key="model_choice")

        premium = (bsm_call['price'] if option_type_payoff == 'Call' else bsm_put['price']) if model_for_payoff == 'Black-Scholes' else (binom_call.get('price', 0) if option_type_payoff == 'Call' else binom_put.get('price', 0))
        
        price_at_exp = np.linspace(S * 0.8, S * 1.2, 100)
        payoff = np.maximum(price_at_exp - K, 0) - premium if option_type_payoff == 'Call' else np.maximum(K - price_at_exp, 0) - premium
        breakeven = K + premium if option_type_payoff == 'Call' else K - premium
            
        payoff_df = pd.DataFrame({'Profit / Loss': payoff}, index=price_at_exp)
        st.line_chart(payoff_df)
        
        c1, c2 = st.columns(2)
        c1.metric("Breakeven Price", f"₹{breakeven:.2f}")
        c2.metric("Max Loss", f"₹{-premium:.2f}")

with col2:
    with st.container(border=True):
        st.markdown(f"#### Volatility Analysis for {selected_company_name}")
        st.metric("1-Year Historical Volatility", f"{hv:.2%}")
        st.line_chart(hist_data['Close'], use_container_width=True)
        st.caption("Historical volatility is based on the standard deviation of logarithmic returns over the last year.")

st.divider()

# --- 3. Binomial Tree Visualization ---
st.subheader("🌳 Binomial Tree Construction")
with st.container(border=True):
    st.markdown("This section visualizes the underlying asset price movements and the corresponding option values at each node of the binomial tree.")
    
    N_viz = st.slider("Select number of steps to visualize", 2, 10, 4, 1)
    option_type_viz = st.radio("Select Option Type to Visualize", ('Call', 'Put'), horizontal=True, key="viz_choice")
    
    # --- Tree Calculations ---
    dt_viz = T / N_viz
    u_viz = np.exp(v * np.sqrt(dt_viz))
    d_viz = 1 / u_viz
    p_viz = (np.exp(r * dt_viz) - d_viz) / (u_viz - d_viz)
    
    # --- Display Formulas ---
    st.markdown("#### Core Formulas")
    f_col1, f_col2, f_col3 = st.columns(3)
    f_col1.latex(fr"u = e^{{\sigma \sqrt{{\Delta t}}}} = {u_viz:.4f}")
    f_col2.latex(fr"d = 1/u = {d_viz:.4f}")
    f_col3.latex(fr"p = \frac{{e^{{r\Delta t}} - d}}{{u - d}} = {p_viz:.4f}")
    st.caption("Where Δt is the time per step (T/N).")
    
    st.markdown("---")
    
    # --- Generate and Display Tree ---
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
        st.error("Arbitrage opportunity detected (Risk-Neutral Probability 'p' is not between 0 and 1). Please adjust parameters.")
        
    st.markdown("---")
    st.markdown("#### Valuation Process")
    st.markdown("""
    1.  **Asset prices** are projected forward from today (Step 0) to expiration (the final step). At each node, the price can either move up by a factor of *u* or down by a factor of *d*.
    2.  **Option valuation** begins at the final nodes (expiration). The value is its intrinsic worth: `max(0, Asset Price - Strike)` for a call, or `max(0, Strike - Asset Price)` for a put.
    3.  **Backward induction** is used to find the option's value today. The value at any node is the discounted expected value of the two possible future nodes, weighted by the risk-neutral probability *p*.
    """)
