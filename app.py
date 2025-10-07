import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd
from datetime import date, timedelta
from streamlit_agraph import agraph, Node, Edge, Config

# --- UI Styling ---
def add_custom_css():
    st.markdown("""
    <style>
    /* Main app background */
    [data-testid="stAppViewContainer"] {
        background-image: linear-gradient(to right top, #e6f2ff, #edf5ff, #f3f8ff, #f9fbff, #ffffff);
        background-size: cover;
    }
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #f1f8ff;
    }
    /* Remove header background */
    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0);
    }
    </style>
    """, unsafe_allow_html=True)

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
    # Ensure there are enough nodes for Greek calculation
    if N < 2:
        return {'price': price, 'delta': np.nan, 'gamma': np.nan, 'theta': np.nan, 'vega': np.nan, 'rho': np.nan}
        
    delta = (option_tree[0, 1] - option_tree[1, 1]) / (asset_tree[0, 1] - asset_tree[1, 1])
    delta_up = (option_tree[0, 2] - option_tree[1, 2]) / (asset_tree[0, 2] - asset_tree[1, 2])
    delta_down = (option_tree[1, 2] - option_tree[2, 2]) / (asset_tree[1, 2] - asset_tree[2, 2])
    gamma = (delta_up - delta_down) / (0.5 * (asset_tree[0, 2] - asset_tree[2, 2]))
    theta = (option_tree[1, 1] - option_tree[0, 0]) / dt # Theta per time step dt
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

@st.cache_data(ttl=900) # Cache for 15 mins
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

def create_tree_graph_elements(asset_tree, option_tree):
    """Generates nodes and edges for the agraph visualization."""
    nodes = []
    edges = []
    n_steps = asset_tree.shape[1] - 1

    for i in range(n_steps + 1): # Time steps
        for j in range(i + 1):   # Nodes at each time step
            node_id = f'T{i}N{j}'
            asset_price = asset_tree[j, i]
            option_price = option_tree[j, i]
            
            label = f"Asset: ₹{asset_price:.2f}\nOption: ₹{option_price:.2f}"
            nodes.append(Node(id=node_id, label=label, shape="box", 
                              color="#d1e4f6", font={'color': '#003366', 'size': 18}))

            if i < n_steps:
                up_node_id = f'T{i+1}N{j}'
                edges.append(Edge(source=node_id, target=up_node_id, label='Up Move (u)', color="#28a745"))
                down_node_id = f'T{i+1}N{j+1}'
                edges.append(Edge(source=node_id, target=down_node_id, label='Down Move (d)', color="#dc3545"))
    return nodes, edges

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Option Edge")

# Apply the custom CSS
add_custom_css()

st.title("Option Edge")
st.markdown("##### A Professional Binomial Model for Financial Decision-Making")

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("Market & Model Inputs")
    nifty50_tickers = get_nifty50_tickers()
    selected_company_name = st.selectbox("Select Asset", list(nifty50_tickers.keys()))
    ticker_symbol = nifty50_tickers[selected_company_name]
    
    latest_price, hist_data, info = fetch_stock_data(ticker_symbol)
    
    if latest_price is None:
        st.error(f"Data unavailable for {ticker_symbol}. Please choose another asset.")
        st.stop()

    st.header("Option Strategy Parameters")
    S = st.number_input("Current Asset Price (S)", value=latest_price, format="%.2f")
    K = st.number_input("Strike Price (K)", value=round(latest_price, -2), step=100.0, format="%.2f")
    
    today = date.today()
    exp_date = st.date_input("Expiration Date", today + timedelta(days=30))
    T = (exp_date - today).days / 365.0
    st.write(f"Days to Expiry: {max(0, (exp_date - today).days)}")
    
    st.header("Market Assumptions")
    r = st.slider("Risk-Free Interest Rate (%)", 0.0, 20.0, 7.1, 0.1, format="%.1f") / 100
    hv = calculate_historical_volatility(hist_data)
    v = st.slider("Implied Volatility (%)", 1.0, 200.0, hv * 100, 1.0, format="%.1f", 
                  help=f"The asset's 1-year historical volatility is {hv:.1%}. Adjust based on your forecast.") / 100

# --- Main Panel ---

# --- Selected Company Data ---
st.subheader(f"Live Market Snapshot: {info.get('longName', selected_company_name)}")
with st.container(border=True):
    price_change = latest_price - info.get('previousClose', latest_price)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Last Market Price", f"₹{latest_price:.2f}", f"{price_change:.2f} (₹)")
    c2.metric("Previous Close", f"₹{info.get('previousClose', 0):.2f}")
    c3.metric("Open", f"₹{info.get('open', 0):.2f}")
    c4.metric("Day High", f"₹{info.get('dayHigh', 0):.2f}")
    c5.metric("Day Low", f"₹{info.get('dayLow', 0):.2f}")

st.divider()

st.subheader("Binomial Model Pricing & Risk Analysis")
st.markdown("The core of the dashboard, providing the theoretical option price based on the Binomial model and the critical risk metrics (Greeks).")
binom_steps = 100
binom_call = binomial_option_pricer(S, K, T, r, v, binom_steps, 'call')
binom_put = binomial_option_pricer(S, K, T, r, v, binom_steps, 'put')

col1, col2 = st.columns(2)
with col1:
    with st.container(border=True):
        st.markdown("#### Call Option (Right to Buy)")
        st.metric(f"Theoretical Option Price", f"₹{binom_call.get('price', 0):.2f}")
        st.markdown("---")
        st.markdown("**Key Risk Metrics (The Greeks)**")
        g1, g2 = st.columns(2)
        g1.metric("Delta", f"{binom_call.get('delta', 0):.4f}")
        g1.caption("Price sensitivity to a ₹1 change in the underlying asset.")
        g2.metric("Gamma", f"{binom_call.get('gamma', 0):.4f}")
        g2.caption("Sensitivity of Delta to changes in the underlying asset.")
        g1.metric("Theta (per day)", f"₹{binom_call.get('theta', 0):.2f}")
        g1.caption("Daily value decay of the option due to time passing.")
        g2.metric("Vega (per 1% vol)", f"₹{binom_call.get('vega', 0):.2f}")
        g2.caption("Price sensitivity to a 1% change in volatility.")
        g1.metric("Rho (per 1% rate)", f"₹{binom_call.get('rho', 0):.2f}")
        g1.caption("Price sensitivity to a 1% change in interest rates.")

with col2:
    with st.container(border=True):
        st.markdown("#### Put Option (Right to Sell)")
        st.metric(f"Theoretical Option Price", f"₹{binom_put.get('price', 0):.2f}")
        st.markdown("---")
        st.markdown("**Key Risk Metrics (The Greeks)**")
        g1, g2 = st.columns(2)
        g1.metric("Delta", f"{binom_put.get('delta', 0):.4f}")
        g1.caption("Price sensitivity to a ₹1 change in the underlying asset.")
        g2.metric("Gamma", f"{binom_put.get('gamma', 0):.4f}")
        g2.caption("Sensitivity of Delta to changes in the underlying asset.")
        g1.metric("Theta (per day)", f"₹{binom_put.get('theta', 0):.2f}")
        g1.caption("Daily value decay of the option due to time passing.")
        g2.metric("Vega (per 1% vol)", f"₹{binom_put.get('vega', 0):.2f}")
        g2.caption("Price sensitivity to a 1% change in volatility.")
        g1.metric("Rho (per 1% rate)", f"₹{binom_put.get('rho', 0):.2f}")
        g1.caption("Price sensitivity to a 1% change in interest rates.")

st.divider()

st.subheader("Visual Analysis Suite")
st.markdown("Interactive charts to help you understand the potential outcomes and key drivers of the option's value.")
col1, col2 = st.columns(2)
with col1:
    with st.container(border=True):
        st.markdown("#### Strategy Payoff at Expiration")
        st.caption("This chart visualizes your potential profit or loss. The point where the line crosses zero is your breakeven price.")
        option_type_payoff = st.radio("Select Option for Payoff", ('Call', 'Put'), horizontal=True)
        premium = binom_call.get('price', 0) if option_type_payoff == 'Call' else binom_put.get('price', 0)
        price_at_exp = np.linspace(S * 0.8, S * 1.2, 100)
        payoff = np.maximum(price_at_exp - K, 0) - premium if option_type_payoff == 'Call' else np.maximum(K - price_at_exp, 0) - premium
        breakeven = K + premium if option_type_payoff == 'Call' else K - premium
        payoff_df = pd.DataFrame({'Profit / Loss (₹)': payoff}, index=price_at_exp)
        st.area_chart(payoff_df)
        p1, p2 = st.columns(2)
        p1.metric("Breakeven Price", f"₹{breakeven:.2f}")
        p2.metric("Max Loss (Premium Paid)", f"₹{-premium:.2f}")

with col2:
    with st.container(border=True):
        st.markdown(f"#### Price History & Volatility")
        st.caption("This chart shows the asset's price swings over the last year. Larger swings result in higher historical volatility.")
        st.metric(f"1-Year Historical Volatility", f"{hv:.2%}")
        st.line_chart(hist_data['Close'], use_container_width=True)
        
st.divider()

st.subheader("Binomial Tree Construction")
st.markdown("This visualizes how the model calculates the option price by building a tree of potential future asset prices and working backward.")
with st.container(border=True):
    c1, c2 = st.columns([1,2])
    N_viz = c1.slider("Steps to Visualize", 2, 8, 4, 1, help="Select the number of time steps for the tree. Fewer steps are easier to visualize.")
    option_type_viz = c2.radio("Option Type to Visualize", ('Call', 'Put'), horizontal=True, key="viz_choice")
    
    if T > 0:
        dt_viz = T / N_viz
        u_viz = np.exp(v * np.sqrt(dt_viz))
        d_viz = 1 / u_viz
        p_viz = (np.exp(r * dt_viz) - d_viz) / (u_viz - d_viz)
    else:
        p_viz = -1 # Invalid p to prevent calculation
    
    if 0 < p_viz < 1 and T > 0:
        asset_tree_viz, option_tree_viz = generate_binomial_tree_data(S, K, T, r, v, N_viz, option_type_viz)
        nodes, edges = create_tree_graph_elements(asset_tree_viz, option_tree_viz)
        
        config = Config(width=1200, 
                        height=800, 
                        directed=True, 
                        physics=False, 
                        hierarchical={'enabled': True, 
                                      'sortMethod': 'directed',
                                      'levelSeparation': 300,
                                      'direction': 'LR'})
        
        agraph(nodes=nodes, edges=edges, config=config)

    elif T <= 0:
        st.warning("Cannot generate a tree for an expired option. Please select a future date.")
    else:
        st.error("Arbitrage Opportunity Detected: The model cannot be built with these parameters. Please adjust Volatility or the Risk-Free Rate.")
        
    st.markdown("---")
    with st.expander("Learn about the Valuation Process"):
        st.markdown("""
        The Binomial Model is a powerful tool that breaks down the time to expiration into a number of discrete time steps. Here’s how it works:
        
        1.  **Asset Price Tree:** The model first builds a tree of all possible future prices for the underlying asset. It starts with today's price and projects forward. At each step, the price can either go up (by a factor of `u`) or down (by a factor of `d`).
        
        2.  **Value at Expiration:** At the final step of the tree (the option's expiration date), the model calculates the option's value at each possible final asset price. This is simply its intrinsic value: `max(0, Asset Price - Strike Price)` for a call, or `max(0, Strike Price - Asset Price)` for a put.
        
        3.  **Backward Induction:** This is the key step. The model works backward from the final step to the present. The option value at any given node is calculated as the discounted average of the two possible future values, weighted by a "risk-neutral probability" (`p`). This process is repeated until it arrives at the first node, which gives the theoretical value of the option today.
        """)

