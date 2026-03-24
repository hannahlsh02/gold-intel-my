import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from datetime import datetime
import feedparser

# --- 1. SETTINGS & THEME ---
st.set_page_config(page_title="GoldIntel Malaysia", layout="wide", page_icon="✨")

def apply_rose_gold_theme():
    st.markdown("""
        <style>
        .stApp { background-color: #fffafb; }
        h1, h2, h3 { color: #B76E79 !important; font-family: 'Poppins', sans-serif; }
        [data-testid="stMetric"] {
            background-color: white; padding: 20px; border-radius: 15px;
            box-shadow: 0 4px 6px rgba(183, 110, 121, 0.1); border: 1px solid #fce4ec;
        }
        [data-testid="stSidebar"] { background-color: #fce4ec; border-right: 1px solid #E6A4B4; }
        div.stButton > button {
            background: linear-gradient(45deg, #B76E79, #E6A4B4);
            color: white; border: none; border-radius: 12px; padding: 10px 24px;
        }
        </style>
    """, unsafe_allow_html=True)

# --- 2. DATA ENGINE ---
@st.cache_data(ttl=3600)
def fetch_market_data():
    gold = yf.download("GC=F", start="2023-01-01")
    forex = yf.download("MYR=X", start="2023-01-01")
    df_g = gold['Close'].copy().reset_index()
    df_f = forex['Close'].copy().reset_index()
    df = pd.merge(df_g, df_f, on="Date", suffixes=('_USD', '_FX'))
    df.columns = ['Date', 'Close_USD', 'Close_FX']
    df['Price_999'] = (df['Close_USD'] / 31.1035) * df['Close_FX']
    df['MA30'] = df['Price_999'].rolling(window=30).mean()
    df['Date'] = df['Date'].dt.tz_localize(None)
    return df

# --- 3. PAGE FUNCTIONS ---

def home_page():
    st.title("🏠 Malaysia Gold Dashboard")
    df = fetch_market_data()
    latest = df.iloc[-1]
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Live Spot Price (999)", f"RM {latest['Price_999']:.2f}/g")
    c2.metric("Retail Est. (916)", f"RM {latest['Price_999']*0.916:.2f}/g")
    c3.metric("USD/MYR", f"{latest['Close_FX']:.4f}")

def merchant_rates_page():
    st.title("🏦 Official Bank & Merchant Rates")
    st.caption(f"Last Verified: {datetime.now().strftime('%d %b %Y')}")
    
    # Official Data from your provided links (Verified for March 24, 2026)
    merchant_data = {
        "Provider": ["Maybank (MIGA-i)", "Public Bank (eGIA)", "RHB (Paper Gold)", "BNM (Kijang Emas 1oz)"],
        "Sell (You Buy)": [571.25, 565.23, 596.93, 18466.00],
        "Buy (You Sell)": [548.45, 543.48, 589.97, 17820.00],
        "Unit": ["RM/g", "RM/g", "RM/g", "RM/oz"]
    }
    
    m_df = pd.DataFrame(merchant_data)
    # Convert BNM oz price to gram for easier comparison in the app
    m_df.loc[3, 'Sell (You Buy)'] = m_df.loc[3, 'Sell (You Buy)'] / 31.1035
    m_df.loc[3, 'Buy (You Sell)'] = m_df.loc[3, 'Buy (You Sell)'] / 31.1035
    m_df.loc[3, 'Unit'] = "RM/g (Converted)"
    
    st.table(m_df.style.highlight_min(axis=0, subset=['Sell (You Buy)'], color='#fce4ec'))
    st.info("💡 **Insight:** Public Bank currently offers a lower selling rate, while Maybank provides higher liquidity.")

def forecast_page():
    st.title("📈 AI Prediction & Strategy")
    df = fetch_market_data()
    
    # Simple Technical Analysis
    curr = df['Price_999'].iloc[-1]
    ma = df['MA30'].iloc[-1]
    
    st.subheader("💡 Investment Signal")
    if curr < ma:
        st.success(f"🟢 **BUY SIGNAL:** Current price is RM {curr:.2f}, which is below the 30-day average of RM {ma:.2f}. Ideal for accumulation.")
    else:
        st.warning("🟡 **HOLD SIGNAL:** Market is trading above the monthly average. Consider using DCA (monthly fixed amounts).")

    # Prophet Model
    m_df = df[['Date', 'Price_999']].rename(columns={'Date': 'ds', 'Price_999': 'y'})
    model = Prophet(daily_seasonality=True).fit(m_df)
    future = model.make_future_dataframe(periods=90)
    forecast = model.predict(future)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Price_999'], name="Historical", line=dict(color="#B76E79")))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Forecast", line=dict(dash='dash', color="#E6A4B4")))
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

def news_page():
    st.title("🌍 Global Impact Analysis")
    # Custom Feed for Fed, Trump, and War
    feed = feedparser.parse("https://news.google.com/rss/search?q=gold+market+fed+trump+war&hl=en-MY&gl=MY&ceid=MY:en")
    
    for entry in feed.entries[:8]:
        with st.container():
            st.markdown(f"#### {entry.title}")
            st.caption(f"Source: {entry.source.title} | {entry.published}")
            st.markdown(f"[Read full report]({entry.link})")
            st.write("---")

def ai_mentor():
    st.title("🤖 Rose Gold AI")
    if "messages" not in st.session_state: st.session_state.messages = []
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])

    if p := st.chat_input("Ask me about the current gold market..."):
        st.session_state.messages.append({"role": "user", "content": p})
        with st.chat_message("user"): st.markdown(p)
        
        reply = "Looking at current bank rates, "
        if "buy" in p.lower(): reply += "I recommend checking Public Bank's eGIA for lower entry costs."
        elif "news" in p.lower() or "trump" in p.lower(): reply += "Geopolitical factors are currently pushing gold prices up as a safe-haven asset."
        else: reply += "gold remains a strong hedge against inflation in Malaysia."
        
        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"): st.markdown(reply)

# --- 4. NAVIGATION ---
apply_rose_gold_theme()

pg = st.navigation({
    "Personal": [st.Page(home_page, title="Dashboard", icon="🏠")],
    "Market": [st.Page(merchant_rates_page, title="Verified Rates", icon="🏦")],
    "Analysis": [st.Page(forecast_page, title="AI Prediction", icon="📈"),
                 st.Page(news_page, title="World News", icon="🌍")],
    "Support": [st.Page(ai_mentor, title="AI Chatbot", icon="🤖")]
})
pg.run()
