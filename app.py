import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# ==========================================
# 1. PAGE CONFIGURATION & SETUP
# ==========================================
st.set_page_config(
    page_title="Cult Quant 2026 | Investment Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Removed strict CSS backgrounds so it works flawlessly in BOTH Light and Dark themes natively!
st.markdown("""
    <style>
    /* Minimal styling that adapts to any theme */
    h1, h2, h3 { font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. DATA LOADER (CACHED FOR PERFORMANCE)
# ==========================================
@st.cache_data
def load_data():
    data = {}
    try:
        data['invest'] = pd.read_csv('investment_recommendations.csv')
        data['allocations'] = pd.read_csv('portfolio_allocations.csv')
        data['risk_sum'] = pd.read_csv('portfolio_risk_summary.csv')
        data['sector'] = pd.read_csv('sector_analytics.csv')
        data['frontier'] = pd.read_csv('efficient_frontier.csv')
        data['forecast'] = pd.read_csv('final_opportunities.csv')
        data['anomalies'] = pd.read_csv('anomalies.csv')
        data['stress'] = pd.read_csv('stress_test_results.csv')
        data['exec'] = pd.read_csv('executive_summary.csv')
    except Exception as e:
        st.sidebar.error(f"Data loading error: {e}")
    return data

data = load_data()

if not data:
    st.error("Please ensure all CSV files are present in the directory.")
    st.stop()

df_invest = data['invest']
df_forecast = data['forecast']
df_allocations = data['allocations']

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("Cult Quant 2026")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate Dashboard", [
    "1. Overview",
    "2. Stock Explorer",
    "3. Portfolio Builder",
    "4. Risk Dashboard",
    "5. Forecast Dashboard",
    "6. Anomaly Dashboard",
    "7. Investment Recommendations",
    "8. Executive Summary"
])

st.sidebar.markdown("---")

# ==========================================
# 4. PAGES RENDERING
# ==========================================

if page == "1. Overview":
    st.title("🌐 Market Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Stocks Analyzed", len(df_forecast))
    col2.metric("Strong Buy Signals", len(df_forecast[df_forecast['Final_Recommendation'] == 'Strong Buy']) if 'Final_Recommendation' in df_forecast else 0)
    col3.metric("Anomalies Detected", len(data['anomalies']))
    col4.metric("Avg Market Expected Return", f"{df_forecast['Expected_Return'].mean():.2f}%" if not df_forecast.empty else "N/A")
    
    st.markdown("### Top Opportunities (Future Score)")
    if not data['exec'].empty:
        top_opps = data['exec'].head(10)
        fig = px.bar(top_opps, x='Symbol', y='Future_Score', color='Expected_Return',
                     color_continuous_scale='Viridis', text='Final_Recommendation')
        st.plotly_chart(fig, use_container_width=True)

elif page == "2. Stock Explorer":
    st.title("🔎 Stock Explorer")
    
    if not df_forecast.empty:
        selected_stock = st.selectbox("Select a Stock Symbol", df_forecast['Symbol'].sort_values())
        stock_data = df_forecast[df_forecast['Symbol'] == selected_stock].iloc[0]
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("### Recommendation")
            st.markdown(f"<h2 style='text-align:center;'>{stock_data.get('Final_Recommendation', 'N/A')}</h2>", unsafe_allow_html=True)
            
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=stock_data.get('Future_Score', 50),
                title={'text': "AI Future Score"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "rgba(30,144,255,0.8)"}}
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with col2:
            st.markdown("### AI Explainability")
            st.info(stock_data.get('AI_Explainability', 'Explanation not available.'))
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Expected Return", f"{stock_data.get('Expected_Return', 0):.2f}%")
            m2.metric("Forecast Volatility", f"{stock_data.get('Forecast_Volatility', 0):.2f}%")
            m3.metric("Sharpe Ratio", f"{stock_data.get('Sharpe', 0):.2f}")

elif page == "3. Portfolio Builder":
    st.title("💼 Portfolio Builder")
    
    st.markdown("### Portfolio Risk-Return Profiles")
    st.dataframe(data['risk_sum'], use_container_width=True)
    
    st.markdown("### Target Allocations")
    tabs = st.tabs(["Conservative (Min Vol)", "Balanced (Max Sharpe)", "Aggressive (Max Return)"])
    
    def plot_donut(col_name, title):
        if col_name not in df_allocations.columns:
            return None
        df_pie = df_allocations[['Symbol', col_name]].dropna()
        # Convert to numeric safely
        df_pie[col_name] = pd.to_numeric(df_pie[col_name], errors='coerce').fillna(0)
        df_pie = df_pie[df_pie[col_name] > 0.01] 
        
        if df_pie.empty: # FIX: Prevents crash if dataframe is empty
            return None
            
        fig = px.pie(df_pie, values=col_name, names='Symbol', hole=0.4, title=title)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        return fig

    with tabs[0]:
        fig = plot_donut('Conservative_Wt', "Conservative Portfolio")
        if fig: st.plotly_chart(fig, use_container_width=True)
        else: st.info("Allocations too small to plot.")
            
    with tabs[1]:
        fig = plot_donut('Balanced_Wt', "Balanced Portfolio")
        if fig: st.plotly_chart(fig, use_container_width=True)
        else: st.info("Allocations too small to plot.")
            
    with tabs[2]:
        fig = plot_donut('Aggressive_Wt', "Aggressive Portfolio")
        if fig: st.plotly_chart(fig, use_container_width=True)
        else: st.info("Allocations too small to plot.")

elif page == "4. Risk Dashboard":
    st.title("⚠️ Risk Dashboard")
    
    st.markdown("### Efficient Frontier")
    if not data['frontier'].empty:
        fig_ef = px.scatter(data['frontier'], x='Volatility', y='Return', color='Sharpe', color_continuous_scale='YlOrRd')
        st.plotly_chart(fig_ef, use_container_width=True)
    
    st.markdown("### Stress Testing (Market Crashes)")
    st.dataframe(data['stress'], use_container_width=True)
    
    st.markdown("### Max Drawdown by Sector (Top 15 Stocks)")
    if 'Max_Drawdown' in df_invest.columns:
        # FIX: Safe numeric conversion before plotting
        df_invest['Max_Drawdown'] = pd.to_numeric(df_invest['Max_Drawdown'], errors='coerce')
        df_dd = df_invest.dropna(subset=['Max_Drawdown']).nsmallest(15, 'Max_Drawdown')
        if not df_dd.empty:
            fig_dd = px.bar(df_dd, x='Symbol', y='Max_Drawdown', color='Recommendation')
            st.plotly_chart(fig_dd, use_container_width=True)

elif page == "5. Forecast Dashboard":
    st.title("🔮 Forecast Dashboard")
    
    if not df_forecast.empty:
        fig_scatter = px.scatter(df_forecast, x='Forecast_Volatility', y='Expected_Return', 
                                 color='Final_Recommendation', hover_data=['Symbol'])
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.markdown("### Sector Outlook")
    st.dataframe(data['sector'], use_container_width=True)

elif page == "6. Anomaly Dashboard":
    st.title("🚨 Anomaly Dashboard")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Cross-Sectional Anomalies")
        anomalies_if = df_forecast[df_forecast.get('Anomaly_Flag', '') == 'Anomaly']
        if anomalies_if.empty: st.success("No anomalies currently.")
        else: st.dataframe(anomalies_if[['Symbol', 'Expected_Return']], use_container_width=True)
            
    with col2:
        st.markdown("### Historical Extreme Events")
        df_anom = data['anomalies'].copy()
        st.dataframe(df_anom.tail(15), use_container_width=True)
        
    st.markdown("### Anomaly Frequency Timeline")
    # FIX: Safe Date Parsing for timeline
    if 'Date' in df_anom.columns and not df_anom.empty:
        df_anom['Date'] = pd.to_datetime(df_anom['Date'], errors='coerce')
        df_anom = df_anom.dropna(subset=['Date'])
        if not df_anom.empty:
            timeline = df_anom.groupby(df_anom['Date'].dt.to_period('M')).size().reset_index(name='Count')
            timeline['Date'] = timeline['Date'].astype(str)
            fig_line = px.line(timeline, x='Date', y='Count', markers=True)
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("Not enough valid dates to plot timeline.")

elif page == "7. Investment Recommendations":
    st.title("📝 Investment Recommendations")
    
    filter_rec = st.selectbox("Filter", options=['All', 'Strong Buy', 'Buy', 'Hold', 'Reduce', 'Sell'])
    df_display = df_forecast.copy()
    if filter_rec != 'All':
        df_display = df_display[df_display.get('Final_Recommendation') == filter_rec]
        
    st.dataframe(df_display, height=500, use_container_width=True)

elif page == "8. Executive Summary":
    st.title("🏆 Executive Summary")
    
    if not data['exec'].empty:
        best_stock = data['exec'].iloc[0]
        st.success(f"**Top Recommended Stock:** {best_stock['Symbol']} (Score: {best_stock.get('Future_Score', 0):.2f})")
    
    st.dataframe(data['exec'], use_container_width=True)
