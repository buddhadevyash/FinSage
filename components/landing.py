import streamlit as st
import subprocess
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="FINSAGE",
    page_icon="💹",
    layout="wide",
)

# Custom CSS - simplified
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Manrope', sans-serif;
        }
        
        /* Adjust main container padding */
        .main .block-container {
            padding: 2rem;
        }
        
        /* Simple footer styling */
        .footer {
            margin-top: 2rem;
            text-align: center;
            font-size: 0.85rem;
            color: #718096;
            padding-top: 1rem;
            border-top: 1px solid #f1f1f1;
        }
    </style>
""", unsafe_allow_html=True)

def run_app(app_file):
    app_path = str(Path(__file__).parent / app_file)
    command = f"streamlit run {app_path}"
    subprocess.Popen(command, shell=True)

# Main content area
# Header
st.title("FINSAGE")
st.subheader("Financial Intelligence Platform for data-driven investment decisions")

st.divider()

# Create tabs for the main modules
tab1, tab2, tab3 = st.tabs(["📊 Financial Dashboard", "🤖 AI Assistant", "🔮 Prediction Platform"])

with tab1:
    st.header("Financial Dashboard")
    
    st.write("Comprehensive financial analytics with portfolio tracking and market data visualization.")
    
    st.markdown("#### Features")
    features = [
        "✓ Performance metrics",
        "✓ Asset allocation",
        "✓ Interactive charts",
        "✓ Real-time updates"
    ]
    for feature in features:
        st.markdown(feature)
    
    if st.button("Launch Dashboard", type="primary", use_container_width=True):
        run_app("finrisk_page/finrisk/FinRisk/main_page.py")

with tab2:
    st.header("AI Financial Assistant")
    
    st.write("AI-powered financial advisor providing personalized insights and recommendations.")
    
    st.markdown("#### Features")
    features = [
        "✓ Natural language queries",
        "✓ Investment recommendations",
        "✓ Market sentiment analysis",
        "✓ Strategy optimization"
    ]
    for feature in features:
        st.markdown(feature)
    
    if st.button("Launch AI Assistant", type="primary", use_container_width=True):
        run_app("ai_assistant_page.py")

with tab3:
    st.header("Prediction Platform")
    
    st.write("Advanced forecasting tools using machine learning to predict market trends.")
    
    st.markdown("#### Features")
    features = [
        "✓ Price predictions",
        "✓ Risk assessment",
        "✓ Scenario analysis", 
        "✓ Market forecasting"
    ]
    for feature in features:
        st.markdown(feature)
    
    if st.button("Launch Prediction Tools", type="primary", use_container_width=True):
        run_app("prediction_page.py")

# Footer
st.markdown("""
    <div class="footer">
        © 2025 FINSAGE - Financial Intelligence Platform
    </div>
""", unsafe_allow_html=True)