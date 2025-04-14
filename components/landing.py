import streamlit as st
import sys
import os
import importlib.util

# Set page configuration
st.set_page_config(
    page_title="FinSage Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to import a module from filepath
def import_module_from_file(module_name, file_path):
    """Import a module from a file path"""
    try:
        # Check if the file exists
        if not os.path.exists(file_path):
            raise ImportError(f"File not found: {file_path}")
            
        # Create a module spec
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None:
            raise ImportError(f"Could not create spec for module: {file_path}")
            
        # Create the module
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        
        # Execute the module
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        st.error(f"Error importing {module_name}: {e}")
        return None

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #4CAF50;
        margin-bottom: 1rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 500;
        color: #555;
        margin-bottom: 2rem;
        text-align: center;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .card-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 0.75rem;
    }
    .card-text {
        color: #666;
        font-size: 0.95rem;
    }
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Initialize session state for page navigation if it doesn't exist
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Page navigation logic
    if st.session_state.page == 'home':
        show_home()
    elif st.session_state.page == 'main_page':
        # Run main page
        try:
            # Use dynamic module loading
            main_page_path = os.path.join(current_dir, 'main_page.py')
            main_page_module = import_module_from_file('main_page', main_page_path)
            
            if main_page_module and hasattr(main_page_module, 'main'):
                main_page_module.main()
            else:
                st.error("Could not find main function in main_page.py")
                if st.button("Return to Home"):
                    st.session_state.page = 'home'
                    st.rerun()
        except Exception as e:
            st.error(f"Error loading main page: {e}")
            if st.button("Return to Home"):
                st.session_state.page = 'home'
                st.rerun()
    elif st.session_state.page == 'ai_assistant':
        # Run AI assistant page
        try:
            # Use dynamic module loading
            ai_assistant_path = os.path.join(current_dir, 'ai_assistant_page.py')
            ai_assistant_module = import_module_from_file('ai_assistant_page', ai_assistant_path)
            
            if ai_assistant_module and hasattr(ai_assistant_module, 'main'):
                ai_assistant_module.main()
            else:
                st.error("Could not find main function in ai_assistant_page.py")
                if st.button("Return to Home"):
                    st.session_state.page = 'home'
                    st.rerun()
        except Exception as e:
            st.error(f"Error loading AI assistant page: {e}")
            if st.button("Return to Home"):
                st.session_state.page = 'home'
                st.rerun()
    elif st.session_state.page == 'prediction':
        # Run prediction page
        try:
            # Use dynamic module loading
            prediction_path = os.path.join(current_dir, 'prediction_page.py')
            prediction_module = import_module_from_file('prediction_page', prediction_path)
            
            if prediction_module and hasattr(prediction_module, 'main'):
                prediction_module.main()
            else:
                st.error("Could not find main function in prediction_page.py")
                if st.button("Return to Home"):
                    st.session_state.page = 'home'
                    st.rerun()
        except Exception as e:
            st.error(f"Error loading prediction page: {e}")
            if st.button("Return to Home"):
                st.session_state.page = 'home'
                st.rerun()

def show_home():
    # Header
    st.markdown("<h1 class='main-header'>Welcome to FinSage</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Your comprehensive financial analytics platform</p>", unsafe_allow_html=True)
    
    # Navigation Cards
    st.write("### Choose a Module to Explore")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='feature-icon'>📊</div>", unsafe_allow_html=True)
        st.markdown("<h3 class='card-title'>Main Dashboard</h3>", unsafe_allow_html=True)
        st.markdown("<p class='card-text'>Access the main financial dashboard with comprehensive analytics and portfolio overview.</p>", unsafe_allow_html=True)
        if st.button("Launch Dashboard", key="main"):
            st.session_state.page = 'main_page'
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='feature-icon'>🤖</div>", unsafe_allow_html=True)
        st.markdown("<h3 class='card-title'>AI Assistant</h3>", unsafe_allow_html=True)
        st.markdown("<p class='card-text'>Get intelligent insights and answers to your financial questions with our AI assistant.</p>", unsafe_allow_html=True)
        if st.button("Open AI Assistant", key="ai"):
            st.session_state.page = 'ai_assistant'
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='feature-icon'>📈</div>", unsafe_allow_html=True)
        st.markdown("<h3 class='card-title'>Predictions</h3>", unsafe_allow_html=True)
        st.markdown("<p class='card-text'>Explore predictive models and forecasts for market trends and risk analysis.</p>", unsafe_allow_html=True)
        if st.button("View Predictions", key="pred"):
            st.session_state.page = 'prediction'
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Features section
    st.write("---")
    st.write("### Key Features")
    
    feature_col1, feature_col2 = st.columns(2)
    
    with feature_col1:
        st.write("#### 📊 Real-time Financial Analytics")
        st.write("Monitor your portfolio performance with live updates and comprehensive metrics.")
        
        st.write("#### 🤖 AI-Powered Insights")
        st.write("Leverage advanced AI to get personalized financial advice and market analysis.")
    
    with feature_col2:
        st.write("#### 🔮 Advanced Prediction Models")
        st.write("Access sophisticated prediction models to forecast market trends and potential risks.")
        
        st.write("#### 🛡️ Risk Assessment Tools")
        st.write("Identify and mitigate financial risks using our specialized risk assessment modules.")
    
    # Footer
    st.write("---")
    st.write("#### About FinSage")
    st.write("FinSage provides cutting-edge financial analytics tools for individuals and organizations. Our platform combines powerful analytics with easy-to-use interfaces to help you make informed decisions.")

# Run the main function
if __name__ == "__main__":
    main()