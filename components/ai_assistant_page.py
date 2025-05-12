import streamlit as st
import google.generativeai as genai
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

def ai_assistant():
    # Download NLTK resources if not already downloaded
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')

    # Simple clean styling
    st.markdown("""
    <style>
        .stApp {
            background-color: #f8f9fa;
        }
        .stChatMessage {
            border-radius: 10px;
            padding: 12px;
            margin: 8px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .stTextInput>div>div>input {
            border-radius: 20px;
            padding: 10px 15px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Configure Gemini
    gemini_api_key = "AIzaSyCZa8e1qnbcxnMIWivihkcny1EZPKEAicY"
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')

    # List of financial terms for topic filtering
    financial_terms = [
        'finance', 'money', 'invest', 'stock', 'bond', 'fund', 'market', 'etf', 'dividend', 'interest', 'loan', 
        'mortgage', 'debt', 'credit', 'budget', 'saving', 'retirement', 'tax', 'income', 'expense', 'asset', 
        'liability', 'portfolio', 'diversification', 'inflation', 'recession', 'bank', 'insurance', 'premium', 
        'roi', 'return', 'equity', 'capital', 'profit', 'loss', 'trading', 'broker', 'nasdaq', 'djia', 's&p', 
        'mutual', 'roth', 'ira', '401k', '401(k)', 'annuity', 'option', 'future', 'commodity', 'forex', 'cryptocurrency', 
        'bitcoin', 'ethereum', 'price', 'hedge', 'leverage', 'margin', 'balance', 'statement', 'account', 
        'amortization', 'compound', 'yield', 'rate', 'depreciation', 'appreciation', 'liquidity', 'solvency', 
        'default', 'bankruptcy', 'risk', 'volatility', 'pension', 'trust', 'beneficiary', 'fiduciary',
        'financial', 'dollar', 'euro', 'currency', 'cost', 'salary', 'wage', 'compensation', 'benefit',
        'economy', 'gdp', 'economic', 'monetary', 'fiscal', 'treasury', 'security', 'bill', 'note', 'bull', 'bear',
        'cash', 'check', 'payment', 'transaction', 'transfer', 'withdrawal', 'deposit', 'atm', 'debit', 'credit card',
        'apr', 'apy', 'index', 'quote', 'exchange', 'trade', 'fee', 'charge', 'commission', 'escrow',
        '401-k', 'dow jones', 'dow', 's&p 500'
    ]

    def is_finance_query(query):
        query = query.lower()
        
        # Standardize common finance terms
        query = re.sub(r'401-k', '401k', query)
        query = re.sub(r'401\s*k', '401k', query)
        query = re.sub(r'401\(k\)', '401k', query)
        
        # Special cases that are always finance-related
        if any(term in query for term in ['401k', '401-k', '401(k)', 'insurance', 'dow', 'djia', 's&p', 'nasdaq', 'index']):
            return True
            
        # Check for financial terms
        if any(term in query for term in financial_terms):
            return True
            
        # Check for money patterns
        money_patterns = [r'\$\d+', r'\d+\s*dollars', r'\d+\s*%', r'percent', r'percentage']
        if any(re.search(pattern, query) for pattern in money_patterns):
            return True
            
        # Educational queries about finance concepts
        if re.search(r'what\s+is|how\s+to|explain|define|meaning\s+of', query):
            return True
            
        return False

    def generate_finance_response(prompt):
        if not is_finance_query(prompt):
            return "I specialize in finance topics only. Please ask about personal finance, investing, retirement, or related topics."
            
        try:
            response = model.generate_content(
                f"""You are a financial expert assistant. Answer the following question strictly about finance:
                
Question: {prompt}

Rules:
1. Only answer if the question is about finance
2. Be accurate and helpful
3. For non-finance questions, say "Please ask about finance topics"
4. For investments, provide balanced advice
5. For retirement questions, be detailed
6. For insurance questions, explain clearly""")
            
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def clear_chat():
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you with finance today?"}]

    # Initialize chat
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you with finance today?"}]

    st.title("💰 Financial AI Assistant")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a finance question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.spinner("Thinking..."):
            response = generate_finance_response(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

    # Clear button
    st.button("Clear Chat", on_click=clear_chat, use_container_width=True)

if __name__ == "__main__":
    ai_assistant()