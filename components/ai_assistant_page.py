import streamlit as st
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv
import pandas as pd
import yfinance as yf
from gtts import gTTS
import speech_recognition as sr
import io
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
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
        nltk.download('averaged_perceptron_tagger')  # Fixed resource name
        
    def set_background(image_url):
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url({image_url});
                background-size: fit;
                background-position: center;
                background-repeat: repeat;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    
    set_background("https://img.freepik.com/free-vector/geometric-pattern-background-vector-white_53876-126684.jpg")

    st.markdown("""
    <style>
            html{
                font-family: Manrope;
                }
            .e1nzilvr2{
                text-align:center;
                text-shadow: 0px 2px 5.3px rgba(0, 0, 0, 0.19);
                font-family: Manrope;
                font-size: 72px;
                font-style: normal;
                font-weight: 600;
                line-height: 83px; 
                letter-spacing: -2.16px;
                opacity: 0;
                animation: fadeIn 2s forwards;
                }
             .ea3mdgi5{
                max-width:100%;
                }
    </style>
        """, unsafe_allow_html=True)
                
    # Load environment variables
    load_dotenv()

    # List of financial terms for better topic filtering
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
        'apr', 'apy', 'index', 'quote', 'exchange', 'trade', 'fee', 'charge', 'commission', 'escrow'
    ]

    # Improved function to check if a query is finance-related - simplified to avoid NLTK POS tagging
    def is_finance_query(query):
        query = query.lower()
        
        # Convert inputs like "401-k" to standard format "401k" for better matching
        query = re.sub(r'401-k', '401k', query)
        query = re.sub(r'401\s*k', '401k', query)
        query = re.sub(r'401\(k\)', '401k', query)
        
        # Special case handling for common finance terms with variations
        special_finance_terms = {
            '401k': ['401-k', '401(k)', '401 k', '401k plan', '401k account', '401k retirement'],
            'ira': ['individual retirement account', 'roth ira', 'traditional ira', 'sep ira'],
            'etf': ['exchange traded fund', 'exchange-traded fund'],
            'cd': ['certificate of deposit', 'certificates of deposit'],
            'apr': ['annual percentage rate'],
            'apy': ['annual percentage yield'],
            'hsa': ['health savings account'],
            'fsa': ['flexible spending account'],
            'rmd': ['required minimum distribution'],
            'mortgage': ['home loan', 'housing loan', 'property loan'],
            'credit score': ['fico', 'credit rating', 'credit report', 'credit history'],
            'stock': ['shares', 'equity', 'ticker'],
            'bond': ['treasury', 't-bill', 'municipal bond', 'corporate bond'],
            'portfolio': ['investment portfolio', 'asset allocation'],
            'retirement': ['pension', 'social security', 'retire'],
            'banking': ['bank account', 'checking', 'savings', 'money market'],
            'insurance': ['life insurance', 'auto insurance', 'home insurance', 'coverage', 'policy']
        }
        
        # Check special case finance terms
        for term, variations in special_finance_terms.items():
            if term in query or any(variation in query for variation in variations):
                return True
        
        # If query explicitly mentions 401k or similar retirement accounts, it's definitely finance
        retirement_terms = ['401k', '401-k', '401(k)', 'ira', 'roth', 'retirement']
        if any(term in query for term in retirement_terms):
            return True
            
        # Check for explicit non-finance topics
        non_finance_keywords = [
            "sports", "cricket", "football", "soccer", "baseball", "basketball", "tennis", "golf", "nfl", "nba", "mlb",
            "actor", "actress", "movie", "film", "cinema", "hollywood", "bollywood", "song", "music", "singer", "album",
            "politics", "politician", "president", "prime minister", "election", "party", "government", "minister",
            "history", "historical", "war", "ancient", "medieval", "century", "king", "queen", "emperor",
            "geography", "country", "continent", "ocean", "mountain", "river", "lake", "capital", "city",
            "science", "physics", "chemistry", "biology", "scientist", "space", "planet", "star", "galaxy",
            "celebrity", "famous", "entertainment", "gossip", "scandal", "award", "ceremony",
            "weather", "climate", "storm", "hurricane", "tornado", "earthquake", "tsunami", "forecast",
            "travel", "tourism", "destination", "vacation", "hotel", "flight", "cruise", "beach", "resort",
            "food", "recipe", "cook", "chef", "restaurant", "cuisine", "ingredient", "meal", "dish", "dessert",
            "religion", "god", "prayer", "worship", "temple", "church", "mosque", "faith", "belief",
            "sports player", "athlete", "player", "team", "match", "tournament", "championship", "world cup"
        ]
        
        # If any financial term is in the query, consider it finance-related
        if any(term in query for term in financial_terms):
            return True
            
        # Check for patterns that might indicate a person question
        if "who is" in query or "who was" in query:
            # Only allow if it also contains finance terms
            return any(term in query for term in financial_terms)
        
        # Check for explicit non-finance keywords
        for keyword in non_finance_keywords:
            if keyword in query:
                # Only reject if there are no finance terms
                if not any(term in query for term in financial_terms):
                    return False
        
        # Additional check: does query mention money, investments, or financial concepts?
        money_patterns = [r'\$\d+', r'\d+\s*dollars', r'\d+\s*%', r'percent', r'percentage']
        if any(re.search(pattern, query) for pattern in money_patterns):
            return True
            
        # If we can't determine, default to allowing the query if it contains words like "what is" or "how to"
        # that indicate the user is asking for educational information
        if re.search(r'what\s+is|how\s+to|explain|define|meaning\s+of', query):
            # Give benefit of doubt for educational queries unless clearly non-finance
            return True
            
        # Default to True for short queries that weren't caught by non-finance filters
        words = query.split()
        if len(words) <= 5:
            return True
            
        # For longer queries, require at least one finance term
        return any(term in query for term in financial_terms)
    
    # Financial AI Assistant functions
    def generate_ai_response(prompt, client, context=""):
        # First check if query is finance-related
        if not is_finance_query(prompt):
            return "This question is outside the scope of finance-related topics."
            
        system_message = f"""You are an AI Assistant specialized in personal finance, insurance, credit scoring, stocks, retirement plans (including 401k, IRA, etc.), and all related financial topics.
You MUST answer all finance-related questions thoroughly and accurately.
For any questions about sports, celebrities, politics, general knowledge, or other non-finance topics, respond ONLY with:
"This question is outside the scope of finance-related topics."
Important: Questions about 401k, retirement plans, insurance, financial planning, and personal finance are definitely within your scope.
Do not provide any information about people unless they are directly relevant to finance.
Use the following additional information in your responses: Context: {context}"""
        
        if "stock" in prompt.lower() or "$" in prompt:
            words = prompt.replace("$", "").split()
            potential_tickers = [word.upper() for word in words if word.isalpha() and len(word) <= 5]
            
            stock_info = ""
            for ticker in potential_tickers:
                stock_info += get_stock_info(ticker) + "\n"
            
            if stock_info:
                prompt += f"\n\nHere's the current stock information:\n{stock_info}"
        
        messages = [    
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        
        response = ""
        for message in client.chat_completion(
            messages=messages,
            max_tokens=120,
            stream=True
        ):
            response += message.choices[0].delta.content or ""
        
        # Double-check response doesn't contain inappropriate content
        if not is_finance_query(prompt) and len(response) > 60:
            return "This question is outside the scope of finance-related topics."
            
        return response

    def get_stock_info(ticker):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            current_price = stock.history(period='1d')['Close'].iloc[-1]
            company_name = info.get('longName', 'Unknown Company')
            return f"{company_name} (${ticker}) current price: ${current_price:.2f}"
        except Exception as e:
            return f"Unable to fetch information for {ticker}. Error: {str(e)}"

    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you with finance-related questions today?"}]

    def download_stock_data(ticker):
        stock = yf.Ticker(ticker)
        return stock.history(period="10y")

    def text_to_speech(text):
        tts = gTTS(text)
        return tts

    def speech_to_text():
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.write("Listening...")
            audio = recognizer.listen(source)
            st.write("Processing...")
        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            st.write("Could not understand audio")
        except sr.RequestError as e:
            st.write(f"Error with the Google Speech Recognition service; {e}")
        return ""

    # PDF Chat functions
    def get_pdf_text(pdf_docs):
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    def get_text_chunks(text):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000, chunk_overlap=1000
        )
        chunks = text_splitter.split_text(text)
        return chunks

    def get_vector_store(text_chunks, api_key):
        embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        vector_store = FAISS.from_texts(text_chunks, embedding=embedding)
        vector_store.save_local("faiss_index")

    def get_conversational_chain(api_key):
        prompt_template = """
        Answer the question strictly based on finance-related context only.

If the question is not related to finance, respond with: "This question is outside the scope of finance-related topics."

If the answer is not available in the provided context, respond with: "Answer is not available in the context."

When the answer is available in the context, provide it in both bullet points and a detailed paragraph.

Questions about 401k, retirement plans, insurance, financial planning, and personal finance are definitely within your scope.

Do not provide any assumptions or unrelated information. Stick strictly to the finance domain.

        Context:
        {context}?
        Question: 
        {question}
        Answer:
        """
        genAI.configure(api_key=api_key)
        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, google_api_key=api_key)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain

    def user_input_pdf(user_question, api_key):
        # Check if the question is finance-related
        if not is_finance_query(user_question):
            return {"output_text": "This question is outside the scope of finance-related topics."}
            
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = new_db.similarity_search(user_question)
            chain = get_conversational_chain(api_key)
            response = chain(
                {"input_documents": docs, "question": user_question},
                return_only_outputs=True
            )
            return response
        except Exception as e:
            return {"output_text": f"Error processing your question: {str(e)}"}

    st.title('🤖 Financial AI Assistant & PDF Chat')

    # Sidebar for settings and file upload
    with st.sidebar:
        st.title('Assistant Settings')
        # Use direct API keys instead of secrets
        hf_api_token = "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"
        gemini_api_key = "AIzaSyCZa8e1qnbcxnMIWivihkcny1EZPKEAicY"
        
        st.button('Clear Chat History', on_click=clear_chat_history)
        
        # Upload a text file
        st.subheader("Upload a Text File (200 MB limit)")
        txt_file = st.file_uploader("Choose a text file", type=["txt"], accept_multiple_files=False)
        text_data = ""
        if txt_file:
            if txt_file.size <= 2e8:  # 200 MB limit
                text_data = txt_file.read().decode("utf-8")
                st.write("Text file uploaded successfully!")
            else:
                st.write("File size exceeds the 200 MB limit.")
        
        # Upload a CSV file
        st.subheader("Upload a CSV File (200 MB limit)")
        csv_file = st.file_uploader("Choose a CSV file", type=["csv"], accept_multiple_files=False)
        df = None
        if csv_file:
            if csv_file.size <= 2e8:  # 200 MB limit
                df = pd.read_csv(csv_file)
                st.write("CSV file uploaded successfully!")
                st.dataframe(df.head())
            else:
                st.write("File size exceeds the 200 MB limit.")

        st.subheader("Stock Data")
        ticker = st.text_input("Enter a stock ticker symbol:")
        stock_data = None
        if ticker:
            stock_data = download_stock_data(ticker)
            st.line_chart(stock_data['Close'])

        # PDF Upload
        st.subheader("Upload PDF Files")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        if st.button("Process PDFs"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks, gemini_api_key)
                    st.session_state.pdf_processed = True
                    st.success("PDFs processed successfully!")
            else:
                st.error("Please upload PDF files before processing.")

    # Initialize the InferenceClient for Hugging Face
    client = InferenceClient(token=hf_api_token)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you with finance-related questions today?"}]
    
    # Prepare context from uploaded files
    context = ""
    if text_data:
        context += f"Text data: {text_data[:1000]}... "  # Limiting context size
    if df is not None:
        # Summarize the CSV data
        csv_summary = df.head(5).to_string(index=False)
        context += f"CSV data sample:\n{csv_summary}\n"
    
    # Display chat history with TTS option
    for i, message in enumerate(st.session_state.messages):
        col1, col2 = st.columns([0.9, 0.1])
        with col1:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        if message["role"] == "assistant":
            with col2:
                if st.button("🔊", key=f"tts_{i}"):
                    tts = text_to_speech(message["content"])
                    # Create temporary file for audio
                    audio_file = f"response_{i}.mp3"
                    tts.save(audio_file)
                    # Play audio
                    with open(audio_file, "rb") as f:
                        audio_bytes = f.read()
                    st.audio(audio_bytes)

    # User input with STT option
    col1, col2, col3 = st.columns([0.8, 0.1, 0.1])
    with col1:
        user_input = st.chat_input("Type your finance-related question here:")
    with col2:
        if st.button("🎤"):
            recognized_text = speech_to_text()
            if recognized_text:
                user_input = recognized_text
    with col3:
        use_pdf = st.checkbox("Use PDF")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        try:
            if use_pdf and st.session_state.get('pdf_processed', False):
                # Generate AI response from PDF
                response = user_input_pdf(user_input, gemini_api_key)
                response_content = response["output_text"]
            else:
                # Generate AI response with context
                response_content = generate_ai_response(user_input, client, context=context)
            
            st.session_state.messages.append({"role": "assistant", "content": response_content})
            with st.chat_message("assistant"):
                st.markdown(response_content)
        except Exception as e:
            error_message = f"Error processing your request: {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            with st.chat_message("assistant"):
                st.markdown(error_message)

if __name__ == "__main__":
    ai_assistant()