import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
import datetime
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error

def stock():
    def set_background(image_url):
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url({image_url});
                background-size: cover;
                background-position: top;
                background-repeat:repeat;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    
    set_background("https://img.freepik.com/free-vector/paper-style-white-monochrome-background_52683-66443.jpg")
    
    st.markdown("""
    <style>
            html{
                font-family: Manrope;
                }
            .e1nzilvr2{
                text-align:center;
                text-shadow: 0px 2px 5.3px rgba(0, 0, 0, 0.19);
                font-family: Manrope;
                font-size: 102;
                font-style: normal;
                font-weight: 600;
                line-height: 100%; 
                letter-spacing: -2.16px;
                opacity: 0;
                animation: fadeIn 2s forwards;
                }
             .ea3mdgi5{
                max-width:100%;
                }
    </style>
        """, unsafe_allow_html=True)
    
    st.title('Stock Price Predictions')
    st.sidebar.info('Welcome to the Stock Price Prediction App. Choose your options below')

    @st.cache_resource
    def download_data(op, start_date, end_date):
        # Set auto_adjust=True explicitly as it's now the default
        df = yf.download(op, start=start_date, end=end_date, progress=False, auto_adjust=True)
        
        # Standardize the data format - if it's multi-index (multiple tickers), get the first ticker
        if isinstance(df.columns, pd.MultiIndex):
            ticker = df.columns.get_level_values(1)[0]  # Get the first ticker
            # Extract just the columns for this ticker and make a clean DataFrame
            clean_df = pd.DataFrame()
            for col_type in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if (col_type, ticker) in df.columns:
                    clean_df[col_type] = df[(col_type, ticker)]
            return clean_df
        else:
            # Already a single ticker DataFrame
            return df

    option = st.sidebar.text_input('Enter a Stock Symbol', value='SPY')
    option = option.upper()
    today = datetime.date.today()
    duration = st.sidebar.number_input('Enter the duration', value=3000)
    before = today - datetime.timedelta(days=duration)
    start_date = st.sidebar.date_input('Start Date', value=before)
    end_date = st.sidebar.date_input('End date', today)
    
    if st.sidebar.button('Send'):
        if start_date < end_date:
            st.sidebar.success('Start date: %s\n\nEnd date: %s' % (start_date, end_date))
            data = download_data(option, start_date, end_date)
        else:
            st.sidebar.error('Error: End date must fall after start date')
            return

    data = download_data(option, start_date, end_date)
    scaler = StandardScaler()

    def tech_indicators():
        st.header('Technical Indicators')
        indicator_option = st.radio('Choose a Technical Indicator to Visualize', 
                                  ['Close', 'BB', 'MACD', 'RSI', 'SMA', 'EMA'])

        # Create a copy of the data to avoid SettingWithCopyWarning
        bb = data.copy()
        
        # Calculate Bollinger Bands
        window = 20
        bb['MA'] = bb['Close'].rolling(window=window).mean()
        bb['STD'] = bb['Close'].rolling(window=window).std()
        bb['bb_h'] = bb['MA'] + (bb['STD'] * 2)
        bb['bb_l'] = bb['MA'] - (bb['STD'] * 2)
        bb_view = bb[['Close', 'bb_h', 'bb_l']]
        
        # Get close prices as Series for indicators
        close_series = data['Close']
        
        # Other indicators
        macd = MACD(close_series).macd()
        rsi = RSIIndicator(close_series).rsi()
        sma = SMAIndicator(close_series, window=14).sma_indicator()
        ema = EMAIndicator(close_series).ema_indicator()

        if indicator_option == 'Close':
            st.write('Close Price')
            st.line_chart(data['Close'])
        elif indicator_option == 'BB':
            st.write('BollingerBands')
            st.line_chart(bb_view)
        elif indicator_option == 'MACD':
            st.write('Moving Average Convergence Divergence')
            st.line_chart(macd)
        elif indicator_option == 'RSI':
            st.write('Relative Strength Indicator')
            st.line_chart(rsi)
        elif indicator_option == 'SMA':
            st.write('Simple Moving Average')
            st.line_chart(sma)
        else:
            st.write('Exponential Moving Average')
            st.line_chart(ema)

    def dataframe():
        st.header('Recent Data')
        st.dataframe(data.tail(10))

    def model_engine(model, num):
        # Create DataFrame for modeling using the standardized data
        df = data[['Close']].copy()
        df['preds'] = df['Close'].shift(-num)
        
        # Drop NaN values created by the shift
        df.dropna(inplace=True)
        
        # Separate features and target
        X = df[['Close']].values
        y = df['preds'].values
        
        # Get data for forecasting (last 'num' days)
        X_forecast = data[['Close']].tail(num).values
        
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_forecast_scaled = scaler.transform(X_forecast)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=7)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        preds = model.predict(X_test)
        
        # Print metrics
        st.text(f'r2_score: {r2_score(y_test, preds)}\nMAE: {mean_absolute_error(y_test, preds)}')
        
        # Make forecast predictions
        forecast_pred = model.predict(X_forecast_scaled)
        
        # Display forecast
        day = 1
        for i in forecast_pred:
            st.text(f'Day {day}: {i}')
            day += 1

    def predict():
        model = st.radio('Choose a model', 
                        ['LinearRegression', 'RandomForestRegressor', 
                         'ExtraTreesRegressor', 'KNeighborsRegressor', 
                         'XGBoostRegressor'])
        num = st.number_input('How many days forecast?', value=5)
        num = int(num)
        if st.button('Predict'):
            if model == 'LinearRegression':
                engine = LinearRegression()
            elif model == 'RandomForestRegressor':
                engine = RandomForestRegressor()
            elif model == 'ExtraTreesRegressor':
                engine = ExtraTreesRegressor()
            elif model == 'KNeighborsRegressor':
                engine = KNeighborsRegressor()
            else:
                engine = XGBRegressor()
            model_engine(engine, num)

    option = st.sidebar.selectbox('Make a choice', ['Visualize', 'Recent Data', 'Predict'])
    if option == 'Visualize':
        tech_indicators()
    elif option == 'Recent Data':
        dataframe()
    else:
        predict()

if __name__ == '__main__':
    stock()