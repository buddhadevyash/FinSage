import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
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

    def calculate_indicators(data):
        """Calculate all technical indicators and return them in a dictionary"""
        # Create a copy of the data to avoid SettingWithCopyWarning
        df = data.copy()
        
        # Get close prices as Series for indicators
        close_series = df['Close']
        
        # Calculate Bollinger Bands
        window = 20
        df['MA'] = df['Close'].rolling(window=window).mean()
        df['STD'] = df['Close'].rolling(window=window).std()
        df['bb_h'] = df['MA'] + (df['STD'] * 2)
        df['bb_l'] = df['MA'] - (df['STD'] * 2)
        
        # Calculate other indicators
        macd_ind = MACD(close_series)
        df['macd'] = macd_ind.macd()
        df['macd_signal'] = macd_ind.macd_signal()
        df['macd_diff'] = macd_ind.macd_diff()
        
        df['rsi'] = RSIIndicator(close_series).rsi()
        df['sma'] = SMAIndicator(close_series, window=14).sma_indicator()
        df['ema'] = EMAIndicator(close_series).ema_indicator()
        
        return df

    def tech_indicators():
        st.header('Technical Indicators')
        
        # Calculate all indicators
        indicators_df = calculate_indicators(data)
        
        visualization_type = st.radio(
            'Choose Visualization Type',
            ['Single Indicator', 'Multiple Indicators Comparison']
        )
        
        if visualization_type == 'Single Indicator':
            indicator_option = st.selectbox(
                'Choose a Technical Indicator to Visualize', 
                ['Close', 'BB', 'MACD', 'RSI', 'SMA', 'EMA']
            )
            
            if indicator_option == 'Close':
                st.write('Close Price')
                st.line_chart(data['Close'])
            elif indicator_option == 'BB':
                st.write('BollingerBands')
                bb_view = indicators_df[['Close', 'bb_h', 'bb_l']]
                st.line_chart(bb_view)
            elif indicator_option == 'MACD':
                st.write('Moving Average Convergence Divergence')
                macd_view = indicators_df[['macd', 'macd_signal']]
                st.line_chart(macd_view)
            elif indicator_option == 'RSI':
                st.write('Relative Strength Indicator')
                st.line_chart(indicators_df['rsi'])
            elif indicator_option == 'SMA':
                st.write('Simple Moving Average')
                st.line_chart(indicators_df['sma'])
            else:
                st.write('Exponential Moving Average')
                st.line_chart(indicators_df['ema'])
        
        else:  # Multiple Indicators Comparison
            st.write('Multiple Indicators Comparison')
            
            # Allow selecting multiple indicators for comparison
            selected_indicators = st.multiselect(
                'Select indicators to compare',
                ['Close', 'BB', 'MACD', 'RSI', 'SMA', 'EMA'],
                default=['Close', 'BB']
            )
            
            if not selected_indicators:
                st.warning('Please select at least one indicator to display')
                return
            
            # Create subplots based on selected indicators
            fig, axes = plt.subplots(len(selected_indicators), 1, figsize=(10, 3*len(selected_indicators)), sharex=True)
            if len(selected_indicators) == 1:
                axes = [axes]  # Convert to list for easier indexing
                
            # Set the time index for proper alignment
            date_range = pd.to_datetime(indicators_df.index)
            
            # Plot each selected indicator
            for i, indicator in enumerate(selected_indicators):
                ax = axes[i]
                
                if indicator == 'Close':
                    ax.plot(date_range, indicators_df['Close'], label='Close')
                    ax.set_title('Close Price')
                
                elif indicator == 'BB':
                    ax.plot(date_range, indicators_df['Close'], label='Close')
                    ax.plot(date_range, indicators_df['bb_h'], label='Upper Band')
                    ax.plot(date_range, indicators_df['bb_l'], label='Lower Band')
                    ax.fill_between(date_range, indicators_df['bb_h'], indicators_df['bb_l'], alpha=0.1)
                    ax.set_title('Bollinger Bands')
                
                elif indicator == 'MACD':
                    ax.plot(date_range, indicators_df['macd'], label='MACD')
                    ax.plot(date_range, indicators_df['macd_signal'], label='Signal')
                    ax.bar(date_range, indicators_df['macd_diff'], label='MACD Diff', alpha=0.5)
                    ax.set_title('MACD')
                
                elif indicator == 'RSI':
                    ax.plot(date_range, indicators_df['rsi'], label='RSI')
                    ax.axhline(70, linestyle='--', color='r', alpha=0.5)
                    ax.axhline(30, linestyle='--', color='g', alpha=0.5)
                    ax.set_title('RSI')
                
                elif indicator == 'SMA':
                    ax.plot(date_range, indicators_df['Close'], label='Close')
                    ax.plot(date_range, indicators_df['sma'], label='SMA')
                    ax.set_title('Simple Moving Average')
                
                elif indicator == 'EMA':
                    ax.plot(date_range, indicators_df['Close'], label='Close')
                    ax.plot(date_range, indicators_df['ema'], label='EMA')
                    ax.set_title('Exponential Moving Average')
                
                ax.legend(loc='upper left')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Also offer option to display raw data
            if st.checkbox('Show indicator data'):
                st.write("Indicator Data:")
                
                # Prepare data to display based on selected indicators
                display_cols = ['Close']
                for indicator in selected_indicators:
                    if indicator == 'BB':
                        display_cols.extend(['bb_h', 'bb_l'])
                    elif indicator == 'MACD':
                        display_cols.extend(['macd', 'macd_signal', 'macd_diff'])
                    elif indicator == 'RSI':
                        display_cols.append('rsi')
                    elif indicator == 'SMA':
                        display_cols.append('sma')
                    elif indicator == 'EMA':
                        display_cols.append('ema')
                
                # Display only the selected columns
                display_cols = list(dict.fromkeys(display_cols))  # Remove duplicates
                st.dataframe(indicators_df[display_cols].tail(20))

    def dataframe():
        st.header('Recent Data')
        st.dataframe(data.tail(10))

    def run_model(model_instance, X_train, X_test, y_train, y_test, X_forecast_scaled):
        """Train a model and return predictions and metrics"""
        # Train the model
        model_instance.fit(X_train, y_train)
        
        # Make predictions on test set
        test_preds = model_instance.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, test_preds)
        mae = mean_absolute_error(y_test, test_preds)
        
        # Make forecast predictions
        forecast_preds = model_instance.predict(X_forecast_scaled)
        
        return test_preds, forecast_preds, r2, mae

    def predict():
        st.header('Stock Price Prediction')
        
        # Choose forecasting period
        num_days = st.number_input('How many days forecast?', value=5, min_value=1, max_value=30)
        num_days = int(num_days)
        
        # Select models to compare
        models_to_compare = st.multiselect(
            'Select models to compare',
            ['LinearRegression', 'RandomForestRegressor', 'ExtraTreesRegressor', 
             'KNeighborsRegressor', 'XGBRegressor'],
            default=['LinearRegression', 'RandomForestRegressor', 'XGBRegressor']
        )
        
        if st.button('Generate Predictions'):
            if not models_to_compare:
                st.warning("Please select at least one model to compare.")
                return
                
            # Prepare data for modeling
            df = data[['Close']].copy()
            df['preds'] = df['Close'].shift(-num_days)
            
            # Drop NaN values created by the shift
            df.dropna(inplace=True)
            
            # Separate features and target
            X = df[['Close']].values
            y = df['preds'].values
            
            # Get data for forecasting (last 'num_days' days)
            X_forecast = data[['Close']].tail(num_days).values
            
            # Scale the features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_forecast_scaled = scaler.transform(X_forecast)
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=7)
            
            # Create model instances
            model_instances = {
                'LinearRegression': LinearRegression(),
                'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=42),
                'ExtraTreesRegressor': ExtraTreesRegressor(n_estimators=100, random_state=42),
                'KNeighborsRegressor': KNeighborsRegressor(n_neighbors=5),
                'XGBRegressor': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
            }
            
            # Dictionary to store results
            results = {}
            
            # Train selected models and get predictions
            for model_name in models_to_compare:
                st.subheader(f"{model_name} Model")
                
                # Get model instance
                model = model_instances[model_name]
                
                # Train model and get predictions
                test_preds, forecast_preds, r2, mae = run_model(
                    model, X_train, X_test, y_train, y_test, X_forecast_scaled
                )
                
                # Store results
                results[model_name] = {
                    'forecast': forecast_preds,
                    'r2': r2,
                    'mae': mae
                }
                
                # Display metrics
                st.write(f"**Model Performance:**")
                st.write(f"R² Score: {r2:.4f}")
                st.write(f"Mean Absolute Error: {mae:.4f}")
            
            # Create comparison table of predictions
            st.subheader("Forecast Comparison")
            
            # Get the last date from the data
            last_date = data.index[-1]
            
            # Create a DataFrame to store all predictions
            forecast_df = pd.DataFrame()
            
            # Add date column
            forecast_dates = [last_date + datetime.timedelta(days=i+1) for i in range(num_days)]
            forecast_df['Date'] = forecast_dates
            
            # Add actual close price (last known)
            last_close = data['Close'].iloc[-1]
            forecast_df['Last Close'] = last_close
            
            # Add predictions from each model
            for model_name in models_to_compare:
                forecast_df[model_name] = results[model_name]['forecast']
                
            # Display forecast comparison table
            st.dataframe(forecast_df.set_index('Date'))
            
            # Create comparison metrics table
            st.subheader("Model Performance Comparison")
            
            metrics_df = pd.DataFrame({
                'Model': list(results.keys()),
                'R² Score': [results[model]['r2'] for model in results],
                'Mean Absolute Error': [results[model]['mae'] for model in results]
            })
            
            st.dataframe(metrics_df.set_index('Model'))
            
            # Create visualization of predictions
            st.subheader("Forecast Visualization")
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot the last 30 days of actual data
            last_30_days = data['Close'].tail(30)
            ax.plot(last_30_days.index, last_30_days.values, label='Historical Close', color='black')
            
            # Plot predictions for each model
            for model_name in models_to_compare:
                ax.plot(forecast_dates, results[model_name]['forecast'], marker='o', linestyle='--', 
                        label=f"{model_name} Forecast")
            
            # Add vertical line to separate historical and forecast periods
            ax.axvline(x=last_date, color='gray', linestyle='--', alpha=0.7)
            ax.text(last_date, ax.get_ylim()[0], 'Forecast Start', rotation=90, verticalalignment='bottom')
            
            # Add chart details
            ax.set_title(f"{option} Stock Price Forecast for Next {num_days} Days")
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)

    option = st.sidebar.selectbox('Make a choice', ['Visualize', 'Recent Data', 'Predict'])
    if option == 'Visualize':
        tech_indicators()
    elif option == 'Recent Data':
        dataframe()
    else:
        predict()

if __name__ == '__main__':
    stock()