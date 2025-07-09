import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import timedelta
import plotly.express as px
import plotly.graph_objects as go

# --- Page Config & Custom Styling ---
st.set_page_config(
    layout="wide",
    page_title="Dynamic Stock Analysis & Forecasting",
    initial_sidebar_state="expanded",
    # You can choose "dark" or "light" theme here
    # theme={
    #     "base": "dark",
    #     "primaryColor": "#6d5ee8",
    #     "backgroundColor": "#1a1a1a",
    #     "secondaryBackgroundColor": "#2d2d2d",
    #     "textColor": "#fafafa",
    #     "font": "sans serif"
    # }
)

# Custom CSS for minor aesthetic tweaks (optional, but can enhance look)
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
    }
    .st-emotion-cache-1jm6g9g {
        background-color: #0E1117; # Match sidebar background
    }
</style>
""", unsafe_allow_html=True)


st.title("📈 Dynamic Stock Data Analysis Dashboard with LSTM Forecasting")
st.markdown("Analyze historical stock performance and predict future trends using a Long Short-Term Memory (LSTM) neural network.")

st.divider() # Visual separator

# Sidebar for inputs
st.sidebar.header("⚙️ Configuration")
stock = st.sidebar.text_input("Enter Stock Symbol (e.g. AAPL, TSLA, RELIANCE.NS)", value="AAPL").upper()
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# LSTM Hyperparameters
st.sidebar.subheader("Neural Network Settings (LSTM)")
seq_len = st.sidebar.slider("Sequence Length (Days to Look Back)", min_value=30, max_value=120, value=60, help="Number of past days to consider for each prediction.")
lstm_units = st.sidebar.slider("LSTM Layer Units", min_value=30, max_value=100, value=50, help="Number of neurons in each LSTM layer.")
epochs = st.sidebar.slider("Training Epochs", min_value=5, max_value=50, value=10, help="Number of times the model will go through the entire training dataset.")
batch_size = st.sidebar.slider("Batch Size", min_value=16, max_value=64, value=32, help="Number of samples processed before updating model weights.")
forecast_days = st.sidebar.slider("Future Forecast Days", min_value=1, max_value=30, value=7, help="Number of future days to predict stock prices for.")

st.sidebar.markdown("---")
st.sidebar.info("Adjust parameters above and click outside the slider/input for changes to take effect.")


# --- Data Fetching and Initial Checks ---
data = pd.DataFrame() # Initialize data as an empty DataFrame
close_data = pd.Series(dtype='float64')
volume_data = pd.Series(dtype='float64')


if stock and start_date < end_date:
    with st.spinner(f"📥 Fetching historical data for {stock}..."):
        try:
            data = yf.download(stock, start=start_date, end=end_date)

            if data.empty:
                st.warning("⚠️ No data found for this symbol or date range. Please double-check the stock symbol or select a different date range.")
            else:
                if "Close" in data.columns:
                    close_data = data["Close"].dropna()
                else:
                    st.error("❌ 'Close' price column not found in the downloaded data. This is essential for analysis. Please try a different stock symbol.")
                    st.stop()

                if "Volume" in data.columns:
                    volume_data = data["Volume"].dropna()
                else:
                    st.warning("⚠️ 'Volume' column not found in data. The Volume chart will not be displayed.")

                if close_data.empty:
                    st.warning("⚠️ 'Close' price data is empty after removing missing values for the selected period. Cannot proceed with analysis. Try a wider date range.")
                    st.stop()
                else:
                    st.success(f"✅ Data for {stock} loaded successfully from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}.")

                    st.divider()

                    # --- Raw Data Preview ---
                    with st.expander("🔍 Click to view Raw Data Preview"):
                        st.dataframe(data.head(10))
                        st.write(f"Total entries: {len(data)}")

                    # --- Summary Statistics ---
                    with st.expander("📊 Click to view Summary Statistics"):
                        st.dataframe(data.describe())

                    st.divider()

                    # --- Visualizations Section ---
                    st.subheader(f"📈 Visualizing {stock} Performance")
                    col1, col2 = st.columns(2)

                    with col1:
                        # --- Closing Price Over Time (Plotly Express) ---
                        st.markdown("#### Closing Price Over Time")
                        try:
                            plot_df = pd.DataFrame({
                                "Date": close_data.index,
                                "Closing Price": close_data.values.flatten()
                            })
                            fig1 = px.line(plot_df, x="Date", y="Closing Price",
                                           title=f"{stock} - Historical Closing Price",
                                           labels={"Closing Price": "Price (USD)"},
                                           hover_data={"Date": "|%Y-%m-%d", "Closing Price": ":.2f"})
                            fig1.update_traces(mode='lines+markers', marker=dict(size=4, opacity=0))
                            fig1.update_layout(hovermode="x unified", height=400)
                            st.plotly_chart(fig1, use_container_width=True)
                        except Exception as plot_error:
                            st.error(f"❌ Error plotting Closing Price Over Time: {plot_error}")

                    with col2:
                        # --- Trading Volume Plot (Plotly Express) ---
                        st.markdown("#### Trading Volume")
                        if not volume_data.empty:
                            try:
                                volume_plot_df = pd.DataFrame({
                                    "Date": volume_data.index,
                                    "Volume": volume_data.values.flatten()
                                })
                                fig2 = px.bar(volume_plot_df, x="Date", y="Volume",
                                              title=f"{stock} - Trading Volume",
                                              labels={"Volume": "Volume"},
                                              hover_data={"Date": "|%Y-%m-%d", "Volume": ":,"})
                                fig2.update_layout(hovermode="x unified", height=400)
                                st.plotly_chart(fig2, use_container_width=True)
                            except Exception as plot_error:
                                st.error(f"❌ Error plotting Trading Volume: {plot_error}")
                        else:
                            st.info("ℹ️ Volume data is not available or empty for plotting.")

                    st.divider()

                    # --- Download Cleaned Data ---
                    st.subheader("⬇️ Download Data")
                    st.markdown("You can download the cleaned historical data in CSV format.")
                    csv = data.to_csv().encode('utf-8')
                    st.download_button(
                        label=f"Download {stock} Data as CSV",
                        data=csv,
                        file_name=f"{stock}_stock_data.csv",
                        mime="text/csv",
                        help="Click to download the fetched stock data."
                    )
                    st.divider()

                    # --- LSTM Model Preparation and Forecasting ---
                    st.subheader("🧠 LSTM Model Training & Future Forecasting")
                    st.markdown("The LSTM model will be trained on the historical closing prices to predict future trends.")

                    close_np = close_data.values.reshape(-1, 1)

                    if len(close_np) < seq_len + forecast_days + 10: # Added a buffer for robust training
                        st.warning(f"⚠️ Not enough data points ({len(close_np)}) to reliably train the LSTM model and make a {forecast_days}-day forecast with a sequence length of {seq_len}. Please select a longer date range. Minimum required: {seq_len + forecast_days + 10} data points.")
                    else:
                        with st.spinner("⏳ Training LSTM model and making predictions... This may take a moment."):
                            try:
                                scaler = MinMaxScaler(feature_range=(0, 1))
                                scaled_data = scaler.fit_transform(close_np)

                                # Function to create sequences for LSTM
                                @st.cache_data(show_spinner=False) # Cache this function as data doesn't change often
                                def create_sequences(dataset, sequence_length):
                                    X, y = [], []
                                    for i in range(sequence_length, len(dataset)):
                                        X.append(dataset[i - sequence_length:i, 0])
                                        y.append(dataset[i, 0])
                                    return np.array(X), np.array(y)

                                # Prepare data for training and testing
                                train_data = scaled_data[:-forecast_days] # Use all data except the last 'forecast_days' for training evaluation

                                if len(train_data) < seq_len:
                                    st.warning(f"⚠️ Insufficient training data ({len(train_data)} points) after reserving {forecast_days} days for forecast evaluation. Required: at least {seq_len} points for training sequences.")
                                    st.stop()

                                X_train, y_train = create_sequences(train_data, seq_len)
                                X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

                                # Build and compile the LSTM model
                                # Use st.cache_resource for the model itself if you want to avoid rebuilding every time
                                @st.cache_resource(show_spinner=False)
                                def build_and_train_model(seq_len_val, lstm_units_val, epochs_val, batch_size_val, X_train_val, y_train_val):
                                    model = Sequential()
                                    model.add(LSTM(lstm_units_val, return_sequences=True, input_shape=(seq_len_val, 1)))
                                    model.add(Dropout(0.2))
                                    model.add(LSTM(lstm_units_val))
                                    model.add(Dropout(0.2))
                                    model.add(Dense(1))
                                    model.compile(optimizer='adam', loss='mean_squared_error')
                                    model.fit(X_train_val, y_train_val, epochs=epochs_val, batch_size=batch_size_val, verbose=0)
                                    return model

                                model = build_and_train_model(seq_len, lstm_units, epochs, batch_size, X_train, y_train)

                                st.success("✅ LSTM Model trained successfully!")

                                st.divider()

                                # --- Accuracy on Last 'forecast_days' Known Days ---
                                st.subheader(f"📊 Model Performance: Last {forecast_days} Known Days")
                                st.markdown("This shows how well the model predicted the most recent historical data it hasn't seen during training.")

                                if len(scaled_data) < seq_len + forecast_days:
                                    st.warning(f"Insufficient data ({len(scaled_data)} points) to evaluate accuracy on the last {forecast_days} known days. Need at least {seq_len + forecast_days} data points.")
                                else:
                                    current_input_seq = scaled_data[-seq_len:].reshape(1, seq_len, 1)
                                    
                                    predicted_known_days = []
                                    for _ in range(forecast_days):
                                        pred = model.predict(current_input_seq, verbose=0)[0][0]
                                        predicted_known_days.append(pred)
                                        current_input_seq = np.append(current_input_seq[:, 1:, :], [[[pred]]], axis=1)

                                    predicted_known_prices = scaler.inverse_transform(np.array(predicted_known_days).reshape(-1, 1))
                                    actual_known_prices = close_np[-forecast_days:]

                                    rmse = np.sqrt(mean_squared_error(actual_known_prices, predicted_known_prices))
                                    mae = mean_absolute_error(actual_known_prices, predicted_known_prices)

                                    st.info(f"📈 **RMSE (Root Mean Squared Error):** {rmse:.2f} (Lower is better)")
                                    st.info(f"📉 **MAE (Mean Absolute Error):** {mae:.2f} (Lower is better)")
                                    st.markdown("RMSE penalizes larger errors more, while MAE gives equal weight to all errors.")

                                    # Plot Accuracy (Plotly Express)
                                    acc_df = pd.DataFrame({
                                        "Date": close_data.index[-forecast_days:],
                                        "Actual Price": actual_known_prices.flatten(),
                                        "Predicted Price": predicted_known_prices.flatten()
                                    })
                                    fig_acc = px.line(acc_df, x="Date", y=["Actual Price", "Predicted Price"],
                                                      title=f"{stock} - Actual vs. Predicted Price (Last {forecast_days} Days)",
                                                      labels={"value": "Price (USD)", "variable": "Type"},
                                                      hover_data={"Date": "|%Y-%m-%d", "value": ":.2f"})
                                    fig_acc.update_layout(hovermode="x unified", height=450)
                                    st.plotly_chart(fig_acc, use_container_width=True)

                                    st.markdown("##### Detailed Accuracy Table:")
                                    st.dataframe(acc_df.set_index("Date").style.format("{:.2f}"))

                                st.divider()

                                # --- Future Forecast (next 'forecast_days' days) ---
                                st.subheader(f"🔮 {forecast_days}-Day Future Price Forecast for {stock}")
                                st.markdown("This forecast is based on the trained LSTM model. Remember, stock market predictions are inherently uncertain.")

                                if len(scaled_data) < seq_len:
                                    st.warning(f"Insufficient historical data ({len(scaled_data)} days) to generate a {forecast_days}-day forecast with a sequence length of {seq_len}. Please extend your 'Start Date'.")
                                else:
                                    last_seq_for_forecast = scaled_data[-seq_len:].reshape(1, seq_len, 1)
                                    
                                    future_predictions = []
                                    for _ in range(forecast_days):
                                        future_pred = model.predict(last_seq_for_forecast, verbose=0)[0][0]
                                        future_predictions.append(future_pred)
                                        last_seq_for_forecast = np.append(last_seq_for_forecast[:, 1:, :], [[[future_pred]]], axis=1)

                                    predicted_future_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

                                    last_known_date = close_data.index[-1]
                                    future_dates_raw = []
                                    current_date = last_known_date + timedelta(days=1)
                                    while len(future_dates_raw) < forecast_days + 7: # Add buffer for weekends/holidays
                                        # Basic check for weekdays. For more robust, need a holiday calendar.
                                        if current_date.weekday() < 5: # Monday=0, Tuesday=1, ..., Friday=4
                                            future_dates_raw.append(current_date)
                                        current_date += timedelta(days=1)

                                    future_dates_business_days = future_dates_raw[:forecast_days]
                                    
                                    num_forecasted_prices = min(len(predicted_future_prices), len(future_dates_business_days))

                                    forecast_df = pd.DataFrame({
                                        "Date": future_dates_business_days[:num_forecasted_prices],
                                        "Predicted Close": predicted_future_prices[:num_forecasted_prices].flatten()
                                    })
                                    forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])

                                    # Forecast Plot (Plotly Express)
                                    historical_plot_df = pd.DataFrame({
                                        "Date": close_data.index,
                                        "Price": close_data.values.flatten(),
                                        "Type": "Historical"
                                    })
                                    forecast_plot_df = pd.DataFrame({
                                        "Date": forecast_df["Date"],
                                        "Price": forecast_df["Predicted Close"],
                                        "Type": "Forecast"
                                    })

                                    combined_plot_df = pd.concat([historical_plot_df, forecast_plot_df])

                                    fig_forecast = px.line(combined_plot_df, x="Date", y="Price", color="Type",
                                                           title=f"{stock} - Historical and {forecast_days}-Day Future Forecast",
                                                           labels={"Price": "Price (USD)"},
                                                           line_dash="Type",
                                                           hover_data={"Date": "|%Y-%m-%d", "Price": ":.2f", "Type": False})
                                    
                                    fig_forecast.update_layout(hovermode="x unified", height=500)
                                    
                                    # Add a rudimentary confidence interval as a shaded area
                                    # This is a very simplistic CI; for real applications, consider more advanced methods.
                                    fig_forecast.add_trace(go.Scatter(
                                        x=forecast_df["Date"],
                                        y=forecast_df["Predicted Close"] * 1.02,
                                        mode='lines',
                                        line=dict(width=0),
                                        showlegend=False,
                                        name='Upper Bound'
                                    ))
                                    fig_forecast.add_trace(go.Scatter(
                                        x=forecast_df["Date"],
                                        y=forecast_df["Predicted Close"] * 0.98,
                                        mode='lines',
                                        line=dict(width=0),
                                        fill='tonexty',
                                        fillcolor='rgba(255,165,0,0.2)', # Orange with transparency
                                        showlegend=True,
                                        name='Approx. 2% CI'
                                    ))

                                    st.plotly_chart(fig_forecast, use_container_width=True)

                                    st.markdown("##### {forecast_days}-Day Forecast Table:")
                                    st.dataframe(forecast_df.set_index("Date").style.format("{:.2f}"))

                            except Exception as model_error:
                                st.error(f"❌ An error occurred during LSTM model training or forecasting: {model_error}")
                                st.info("💡 Please check your data, sequence length, or try adjusting model parameters in the sidebar. Common issues include insufficient data points or overly complex model parameters for the given data.")

        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {e}. Please ensure the stock symbol is correct and you have an active internet connection.")
            st.info("💡 If the issue persists, try refreshing the page or using a different stock symbol.")
else:
    st.info("ℹ️ Please enter a valid stock symbol and ensure the 'Start Date' is before the 'End Date' in the sidebar to begin analysis.")

st.markdown("---")
st.markdown("Disclaimer: This dashboard is for educational purposes only and should not be considered financial advice. Stock market investing involves risks.")
st.markdown("Developed by using Streamlit, yfinance, and TensorFlow.")
st.markdown(
    "🔗 <a href='https://github.com/srikanth-ankam' target='_blank'>GitHub Repository</a> | "
    "© 2025 <b>Ankam Srikanth</b></center>",
    unsafe_allow_html=True
)
