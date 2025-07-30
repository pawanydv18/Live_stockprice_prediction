import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from keras.models import load_model
import pickle
import time
import uuid

# Streamlit page setup
st.set_page_config(page_title="📈 Stock Candlestick Predictor", layout="wide")
st.title("📈 Live Candlestick + Volume Chart with Prediction History")

# Input: Ticker
ticker = st.text_input("Enter Stock Ticker", "AAPL")
interval = "1m"
period = "1d"

# Start/Stop buttons
col1, col2 = st.columns([1, 2])
start_button = col1.button("▶ Start Live Update")
stop_button = col2.button("⏹ Stop")

# Session state
if 'run' not in st.session_state:
    st.session_state.run = False
if start_button:
    st.session_state.run = True
if stop_button:
    st.session_state.run = False

# Store previous predictions
if 'predictions' not in st.session_state:
    st.session_state.predictions = []

# Load model and scaler
try:
    model = load_model('model.h5', compile=False)
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except Exception as e:
    st.error(f"❌ Failed to load model or scaler: {e}")
    st.stop()

# Predict function
def get_prediction(data):
    if data.shape[0] < 60:
        return None
    last_60_days = data[-60:][['Close']].values
    scaled = scaler.transform(last_60_days)
    x_input = scaled.reshape(1, 60, 1)
    pred_scaled = model.predict(x_input)
    pred = scaler.inverse_transform(pred_scaled)
    return pred[0][0]

# UI output containers
chart = st.empty()
prediction_area = st.empty()

previous_index = None

# Start live update loop
while st.session_state.run:
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

        if df.empty or df.shape[0] < 60:
            prediction_area.warning("⚠ Waiting for enough data...")
            time.sleep(5)
            continue

        latest_index = df.index[-1]
        if previous_index == latest_index:
            time.sleep(1)
            continue  # wait for new data

        previous_index = latest_index

        predicted_price = get_prediction(df)
        next_time = latest_index + pd.Timedelta(minutes=1)

        if predicted_price:
            st.session_state.predictions.append((next_time, predicted_price))
            prediction_area.markdown(f"### 📍 Predicted Next Price: *${predicted_price:.2f}*")
        else:
            prediction_area.warning("⚠ Prediction failed.")
            time.sleep(5)
            continue

        # Create candlestick + volume plot
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.7, 0.3], vertical_spacing=0.03)

        # 🕯 Candlestick
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Candlestick'
        ), row=1, col=1)

        # 🔴 All previous predictions
        if st.session_state.predictions:
            pred_times, pred_values = zip(*st.session_state.predictions)
            fig.add_trace(go.Scatter(
                x=pred_times,
                y=pred_values,
                mode='markers+lines+text',
                name='Predictions',
                marker=dict(color='red', size=8),
                text=[f"{p:.2f}" for p in pred_values],
                textposition="top center",
                line=dict(color='red', dash='dot')
            ), row=1, col=1)

        # 📊 Volume bars
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['Volume'],
            marker_color='lightblue',
            name='Volume'
        ), row=2, col=1)

        fig.update_layout(
            title=f"{ticker} Live Candlestick + Prediction History",
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            height=700,
            showlegend=False
        )

        # Update chart
        chart.plotly_chart(fig, use_container_width=True, key=str(uuid.uuid4()))
        time.sleep(1)

    except Exception as e:
        prediction_area.error(f"❌ Error occurred: {e}")
        time.sleep(5)