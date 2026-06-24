import streamlit as st
import requests
import plotly.io as pio
import pandas as pd
from datetime import datetime

# Конфигурация страницы
st.set_page_config(page_title="Time Series Analysis", layout="wide")

# Базовый URL бэкенда и номер порта
API_URL, PORT = "http://localhost", "8000"

# Словарь тикеров
TICKERS = {
    'META': 'Meta',
    'AAPL': 'Apple',
    'AMZN': 'Amazon',
    'NFLX': 'Netflix',
    'GOOGL': 'Google'
}

# Боковая панель с выбором тикера и временным срезом
st.sidebar.title("Parameters")
ticker = st.sidebar.selectbox("Ticker: company", options=list(TICKERS.keys()), format_func=lambda x: TICKERS[x])
start_date = st.sidebar.date_input("Date: start", value=pd.to_datetime('2020-01-01'))
end_date = st.sidebar.date_input("Date: end", value=pd.to_datetime('today'))

if st.sidebar.button("RUN ANALYSIS"):
    with st.spinner("Data loading..."):
        try:
            response = requests.post(
                f"{API_URL}:{PORT}/analyze",
                json={
                    "ticker": ticker,
                    "start_date": start_date.strftime("%Y-%m-%d"),
                    "end_date": end_date.strftime("%Y-%m-%d")
                }
            )
            if response.status_code == 200:
                st.session_state['data'] = response.json()
                # st.success("Data has been loaded!")
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Connection error: {e}")

# Основная область
st.title("Time Series Analysis Dashboard")

if 'data' not in st.session_state:
    st.info("Setup parameters and press RUN ANALYSIS")
    st.stop()

data = st.session_state['data']

# Навигация по страницам
page = st.radio(
    "Analysis steps:",
    ["1. Overview", "2. Analytics", "3. Forecast"],
    horizontal=True
)

# Part 1: Overview
if page == "1. Overview":
    st.header("Time Series Overview")
    col1, col2 = st.columns([1, 3])

    with col1:
        # Значения временного ряда
        st.subheader("Data Preview")
        df = pd.DataFrame(data['data_preview'])
        st.dataframe(df, use_container_width=True)
    with col2:
        # График временного ряда
        st.subheader("Time Series Plot")
        fig = pio.from_json(data['overview_fig'])
        st.plotly_chart(fig, use_container_width=True)

# Part 2: Analytics
elif page == "2. Analytics":
    st.header("Time Series Analytics")

    # Сглаживание
    st.subheader("Smoothing (SMA)")
    fig_smooth = pio.from_json(data['smoothing_fig'])
    st.plotly_chart(fig_smooth, use_container_width=True)

    # Декомпозиция
    st.subheader("Decomposition")
    fig_decomp = pio.from_json(data['decomposition_fig'])
    st.plotly_chart(fig_decomp, use_container_width=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        # Автокорреляция
        st.subheader("Autocorrelation")
        # df_ac = pd.DataFrame(data['autocorr_df'])
        # st.dataframe(df_ac, use_container_width=True)
        st.write("**Вывод:**", data['autocorr_result'])
    with col2:
        # Стационарность
        st.subheader("Stationarity")
        st.write("**Вывод:**", data['stationarity_result'])

# Part 3: Forecast
elif page == "3. Forecast":
    st.header("Time Series Forecasting")

    col1, col2 = st.columns([1, 3])
    with col1:
        # Модель и метрики
        st.subheader("Model & Metrics")
        model_df = pd.DataFrame({
            'Parameter': ['Ensemble', 'Base model', 'Count', 'Alpha'],
            'Value': ['Bagging', 'Ridge', '100', '1.0'],
        })
        st.dataframe(model_df, use_container_width=True)
        metrics_df = pd.DataFrame(data['metrics'].items(), columns=["Metric", "Value"])
        st.dataframe(metrics_df, use_container_width=True)
    with col2:
        # График прогноза
        st.subheader("Forecast Plot")
        fig_forecast = pio.from_json(data['forecast_fig'])
        st.plotly_chart(fig_forecast, use_container_width=True)
