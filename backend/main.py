from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import plotly.io as pio
from models import AnalyzeRequest
from services import *
from utils import get_ticker_name

app = FastAPI(title="Time Series Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    try:
        # Загрузка данных
        stocks = data_stocks_load(request.ticker, request.start_date, request.end_date)
        raw_TS = data_to_timeseries(stocks)
        TS = timeseries_missing_fill(raw_TS)
        ticker_name = get_ticker_name(request.ticker)

        # Part 1 – Overview
        data_preview = TS.reset_index().to_dict(orient='records')
        overview_fig = get_overview_fig(TS, ticker_name)

        # Part 2 – Analytics
        smoothing_fig = get_smoothing_fig(TS, request.smoothings, ticker_name)
        decomposition_fig = get_decomposition_fig(TS, request.model_type, request.period_value)
        autocorr_df, autocorr_result = timeseries_autocorrelation(TS, lags=100)
        stationarity_result = timeseries_stationarity(TS['Close'])

        # Part 3 – Forecasting
        X_train, X_test, y_train, y_test = timeseries_preprocessing_to_dataset_with_train_test_split(
            TS,
            test_size=request.test_size,
            lag_begin=request.lag_begin,
            lag_end=request.lag_end
        )
        model = BaggingRegressor(
            estimator=Ridge(alpha=1.0, solver='auto'),
            n_estimators=100,
            max_samples=0.8,
            bootstrap=True,
            bootstrap_features=False,
            n_jobs=-1,
            random_state=42
        )
        y_pred = timeseries_model_training_and_prediction(model, X_train, X_test, y_train)
        metrics_names, metrics_values = timeseries_model_metrics(y_test, y_pred)
        metrics_dict = dict(zip(metrics_names, metrics_values))
        forecast_fig = get_forecast_fig(y_train, y_test, y_pred, ticker_name)

        # Сериализация в JSON
        return {
            "overview_fig": pio.to_json(overview_fig),
            "data_preview": data_preview,
            "smoothing_fig": pio.to_json(smoothing_fig),
            "decomposition_fig": pio.to_json(decomposition_fig),
            "autocorr_df": autocorr_df.to_dict(orient='records'),
            "autocorr_result": autocorr_result,
            "stationarity_result": stationarity_result,
            "forecast_fig": pio.to_json(forecast_fig),
            "metrics": metrics_dict,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
