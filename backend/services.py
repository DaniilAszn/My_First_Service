import yfinance as yf
import numpy as np
import pandas as pd
import datetime as dt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from typing import Union, Tuple, Dict, Any
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


'''
Overview
'''
def data_stocks_load(
    ticker: str,
    start_date: str,
    end_date: str
    ) -> pd.DataFrame:
    return yf.download(ticker, start=start_date, end=end_date)


def data_to_timeseries(
    data: pd.DataFrame
    ) -> pd.DataFrame:
    data.reset_index(inplace=True)
    timeseries = pd.DataFrame()
    timeseries['Date'] = data['Date']
    timeseries['Close'] = data['Close']
    timeseries = timeseries.set_index('Date')

    return timeseries


def timeseries_missing_fill(
    timeseries: pd.DataFrame
    ) -> pd.DataFrame:
    full_index = pd.date_range(start=timeseries.index.min(), end=timeseries.index.max(), freq='D')
    timeseries = timeseries.reindex(full_index)
    timeseries['Close'] = timeseries['Close'].interpolate(method='time')

    return timeseries


def get_overview_fig(
    timeseries: pd.DataFrame,
    ticker_name: str
    ) -> go.Figure:
    fig = px.line(timeseries, x=timeseries.index, y=timeseries['Close'], title=f"{ticker_name}'s stocks")

    fig.update_layout(template='plotly_white', width=1000, height=500)
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Close")

    return fig


'''
Analytics
'''
def get_smoothing_fig(
    timeseries: pd.DataFrame,
    smoothings: list,
    ticker_name: str
    ) -> go.Figure:
    fig = px.line(title=f"{ticker_name}'s stocks with SMA smoothing")

    fig.add_scatter(x=timeseries.index, y=timeseries['Close'], mode='lines', name='Original stocks', line=dict(color='silver'))
    colors = ['yellow', 'orange', 'red']
    for i, w in enumerate(smoothings):
        fig.add_scatter(
            x=timeseries.index,
            y=timeseries.rolling(window=w).mean()['Close'],
            mode='lines',
            name=f'SMA, window={w}',
            line=dict(color=colors[i % len(colors)])
        )

    fig.update_layout(template='plotly_white', width=1000, height=500)
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Close")

    return fig


def get_decomposition_fig(
    timeseries: pd.DataFrame,
    model_type: str,
    period_value: int
    ) -> go.Figure:
    timeseries.index = pd.to_datetime(timeseries.index)
    decompose = seasonal_decompose(timeseries['Close'], model=model_type, period=period_value)
    decompose_observed = decompose.observed
    decompose_trend = decompose.trend
    decompose_seasonal = decompose.seasonal
    decompose_resid = decompose.resid

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Observed', 'Trend', 'Seasonal', 'Residual')
    )
    fig.add_trace(go.Scatter(
        x=decompose_observed.index,
        y=decompose_observed,
        mode='lines',
        name='Observed',
        line=dict(color='blue', width=1.5)),
        row=1,
        col=1
    )
    fig.add_trace(go.Scatter(
        x=decompose_trend.index,
        y=decompose_trend,
        mode='lines',
        name='Trend',
        line=dict(color='red', width=1.5)),
        row=2,
        col=1
    )
    fig.add_trace(go.Scatter(
        x=decompose_seasonal.index,
        y=decompose_seasonal,
        mode='lines',
        name='Seasonal',
        line=dict(color='green', width=1.5)),
        row=3,
        col=1
    )
    fig.add_trace(go.Scatter(
        x=decompose_resid.index,
        y=decompose_resid,
        mode='markers',
        name='Residual',
        marker=dict(color='black', size=2, opacity=0.5)),
        row=4,
        col=1
    )

    fig.update_layout(
        width=1000,
        height=600,
        title_text='Time series decomposition',
        showlegend=False,
        template='plotly_white'
    )
    fig.update_yaxes(title_text="Close", row=1, col=1)
    fig.update_yaxes(title_text="Close", row=2, col=1)
    fig.update_yaxes(title_text="Close", row=3, col=1)
    fig.update_yaxes(title_text="Close", row=4, col=1)
    fig.update_xaxes(title_text="Data", row=4, col=1)

    return fig


def timeseries_autocorrelation(
    timeseries: pd.DataFrame,
    lags: int=100
    ) -> Tuple[pd.DataFrame, str]:
    acorr_ljungbox_df = acorr_ljungbox(timeseries['Close'], return_df=True, lags=lags)
    len_acorr_ljungbox_df = acorr_ljungbox_df[acorr_ljungbox_df['lb_pvalue'] >= 0.05].shape[0]

    if len_acorr_ljungbox_df == 0:
        return acorr_ljungbox_df, "Автокорреляция присутствует на всех проверяемых лагах"
    else:
        return acorr_ljungbox_df, "Автокорреляция отсутствует на некоторых лагах"


def adf_test(
    timeseries: pd.Series
    ) -> None:
    print("Results of Dickey-Fuller Test:")
    dftest = adfuller(timeseries, autolag="AIC")
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "Lags Used",
            "Number of Observations Used",
        ]
    )
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    print(dfoutput)


def kpss_test(
    timeseries: pd.Series
    ) -> None:
    print("Results of KPSS Test:")
    kpsstest = kpss(timeseries, regression="c", nlags="auto")
    kpss_output = pd.Series(
        kpsstest[0:3],
        index=[
            "Test Statistic",
            "p-value",
            "Lags Used"
        ]
    )
    for key, value in kpsstest[3].items():
        kpss_output["Critical Value (%s)" % key] = value
    print(kpss_output)


def timeseries_stationarity(
    timeseries: pd.Series
    ) -> str:
    df_test = adfuller(timeseries, autolag="AIC")
    kpss_test = kpss(timeseries, regression="c", nlags="auto")

    df_p_value = round(df_test[1], 5)
    kpss_p_value = round(kpss_test[1], 5)

    if (df_p_value <= 0.05) and (kpss_p_value > 0.05):
        return "Временной ряд стационарный"
    elif (df_p_value > 0.05) and (kpss_p_value <= 0.05):
        return "Временной ряд нестационарный"
    elif (df_p_value <= 0.05) and (kpss_p_value <= 0.05):
        return "Результаты противоречивы"
    elif (df_p_value > 0.05) and (kpss_p_value > 0.05):
        return "Требуется больше данных"


'''
Forecasting
'''
def code_mean(
    data: pd.DataFrame,
    time_feature: str,
    real_feature: str
    ) -> Dict[Any, float]:
    return dict(data.groupby(time_feature)[real_feature].mean())


def code_std(
    data: pd.DataFrame,
    time_feature: str,
    real_feature: str
    ) -> Dict[Any, float]:
    return dict(data.groupby(time_feature)[real_feature].std(ddof=1))


def timeseries_preprocessing_to_dataset_with_train_test_split(
    timeseries: pd.DataFrame,
    test_size: float=0.2,
    lag_begin: int=1,
    lag_end: int=7
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    timeseries = pd.DataFrame(timeseries.copy())

    # считаем индекс в датафрейме, после которого начинается тестовый отрезок
    test_index = int(len(timeseries) * (1 - test_size))

    # добавляем лаги исходного ряда в качестве признаков
    for i in range(lag_begin, lag_end + 1):
        timeseries[f"lag_{i}"] = timeseries.Close.shift(i)

    timeseries["weekday"] = timeseries.index.weekday
    # timeseries["week"] = timeseries.index.week
    timeseries["month"] = timeseries.index.month
    timeseries["year"] = timeseries.index.year

    # считаем средние только по тренировочной части, чтобы избежать лика (data leak)
    timeseries["weekday_mean"] = list(map(code_mean(timeseries[:test_index], "weekday", "Close").get, timeseries.weekday))
    # timeseries["week_mean"] = list(map(code_mean(timeseries[:test_index], "week", "Close").get, timeseries.week))
    timeseries["month_mean"] = list(map(code_mean(timeseries[:test_index], "month", "Close").get, timeseries.month))
    timeseries["year_mean"] = list(map(code_mean(timeseries[:test_index], "year", "Close").get, timeseries.year))

    # считаем отклонения только по тренировочной части, чтобы избежать лика (data leak)
    timeseries["weekday_std"] = list(map(code_std(timeseries[:test_index], "weekday", "Close").get, timeseries.weekday))
    # timeseries["week_std"] = list(map(code_std(timeseries[:test_index], "week", "Close").get, timeseries.week))
    timeseries["month_std"] = list(map(code_std(timeseries[:test_index], "month", "Close").get, timeseries.month))
    timeseries["year_std"] = list(map(code_std(timeseries[:test_index], "year", "Close").get, timeseries.year))

    # выкидываем закодированные средними признаки
    timeseries.drop(["weekday", 'month', 'year'], axis=1, inplace=True)

    timeseries = timeseries.dropna()
    timeseries = timeseries.reset_index(drop=True)

    # разбиваем весь датасет на тренировочную и тестовую выборку
    X_train = timeseries.loc[:test_index].drop(["Close"], axis=1)
    y_train = timeseries.loc[:test_index]["Close"]
    X_test = timeseries.loc[test_index:].drop(["Close"], axis=1)
    y_test = timeseries.loc[test_index:]["Close"]

    return X_train, X_test, y_train, y_test


def timeseries_model_training_and_prediction(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: Union[pd.Series, np.ndarray]
    ) -> Union[pd.Series, np.ndarray]:
    model.fit(X_train, y_train)
    y_pred = pd.Series(model.predict(X_test))

    return y_pred


def symmetric_mean_absolute_percentage_error(
    y_test: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray]
    ) -> float:
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)

    denominator = (np.abs(y_test) + np.abs(y_pred)) / 2
    mask = denominator != 0

    if not np.any(mask):
        return 0.0  # все значения нулевые – ошибка 0

    return np.mean(np.abs(y_test[mask] - y_pred[mask]) / denominator[mask])


def timeseries_model_metrics(
    y_test: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray]
    ) -> Tuple[Tuple[str], Tuple[float]]:
    R2 = round(r2_score(y_test, y_pred), 5)
    RMSE = round(root_mean_squared_error(y_test, y_pred), 5)
    MAE = round(mean_absolute_error(y_test, y_pred), 5)
    MAPE = round(mean_absolute_percentage_error(y_test, y_pred) * 100, 5)
    SMAPE = round(symmetric_mean_absolute_percentage_error(y_test, y_pred)  * 100, 5)

    return ('R2', 'RMSE', 'MAE', 'MAPE', 'SMAPE'), (R2, RMSE, MAE, MAPE, SMAPE)


def get_forecast_fig(
    y_train: Union[pd.Series, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    ticker_name: str
    ) -> go.Figure:
    fig = px.line(title=f"{ticker_name}'s stocks forecasting")

    fig.add_scatter(x=y_train.index, y=y_train, mode='lines', name='y_train', line=dict(color='blue'))
    fig.add_scatter(x=y_test.index, y=y_test, mode='lines', name='y_test', line=dict(color='green'))
    fig.add_scatter(x=y_test.index, y=y_pred, mode='lines', name='y_pred', line=dict(color='red'))

    fig.update_layout(template='plotly_white', width=1000, height=500)
    fig.update_xaxes(title_text="Index")
    fig.update_yaxes(title_text="Values")

    return fig