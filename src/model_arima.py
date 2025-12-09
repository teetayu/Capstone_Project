# model_arima.py
# -*- coding: utf-8 -*-
"""
ARIMA model for Poverty Percent forecasting
- Simple ARIMA(1,1,1) example
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


def load_data(path: str = "Book.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_area_series(df: pd.DataFrame, area_name: str) -> pd.Series:
    sub = df[df["Geography Name"] == area_name].copy()
    if sub.empty:
        raise ValueError(f"No data found for area: {area_name}")
    sub = sub.sort_values("Year")
    # ใช้ Year เป็น index เฉย ๆ (ไม่บังคับเป็น datetime)
    sub = sub.set_index("Year")
    y = sub["Poverty Percent"].astype(float)
    return y


def train_test_split_series(y: pd.Series, test_size: int = 2):
    if len(y) <= test_size:
        raise ValueError("Not enough data points for the requested test_size.")
    train = y.iloc[:-test_size]
    test = y.iloc[-test_size:]
    return train, test


def train_arima(train: pd.Series, order=(1, 1, 1)) -> ARIMA:
    model = ARIMA(train, order=order)
    fitted = model.fit()
    return fitted


def evaluate_model(model, test: pd.Series):
    # forecast same length as test
    n_test = len(test)
    forecast = model.forecast(steps=n_test)

    y_true = test.values
    y_pred = forecast.values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return mae, rmse, forecast


def forecast_future(model, periods: int = 5):
    future_forecast = model.forecast(steps=periods)
    return future_forecast


def train_and_evaluate(
    area_name: str,
    test_size: int = 2,
    future_periods: int = 5,
    order=(1, 1, 1),
    plot: bool = True,
):
    df = load_data("Book.csv")
    y = prepare_area_series(df, area_name)

    train, test = train_test_split_series(y, test_size=test_size)
    fitted = train_arima(train, order=order)

    mae, rmse, forecast_test = evaluate_model(fitted, test)

    # ทำ series สำหรับ plot
    full_index = y.index
    full_model = train_arima(y, order=order)
    future_forecast = forecast_future(full_model, periods=future_periods)

    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(y.index, y.values, label="Actual")
        plt.plot(test.index, forecast_test.values, "r--", label="Test Forecast")

        # index ของอนาคต (ต่อจากปีสุดท้าย)
        last_year = int(y.index.max())
        future_years = [last_year + i for i in range(1, future_periods + 1)]
        plt.plot(future_years, future_forecast.values, "g--", label="Future Forecast")

        plt.title(f"ARIMA Forecast for {area_name}")
        plt.xlabel("Year")
        plt.ylabel("Poverty Percent")
        plt.legend()
        plt.tight_layout()
        plt.show()

    result = {
        "model": "ARIMA",
        "area": area_name,
        "MAE": float(mae),
        "RMSE": float(rmse),
        "order": order,
    }
    return result


def main():
    area = "San Diego County"  # TODO: เปลี่ยนตามพื้นที่
    print(f" Training ARIMA for: {area}")
    res = train_and_evaluate(area, test_size=2, future_periods=5, order=(1, 1, 1))
    print("\n ARIMA performance")
    print(res)


if __name__ == "__main__":
    main()