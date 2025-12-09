# model_prophet.py
# -*- coding: utf-8 -*-
"""
Prophet model for Poverty Percent forecasting
- Train/Test split
- Train model
- Evaluate with MAE, RMSE
- Forecast future
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


def load_data(path: str = "Book.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_area(df: pd.DataFrame, area_name: str) -> pd.DataFrame:
    sub = df[df["Geography Name"] == area_name].copy()
    if sub.empty:
        raise ValueError(f"No data found for area: {area_name}")
    sub["ds"] = pd.to_datetime(sub["Year"].astype(str) + "-01-01")
    sub["y"] = sub["Poverty Percent"]
    sub = sub.sort_values("ds")
    return sub[["ds", "y"]]


def train_test_split(df_area: pd.DataFrame, test_size: int = 2):
    if len(df_area) <= test_size:
        raise ValueError("Not enough data points for the requested test_size.")
    train = df_area.iloc[:-test_size]
    test = df_area.iloc[-test_size:]
    return train, test


def train_prophet(train_df: pd.DataFrame) -> Prophet:
    model = Prophet()
    model.fit(train_df)
    return model


def evaluate_model(model: Prophet, test_df: pd.DataFrame):
    future = test_df[["ds"]]  # predict on test dates
    forecast = model.predict(future)

    y_true = test_df["y"].values
    y_pred = forecast["yhat"].values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return mae, rmse, forecast


def forecast_future(model: Prophet, periods: int = 5):
    future = model.make_future_dataframe(periods=periods, freq="Y")
    forecast = model.predict(future)
    return forecast


def train_and_evaluate(
    area_name: str,
    test_size: int = 2,
    future_periods: int = 5,
    plot: bool = True,
):
    df = load_data("Book.csv")
    df_area = prepare_area(df, area_name)

    train_df, test_df = train_test_split(df_area, test_size=test_size)
    model = train_prophet(train_df)

    mae, rmse, forecast_test = evaluate_model(model, test_df)

    forecast_full = forecast_future(model, periods=future_periods)

    if plot:
        fig = model.plot(forecast_full)
        plt.title(f"Prophet Forecast for {area_name}")
        plt.xlabel("Year")
        plt.ylabel("Poverty Percent")
        plt.tight_layout()
        plt.show()

    result = {
        "model": "Prophet",
        "area": area_name,
        "MAE": float(mae),
        "RMSE": float(rmse),
    }
    return result


def main():
    area = "San Diego County"  # TODO: เปลี่ยนเป็นพื้นที่ของ Tee
    print(f" Training Prophet for: {area}")
    res = train_and_evaluate(area, test_size=2, future_periods=5, plot=True)
    print("\n Prophet performance")
    print(res)


if __name__ == "__main__":
    main()