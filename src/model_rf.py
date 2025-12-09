# model_rf.py
# -*- coding: utf-8 -*-
"""
Random Forest Regression for Poverty Percent forecasting
- ใช้ Year เป็น feature เดียว (เรียบง่ายสำหรับเทียบกับ Prophet/ARIMA)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


def load_data(path: str = "Book.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_area_features(df: pd.DataFrame, area_name: str):
    sub = df[df["Geography Name"] == area_name].copy()
    if sub.empty:
        raise ValueError(f"No data found for area: {area_name}")
    sub = sub.sort_values("Year")
    X = sub[["Year"]].astype(float).values  # feature = year
    y = sub["Poverty Percent"].astype(float).values
    years = sub["Year"].values
    return X, y, years


def train_test_split_xy(X, y, years, test_size: int = 2):
    if len(y) <= test_size:
        raise ValueError("Not enough data points for the requested test_size.")
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]
    years_train, years_test = years[:-test_size], years[-test_size:]
    return X_train, X_test, y_train, y_test, years_train, years_test


def train_rf(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return mae, rmse, y_pred


def forecast_future(model, last_year: int, periods: int = 5):
    future_years = np.array([last_year + i for i in range(1, periods + 1)]).reshape(-1, 1)
    future_pred = model.predict(future_years)
    return future_years.flatten(), future_pred


def train_and_evaluate(
    area_name: str,
    test_size: int = 2,
    future_periods: int = 5,
    plot: bool = True,
):
    df = load_data("Book.csv")
    X, y, years = prepare_area_features(df, area_name)

    X_train, X_test, y_train, y_test, years_train, years_test = train_test_split_xy(
        X, y, years, test_size=test_size
    )

    model = train_rf(X_train, y_train)

    mae, rmse, y_pred_test = evaluate_model(model, X_test, y_test)

    last_year = int(years.max())
    future_years, future_pred = forecast_future(model, last_year, periods=future_periods)

    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(years, y, label="Actual")
        plt.plot(years_test, y_pred_test, "r--", label="Test Forecast")
        plt.plot(future_years, future_pred, "g--", label="Future Forecast")

        plt.title(f"Random Forest Forecast for {area_name}")
        plt.xlabel("Year")
        plt.ylabel("Poverty Percent")
        plt.legend()
        plt.tight_layout()
        plt.show()

    result = {
        "model": "RandomForest",
        "area": area_name,
        "MAE": float(mae),
        "RMSE": float(rmse),
    }
    return result


def main():
    area = "San Diego County"  # TODO: เปลี่ยนตามพื้นที่
    print(f" Training Random Forest for: {area}")
    res = train_and_evaluate(area, test_size=2, future_periods=5, plot=True)
    print("\n Random Forest performance")
    print(res)


if __name__ == "__main__":
    main()