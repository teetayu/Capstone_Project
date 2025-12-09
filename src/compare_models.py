# compare_models.py
# -*- coding: utf-8 -*-
"""
Compare 4 models for a given area:
- Prophet
- ARIMA
- Random Forest
- XGBoost

และสร้าง Bar Chart เปรียบเทียบ MAE / RMSE
"""

import pandas as pd
import matplotlib.pyplot as plt

from model_prophet import train_and_evaluate as prophet_run
from model_arima import train_and_evaluate as arima_run
from model_rf import train_and_evaluate as rf_run
from model_xgboost import train_and_evaluate as xgb_run


def plot_comparison_bar(df_res, area):
    """
    สร้าง Bar Chart เปรียบเทียบ MAE และ RMSE
    """
    models = df_res["model"]
    mae = df_res["MAE"]
    rmse = df_res["RMSE"]

    x = range(len(models))
    width = 0.35

    plt.figure(figsize=(10, 6))

    # MAE bars
    plt.bar([i - width/2 for i in x], mae, width, label='MAE')

    # RMSE bars
    plt.bar([i + width/2 for i in x], rmse, width, label='RMSE')

    plt.xticks(x, models)
    plt.ylabel("Error Value")
    plt.title(f"Model Comparison (MAE & RMSE)\n{area}")
    plt.legend()
    plt.tight_layout()

    filename = f"model_comparison_{area.replace(' ', '_')}.png"
    plt.savefig(filename)
    plt.close()

    print(f" Bar Chart saved as: {filename}")


def compare_for_area(area_name: str, test_size: int = 2, future_periods: int = 5):
    results = []

    print(f"\n========== Comparing models for: {area_name} ==========\n")

    # Prophet
    res_prophet = prophet_run(
        area_name,
        test_size=test_size,
        future_periods=future_periods,
        plot=False,
    )
    print(" Prophet done")
    results.append(res_prophet)

    # ARIMA
    res_arima = arima_run(
        area_name,
        test_size=test_size,
        future_periods=future_periods,
        order=(1, 1, 1),
        plot=False,
    )
    print(" ARIMA done")
    results.append(res_arima)

    # Random Forest
    res_rf = rf_run(
        area_name,
        test_size=test_size,
        future_periods=future_periods,
        plot=False,
    )
    print(" Random Forest done")
    results.append(res_rf)

    # XGBoost
    res_xgb = xgb_run(
        area_name,
        test_size=test_size,
        future_periods=future_periods,
        plot=False,
    )
    print(" XGBoost done")
    results.append(res_xgb)

    # Convert to DataFrame
    df_res = pd.DataFrame(results)
    print("\n Model comparison (lower is better):")
    print(df_res[["model", "MAE", "RMSE"]])

    # Save table
    df_res.to_csv(f"model_comparison_{area_name.replace(' ', '_')}.csv", index=False)
    print(f"\n Saved comparison table: model_comparison_{area_name.replace(' ', '_')}.csv")

    # Plot Bar Chart
    plot_comparison_bar(df_res, area_name)

    return df_res


def main():
    area = "San Diego County"  # ← เปลี่ยนพื้นที่ตรงนี้
    compare_for_area(area, test_size=2, future_periods=5)


if __name__ == "__main__":
    main()