# -*- coding: utf-8 -*-
"""
Forecast Poverty Percent for ALL Areas using Prophet
เซฟกราฟอัตโนมัติเป็น .png สำหรับแต่ละพื้นที่
"""

import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet


def load_data(path="Book.csv"):
    return pd.read_csv(path)


def forecast_and_plot(df, area_name, periods=5):
    """
    ทำนายความยากจนของพื้นที่ + เซฟกราฟ
    """

    df_area = df[df["Geography Name"] == area_name].copy().sort_values("Year")

    if df_area.empty:
        print(f"⚠ ไม่มีข้อมูล: {area_name}")
        return

    # เตรียมข้อมูล
    df_area["ds"] = pd.to_datetime(df_area["Year"].astype(str) + "-01-01")
    df_area["y"] = df_area["Poverty Percent"]

    # Train model
    model = Prophet()
    model.fit(df_area[["ds", "y"]])

    # Forecast
    future = model.make_future_dataframe(periods=periods, freq="Y")
    forecast = model.predict(future)

    # Plot
    fig = model.plot(forecast)
    plt.title(f"Poverty Forecast: {area_name}")
    plt.xlabel("Year")
    plt.ylabel("Poverty Percent")
    plt.tight_layout()

    # Save as PNG
    filename = f"Forecast_{area_name.replace(' ', '_')}.png"
    plt.savefig(filename)
    plt.close()

    print(f" Saved forecast graph: {filename}")


def main():
    df = load_data("Book.csv")

    areas = df["Geography Name"].unique()

    print(" เริ่มทำนายทุกพื้นที่...")
    for area in areas:
        forecast_and_plot(df, area, periods=5)

    print("\n เสร็จสิ้น! ไฟล์ทั้งหมดถูกเซฟเป็น PNG แล้ว")


if __name__ == "__main__":
    main()