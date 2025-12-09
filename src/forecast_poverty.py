# -*- coding: utf-8 -*-
"""
Poverty Forecast using Facebook Prophet
แยกไฟล์ออกจาก EDA เพื่อให้โปรเจกต์ดูเป็นสัดส่วนชัดเจน
"""

import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet


# -----------------------------
# Load data
# -----------------------------
def load_data(path="Book.csv"):
    df = pd.read_csv(path)
    print("Loaded:", df.shape, "rows")
    return df


# -----------------------------
# Forecast function
# -----------------------------
def forecast_poverty(df, area_name, periods=5):
    """
    พยากรณ์ความยากจนด้วย Prophet แยกตามพื้นที่
    - df: DataFrame
    - area_name: ชื่อพื้นที่ เช่น "San Diego County"
    - periods: ต้องการพยากรณ์กี่ปีข้างหน้า
    """

    year_col = "Year"
    area_col = "Geography Name"
    target_col = "Poverty Percent"

    # 1) เลือกข้อมูลเฉพาะพื้นที่
    area_df = df[df[area_col] == area_name].copy().sort_values(year_col)

    if area_df.empty:
        print(f"⚠ ไม่มีข้อมูลพื้นที่: {area_name}")
        print("พื้นที่ที่มีอยู่:", df[area_col].unique())
        return

    print(f"พื้นที่ที่เลือก:", area_name)
    print(area_df[[year_col, target_col]])

    # 2) เตรียมข้อมูลให้ตรงกับรูปแบบ Prophet
    area_df["ds"] = pd.to_datetime(area_df[year_col].astype(str) + "-01-01")
    area_df["y"] = area_df[target_col]

    # 3) สร้างโมเดล + Training
    model = Prophet()
    model.fit(area_df[["ds", "y"]])

    # 4) Create future dataframe
    future = model.make_future_dataframe(periods=periods, freq="Y")

    # 5) Predict
    forecast = model.predict(future)

    # 6) Plot ผลลัพธ์
    fig = model.plot(forecast)
    plt.title(f"Poverty Forecast: {area_name}")
    plt.xlabel("Year")
    plt.ylabel("Poverty Percent")
    plt.tight_layout()
    plt.show()

    # 7) แสดงเฉพาะค่าทำนายในอนาคต
    print("\n ค่าพยากรณ์อนาคต:")
    future_only = forecast[forecast["ds"] > area_df["ds"].max()][
        ["ds", "yhat", "yhat_lower", "yhat_upper"]
    ]
    print(future_only)


# -----------------------------
# Main
# -----------------------------
def main():
    df = load_data("Book.csv")

    #ตัวอย่างการทำนาย
    forecast_poverty(
        df,
        area_name="San Diego County",  # ← เปลี่ยนตาม dataset ของ Tee
        periods=5                      # ทำนายอีก 5 ปี
    )


if __name__ == "__main__":
    main()