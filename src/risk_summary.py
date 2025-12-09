# -*- coding: utf-8 -*-
"""
Risk Summary Table – พื้นที่ที่เสี่ยงความยากจนเพิ่มขึ้น
ใช้ Prophet เพื่อทำนายทุกพื้นที่ แล้วสร้างตารางสรุปความเสี่ยง
"""

import pandas as pd
from prophet import Prophet


def load_data(path="Book.csv"):
    df = pd.read_csv(path)
    return df


def forecast_next_year(df, area_name):
    """
    ทำนาย Poverty Percent ของพื้นที่ 1 ปีถัดไป
    คืนค่า:
    - last_actual
    - next_year_pred
    - increase (next_year_pred - last_actual)
    """

    year_col = "Year"
    area_col = "Geography Name"
    target_col = "Poverty Percent"

    area_df = df[df[area_col] == area_name].copy().sort_values(year_col)

    if len(area_df) < 2:
        return None  # ต้องมีข้อมูลอย่างน้อย 2 ปี

    # เตรียมข้อมูลสำหรับ Prophet
    area_df["ds"] = pd.to_datetime(area_df[year_col].astype(str) + "-01-01")
    area_df["y"] = area_df[target_col]

    model = Prophet()
    model.fit(area_df[["ds", "y"]])

    # สร้าง dataframe สำหรับทำนาย 1 ปีข้างหน้า
    future = model.make_future_dataframe(periods=1, freq="Y")
    forecast = model.predict(future)

    # ค่าปีล่าสุดจริง
    last_actual = area_df["y"].iloc[-1]

    # ค่าทำนายปีถัดไป
    next_pred = forecast["yhat"].iloc[-1]

    increase = next_pred - last_actual

    return {
        "Area": area_name,
        "Last Actual": round(float(last_actual), 3),
        "Next Year Forecast": round(float(next_pred), 3),
        "Increase (Risk)": round(float(increase), 3)
    }


def build_risk_table(df):
    """
    สร้างตารางสรุปความเสี่ยงทุกพื้นที่
    """

    area_col = "Geography Name"

    results = []

    for area in df[area_col].unique():
        result = forecast_next_year(df, area)
        if result:
            results.append(result)

    risk_df = pd.DataFrame(results)

    # เรียงจากความเสี่ยงเพิ่มขึ้นมากที่สุด → น้อยที่สุด
    risk_df = risk_df.sort_values("Increase (Risk)", ascending=False)

    return risk_df


def main():
    df = load_data("Book.csv")

    risk_df = build_risk_table(df)

    print("\n ตารางสรุปพื้นที่เสี่ยงความยากจนเพิ่มขึ้น (เรียงจากมาก → น้อย)")
    print(risk_df)

    # เซฟเป็นไฟล์ Excel หรือ CSV ก็ได้
    risk_df.to_csv("Risk_Summary.csv", index=False)
    print("\n บันทึกไฟล์: Risk_Summary.csv")


if __name__ == "__main__":
    main()