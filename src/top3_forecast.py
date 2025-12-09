# -*- coding: utf-8 -*-
"""
Top 3 Risky Areas – Forecast Plot using Prophet
1) หาพื้นที่ที่มีความยากจนเสี่ยงเพิ่มขึ้นมากที่สุด 3 พื้นที่ (จากปีหน้า)
2) สร้างกราฟ forecast และเซฟเป็น PNG
"""

import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet


# -----------------------------
# โหลดข้อมูล
# -----------------------------
def load_data(path="Book.csv"):
    df = pd.read_csv(path)
    print(" Loaded data:", df.shape)
    return df


# -----------------------------
# ทำนายปีหน้า + คำนวณความเสี่ยงของพื้นที่เดียว
# -----------------------------
def forecast_next_year(df, area_name):
    """
    ทำนาย Poverty Percent ปีหน้า สำหรับพื้นที่เดียว
    คืน dict ที่มี: last_actual, next_year_pred, increase
    """

    year_col = "Year"
    area_col = "Geography Name"
    target_col = "Poverty Percent"

    area_df = df[df[area_col] == area_name].copy().sort_values(year_col)

    # ต้องมีอย่างน้อย 2 ปี ถึงจะดูแนวโน้มได้
    if len(area_df) < 2:
        return None

    # เตรียมข้อมูลสำหรับ Prophet
    area_df["ds"] = pd.to_datetime(area_df[year_col].astype(str) + "-01-01")
    area_df["y"] = area_df[target_col]

    model = Prophet()
    model.fit(area_df[["ds", "y"]])

    # ทำนายเพิ่ม 1 ปี
    future = model.make_future_dataframe(periods=1, freq="Y")
    forecast = model.predict(future)

    last_actual = float(area_df["y"].iloc[-1])
    next_pred = float(forecast["yhat"].iloc[-1])
    increase = next_pred - last_actual

    return {
        "Area": area_name,
        "Last Actual": round(last_actual, 3),
        "Next Year Forecast": round(next_pred, 3),
        "Increase": round(increase, 3),
    }


# -----------------------------
# หา Top 3 พื้นที่เสี่ยงสุด
# -----------------------------
def get_top3_risky_areas(df):
    area_col = "Geography Name"
    results = []

    for area in df[area_col].unique():
        r = forecast_next_year(df, area)
        if r is not None:
            results.append(r)

    risk_df = pd.DataFrame(results)

    # เรียงจาก Increase มาก -> น้อย แล้วเอา Top 3
    risk_df = risk_df.sort_values("Increase", ascending=False)
    top3 = risk_df.head(3).reset_index(drop=True)

    print("\n Top 3 พื้นที่ที่เสี่ยงความยากจนเพิ่มขึ้น (ดูจากปีหน้า):")
    print(top3)

    # เซฟตารางไว้ด้วย เผื่อใช้ในรายงาน
    top3.to_csv("Top3_Risk_Areas.csv", index=False)
    print("\n บันทึกตาราง: Top3_Risk_Areas.csv")

    return top3


# -----------------------------
# สร้างกราฟ forecast และเซฟรูป สำหรับพื้นที่เดียว
# -----------------------------
def forecast_and_plot(df, area_name, periods=5):
    year_col = "Year"
    area_col = "Geography Name"
    target_col = "Poverty Percent"

    area_df = df[df[area_col] == area_name].copy().sort_values(year_col)

    if area_df.empty:
        print(f"⚠ ไม่มีข้อมูลพื้นที่: {area_name}")
        return

    area_df["ds"] = pd.to_datetime(area_df[year_col].astype(str) + "-01-01")
    area_df["y"] = area_df[target_col]

    model = Prophet()
    model.fit(area_df[["ds", "y"]])

    future = model.make_future_dataframe(periods=periods, freq="Y")
    forecast = model.predict(future)

    fig = model.plot(forecast)
    plt.title(f"Poverty Forecast – {area_name}")
    plt.xlabel("Year")
    plt.ylabel("Poverty Percent")
    plt.tight_layout()

    filename = f"Forecast_TOP3_{area_name.replace(' ', '_')}.png"
    plt.savefig(filename)
    plt.close()

    print(f" เซฟกราฟ: {filename}")


# -----------------------------
# main
# -----------------------------
def main():
    df = load_data("Book.csv")

    # 1) หา Top 3 พื้นที่เสี่ยง
    top3_df = get_top3_risky_areas(df)

    # 2) สร้างกราฟ forecast สำหรับ Top 3
    print("\n กำลังสร้างกราฟ forecast สำหรับ Top 3 ...")
    for area in top3_df["Area"]:
        forecast_and_plot(df, area, periods=5)

    print("\n เสร็จแล้ว! ไฟล์รูปเริ่มต้นด้วยชื่อ 'Forecast_TOP3_' พร้อมใช้ในสไลด์")


if __name__ == "__main__":
    main()