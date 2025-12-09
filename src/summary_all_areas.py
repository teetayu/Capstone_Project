# -*- coding: utf-8 -*-
"""
Summary analysis for ALL areas in Book.csv
วิเคราะห์ทุกพื้นที่พร้อมกัน:
- CAGR
- Volatility
- Max/Min
- Trend (เพิ่ม/ลด)
แล้วสรุปเป็นตาราง CSV
"""

import pandas as pd
import numpy as np


def load_data(path="Book.csv"):
    return pd.read_csv(path)


# -----------------------------
# คำนวณ CAGR
# -----------------------------
def calculate_cagr(df_area):
    df_area = df_area.sort_values("Year")
    first = df_area["Poverty Percent"].iloc[0]
    last = df_area["Poverty Percent"].iloc[-1]
    n = len(df_area) - 1

    if n <= 0:
        return None

    cagr = (last / first) ** (1 / n) - 1
    return round(cagr * 100, 3)


# -----------------------------
# คำนวณ Trend
# -----------------------------
def determine_trend(df_area):
    df_area = df_area.sort_values("Year")

    if df_area["Poverty Percent"].iloc[-1] > df_area["Poverty Percent"].iloc[0]:
        return "Increasing"
    elif df_area["Poverty Percent"].iloc[-1] < df_area["Poverty Percent"].iloc[0]:
        return "Decreasing"
    else:
        return "Stable"


# -----------------------------
# คำนวณ Max/Min
# -----------------------------
def find_extremes(df_area):
    max_row = df_area.loc[df_area["Poverty Percent"].idxmax()]
    min_row = df_area.loc[df_area["Poverty Percent"].idxmin()]

    return (
        int(max_row["Year"]),
        float(max_row["Poverty Percent"]),
        int(min_row["Year"]),
        float(min_row["Poverty Percent"]),
    )


# -----------------------------
# สรุปผลทุกพื้นที่
# -----------------------------
def summarize_all(df):
    area_col = "Geography Name"

    summaries = []

    for area in df[area_col].unique():
        df_area = df[df[area_col] == area].copy().sort_values("Year")

        if len(df_area) < 2:
            continue  # ต้องมีข้อมูลอย่างน้อย 2 ปี

        cagr = calculate_cagr(df_area)
        trend = determine_trend(df_area)
        vol = round(df_area["Poverty Percent"].std(), 3)
        max_year, max_val, min_year, min_val = find_extremes(df_area)

        summaries.append({
            "Area": area,
            "CAGR (%)": cagr,
            "Volatility": vol,
            "Max Year": max_year,
            "Max Value": max_val,
            "Min Year": min_year,
            "Min Value": min_val,
            "Trend": trend,
        })

    summary_df = pd.DataFrame(summaries)
    return summary_df.sort_values("CAGR (%)", ascending=False)


# -----------------------------
# main
# -----------------------------
def main():
    df = load_data("Book.csv")

    summary_df = summarize_all(df)

    print("\n Summary for ALL Areas:")
    print(summary_df)

    summary_df.to_csv("Area_Summary.csv", index=False)
    print("\n Saved: Area_Summary.csv")


if __name__ == "__main__":
    main()