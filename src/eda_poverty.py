# -*- coding: utf-8 -*-
"""
EDA Charts for Poverty Dataset (Book.csv)
ใช้กับ VS Code ได้เลย
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ตั้งค่าธีมกราฟให้ดูง่ายหน่อย
sns.set(style="whitegrid")


# -----------------------------
# 0) ฟังก์ชันโหลดข้อมูล
# -----------------------------
def load_data(csv_path: str = "Book.csv") -> pd.DataFrame:
    """
    อ่านไฟล์ CSV และคืนค่าเป็น DataFrame
    """
    df = pd.read_csv(csv_path)

    print("Loaded data shape:", df.shape)
    print("Columns:", list(df.columns))
    print(df.head())

    return df


# -----------------------------
# 1) Line Chart – Poverty Percent by Year
# -----------------------------
def plot_line_trend(df: pd.DataFrame):
    """
    แนวโน้มความยากจน (Poverty Percent) ของแต่ละพื้นที่ตามปี
    """

    # ป้องกันกรณี column ชื่อไม่ตรง (ปรับเองตามจริงได้)
    year_col = "Year"
    area_col = "Geography Name"
    poverty_pct_col = "Poverty Percent"

    plt.figure(figsize=(10, 6))

    for name, g in df.groupby(area_col):
        g_sorted = g.sort_values(year_col)
        plt.plot(
            g_sorted[year_col],
            g_sorted[poverty_pct_col],
            marker="o",
            alpha=0.4,
            label=name,
        )

    plt.title("Poverty Percent by Year(all areas)")
    plt.xlabel("Year")
    plt.ylabel("Poverty Percent")
    # legend ด้านข้าง
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.show()


# -----------------------------
# 2) Bar Chart – Poverty Percent Comparison Across Areas (รวมทุกปี)
# -----------------------------
def plot_bar_comparison(df: pd.DataFrame, top_n: int = 20):
    """
    เปรียบเทียบ Poverty Percent ระหว่างพื้นที่ (รวมทุกปี)
    - ใช้ค่าเฉลี่ย Poverty Percent ของแต่ละพื้นที่ตลอดทุกปี
    - เรียงจากจนมาก -> น้อย
    - แสดงเฉพาะ Top N เพื่อให้กราฟอ่านง่าย (default = 20)
    """

    area_col = "Geography Name"
    poverty_pct_col = "Poverty Percent"

    # รวมทุกปี -> เอาค่าเฉลี่ยของแต่ละพื้นที่
    df_avg = (
        df.groupby(area_col)[poverty_pct_col]
        .mean()
        .reset_index()
        .rename(columns={poverty_pct_col: "Poverty Percent Mean"})
    )

    # เรียงจากจนมาก -> น้อย
    df_avg = df_avg.sort_values("Poverty Percent Mean", ascending=False)

    # เลือกเฉพาะ Top N (ถ้าข้อมูลน้อยกว่า N ก็ใช้ทั้งหมด)
    if top_n is not None and top_n < len(df_avg):
        df_plot = df_avg.head(top_n)
        title = f"Top {top_n} Areas by Average Poverty Percent (All Years)"
    else:
        df_plot = df_avg
        title = "Average Poverty Percent by Area (All Years)"

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df_plot,
        x=area_col,
        y="Poverty Percent Mean",
    )

    plt.title(title)
    plt.xlabel("Geography Name")
    plt.ylabel("Average Poverty Percent")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


# -----------------------------
# 3) Histogram – Distribution of Poverty Percent
# -----------------------------
def plot_histogram(df: pd.DataFrame):
    """
    การกระจายตัวของ Poverty Percent ทั้ง dataset
    """

    poverty_pct_col = "Poverty Percent"

    plt.figure(figsize=(8, 5))
    sns.histplot(
        df[poverty_pct_col],
        bins=20,
        kde=True,  # เส้น density ช่วยให้เห็นรูปทรง distribution
    )
    plt.title("Distribution of Poverty Percent")
    plt.xlabel("Poverty Percent")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


# -----------------------------
# 4) Box Plot – Outlier Detection
# -----------------------------
def plot_boxplot(df: pd.DataFrame):
    """
    หา outliers ของ Poverty Percent แยกตามพื้นที่
    """

    area_col = "Geography Name"
    poverty_pct_col = "Poverty Percent"

    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=df,
        x=area_col,
        y=poverty_pct_col,
    )

    plt.title("Boxplot of Poverty Percent by Area (every year combined)")
    plt.xlabel("Geography Name")
    plt.ylabel("Poverty Percent")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


# -----------------------------
# 5) Heatmap (Correlation Matrix)
# -----------------------------
def plot_corr_heatmap(df: pd.DataFrame):
    """
    Heatmap ความสัมพันธ์ระหว่างตัวแปรตัวเลข
    เช่น Poverty Percent, Poverty Estimate, Population Estimate
    """

    # ปรับ list นี้ให้ตรงกับ column จริงใน Book.csv ของ Tee
    numeric_cols = [
        "Poverty Percent",
        "Poverty Estimate",
        "Population Estimate",
    ]

    # เลือกเฉพาะคอลัมน์ที่มีอยู่จริง
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    if not numeric_cols:
        print("⚠ ไม่มีคอลัมน์ตัวเลขที่ตรงกับชื่อที่กำหนดไว้")
        return

    corr = df[numeric_cols].corr()

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
    )

    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()


# -----------------------------
# main() – จุดเริ่มรันไฟล์
# -----------------------------
def main():
    df = load_data("Book.csv")

    # เรียกทีละกราฟ
    plot_line_trend(df)
    plot_bar_comparison(df, top_n=20)
    plot_histogram(df)
    plot_boxplot(df)
    plot_corr_heatmap(df)


if __name__ == "__main__":
    main()