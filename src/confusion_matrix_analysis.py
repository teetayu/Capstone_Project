# -*- coding: utf-8 -*-
"""
confusion_matrix_analysis.py

สร้าง Confusion Matrix สำหรับงาน:
CLASS 1 = พื้นที่ที่ความยากจน "เพิ่มขึ้น"
CLASS 0 = พื้นที่ที่ความยากจน "ไม่เพิ่ม"

ขั้นตอน:
1) เลือกพื้นที่ (area_name)
2) เทรน Prophet จากข้อมูลของพื้นที่นั้น
3) ให้ Prophet ทำนายค่าความยากจนในทุกปีในอดีต
4) แปลงค่าจริง และค่าทำนาย เป็น Class 0/1 ตามว่า "เพิ่มขึ้นจากปีก่อน" หรือไม่
5) คำนวณ Confusion Matrix (TP, TN, FP, FN) + metrics อื่น ๆ
6) (option) วาด Confusion Matrix เป็นรูป
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


# -----------------------------
# 1) โหลดข้อมูล & เตรียมเฉพาะพื้นที่
# -----------------------------
def load_data(path: str = "Book.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def prepare_area(df: pd.DataFrame, area_name: str) -> pd.DataFrame:
    """
    ดึงข้อมูลเฉพาะพื้นที่ แล้วเตรียมคอลัมน์ ds, y สำหรับ Prophet
    """
    sub = df[df["Geography Name"] == area_name].copy()
    if sub.empty:
        raise ValueError(f"ไม่พบพื้นที่: {area_name}")
    sub = sub.sort_values("Year")
    sub["ds"] = pd.to_datetime(sub["Year"].astype(str) + "-01-01")
    sub["y"] = sub["Poverty Percent"].astype(float)
    return sub[["Year", "ds", "y"]]


# -----------------------------
# 2) สร้าง label จริง (y_true_class)
# -----------------------------
def create_true_labels(area_df: pd.DataFrame) -> pd.DataFrame:
    """
    label จริง: 1 ถ้าปีนี้ > ปีที่แล้ว, 0 ถ้าไม่เพิ่มขึ้น
    แถวแรกจะไม่มี label เพราะไม่มีปีที่แล้ว -> ตัดทิ้ง
    """
    df = area_df.copy()
    df["diff_true"] = df["y"].diff()
    df["y_true_class"] = (df["diff_true"] > 0).astype(int)

    # ตัดแถวแรก (diff = NaN)
    df = df.iloc[1:].reset_index(drop=True)
    return df


# -----------------------------
# 3) เทรน Prophet และทำนายทุกปีในอดีต
# -----------------------------
def prophet_predict(area_df: pd.DataFrame) -> np.ndarray:
    """
    เทรน Prophet ด้วยข้อมูลทั้งชุด แล้วทำนายค่าทุกปีในอดีต
    (ใช้สำหรับสร้าง y_pred_class)
    """
    m = Prophet()
    m.fit(area_df[["ds", "y"]])

    future = area_df[["ds"]]  # ทำนายเฉพาะจุดเวลาเดิม
    forecast = m.predict(future)

    y_pred = forecast["yhat"].values
    return y_pred


def create_pred_labels(y_pred: np.ndarray) -> np.ndarray:
    """
    แปลงค่าทำนายเป็น class:
    1 ถ้าปีนี้ > ปีก่อน, 0 ถ้าไม่เพิ่มขึ้น
    ความยาว = n-1 (ตัดตัวแรกออกเหมือน y_true_class)
    """
    diff_pred = np.diff(y_pred)
    y_pred_class = (diff_pred > 0).astype(int)
    return y_pred_class


# -----------------------------
# 4) คำนวณ Confusion Matrix + Metrics
# -----------------------------
def compute_confusion_metrics(y_true_class: np.ndarray, y_pred_class: np.ndarray):
    cm = confusion_matrix(y_true_class, y_pred_class)
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true_class, y_pred_class)
    prec = precision_score(y_true_class, y_pred_class, zero_division=0)
    rec = recall_score(y_true_class, y_pred_class, zero_division=0)
    f1 = f1_score(y_true_class, y_pred_class, zero_division=0)

    metrics = {
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "accuracy": round(acc, 3),
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "f1": round(f1, 3),
        "cm": cm,
    }
    return metrics


def plot_confusion_matrix(cm, area_name: str):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix – {area_name}\nClass 1 = Poverty Increased")
    plt.tight_layout()
    filename = f"confusion_matrix_{area_name.replace(' ', '_')}.png"
    plt.savefig(filename)
    plt.close()
    print(f" เซฟรูป Confusion Matrix ไว้ที่: {filename}")


# -----------------------------
# 5) main
# -----------------------------
def main():
    # 🔧 เปลี่ยนตรงนี้เป็นพื้นที่ที่ Tee ต้องการวิเคราะห์
    area_name = "San Diego County"

    df = load_data("Book.csv")
    area_df = prepare_area(df, area_name)

    # y_true_class
    labeled_df = create_true_labels(area_df)
    y_true_class = labeled_df["y_true_class"].values

    # y_pred_class จาก Prophet
    y_pred_all = prophet_predict(area_df)
    y_pred_class = create_pred_labels(y_pred_all)

    # ให้ length เท่ากันเผื่อความต่างจากการ diff
    min_len = min(len(y_true_class), len(y_pred_class))
    y_true_class = y_true_class[:min_len]
    y_pred_class = y_pred_class[:min_len]

    metrics = compute_confusion_metrics(y_true_class, y_pred_class)

    print(f"\n Confusion Matrix – {area_name}")
    print("TP:", metrics["TP"])
    print("TN:", metrics["TN"])
    print("FP:", metrics["FP"])
    print("FN:", metrics["FN"])
    print("\n Metrics")
    print("Accuracy :", metrics["accuracy"])
    print("Precision:", metrics["precision"])
    print("Recall   :", metrics["recall"])
    print("F1-score :", metrics["f1"])

    # วาดรูป Confusion Matrix
    plot_confusion_matrix(metrics["cm"], area_name)


if __name__ == "__main__":
    main()