# -*- coding: utf-8 -*-
"""
Find Unique Areas in the Dataset
อ่านไฟล์ Book.csv แล้วดึงรายชื่อพื้นที่ทั้งหมดแบบไม่ซ้ำกัน
"""

import pandas as pd

def load_data(path="Book.csv"):
    """
    โหลดข้อมูล CSV
    """
    df = pd.read_csv(path)
    print(" Loaded dataset:", df.shape)
    return df


def find_unique_areas(df):
    """
    คืนค่า list ของพื้นที่ที่ไม่ซ้ำกัน
    """
    areas = df["Geography Name"].unique().tolist()
    print("\n Unique Areas Found:")
    for a in areas:
        print("-", a)

    # เซฟลงไฟล์ (optional แต่มีประโยชน์ตอนเลือกในโมเดล)
    with open("unique_areas.txt", "w", encoding="utf-8") as f:
        for a in areas:
            f.write(a + "\n")

    print("\n Saved to: unique_areas.txt")

    return areas


def main():
    df = load_data("Book.csv")
    find_unique_areas(df)


if __name__ == "__main__":
    main()