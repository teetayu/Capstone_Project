# generate_poverty_population_chart.py
import pandas as pd
import matplotlib.pyplot as plt

def load_data(path="Book.csv"):
    return pd.read_csv(path)

def plot_poverty_population(df, area_name):
    sub = df[df["Geography Name"] == area_name].sort_values("Year")

    if "Poverty Estimate" not in sub.columns:
        raise ValueError("Dataset ไม่มีคอลัมน์ Poverty Estimate")

    plt.figure(figsize=(10,5))
    plt.plot(sub["Year"], sub["Poverty Estimate"], marker='o', linewidth=2, color='darkred')

    plt.title(f"Total Population Below Poverty Line – {area_name}")
    plt.xlabel("Year")
    plt.ylabel("Population Count")
    plt.grid(True)
    plt.tight_layout()

    filename = f"BelowPoverty_{area_name.replace(' ','_')}.png"
    plt.savefig(filename)
    plt.close()
    print(f" Saved graph: {filename}")

def main():
    df = load_data()
    area = "San Diego County"  # เปลี่ยนชื่อพื้นที่ที่ต้องการ
    plot_poverty_population(df, area)

if __name__ == "__main__":
    main()
