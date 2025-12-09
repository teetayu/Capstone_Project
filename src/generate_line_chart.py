# generate_line_chart.py
import pandas as pd
import matplotlib.pyplot as plt

def load_data(path="Book.csv"):
    return pd.read_csv(path)

def plot_line(df, area_name):
    sub = df[df["Geography Name"] == area_name].sort_values("Year")

    plt.figure(figsize=(10,5))
    plt.plot(sub["Year"], sub["Poverty Percent"], marker='o', linewidth=2)

    plt.title(f"Poverty Yearly Trend – {area_name}")
    plt.xlabel("Year")
    plt.ylabel("Poverty Percent")
    plt.grid(True)
    plt.tight_layout()

    filename = f"LineTrend_{area_name.replace(' ','_')}.png"
    plt.savefig(filename)
    plt.close()
    print(f" Saved graph: {filename}")

def main():
    df = load_data()
    area = "San Diego County"  # <– เปลี่ยนตรงนี้
    plot_line(df, area)

if __name__ == "__main__":
    main()