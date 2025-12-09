import pandas as pd

def load_data(path="Book.csv"):
    return pd.read_csv(path)

def find_first_increase(df):
    results = []

    for area in df["Geography Name"].unique():
        sub = df[df["Geography Name"] == area].sort_values("Year")

        prev = None
        first_increase_year = None
        increase_value = None

        for _, row in sub.iterrows():
            if prev is not None:
                diff = row["Poverty Percent"] - prev
                if diff > 0:
                    first_increase_year = row["Year"]
                    increase_value = diff
                    break
            prev = row["Poverty Percent"]

        results.append({
            "Area": area,
            "First Increase Year": first_increase_year,
            "Increase Amount": increase_value
        })

    return pd.DataFrame(results)

def main():
    df = load_data()
    result = find_first_increase(df)
    print(result)
    result.to_csv("first_poverty_increase_year.csv", index=False)
    print("\n Saved: first_poverty_increase_year.csv")

if __name__ == "__main__":
    main()