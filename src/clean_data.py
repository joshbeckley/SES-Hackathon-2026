import pandas as pd

# Load the raw series pulled by fetch_data.py
df = pd.read_csv("data/usdzar_raw.csv", parse_dates=["TIME_PERIOD"])
df = df.rename(columns={"TIME_PERIOD": "date", "OBS_VALUE": "usdzar"})
df = df.sort_values("date").reset_index(drop=True)

# Basic sanity checks
print("Date range:", df["date"].min(), "to", df["date"].max())
print("Missing values:", df["usdzar"].isna().sum())
print("Duplicate dates:", df["date"].duplicated().sum())

# Save the full cleaned daily series (your master file for training)
df.to_csv("data/usdzar_clean.csv", index=False)

# Build the Friday evaluation grid for 2021-01-01 to 2025-12-31
fridays = pd.date_range("2021-01-01", "2025-12-31", freq="W-FRI")

# For each Friday, find the actual observed rate on/before that date
# (handles public holidays where the exact Friday has no observation)
df_indexed = df.set_index("date")["usdzar"]
grid = []
for f in fridays:
    available = df_indexed[df_indexed.index <= f]
    if not available.empty:
        actual_date = available.index[-1]
        grid.append({"target_friday": f, "actual_date": actual_date, "usdzar": available.iloc[-1]})

friday_grid = pd.DataFrame(grid)
print("\nFriday grid shape:", friday_grid.shape)
print(friday_grid.head())
print(friday_grid.tail())

friday_grid.to_csv("data/friday_grid.csv", index=False)