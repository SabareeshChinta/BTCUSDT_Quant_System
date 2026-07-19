# download_data.py

from binance.client import Client
import pandas as pd

client = Client()

klines = client.get_historical_klines(
    "BTCUSDT",
    Client.KLINE_INTERVAL_1HOUR,
    "1 Jan, 2019"
)

print(len(klines))
df = pd.DataFrame(
    klines,
    columns=[
        "Open Time",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Close Time",
        "Quote Asset Volume",
        "Number of Trades",
        "Taker Buy Base",
        "Taker Buy Quote",
        "Ignore"
    ]
)

print(df.head())
print(df.shape)
# Convert timestamps
df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms")
df["Close Time"] = pd.to_datetime(df["Close Time"], unit="ms")

# Convert important columns to numeric
numeric_cols = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col])

# Save data
df.to_parquet("btcusdt_1h.parquet", index=False)

print("\nSaved successfully!")
print(df.dtypes)