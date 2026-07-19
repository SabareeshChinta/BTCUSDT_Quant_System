import pandas as pd

df = pd.read_parquet("btcusdt_1h.parquet")

df["prev_close"] = df["Close"].shift(1)

df["tr1"] = df["High"] - df["Low"]
df["tr2"] = abs(df["High"] - df["prev_close"])
df["tr3"] = abs(df["Low"] - df["prev_close"])

df["TR"] = df[["tr1", "tr2", "tr3"]].max(axis=1)

df["ATR"] = df["TR"].rolling(14).mean()

print(df[["Close", "ATR"]].tail())
df.to_parquet("btcusdt_atr.parquet", index=False)