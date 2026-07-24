from econdatapy import read
import pandas as pd

x = read.dataset("MARKET_RATES")

usdzar = x["data"]["EXCX135.B.A"]
df = pd.DataFrame(usdzar)

print(df.head())
print(df.tail())
print(df.shape)

df.to_csv("data/usdzar_raw.csv", index=False)