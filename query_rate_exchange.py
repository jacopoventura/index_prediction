import yfinance as yf
import datetime, warnings
warnings.filterwarnings("ignore")
print(" ")

mydate = datetime.datetime(2022, 4, 11).date()

# Query exchange rate history
df = yf.Ticker("EUR=X")
df = df.history(period="max")
df.drop(columns=["Volume", "Dividends", "Stock Splits", "Open"], inplace=True)
df["Trading date"] = [d.date() for d in df.index.to_list()]
df.set_index("Trading date", inplace=True)

rate = df.loc[mydate]
print(rate)

