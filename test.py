import pandas as pd
import yfinance as yf
# yf.pdr_override() # <== that's all it takes :-)

# download dataframe using pandas_datareader
yfi_ticker = ['DAX', 'IBM']
for item in yfi_ticker:
    df = yf.download(item, auto_adjust=True, back_adjust=False)
    df.reset_index(inplace=True)
    df.insert(0, "ts_code", item)
    df.columns = [
                    "ts_code",
                    "trade_date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "vol"
                ]
    df["trade_date"] = pd.to_datetime(df['trade_date']).dt.strftime("%Y%m%d")
    df.sort_values(by=['trade_date'], ascending=False, inplace=True)
    df = df.reindex(columns=[
                        "ts_code",
                        "trade_date",
                        "open",
                        "high",
                        "low",
                        "close",
                        "vol"
                    ])
    print(df)