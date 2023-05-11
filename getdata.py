import datetime
import tushare as ts
import akshare as ak
import yfinance as yf
import argparse
from common import stock_list_queue,tqdm,time,stock_data_queue,NoneDataFrame,pd,daily_path,threading
from init import TQDM_NCOLS

parser = argparse.ArgumentParser()
parser.add_argument('--code', default="", type=str, help="code")
parser.add_argument('--api', default="akshare", type=str, help="api-interface, tushare, akshare or yfinance")
parser.add_argument('--adjust', default="hfq", type=str, help="adjust: none or qfq or hfq, Note if you have permission")
args = parser.parse_args()

if args.api == "tushare":
    api_token = ""
    with open('api.txt', 'r') as file:
        api_token = file.read()
    pro = ts.pro_api(api_token)

def set_adjust(adjust):
    args.adjust=adjust

def get_stock_list():
    if args.api == "tushare":
        # Get stock list
        df = pro.stock_basic(fields=["ts_code"])
        stock_list = df["ts_code"].tolist()
        
        # Put stock_list into the queue
        stock_list_queue.put(stock_list)

        return stock_list
    elif args.api == "akshare":
        stock_list = ak.stock_zh_a_spot_em()
        stock_list = stock_list["代码"].tolist()
        stock_list_queue.put(stock_list)
        return stock_list

def get_stock_data(ts_code="", save=True, start_code="", save_path="", datediff=-1):
    if args.api == "tushare":
        if ts_code == "":
            stock_list = get_stock_list()
        else:
            stock_list = [ts_code]

        if start_code != "":
            stock_list = stock_list[stock_list.index(start_code):]

        if save:
            pbar = tqdm(total=len(stock_list), leave=False, ncols=TQDM_NCOLS)

        lock = threading.Lock()

        with lock:
            adjust = ""
            if args.adjust != "":
                adjust = "_" + args.adjust
            for code in stock_list:
                try:
                    if args.adjust != "":
                        df = pro.stk_factor(ts_code=code, fields=[
                            "ts_code",
                            "trade_date",
                            "open"+adjust,
                            "high"+adjust,
                            "low"+adjust,
                            "close"+adjust,
                            "pre_close"+adjust,
                            "change",
                            "pct_change",
                            "vol",
                            "amount"
                        ])
                        
                        df.columns = [
                            "ts_code",
                            "trade_date",
                            "open",
                            "high",
                            "low",
                            "close",
                            "pre_close",
                            "change",
                            "pct_change",
                            "vol",
                            "amount"
                        ]
                    else:
                        df = pro.daily(ts_code=code, fields=[
                            "ts_code",
                            "trade_date",
                            "open",
                            "high",
                            "low",
                            "close",
                            "pre_close",
                            "change",
                            "pct_chg",
                            "vol",
                            "amount"
                        ])
                    df = df.reindex(columns=[
                            "ts_code",
                            "trade_date",
                            "open",
                            "high",
                            "low",
                            "close",
                            "change",
                            "pct_chg",
                            "vol",
                            "amount",
                            "pre_close"
                        ])
                except Exception as e:
                    if save:
                        tqdm.write(f"{code} {e}")
                        pbar.update(1)
                    else:
                        print(f"{code} {e}")
                    continue
                
                time.sleep(0.1)

                if save:
                    df.to_csv(save_path+f"/{code}.csv", index=False)
                    pbar.update(1)

                else:
                    if not df.empty:
                        stock_data_queue.put(df)
                        return df
                    else:
                        stock_data_queue.put(NoneDataFrame)
                        return None
    elif args.api == "akshare":
        if ts_code == "":
            stock_list = get_stock_list()
        else:
            stock_list = [ts_code]

        if start_code != "":
            stock_list = stock_list[stock_list.index(start_code):]

        if save:
            pbar = tqdm(total=len(stock_list), leave=False, ncols=TQDM_NCOLS)

        lock = threading.Lock()

        with lock:
            now_time = datetime.datetime.now()
            end_time = now_time +  datetime.timedelta(days = datediff)
            enddate = end_time.strftime('%Y%m%d')
            for code in stock_list:
                try:
                    df = ak.stock_zh_a_hist(symbol=code, period="daily", end_date=enddate, adjust=args.adjust)
                    df.insert(0, "ts_code", code)
                    df.columns = [
                            "ts_code",
                            "trade_date",
                            "open",
                            "close",
                            "high",
                            "low",
                            "vol",
                            "amount",
                            "amplitude",
                            "pct_change",
                            "change",
                            "exchange_rate"
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
                            "change",
                            "pct_change",
                            "vol",
                            "amount",
                            "amplitude",
                            "exchange_rate"
                        ])
                except Exception as e:
                    if save:
                        tqdm.write(f"{code} {e}")
                        pbar.update(1)
                    else:
                        print(f"{code} {e}")
                    continue

                time.sleep(0.1)

                if save:
                    df.to_csv(save_path+f"/{code}.csv", index=False)
                    pbar.update(1)

                else:
                    if not df.empty:
                        stock_data_queue.put(df)
                        return df
                    else:
                        stock_data_queue.put(NoneDataFrame)
                        return None
    elif args.api == "yfinance":
        auto_adjust = False
        back_adjust = False
        if args.adjust == "":
            auto_adjust = False
            back_adjust = False
        elif args.adjust == "qfq":
            auto_adjust = True
            back_adjust = False
        elif args.adjust == "hfq":
            auto_adjust = True
            back_adjust = True
        stock_list = ts_code
        if save:
            pbar = tqdm(total=len(stock_list), leave=False, ncols=TQDM_NCOLS)
        lock = threading.Lock()
        with lock:
            for code in stock_list:
                try:
                    df = yf.download(code, auto_adjust=auto_adjust, back_adjust=back_adjust)
                    df.reset_index(inplace=True)
                    df.insert(0, "ts_code", code)
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
                except Exception as e:
                    if save:
                        tqdm.write(f"{code} {e}")
                        pbar.update(1)
                    else:
                        print(f"{code} {e}")
                    continue
                if save:
                    df.to_csv(save_path+f"/{code}.csv", index=False)
                    pbar.update(1)

                else:
                    if not df.empty:
                        stock_data_queue.put(df)
                        return df
                    else:
                        stock_data_queue.put(NoneDataFrame)
                        return None
    if save:
        pbar.close()

if __name__ == "__main__":
    # if os.path.exists(daily_path) == False:
    #     os.mkdir(daily_path)
    yfi_ticker = ['DAX', 'IBM']
    if args.api == "yfinance":
        assert len(yfi_ticker) > 0, "Please input ticker"
        get_stock_data(yfi_ticker, save=True, save_path=daily_path)
    else:
        if args.code != "":
            get_stock_data(args.code, save=True, save_path=daily_path)
        else:
            get_stock_data("", save=True, save_path=daily_path, datediff=-1) 