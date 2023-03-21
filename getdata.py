import os
import threading
import time
import tushare as ts
import common
from tqdm import tqdm

pro = ts.pro_api('546dfa0ae0cee993337b8e1d755912059be77d436b49b95e7cf7732d')

def get_stock_list():
    # get stock list
    df = pro.stock_basic(**{
        "ts_code": "",
        "name": "",
        "exchange": "",
        "market": "",
        "is_hs": "",
        "list_status": "",
        "limit": "",
        "offset": ""
    }, fields=[
        "ts_code"
    ])
    stock_list = df["ts_code"].tolist()
    # print(stock_list)
    # 
    common.stock_list_queue.put(stock_list)
    return stock_list

def get_stock_data(ts_code="", save=True, start_code=""):
    if ts_code == "":
        get_stock_list()
        stock_list = common.stock_list_queue.get()
    else:
        stock_list = [ts_code]
    if start_code != "":
        for code in stock_list:
            if code == start_code:
                break
            stock_list.remove(code)
    if save:
        pbar = tqdm(total=len(stock_list), leave=False)
    lock = threading.Lock()
    with lock:
        for code in stock_list:
                df = pro.daily(**{
                "ts_code": code,
                "trade_date": "",
                "start_date": "",
                "end_date": "",
                "offset": "",
                "limit": ""
                }, fields=[
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
                time.sleep(0.1)
                if save:
                    df.to_csv("./stock_daily/"+code+".csv", index=False)
                    pbar.update(1)
        if save: pbar.close()
        if save == False:
            # return df
            if df.empty == False:
                common.stock_data_queue.put(df)
                return df
            else:
                common.stock_data_queue.put(common.NoneDataFrame)
                return None

if __name__ == "__main__":
    if os.path.exists("./stock_daily") == False:
        os.mkdir("./stock_daily")
    get_stock_data("", save=True)