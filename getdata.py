import threading
import tushare as ts
import common
from init import *
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--code', default="", type=str, help="code")
args = parser.parse_args()

api_token = ""
with open('api.txt', 'r') as file:
    api_token = file.read()
pro = ts.pro_api(api_token)

def get_stock_list():
    # Get stock list
    df = pro.stock_basic(fields=["ts_code"])
    stock_list = df["ts_code"].tolist()
    
    # Put stock_list into the queue
    common.stock_list_queue.put(stock_list)

    return stock_list

def get_stock_data(ts_code="", save=True, start_code=""):
    if ts_code == "":
        stock_list = get_stock_list()
    else:
        stock_list = [ts_code]

    if start_code != "":
        stock_list = stock_list[stock_list.index(start_code):]

    if save:
        pbar = tqdm(total=len(stock_list), leave=False)

    lock = threading.Lock()

    with lock:
        for code in stock_list:
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

            time.sleep(0.1)

            if save:
                df.to_csv(daily_path+f"/{code}.csv", index=False)
                pbar.update(1)

            else:
                if not df.empty:
                    common.stock_data_queue.put(df)
                    return df
                else:
                    common.stock_data_queue.put(common.NoneDataFrame)
                    return None

    if save:
        pbar.close()

if __name__ == "__main__":
    # if os.path.exists(daily_path) == False:
    #     os.mkdir(daily_path)
    if args.code != "":
        get_stock_data(args.code, save=True)
    else:
        get_stock_data("", save=True) 