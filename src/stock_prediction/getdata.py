import datetime
import argparse
import random

# 处理相对导入问题
try:
    from .init import stock_list_queue, tqdm, time, stock_data_queue, NoneDataFrame, pd, daily_path, threading, TQDM_NCOLS
except ImportError:
    # 如果直接运行此文件，使用绝对导入
    import sys
    from pathlib import Path
    current_dir = Path(__file__).resolve().parent
    root_dir = current_dir.parent.parent
    src_dir = root_dir / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from stock_prediction.init import stock_list_queue, tqdm, time, stock_data_queue, NoneDataFrame, pd, daily_path, threading, TQDM_NCOLS

# 外部依赖需要用户安装
try:
    import tushare as ts
except ImportError:
    ts = None

try:
    import akshare as ak
except ImportError:
    ak = None

try:
    import yfinance as yf
except ImportError:
    yf = None


class DataConfig:
    """数据获取配置"""
    def __init__(self):
        self.api = "akshare"
        self.adjust = "hfq"
        self.code = ""


# 全局配置实例
config = DataConfig()


def set_adjust(adjust):
    """设置复权方式"""
    config.adjust = adjust


def get_stock_list():
    """获取股票列表"""
    if config.api == "tushare":
        if ts is None:
            raise ImportError("tushare not installed")
        
        # Get stock list
        df = ts.pro_api().stock_basic(fields=["ts_code"])
        stock_list = df["ts_code"].tolist()
        
        # Put stock_list into the queue
        stock_list_queue.put(stock_list)
        return stock_list
        
    elif config.api == "akshare":
        if ak is None:
            raise ImportError("akshare not installed")
            
        stock_list = ak.stock_zh_a_spot_em()
        stock_list = stock_list["代码"].tolist()
        stock_list_queue.put(stock_list)
        return stock_list


def get_stock_data(ts_code="", save=True, start_code="", save_path="", datediff=-1):
    """获取股票数据"""
    if config.api == "tushare":
        if ts is None:
            raise ImportError("tushare not installed")
            
        pro = ts.pro_api()
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
            if config.adjust != "":
                adjust = "_" + config.adjust
            for code in stock_list:
                try:
                    if config.adjust != "":
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
                
                time.sleep(random.uniform(0.1, 0.9))  # 避免频率过快

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
                        
    elif config.api == "akshare":
        if ak is None:
            raise ImportError("akshare not installed")
            
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
                    df = ak.stock_zh_a_hist(symbol=code, period="daily", end_date=enddate, adjust=config.adjust)
                    df.columns = [
                            "trade_date",
                            "ts_code",
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
                    columns = list(df.columns)
                    columns[0], columns[1] = columns[1], columns[0]  # 交换第一列和第二列
                    df = df[columns]
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
                    if e.args[0].args[0] == 'Connection aborted.' or e.args[0].args[1].args[0] == 'Remote end closed connection without response':
                        break
                    else:
                        continue

                time.sleep(random.uniform(0.1, 0.9))  # 避免频率过快

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
                        
    elif config.api == "yfinance":
        if yf is None:
            raise ImportError("yfinance not installed")
            
        auto_adjust = False
        back_adjust = False
        if config.adjust == "":
            auto_adjust = False
            back_adjust = False
        elif config.adjust == "qfq":
            auto_adjust = True
            back_adjust = False
        elif config.adjust == "hfq":
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


def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--code', default="", type=str, help="code")
    parser.add_argument('--api', default="akshare", type=str, help="api-interface, tushare, akshare or yfinance")
    parser.add_argument('--adjust', default="hfq", type=str, help="adjust: none or qfq or hfq, Note if you have permission")
    args = parser.parse_args()
    
    config.api = args.api
    config.adjust = args.adjust
    config.code = args.code
    
    yfi_ticker = ['DAX', 'IBM']
    if args.api == "yfinance":
        assert len(yfi_ticker) > 0, "Please input ticker"
        get_stock_data(yfi_ticker, save=True, save_path=daily_path)
    else:
        if args.code != "":
            get_stock_data(args.code, save=True, save_path=daily_path)
        else:
            get_stock_data("", save=True, save_path=daily_path, datediff=-1)


if __name__ == "__main__":
    main()