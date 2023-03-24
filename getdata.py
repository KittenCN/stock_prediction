import os
import threading
import time
import numpy as np
import tushare as ts
import common
import target
from tqdm import tqdm

api_token = ""
with open('api.txt', 'r') as file:
    api_token = file.read()
    print(api_token)
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
            
            if 'trade_date' in df.columns:
                # times = [datetime.datetime.fromtimestamp(int(str(ts.value)[:10])).strftime('%Y%m%d') for ts in df['trade_date'].tolist()]
                times = np.array(df['trade_date'].values)
                close = np.array(df['close'].values)
                hpri = np.array(df['high'].values)
                lpri = np.array(df['low'].values)
                vol = np.array(df['vol'].values)
            elif 'Date' in df.columns:
                times = np.array(df['Date'].values)
                close = np.array(df['Close'].values)
                hpri = np.array(df['High'].values)
                lpri = np.array(df['Low'].values)
                vol = np.array(df['Volume'].values)
            
            times = times[::-1]
            close = close[::-1]
            hpri = hpri[::-1]
            lpri = lpri[::-1]
            vol = vol[::-1]

            macd_dif, macd_dea, macd_bar = target.MACD(close)
            df["macd_dif"] = common.cmp_append(macd_dif[::-1], df)
            df["macd_dea"] = common.cmp_append(macd_dea[::-1], df)
            df["macd_bar"] = common.cmp_append(macd_bar[::-1], df)
            k, d, j = target.KDJ(close, hpri, lpri)
            df['k'] = common.cmp_append(k[::-1], df)
            df['d'] = common.cmp_append(d[::-1], df)
            df['j'] = common.cmp_append(j[::-1], df)
            boll_upper, boll_mid, boll_lower = target.BOLL(close)
            df['boll_upper'] = common.cmp_append(boll_upper[::-1], df)
            df['boll_mid'] = common.cmp_append(boll_mid[::-1], df)
            df['boll_lower'] = common.cmp_append(boll_lower[::-1], df)
            cci = target.CCI(close, hpri, lpri)
            df['cci'] = common.cmp_append(cci[::-1], df)
            pdi, mdi, adx, adxr = target.DMI(close, hpri, lpri)
            df['pdi'] = common.cmp_append(pdi[::-1], df)
            df['mdi'] = common.cmp_append(mdi[::-1], df)
            df['adx'] = common.cmp_append(adx[::-1], df)
            df['adxr'] = common.cmp_append(adxr[::-1], df)
            taq_up, taq_mid, taq_down = target.TAQ(hpri, lpri, 5)
            df['taq_up'] = common.cmp_append(taq_up[::-1], df)
            df['taq_mid'] = common.cmp_append(taq_mid[::-1], df)
            df['taq_down'] = common.cmp_append(taq_down[::-1], df)
            trix, trma = target.TRIX(close)
            df['trix'] = common.cmp_append(trix[::-1], df)
            df['trma'] = common.cmp_append(trma[::-1], df)
            mfi = target.MFI(close, hpri, lpri, vol)
            df['mfi'] = common.cmp_append(mfi[::-1], df)
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
                "macd_dif",
                "macd_dea",
                "macd_bar",
                "k",
                "d",
                "j",
                "boll_upper",
                "boll_mid",
                "boll_lower",
                "cci",
                "pdi",
                "mdi",
                "adx",
                "adxr",
                "taq_up",
                "taq_mid",
                "taq_down",
                "trix",
                "trma",
                "mfi",
                "pre_close"
            ])

            time.sleep(0.1)

            if save:
                df.to_csv(f"./stock_daily/{code}.csv", index=False)
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
    if os.path.exists("./stock_daily") == False:
        os.mkdir("./stock_daily")
    get_stock_data("", save=True)