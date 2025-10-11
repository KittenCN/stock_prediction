import pandas as pd
import json
import os

# 处理相对导入问题
try:
    from .init import data_path, daily_path
except ImportError:
    # 如果直接运行此文件，使用绝对导入
    import sys
    from pathlib import Path
    current_dir = Path(__file__).resolve().parent
    root_dir = current_dir.parent.parent
    src_dir = root_dir / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from stock_prediction.init import data_path, daily_path


def write_page(stock_id, logstr, log_path=None):
    """写入页面日志"""
    if log_path is None:
        log_path = f"{data_path}/log"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    with open(f"{log_path}/{stock_id}", 'w', encoding='utf-8') as f:
        f.write(logstr)


def read_page(stock_id, log_path=None):
    """读取页面日志"""
    if log_path is None:
        log_path = f"{data_path}/log"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    file_path = f"{log_path}/{stock_id}"
    if os.path.isfile(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            line_list = f.readlines()
            start_page = line_list[0].rstrip('\n')
            return int(start_page)
    else:
        return 0


def write_log(stock_id, logstr, log_path=None):
    """写入日志"""
    if log_path is None:
        log_path = f"{data_path}/log"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    with open(f"{log_path}/{stock_id}", 'a', encoding='utf-8') as f:
        f.write(logstr + '\n')


def read_log(stock_id, log_path=None):
    """读取日志"""
    if log_path is None:
        log_path = f"{data_path}/log"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    
    file_path = f"{log_path}/{stock_id}"
    if os.path.isfile(file_path):
        comment_urls = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            line_list = f.readlines()
            for i in range(0, len(line_list)):
                record = json.loads(line_list[i].rstrip('\n') + "")
                comment_urls[record['comment_url']] = record
        return comment_urls
    else:
        return {}


def write_url(stock_id, logstr, log_path=None):
    """写入URL日志"""
    if log_path is None:
        log_path = f"{data_path}/log"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    with open(f"{log_path}/{stock_id}", 'a', encoding='utf-8') as f:
        f.write(logstr + '\n')


def read_url(stock_id, log_path=None):
    """读取URL日志"""
    if log_path is None:
        log_path = f"{data_path}/log"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    
    file_path = f"{log_path}/{stock_id}"
    if os.path.isfile(file_path):
        comment_urls = []
        with open(file_path, 'r', encoding='utf-8') as f:
            line_list = f.readlines()
            for i in range(0, len(line_list)):
                comment_urls.append(line_list[i].rstrip('\n'))
        return set(comment_urls)
    else:
        return set()


def json2csv(path, save_path):
    """将JSON文件夹转换为CSV"""
    df = pd.DataFrame()
    for file in os.listdir(path):  
        file_path = os.path.join(path, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            item = json.load(f)
            row = pd.DataFrame(item, index=[0])
            df = pd.concat([df, row], ignore_index=True)
    df = df.set_index('time')
    df.drop(['read', 'subcomments', 'comment_url', 'comment_id'], inplace=True, axis=1)
    df.to_csv(save_path)


def cal_compounding_factor(ts_code=""):
    """计算复权因子"""
    import random
    import numpy as np
    import glob
    try:
        from .getdata import set_adjust, get_stock_data
    except ImportError:
        from stock_prediction.getdata import set_adjust, get_stock_data
    
    # First, find the day on which the reweighting event occurred. Usually, the difference between the post-weighting data and the non-weighting data is greatest on this day.
    # Calculate the ratio of the post-weighted data of the two adjacent days:
    # Ratio of post-weighted data = Post-weighted data of the second day / Post-weighted data of the first day
    # Calculate the ratio of non-rev weighted data for two adjacent days:
    # Non-rev weighted data ratio = Non-rev weighted data of the second day / Non-rev weighted data of the first day
    # Divide the proportion of post-weighted data by the proportion of non-weighted data to obtain the compounding factor:
    # Compounding factor = Proportion of post-weighted data / Proportion of non-weighted data
    if ts_code == "":
        csv_files = glob.glob(daily_path + "/*.csv")
        ts_codes = []
        for csv_file in csv_files:
            ts_codes.append(os.path.basename(csv_file).rsplit(".", 1)[0])
        ts_code = random.sample(ts_codes, 1)[0]
    
    data_file_path = f"{data_path}/{ts_code}.csv"
    daily_file_path = f"{daily_path}/{ts_code}.csv"
    set_adjust("")
    get_stock_data(ts_code, save=True, save_path=data_path) 
    
    if os.path.exists(daily_file_path) and os.path.exists(data_file_path):
        df = pd.read_csv(daily_file_path)
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        df = df.sort_values(by='trade_date', ascending=False)
        daily_open = df['open'].iloc[0] 
        daily_close = df['close'].iloc[0]
        daily_high = df['high'].iloc[0]
        daily_low = df['low'].iloc[0]
        _date = df['trade_date'].iloc[0]
        
        df = pd.read_csv(data_file_path)
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        df = df.sort_values(by='trade_date', ascending=False)
        data_open = None
        data_close = None
        data_high = None
        data_low = None
        
        for i in range(len(df)):
            if df['trade_date'].iloc[i] == _date:
                data_open = df['open'].iloc[i]
                data_close = df['close'].iloc[i]
                data_high = df['high'].iloc[i]
                data_low = df['low'].iloc[i]
                break
                
        if data_open is None or data_close is None or data_high is None or data_low is None:
            return None
            
        factor = [data_open/daily_open, data_close/daily_close, data_high/daily_high, data_low/daily_low]
        compounding_factor = np.mean(factor)
        return compounding_factor
    else:
        return None