import pandas as pd
import json
import os

root_path="."
stock_data_path=root_path+"/stock_data"

def write_page(stock_id, logstr):
	if not os.path.exists(stock_data_path+'/log'):
		os.makedirs(stock_data_path+'/log')
	with open(stock_data_path+'/log/'+stock_id, 'w',encoding='utf-8') as f:
		f.write(logstr)

def read_page(stock_id):
	if not os.path.exists(stock_data_path+'/log'):
		os.makedirs(stock_data_path+'/log')
	if os.path.isfile(stock_data_path+'/log/'+stock_id):
		with open(stock_data_path+'/log/'+stock_id, 'r',encoding='utf-8') as f:
			line_list = f.readlines()
			start_page = line_list[0].rstrip('\n')
			return int(start_page)
	else:
		return 0

def write_log(stock_id, logstr):
    result_file_open = open(stock_data_path+'/log/'+stock_id, 'a', encoding='utf-8')
    result_file_open.write(logstr+'\n')
    result_file_open.close()

def read_log(stock_id):
	if not os.path.exists(stock_data_path+'/log'):
		os.makedirs(stock_data_path+'/log')
	
	if os.path.isfile(stock_data_path+'/log/'+stock_id):
		comment_urls = {}
		with open(stock_data_path+'/log/'+stock_id, 'r',encoding='utf-8') as f:
			line_list = f.readlines()
			for i in range(0, len(line_list)):
				record = json.loads(line_list[i].rstrip('\n')+"")
				comment_urls[record['comment_url']] = record
		return comment_urls
	else:
		return {}

def write_url(stock_id, logstr):
    result_file_open = open(stock_data_path+'/log/'+stock_id, 'a', encoding='utf-8')
    result_file_open.write(logstr+'\n')
    result_file_open.close()

def read_url(stock_id):
	if not os.path.exists(stock_data_path+'/log'):
		os.makedirs(stock_data_path+'/log')
	
	if os.path.isfile(stock_data_path+'/log/'+stock_id):
		comment_urls = []
		with open(stock_data_path+'/log/'+stock_id, 'r',encoding='utf-8') as f:
			line_list = f.readlines()
			for i in range(0, len(line_list)):
				comment_urls.append(line_list[i].rstrip('\n'))
		return set(comment_urls)
	else:
		return set()

def json2csv(path, save_path):
    df = pd.DataFrame()
    for file in os.listdir(path):  
        file_path = os.path.join(path, file)
        with open(file_path,'r',encoding='utf-8') as f:
            item = json.load(f)
            row = pd.DataFrame(item,index=[0])
            df = df.append(row,ignore_index=True)
    df = df.set_index('time')
    df.drop(['read','subcomments','comment_url','comment_id'],inplace=True,axis=1)
    df.to_csv(save_path)

# df = json2csv('comment\\000983','00983.csv')

def cal_compounding_factor(ts_code=""):
    import random
    from getdata import set_adjust, get_stock_data
    from init import glob, daily_path, data_path, np
    # First, find the day on which the reweighting event occurred. Usually, the difference between the post-weighting data and the non-weighting data is greatest on this day.
    # Calculate the ratio of the post-weighted data of the two adjacent days:
    # Ratio of post-weighted data = Post-weighted data of the second day / Post-weighted data of the first day
    # Calculate the ratio of non-rev weighted data for two adjacent days:
    # Non-rev weighted data ratio = Non-rev weighted data of the second day / Non-rev weighted data of the first day
    # Divide the proportion of post-weighted data by the proportion of non-weighted data to obtain the compounding factor:
    # Compounding factor = Proportion of post-weighted data / Proportion of non-weighted data
    if ts_code == "":
        csv_files = glob.glob(daily_path+"/*.csv")
        ts_codes =[]
        for csv_file in csv_files:
            ts_codes.append(os.path.basename(csv_file).rsplit(".", 1)[0])
        ts_code = random.sample(ts_codes, 1)
    data_file_path = f"{data_path}/{ts_code}.csv"
    daily_file_path = f"{daily_path}/{ts_code}.csv"
    set_adjust("")
    get_stock_data(ts_code, save=True, save_path=data_path) 
    if os.path.exists(daily_file_path) and os.path.exists(data_file_path):
        df = pd.read_csv(daily_file_path)
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        df = df.sort_values(by='trade_date', ascending=False)
        daily_open = df['open'][0]
        daily_close = df['close'][0]
        daily_high = df['high'][0]
        daily_low = df['low'][0]
        _date = df['trade_date'][0]
        df = pd.read_csv(data_file_path)
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        df = df.sort_values(by='trade_date', ascending=False)
        data_open = None
        data_close = None
        data_high = None
        data_low = None
        for i in range(len(df)):
            if df['trade_date'][i] == _date:
                data_open = df['open'][i]
                data_close = df['close'][i]
                data_high = df['high'][i]
                data_low = df['low'][i]
                break
        if data_open is None or data_close is None or data_high is None or data_low is None:
            return None
        factor = [data_open/daily_open, data_close/daily_close, data_high/daily_high, data_low/daily_low]
        compounding_factor = np.mean(factor)
        return compounding_factor
    else:
        return None