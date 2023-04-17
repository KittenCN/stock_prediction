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