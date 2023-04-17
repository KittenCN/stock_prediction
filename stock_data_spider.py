from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from time import sleep
import time
import json
import pandas as pd
# import schedule

import os
from tqdm import tqdm
import threading

from utils import write_log, read_log, write_page, read_page, write_url, read_url, stock_data_path

def fake_refresh(driver):
	driver.execute_script(
	'''
	(function () {
		var y = 0;
		var step = 100;
		window.scroll(0, 0);

		function f() {
		if (y < (document.body.scrollHeight)/5) {
			y += step;
			window.scroll(0, y);
			setTimeout(f, 100);
		} else {
			window.scroll(0, 0);   //滑动到顶部
			document.title += 'scroll-done';
		}
		}
		setTimeout(f, 1000);
	})();
	'''
	)
	sleep(5)

class SimpleThread(threading.Thread):

    def __init__(self,func,args={}):
        super(SimpleThread,self).__init__()
        self.func = func # 执行函数
        self.args = args # 执行参数，其中包含切分后的数据块，字典类型

    def run(self):
        self.result = self.func(**self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None

def get_driver():
	# PROXY='182.84.144.68:3256'
	# webdriver.DesiredCapabilities.CHROME['proxy'] = {
    # 'httpProxy': PROXY,
    # 'ftpProxy': PROXY,
    # 'sslProxy': PROXY,
    # 'proxyType': 'MANUAL',

	# }

	webdriver.DesiredCapabilities.CHROME['acceptSslCerts']=True
	chrome_options = Options()
	chrome_options.add_argument('--no-sandbox')
	chrome_options.add_argument('--headless')
	chrome_options.add_argument('--disable-dev-shm-usage')
	chrome_options.add_argument('--disable-gpu')
	# chrome_options.add_argument('window-size=1024,768')
	# chrome_options.add_argument('--ignore-certificate-errors')
	# chrome_options.add_argument('--ignore-ssl-errors')
	# chrome_options.add_argument('--proxy-server={}'.format('27.40.124.240:9999'))
	

	chrome_driver = 'D:\software\chromedriver.exe'  # Win chromedriver的文件位置
	# chrome_driver = 'config/chromedriver-mac' # MAC chromedriver的文件位置
	driver = webdriver.Chrome(chrome_options=chrome_options, executable_path = chrome_driver)
	driver.delete_all_cookies()
	return driver

def text_format(text):
	text = text.replace(' ','')
	text = text.replace('\n','')
	text = text.replace('\r','')
	text = text.replace('查看PDF原文','')
	text = text.replace('查看原文','')
	text = text.replace('[点击]','')
	text = text.replace('提示：本网不保证其真实性和客观性，一切有关该股的有效信息，以交易所的公告为准，敬请投资者注意风险。','')
	return text

def get_base_info(driver, stock_id, record):
	if record['comment_url'].startswith('/news,{}'.format(stock_id)):
		comment_url = 'http://guba.eastmoney.com'+record['comment_url']
		driver.get(comment_url)
		soup = BeautifulSoup(driver.page_source, 'html.parser')
		if soup.select('#zwconttbt'):
			record['title'] = text_format(soup.select('#zwconttbt')[0].get_text())
			time_str = soup.select('#zwconttb > div.zwfbtime')[0].get_text().split(' ')
			record['time'] = time_str[1]+' '+time_str[2]
			record['content'] = text_format(soup.select('#zwconbody > div')[0].get_text())
			return record
		else:
			return {}
	else:
		return {}

def get_url_item(driver, stock_id, record, all_ready_comment_urls):
	save_path = os.path.join('comment',stock_id)
	if record['comment_url'] in all_ready_comment_urls:
		return
	else:
		sleep(0.5)
		record = get_base_info(driver, stock_id, record)
		if record:
			record['comment_id'] = record['comment_url'].split(',')[-1].split('.')[0]
			json2file(save_path, record)
			write_url(stock_id+'-ok', record['comment_url'])
			all_ready_comment_urls.add(record['comment_url'])


def get_all_urls(stock_id, page_id, all_pre_comment_records):
	driver = get_driver()
	driver.maximize_window()
	sleep(1)
	base_url = 'http://guba.eastmoney.com/list,{}_{}.html'.format(stock_id,page_id)
	driver.get(base_url)
	# driver.maximize_window()
	soup = BeautifulSoup(driver.page_source, 'html.parser')
	items = soup.select('#articlelistnew > div')
	records = []
	flag = False
	for item in items:
		if item.get('class')[0] == 'articleh':
			record = {}
			record['read'] = item.select('span.l1.a1')[0].get_text()
			record['subcomments'] = item.select('span.l2.a2')[0].get_text()
			record['comment_url'] = item.select('span.l3.a3 > a')[0].get('href') #.split(',')[-1].split('.')[0]
			flag = flag | record['comment_url'].startswith('/news,{}'.format(stock_id))
			if record['comment_url'] in all_pre_comment_records:
				print(record['comment_url'] + ' Pass!')
				continue

			print(record)
			record_str = '{{"read": "{}", "subcomments": "{}", "comment_url": "{}"}}'.format(record['read'],record['subcomments'],record['comment_url'])
			write_log(stock_id+'-pre', record_str)
			records.append(record)
	driver.quit()
	return records, flag


def get_data(stock_id):
	if not os.path.exists(stock_data_path+'/comment/'+stock_id):
		os.makedirs(stock_data_path+'/comment/'+stock_id)
	driver = get_driver()

	driver.get('http://guba.eastmoney.com/list,{}.html'.format(stock_id))
	driver.maximize_window()
	sleep(1)

	soup = BeautifulSoup(driver.page_source, 'html.parser')
	page_num = soup.select('#articlelistnew > div.pager > span > span > span:nth-child(1) > span')[0].get_text()
	# page_num = 37
	driver.quit()

	records = []

	all_pre_comment_records=read_log(stock_id+'-pre')

	total_num = 0
	# page_num = 2
	start_page = read_page(stock_id+'-page')
	print('start_page:', start_page,' total_page:',page_num)
	for i in range(start_page, int(page_num)):
		page_id = i+1
		print(page_id, str(len(records)+len(all_pre_comment_records)))
		
		rs, flag = get_all_urls(stock_id, page_id, all_pre_comment_records)
		if flag:
			records.extend(rs)
			write_page(stock_id+'-page', str(page_id-1))
		else:
			return 

	print('编号：{}，共计{}页, {}条记录。'.format(stock_id, page_num, len(records)+len(all_pre_comment_records)))

	records.extend(all_pre_comment_records.values())

	all_ready_comment_urls=read_url(stock_id+'-ok')
	driver = get_driver()
	driver.maximize_window()
	sleep(1)

	for record in records:
		get_url_item(driver, stock_id, record, all_ready_comment_urls)
		

def json2file(save_path, record):
	with open(stock_data_path+'/'+save_path+record['comment_id']+'.json','w',encoding='utf-8') as f:
		json.dump(record,f,ensure_ascii=False, indent=4)

def main():
	if not os.path.exists(stock_data_path+'/comment/'):
		os.makedirs(stock_data_path+'/comment/')
	get_data('6005811')

if __name__ == '__main__':
	main()