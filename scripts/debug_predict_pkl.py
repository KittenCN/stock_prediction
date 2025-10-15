
import sys, os
sys.path.insert(0, os.path.abspath('src'))
import dill
import pandas as pd
from stock_prediction.config import Config

config = Config()
train_pkl_path = config.train_pkl_path

test_code = '000615'
with open(train_pkl_path, 'rb') as f:
    data_queue = dill.load(f)

found = False
while not data_queue.empty():
    item = data_queue.get()
    ts_code = str(item['ts_code'].iloc[0]).zfill(6)
    if ts_code == test_code:
        print(f"Found ts_code: {ts_code}")
        print(f"item.shape: {item.shape}")
        print(f"item.columns: {list(item.columns)}")
        print(item.head())
        found = True
        break
if not found:
    print(f"ts_code {test_code} not found in pkl!")
