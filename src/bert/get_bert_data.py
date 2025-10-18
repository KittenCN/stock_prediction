import os
import sys
from pathlib import Path

# 运行时确保可以导入 stock_prediction 包
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import datasets
from stock_prediction.init import *

my_dataset_all = datasets.load_dataset(path='seamew/ChnSentiCorp', cache_dir=bert_data_path + '/data/')
my_dataset_train = my_dataset_all['train']
my_dataset_validation = my_dataset_all['validation']
my_dataset_test = my_dataset_all['test']

my_dataset_all_git = datasets.load_from_disk(bert_data_path + '/data/' + 'ChnSentiCorp')
print(my_dataset_all_git)
