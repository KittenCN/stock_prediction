# 备份：原始 BERT 数据获取脚本（迁移前实现）。
# 注意：该文件仅用于归档，不随主代码路径更新。

import datasets
from stock_prediction.init import *

my_dataset_all = datasets.load_dataset(path='seamew/ChnSentiCorp', cache_dir=bert_data_path+'/data/')  # 获取整个数据集
my_dataset_train = my_dataset_all['train']
my_dataset_validation = my_dataset_all['validation']
my_dataset_test = my_dataset_all['test']

my_dataset_all_git = datasets.load_from_disk(bert_data_path+'/data/'+'ChnSentiCorp')
print(my_dataset_all_git)
