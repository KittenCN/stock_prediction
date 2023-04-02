import multiprocessing
import os
import queue
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader

TRAIN_WEIGHT=0.9
SEQ_LEN=180
LEARNING_RATE=0.00001   # 0.00001
WEIGHT_DECAY=0.05   # 0.05
BATCH_SIZE=16
EPOCH=1
SAVE_NUM_ITER=10
SAVE_NUM_EPOCH=1
# GET_DATA=True
TEST_NUM=25
SAVE_INTERVAL=300
OUTPUT_DIMENSION=8
INPUT_DIMENSION=20
TQDM_NCOLS = 100
NUM_WORKERS = 1
PKL = True

mean_list=[]
std_list=[]
data_queue=multiprocessing.Queue()
stock_data_queue=queue.Queue()
stock_list_queue = queue.Queue()
csv_queue=queue.Queue()
df_queue=queue.Queue()

NoneDataFrame = pd.DataFrame(columns=["ts_code"])
NoneDataFrame["ts_code"] = ["None"]

name_list = ["open", "high", "low", "close", "change", "pct_chg", "vol", "amount"]
use_list = [1,1,1,1,1,1,1,1]
OUTPUT_DIMENSION = sum(use_list)

assert OUTPUT_DIMENSION > 0

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

def check_exist(address):
    if os.path.exists(address) == False:
        os.mkdir(address)

check_exist("./stock_handle")
check_exist("./stock_daily")
check_exist("./pkl_handle")
check_exist("./png")
check_exist("./png/train_loss/")
check_exist("./png/predict/")
check_exist("./png/test/")

train_path="./stock_handle/stock_train.csv"
test_path="./stock_handle/stock_test.csv"
train_pkl_path="./pkl_handle/train.pkl"