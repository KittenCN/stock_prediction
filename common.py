import math
import os
import queue
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from prefetch_generator import BackgroundGenerator

TRAIN_WEIGHT=0.9
SEQ_LEN=179
LEARNING_RATE=0.001   # 0.00001
WEIGHT_DECAY=0.0001   # 0.05
BATCH_SIZE=512
EPOCH=100
SAVE_NUM_ITER=1000
SAVE_NUM_EPOCH=50
GET_DATA=True
TEST_NUM=25
SAVE_INTERVAL=300
OUTPU_DIMENSION=8
INPUT_DIMENSION=8
TQDM_NCOLS = 100

mean_list=[]
std_list=[]
data_queue=queue.Queue()
stock_data_queue=queue.Queue()
stock_list_queue = queue.Queue()
csv_queue=queue.Queue()

NoneDataFrame = pd.DataFrame(columns=["ts_code"])
NoneDataFrame["ts_code"] = ["None"]

name_list = ["open", "high", "low", "close", "change", "pct_chg", "vol", "amount"]
use_list = [1,1,1,1,0,0,0,0]
OUTPU_DIMENSION = sum(use_list)
assert OUTPU_DIMENSION > 0

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def check_exist(address):
    if os.path.exists(address) == False:
        os.mkdir(address)

check_exist("./stock_handle")
check_exist("./stock_daily")
check_exist("./png")
check_exist("./png/train_loss/")
check_exist("./png/predict/")

train_path="./stock_handle/stock_train.csv"
test_path="./stock_handle/stock_test.csv"

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

#完成数据集类
class Stock_Data(Dataset):
    def __init__(self,train=True,transform=None,dataFrame=None,label_num=1):       
        try:
            if train==True:
                if dataFrame is None:
                    with open(train_path) as f:
                        self.data = np.loadtxt(f,delimiter = ",")
                        #可以注释
                        #addi=np.zeros((self.data.shape[0],1))
                        #self.data=np.concatenate((self.data,addi),axis=1)
                else:
                    self.data=dataFrame.values
                self.data=self.data[:,0:INPUT_DIMENSION]
                for i in range(len(self.data[0])):
                    mean_list.append(np.mean(self.data[:,i]))
                    std_list.append(np.std(self.data[:,i]))
                    self.data[:,i]=(self.data[:,i]-np.mean(self.data[:,i]))/(np.std(self.data[:,i])+1e-8)
                self.value=torch.rand(self.data.shape[0]-SEQ_LEN,SEQ_LEN,self.data.shape[1])
                self.label=torch.rand(self.data.shape[0]-SEQ_LEN,label_num)
                for i in range(self.data.shape[0]-SEQ_LEN):                  
                    self.value[i,:,:]=torch.from_numpy(self.data[i:i+SEQ_LEN,:].reshape(SEQ_LEN,self.data.shape[1]))  
                    # self.label[i,:]=self.data[i+SEQ_LEN,0]
                    _tmp = []
                    for index in range(OUTPU_DIMENSION):  
                        if use_list[index] == 1:
                            _tmp.append(self.data[i+SEQ_LEN,index])
                    self.label[i,:]=torch.Tensor(_tmp)
                self.data=self.value
            else:
                if dataFrame is None:
                    with open(test_path) as f:
                        self.data = np.loadtxt(f,delimiter = ",")
                        #可以注释
                        #addi=np.zeros((self.data.shape[0],1))
                        #self.data=np.concatenate((self.data,addi),axis=1)
                else:
                    self.data=dataFrame.values
                self.data=self.data[:,0:INPUT_DIMENSION]
                for i in range(len(self.data[0])):
                    self.data[:,i]=(self.data[:,i]-mean_list[i])/(std_list[i]+1e-8)
                self.value=torch.rand(self.data.shape[0]-SEQ_LEN,SEQ_LEN,self.data.shape[1])
                self.label=torch.rand(self.data.shape[0]-SEQ_LEN,label_num)
                for i in range(self.data.shape[0]-SEQ_LEN):                  
                    self.value[i,:,:]=torch.from_numpy(self.data[i:i+SEQ_LEN,:].reshape(SEQ_LEN,self.data.shape[1]))    
                    # self.label[i,:]=self.data[i+SEQ_LEN,0]
                    _tmp = []
                    for index in range(OUTPU_DIMENSION):  
                        if use_list[index] == 1:
                            _tmp.append(self.data[i+SEQ_LEN,index])
                    self.label[i,:]=torch.Tensor(_tmp)
                self.data=self.value
        except Exception as e:
            print(e)
            return None
    def __getitem__(self,index):
        return self.data[index],self.label[index]
    def __len__(self):
        return len(self.data[:,0])
#LSTM模型
class LSTM(nn.Module):
    def __init__(self,dimension):
        super(LSTM,self).__init__()
        self.lstm=nn.LSTM(input_size=dimension,hidden_size=128,num_layers=3,batch_first=True)
        self.linear1=nn.Linear(in_features=128,out_features=16)
        self.linear2=nn.Linear(16,OUTPU_DIMENSION)
        self.LeakyReLU=nn.LeakyReLU()
        # self.ELU = nn.ELU()
        # self.ReLU = nn.ReLU()
    def forward(self,x):
        # out,_=self.lstm(x)
        lengths = [s.size(0) for s in x] # 获取数据真实的长度
        x_packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out_packed, _ = self.lstm(x_packed)
        out, lengths = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        x=out[:,-1,:]        
        x=self.linear1(x)
        x=self.LeakyReLU(x)
        # x=self.ELU(x)
        x=self.linear2(x)
        return x
#传入tensor进行位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=SEQ_LEN):
        super(PositionalEncoding, self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.div_term = nn.Parameter(torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)), requires_grad=False)
        self.pe = nn.Parameter(torch.zeros(max_len, d_model), requires_grad=False)
        self._init_pe()

    def _init_pe(self):
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        self.pe[:, 0::2] = torch.sin(position * self.div_term)
        self.pe[:, 1::2] = torch.cos(position * self.div_term)

    def forward(self, x):
        pe = self.pe[:x.size(1), :]
        pe = pe.unsqueeze(0).expand(x.size(0), -1, -1)
        pe = pe.to(x.device).float()
        return x + pe

class TransAm(nn.Module):
    def __init__(self, feature_size: int = 8, num_layers: int = 6, dropout: float = 0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=8, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, 1)
        self.linear1 = nn.Linear(SEQ_LEN, OUTPU_DIMENSION)
        self.init_weights()
        self.src_key_padding_mask = None

    def init_weights(self):
        initrange = 0.1
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src: torch.Tensor, seq_len: int = SEQ_LEN) -> torch.Tensor:
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        output = torch.squeeze(output)
        output = self.linear1(output)
        return output