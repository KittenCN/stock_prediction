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
BATCH_SIZE=2048
EPOCH=100
SAVE_NUM_ITER=1000
SAVE_NUM_EPOCH=50
GET_DATA=True
TEST_NUM=25
SAVE_INTERVAL=300
OUTPU_DIMENSION=8
INPUT_DIMENSION=8

mean_list=[]
std_list=[]
data_queue=queue.Queue()
stock_data_queue=queue.Queue()
stock_list_queue = queue.Queue()
csv_queue=queue.Queue()

NoneDataFrame = pd.DataFrame(columns=["ts_code"])
NoneDataFrame["ts_code"] = ["None"]

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
        self.ReLU=nn.ReLU()
    def forward(self,x):
        out,_=self.lstm(x)
        x=out[:,-1,:]        
        x=self.linear1(x)
        x=self.ReLU(x)
        x=self.linear2(x)
        return x
#传入tensor进行位置编码
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len=SEQ_LEN):
        super(PositionalEncoding,self).__init__()
        #序列长度，dimension d_model
        pe=torch.zeros(max_len,d_model)
        position=torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)
        div_term=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)
        pe=pe.unsqueeze(0).transpose(0,1)
        self.register_buffer('pe',pe)
        
    def forward(self,x):
        return x+self.pe[:x.size(0),:]

class TransAm(nn.Module):
    def __init__(self,feature_size=INPUT_DIMENSION,num_layers=6,dropout=0.1,nhead=8,d_model=512):
        super(TransAm,self).__init__()
        self.model_type='Transformer'
        self.src_mask=None
        self.embedding=nn.Linear(feature_size,d_model)
        self.pos_encoder=PositionalEncoding(d_model)
        self.encoder_layer=nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,dropout=dropout)
        self.transformer_encoder=nn.TransformerEncoder(self.encoder_layer,num_layers=num_layers)
        #全连接层代替decoder
        self.decoder=nn.Linear(d_model,1)
        self.linear1=nn.Linear(SEQ_LEN,1)
        self.init_weights()
        self.src_key_padding_mask=None
    
    def init_weights(self):
        initrange=0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange,initrange)
        
    def forward(self,src,seq_len=SEQ_LEN):       
        src=self.pos_encoder(src)
        #print(src)
        #print(self.src_mask)
        #print(self.src_key_padding_mask)
        #output=self.transformer_encoder(src,self.src_mask,self.src_key_padding_mask)
        output=self.transformer_encoder(src)
        output=self.decoder(output)
        output=np.squeeze(output)
        output=self.linear1(output)
        return output