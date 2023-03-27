import math
import os
import queue
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import re
import target
import mplfinance as mpf
import matplotlib as mpl# 用于设置曲线参数
from cycler import cycler# 用于定制线条颜色
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from prefetch_generator import BackgroundGenerator

TRAIN_WEIGHT=0.9
SEQ_LEN=179
LEARNING_RATE=0.001   # 0.00001
WEIGHT_DECAY=0.0001   # 0.05
BATCH_SIZE=512
EPOCH=100
SAVE_NUM_ITER=100
SAVE_NUM_EPOCH=10
GET_DATA=True
TEST_NUM=25
SAVE_INTERVAL=300
OUTPUT_DIMENSION=4
INPUT_DIMENSION=20
TQDM_NCOLS = 100
NUM_WORKERS = 4
PKL = True

mean_list=[]
std_list=[]
data_queue=queue.Queue()
stock_data_queue=queue.Queue()
stock_list_queue = queue.Queue()
csv_queue=queue.Queue()
df_queue=queue.Queue()

NoneDataFrame = pd.DataFrame(columns=["ts_code"])
NoneDataFrame["ts_code"] = ["None"]

name_list = ["open", "high", "low", "close", "change", "pct_chg", "vol", "amount"]
use_list = [1,1,1,1,0,0,0,0]
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

train_path="./stock_handle/stock_train.csv"
test_path="./stock_handle/stock_test.csv"
train_pkl_path="./pkl_handle/train.pkl"

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

#完成数据集类
class Stock_Data(Dataset):
    # mode 0:train 1:test 2:predict
    def __init__(self, mode=0, transform=None, dataFrame=None, label_num=1):
        try:
            assert mode in [0, 1, 2]
            self.mode = mode
            self.data = self.load_data(dataFrame)
            self.normalize_data()
            self.value, self.label = self.generate_value_label_tensors(label_num)
        except Exception as e:
            print(e)
            return None

    def load_data(self, dataFrame):
        if self.mode in [0, 2]:
            path = train_path
        else:
            path = test_path

        if dataFrame is None:
            with open(path) as f:
                data = np.loadtxt(f, delimiter=",")
        else:
            data = dataFrame.values

        return data[:, 0:INPUT_DIMENSION]

    def normalize_data(self):
        for i in range(len(self.data[0])):
            if self.mode == 0:
                mean_list.append(np.mean(self.data[:, i]))
                std_list.append(np.std(self.data[:, i]))

            self.data[:, i] = (self.data[:, i] - mean_list[i]) / (std_list[i] + 1e-8)

    def generate_value_label_tensors(self, label_num):
        if self.mode in [0, 1]:
            value = torch.rand(self.data.shape[0] - SEQ_LEN, SEQ_LEN, self.data.shape[1])
            label = torch.rand(self.data.shape[0] - SEQ_LEN, label_num)

            for i in range(self.data.shape[0] - SEQ_LEN):
                _value_tmp = np.copy(np.flip(self.data[i + 1:i + SEQ_LEN + 1, :].reshape(SEQ_LEN, self.data.shape[1]), 0))
                value[i, :, :] = torch.from_numpy(_value_tmp)

                _tmp = []
                for index in range(OUTPUT_DIMENSION):
                    if use_list[index] == 1:
                        _tmp.append(self.data[i, index])

                label[i, :] = torch.Tensor(_tmp)
        elif self.mode == 2:
            value = torch.rand(1, SEQ_LEN, self.data.shape[1])
            label = torch.rand(1, label_num)
            for i in range(0, SEQ_LEN):
                _i = SEQ_LEN - i - 1
                value[0, i, :] = torch.from_numpy(self.data[_i, :].reshape(1, self.data.shape[1]))
                
        _value = value.flip(0)
        _label = label.flip(0)
        return _value, _label

    def __getitem__(self, index):
        return self.value[index], self.label[index]

    def __len__(self):
        return len(self.value)
#LSTM模型
class LSTM(nn.Module):
    def __init__(self,dimension):
        super(LSTM,self).__init__()
        self.lstm=nn.LSTM(input_size=dimension,hidden_size=128,num_layers=3,batch_first=True, dropout=0.5)
        self.linear1=nn.Linear(in_features=128,out_features=16)
        self.linear2=nn.Linear(16,OUTPUT_DIMENSION)
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
        pe = pe.to(x.device, non_blocking=True).float()
        return x + pe

class TransAm(nn.Module):
    def __init__(self, feature_size: int = 8, num_layers: int = 6, dropout: float = 0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        # self.decoder = nn.Linear(feature_size, 1)
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers)
        self.linear1 = nn.Linear(SEQ_LEN, OUTPUT_DIMENSION)
        self.init_weights()
        self.src_key_padding_mask = None

    def init_weights(self):
        initrange = 0.1
        # nn.init.zeros_(self.decoder.bias)
        # nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src: torch.Tensor, seq_len: int = SEQ_LEN) -> torch.Tensor:
r        src = self.pos_encoder(src)
        tgt = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        # output = self.decoder(output)
        output = self.transformer_decoder(tgt, output)
        # output = torch.squeeze(output)
        output = self.linear1(output)
        return output
    
def is_number(num):
    pattern = re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$')
    result = pattern.match(num)
    if result:
        return True
    else:
        return False
    
#数据清洗：丢弃行，或用上一行的值填充
def data_wash(dataset,keepTime=False):
    if keepTime:
        dataset.fillna(axis=1,method='ffill')
    else:
        dataset.dropna()
    df_queue.put(dataset)
    return dataset

def import_csv(stock_code, dataFrame=None):
    if dataFrame is None:
        file_path = f'stock_daily/{stock_code}.csv'
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
        else:
            csv_queue.put(NoneDataFrame)
            return None
    else:
        df = dataFrame
    try:
        add_target(df)
        df = df_queue.get()

        data_wash(df, keepTime=False)
        df = df_queue.get()
        df.rename(
            columns={
                'trade_date': 'Date', 'open': 'Open',
                'high': 'High', 'low': 'Low',
                'close': 'Close', 'vol': 'Volume'},
            inplace=True)

        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
        df.set_index(df['Date'], inplace=True)
    except Exception as e:
        print(stock_code, e)
        csv_queue.put(NoneDataFrame)
        return None

    if df.empty:
        csv_queue.put(NoneDataFrame)
        return None

    csv_queue.put(df)
    return df

def draw_Kline(df, period, symbol):
    kwargs = dict(
        type='candle',
        mav=(7, 30, 60),
        volume=True,
        title=f'\nA_stock {symbol} candle_line',
        ylabel='OHLC Candles',
        ylabel_lower='Shares\nTraded Volume',
        figratio=(15, 10),
        figscale=2)

    mc = mpf.make_marketcolors(
        up='red',
        down='green',
        edge='i',
        wick='i',
        volume='in',
        inherit=True)

    s = mpf.make_mpf_style(
        gridaxis='both',
        gridstyle='-.',
        y_on_right=False,
        marketcolors=mc)

    mpl.rcParams['axes.prop_cycle'] = cycler(
        color=['dodgerblue', 'deeppink',
               'navy', 'teal', 'maroon', 'darkorange',
               'indigo'])

    mpl.rcParams['lines.linewidth'] = .5

    mpf.plot(df,
             **kwargs,
             style=s,
             show_nontrading=False)

    mpf.plot(df,
             **kwargs,
             style=s,
             show_nontrading=False,
             savefig=f'A_stock-{symbol} {period}_candle_line.jpg')
    plt.show()

def data_replace(data):  #截取小数点后两位
    data = str(data)
    index = data.index('.')
    return float(data[:index+3])

def cmp_append(data, cmp_data):  #比较数据，如果数据不同则添加到列表
    # while len(data) < len(cmp_data):
    #     data.append(0)
    if len(cmp_data) - len(data) > 0:
        data += [0] * (len(cmp_data) - len(data))
    # data = np.nan_to_num(data)
    return data

def add_target(df):
    if 'trade_date' in df.columns:
        # times = [datetime.datetime.fromtimestamp(int(str(ts.value)[:10])).strftime('%Y%m%d') for ts in df['trade_date'].tolist()]
        times = np.array(df['trade_date'].values)
        close = np.array(df['close'].values)
        hpri = np.array(df['high'].values)
        lpri = np.array(df['low'].values)
        vol = np.array(df['vol'].values)
    elif 'Date' in df.columns:
        times = np.array(df['Date'].values)
        close = np.array(df['Close'].values)
        hpri = np.array(df['High'].values)
        lpri = np.array(df['Low'].values)
        vol = np.array(df['Volume'].values)
    
    times = times[::-1]
    close = close[::-1]
    hpri = hpri[::-1]
    lpri = lpri[::-1]
    vol = vol[::-1]

    macd_dif, macd_dea, macd_bar = target.MACD(close)
    df["macd_dif"] = cmp_append(macd_dif[::-1], df)
    df["macd_dea"] = cmp_append(macd_dea[::-1], df)
    df["macd_bar"] = cmp_append(macd_bar[::-1], df)
    k, d, j = target.KDJ(close, hpri, lpri)
    df['k'] = cmp_append(k[::-1], df)
    df['d'] = cmp_append(d[::-1], df)
    df['j'] = cmp_append(j[::-1], df)
    boll_upper, boll_mid, boll_lower = target.BOLL(close)
    df['boll_upper'] = cmp_append(boll_upper[::-1], df)
    df['boll_mid'] = cmp_append(boll_mid[::-1], df)
    df['boll_lower'] = cmp_append(boll_lower[::-1], df)
    cci = target.CCI(close, hpri, lpri)
    df['cci'] = cmp_append(cci[::-1], df)
    pdi, mdi, adx, adxr = target.DMI(close, hpri, lpri)
    df['pdi'] = cmp_append(pdi[::-1], df)
    df['mdi'] = cmp_append(mdi[::-1], df)
    df['adx'] = cmp_append(adx[::-1], df)
    df['adxr'] = cmp_append(adxr[::-1], df)
    taq_up, taq_mid, taq_down = target.TAQ(hpri, lpri, 5)
    df['taq_up'] = cmp_append(taq_up[::-1], df)
    df['taq_mid'] = cmp_append(taq_mid[::-1], df)
    df['taq_down'] = cmp_append(taq_down[::-1], df)
    trix, trma = target.TRIX(close)
    df['trix'] = cmp_append(trix[::-1], df)
    df['trma'] = cmp_append(trma[::-1], df)
    atr = target.ATR(close, hpri, lpri)
    df['atr'] = cmp_append(atr[::-1], df)
    df = df.reindex(columns=[
        "ts_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "change",
        "pct_chg",
        "vol",
        "amount",
        "macd_dif",
        "macd_dea",
        "macd_bar",
        "k",
        "d",
        "j",
        "boll_upper",
        "boll_mid",
        "boll_lower",
        "cci",
        "pdi",
        "mdi",
        "adx",
        "adxr",
        "taq_up",
        "taq_mid",
        "taq_down",
        "trix",
        "trma",
        "atr",
        "pre_close"
    ])
    df_queue.put(df)
    return df

def load_data(ts_codes):
    for ts_code in ts_codes:
        if data_queue.empty():
            print("data_queue is empty, loading data...")
        if GET_DATA:
            # get_stock_data(ts_code, False)
            # dataFrame = stock_data_queue.get()
            import_csv(ts_code, None)
            data = csv_queue.get()
            data_queue.put(data)
            # data_list.append(data)
            