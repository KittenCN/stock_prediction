import re
import target
import mplfinance as mpf
import matplotlib as mpl# 用于设置曲线参数
from cycler import cycler# 用于定制线条颜色
import matplotlib.pyplot as plt
from prefetch_generator import BackgroundGenerator
from init import *

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
            mean_list.clear()
            std_list.clear()
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
            if self.mode in [0, 2]:
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
                for index in range(label_num):
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

class stock_queue_dataset(Dataset):
    # mode 0:train 1:test 2:predict
    def __init__(self, mode=0, data_queue=None, label_num=1, buffer_size=100, total_length=0):
        try:
            assert mode in [0, 1, 2]
            self.mode = mode
            self.data_queue = data_queue
            self.label_num = label_num
            self.buffer_size = buffer_size
            self.buffer_index = 0
            self.value_buffer = []
            self.label_buffer = []
            self.total_length = total_length
            # if data_queue is not None:
            #     self.total_length = 0
            #     while not data_queue.empty():
            #         data_frame = data_queue.get()
            #         data_frame = data_frame.dropna()
            #         self.total_length += len(data_frame) - SEQ_LEN
            #         self.data_queue.put(data_frame)
        except Exception as e:
            print(e)
            return None

    def load_data(self):
        if self.data_queue.empty():
            return None
        else:
            dataFrame = self.data_queue.get()
            dataFrame.drop(['ts_code', 'Date'], axis=1, inplace=True)
            dataFrame = dataFrame.dropna()
            data = dataFrame.values[:, 0:INPUT_DIMENSION]
            return data

    def normalize_data(self, data):
        if self.mode in [0, 2]:
            mean_list.clear()
            std_list.clear()
        for i in range(len(data[0])):
            if self.mode in [0, 2]:
                mean_list.append(np.mean(data[:, i]))
                std_list.append(np.std(data[:, i]))

            data[:, i] = (data[:, i] - mean_list[i]) / (std_list[i] + 1e-8)
        return data

    def generate_value_label_tensors(self, data, label_num):
        value = torch.rand(data.shape[0] - SEQ_LEN, SEQ_LEN, data.shape[1])
        label = torch.rand(data.shape[0] - SEQ_LEN, label_num)

        for i in range(data.shape[0] - SEQ_LEN):
            _value_tmp = np.copy(np.flip(data[i + 1:i + SEQ_LEN + 1, :].reshape(SEQ_LEN, data.shape[1]), 0))
            value[i, :, :] = torch.from_numpy(_value_tmp)

            _tmp = []
            for index in range(OUTPUT_DIMENSION):
                if use_list[index] == 1:
                    _tmp.append(data[i, index])

            label[i, :] = torch.Tensor(_tmp)

        _value = value.flip(0)
        _label = label.flip(0)
        return _value, _label

    # def process_data(self):
    #     raw_data = self.load_data()
    #     if raw_data is not None:
    #         while len(raw_data) < SEQ_LEN:
    #             raw_data = self.load_data()
    #             if raw_data is None:
    #                 return None
    #     if raw_data is not None:
    #         normalized_data = self.normalize_data(raw_data)
    #         value, label = self.generate_value_label_tensors(normalized_data, self.label_num)
    #         self.value_buffer.extend(value)
    #         self.label_buffer.extend(label)
    #     if raw_data is None:
    #         return None

    def process_data(self):
        # Check if there is data in the queue
        if self.data_queue.empty():
            return None

        for _ in range(self.buffer_size):  # Loop for buffer_size times
            try:
                raw_data = self.load_data()
                if raw_data is not None:
                    while len(raw_data) < SEQ_LEN:
                        raw_data = self.load_data()
                        if raw_data is None:
                            break
                    if raw_data is not None:
                        normalized_data = self.normalize_data(raw_data)
                        value, label = self.generate_value_label_tensors(normalized_data, self.label_num)
                        
                        self.value_buffer.extend(value)
                        self.label_buffer.extend(label)
                    else:
                        continue
                else:
                    continue
            except:
                continue

        if len(self.value_buffer) == 0 or len(self.label_buffer) == 0:
            return None

    def __getitem__(self, index):
        while self.buffer_index >= len(self.value_buffer):
            self.value_buffer.clear()
            self.label_buffer.clear()
            self.buffer_index = 0
            ans = self.process_data()
            if ans is None:
                break
        if self.buffer_index >= len(self.value_buffer):
            return None, None
        value, label = self.value_buffer[self.buffer_index], self.label_buffer[self.buffer_index]
        self.buffer_index += 1
        return value, label

    def __len__(self):
        # if self.data_queue is None:
        #     return len(self.value_buffer)
        # else:
        #     return self.total_length
        return self.total_length
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
    def forward(self,x, tgt):
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

class TransformerEncoderLayerWithNorm(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", norm=None):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)
        if norm is not None:
            self.norm1 = norm
            self.norm2 = norm

class TransformerDecoderLayerWithNorm(nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", norm=None):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)
        if norm is not None:
            self.norm1 = norm
            self.norm2 = norm
            self.norm3 = norm

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, output_dim, max_len=5000):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Embedding(max_len, d_model)

        self.transformer_encoder_layer = TransformerEncoderLayerWithNorm(d_model, nhead, dim_feedforward, norm=nn.LayerNorm(d_model))
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers)

        self.transformer_decoder_layer = TransformerDecoderLayerWithNorm(d_model, nhead, dim_feedforward, norm=nn.LayerNorm(d_model))
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers)


        self.target_embedding = nn.Linear(output_dim, d_model)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, output_dim)

        self._initialize_weights()

    def forward(self, src, tgt):
        src = src.permute(1, 0, 2) # (batch_size, seq_len, input_dim) -> (seq_len, batch_size, input_dim)

        src_embedding = self.embedding(src)
        src_seq_length = src.size(0)
        src_batch_size = src.size(1)

        src_positions = torch.arange(src_seq_length, device=src.device).unsqueeze(1).expand(src_seq_length, src_batch_size)
        src = src_embedding + self.positional_encoding(src_positions)

        memory = self.transformer_encoder(src)

        tgt = tgt.unsqueeze(1)
        tgt_embedding = self.target_embedding(tgt)
        tgt_seq_length = tgt.size(1)

        tgt_positions = torch.arange(tgt_seq_length, device=tgt.device).unsqueeze(1).expand(tgt_seq_length, src_batch_size)
        tgt = tgt_embedding + self.positional_encoding(tgt_positions)

        output = self.transformer_decoder(tgt.transpose(0, 1), memory)

        output = output.permute(1, 0, 2)  # (batch_size, seq_len, d_model) -> (seq_len, batch_size, d_model)
        pooled_output = self.pooling(output.permute(0, 2, 1))
        output = self.fc(pooled_output.squeeze(2))

        return output

    def generate_positional_encoding(self, d_model, max_len=5000):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        return pe

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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

def import_csv(stock_code, dataFrame=None, csv_file=None):
    try:
        if dataFrame is None:
            if csv_file is not None:
                file_path = csv_file
            else:
                file_path = daily_path+f"/{stock_code}.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
            else:
                # csv_queue.put(NoneDataFrame)
                return None
        elif isinstance(dataFrame, pd.DataFrame) and not dataFrame.empty:
            df = copy.deepcopy(dataFrame)
        else:
            # csv_queue.put(NoneDataFrame)
            return None
        add_target(df)
        df = df_queue.get()
        # data_wash(df, keepTime=False)
        # df = df_queue.get()
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
        # csv_queue.put(NoneDataFrame)
        return None

    if df.empty:
        # csv_queue.put(NoneDataFrame)
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
    times = times[::-1]
    df_queue.put(df)
    return df


def load_data(ts_codes, pbar=False, csv_file=None):
    if pbar: 
        pbar = tqdm(total=len(ts_codes))
    for ts_code in ts_codes:
        ans = import_csv(ts_code, None, csv_file)
        
        if ans is None:
            continue
        data = csv_queue.get()
        data_queue.put(data)
        if pbar:
            pbar.update(1)
    if pbar:
        pbar.close()

def cross_entropy(pred, target):
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(-target * logsoftmax(pred), dim=1))

def custom_collate(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if len(batch) > 0:
        return torch.utils.data.dataloader.default_collate(batch)
    else:
        return None

def save_model(model, optimizer, save_path):
    torch.save(model.state_dict(), save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_Model.pkl")
    torch.save(optimizer.state_dict(), save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_Optimizer.pkl")

def thread_save_model(model, optimizer, save_path):
    _model = copy.deepcopy(model)
    _optimizer = copy.deepcopy(optimizer)
    data_thread = threading.Thread(target=save_model, args=(_model, _optimizer, save_path,))
    data_thread.start()

def deep_copy_queue(q):
    new_q = multiprocessing.Queue()
    temp_q = []
    while not q.empty():
        try:
            item = q.get(timeout=1)
            temp_q.append(item)
        except queue.Empty:
            break
    for item in temp_q:
        new_q.put(item)
        q.put(item)
    return new_q