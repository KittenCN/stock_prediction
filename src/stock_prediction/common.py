import re
import queue
import copy
import matplotlib as mpl
import mplfinance as mpf
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
from cycler import cycler
from prefetch_generator import BackgroundGenerator
from datetime import datetime, timedelta
from torchvision.models import resnet101  # 未使用，保留占位

# 处理相对导入问题
try:
    from .init import *
    from . import target
except ImportError:
    # 如果直接运行此文件，使用绝对导入
    import sys
    from pathlib import Path
    current_dir = Path(__file__).resolve().parent
    root_dir = current_dir.parent.parent
    src_dir = root_dir / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from stock_prediction.init import *
    import stock_prediction.target as target


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class CustomSchedule(object):
    def __init__(self, d_model, warmup_steps=4000, optimizer=None):
        super(CustomSchedule, self).__init__()
        self.d_model = torch.tensor(d_model, dtype=torch.float32)
        self.warmup_steps = warmup_steps
        self.steps = 1.0
        self.optimizer = optimizer

    def step(self):
        arg1 = self.steps ** -0.5
        arg2 = self.steps * (self.warmup_steps ** -1.5)
        self.steps += 1.0
        # 计算为 Python float，避免将 tensor 赋给 optimizer 学习率
        lr = float((self.d_model.item() ** -0.5) * min(arg1, arg2))
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        return lr


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]

        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertDataSet(torch.utils.data.Dataset):
    def __init__(self, root, is_train=1):
        self.tokenizer = BertTokenizer.from_pretrained(bert_data_path+'/base_model/bert-base-chinese', cache_dir=bert_data_path+'/model/')
        self.data_num = 7346
        self.x_list = []
        self.y_list = []
        self.posi = []

        with open(bert_data_path+'/data'+'/Train_DataSet.csv', encoding='UTF-8') as f:
            for i in range(self.data_num+1):
                line = f.readline()[:-1] + '这是一个中性的数据'
                data_one_str = line.split(',')[len(line.split(','))-2]
                data_two_str = line.split(',')[len(line.split(','))-1]
                if len(data_one_str) < 6:
                    data_one_str = data_one_str + '，' + data_two_str[0:min(200, len(data_two_str))]
                if i == 0:
                    continue
                word_l = self.tokenizer.encode(data_one_str, add_special_tokens=False)
                if len(word_l) < 100:
                    while len(word_l) != 100:
                        word_l.append(0)
                else:
                    word_l = word_l[0:100]
                word_l.append(102)
                l = word_l
                word_l = [101]
                word_l.extend(l)
                self.x_list.append(torch.tensor(word_l))
                self.posi.append(torch.tensor([i for i in range(102)]))

        with open(bert_data_path+'/data'+'/Train_DataSet_Label.csv', encoding='UTF-8') as f:
            for i in range(self.data_num+1):
                label_one = f.readline()[-2]
                if i == 0:
                    continue
                label_one = int(label_one)
                self.y_list.append(torch.tensor(label_one))

        if is_train == 1:
            self.x_list = self.x_list[0:6000]
            self.y_list = self.y_list[0:6000]
            self.posi = self.posi[0:6000]
        else:
            self.x_list = self.x_list[6000:]
            self.y_list = self.y_list[6000:]
            self.posi = self.posi[6000:]

        self.len = len(self.x_list)

    def __getitem__(self, index):
        return self.x_list[index], self.y_list[index], self.posi[index]

    def __len__(self):
        return self.len


class Bert_Model(torch.nn.Module):
    def __init__(self, pretrained_model, opt):
        super().__init__()
        self.pretrain_model = pretrained_model
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(768)
        self.fc = torch.nn.Linear(768, opt.num_labels)
        self.opt = opt

    def forward(self, input_ids, attention_mask, token_type_ids):
        if self.opt.no_grad == 1:
            with torch.no_grad():
                output = self.pretrain_model(input_ids=input_ids,
                                             attention_mask=attention_mask,
                                             token_type_ids=token_type_ids)
        else:
            output = self.pretrain_model(input_ids=input_ids,
                                         attention_mask=attention_mask,
                                         token_type_ids=token_type_ids)
        output = self.dropout(output[0][:, 0])
        output = self.layer_norm(output)
        output = self.fc(output)
        output = output.softmax(dim=1)
        return output


class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['label'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


class Stock_Data(Dataset):
    # mode 0:train 1:test 2:predict
    def __init__(self, mode=0, transform=None, dataFrame=None, label_num=1, predict_days=0, trend=0):
        try:
            assert mode in [0, 1, 2]
            self.mode = mode
            self.predict_days = predict_days
            self.data = self.load_data(dataFrame)
            self.normalize_data()
            self.value, self.label = self.generate_value_label_tensors(label_num)
            self.trend = trend
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
        if self.mode in [0, 2]:
            mean_list.clear()
            std_list.clear()
            for i in range(len(self.data[0])):
                if self.mode in [0, 2]:
                    mean_list.append(np.mean(self.data[:, i]))
                    std_list.append(np.std(self.data[:, i]))

                self.data[:, i] = (self.data[:, i] - mean_list[i]) / (std_list[i] + 1e-8)
        else:
            test_mean_list.clear()
            test_std_list.clear()
            for i in range(len(self.data[0])):
                if self.mode not in [0, 2]:
                    test_mean_list.append(np.mean(self.data[:, i]))
                    test_std_list.append(np.std(self.data[:, i]))

                self.data[:, i] = (self.data[:, i] - test_mean_list[i]) / (test_std_list[i] + 1e-8)
        return self.data

    def generate_value_label_tensors(self, label_num):
        if self.mode in [0, 1]:
            value = torch.rand(self.data.shape[0] - SEQ_LEN, SEQ_LEN, self.data.shape[1])
            if self.predict_days > 0:
                label = torch.rand(self.data.shape[0] - SEQ_LEN, self.predict_days, label_num)
            elif self.predict_days <= 0:
                label = torch.rand(self.data.shape[0] - SEQ_LEN, label_num)

            for i in range(self.data.shape[0] - SEQ_LEN):
                _value_tmp = np.copy(np.flip(self.data[i + 1:i + SEQ_LEN + 1, :].reshape(SEQ_LEN, self.data.shape[1]), 0))
                value[i, :, :] = torch.from_numpy(_value_tmp)
                _tmp = []
                for index in range(len(use_list)):
                    if use_list[index] == 1:
                        if self.predict_days <= 0:
                            _tmp.append(self.data[i, index])
                        elif self.predict_days > 0:
                            _tmp.append(self.data[i:i + self.predict_days, index])
                if self.predict_days <= 0:
                    label[i, :] = torch.Tensor(np.array(_tmp))
                elif self.predict_days > 0:
                    label[i, :, :] = torch.Tensor(np.array(_tmp)).permute(1, 0)
        elif self.mode == 2:
            value = torch.rand(1, SEQ_LEN, self.data.shape[1])
            if self.predict_days <= 0:
                label = torch.rand(1, label_num)
            elif self.predict_days > 0:
                label = torch.rand(1, self.predict_days, label_num)
            _value_tmp = np.copy(np.flip(self.data[0:SEQ_LEN, :].reshape(SEQ_LEN, self.data.shape[1]), 0))
            value[0, :, :] = torch.from_numpy(_value_tmp)
            if self.trend == 1:
                if self.predict_days > 0:
                    label[i][0] = compare_tensor(label[i][0], value[0][-1][:OUTPUT_DIMENSION])

        _value = value.flip(0)
        _label = label.flip(0)
        return _value, _label

    def __getitem__(self, index):
        return self.value[index], self.label[index]

    def __len__(self):
        return len(self.value)


class stock_queue_dataset(Dataset):
    # mode 0:train 1:test 2:predict
    def __init__(self, mode=0, data_queue=None, label_num=1, buffer_size=100, total_length=0, predict_days=0, trend=0):
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
            self.predict_days = predict_days
            self.trend = trend
        except Exception as e:
            print(e)
            return None

    def load_data(self):
        if self.data_queue.empty():
            return None
        else:
            try:
                dataFrame = self.data_queue.get(timeout=30)
            except queue.Empty:
                return None
            dataFrame.drop(['ts_code', 'Date'], axis=1, inplace=True)
            dataFrame = dataFrame.fillna(dataFrame.median(numeric_only=True))
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
        else:
            test_mean_list.clear()
            test_std_list.clear()
            for i in range(len(data[0])):
                if self.mode not in [0, 2]:
                    test_mean_list.append(np.mean(data[:, i]))
                    test_std_list.append(np.std(data[:, i]))

                data[:, i] = (data[:, i] - test_mean_list[i]) / (test_std_list[i] + 1e-8)
        return data

    def generate_value_label_tensors(self, data, label_num):
        value = torch.rand(data.shape[0] - SEQ_LEN, SEQ_LEN, data.shape[1])
        if self.predict_days > 0:
            label = torch.rand(data.shape[0] - SEQ_LEN, self.predict_days, label_num)
        elif self.predict_days <= 0:
            label = torch.rand(data.shape[0] - SEQ_LEN, label_num)

        for i in range(data.shape[0] - SEQ_LEN):
            _value_tmp = np.copy(np.flip(data[i + 1:i + SEQ_LEN + 1, :].reshape(SEQ_LEN, data.shape[1]), 0))
            value[i, :, :] = torch.from_numpy(_value_tmp)

            _tmp = []
            for index in range(len(use_list)):
                if use_list[index] == 1:
                    if self.predict_days > 0:
                        _tmp.append(data[i:i + self.predict_days, index])
                    elif self.predict_days <= 0:
                        _tmp.append(data[i, index])
            if self.predict_days > 0:
                label[i, :, :] = torch.Tensor(np.array(_tmp)).permute(1, 0)
            elif self.predict_days <= 0:
                label[i, :] = torch.Tensor(np.array(_tmp))
            if self.trend == 1:
                if self.predict_days > 0:
                    label[i][0] = compare_tensor(label[i][0], value[0][-1][:OUTPUT_DIMENSION])
        _value = value.flip(0)
        _label = label.flip(0)
        return _value, _label

    def process_data(self):
        if self.data_queue.empty():
            return None

        for _ in range(self.buffer_size):
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
        else:
            return True

    def __getitem__(self, index):
        try:
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
        except Exception as e:
            print(e)
            return None, None

    def __len__(self):
        if self.data_queue is None:
            return len(self.value_buffer)
        else:
            return self.total_length


class LSTM(nn.Module):
    def __init__(self, dimension):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=dimension, hidden_size=128, num_layers=3, batch_first=True, dropout=0.5)
        self.linear1 = nn.Linear(in_features=128, out_features=16)
        self.linear2 = nn.Linear(16, OUTPUT_DIMENSION)
        self.LeakyReLU = nn.LeakyReLU()

    def forward(self, x, tgt, predict_days=0):
        out, _ = self.lstm(x)
        x = out[:, -1, :]
        x = self.linear1(x)
        x = self.LeakyReLU(x)
        x = self.linear2(x)
        if predict_days > 0:
            x = x.unsqueeze(1)
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


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)


class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, output_dim, max_len=5000, mode=0):
        super(TransformerModel, self).__init__()
        assert d_model % 2 == 0, "d_model must be a multiple of 2"
        self.embedding = MLP(input_dim, d_model // 2, d_model)
        self.positional_encoding = None

        if mode in [0]:
            dropout = 0.5
        elif mode in [1, 2]:
            dropout = 0

        self.transformer_encoder_layer = TransformerEncoderLayerWithNorm(d_model, nhead, dim_feedforward, norm=nn.LayerNorm(d_model), dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers)

        self.transformer_decoder_layer = TransformerDecoderLayerWithNorm(d_model, nhead, dim_feedforward, norm=nn.LayerNorm(d_model), dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers)

        self.target_embedding = nn.Linear(output_dim, d_model)
        self.pooling = nn.AdaptiveAvgPool1d
        self.fc = nn.Linear(d_model, output_dim)

        self.d_model = d_model
        self.output_dim = output_dim

        self._initialize_weights()

    def forward(self, src, tgt, predict_days=0):
        src = src.permute(1, 0, 2)

        attention_mask = generate_attention_mask(src)
        src_embedding = self.embedding(src)
        src_seq_length = src.size(0)
        src_batch_size = src.size(1)

        if self.positional_encoding is None or self.positional_encoding.size(0) < src_seq_length:
            self.positional_encoding = self.generate_positional_encoding(src_seq_length, self.d_model).to(src.device)

        src_positions = torch.arange(src_seq_length, device=src.device).unsqueeze(1).expand(src_seq_length, src_batch_size)
        src = src_embedding + self.positional_encoding[src_positions]

        memory = self.transformer_encoder(src, src_key_padding_mask=attention_mask)

        if predict_days <= 0:
            tgt = tgt.unsqueeze(0)
        else:
            tgt = tgt.permute(1, 0, 2)
        tgt_embedding = self.target_embedding(tgt)
        tgt_seq_length = tgt.size(0)

        tgt_positions = torch.arange(tgt_seq_length, device=tgt.device).unsqueeze(1).expand(tgt_seq_length, src_batch_size)
        tgt = tgt_embedding + self.positional_encoding[tgt_positions]

        output = self.transformer_decoder(tgt, memory)

        if predict_days <= 0:
            output = output.permute(1, 2, 0)
            pooled_output = self.pooling(1)(output)
            output = self.fc(pooled_output.squeeze(2))
        else:
            output = output.permute(1, 2, 0)
            pooled_output = self.pooling(predict_days)(output)
            output = self.fc(pooled_output.permute(0, 2, 1))

        return output

    def generate_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def is_number(num):
    """检查字符串是否为有效数字（整数或浮点数）"""
    try:
        float(num)
        return True
    except (ValueError, TypeError):
        return False


def data_wash(dataset, keepTime=False):
    if keepTime:
        dataset.fillna(axis=1, method='ffill')
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
                return None
        elif isinstance(dataFrame, pd.DataFrame) and not dataFrame.empty:
            df = copy.deepcopy(dataFrame)
        else:
            return None
        result = add_target(df)
        if result is None:
            return None
        df = df_queue.get(timeout=30)
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
        return None

    if df.empty:
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


def data_replace(data):
    data = str(data)
    index = data.index('.')
    return float(data[:index+3])


def cmp_append(data, cmp_data):
    if len(cmp_data) - len(data) > 0:
        data += [0] * (len(cmp_data) - len(data))
    return data


def add_target(df):
    if 'trade_date' in df.columns:
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

    try:
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
        if 'amplitude' not in df.columns and 'exchange_rate' not in df.columns and 'pre_close' in df.columns:
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
        elif 'amplitude' in df.columns and 'exchange_rate' in df.columns and 'pre_close' not in df.columns:
            df = df.reindex(columns=[
                "ts_code",
                "trade_date",
                "open",
                "high",
                "low",
                "close",
                "change",
                "pct_change",
                "vol",
                "amount",
                "amplitude",
                "exchange_rate",
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
                "atr"
            ])
        times = times[::-1]
        df_queue.put(df)
        return df
    except Exception as e:
        df_queue.queue.clear()
        print(f"{df['ts_code']} {e}")
        return None


def load_data(ts_codes, pbar=False, csv_file=None, data_queue=data_queue):
    if pbar:
        pbar = tqdm(total=len(ts_codes))
    for ts_code in ts_codes:
        ans = import_csv(ts_code, None, csv_file)
        if ans is None:
            continue
        try:
            data = csv_queue.get(timeout=30)
        except queue.Empty:
            continue
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


def _move_state_to_cpu(state):
    if isinstance(state, dict):
        return {k: _move_state_to_cpu(v) for k, v in state.items()}
    if isinstance(state, list):
        return [_move_state_to_cpu(v) for v in state]
    if isinstance(state, tuple):
        return tuple(_move_state_to_cpu(v) for v in state)
    if torch.is_tensor(state):
        return state.detach().cpu()
    return copy.deepcopy(state)


def save_model(model, optimizer, save_path, best_model=False, predict_days=0):
    if isinstance(model, dict):
        model_state = model
    else:
        model_state = {k: v.detach().cpu().clone() if torch.is_tensor(v) else v for k, v in model.state_dict().items()}
    if isinstance(optimizer, dict):
        optimizer_state = optimizer
    else:
        optimizer_state = _move_state_to_cpu(optimizer.state_dict())

    if predict_days > 0:
        if best_model is False:
            torch.save(model_state, save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_pre" + str(predict_days) + "_Model.pkl")
            torch.save(optimizer_state, save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_pre" + str(predict_days) + "_Optimizer.pkl")
        elif best_model is True:
            torch.save(model_state, save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_pre" + str(predict_days) + "_Model_best.pkl")
            torch.save(optimizer_state, save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_pre" + str(predict_days) + "_Optimizer_best.pkl")
    else:
        if best_model is False:
            torch.save(model_state, save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_Model.pkl")
            torch.save(optimizer_state, save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_Optimizer.pkl")
        elif best_model is True:
            torch.save(model_state, save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_Model_best.pkl")
            torch.save(optimizer_state, save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_Optimizer_best.pkl")


def thread_save_model(model, optimizer, save_path, best_model=False, predict_days=0):
    model_state = {k: v.detach().cpu().clone() if torch.is_tensor(v) else v for k, v in model.state_dict().items()}
    optimizer_state = _move_state_to_cpu(optimizer.state_dict())
    data_thread = threading.Thread(target=save_model, args=(model_state, optimizer_state, save_path, best_model, predict_days,))
    data_thread.start()


def deep_copy_queue(q):
    new_q = multiprocessing.Queue()
    temp_q = []
    while not q.empty():
        try:
            item = q.get_nowait()
            temp_q.append(item)
        except queue.Empty:
            break
    for item in temp_q:
        new_q.put(item)
        q.put(item)
    return new_q


def ensure_queue_compatibility(q_obj):
    """为旧版序列化的 queue.Queue 补齐缺失属性，保证在新版本 Python 中可用。"""
    if isinstance(q_obj, queue.Queue) and not hasattr(q_obj, "is_shutdown"):
        q_obj.is_shutdown = False
    return q_obj


def read_csv_file(file_path):
    csv_files = glob.glob(file_path+"/*.csv")
    df = None
    for csv_file in csv_files:
        if df is None:
            df = pd.read_csv(csv_file, delimiter=',')
        else:
            df = pd.concat([df, pd.read_csv(csv_file, delimiter=',')])
    return df


def tokenize_data(df, tokenizer=None, max_length=None):
    encodings = df['text'].tolist()
    labels = df['label'].tolist()
    return encodings, labels


def csvToDataset(csvfile):
    df = read_csv_file(csvfile)
    encodings, labels = tokenize_data(df)
    dataset = TextDataset({'text': encodings}, labels)
    return dataset


def pad_input(input_data, max_features=INPUT_DIMENSION):
    padded_data = []
    for data in input_data:
        padding = torch.full((data.size(0), max_features - data.shape[-1]), -0.0).to(input_data.device)
        padded_data.append(torch.cat((data, padding), dim=-1))
    return torch.stack(padded_data)


def generate_attention_mask(input_data):
    mask = (input_data != -0.0).any(dim=-1)
    mask = mask.to(torch.float32)
    mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.T
    return mask


def generate_dates(start_date, num_days):
    start_date = datetime.strptime(start_date, '%Y%m%d')
    if num_days < 0:
        return np.array([(start_date + timedelta(days=i)).strftime('%Y%m%d') for i in range(num_days, 1)])
    else:
        return np.array([(start_date + timedelta(days=i)).strftime('%Y%m%d') for i in range(num_days+1)])


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, outputs):
        attn_weights = self.softmax(self.attn(outputs))
        return torch.bmm(attn_weights.transpose(1, 2), outputs).squeeze(1)


class CNNLSTM(nn.Module):
    def __init__(self, input_dim, num_classes=2, predict_days=1, dropout_rate=0.5):
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=3, batch_first=True)
        self.attention = Attention(256)
        self.fc1 = nn.Linear(256, 128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.predict_days = predict_days

    def forward(self, x_3d, _tgt, _predict_days=1):
        self.lstm.flatten_parameters()
        batch_size, seq_len, C = x_3d.size()
        tar = torch.zeros(batch_size, 1, OUTPUT_DIMENSION).to(x_3d.device)
        for index in range(x_3d.size(0)):
            tar[index] = x_3d[index][-1][:OUTPUT_DIMENSION]
        x_3d = x_3d.transpose(1, 2)
        c_out = F.relu(self.conv1(x_3d))
        c_out = F.relu(self.conv2(c_out))
        r_in = c_out.transpose(1, 2)
        r_out, _ = self.lstm(r_in)
        attn_out = self.attention(r_out)
        x = F.relu(self.fc1(attn_out))
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x.view(batch_size, self.predict_days, -1)


def compare_tensor(original, target):
    assert original.size() == target.size()
    result = torch.zeros(original.size()[0])
    for index in range(original.size()[0]):
        if original[index] == target[index]:
            result[index] = 0
        elif original[index] > target[index]:
            result[index] = 1
        else:
            result[index] = -1
    return result
