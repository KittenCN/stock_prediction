import re
import queue
import copy
import matplotlib as mpl
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
from cycler import cycler
from prefetch_generator import BackgroundGenerator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple
from torchvision.models import resnet101  # unused placeholder import

from .models.lstm_basic import LSTM
from .models.transformer_classic import TransformerModel
from .models.cnn_lstm import CNNLSTM




import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


try:
    from .init import *
    from . import target
except ImportError:
    from stock_prediction.init import *
    import stock_prediction.target as target


from stock_prediction.app_config import AppConfig
from stock_prediction.feature_engineering import FeatureEngineer
config = AppConfig.from_env_and_yaml(str(root_dir / 'config' / 'config.yaml'))
train_pkl_path = config.train_pkl_path
png_path = config.png_path
model_path = config.model_path
feature_engineer = FeatureEngineer.from_app_config(config)

def canonical_symbol(symbol: Optional[str]) -> Optional[str]:
    if symbol is None:
        return None
    s = str(symbol).strip()
    if not s:
        return None
    if "." in s:
        s = s.split(".", 1)[0]
    if s.isdigit():
        s = s.zfill(6)
    return s.upper()


def record_symbol_norm(symbol: Optional[str], means: list, stds: list) -> None:
    symbol_key = canonical_symbol(symbol)
    if not symbol_key:
        return
    if not means or not stds:
        return
    try:
        symbol_norm_map[symbol_key] = {
            "mean_list": [float(x) for x in means],
            "std_list": [float(x) for x in stds],
        }
    except Exception:
        pass


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
                line = f.readline()[:-1] + 'This is a neutral data sample'
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
    """Dataset wrapper that optionally emits per-sample symbol indices for embedding."""

    # mode 0:train 1:test 2:predict
    def __init__(self, mode=0, transform=None, dataFrame=None, label_num=1, predict_days=0, trend=0, norm_symbol=None):
        try:
            assert mode in [0, 1, 2]
            self.mode = mode
            self.predict_days = predict_days
            self.trend = trend
            self.norm_symbol = canonical_symbol(norm_symbol)
            self.use_symbol_embedding = feature_engineer.settings.use_symbol_embedding
            self.symbol_series: Optional[np.ndarray] = None
            self.symbol_tensor: Optional[torch.Tensor] = None
            self.data = self.load_data(dataFrame)
            if self.data is None or len(self.data) == 0:
                self.data = np.empty((0, INPUT_DIMENSION), dtype=np.float32)
            self.data = self.normalize_data(self.data, symbol=self.norm_symbol)
            tensors = self.generate_value_label_tensors(label_num)
            if tensors is None:
                self.value = torch.empty(0, SEQ_LEN, INPUT_DIMENSION, dtype=torch.float32)
                if self.predict_days > 0:
                    self.label = torch.empty(0, self.predict_days, label_num, dtype=torch.float32)
                else:
                    self.label = torch.empty(0, label_num, dtype=torch.float32)
                self.symbol_tensor = (
                    torch.empty(0, dtype=torch.long) if self.use_symbol_embedding else None
                )
            else:
                if isinstance(tensors, tuple) and len(tensors) == 3:
                    self.value, self.label, self.symbol_tensor = tensors
                else:
                    self.value, self.label = tensors
                    self.symbol_tensor = None
        except Exception as e:
            print(e)
            self.data = np.empty((0, INPUT_DIMENSION), dtype=np.float32)
            self.value = torch.empty(0, SEQ_LEN, INPUT_DIMENSION, dtype=torch.float32)
            if self.predict_days > 0:
                self.label = torch.empty(0, self.predict_days, label_num, dtype=torch.float32)
            else:
                self.label = torch.empty(0, label_num, dtype=torch.float32)
            self.symbol_tensor = torch.empty(0, dtype=torch.long) if self.use_symbol_embedding else None
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
                raw = np.loadtxt(f, delimiter=",")
            data_df = pd.DataFrame(raw)
        elif isinstance(dataFrame, pd.DataFrame):
            data_df = dataFrame.copy()
        else:
            data_df = pd.DataFrame(dataFrame)

        if self.norm_symbol is None and isinstance(dataFrame, pd.DataFrame):
            if "ts_code" in dataFrame.columns and not dataFrame.empty:
                self.norm_symbol = canonical_symbol(dataFrame["ts_code"].iloc[0])

        try:
            data_df = feature_engineer.transform(data_df)
        except Exception as exc:  # pragma: no cover - logging fallback
            tqdm.write(f"feature engineering failed: {exc}")
            data_df = data_df.fillna(0.0)

        if self.use_symbol_embedding and "_symbol_index" in data_df.columns:
            numeric_ids = pd.to_numeric(data_df["_symbol_index"], errors="coerce").fillna(-1).astype(int)
            self.symbol_series = numeric_ids.to_numpy()
        else:
            self.symbol_series = None

        numeric_df = (
            data_df.select_dtypes(include=[np.number])
            .drop(columns=["_symbol_index", "ts_code"], errors="ignore")
        )
        data = numeric_df.to_numpy(dtype=np.float32, copy=True)

        if data.shape[1] > INPUT_DIMENSION:
            data = data[:, :INPUT_DIMENSION]

        if np.isnan(data).any() or np.isinf(data).any():
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        return data

    def normalize_data(self, data=None, symbol=None):
        global saved_mean_list, saved_std_list
        if data is None:
            data = self.data
        symbol_key = canonical_symbol(symbol if symbol is not None else self.norm_symbol)

        if data is None or len(data) == 0:
            return data

        if self.mode in [0, 2]:
            mean_list.clear()
            std_list.clear()
            for i in range(len(data[0])):
                col_data = data[:, i]

                if np.isnan(col_data).any() or np.isinf(col_data).any():
                    col_data = np.nan_to_num(col_data, nan=0.0, posinf=0.0, neginf=0.0)
                    data[:, i] = col_data

                mean_val = np.mean(col_data)
                std_val = np.std(col_data)

                if np.isnan(mean_val) or np.isinf(mean_val):
                    mean_val = 0.0
                if np.isnan(std_val) or np.isinf(std_val) or std_val < 1e-8:
                    std_val = 1.0

                mean_list.append(mean_val)
                std_list.append(std_val)
                data[:, i] = (col_data - mean_val) / (std_val + 1e-8)

            if not saved_mean_list or not saved_std_list:
                saved_mean_list = mean_list.copy()
                saved_std_list = std_list.copy()
            if symbol_key:
                record_symbol_norm(symbol_key, mean_list.copy(), std_list.copy())
        else:
            saved_stats = symbol_norm_map.get(symbol_key) if symbol_key else None
            test_mean_list.clear()
            test_std_list.clear()
            if saved_stats and saved_stats.get('mean_list') and saved_stats.get('std_list'):
                means = saved_stats['mean_list']
                stds = saved_stats['std_list']
                for i in range(len(data[0])):
                    col_data = data[:, i]
                    if np.isnan(col_data).any() or np.isinf(col_data).any():
                        col_data = np.nan_to_num(col_data, nan=0.0, posinf=0.0, neginf=0.0)
                        data[:, i] = col_data
                    mean_val = means[i] if i < len(means) else 0.0
                    std_val = stds[i] if i < len(stds) else 1.0
                    test_mean_list.append(mean_val)
                    test_std_list.append(std_val)
                    data[:, i] = (col_data - mean_val) / (std_val + 1e-8)
            else:
                for i in range(len(data[0])):
                    col_data = data[:, i]
                    if np.isnan(col_data).any() or np.isinf(col_data).any():
                        col_data = np.nan_to_num(col_data, nan=0.0, posinf=0.0, neginf=0.0)
                        data[:, i] = col_data

                    mean_val = np.mean(col_data)
                    std_val = np.std(col_data)

                    if np.isnan(mean_val) or np.isinf(mean_val):
                        mean_val = 0.0
                    if np.isnan(std_val) or np.isinf(std_val) or std_val < 1e-8:
                        std_val = 1.0

                    test_mean_list.append(mean_val)
                    test_std_list.append(std_val)
                    data[:, i] = (col_data - mean_val) / (std_val + 1e-8)
                if symbol_key:
                    record_symbol_norm(symbol_key, test_mean_list.copy(), test_std_list.copy())
        if data is not self.data:
            return data
        self.data = data
        return self.data

    def _get_symbol_value(self, index: int) -> int:
        if self.symbol_series is None or len(self.symbol_series) == 0:
            return 0
        if index < len(self.symbol_series):
            symbol_id = int(self.symbol_series[index])
        else:
            symbol_id = int(self.symbol_series[-1])
        return max(symbol_id, 0)

    def generate_value_label_tensors(self, label_num):
        if self.mode in [0, 1]:
            sequence_count = self.data.shape[0] - SEQ_LEN
            if sequence_count <= 0:
                return None
            value = torch.rand(sequence_count, SEQ_LEN, self.data.shape[1])
            if self.predict_days > 0:
                label = torch.rand(sequence_count, self.predict_days, label_num)
            else:
                label = torch.rand(sequence_count, label_num)
            symbol_values: list[int] = []

            for i in range(sequence_count):
                window = self.data[i + 1 : i + SEQ_LEN + 1, :].reshape(SEQ_LEN, self.data.shape[1])
                flipped = np.flip(window, 0).astype(np.float32, copy=True)
                value[i, :, :] = torch.from_numpy(flipped)
                targets = []
                for index in range(len(use_list)):
                    if use_list[index] == 1:
                        if self.predict_days <= 0:
                            targets.append(self.data[i, index])
                        else:
                            targets.append(self.data[i : i + self.predict_days, index])
                if self.predict_days <= 0:
                    label[i, :] = torch.tensor(targets, dtype=torch.float32)
                else:
                    stacked = torch.tensor(targets, dtype=torch.float32).permute(1, 0)
                    label[i, :, :] = stacked
                if self.symbol_series is not None:
                    symbol_values.append(self._get_symbol_value(i))
        elif self.mode == 2:
            value = torch.rand(1, SEQ_LEN, self.data.shape[1])
            if self.predict_days <= 0:
                label = torch.rand(1, label_num)
            else:
                label = torch.rand(1, self.predict_days, label_num)
            if self.data.shape[0] < SEQ_LEN:
                return None
            window = self.data[0:SEQ_LEN, :].reshape(SEQ_LEN, self.data.shape[1])
            flipped = np.flip(window, 0).astype(np.float32, copy=True)
            value[0, :, :] = torch.from_numpy(flipped)
            if self.trend == 1 and self.predict_days > 0:
                label[0][0] = compare_tensor(label[0][0], value[0][-1][:OUTPUT_DIMENSION])
            symbol_values = [self._get_symbol_value(0)] if self.symbol_series is not None else []
        else:
            value = torch.tensor([])
            label = torch.tensor([])
            symbol_values = []

        _value = value.flip(0)
        _label = label.flip(0)
        if self.symbol_series is not None:
            if symbol_values:
                symbol_tensor = torch.tensor(symbol_values, dtype=torch.long).flip(0)
            else:
                symbol_tensor = torch.zeros(_value.size(0), dtype=torch.long)
            return _value, _label, symbol_tensor
        return _value, _label

    def __getitem__(self, index):
        if self.symbol_tensor is not None:
            return self.value[index], self.label[index], self.symbol_tensor[index]
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
            self.symbol_buffer: list[torch.Tensor] = []
            self.total_length = total_length
            self.predict_days = predict_days
            self.trend = trend
            self.use_symbol_embedding = feature_engineer.settings.use_symbol_embedding
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
            frame = dataFrame.copy()
            symbol_code = None
            if "ts_code" in frame.columns and not frame.empty:
                symbol_code = canonical_symbol(frame["ts_code"].iloc[0])
            if "trade_date" not in frame.columns and "Date" in frame.columns:
                frame = frame.rename(columns={"Date": "trade_date"})
            try:
                frame = feature_engineer.transform(frame)
            except Exception as exc:  # pragma: no cover - defensive logging
                tqdm.write(f"queue feature engineering failed: {exc}")
                frame = frame.fillna(frame.median(numeric_only=True)).fillna(0.0)

            symbol_series = None
            if self.use_symbol_embedding and "_symbol_index" in frame.columns:
                numeric_ids = pd.to_numeric(frame["_symbol_index"], errors="coerce").fillna(-1).astype(int)
                symbol_series = numeric_ids.to_numpy()

            numeric_frame = (
                frame.select_dtypes(include=[np.number])
                .drop(columns=["_symbol_index", "ts_code"], errors="ignore")
            )
            if numeric_frame.empty:
                return None
            data = numeric_frame.values
            if data.shape[1] > INPUT_DIMENSION:
                data = data[:, :INPUT_DIMENSION]
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            return data, symbol_series, symbol_code

    def normalize_data(self, data, symbol=None):
        global saved_mean_list, saved_std_list
        symbol_key = canonical_symbol(symbol)
        if self.mode in [0, 2]:
            mean_list.clear()
            std_list.clear()
            for i in range(len(data[0])):
                col_data = data[:, i]

                if np.isnan(col_data).any() or np.isinf(col_data).any():
                    col_data = np.nan_to_num(col_data, nan=0.0, posinf=0.0, neginf=0.0)
                    data[:, i] = col_data
                
                mean_val = np.mean(col_data)
                std_val = np.std(col_data)
                

                if np.isnan(mean_val) or np.isinf(mean_val):
                    mean_val = 0.0
                if np.isnan(std_val) or np.isinf(std_val) or std_val < 1e-8:
                    std_val = 1.0
                
                mean_list.append(mean_val)
                std_list.append(std_val)
                data[:, i] = (col_data - mean_val) / (std_val + 1e-8)
            
            # 保存稳定副本用于模型保存（只在第一次或列表为空时更新）
            if not saved_mean_list or not saved_std_list:
                saved_mean_list = mean_list.copy()
                saved_std_list = std_list.copy()
            if symbol_key:
                record_symbol_norm(symbol_key, mean_list.copy(), std_list.copy())
        else:
            saved_stats = symbol_norm_map.get(symbol_key) if symbol_key else None
            test_mean_list.clear()
            test_std_list.clear()
            if saved_stats and saved_stats.get('mean_list') and saved_stats.get('std_list'):
                means = saved_stats['mean_list']
                stds = saved_stats['std_list']
                for i in range(len(data[0])):
                    col_data = data[:, i]
                    if np.isnan(col_data).any() or np.isinf(col_data).any():
                        col_data = np.nan_to_num(col_data, nan=0.0, posinf=0.0, neginf=0.0)
                        data[:, i] = col_data
                    mean_val = means[i] if i < len(means) else 0.0
                    std_val = stds[i] if i < len(stds) else 1.0
                    test_mean_list.append(mean_val)
                    test_std_list.append(std_val)
                    data[:, i] = (col_data - mean_val) / (std_val + 1e-8)
            else:
                for i in range(len(data[0])):
                    col_data = data[:, i]
                    if np.isnan(col_data).any() or np.isinf(col_data).any():
                        col_data = np.nan_to_num(col_data, nan=0.0, posinf=0.0, neginf=0.0)
                        data[:, i] = col_data
                    
                    mean_val = np.mean(col_data)
                    std_val = np.std(col_data)
                    
                    if np.isnan(mean_val) or np.isinf(mean_val):
                        mean_val = 0.0
                    if np.isnan(std_val) or np.isinf(std_val) or std_val < 1e-8:
                        std_val = 1.0
                    
                    test_mean_list.append(mean_val)
                    test_std_list.append(std_val)
                    data[:, i] = (col_data - mean_val) / (std_val + 1e-8)
                if symbol_key:
                    record_symbol_norm(symbol_key, test_mean_list.copy(), test_std_list.copy())
        return data

    def _queue_symbol_value(self, symbol_series: Optional[np.ndarray], index: int) -> int:
        if symbol_series is None or len(symbol_series) == 0:
            return 0
        if index < len(symbol_series):
            value = int(symbol_series[index])
        else:
            value = int(symbol_series[-1])
        return max(value, 0)

    def generate_value_label_tensors(self, data, label_num, symbol_series=None):
        value = torch.rand(data.shape[0] - SEQ_LEN, SEQ_LEN, data.shape[1])
        if self.predict_days > 0:
            label = torch.rand(data.shape[0] - SEQ_LEN, self.predict_days, label_num)
        elif self.predict_days <= 0:
            label = torch.rand(data.shape[0] - SEQ_LEN, label_num)
        symbol_values: list[int] = []

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
            if symbol_series is not None:
                symbol_values.append(self._queue_symbol_value(symbol_series, i))
        _value = value.flip(0)
        _label = label.flip(0)
        if symbol_series is not None:
            if symbol_values:
                symbol_tensor = torch.tensor(symbol_values, dtype=torch.long).flip(0)
            else:
                symbol_tensor = torch.zeros(_value.size(0), dtype=torch.long)
            return _value, _label, symbol_tensor
        return _value, _label

    def process_data(self):
        if self.data_queue.empty():
            return None

        for _ in range(self.buffer_size):
            try:
                raw_entry = self.load_data()
                if raw_entry is not None:
                    raw_data, raw_symbol, raw_symbol_code = raw_entry
                    while raw_data is not None and len(raw_data) < SEQ_LEN:
                        next_entry = self.load_data()
                        if next_entry is None:
                            raw_data = None
                            break
                        raw_data, raw_symbol, raw_symbol_code = next_entry
                    if raw_data is not None:
                        normalized_data = self.normalize_data(raw_data, symbol=raw_symbol_code)
                        tensors = self.generate_value_label_tensors(normalized_data, self.label_num, raw_symbol)
                        if isinstance(tensors, tuple) and len(tensors) == 3:
                            value, label, symbol_tensor = tensors
                        else:
                            value, label = tensors
                            symbol_tensor = None
                        self.value_buffer.extend(value)
                        self.label_buffer.extend(label)
                        if self.use_symbol_embedding and symbol_tensor is not None:
                            self.symbol_buffer.extend(symbol_tensor.tolist())
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
                self.symbol_buffer.clear()
                self.buffer_index = 0
                ans = self.process_data()
                if ans is None:
                    break
            if self.buffer_index >= len(self.value_buffer):
                return (None, None, None) if self.use_symbol_embedding else (None, None)
            value = self.value_buffer[self.buffer_index]
            label = self.label_buffer[self.buffer_index]
            symbol = None
            if self.use_symbol_embedding:
                if self.symbol_buffer and self.buffer_index < len(self.symbol_buffer):
                    symbol_id = int(self.symbol_buffer[self.buffer_index])
                elif self.symbol_buffer:
                    symbol_id = int(self.symbol_buffer[-1])
                else:
                    symbol_id = 0
                symbol = torch.tensor(symbol_id, dtype=torch.long)
            self.buffer_index += 1
            if self.use_symbol_embedding:
                return value, label, symbol
            return value, label
        except Exception as e:
            print(e)
            return (None, None, None) if self.use_symbol_embedding else (None, None)

    def __len__(self):
        if self.data_queue is None:
            return len(self.value_buffer)
        else:
            return self.total_length


def is_number(num):
    """Check whether the provided string is a valid number (int or float)"""
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
    """保存模型与优化器，并确保归一化参数文件总是有效。

    逻辑：
    - 优先保存模型与优化器
    - 归一化参数优先取稳定副本(saved_mean_list/saved_std_list)；为空则回退全局(mean_list/std_list)
    - 若仍为空（常见于 PKL 模式），将从 train_pkl_path 自动计算并写入
    """
    import json, os, dill
    import queue as _q
    import pandas as _pd
    import numpy as _np

    # 组装可序列化状态
    if isinstance(model, dict):
        model_state = model
        model_args = None
    else:
        model_state = {k: v.detach().cpu().clone() if torch.is_tensor(v) else v for k, v in model.state_dict().items()}
        model_args = getattr(model, '_init_args', None)
    if isinstance(optimizer, dict):
        optimizer_state = optimizer
    else:
        optimizer_state = _move_state_to_cpu(optimizer.state_dict())

    def _compute_norm_params_from_pkl(pkl_path):
        try:
            if not pkl_path:
                return None, None
            pkl_path_str = str(pkl_path)
            if not os.path.exists(pkl_path_str):
                return None, None
            with open(pkl_path_str, 'rb') as f:
                dq = dill.load(f)
            dq = ensure_queue_compatibility(dq)
            items = []
            while not dq.empty():
                try:
                    items.append(dq.get_nowait())
                except _q.Empty:
                    break
            if not items:
                return None, None
            df = _pd.concat(items, ignore_index=True)
            df = normalize_date_column(df)
            feat_df = df.drop(columns=['ts_code', 'Date'], errors='ignore')
            feat_df = feat_df.fillna(feat_df.median(numeric_only=True))
            X = feat_df.values
            means, stds = [], []
            for i in range(X.shape[1]):
                col = _np.nan_to_num(X[:, i], nan=0.0, posinf=0.0, neginf=0.0)
                m = float(_np.mean(col))
                s = float(_np.std(col))
                if not _np.isfinite(m):
                    m = 0.0
                if (not _np.isfinite(s)) or s < 1e-8:
                    s = 1.0
                means.append(m)
                stds.append(s)
            print(f"[LOG] Auto-computed normalization params from PKL: {len(means)} features")
            return means, stds
        except Exception as e:
            print(f"[WARN] Auto-compute norm params from PKL failed: {e}")
            return None, None

    def save_with_args(model_path, optimizer_path, args_path, norm_path):
        # 1) 保存模型与优化器
        torch.save(model_state, model_path)
        torch.save(optimizer_state, optimizer_path)

        # 2) 保存模型初始化参数（若有）
        if model_args is not None:
            try:
                with open(args_path, 'w', encoding='utf-8') as f:
                    json.dump(model_args, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[WARN] Failed to save model args: {e}")

        # 3) 保存归一化参数（稳定副本 → 全局 → 自动计算）
        try:
            use_mean = saved_mean_list if saved_mean_list else mean_list
            use_std = saved_std_list if saved_std_list else std_list

            if (not use_mean or not use_std or len(use_mean) == 0 or len(use_std) == 0):
                auto_mean, auto_std = _compute_norm_params_from_pkl(train_pkl_path)
                if auto_mean and auto_std and len(auto_mean) == len(auto_std) and len(auto_mean) > 0:
                    use_mean, use_std = auto_mean, auto_std
                    try:
                        saved_mean_list.clear(); saved_mean_list.extend(use_mean)  # type: ignore
                        saved_std_list.clear(); saved_std_list.extend(use_std)    # type: ignore
                        print("[LOG] Filled normalization params into saved_mean_list/saved_std_list from PKL")
                    except Exception:
                        pass

            def _backfill_symbol_stats():
                if not os.path.exists(train_pkl_path):
                    return
                try:
                    with open(train_pkl_path, 'rb') as f_pkl:
                        dq = ensure_queue_compatibility(dill.load(f_pkl))
                    items = []
                    while not dq.empty():
                        try:
                            items.append(dq.get_nowait())
                        except _q.Empty:
                            break
                    for item in items:
                        try:
                            sym_series = item.get('ts_code')
                            sym = canonical_symbol(str(sym_series.iloc[0])) if sym_series is not None else None
                        except Exception:
                            sym = None
                        if not sym or sym in symbol_norm_map:
                            continue
                        frame = normalize_date_column(item.copy())
                        try:
                            transformed = feature_engineer.transform(frame)
                        except Exception:
                            transformed = frame
                        numeric = (
                            transformed.select_dtypes(include=[_np.number])
                            .drop(columns=['_symbol_index', 'ts_code'], errors='ignore')
                        )
                        if numeric.empty:
                            continue
                        data_np = _np.nan_to_num(numeric.to_numpy(dtype=_np.float64, copy=True), nan=0.0, posinf=0.0, neginf=0.0)
                        means = [_np.float64(m).item() if _np.isfinite(m) else 0.0 for m in data_np.mean(axis=0)]
                        stds = []
                        for s in data_np.std(axis=0):
                            if not _np.isfinite(s) or s < 1e-8:
                                stds.append(1.0)
                            else:
                                stds.append(_np.float64(s).item())
                        if len(means) == len(stds):
                            symbol_norm_map[sym] = {
                                'mean_list': means,
                                'std_list': stds,
                            }
                except Exception as exc:
                    print(f"[WARN] Failed to backfill per-symbol norm stats: {exc}")

            _backfill_symbol_stats()

            per_symbol_payload = {}
            for sym, stats in symbol_norm_map.items():
                means = stats.get('mean_list')
                stds = stats.get('std_list')
                if means and stds and len(means) == len(stds):
                    per_symbol_payload[sym] = {
                        'mean_list': list(means),
                        'std_list': list(stds),
                    }

            norm_params = {
                'mean_list': list(use_mean),
                'std_list': list(use_std),
                'show_list': list(show_list),
                'name_list': list(name_list),
                'version': 2,
            }
            if per_symbol_payload:
                norm_params['per_symbol'] = per_symbol_payload
            else:
                print('[WARN] per_symbol stats empty; only global normalization will be saved.')
            with open(norm_path, 'w', encoding='utf-8') as f:
                json.dump(norm_params, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"[WARN] Failed to save normalization params: {e}")

    # 目标路径拼装与调用
    if predict_days > 0:
        if best_model is False:
            model_path = save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_pre" + str(predict_days) + "_Model.pkl"
            optimizer_path = save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_pre" + str(predict_days) + "_Optimizer.pkl"
            args_path = save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_pre" + str(predict_days) + "_Model_args.json"
            norm_path = save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_pre" + str(predict_days) + "_norm_params.json"
            save_with_args(model_path, optimizer_path, args_path, norm_path)
        else:
            model_path = save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_pre" + str(predict_days) + "_Model_best.pkl"
            optimizer_path = save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_pre" + str(predict_days) + "_Optimizer_best.pkl"
            args_path = save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_pre" + str(predict_days) + "_Model_best_args.json"
            norm_path = save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_pre" + str(predict_days) + "_norm_params_best.json"
            save_with_args(model_path, optimizer_path, args_path, norm_path)
    else:
        if best_model is False:
            model_path = save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_Model.pkl"
            optimizer_path = save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_Optimizer.pkl"
            args_path = save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_Model_args.json"
            norm_path = save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_norm_params.json"
            save_with_args(model_path, optimizer_path, args_path, norm_path)
        else:
            model_path = save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_Model_best.pkl"
            optimizer_path = save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_Optimizer_best.pkl"
            args_path = save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_Model_best_args.json"
            norm_path = save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_norm_params_best.json"
            save_with_args(model_path, optimizer_path, args_path, norm_path)


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
    """Fill missing attributes on legacy serialized queue.Queue objects so they remain usable on newer Python versions."""
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



PLOT_FEATURE_COLUMNS = ["Open", "High", "Low", "Close"]
PLOT_FEATURE_LABELS = {
    "Open": "Open Price (Open)",
    "High": "High Price (High)",
    "Low": "Low Price (Low)",
    "Close": "Close Price (Close)",
}


def plot_feature_comparison(symbol, model_name, feature, history_series, prediction_series,
                            output_dir, prefix="predict"):
    """
    Plot a single price feature comparing historical actual values with predictions, shared by train/predict.

    Args:
        symbol: Stock symbol string.
        model_name: Name of the current model.
        feature: Feature name to plot (case sensitive, must match the DataFrame column).
        history_series: pandas.Series，index is the date and the value represents the actual observation
        prediction_series: pandas.Series，index is the date and the value represents the forecast
        output_dir: Output directory path (str or Path).
        prefix: Filename prefix to distinguish different call sites.
    """
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        plt.rcParams['font.sans-serif'] = [
            "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans", "sans-serif"
        ]
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass

    # Ensure the index is sorted ascending to avoid inverted line plots.
    history_series = history_series.dropna().sort_index()
    prediction_series = prediction_series.dropna().sort_index()

    if history_series.empty and prediction_series.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, 6))

    if not history_series.empty:
        ax.plot(
            history_series.index,
            history_series.values,
            marker="o",
            markersize=3,
            linewidth=1.0,
            alpha=0.9,
            label="Actual",
        )

    plotted_pred_series = pd.Series(dtype=float)
    if not prediction_series.empty:
        # If predictions start after the last historical point, attach the first forecast to the last actual value for continuity.
        plot_index = prediction_series.index
        plot_values = prediction_series.values
        if not history_series.empty and prediction_series.index[0] > history_series.index[-1]:
            plot_index = prediction_series.index.insert(0, history_series.index[-1])
            plot_values = np.insert(prediction_series.values, 0, history_series.iloc[-1])
        # Keep a copy of what is actually plotted for CSV export
        plotted_pred_series = pd.Series(plot_values, index=plot_index, dtype=float)
        ax.plot(
            plot_index,
            plot_values,
            marker="o",
            markersize=3,
            linestyle="--",
            linewidth=1.0,
            alpha=0.9,
            label="Forecast",
        )

    display_name = PLOT_FEATURE_LABELS.get(feature, feature)
    ax.set_title(f"{symbol} {display_name} Actual vs Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel(display_name)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend()
    fig.autofmt_xdate()

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    safe_symbol = str(symbol).replace(".", "")
    file_name = f"{safe_symbol}_{model_name}_{prefix}_{feature.lower()}_{timestamp}.png"
    save_path = output_dir / file_name
    fig.savefig(save_path, dpi=600)
    plt.close(fig)
    # Also export the data used in the plot to CSV with the same filename (different suffix)
    try:
        hist_df = history_series.rename("Actual").to_frame()
        pred_src = plotted_pred_series if not plotted_pred_series.empty else prediction_series
        pred_df = pred_src.rename("Forecast").to_frame()
        merged = hist_df.join(pred_df, how="outer").sort_index()
        merged.reset_index(inplace=True)
        merged.rename(columns={merged.columns[0]: "Date"}, inplace=True)
        csv_path = save_path.with_suffix('.csv')
        merged.to_csv(csv_path, index=False, encoding='utf-8')
    except Exception as e:
        print(f"[WARN] Failed to export CSV for plot {save_path.name}: {e}")
    return save_path


def normalize_date_column(df, inplace=False):
    """
    Normalize date columns by ensuring a Date field and converting it to pandas datetime.
    Supports inputs where the column is named either Date or trade_date.
    """
    if df is None:
        return None
    target = df if inplace else df.copy()
    if "Date" in target.columns:
        target["Date"] = pd.to_datetime(target["Date"])
    elif "trade_date" in target.columns:
        target["Date"] = pd.to_datetime(target["trade_date"])
    return target


def generate_dates(start_date, num_days):
    start_date = datetime.strptime(start_date, '%Y%m%d')
    if num_days < 0:
        return np.array([(start_date + timedelta(days=i)).strftime('%Y%m%d') for i in range(num_days, 1)])
    else:
        return np.array([(start_date + timedelta(days=i)).strftime('%Y%m%d') for i in range(num_days+1)])


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
