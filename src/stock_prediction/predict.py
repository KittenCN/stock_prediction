#!/usr/bin/env python
# coding: utf-8
import argparse
import copy
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
import sys
import queue

import dill
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from stock_prediction.models import (
    LSTM,
    AttentionLSTM,
    BiLSTM,
    TCN,
    MultiBranchNet,
    TransformerModel,
    CNNLSTM,
    TemporalHybridNet,
    PTFTVSSMEnsemble,
    PTFTVSSMLoss,
)

try:
    from .common import *
    from .config import Config
except ImportError:
    from stock_prediction.common import *
    from stock_prediction.config import Config

# 初始化配置
config = Config()
train_pkl_path = config.train_pkl_path

parser = argparse.ArgumentParser(description="Stock price inference CLI")
parser.add_argument('--model', default="transformer", type=str, help="model name, e.g. lstm / transformer / hybrid / ptft_vssm")
parser.add_argument('--test_code', default="", type=str, help="stock code to predict")
parser.add_argument('--cpu', default=0, type=int, help="set 1 to run on CPU only")
parser.add_argument('--pkl', default=1, type=int, help="whether to use preprocessed pkl data (1 means use)")
parser.add_argument('--pkl_queue', default=1, type=int, help="whether to use pkl queue data")
parser.add_argument('--predict_days', default=0, type=int, help=">0 for interval prediction, <=0 for day-by-day prediction")
parser.add_argument('--api', default="akshare", type=str, help="data source: tushare / akshare / yfinance")
parser.add_argument('--trend', default=0, type=int, help="set 1 to predict trend instead of price")
parser.add_argument('--test_gpu', default=1, type=int, help="set 1 to run inference on GPU")

class DefaultArgs:
    model = "transformer"
    test_code = ""
    cpu = 0
    pkl = 1
    pkl_queue = 1
    predict_days = 0
    api = "akshare"
    trend = 0
    test_gpu = 1

args = DefaultArgs()
model_mode: str | None = None
model: nn.Module | None = None
test_model: nn.Module | None = None
save_path: str | None = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PKL = True
drop_last = False

# 用于测试的全局变量（正常情况下会在运行时从 common.py 设置）
total_test_length = 0


def _init_models(symbol: str) -> None:
    global model_mode, model, test_model, save_path
    model_mode = args.model.upper()
    if model_mode == "LSTM":
        model = LSTM(input_dim=INPUT_DIMENSION)
        test_model = LSTM(input_dim=INPUT_DIMENSION)
        save_path = str(config.get_model_path("LSTM", symbol))
    elif model_mode == "ATTENTION_LSTM":
        model = AttentionLSTM(input_dim=INPUT_DIMENSION, hidden_dim=128, num_layers=2, output_dim=OUTPUT_DIMENSION)
        test_model = AttentionLSTM(input_dim=INPUT_DIMENSION, hidden_dim=128, num_layers=2, output_dim=OUTPUT_DIMENSION)
        save_path = "output/attention_lstm"
    elif model_mode == "BILSTM":
        model = BiLSTM(input_dim=INPUT_DIMENSION, hidden_dim=128, num_layers=2, output_dim=OUTPUT_DIMENSION)
        test_model = BiLSTM(input_dim=INPUT_DIMENSION, hidden_dim=128, num_layers=2, output_dim=OUTPUT_DIMENSION)
        save_path = "output/bilstm"
    elif model_mode == "TCN":
        model = TCN(input_dim=INPUT_DIMENSION, output_dim=OUTPUT_DIMENSION, num_channels=[64, 64, 64], kernel_size=3)
        test_model = TCN(input_dim=INPUT_DIMENSION, output_dim=OUTPUT_DIMENSION, num_channels=[64, 64, 64], kernel_size=3)
        save_path = "output/tcn"
    elif model_mode == "MULTIBRANCH":
        price_dim = INPUT_DIMENSION // 2
        tech_dim = INPUT_DIMENSION - price_dim
        model = MultiBranchNet(price_dim=price_dim, tech_dim=tech_dim, hidden_dim=64, output_dim=OUTPUT_DIMENSION)
        test_model = MultiBranchNet(price_dim=price_dim, tech_dim=tech_dim, hidden_dim=64, output_dim=OUTPUT_DIMENSION)
        save_path = "output/multibranch"
    elif model_mode == "TRANSFORMER":
        model = TransformerModel(input_dim=INPUT_DIMENSION, d_model=D_MODEL, nhead=NHEAD, num_layers=6,
                                 dim_feedforward=2048, output_dim=OUTPUT_DIMENSION, max_len=SEQ_LEN, mode=0)
        test_model = TransformerModel(input_dim=INPUT_DIMENSION, d_model=D_MODEL, nhead=NHEAD, num_layers=6,
                                      dim_feedforward=2048, output_dim=OUTPUT_DIMENSION, max_len=SEQ_LEN, mode=1)
        save_path = config.get_model_path("TRANSFORMER", symbol)
    elif model_mode == "HYBRID":
        hybrid_steps = abs(int(args.predict_days)) if int(args.predict_days) > 0 else 1
        model = TemporalHybridNet(input_dim=INPUT_DIMENSION, output_dim=OUTPUT_DIMENSION, predict_steps=hybrid_steps)
        test_model = TemporalHybridNet(input_dim=INPUT_DIMENSION, output_dim=OUTPUT_DIMENSION, predict_steps=hybrid_steps)
        save_path = config.get_model_path("HYBRID", symbol)
    elif model_mode == "PTFT_VSSM":
        ensemble_steps = abs(int(args.predict_days)) if int(args.predict_days) > 0 else 1
        model = PTFTVSSMEnsemble(input_dim=INPUT_DIMENSION, output_dim=OUTPUT_DIMENSION, predict_steps=ensemble_steps)
        test_model = PTFTVSSMEnsemble(input_dim=INPUT_DIMENSION, output_dim=OUTPUT_DIMENSION, predict_steps=ensemble_steps)
        save_path = config.get_model_path("PTFT_VSSM", symbol)
    elif model_mode == "CNNLSTM":
        predict_days = max(1, abs(int(args.predict_days)))
        model = CNNLSTM(input_dim=INPUT_DIMENSION, num_classes=OUTPUT_DIMENSION, predict_days=predict_days)
        test_model = CNNLSTM(input_dim=INPUT_DIMENSION, num_classes=OUTPUT_DIMENSION, predict_days=predict_days)
        save_path = config.get_model_path("CNNLSTM", symbol)
    else:
        raise ValueError(f"Unsupported model: {model_mode}")
    model.to(device)
    test_model.to(device if args.test_gpu == 1 else torch.device("cpu"))


def test(dataset, testmodel=None, dataloader_mode=0):
    global test_model
    predict_list = []
    accuracy_list = []
    if dataloader_mode in [0, 2]:
        stock_predict = Stock_Data(mode=dataloader_mode, dataFrame=dataset, label_num=OUTPUT_DIMENSION,
                                   predict_days=int(args.predict_days), trend=int(args.trend))
        dataloader = DataLoader(dataset=stock_predict, batch_size=BATCH_SIZE, shuffle=False,
                                drop_last=drop_last, num_workers=NUM_WORKERS, pin_memory=True)
    elif dataloader_mode in [1]:
        _stock_test_data_queue = deep_copy_queue(dataset)
        total_test_length = _stock_test_data_queue.qsize()
        stock_test = stock_queue_dataset(mode=1, data_queue=_stock_test_data_queue, label_num=OUTPUT_DIMENSION,
                                         buffer_size=BUFFER_SIZE, total_length=total_test_length,
                                         predict_days=int(args.predict_days), trend=int(args.trend))
        dataloader = DataLoader(dataset=stock_test, batch_size=BATCH_SIZE, shuffle=False, drop_last=drop_last,
                                num_workers=NUM_WORKERS, pin_memory=True, collate_fn=custom_collate)
    else:
        raise ValueError("Unsupported dataloader mode")

    if testmodel is None:
        if int(args.predict_days) > 0:
            model_path = f"{save_path}_out{OUTPUT_DIMENSION}_time{SEQ_LEN}_pre{args.predict_days}_Model.pkl"
            if not os.path.exists(model_path):
                raise FileNotFoundError(model_path)
            test_model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            model_path = f"{save_path}_out{OUTPUT_DIMENSION}_time{SEQ_LEN}_Model.pkl"
            if not os.path.exists(model_path):
                raise FileNotFoundError(model_path)
            test_model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        test_model = testmodel

    test_model.eval()
    criterion = nn.MSELoss()
    pbar = tqdm(total=len(dataloader), leave=False, ncols=TQDM_NCOLS)
    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                pbar.update(1)
                continue
            data, label = batch
            device_target = device if args.test_gpu == 1 else torch.device("cpu")
            data = data.to(device_target, non_blocking=True)
            label = label.to(device_target, non_blocking=True)
            data = pad_input(data)
            if model_mode == "MULTIBRANCH":
                price_dim = INPUT_DIMENSION // 2
                tech_dim = INPUT_DIMENSION - price_dim
                predict = test_model(price_x=data[:, :, :price_dim], tech_x=data[:, :, price_dim:])
            elif model_mode == "TRANSFORMER":
                predict = test_model(data, label, int(args.predict_days))
            else:
                predict = test_model(data)
            predict_list.append(predict.to('cpu'))
            if predict.shape == label.shape:
                loss = criterion(predict, label)
                if torch.isfinite(loss):
                    accuracy_list.append(loss.item())
            pbar.update(1)
    pbar.close()
    if not accuracy_list:
        accuracy_list = [0]
    return float(np.mean(accuracy_list)), predict_list, dataloader


def predict(test_codes):
    print(f"[LOG] predict() called with test_codes={test_codes}")
    if not test_codes:
        print("[LOG] test_codes is empty, abort.")
        raise ValueError("test_codes is empty")
    if PKL == 0:
        print("[LOG] Using CSV data loading mode.")
        load_data(test_codes, data_queue=data_queue)
        try:
            data = data_queue.get(timeout=30)
        except queue.Empty:
            print("[LOG] data_queue is empty after load_data.")
            raise RuntimeError("data_queue is empty")
    else:
        print(f"[LOG] Using PKL mode, loading from {train_pkl_path}")
        with open(train_pkl_path, 'rb') as f:
            data_queue = ensure_queue_compatibility(dill.load(f))
        data = None
        while not data_queue.empty():
            item = data_queue.get()
            print(f"[LOG] Checking ts_code in pkl: {str(item['ts_code'].iloc[0]).zfill(6)}")
            if str(item['ts_code'].iloc[0]).zfill(6) == str(test_codes[0]):
                data = copy.deepcopy(item)
                print(f"[LOG] Found target ts_code: {test_codes[0]}, shape={data.shape}")
                break
        if data is None:
            print(f"[LOG] 目标股票 {test_codes[0]} 未在 pkl 队列中找到")
            raise RuntimeError("目标股票未在 pkl 队列中找到")

    if data.empty or data['ts_code'].iloc[0] == "None":
        print(f"[LOG] 数据为空或 ts_code 无效, data.empty={data.empty}, ts_code={data['ts_code'].iloc[0]}")
        raise RuntimeError("数据为空或 ts_code 无效")

    predict_data = copy.deepcopy(data)
    spliced_data = copy.deepcopy(data)
    print(f"[LOG] predict_data.columns: {list(predict_data.columns)}")
    print(f"[LOG] predict_data.shape: {predict_data.shape}")
    if int(args.predict_days) <= 0:
        predict_days = abs(int(args.predict_days)) or 1
        print(f"[LOG] predict_days={predict_days}")
        pbar = tqdm(total=predict_days, leave=False, ncols=TQDM_NCOLS)
        while predict_days > 0:
            predict_days -= 1
            lastdate = predict_data['Date'].iloc[0].strftime('%Y%m%d')
            print(f"[LOG] lastdate={lastdate}")
            predict_data.drop(['ts_code', 'Date'], axis=1, inplace=True)
            predict_data = predict_data.fillna(predict_data.median(numeric_only=True))
            _, predict_list, _ = test(predict_data, dataloader_mode=2)
            print(f"[LOG] predict_list len: {len(predict_list)}")
            rows = []
            for items in predict_list:
                items = items.to('cpu')
                for idxs in items:
                    row = []
                    for index, item in enumerate(idxs):
                        if use_list[index] == 1:
                            row.append((item * std_list[index] + mean_list[index]).detach().numpy())
                    rows.append(row)
            print(f"[LOG] rows[0] len: {len(rows[0]) if rows else 0}")
            date_obj = datetime.strptime(lastdate, '%Y%m%d') + timedelta(days=1)
            new_row = [test_codes[0], date_obj]
            if rows:
                new_row.extend(rows[0])
            print(f"[LOG] new_row len: {len(new_row)}, columns: {len(spliced_data.columns)}")
            # 补齐 new_row 长度到 columns 数量
            while len(new_row) < len(spliced_data.columns):
                new_row.append(0.0)
            # 截断多余
            new_row = new_row[:len(spliced_data.columns)]
            new_df = pd.DataFrame([new_row], columns=spliced_data.columns)
            print(f"[LOG] Saving predict image, DataFrame shape: {new_df.shape}")
            # 保存图片前加日志
            try:
                # 只画 open/high/low/close 四个字段，分别生成四张图片
                save_dir = root_dir / "png" / "predict"
                save_dir.mkdir(parents=True, exist_ok=True)
                plot_cols = ["Open", "High", "Low", "Close"]
                for col in plot_cols:
                    if col in new_df.columns:
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.plot(new_df["Date"], new_df[col], marker='o', label=col)
                        ax.set_title(f"预测结果: {test_codes[0]} {col} {date_obj.strftime('%Y-%m-%d')}")
                        ax.set_xlabel("日期")
                        ax.set_ylabel(f"{col} 预测值")
                        ax.legend()
                        img_name = f"{test_codes[0]}_{col}_{date_obj.strftime('%Y%m%d')}_Pre.png"
                        img_path = save_dir / img_name
                        plt.tight_layout()
                        plt.savefig(img_path, dpi=600)
                        plt.close(fig)
                        print(f"[LOG] 图片已保存: {img_path}")
            except Exception as e:
                print(f"[LOG] Error saving image: {e}")
            predict_data = pd.concat([new_df, spliced_data], ignore_index=True)
            spliced_data = copy.deepcopy(predict_data)
            predict_data['Date'] = pd.to_datetime(predict_data['Date'])
            pbar.update(1)
        pbar.close()
    else:
        predict_data.drop(['ts_code', 'Date'], axis=1, inplace=True)
        predict_data = predict_data.fillna(predict_data.median(numeric_only=True))
        test_loss, predict_list, _ = test(predict_data, dataloader_mode=2)
        print("test loss:", test_loss)
        return predict_list


def main(argv=None):
    global args, PKL, drop_last, device
    args = parser.parse_args(argv)
    if args.cpu == 1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    PKL = False if args.pkl <= 0 else True
    drop_last = False
    if not args.test_code:
        raise ValueError('test_code 参数不能为空')
    # 修复：使用 symbol 作为模型路径，而不是 test_code
    _init_models(symbol)
    predict([args.test_code])


class Predictor:
    def __init__(self, model_type='lstm', device_type='cpu'):
        self.args = argparse.Namespace(**vars(DefaultArgs()))
        self.args.model = model_type
        self.args.cpu = 1 if device_type.lower() == 'cpu' else 0

    def predict(self, test_code: str):
        global args
        self.args.test_code = test_code
        args = self.args
        _init_models(test_code)
        return predict([test_code])


def create_predictor(model_type='lstm', device_type='cpu'):
    return Predictor(model_type, device_type)


if __name__ == '__main__':
    main()