#!/usr/bin/env python
# coding: utf-8
import argparse
import copy
import os
import random
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
    DiffusionForecaster,
    GraphTemporalModel,
)
try:
    from .common import *
except ImportError:
    from stock_prediction.common import *

# Load shared configuration
from stock_prediction.app_config import AppConfig
config = AppConfig.from_env_and_yaml(str(root_dir / 'config' / 'config.yaml'))
train_pkl_path = config.train_pkl_path
png_path = config.png_path
model_path = config.model_path
feature_settings = getattr(config, "features", None)
SYMBOL_EMBED_ENABLED = bool(getattr(feature_settings, "use_symbol_embedding", False))
SYMBOL_EMBED_DIM = int(getattr(feature_settings, "symbol_embedding_dim", 16))
SYMBOL_EMBED_MAX = int(os.getenv("SYMBOL_EMBED_MAX", "4096"))
_symbol_vocab = len(feature_engineer.get_symbol_indices()) if SYMBOL_EMBED_ENABLED else 0
if SYMBOL_EMBED_ENABLED:
    SYMBOL_VOCAB_SIZE = _symbol_vocab if _symbol_vocab > 0 else SYMBOL_EMBED_MAX
    SYMBOL_VOCAB_SIZE = min(max(SYMBOL_VOCAB_SIZE, 1), SYMBOL_EMBED_MAX)
else:
    SYMBOL_VOCAB_SIZE = max(_symbol_vocab, 1)

# Load symbol mapping for embedding
import dill
train_data = dill.load(open(train_pkl_path, 'rb'))
if isinstance(train_data, queue.Queue):
    temp_queue = deep_copy_queue(train_data)
    all_data = []
    while not temp_queue.empty():
        item = temp_queue.get()
        all_data.append(item)
    if all_data:
        train_df = pd.concat(all_data, ignore_index=True)
        unique_symbols = train_df['ts_code'].unique()
        symbol_to_id = {symbol: i for i, symbol in enumerate(unique_symbols)}
    else:
        symbol_to_id = {}
else:
    unique_symbols = train_data['ts_code'].unique()
    symbol_to_id = {symbol: i for i, symbol in enumerate(unique_symbols)}

parser = argparse.ArgumentParser(description="Stock price inference CLI")
parser.add_argument('--model', default="hybrid", type=str, help="model name, e.g. lstm / transformer / hybrid / ptft_vssm / diffusion / graph")
parser.add_argument('--test_code', default="", type=str, help="stock code to predict")
parser.add_argument('--cpu', default=0, type=int, help="set 1 to run on CPU only")
parser.add_argument('--pkl', default=1, type=int, help="whether to use preprocessed pkl data (1 means use)")
parser.add_argument('--pkl_queue', default=1, type=int, help="whether to use pkl queue data")
parser.add_argument('--predict_days', default=0, type=int, help=">0 for interval prediction, <=0 for day-by-day prediction")
parser.add_argument('--api', default="akshare", type=str, help="data source: tushare / akshare / yfinance")
parser.add_argument('--trend', default=0, type=int, help="set 1 to predict trend instead of price")
parser.add_argument('--test_gpu', default=1, type=int, help="set 1 to run inference on GPU")

class DefaultArgs:
    model = "hybrid"
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

# Provide global handles for tests when common.py is unavailable
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
        model = TemporalHybridNet(
            input_dim=INPUT_DIMENSION,
            output_dim=OUTPUT_DIMENSION,
            predict_steps=hybrid_steps,
            use_symbol_embedding=SYMBOL_EMBED_ENABLED,
            symbol_embedding_dim=SYMBOL_EMBED_DIM,
            max_symbols=SYMBOL_VOCAB_SIZE,
        )
        test_model = TemporalHybridNet(
            input_dim=INPUT_DIMENSION,
            output_dim=OUTPUT_DIMENSION,
            predict_steps=hybrid_steps,
            use_symbol_embedding=SYMBOL_EMBED_ENABLED,
            symbol_embedding_dim=SYMBOL_EMBED_DIM,
            max_symbols=SYMBOL_VOCAB_SIZE,
        )
        save_path = config.get_model_path("HYBRID", symbol)
    elif model_mode == "PTFT_VSSM":
        ensemble_steps = abs(int(args.predict_days)) if int(args.predict_days) > 0 else 1
        model = PTFTVSSMEnsemble(
            input_dim=INPUT_DIMENSION,
            output_dim=OUTPUT_DIMENSION,
            predict_steps=ensemble_steps,
            use_symbol_embedding=SYMBOL_EMBED_ENABLED,
            symbol_embedding_dim=SYMBOL_EMBED_DIM,
            max_symbols=SYMBOL_VOCAB_SIZE,
        )
        test_model = PTFTVSSMEnsemble(
            input_dim=INPUT_DIMENSION,
            output_dim=OUTPUT_DIMENSION,
            predict_steps=ensemble_steps,
            use_symbol_embedding=SYMBOL_EMBED_ENABLED,
            symbol_embedding_dim=SYMBOL_EMBED_DIM,
            max_symbols=SYMBOL_VOCAB_SIZE,
        )
        save_path = config.get_model_path("PTFT_VSSM", symbol)
    elif model_mode == "DIFFUSION":
        diffusion_steps = abs(int(args.predict_days)) if int(args.predict_days) > 0 else 1
        model = DiffusionForecaster(
            input_dim=INPUT_DIMENSION,
            output_dim=OUTPUT_DIMENSION,
            predict_steps=diffusion_steps,
            use_symbol_embedding=SYMBOL_EMBED_ENABLED,
            symbol_embedding_dim=SYMBOL_EMBED_DIM,
            max_symbols=SYMBOL_VOCAB_SIZE,
        )
        test_model = DiffusionForecaster(
            input_dim=INPUT_DIMENSION,
            output_dim=OUTPUT_DIMENSION,
            predict_steps=diffusion_steps,
            use_symbol_embedding=SYMBOL_EMBED_ENABLED,
            symbol_embedding_dim=SYMBOL_EMBED_DIM,
            max_symbols=SYMBOL_VOCAB_SIZE,
        )
        save_path = str(config.get_model_path("DIFFUSION", symbol))
    elif model_mode == "GRAPH":
        graph_steps = abs(int(args.predict_days)) if int(args.predict_days) > 0 else 1
        model = GraphTemporalModel(
            input_dim=INPUT_DIMENSION,
            output_dim=OUTPUT_DIMENSION,
            predict_steps=graph_steps,
            use_symbol_embedding=SYMBOL_EMBED_ENABLED,
            symbol_embedding_dim=SYMBOL_EMBED_DIM,
            max_symbols=SYMBOL_VOCAB_SIZE,
        )
        test_model = GraphTemporalModel(
            input_dim=INPUT_DIMENSION,
            output_dim=OUTPUT_DIMENSION,
            predict_steps=graph_steps,
            use_symbol_embedding=SYMBOL_EMBED_ENABLED,
            symbol_embedding_dim=SYMBOL_EMBED_DIM,
            max_symbols=SYMBOL_VOCAB_SIZE,
        )
        save_path = str(config.get_model_path("GRAPH", symbol))
    elif model_mode == "CNNLSTM":
        predict_days = max(1, abs(int(args.predict_days)))
        model = CNNLSTM(input_dim=INPUT_DIMENSION, num_classes=OUTPUT_DIMENSION, predict_days=predict_days)
        test_model = CNNLSTM(input_dim=INPUT_DIMENSION, num_classes=OUTPUT_DIMENSION, predict_days=predict_days)
        save_path = config.get_model_path("CNNLSTM", symbol)
    else:
        raise ValueError(f"Unsupported model: {model_mode}")
    model.to(device)
    test_model.to(device if args.test_gpu == 1 else torch.device("cpu"))


def test(dataset, testmodel=None, dataloader_mode=0, symbol_index=None):
    global test_model
    predict_list = []
    accuracy_list = []
    use_gpu = device.type == "cuda" and getattr(args, "test_gpu", 1) == 1 and torch.cuda.is_available()
    pin_memory = use_gpu
    if dataloader_mode in [0, 2]:
        stock_predict = Stock_Data(mode=dataloader_mode, dataFrame=dataset, label_num=OUTPUT_DIMENSION,
                                   predict_days=int(args.predict_days), trend=int(args.trend))
        dataloader = DataLoader(dataset=stock_predict, batch_size=BATCH_SIZE, shuffle=False,
                                drop_last=drop_last, num_workers=NUM_WORKERS, pin_memory=pin_memory)
    elif dataloader_mode in [1]:
        _stock_test_data_queue = deep_copy_queue(dataset)
        total_test_length = _stock_test_data_queue.qsize()
        stock_test = stock_queue_dataset(mode=1, data_queue=_stock_test_data_queue, label_num=OUTPUT_DIMENSION,
                                         buffer_size=BUFFER_SIZE, total_length=total_test_length,
                                         predict_days=int(args.predict_days), trend=int(args.trend))
        dataloader = DataLoader(dataset=stock_test, batch_size=BATCH_SIZE, shuffle=False, drop_last=drop_last,
                                num_workers=NUM_WORKERS, pin_memory=pin_memory, collate_fn=custom_collate)
    else:
        raise ValueError("Unsupported dataloader mode")

    import json
    if testmodel is None:
        predict_days = int(args.predict_days)
        ckpt_prefix = f"{save_path}_out{OUTPUT_DIMENSION}_time{SEQ_LEN}"
        if predict_days > 0:
            ckpt_prefix += f"_pre{predict_days}"
        candidates = [
            f"{ckpt_prefix}_Model.pkl",
            f"{ckpt_prefix}_Model_best.pkl",
        ]
        args_candidates = [
            f"{ckpt_prefix}_Model_args.json",
            f"{ckpt_prefix}_Model_best_args.json",
        ]
        # 1. 自动读取参数 json
        model_args = None
        for args_path in args_candidates:
            if os.path.exists(args_path):
                try:
                    with open(args_path, 'r', encoding='utf-8') as f:
                        model_args = json.load(f)
                    print(f"[INFO] Loaded model args from {args_path}")
                    break
                except Exception as e:
                    print(f"[WARN] Failed to load model args: {e}")
        # 2. 支持手动覆盖（命令行参数优先）
        hybrid_size = getattr(args, "hybrid_size", None)
        if model_mode == "HYBRID":
            # 优先用 json 参数，其次命令行参数
            from stock_prediction.hybrid_config import get_adaptive_hybrid_config
            if model_args is not None:
                # json 优先
                hybrid_config = get_adaptive_hybrid_config(size_hint=model_args.get("hybrid_size", "auto"), data_size=0)
                test_model = TemporalHybridNet(
                    input_dim=model_args.get("input_dim", INPUT_DIMENSION),
                    output_dim=model_args.get("output_dim", OUTPUT_DIMENSION),
                    hidden_dim=model_args.get("hidden_dim", hybrid_config["hidden_dim"]),
                    predict_steps=model_args.get("predict_steps", int(args.predict_days)),
                    branch_config=model_args.get("branch_config", hybrid_config["branch_config"]),
                    use_symbol_embedding=model_args.get("use_symbol_embedding", SYMBOL_EMBED_ENABLED),
                    symbol_embedding_dim=model_args.get("symbol_embedding_dim", SYMBOL_EMBED_DIM),
                    max_symbols=model_args.get("max_symbols", SYMBOL_VOCAB_SIZE),
                )
            elif hybrid_size is not None:
                # 命令行覆盖
                hybrid_config = get_adaptive_hybrid_config(size_hint=hybrid_size, data_size=0)
                test_model = TemporalHybridNet(
                    input_dim=INPUT_DIMENSION,
                    output_dim=OUTPUT_DIMENSION,
                    hidden_dim=hybrid_config["hidden_dim"],
                    predict_steps=int(args.predict_days),
                    branch_config=hybrid_config["branch_config"],
                    use_symbol_embedding=SYMBOL_EMBED_ENABLED,
                    symbol_embedding_dim=SYMBOL_EMBED_DIM,
                    max_symbols=SYMBOL_VOCAB_SIZE,
                )
            else:
                # 默认 auto
                hybrid_config = get_adaptive_hybrid_config(size_hint="auto", data_size=0)
                test_model = TemporalHybridNet(
                    input_dim=INPUT_DIMENSION,
                    output_dim=OUTPUT_DIMENSION,
                    hidden_dim=hybrid_config["hidden_dim"],
                    predict_steps=int(args.predict_days),
                    branch_config=hybrid_config["branch_config"],
                    use_symbol_embedding=SYMBOL_EMBED_ENABLED,
                    symbol_embedding_dim=SYMBOL_EMBED_DIM,
                    max_symbols=SYMBOL_VOCAB_SIZE,
                )
        # 3. 加载权重
        loaded = False
        for candidate in candidates:
            if os.path.exists(candidate):
                test_model.load_state_dict(torch.load(candidate, map_location=device))
                loaded = True
                break
        if not loaded:
            raise FileNotFoundError(candidates[0])
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
            if isinstance(batch, (list, tuple)):
                if len(batch) == 3:
                    data, label, symbol_idx = batch
                else:
                    data, label = batch[0], batch[1]
                    symbol_idx = None
            else:
                data, label = batch
                symbol_idx = None
            device_target = device if args.test_gpu == 1 else torch.device("cpu")
            data = data.to(device_target, non_blocking=True)
            label = label.to(device_target, non_blocking=True)
            if symbol_idx is not None:
                symbol_idx = symbol_idx.to(device_target, non_blocking=True).long()
            data = pad_input(data)
            if model_mode == "MULTIBRANCH":
                price_dim = INPUT_DIMENSION // 2
                tech_dim = INPUT_DIMENSION - price_dim
                predict = test_model(price_x=data[:, :, :price_dim], tech_x=data[:, :, price_dim:])
            elif model_mode == "TRANSFORMER":
                predict = test_model(data, label, int(args.predict_days))
            else:
                if symbol_index is not None:
                    predict = test_model(data, symbol_index=symbol_index)
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
            print(f"[LOG] Target symbol {test_codes[0]} not found inside pkl queue")
            raise RuntimeError("Target symbol missing from preprocessed queue")

    if data.empty or data['ts_code'].iloc[0] == "None":
        print(f"[LOG] Data is empty or ts_code invalid, data.empty={data.empty}, ts_code={data['ts_code'].iloc[0]}")
        raise RuntimeError("Data is empty or ts_code is invalid")

    data = normalize_date_column(data)
    predict_data = normalize_date_column(copy.deepcopy(data))
    spliced_data = normalize_date_column(copy.deepcopy(data))
    print(f"[LOG] predict_data.columns: {list(predict_data.columns)}")
    print(f"[LOG] predict_data.shape: {predict_data.shape}")

    # Prepare symbol index for embedding
    symbol_index = None
    if SYMBOL_EMBED_ENABLED:
        symbol_index = torch.tensor([symbol_to_id.get(test_codes[0], 0)])
    if int(args.predict_days) <= 0:
        predict_days = abs(int(args.predict_days)) or 1
        print(f"[LOG] predict_days={predict_days}")
        pbar = tqdm(total=predict_days, leave=False, ncols=TQDM_NCOLS)
        predicted_rows: list[dict] = []
        history_window = max(SEQ_LEN * 8, 40)
        while predict_days > 0:
            predict_days -= 1
            lastdate = pd.to_datetime(predict_data['Date'].iloc[0])
            print(f"[LOG] lastdate={lastdate.strftime('%Y%m%d')}")

            normalized_predict = normalize_date_column(predict_data)
            features_df = normalized_predict.drop(columns=['ts_code', 'Date']).copy()
            features_df = features_df.fillna(features_df.median(numeric_only=True))
            _, predict_list, _ = test(features_df, dataloader_mode=2, symbol_index=symbol_index)
            print(f"[LOG] predict_list len: {len(predict_list)}")

            rows = []
            for items in predict_list:
                items = items.to('cpu')
                for idxs in items:
                    row = []
                    for index, item in enumerate(idxs):
                        if use_list[index] == 1:
                            row.append(float((item * std_list[index] + mean_list[index]).detach().numpy()))
                    if row:
                        rows.append(row)

            if not rows:
                print("[LOG] No valid prediction results, ending early.")
            date_obj = lastdate + timedelta(days=1)
            new_row = [test_codes[0], date_obj]
            if rows:
                new_row.extend(rows[0])
            while len(new_row) < len(spliced_data.columns):
                new_row.append(0.0)
            new_row = new_row[:len(spliced_data.columns)]
            new_df = pd.DataFrame([new_row], columns=spliced_data.columns)
            predicted_rows.append(new_df.iloc[0].to_dict())

            predict_data = pd.concat([new_df, spliced_data], ignore_index=True)
            spliced_data = normalize_date_column(copy.deepcopy(predict_data))
            predict_data['Date'] = pd.to_datetime(predict_data['Date'])
            pbar.update(1)
        pbar.close()

        # Render open/high/low/close comparison charts
        full_df = spliced_data.copy()
        full_df['Date'] = pd.to_datetime(full_df['Date'])
        predicted_df = pd.DataFrame(predicted_rows)
        if not predicted_df.empty:
            predicted_df['Date'] = pd.to_datetime(predicted_df['Date'])

        history_df = full_df.iloc[len(predicted_rows):len(predicted_rows) + history_window].copy()
        history_df['Date'] = pd.to_datetime(history_df['Date'])

        symbol_code = str(test_codes[0]).split('.')[0].zfill(6)
        for col in PLOT_FEATURE_COLUMNS:
            if col not in history_df.columns:
                continue
            history_series = history_df.set_index('Date')[col].astype(float).dropna()
            prediction_series = pd.Series(dtype=float)
            if not predicted_df.empty and col in predicted_df.columns:
                prediction_series = predicted_df.set_index('Date')[col].astype(float).dropna()
            plot_feature_comparison(
                symbol_code,
                model_mode,
                col,
                history_series,
                prediction_series,
                Path(png_path) / "predict",
                prefix="predict",
            )
    else:
        normalized_predict = normalize_date_column(predict_data)
        feature_frame = normalized_predict.drop(columns=['ts_code', 'Date']).copy()
        feature_frame = feature_frame.fillna(feature_frame.median(numeric_only=True))
        test_loss, predict_list, _ = test(feature_frame, dataloader_mode=2, symbol_index=symbol_index)
        print("test loss:", test_loss)

        predictions = []
        for items in predict_list:
            items = items.to("cpu")
            for idxs in items:
                row = []
                for index, item in enumerate(idxs):
                    if show_list[index] == 1:
                        row.append(float(item * std_list[index] + mean_list[index]))
                if row:
                    predictions.append(row)

        selected_features = [name_list[idx] for idx, flag in enumerate(show_list) if flag == 1]
        rename_map = {"open": "Open", "high": "High", "low": "Low", "close": "Close"}
        pred_columns = [rename_map.get(name, name.title()) for name in selected_features]
        pred_df = pd.DataFrame(predictions, columns=pred_columns)
        actual_df = normalize_date_column(predict_data).head(len(pred_df))
        actual_df['Date'] = pd.to_datetime(actual_df['Date'])
        pred_df['Date'] = actual_df['Date'].iloc[:len(pred_df)].values

        symbol_code = str(test_codes[0]).split('.')[0].zfill(6)
        import json
        from datetime import datetime
        from stock_prediction.metrics import metrics_report
        # 采集并保存指标
        metrics_out = {}
        for col in PLOT_FEATURE_COLUMNS:
            if col not in actual_df.columns:
                actual_df[col] = np.nan
            history_series = actual_df.set_index('Date')[col].astype(float).dropna()
            prediction_series = pd.Series(dtype=float)
            if col in pred_df.columns:
                prediction_series = pred_df.set_index('Date')[col].astype(float).dropna()
            # 绘图
            plot_feature_comparison(
                symbol_code,
                model_mode,
                col,
                history_series,
                prediction_series,
                Path(png_path) / "predict",
                prefix="predict",
            )
            # 指标采集
            if len(history_series) == len(prediction_series) and len(history_series) > 0:
                metrics_out[col] = metrics_report(history_series.values, prediction_series.values)
        # 保存指标到 output/metrics_xxx.json
        if metrics_out:
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d%H%M%S")
            metrics_path = output_dir / f"metrics_{symbol_code}_{model_mode}_{ts}.json"
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics_out, f, ensure_ascii=False, indent=2)
            print(f"[LOG] Metrics saved: {metrics_path}")
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
        fallback_path = root_dir / 'test_codes.txt'
        candidate_code = None
        if fallback_path.exists():
            try:
                lines = fallback_path.read_text(encoding='utf-8').splitlines()
            except UnicodeDecodeError:
                lines = fallback_path.read_text(encoding='gbk', errors='ignore').splitlines()
            candidates = [line.strip() for line in lines if line.strip()]
            if candidates:
                candidate_code = random.choice(candidates)
                print(f'[LOG] test_code not provided; randomly selected {candidate_code}')
        if candidate_code is None:
            raise ValueError('test_code is empty and no valid entries found in test_codes.txt')
        args.test_code = candidate_code
    # �޸���ʹ�� symbol ��Ϊģ��·���������� test_code
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
