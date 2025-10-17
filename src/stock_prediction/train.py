#!/usr/bin/env python
# coding: utf-8

import random
import argparse
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import os
import sys
from pathlib import Path

# Ensure the stock_prediction package is importable before importing internal modules
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import torch
from stock_prediction.trainer import Trainer, EarlyStopping, EarlyStoppingConfig

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
    HybridLoss,
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

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default="train", type=str, help="select running mode: train, test, predict")
parser.add_argument('--model', default="hybrid", type=str, help="available model names (e.g. lstm, transformer, hybrid, ptft_vssm, diffusion, graph)")
parser.add_argument('--begin_code', default="", type=str, help="begin code")
parser.add_argument('--cpu', default=0, type=int, help="only use cpu")
parser.add_argument('--pkl', default=1, type=int, help="use pkl file instead of csv file")
parser.add_argument('--pkl_queue', default=1, type=int, help="use pkl queue instead of csv file")
parser.add_argument('--test_code', default="", type=str, help="test code")
parser.add_argument('--test_gpu', default=1, type=int, help="test method use gpu or not")
parser.add_argument('--predict_days', default=0, type=int, help="number of the predict days,Positive numbers use interval prediction algorithm, 0 and negative numbers use date prediction algorithm")
parser.add_argument('--api', default="akshare", type=str, help="api-interface, tushare, akshare or yfinance")
parser.add_argument('--trend', default=0, type=int, help="predict the trend of stock, not the price")
parser.add_argument('--epoch', default=5, type=int, help="training epochs")
parser.add_argument('--plot_days', default=30, type=int, help="history days to display in test/predict plots")
parser.add_argument('--full_train', default=1, type=int, help="train on full dataset without validation/test (1 to enable)")

# Default args reused by tests and direct imports
class DefaultArgs:
    mode = "train"
    model = "hybrid"
    begin_code = ""
    cpu = 0
    pkl = 1
    pkl_queue = 1
    test_code = ""
    test_gpu = 1
    predict_days = 0
    api = "akshare"
    trend = 0
    epoch = 5
    plot_days = 30
    full_train = 1

args = DefaultArgs()

# Initialise module-level state
last_save_time = 0
iteration = 0
batch_none = 0
data_none = 0
loss = None
lo_list = []
loss_list = []
last_loss = 1e10
lr_scheduler = None

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True


# --- Trainer integration ---
def build_scheduler(cfg, optimizer):
    if cfg.scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.scheduler_step_size, gamma=cfg.scheduler_gamma)
    elif cfg.scheduler_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=cfg.early_stopping_patience, factor=cfg.scheduler_gamma)
    else:
        return None

def build_early_stopping(cfg):
    return EarlyStopping(EarlyStoppingConfig(patience=cfg.early_stopping_patience, min_delta=cfg.early_stopping_min_delta, mode="min"))

def save_best_callback(context):
    # context: {"epoch", "best", "train_loss", "val_loss"}
    thread_save_model(model, optimizer, save_path, True, int(args.predict_days))
    with open('loss.txt', 'w') as file:
        file.write(str(context["best"]))

def train_with_trainer(train_loader, val_loader=None, epoch_count=1):
    scheduler = build_scheduler(config, optimizer)
    early_stopping = build_early_stopping(config)
    epoch_count = max(1, int(epoch_count))
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        early_stopping=early_stopping,
        epoch_count=epoch_count,
        scaler=None,
        use_amp=False,
        callbacks={"on_improve": save_best_callback},
        show_progress=True,
    )
    history = trainer.fit()
    train_with_trainer.last_history = history  # 挂载 loss 记录，供全局访问
    return history


def test(dataset, testmodel=None, dataloader_mode=0):
    if getattr(args, "full_train", False):
        return -1, -1, None
    global drop_last, total_test_length
    global test_model
    predict_list = []
    accuracy_list = []
    use_gpu = device.type == "cuda" and getattr(args, "test_gpu", 1) == 1 and torch.cuda.is_available()
    pin_memory = use_gpu
    if dataloader_mode in [0, 2]:
        stock_predict = Stock_Data(mode=dataloader_mode, dataFrame=dataset, label_num=OUTPUT_DIMENSION,predict_days=int(args.predict_days),trend=int(args.trend))
        dataloader = DataLoader(dataset=stock_predict, batch_size=BATCH_SIZE, shuffle=False, drop_last=drop_last, num_workers=NUM_WORKERS, pin_memory=pin_memory)
    elif dataloader_mode in [1]:
        _stock_test_data_queue = deep_copy_queue(dataset)
        stock_test = stock_queue_dataset(mode=1, data_queue=_stock_test_data_queue, label_num=OUTPUT_DIMENSION, buffer_size=BUFFER_SIZE, total_length=total_test_length,predict_days=int(args.predict_days),trend=int(args.trend))
        dataloader=DataLoader(dataset=stock_test,batch_size=BATCH_SIZE,shuffle=False,drop_last=drop_last, num_workers=NUM_WORKERS, pin_memory=pin_memory, collate_fn=custom_collate)
    elif dataloader_mode in [3]:
        stock_predict = Stock_Data(mode=1, dataFrame=dataset, label_num=OUTPUT_DIMENSION,predict_days=int(args.predict_days),trend=int(args.trend))
        dataloader = DataLoader(dataset=stock_predict, batch_size=BATCH_SIZE, shuffle=False, drop_last=drop_last, num_workers=NUM_WORKERS, pin_memory=pin_memory)

    if testmodel is None:
        predict_days = int(args.predict_days)
        ckpt_prefix = f"{save_path}_out{OUTPUT_DIMENSION}_time{SEQ_LEN}"
        if predict_days > 0:
            ckpt_prefix += f"_pre{predict_days}"
        candidates = [
            f"{ckpt_prefix}_Model.pkl",
            f"{ckpt_prefix}_Model_best.pkl",
        ]
        loaded = False
        for candidate in candidates:
            if os.path.exists(candidate):
                test_model.load_state_dict(torch.load(candidate))
                loaded = True
                break
        if not loaded:
            tqdm.write("No model found")
            return -1, -1, -1
    else:
        test_model = testmodel
    test_model.eval()
    accuracy_fn = nn.MSELoss()
    pbar = tqdm(total=len(dataloader), leave=False, ncols=TQDM_NCOLS)
    with torch.no_grad():
        for batch in dataloader:
            try:
                if batch is None:
                    # tqdm.write(f"test error: batch is None")
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
                if data is None or label is None:
                    # tqdm.write(f"test error: data is None or label is None")
                    pbar.update(1)
                    continue
                if args.test_gpu == 1:
                    data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)
                    if symbol_idx is not None:
                        symbol_idx = symbol_idx.to(device, non_blocking=True).long()
                else:
                    data, label = data.to("cpu", non_blocking=True), label.to("cpu", non_blocking=True)
                    if symbol_idx is not None:
                        symbol_idx = symbol_idx.to("cpu", non_blocking=True).long()
                # Ensure data tensors do not contain NaN or inf
                if torch.isnan(data).any() or torch.isinf(data).any():
                    tqdm.write(f"test error: data has nan or inf, skip batch")
                    pbar.update(1)
                    continue
                if torch.isnan(label).any() or torch.isinf(label).any():
                    tqdm.write(f"test error: label has nan or inf, skip batch")
                    pbar.update(1)
                    continue
                # test_optimizer.zero_grad()
                if model_mode == "MULTIBRANCH":
                    price_dim = INPUT_DIMENSION // 2
                    tech_dim = INPUT_DIMENSION - price_dim
                    price_x = data[:, :, :price_dim]
                    tech_x = data[:, :, price_dim:]
                    predict = test_model.forward(price_x, tech_x)
                elif model_mode == "TRANSFORMER":
                    data = pad_input(data)
                    predict = test_model.forward(data, label, int(args.predict_days))
                else:
                    data = pad_input(data)
                    if symbol_idx is not None:
                        predict = test_model(data, symbol_index=symbol_idx)
                    else:
                        predict = test_model(data)
                predict_list.append(predict)
                if(predict.shape == label.shape):
                    accuracy = accuracy_fn(predict, label)
                    # Ensure accuracy is finite
                    if not torch.isfinite(accuracy):
                        tqdm.write(f"test warning: accuracy is not finite (nan/inf), skip batch")
                        pbar.update(1)
                        continue
                    if is_number(str(accuracy.item())):
                        accuracy_list.append(accuracy.item())
                    else:
                        pass
                    if dataloader_mode not in [2]:
                        pbar.set_description(f"test accuracy: {np.mean(accuracy_list):.2e}")
                    pbar.update(1)
                else:
                    tqdm.write(f"test error: predict.shape != label.shape")
                    pbar.update(1)
                    continue
            except Exception as e:
                tqdm.write(f"test error: {e}")
                pbar.update(1)
                continue
    if dataloader_mode not in [2]:
        tqdm.write(f"test accuracy: {np.mean(accuracy_list)}")
    pbar.close()
    if not accuracy_list:
        accuracy_list = [0]

    test_loss = np.mean(accuracy_list)
    return test_loss, predict_list, dataloader


def predict(test_codes):
    global model_mode
    print("test_code=", test_codes)
    if PKL == 0:
        load_data(test_codes,data_queue=data_queue)
        try:
            data = data_queue.get(timeout=30)
        except queue.Empty:
            print("Error: data_queue is empty")
            return
    else:
        _data = NoneDataFrame
        with open(train_pkl_path, 'rb') as f:
            data_queue = ensure_queue_compatibility(dill.load(f))
        while data_queue.empty() == False:
            try:
                item = data_queue.get(timeout=30)
                if str(item['ts_code'].iloc[0]).zfill(6) in test_codes:
                    _data = item
                    break
            except queue.Empty:
                break
    data_queue = queue.Queue()
    data = copy.deepcopy(_data)

    data = normalize_date_column(data)

    if data.empty or data["ts_code"][0] == "None":
        print("Error: data is empty or ts_code is None")
        return

    if str(data['ts_code'][0]).zfill(6) != str(test_codes[0]):
        print("Error: ts_code is not match")
        return

    predict_size = int(data.shape[0])
    if predict_size < SEQ_LEN:
        print("Error: train_size is too small or too large")
        return

    predict_data = normalize_date_column(copy.deepcopy(data))
    spliced_data = normalize_date_column(copy.deepcopy(data))
    history_window = max(1, int(getattr(args, "plot_days", 30)))
    predicted_rows: list[dict] = []
    if predict_data.empty:
        print("Error: Train_data or Test_data is None")
        return
    symbol_code = str(test_codes[0]).split('.')[0].zfill(6)
    if int(args.predict_days) <= 0:
        predict_days = abs(int(args.predict_days)) or 1
        pbar = tqdm(total=predict_days, leave=False, ncols=TQDM_NCOLS)
        while predict_days > 0:
            lastdate = pd.to_datetime(predict_data["Date"].iloc[0])
            if args.api == "tushare":
                lastclose = float(predict_data["Close"].iloc[0])
            feature_frame = predict_data.drop(columns=['ts_code', 'Date']).copy()
            feature_frame = feature_frame.fillna(feature_frame.median(numeric_only=True))
            test_loss, predict_list, _ = test(feature_frame, dataloader_mode=2)
            if test_loss == -1 and predict_list == -1:
                return
            rows = []
            for items in predict_list:
                items = items.to("cpu", non_blocking=True)
                for idxs in items:
                    row = []
                    for index, item in enumerate(idxs):
                        if use_list[index] == 1:
                            row.append(float((item * std_list[index] + mean_list[index]).detach().numpy()))
                    if row:
                        rows.append(row)
            date_obj = lastdate + timedelta(days=1)
            tmp_data = [test_codes[0], date_obj]
            if rows:
                tmp_data.extend(rows[0])
            _splice_data = copy.deepcopy(spliced_data).drop(columns=['ts_code', 'Date'])
            df_mean = _splice_data.mean().tolist()
            if args.api == "tushare":
                for index in range(len(tmp_data) - 2, len(df_mean) - 1):
                    tmp_data.append(df_mean[index])
                tmp_data.append(lastclose)
            else:
                for index in range(len(tmp_data) - 2, len(df_mean)):
                    tmp_data.append(-0.0)
            tmp_df = pd.DataFrame([tmp_data], columns=spliced_data.columns)
            predicted_rows.append(tmp_df.iloc[0].to_dict())
            predict_data = pd.concat([tmp_df, spliced_data], axis=0, ignore_index=True)
            spliced_data = normalize_date_column(copy.deepcopy(predict_data))
            predict_data['Date'] = pd.to_datetime(predict_data['Date'])

            if args.api in ("akshare", "yfinance"):
                predict_data[['Open', 'High', 'Low', 'Close', 'change', 'pct_change', 'Volume', 'amount', 'amplitude', 'exchange_rate']] = (
                    predict_data[['Open', 'High', 'Low', 'Close', 'change', 'pct_change', 'Volume', 'amount', 'amplitude', 'exchange_rate']].astype('float64')
                )
                predict_data['Date'] = predict_data['Date'].dt.strftime('%Y%m%d')
                predict_data.rename(
                    columns={
                        'Date': 'trade_date', 'Open': 'open',
                        'High': 'high', 'Low': 'low',
                        'Close': 'close', 'Volume': 'vol'},
                    inplace=True)
                predict_data = predict_data.loc[:, [
                    "ts_code", "trade_date", "open", "high", "low", "close",
                    "change", "pct_change", "vol", "amount", "amplitude", "exchange_rate"
                ]]
            elif args.api == "tushare":
                predict_data['Date'] = predict_data['Date'].dt.strftime('%Y%m%d')
                predict_data = predict_data.loc[:, [
                    "ts_code", "Date", "Open", "Close", "High", "Low",
                    "Volume", "amount", "amplitude", "pct_change", "change", "exchange_rate"
                ]]
                predict_data.rename(
                    columns={
                        'Date': 'trade_date', 'Open': 'open',
                        'High': 'high', 'Low': 'low',
                        'Close': 'close', 'Volume': 'vol'},
                    inplace=True)

            predict_data.to_csv(test_path, sep=',', index=False, header=True)
            load_data([test_codes[0]], None, test_path, data_queue=data_queue)
            while data_queue.empty() is False:
                try:
                    predict_data = data_queue.get(timeout=30)
                    predict_data = normalize_date_column(predict_data)
                    break
                except queue.Empty:
                    break

            predict_days -= 1
            pbar.update(1)
        pbar.close()

        full_df = spliced_data.copy()
        full_df['Date'] = pd.to_datetime(full_df['Date'])
        predicted_df = pd.DataFrame(predicted_rows)
        if not predicted_df.empty:
            predicted_df['Date'] = pd.to_datetime(predicted_df['Date'])
        history_df = full_df.iloc[len(predicted_rows):len(predicted_rows) + history_window].copy()
        history_df['Date'] = pd.to_datetime(history_df['Date'])
        history_df = history_df.sort_values('Date').tail(history_window)

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
        # 指标采集
        import json
        from stock_prediction.metrics import metrics_report
        print("[DEBUG] Starting metrics collection for predict_days <= 0")
        metrics_out = {}
        for col in PLOT_FEATURE_COLUMNS:
            if col not in history_df.columns:
                continue
            history_series = history_df.set_index('Date')[col].astype(float).dropna()
            prediction_series = pd.Series(dtype=float)
            if not predicted_df.empty and col in predicted_df.columns:
                prediction_series = predicted_df.set_index('Date')[col].astype(float).dropna()
            if len(history_series) == len(prediction_series) and len(history_series) > 0:
                metrics_out[col] = metrics_report(history_series.values, prediction_series.values)
        print(f"[DEBUG] metrics_out: {metrics_out}")
        if metrics_out:
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d%H%M%S")
            metrics_path = output_dir / f"metrics_{symbol_code}_{model_mode}_{ts}.json"
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics_out, f, ensure_ascii=False, indent=2)
            print(f"[LOG] Metrics saved: {metrics_path}")
        return
    else:
        normalized_predict = normalize_date_column(predict_data)
        feature_frame = normalized_predict.drop(columns=['ts_code', 'Date']).copy()
        feature_frame = feature_frame.fillna(feature_frame.median(numeric_only=True))
        test_loss, predict_list, _ = test(feature_frame, dataloader_mode=2)
        predictions = []
        for items in predict_list:
            items = items.to("cpu", non_blocking=True)
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
        actual_df = normalize_date_column(predict_data)
        actual_df['Date'] = pd.to_datetime(actual_df['Date'])
        actual_df = actual_df.sort_values('Date').tail(max(1, int(getattr(args, "plot_days", 30))))
        if not pred_df.empty:
            pred_df = pred_df.tail(len(actual_df))
        pred_df['Date'] = actual_df['Date'].iloc[:len(pred_df)].values

        for col in PLOT_FEATURE_COLUMNS:
            if col not in actual_df.columns:
                actual_df[col] = np.nan
            history_series = actual_df.set_index('Date')[col].astype(float).dropna()
            prediction_series = pd.Series(dtype=float)
            if col in pred_df.columns:
                prediction_series = pred_df.set_index('Date')[col].astype(float).dropna()
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
        import json
        from stock_prediction.metrics import metrics_report
        metrics_out = {}
        for col in PLOT_FEATURE_COLUMNS:
            if col not in actual_df.columns:
                actual_df[col] = np.nan
            history_series = actual_df.set_index('Date')[col].astype(float).dropna()
            prediction_series = pd.Series(dtype=float)
            if col in pred_df.columns:
                prediction_series = pred_df.set_index('Date')[col].astype(float).dropna()
            if len(history_series) == len(prediction_series) and len(history_series) > 0:
                metrics_out[col] = metrics_report(history_series.values, prediction_series.values)
        if metrics_out:
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d%H%M%S")
            metrics_path = output_dir / f"metrics_{symbol_code}_{model_mode}_{ts}.json"
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics_out, f, ensure_ascii=False, indent=2)
            print(f"[LOG] Metrics saved: {metrics_path}")
        return

def loss_curve(loss_list):
    global model_mode
    try:
        if not loss_list:
            print("[WARN] loss_curve: loss_list is empty, skip plotting")
            return
        save_dir = Path(png_path) / "train_loss"
        save_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(12, 6))
        steps = np.arange(1, len(loss_list) + 1)
        ax.plot(steps, np.array(loss_list), label="Training Loss", linewidth=1.2)
        ax.set_ylabel("MSELoss")
        ax.set_xlabel("iteration")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
        ax.legend()
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        img_path = save_dir / f"{cnname}_{model_mode}_{timestamp}_train_loss.png"
        fig.savefig(img_path, dpi=600)
        plt.close(fig)
        print(f"[LOG] Training loss figure saved: {img_path}")
    except Exception as e:
        print("Error: loss_curve", e)


def ensure_checkpoint_ready() -> None:
    """同步保存最新权重与最优权重，保证后续可加载。"""

    predict_days = int(getattr(args, "predict_days", 0))
    save_model(model, optimizer, save_path, False, predict_days)
    save_model(model, optimizer, save_path, True, predict_days)

def contrast_lines(test_codes):
    if getattr(args, "full_train", False):
        print("[LOG] full_train enabled, skip contrast_lines.")
        return
    global model_mode
    data = NoneDataFrame
    if PKL is False:
        load_data(test_codes,data_queue=data_queue)
        try:
            data = data_queue.get(timeout=30)
        except queue.Empty:
            print("Error: data_queue is empty")
            return
    else:
        with open(train_pkl_path, 'rb') as f:
            data_queue = ensure_queue_compatibility(dill.load(f))
        while data_queue.empty() == False:
            try:
                item = data_queue.get(timeout=30)
            except queue.Empty:
                break
            if str(item['ts_code'].iloc[0]).zfill(6) in test_codes:
                data = copy.deepcopy(item)
                break
        if data is NoneDataFrame:
            print("Error: data is None")
            return
        data_queue = queue.Queue()

    data = normalize_date_column(data)
    feature_data = data.drop(columns=['ts_code', 'Date'], errors='ignore').copy()
    feature_data = feature_data.fillna(feature_data.median(numeric_only=True))
    print("test_code=", test_codes)
    if feature_data.empty:
        print("Error: data is empty or ts_code is None")
        return -1

    if PKL is False and ('ts_code' in data.columns and data["ts_code"][0] == "None"):
        print("Error: data is empty or ts_code is None")
        return -1

    train_size = int(TRAIN_WEIGHT * (feature_data.shape[0]))
    if train_size < SEQ_LEN or train_size + SEQ_LEN > feature_data.shape[0]:
        print("Error: train_size is too small or too large")
        return -1

    Train_data = copy.deepcopy(feature_data)
    Test_data = copy.deepcopy(feature_data)
    if Train_data.empty or Test_data.empty or Train_data is None or Test_data is None:
        print("Error: Train_data or Test_data is None")
        return -1

    accuracy_list, predict_list = [], []
    real_list = []
    prediction_list = []
    test_loss, predict_list, dataloader = test(Test_data, dataloader_mode=3)
    if test_loss == -1 and predict_list == -1:
        print("Error: No model excist")
        try:
            import os
            import matplotlib
            # Configure fonts to avoid missing glyph warnings across operating systems
            import platform
            sys_plat = platform.system()
            # Use only English/Western fonts to avoid Chinese glyph warnings
            matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
            matplotlib.rcParams['axes.unicode_minus'] = False
            save_dir = os.path.join(png_path, "train_loss")
            os.makedirs(save_dir, exist_ok=True)
            plt.figure()
            x=np.linspace(1,len(loss_list),len(loss_list))
            x=20*x
            plt.plot(x,np.array(loss_list),label="train_loss")
            plt.ylabel("MSELoss")
            plt.xlabel("iteration")
            now = datetime.now()
            date_string = now.strftime("%Y%m%d%H%M%S")
            img_path = os.path.join(save_dir, f"{cnname}_{model_mode}_{date_string}_train_loss.png")
            plt.savefig(img_path, dpi=600)
            plt.close()
            print(f"[LOG] Training loss figure saved: {img_path}")
        except Exception as e:
            print("Error: loss_curve", e)
    else:

        for i, (_, label) in enumerate(dataloader):
            for idx in range(label.shape[0]):
                _tmp = []
                for index in range(len(show_list)):
                    value = label[idx]
                    # Support both scalar and higher-dimensional tensors
                    if hasattr(value, 'dim'):
                        if value.dim() == 0:
                            v = value.item()
                        elif value.dim() == 1:
                            v = value[index].item() if index < value.shape[0] else value[-1].item()
                        else:
                            v = value[0][index].item() if index < value.shape[-1] else value[0][-1].item()
                    else:
                        v = value
                    if show_list[index] == 1:
                        _tmp.append(v * test_std_list[index] + test_mean_list[index])
                real_list.append(np.array(_tmp))

        for items in predict_list:
            items = items.to("cpu", non_blocking=True)
            for idxs in items:
                _tmp = []
                # Handle idxs tensors with dimension 0/1/2
                if hasattr(idxs, 'dim'):
                    if idxs.dim() == 0:
                        values = [idxs.item()]
                    elif idxs.dim() == 1:
                        values = idxs.tolist()
                    elif idxs.dim() == 2:
                        values = idxs[0].tolist()
                    else:
                        values = idxs.flatten().tolist()
                else:
                    values = [idxs]
                for index, item in enumerate(values):
                    if show_list[index] == 1:
                        _tmp.append(item * test_std_list[index] + test_mean_list[index])
                prediction_list.append(np.array(_tmp))
    selected_features = [name_list[idx] for idx, flag in enumerate(show_list) if flag == 1]
    rename_map = {"open": "Open", "high": "High", "low": "Low", "close": "Close"}
    real_array = np.array(real_list)
    pred_array = np.array(prediction_list)
    min_len = min(len(real_array), len(pred_array))
    if min_len == 0:
        print("Error: No valid prediction results for plotting.")
        return
    plot_window = max(1, int(getattr(args, "plot_days", 30)))
    real_array = real_array[:min_len][-plot_window:]
    pred_array = pred_array[:min_len][-plot_window:]
    column_names = [rename_map.get(name, name.title()) for name in selected_features]
    real_df = pd.DataFrame(real_array, columns=column_names)
    pred_df = pd.DataFrame(pred_array, columns=column_names)
    if data is not None and "Date" in data.columns:
        date_series = pd.to_datetime(data['Date']).iloc[:min_len][-plot_window:]
    else:
        print("Error: contrast_lines cannot find date column.")
        return -1
    real_df['Date'] = date_series.values
    pred_df['Date'] = date_series.values
    real_df.sort_values('Date', inplace=True)
    pred_df.sort_values('Date', inplace=True)
    symbol_code = str(test_codes[0]).split('.')[0].zfill(6)
    for col in PLOT_FEATURE_COLUMNS:
        if col not in real_df.columns:
            continue
        history_series = real_df.set_index('Date')[col].astype(float).dropna()
        prediction_series = pd.Series(dtype=float)
        if col in pred_df.columns:
            prediction_series = pred_df.set_index('Date')[col].astype(float).dropna()
        plot_feature_comparison(
            symbol_code,
            model_mode,
            col,
            history_series,
            prediction_series,
            Path(png_path) / "test",
            prefix="test",
        )
    # 指标采集
    import json
    from stock_prediction.metrics import metrics_report
    metrics_out = {}
    for col in PLOT_FEATURE_COLUMNS:
        if col in real_df.columns and col in pred_df.columns:
            history_series = real_df.set_index('Date')[col].astype(float).dropna()
            prediction_series = pred_df.set_index('Date')[col].astype(float).dropna()
            if len(history_series) == len(prediction_series) and len(history_series) > 0:
                metrics_out[col] = metrics_report(history_series.values, prediction_series.values)
    if metrics_out:
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        metrics_path = output_dir / f"metrics_{symbol_code}_{model_mode}_{ts}.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_out, f, ensure_ascii=False, indent=2)
        print(f"[LOG] Metrics saved: {metrics_path}")
    return

def main():
    """Main entry point: training, testing, and prediction."""
    global args
    global last_loss,test_model,model,total_test_length,lr_scheduler,drop_last
    global criterion, optimizer, model_mode, save_path, device, last_save_time
    global lo_list
    
    # Parse command-line arguments only when executed as a script
    args = parser.parse_args()
    args.full_train = bool(int(getattr(args, "full_train", 0)))
    # b_size * (p_days * n_head) * (d_model // n_head) = b_size * seq_len * d_model
    # if int(args.predict_days) > 0:
    #     assert BATCH_SIZE * (int(args.predict_days) * NHEAD) * (D_MODEL // NHEAD) == BATCH_SIZE * SEQ_LEN * D_MODEL and D_MODEL % NHEAD == 0, "Error: assert error"

    # if args.predict_days <= 0:
    #     drop_last = False
    # else:
    #     drop_last = True
    drop_last = False
    last_loss = 1e10
    if os.path.exists('loss.txt'):
        with open('loss.txt', 'r') as file:
            last_loss = float(file.read())
    print("last_loss=", last_loss)

    mode = args.mode
    model_mode = args.model.upper()
    PKL = False if args.pkl <= 0 else True
    if args.cpu == 1:
        device = torch.device("cpu")

    if model_mode == "LSTM":
        model = LSTM(input_dim=INPUT_DIMENSION)
        model._init_args = dict(input_dim=INPUT_DIMENSION)
        test_model = LSTM(input_dim=INPUT_DIMENSION)
        test_model._init_args = dict(input_dim=INPUT_DIMENSION)
        save_path = lstm_path
        criterion = nn.MSELoss()
    elif model_mode == "ATTENTION_LSTM":
        model = AttentionLSTM(input_dim=INPUT_DIMENSION, hidden_dim=128, num_layers=2, output_dim=OUTPUT_DIMENSION)
        model._init_args = dict(input_dim=INPUT_DIMENSION, hidden_dim=128, num_layers=2, output_dim=OUTPUT_DIMENSION)
        test_model = AttentionLSTM(input_dim=INPUT_DIMENSION, hidden_dim=128, num_layers=2, output_dim=OUTPUT_DIMENSION)
        test_model._init_args = dict(input_dim=INPUT_DIMENSION, hidden_dim=128, num_layers=2, output_dim=OUTPUT_DIMENSION)
        save_path = "output/attention_lstm"
        criterion = nn.MSELoss()
    elif model_mode == "BILSTM":
        model = BiLSTM(input_dim=INPUT_DIMENSION, hidden_dim=128, num_layers=2, output_dim=OUTPUT_DIMENSION)
        model._init_args = dict(input_dim=INPUT_DIMENSION, hidden_dim=128, num_layers=2, output_dim=OUTPUT_DIMENSION)
        test_model = BiLSTM(input_dim=INPUT_DIMENSION, hidden_dim=128, num_layers=2, output_dim=OUTPUT_DIMENSION)
        test_model._init_args = dict(input_dim=INPUT_DIMENSION, hidden_dim=128, num_layers=2, output_dim=OUTPUT_DIMENSION)
        save_path = "output/bilstm"
        criterion = nn.MSELoss()
    elif model_mode == "TCN":
        model = TCN(input_dim=INPUT_DIMENSION, output_dim=OUTPUT_DIMENSION, num_channels=[64, 64, 64], kernel_size=3)
        model._init_args = dict(input_dim=INPUT_DIMENSION, output_dim=OUTPUT_DIMENSION, num_channels=[64, 64, 64], kernel_size=3)
        test_model = TCN(input_dim=INPUT_DIMENSION, output_dim=OUTPUT_DIMENSION, num_channels=[64, 64, 64], kernel_size=3)
        test_model._init_args = dict(input_dim=INPUT_DIMENSION, output_dim=OUTPUT_DIMENSION, num_channels=[64, 64, 64], kernel_size=3)
        save_path = "output/tcn"
        criterion = nn.MSELoss()
    elif model_mode == "MULTIBRANCH":
        price_dim = INPUT_DIMENSION // 2
        tech_dim = INPUT_DIMENSION - price_dim
        model = MultiBranchNet(price_dim=price_dim, tech_dim=tech_dim, hidden_dim=64, output_dim=OUTPUT_DIMENSION)
        model._init_args = dict(price_dim=price_dim, tech_dim=tech_dim, hidden_dim=64, output_dim=OUTPUT_DIMENSION)
        test_model = MultiBranchNet(price_dim=price_dim, tech_dim=tech_dim, hidden_dim=64, output_dim=OUTPUT_DIMENSION)
        test_model._init_args = dict(price_dim=price_dim, tech_dim=tech_dim, hidden_dim=64, output_dim=OUTPUT_DIMENSION)
        save_path = "output/multibranch"
        criterion = nn.MSELoss()
    elif model_mode == "TRANSFORMER":
        model = TransformerModel(input_dim=INPUT_DIMENSION, d_model=D_MODEL, nhead=NHEAD, num_layers=6, 
                               dim_feedforward=2048, output_dim=OUTPUT_DIMENSION, max_len=SEQ_LEN, mode=0)
        model._init_args = dict(input_dim=INPUT_DIMENSION, d_model=D_MODEL, nhead=NHEAD, num_layers=6,
                               dim_feedforward=2048, output_dim=OUTPUT_DIMENSION, max_len=SEQ_LEN, mode=0)
        test_model = TransformerModel(input_dim=INPUT_DIMENSION, d_model=D_MODEL, nhead=NHEAD, 
                                    num_layers=6, dim_feedforward=2048, output_dim=OUTPUT_DIMENSION, max_len=SEQ_LEN, mode=1)
        test_model._init_args = dict(input_dim=INPUT_DIMENSION, d_model=D_MODEL, nhead=NHEAD, num_layers=6,
                                    dim_feedforward=2048, output_dim=OUTPUT_DIMENSION, max_len=SEQ_LEN, mode=1)
        save_path = transformer_path
        criterion = nn.MSELoss()
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
        model._init_args = dict(
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
        test_model._init_args = dict(
            input_dim=INPUT_DIMENSION,
            output_dim=OUTPUT_DIMENSION,
            predict_steps=hybrid_steps,
            use_symbol_embedding=SYMBOL_EMBED_ENABLED,
            symbol_embedding_dim=SYMBOL_EMBED_DIM,
            max_symbols=SYMBOL_VOCAB_SIZE,
        )
        save_path = str(config.get_model_path("HYBRID", symbol))
        criterion = HybridLoss(model, mse_weight=1.0, quantile_weight=0.1, direction_weight=0.05, regime_weight=0.05)
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
        model._init_args = dict(
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
        test_model._init_args = dict(
            input_dim=INPUT_DIMENSION,
            output_dim=OUTPUT_DIMENSION,
            predict_steps=ensemble_steps,
            use_symbol_embedding=SYMBOL_EMBED_ENABLED,
            symbol_embedding_dim=SYMBOL_EMBED_DIM,
            max_symbols=SYMBOL_VOCAB_SIZE,
        )
        save_path = str(config.get_model_path("PTFT_VSSM", symbol))
        criterion = PTFTVSSMLoss(model, mse_weight=1.0, kl_weight=1e-3)
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
        model._init_args = dict(
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
        test_model._init_args = dict(
            input_dim=INPUT_DIMENSION,
            output_dim=OUTPUT_DIMENSION,
            predict_steps=diffusion_steps,
            use_symbol_embedding=SYMBOL_EMBED_ENABLED,
            symbol_embedding_dim=SYMBOL_EMBED_DIM,
            max_symbols=SYMBOL_VOCAB_SIZE,
        )
        save_path = str(config.get_model_path("DIFFUSION", symbol))
        criterion = nn.MSELoss()
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
        model._init_args = dict(
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
        test_model._init_args = dict(
            input_dim=INPUT_DIMENSION,
            output_dim=OUTPUT_DIMENSION,
            predict_steps=graph_steps,
            use_symbol_embedding=SYMBOL_EMBED_ENABLED,
            symbol_embedding_dim=SYMBOL_EMBED_DIM,
            max_symbols=SYMBOL_VOCAB_SIZE,
        )
        save_path = str(config.get_model_path("GRAPH", symbol))
        criterion = nn.MSELoss()
    elif model_mode == "CNNLSTM":
        assert abs(abs(int(args.predict_days))) > 0, "Error: predict_days must be greater than 0"
        model = CNNLSTM(input_dim=INPUT_DIMENSION, num_classes=OUTPUT_DIMENSION, predict_days=abs(int(args.predict_days)))
        model._init_args = dict(input_dim=INPUT_DIMENSION, num_classes=OUTPUT_DIMENSION, predict_days=abs(int(args.predict_days)))
        test_model = CNNLSTM(input_dim=INPUT_DIMENSION, num_classes=OUTPUT_DIMENSION, predict_days=abs(int(args.predict_days)))
        test_model._init_args = dict(input_dim=INPUT_DIMENSION, num_classes=OUTPUT_DIMENSION, predict_days=abs(int(args.predict_days)))
        save_path = cnnlstm_path
        criterion = nn.MSELoss()
    else:
        print(f"No such model: {model_mode}")
        exit(0)

    model=model.to(device, non_blocking=True)
    if args.test_gpu == 0:
        test_model=test_model.to('cpu', non_blocking=True)
    else:
        test_model=test_model.to(device, non_blocking=True)
    if torch.cuda.device_count() >= 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            if args.test_gpu == 1:
                test_model = nn.DataParallel(test_model)
    else:
        print("Let's use CPU!")

    print(model)
    optimizer=optim.Adam(model.parameters(),lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    lr_scheduler = CustomSchedule(d_model=D_MODEL, warmup_steps=WARMUP_STEPS, optimizer=optimizer)
    if int(args.predict_days) > 0:
        if os.path.exists(save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_pre" + str(args.predict_days) + "_Model.pkl") and os.path.exists(save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_pre" + str(args.predict_days) + "_Optimizer.pkl"):
            print("Load model and optimizer from file")
            model.load_state_dict(torch.load(save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_pre" + str(args.predict_days) + "_Model.pkl"))
            optimizer.load_state_dict(torch.load(save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_pre" + str(args.predict_days) + "_Optimizer.pkl"))
        else:
            print("No model and optimizer file, train from scratch")
    else:
        if os.path.exists(save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_Model.pkl") and os.path.exists(save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_Optimizer.pkl"):
            print("Load model and optimizer from file")
            model.load_state_dict(torch.load(save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_Model.pkl"))
            optimizer.load_state_dict(torch.load(save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_Optimizer.pkl"))
        else:
            print("No model and optimizer file, train from scratch")

    # period = 100
    train_codes = []
    test_codes = []
    print("Clean the data...")
    if symbol == 'Generic.Data':
        # ts_codes = get_stock_list()
        csv_files = glob.glob(daily_path+"/*.csv")
        ts_codes =[]
        for csv_file in csv_files:
            ts_codes.append(os.path.basename(csv_file).rsplit(".", 1)[0])
    else:
        ts_codes = [symbol]
    
    if len(ts_codes) > 1:
        # train_codes = ts_codes[:int(TRAIN_WEIGHT*len(ts_codes))]
        # test_codes = ts_codes[int(TRAIN_WEIGHT*len(ts_codes)):]
        if os.path.exists("test_codes.txt"):
            with open("test_codes.txt", 'r') as f:
                test_codes = f.read().splitlines()
            train_codes = list(set(ts_codes) - set(test_codes))
        else:
            train_codes = random.sample(ts_codes, int(TRAIN_WEIGHT*len(ts_codes)))
            test_codes = list(set(ts_codes) - set(train_codes))
            with open("test_codes.txt", 'w') as f:
                for test_code in test_codes:
                    f.write(test_code + "\n")
    else:
        train_codes = ts_codes
        test_codes = ts_codes
    if getattr(args, "full_train", False):
        train_codes = ts_codes
        test_codes = []
    random.shuffle(ts_codes)
    random.shuffle(train_codes)
    random.shuffle(test_codes)

    if mode == 'train':
        lo_list.clear()
        data_len=0
        total_length = 0
        total_test_length = 0
        if PKL is False:
            print("Load data from csv using thread ...")
            data_thread = threading.Thread(target=load_data, args=(ts_codes,))
            data_thread.start()
            codes_len = len(ts_codes)
        else:
            _datas = []
            with open(train_pkl_path, 'rb') as f:
                _data_queue = ensure_queue_compatibility(dill.load(f))
                while _data_queue.empty() == False:
                    try:
                        _datas.append(_data_queue.get(timeout=30))
                    except queue.Empty:
                        break
                random.shuffle(_datas)
                init_bar = tqdm(total=len(_datas), ncols=TQDM_NCOLS)
                for _data in _datas:
                    init_bar.update(1)
                    # _data = _data.dropna()
                    # _data = _data.fillna(-0.0)
                    _data = _data.fillna(_data.median(numeric_only=True))
                    if _data.empty:
                        continue
                    _ts_code = str(_data['ts_code'].iloc[0]).zfill(6)
                    if args.api == "akshare":
                        _ts_code = _ts_code.zfill(6)
                    if _ts_code in train_codes:
                        data_queue.put(_data)
                        total_length += _data.shape[0] - SEQ_LEN
                    if _ts_code in test_codes:
                        test_queue.put(_data)
                        total_test_length += _data.shape[0] - SEQ_LEN
                    if not getattr(args, "full_train", False) and _ts_code not in train_codes and _ts_code not in test_codes:
                        print("Error: %s not in train or test"%_ts_code)
                        continue
                    if not getattr(args, "full_train", False) and _ts_code in train_codes and _ts_code in test_codes:
                        print("Error: %s in train and test"%_ts_code)
                        continue
                init_bar.close()
            codes_len = data_queue.qsize()
        print("data_queue size: %d" % (data_queue.qsize()))
        print("total codes: %d, total length: %d"%(codes_len, total_length))
        print("total test codes: %d, total test length: %d"%(test_queue.qsize(), total_test_length))
        # batch_none = 0
        # data_none = 0
        # scaler = GradScaler('cuda')
        pbar = tqdm(total=args.epoch, leave=False, ncols=TQDM_NCOLS)
        last_epoch = 0
        for epoch in range(0, args.epoch):
            if len(lo_list) == 0:
                    m_loss = 0
            else:
                m_loss = np.mean(lo_list)
            pbar.set_description("%d, %e"%(epoch+1,m_loss))
            if args.pkl_queue == 0:
                if epoch == 0:
                    tqdm.write("pkl_queue is disabled")
                code_bar = tqdm(total=codes_len, ncols=TQDM_NCOLS)
                for index in range (codes_len):
                    try:
                        if PKL is False:
                            while data_queue.empty() == False:
                                try:
                                    data_list += [data_queue.get(timeout=30)]
                                except queue.Empty:
                                    break
                                data_len = max(data_len, data_queue.qsize())
                            Err_nums = 5
                            while index >= len(data_list):
                                if data_queue.empty() == False:
                                    try:
                                        data_list += [data_queue.get(timeout=30)]
                                    except queue.Empty:
                                        break
                                time.sleep(5)
                                Err_nums -= 1
                                if Err_nums == 0:
                                    tqdm.write("Error: data_list is empty")
                                    exit(0)
                        elif index >= len(data_list):
                            tqdm.write("Error: data_list is empty")
                            code_bar.close()
                            break
                        data = data_list[index].copy(deep=True)
                        # data = data.dropna()
                        # data = data.fillna(-0.0)
                        data = data.fillna(data.median(numeric_only=True))
                        if data.empty or data["ts_code"][0] == "None":
                            tqdm.write("data is empty or data has invalid col")
                            code_bar.update(1)
                            continue
                        ts_code = str(data['ts_code'].iloc[0]).zfill(6)
                        if args.begin_code != "":
                            if ts_code != args.begin_code:
                                code_bar.update(1)
                                continue
                            else:
                                args.begin_code = ""
                        data.drop(['ts_code','Date'],axis=1,inplace = True)    
                        if getattr(args, "full_train", False):
                            Train_data = data
                        else:
                            train_size=int(TRAIN_WEIGHT*(data.shape[0]))
                            if train_size<SEQ_LEN or train_size+SEQ_LEN>data.shape[0]:
                                code_bar.update(1)
                                continue
                            Train_data=data[:train_size+SEQ_LEN]
                        # Test_data=data[train_size-SEQ_LEN:]
                        if Train_data.shape[0] < SEQ_LEN:
                            code_bar.update(1)
                            continue
                        if Train_data.empty or Train_data is None:
                            tqdm.write(ts_code + ":Train_data is None")
                            code_bar.update(1)
                            continue
                        stock_train=Stock_Data(mode=0, dataFrame=Train_data, label_num=OUTPUT_DIMENSION)
                        if len(loss_list) == 0:
                            m_loss = 0
                        else:
                            m_loss = np.mean(loss_list)
                        code_bar.set_description("%s, %d, %e" % (ts_code,data_len,m_loss))
                    except Exception as e:
                        print(ts_code,"main function ", e)
                        code_bar.update(1)
                        continue
            else:
                if epoch == 0:
                    tqdm.write("pkl_queue is enabled")
                ts_code = "data_queue"
                index = len(ts_codes) - 1
                # tqdm.write("epoch: %d, data_queue size before deep copy: %d" % (epoch, data_queue.qsize()))
                _stock_data_queue = deep_copy_queue(data_queue)

                # tqdm.write("epoch: %d, data_queue size after deep copy: %d" % (epoch, data_queue.qsize()))
                # tqdm.write("epoch: %d, _stock_data_queue size: %d" % (epoch, _stock_data_queue.qsize()))
                
                stock_train = stock_queue_dataset(mode=0, data_queue=_stock_data_queue, label_num=OUTPUT_DIMENSION, 
                                                  buffer_size=BUFFER_SIZE, total_length=total_length,
                                                  predict_days=int(args.predict_days),trend=int(args.trend))
            # iteration=0
            loss_list=[]
            
            train_pin_memory = torch.cuda.is_available() and device.type == "cuda" and getattr(args, "cpu", 0) == 0
            train_dataloader=DataLoader(dataset=stock_train,batch_size=BATCH_SIZE,shuffle=False,drop_last=drop_last, 
                                        num_workers=NUM_WORKERS, pin_memory=train_pin_memory, collate_fn=custom_collate)
            # predict_list=[]
            # accuracy_list=[]
            train_with_trainer(train_dataloader, epoch_count=1)
            if args.pkl_queue == 0:
                code_bar.update(1)
            if (time.time() - last_save_time >= SAVE_INTERVAL or index == len(ts_codes) - 1) and safe_save == True:
                thread_save_model(model, optimizer, save_path, False, int(args.predict_days))
                last_save_time = time.time()
            if args.pkl_queue == 0:
                code_bar.close()
            pbar.update(1)
            last_epoch = epoch
        pbar.close()
        print("Training finished!")
        ensure_checkpoint_ready()
        # 绘制 Trainer 记录的 loss 曲线
        if hasattr(train_with_trainer, "last_history") and train_with_trainer.last_history is not None:
            train_loss_list = train_with_trainer.last_history.get("batch_loss", [])
            if train_loss_list:
                print("Start create image for loss (final)")
                loss_curve(train_loss_list)
        elif len(lo_list) > 0:
            print("Start create image for loss (final)")
            loss_curve(lo_list)
        if getattr(args, "full_train", False) or len(test_codes) == 0:
            print("[LOG] full_train enabled或无测试代码，跳过预测对比。")
        else:
            print("Start create image for pred-real")
            test_index = random.randint(0, len(test_codes) - 1)
            test_code = [test_codes[test_index]]
            # 采集并保存指标
            try:
                import json
                from stock_prediction.metrics import metrics_report
                pred_result = contrast_lines(test_code)
                if isinstance(pred_result, tuple) and len(pred_result) == 2:
                    y_true, y_pred = pred_result
                    metrics_out = {"close": metrics_report(y_true, y_pred)}
                    output_dir = Path("output")
                    output_dir.mkdir(exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d%H%M%S")
                    metrics_path = output_dir / f"metrics_train_{test_code[0]}_{model_mode}_{ts}.json"
                    with open(metrics_path, "w", encoding="utf-8") as f:
                        json.dump(metrics_out, f, ensure_ascii=False, indent=2)
                    print(f"[LOG] Train metrics saved: {metrics_path}")
            except Exception as e:
                print(f"[WARN] Train metrics collection failed: {e}")
            while contrast_lines(test_code) == -1:
                test_index = random.randint(0, len(test_codes) - 1)
                test_code = [test_codes[test_index]]
        print("train epoch: %d" % (last_epoch))
    elif mode == "test":
        if getattr(args, "full_train", False) and not test_codes and args.test_code == "":
            print("[LOG] 无测试数据可用于对比。")
            return
        if args.test_code != "" or args.test_code == "all":
            test_code = [args.test_code]
        else:
            test_index = random.randint(0, len(test_codes) - 1)
            test_code = [test_codes[test_index]]
        while contrast_lines(test_code) == -1:
            test_index = random.randint(0, len(test_codes) - 1)
            test_code = [test_codes[test_index]]
    elif mode == "predict":
        if args.test_code == "":
            print("Error: test_code is empty")
            exit(0)
        elif args.test_code in ts_codes or PKL == True:
            test_code = [args.test_code]
            predict(test_code)
        else:
            print("Error: test_code is not in ts_codes")
            exit(0)


def create_predictor(model_type="lstm", device_type="cpu"):
    
    """Create a predictor instance (used by tests and external callers)."""


    class Predictor:
        def __init__(self, model_type, device_type):
            self.model_type = model_type.upper()
            self.device = torch.device(device_type)
            
    return Predictor(model_type, device_type)


if __name__ == "__main__":
    main()
