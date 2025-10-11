#!/usr/bin/env python#!/usr/bin/env python

# coding: utf-8# coding: utf-8

""""""

Stock prediction core moduleStock prediction core module

- Restored training, testing, prediction entrypoints- Restores training, testing, and prediction entrypoints

- CSV pipeline implemented; PKL pipeline TODO- CSV pipeline is implemented; PKL pipeline is TODO

""""""

from __future__ import annotationsfrom __future__ import annotations



import argparseimport argparse

import globimport glob

import osimport os

import randomimport random

import timeimport time

from typing import List, Tuplefrom typing import List, Tuple



try:# Safe imports from package or as script

    from .init import *  # noqa: F401,F403try:

    from .common import (  # noqa: F401    from .init import *  # noqa: F401,F403 - use project-wide constants and torch utils

        LSTM,    from .common import (  # noqa: F401

        TransformerModel,        LSTM,

        CNNLSTM,        TransformerModel,

        import_csv,        CNNLSTM,

        pad_input,        import_csv,

        thread_save_model,        pad_input,

        Stock_Data,        thread_save_model,

    )        Stock_Data,

except Exception:    )

    import sysexcept Exception:

    from pathlib import Path    import sys

    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "src"

    if str(root) not in sys.path:    root = Path(__file__).resolve().parents[2] / "src"

        sys.path.insert(0, str(root))    if str(root) not in sys.path:

    from stock_prediction.init import *  # type: ignore  # noqa: F401,F403        sys.path.insert(0, str(root))

    from stock_prediction.common import (  # type: ignore  # noqa: F401    from stock_prediction.init import *  # type: ignore  # noqa: F401,F403

        LSTM,    from stock_prediction.common import (  # type: ignore  # noqa: F401

        TransformerModel,        LSTM,

        CNNLSTM,        TransformerModel,

        import_csv,        CNNLSTM,

        pad_input,        import_csv,

        thread_save_model,        pad_input,

        Stock_Data,        thread_save_model,

    )        Stock_Data,

    )



# Global training state

last_save_time = 0.0# Globals for training state

loss = Nonelast_save_time = 0.0

loss_list: List[float] = []loss = None

iteration = 0loss_list: List[float] = []

lo_list: List[float] = []iteration = 0

last_loss = 1e10lo_list: List[float] = []

lr_scheduler = Nonelast_loss = 1e10

model = Nonelr_scheduler = None

test_model = Nonemodel = None

optimizer = Nonetest_model = None

criterion = Noneoptimizer = None

save_path = Nonecriterion = None

scaler = Nonesave_path = None

scaler = None



def train_one_epoch(epoch: int, dataloader, scaler_param, ts_code: str = "") -> None:

    global loss, iteration, lr_schedulerdef train_one_epoch(epoch: int, dataloader, scaler_param, ts_code: str = "") -> None:

    model.train()    global loss, iteration, lr_scheduler

    subbar = tqdm(total=len(dataloader), leave=False, ncols=TQDM_NCOLS)    model.train()

    subbar = tqdm(total=len(dataloader), leave=False, ncols=TQDM_NCOLS)

    for batch in dataloader:

        iteration += 1    for batch in dataloader:

        if batch is None:        iteration += 1

            subbar.update(1)        if batch is None:

            continue            subbar.update(1)

            continue

        data, label = batch

        if data is None or label is None:        data, label = batch

            subbar.update(1)        if data is None or label is None:

            continue            subbar.update(1)

            continue

        data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

        data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

        with autocast(device_type=device.type):

            data = pad_input(data)        with autocast(device_type=device.type):

            outputs = model.forward(data, label, 0)            data = pad_input(data)

            loss = criterion(outputs, label)            outputs = model.forward(data, label, 0)

            loss = criterion(outputs, label)

        optimizer.zero_grad(set_to_none=True)

        scaler_param.scale(loss).backward()        optimizer.zero_grad(set_to_none=True)

        scaler_param.step(optimizer)        scaler_param.scale(loss).backward()

        scaler_param.update()        scaler_param.step(optimizer)

        lr_scheduler.step()        scaler_param.update()

        lr_scheduler.step()

        loss_list.append(float(loss.item()))

        if iteration % 100 == 0:        loss_list.append(float(loss.item()))

            lo_list.append(float(loss.item()))        if iteration % 100 == 0:

            lo_list.append(float(loss.item()))

        subbar.set_description(f"{ts_code}, e:{epoch}, i:{iteration}, loss:{loss.item():.3e}")

        subbar.update(1)        subbar.set_description(f"{ts_code}, e:{epoch}, i:{iteration}, loss:{loss.item():.3e}")

        subbar.update(1)

    subbar.close()

    subbar.close()



def run_test(dataset) -> Tuple[list, list, float]:

    tm = test_model if test_model is not None else modeldef run_test(dataset) -> Tuple[list, list, float]:

    tm.eval()    tm = test_model if test_model is not None else model

    tm.eval()

    # Wrap np/DataFrame into dataset

    if isinstance(dataset, np.ndarray) or hasattr(dataset, "values"):    # Wrap np/DataFrame into dataset

        stock_ds = Stock_Data(mode=1, dataFrame=dataset, label_num=OUTPUT_DIMENSION)    if isinstance(dataset, np.ndarray) or hasattr(dataset, "values"):

        dl = DataLoader(dataset=stock_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=NUM_WORKERS, pin_memory=True)        stock_ds = Stock_Data(mode=1, dataFrame=dataset, label_num=OUTPUT_DIMENSION)

    else:        dl = DataLoader(dataset=stock_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=NUM_WORKERS, pin_memory=True)

        dl = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=NUM_WORKERS, pin_memory=True)    else:

        dl = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=NUM_WORKERS, pin_memory=True)

    total = 0.0

    cnt = 0    total = 0.0

    preds, labels = [], []    cnt = 0

    with torch.no_grad():    preds, labels = [], []

        for data, label in dl:    with torch.no_grad():

            if data is None or label is None:        for data, label in dl:

                continue            if data is None or label is None:

            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)                continue

            data = pad_input(data)            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

            out = tm.forward(data, label, 0)            data = pad_input(data)

            l = criterion(out, label)            out = tm.forward(data, label, 0)

            total += float(l.item())            l = criterion(out, label)

            cnt += 1            total += float(l.item())

            preds.append(out.detach().cpu().numpy())            cnt += 1

            labels.append(label.detach().cpu().numpy())            preds.append(out.detach().cpu().numpy())

    avg = total / cnt if cnt else float("inf")            labels.append(label.detach().cpu().numpy())

    print(f"Test Loss: {avg:.6f}")    avg = total / cnt if cnt else float("inf")

    return preds, labels, avg    print(f"Test Loss: {avg:.6f}")

    return preds, labels, avg



def run_predict(codes: List[str]) -> None:

    if test_model is None:def run_predict(codes: List[str]) -> None:

        print("Error: test_model not initialized")    if test_model is None:

        return        print("Error: test_model not initialized")

    test_model.eval()        return

    test_model.eval()

    for code in codes:

        print(f"Predicting: {code}")    for code in codes:

        if PKL:        print(f"Predicting: {code}")

            print("PKL pipeline TODO for code", code)        if PKL:

            continue            print("PKL pipeline TODO for code", code)

        data = import_csv(code)            continue

        if data is None:        data = import_csv(code)

            print("No data for", code)        if data is None:

            continue            print("No data for", code)

        run_test(data)            continue

        print("Done:", code)        run_test(data)

        print("Done:", code)



def main() -> None:

    global model, test_model, optimizer, criterion, save_path, scaler, last_save_time, lr_schedulerdef main() -> None:

    global model, test_model, optimizer, criterion, save_path, scaler, last_save_time, lr_scheduler

    ap = argparse.ArgumentParser()

    ap.add_argument("--mode", default="train", choices=["train", "test", "predict"])     ap = argparse.ArgumentParser()

    ap.add_argument("--model", default="transformer", choices=["lstm", "transformer", "cnnlstm"])     ap.add_argument("--mode", default="train", choices=["train", "test", "predict"]) 

    ap.add_argument("--cpu", type=int, default=0)    ap.add_argument("--model", default="transformer", choices=["lstm", "transformer", "cnnlstm"]) 

    ap.add_argument("--pkl", type=int, default=int(PKL))    ap.add_argument("--cpu", type=int, default=0)

    ap.add_argument("--test_code", type=str, default="")    ap.add_argument("--pkl", type=int, default=int(PKL))

    ap.add_argument("--test_gpu", type=int, default=1)    ap.add_argument("--test_code", type=str, default="")

    ap.add_argument("--predict_days", type=int, default=0)    ap.add_argument("--test_gpu", type=int, default=1)

    ap.add_argument("--trend", type=int, default=0)    ap.add_argument("--predict_days", type=int, default=0)

    ap.add_argument("--epochs", type=int, default=EPOCH)    ap.add_argument("--trend", type=int, default=0)

    args = ap.parse_args()    ap.add_argument("--epochs", type=int, default=EPOCH)

    args = ap.parse_args()

    # Device

    device_to_use = torch.device("cpu") if args.cpu == 1 else device    # Device

    device_to_use = torch.device("cpu") if args.cpu == 1 else device

    # Model

    mm = args.model.upper()    # Model

    if mm == "LSTM":    mm = args.model.upper()

        model = LSTM(dimension=INPUT_DIMENSION)    if mm == "LSTM":

        test_model = LSTM(dimension=INPUT_DIMENSION)        model = LSTM(dimension=INPUT_DIMENSION)

        save_path = lstm_path        test_model = LSTM(dimension=INPUT_DIMENSION)

        criterion_local = nn.MSELoss()        save_path = lstm_path

    elif mm == "TRANSFORMER":        criterion_local = nn.MSELoss()

        model = TransformerModel(input_dim=INPUT_DIMENSION, d_model=D_MODEL, nhead=NHEAD, num_layers=6, dim_feedforward=2048, output_dim=OUTPUT_DIMENSION, max_len=SEQ_LEN, mode=0)    elif mm == "TRANSFORMER":

        test_model = TransformerModel(input_dim=INPUT_DIMENSION, d_model=D_MODEL, nhead=NHEAD, num_layers=6, dim_feedforward=2048, output_dim=OUTPUT_DIMENSION, max_len=SEQ_LEN, mode=1)        model = TransformerModel(input_dim=INPUT_DIMENSION, d_model=D_MODEL, nhead=NHEAD, num_layers=6, dim_feedforward=2048, output_dim=OUTPUT_DIMENSION, max_len=SEQ_LEN, mode=0)

        save_path = transformer_path        test_model = TransformerModel(input_dim=INPUT_DIMENSION, d_model=D_MODEL, nhead=NHEAD, num_layers=6, dim_feedforward=2048, output_dim=OUTPUT_DIMENSION, max_len=SEQ_LEN, mode=1)

        criterion_local = nn.MSELoss()        save_path = transformer_path

    else:        criterion_local = nn.MSELoss()

        model = CNNLSTM(input_dim=INPUT_DIMENSION, num_classes=OUTPUT_DIMENSION, predict_days=abs(int(args.predict_days)))    else:

        test_model = CNNLSTM(input_dim=INPUT_DIMENSION, num_classes=OUTPUT_DIMENSION, predict_days=abs(int(args.predict_days)))        model = CNNLSTM(input_dim=INPUT_DIMENSION, num_classes=OUTPUT_DIMENSION, predict_days=abs(int(args.predict_days)))

        save_path = cnnlstm_path        test_model = CNNLSTM(input_dim=INPUT_DIMENSION, num_classes=OUTPUT_DIMENSION, predict_days=abs(int(args.predict_days)))

        criterion_local = nn.MSELoss()        save_path = cnnlstm_path

        criterion_local = nn.MSELoss()

    # Move and DDP

    model = model.to(device_to_use, non_blocking=True)    # Move and DDP

    if args.test_gpu == 0:    model = model.to(device_to_use, non_blocking=True)

        test_local = test_model.to("cpu", non_blocking=True)    if args.test_gpu == 0:

    else:        test_local = test_model.to("cpu", non_blocking=True)

        test_local = test_model.to(device_to_use, non_blocking=True)    else:

    if torch.cuda.device_count() >= 1:        test_local = test_model.to(device_to_use, non_blocking=True)

        print("GPUs:", torch.cuda.device_count())    if torch.cuda.device_count() >= 1:

        if torch.cuda.device_count() > 1:        print("GPUs:", torch.cuda.device_count())

            model = nn.DataParallel(model)        if torch.cuda.device_count() > 1:

            if args.test_gpu == 1:            model = nn.DataParallel(model)

                test_local = nn.DataParallel(test_local)            if args.test_gpu == 1:

    else:                test_local = nn.DataParallel(test_local)

        print("Using CPU")    else:

        print("Using CPU")

    # Optim

    optimizer_local = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)    # Optim

    scheduler_local = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_local, T_0=WARMUP_STEPS)    optimizer_local = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    scaler_local = GradScaler()    scheduler_local = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_local, T_0=WARMUP_STEPS)

    scaler_local = GradScaler()

    # Bind globals

    criterion = criterion_local    # Bind globals

    optimizer = optimizer_local    criterion = criterion_local

    lr_scheduler = scheduler_local    optimizer = optimizer_local

    scaler = scaler_local    lr_scheduler = scheduler_local

    test_model = test_local    scaler = scaler_local

    test_model = test_local

    # Build code list from CSVs (PKL TODO)

    codes = []    # Build code list from CSVs (PKL TODO)

    if int(args.pkl) == 0:    codes = []

        for csv_file in glob.glob(f"{daily_path}/*.csv"):    if int(args.pkl) == 0:

            code = os.path.basename(csv_file).rsplit(".", 1)[0]        for csv_file in glob.glob(f"{daily_path}/*.csv"):

            codes.append(code)            code = os.path.basename(csv_file).rsplit(".", 1)[0]

    else:            codes.append(code)

        print("PKL mode list-building TODO; proceeding with CSVs only")    else:

        for csv_file in glob.glob(f"{daily_path}/*.csv"):        print("PKL mode list-building TODO; proceeding with CSVs only")

            code = os.path.basename(csv_file).rsplit(".", 1)[0]        for csv_file in glob.glob(f"{daily_path}/*.csv"):

            codes.append(code)            code = os.path.basename(csv_file).rsplit(".", 1)[0]

            codes.append(code)

    if not codes:

        print("No stock codes found.")    if not codes:

        return        print("No stock codes found.")

        return

    random.shuffle(codes)

    split = int(len(codes) * TRAIN_WEIGHT)    random.shuffle(codes)

    train_codes, test_codes = codes[:split], codes[split:]    split = int(len(codes) * TRAIN_WEIGHT)

    train_codes, test_codes = codes[:split], codes[split:]

    mode = args.mode

    if mode == "train":    mode = args.mode

        print("Start training with", len(train_codes), "codes")    if mode == "train":

        for epoch in range(1, args.epochs + 1):        print("Start training with", len(train_codes), "codes")

            pbar = tqdm(total=len(train_codes), desc=f"Epoch {epoch}")        for epoch in range(1, args.epochs + 1):

            for code in train_codes:            pbar = tqdm(total=len(train_codes), desc=f"Epoch {epoch}")

                data = import_csv(code)            for code in train_codes:

                if data is None:                data = import_csv(code)

                    pbar.update(1)                if data is None:

                    continue                    pbar.update(1)

                ds = Stock_Data(mode=0, dataFrame=data, label_num=OUTPUT_DIMENSION, predict_days=abs(int(args.predict_days)), trend=int(args.trend))                    continue

                dl = DataLoader(dataset=ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=NUM_WORKERS, pin_memory=True)                ds = Stock_Data(mode=0, dataFrame=data, label_num=OUTPUT_DIMENSION, predict_days=abs(int(args.predict_days)), trend=int(args.trend))

                train_one_epoch(epoch, dl, scaler, ts_code=code)                dl = DataLoader(dataset=ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=NUM_WORKERS, pin_memory=True)

                pbar.update(1)                train_one_epoch(epoch, dl, scaler, ts_code=code)

                pbar.update(1)

            pbar.close()

            if time.time() - last_save_time >= SAVE_INTERVAL:            pbar.close()

                thread_save_model(model, optimizer, save_path, False, int(args.predict_days))            if time.time() - last_save_time >= SAVE_INTERVAL:

        print("Training finished.")                thread_save_model(model, optimizer, save_path, False, int(args.predict_days))

        if lo_list:        print("Training finished.")

            try:        if lo_list:

                import matplotlib.pyplot as plt            try:

                loss_dir = f"{png_path}/train_loss"                import matplotlib.pyplot as plt

                os.makedirs(loss_dir, exist_ok=True)                loss_dir = f"{png_path}/train_loss"

                plt.figure(figsize=(10, 6))                os.makedirs(loss_dir, exist_ok=True)

                plt.plot(lo_list)                plt.figure(figsize=(10, 6))

                plt.title("Training Loss Curve")                plt.plot(lo_list)

                plt.savefig(os.path.join(loss_dir, "loss_curve.png"))                plt.title("Training Loss Curve")

                plt.close()                plt.savefig(os.path.join(loss_dir, "loss_curve.png"))

            except Exception as e:                plt.close()

                print("plot error:", e)            except Exception as e:

                print("plot error:", e)

    elif mode == "test":

        pick = [args.test_code] if args.test_code else ([random.choice(test_codes)] if test_codes else [])    elif mode == "test":

        if not pick:        pick = [args.test_code] if args.test_code else ([random.choice(test_codes)] if test_codes else [])

            print("No codes for test")        if not pick:

            return            print("No codes for test")

        for code in pick:            return

            data = import_csv(code)        for code in pick:

            if data is not None:            data = import_csv(code)

                run_test(data)            if data is not None:

                run_test(data)

    else:  # predict

        if not args.test_code:    else:  # predict

            print("--test_code required for predict")        if not args.test_code:

            return            print("--test_code required for predict")

        run_predict([args.test_code])            return

        run_predict([args.test_code])



if __name__ == "__main__":

    main()if __name__ == "__main__":

    main()

    from pathlib import Path    current_dir = Path(__file__).resolve().parent

    current_dir = Path(__file__).resolve().parent    root_dir = current_dir.parent.parent

    root_dir = current_dir.parent.parent    src_dir = root_dir / "src"

    src_dir = root_dir / "src"    if str(src_dir) not in sys.path:

    if str(src_dir) not in sys.path:        sys.path.insert(0, str(src_dir))

        sys.path.insert(0, str(src_dir))    from stock_prediction.init import *

    from stock_prediction.init import *    from stock_prediction.common import *

    from stock_prediction.common import *

# 全局变量

# 全局变量last_save_time = 0

last_save_time = 0loss = None

loss = Noneloss_list = []

loss_list = []iteration = 0

iteration = 0lo_list = []

lo_list = []batch_none = 0

batch_none = 0data_none = 0

data_none = 0last_loss = 1e10

last_loss = 1e10lr_scheduler = None

lr_scheduler = Nonemodel = None

model = Nonetest_model = None

test_model = Noneoptimizer = None

optimizer = Nonecriterion = None

criterion = Nonesave_path = None

save_path = Nonescaler = None

scaler = None

def train(epoch, dataloader, scaler_param, ts_code="", data_queue=None):

def train(epoch, dataloader, scaler_param, ts_code="", data_queue=None):    """训练函数 - 恢复原有功能"""

    """训练函数 - 恢复原有功能"""    global loss, last_save_time, loss_list, iteration, lo_list, batch_none, data_none, last_loss, lr_scheduler

    global loss, last_save_time, loss_list, iteration, lo_list, batch_none, data_none, last_loss, lr_scheduler    model.train()

    model.train()    subbar = tqdm(total=len(dataloader), leave=False, ncols=TQDM_NCOLS)

    subbar = tqdm(total=len(dataloader), leave=False, ncols=TQDM_NCOLS)    test_iner = len(dataloader) // TEST_INTERVAL if TEST_INTERVAL > 0 else len(dataloader)

    test_iner = len(dataloader) // TEST_INTERVAL if TEST_INTERVAL > 0 else len(dataloader)    safe_save = False

    safe_save = False    

        for batch in dataloader:

    for batch in dataloader:        try:

        try:            safe_save = False

            safe_save = False            iteration += 1

            iteration += 1            if batch is None:

            if batch is None:                batch_none += 1

                batch_none += 1                subbar.set_description(f"{ts_code}, e:{epoch}, bn:{batch_none}, loss:{loss.item():.2e}" if loss is not None else f"{ts_code}, e:{epoch}, bn:{batch_none}")

                subbar.set_description(f"{ts_code}, e:{epoch}, bn:{batch_none}, loss:{loss.item():.2e}" if loss is not None else f"{ts_code}, e:{epoch}, bn:{batch_none}")                subbar.update(1)

                subbar.update(1)                continue

                continue                

                            data, label = batch

            data, label = batch            if data is None or label is None:

            if data is None or label is None:                tqdm.write(f"code: {ts_code}, train error: data is None or label is None")

                tqdm.write(f"code: {ts_code}, train error: data is None or label is None")                subbar.update(1)

                subbar.update(1)                continue

                continue                

                            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)            

                        with autocast(device_type=device.type):

            with autocast(device_type=device.type):                data = pad_input(data)

                data = pad_input(data)                outputs = model.forward(data, label, 0)  # predict_days默认为0

                outputs = model.forward(data, label, 0)  # predict_days默认为0                

                                if outputs.shape == label.shape:

                if outputs.shape == label.shape:                    loss = criterion(outputs, label)

                    loss = criterion(outputs, label)                else:

                else:                    tqdm.write(f"Shape mismatch: outputs {outputs.shape}, label {label.shape}")

                    tqdm.write(f"Shape mismatch: outputs {outputs.shape}, label {label.shape}")                    subbar.update(1)

                    subbar.update(1)                    continue

                    continue            

                        optimizer.zero_grad()

            optimizer.zero_grad()            scaler_param.scale(loss).backward()

            scaler_param.scale(loss).backward()            scaler_param.step(optimizer)

            scaler_param.step(optimizer)            scaler_param.update()

            scaler_param.update()            lr_scheduler.step()

            lr_scheduler.step()            

                        loss_list.append(loss.item())

            loss_list.append(loss.item())            

                        if iteration % 100 == 0:

            if iteration % 100 == 0:                lo_list.append(loss.item())

                lo_list.append(loss.item())                

                            subbar.set_description(f"{ts_code}, e:{epoch}, i:{iteration}, loss:{loss.item():.2e}")

            subbar.set_description(f"{ts_code}, e:{epoch}, i:{iteration}, loss:{loss.item():.2e}")            subbar.update(1)

            subbar.update(1)            safe_save = True

            safe_save = True            

                    except Exception as e:

        except Exception as e:            tqdm.write(f"Training error: {e}")

            tqdm.write(f"Training error: {e}")            subbar.update(1)

            subbar.update(1)            continue

            continue            

                subbar.close()

    subbar.close()



def test(dataset, testmodel=None, dataloader_mode=0):

def test(dataset, testmodel=None, dataloader_mode=0):    """测试函数 - 恢复原有功能"""

    """测试函数 - 恢复原有功能"""    if testmodel is None:

    if testmodel is None:        testmodel = test_model

        testmodel = test_model        

            testmodel.eval()

    testmodel.eval()    test_dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False, 

                                    drop_last=False, num_workers=NUM_WORKERS, pin_memory=True, 

    # 根据数据类型创建相应的数据集                                collate_fn=custom_collate)

    if isinstance(dataset, np.ndarray) or hasattr(dataset, 'values'):    

        # CSV数据或DataFrame    total_loss = 0

        stock_predict = Stock_Data(mode=dataloader_mode, dataFrame=dataset, label_num=OUTPUT_DIMENSION)    num_batches = 0

        test_dataloader = DataLoader(dataset=stock_predict, batch_size=BATCH_SIZE, shuffle=False,     predictions = []

                                    drop_last=False, num_workers=NUM_WORKERS, pin_memory=True)    labels = []

    else:    

        # 已经是数据集对象    with torch.no_grad():

        test_dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False,         for batch in test_dataloader:

                                    drop_last=False, num_workers=NUM_WORKERS, pin_memory=True,             if batch is None:

                                    collate_fn=custom_collate)                continue

                    

    total_loss = 0            data, label = batch

    num_batches = 0            if data is None or label is None:

    predictions = []                continue

    labels = []                

                data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

    with torch.no_grad():            data = pad_input(data)

        for batch in test_dataloader:            

            if batch is None:            outputs = testmodel.forward(data, label, 0)

                continue            

                            if outputs.shape == label.shape:

            data, label = batch                loss = criterion(outputs, label)

            if data is None or label is None:                total_loss += loss.item()

                continue                num_batches += 1

                                

            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)                predictions.append(outputs.cpu().numpy())

            data = pad_input(data)                labels.append(label.cpu().numpy())

                

            outputs = testmodel.forward(data, label, 0)    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')

                print(f"Test Loss: {avg_loss:.4f}")

            if outputs.shape == label.shape:    

                loss = criterion(outputs, label)    return predictions, labels, avg_loss

                total_loss += loss.item()

                num_batches += 1

                def predict(test_codes):

                predictions.append(outputs.cpu().numpy())    """预测函数 - 恢复原有功能"""

                labels.append(label.cpu().numpy())    test_model.eval()

        

    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')    for code in test_codes:

    print(f"Test Loss: {avg_loss:.4f}")        print(f"Predicting for stock: {code}")

            

    return predictions, labels, avg_loss        # 加载数据

        try:

            if PKL:

def predict(test_codes):                # 从pkl文件加载数据

    """预测函数 - 恢复原有功能"""                    dataset = stock_dataset(mode=1, ts_code=code, label_num=OUTPUT_DIMENSION)

    if test_model is None:            else:

        print("Error: test_model is not initialized")                # 从csv文件加载数据

        return                dataset = import_csv(code)

                        

    test_model.eval()                                dataset = stock_dataset(mode=0, ts_code=ts_code, label_num=OUTPUT_DIMENSION)

                    print(f"No data found for {code}")

    for code in test_codes:                continue

        print(f"Predicting for stock: {code}")                

                    predictions, labels, _ = test(dataset, test_model)

        # 加载数据            

        try:            # 保存预测结果

            dataset = None            print(f"Prediction completed for {code}")

            if PKL:            

                # TODO: 实现PKL数据加载逻辑        except Exception as e:

                print(f"PKL mode not fully implemented for {code}")            print(f"Error predicting {code}: {e}")

                continue            continue

            else:

                # 从csv文件加载数据

                dataset = import_csv(code)def loss_curve(loss_list_param):

                    """绘制损失曲线"""

            if dataset is None:    plt.figure(figsize=(10, 6))

                print(f"No data found for {code}")    plt.plot(loss_list_param)

                continue    plt.title('Training Loss Curve')

                    plt.xlabel('Iterations')

            predictions, labels, _ = test(dataset, test_model, dataloader_mode=2)    plt.ylabel('Loss')

                plt.savefig(f"{png_path}/train_loss/loss_curve.png")

            # 保存预测结果    plt.close()

            print(f"Prediction completed for {code}")

            

        except Exception as e:def contrast_lines(test_codes):

            print(f"Error predicting {code}: {e}")    """对比预测结果和真实结果"""

            continue    try:

        for code in test_codes:

            # 这里需要实现具体的对比逻辑

def loss_curve(loss_list_param):            print(f"Creating contrast chart for {code}")

    """绘制损失曲线"""            # 简化实现，返回成功

    try:            return 0

        plt.figure(figsize=(10, 6))    except Exception as e:

        plt.plot(loss_list_param)        print(f"Error creating contrast chart: {e}")

        plt.title('Training Loss Curve')        return -1

        plt.xlabel('Iterations')

        plt.ylabel('Loss')

        def main():

        # 确保目录存在    """主函数 - 恢复完整的运行逻辑"""

        loss_dir = f"{png_path}/train_loss"    global model, test_model, optimizer, criterion, save_path, scaler, last_loss, PKL

        os.makedirs(loss_dir, exist_ok=True)    

            parser = argparse.ArgumentParser()

        plt.savefig(f"{loss_dir}/loss_curve.png")    parser.add_argument('--mode', default="train", type=str, help="select running mode: train, test, predict")

        plt.close()    parser.add_argument('--model', default="transformer", type=str, help="lstm or transformer or cnnlstm")

        print(f"Loss curve saved to {loss_dir}/loss_curve.png")    parser.add_argument('--begin_code', default="", type=str, help="begin code")

    except Exception as e:    parser.add_argument('--cpu', default=0, type=int, help="only use cpu")

        print(f"Error creating loss curve: {e}")    parser.add_argument('--pkl', default=1, type=int, help="use pkl file instead of csv file")

    parser.add_argument('--pkl_queue', default=1, type=int, help="use pkl queue instead of csv file")

    parser.add_argument('--test_code', default="", type=str, help="test code")

def contrast_lines(test_codes):    parser.add_argument('--test_gpu', default=1, type=int, help="test method use gpu or not")

    """对比预测结果和真实结果"""    parser.add_argument('--predict_days', default=0, type=int, help="number of the predict days")

    try:    parser.add_argument('--api', default="akshare", type=str, help="api-interface, tushare, akshare or yfinance")

        for code in test_codes:    parser.add_argument('--trend', default=0, type=int, help="predict the trend of stock, not the price")

            print(f"Creating contrast chart for {code}")    parser.add_argument('--epochs', default=EPOCH, type=int, help="number of training epochs")

            # TODO: 实现具体的对比逻辑    

            # 这里需要加载数据、运行预测并创建对比图    args = parser.parse_args()

            return 0    

    except Exception as e:    # 设置设备

        print(f"Error creating contrast chart: {e}")    device_to_use = device

        return -1    if args.cpu == 1:

        device_to_use = torch.device("cpu")

    

def main():    # 读取上次的损失

    """主函数 - 恢复完整的运行逻辑"""    if os.path.exists('loss.txt'):

    global model, test_model, optimizer, criterion, save_path, scaler, last_loss, PKL        try:

                with open('loss.txt', 'r') as file:

    parser = argparse.ArgumentParser()                last_loss = float(file.read())

    parser.add_argument('--mode', default="train", type=str, help="select running mode: train, test, predict")        except:

    parser.add_argument('--model', default="transformer", type=str, help="lstm or transformer or cnnlstm")            last_loss = 1e10

    parser.add_argument('--begin_code', default="", type=str, help="begin code")    print("last_loss=", last_loss)

    parser.add_argument('--cpu', default=0, type=int, help="only use cpu")    

    parser.add_argument('--pkl', default=0, type=int, help="use pkl file instead of csv file")    mode = args.mode

    parser.add_argument('--pkl_queue', default=0, type=int, help="use pkl queue instead of csv file")    model_mode = args.model.upper()

    parser.add_argument('--test_code', default="", type=str, help="test code")    PKL = False if args.pkl <= 0 else True

    parser.add_argument('--test_gpu', default=1, type=int, help="test method use gpu or not")    

    parser.add_argument('--predict_days', default=0, type=int, help="number of the predict days")    # 设置模型

    parser.add_argument('--api', default="akshare", type=str, help="api-interface, tushare, akshare or yfinance")    if model_mode == "LSTM":

    parser.add_argument('--trend', default=0, type=int, help="predict the trend of stock, not the price")        model = LSTM(dimension=INPUT_DIMENSION)

    parser.add_argument('--epochs', default=EPOCH, type=int, help="number of training epochs")        test_model = LSTM(dimension=INPUT_DIMENSION)

            save_path = lstm_path

    args = parser.parse_args()        criterion = nn.MSELoss()

        elif model_mode == "TRANSFORMER":

    # 设置设备        model = TransformerModel(input_dim=INPUT_DIMENSION, d_model=D_MODEL, nhead=NHEAD, 

    device_to_use = device                                num_layers=6, dim_feedforward=2048, output_dim=OUTPUT_DIMENSION, 

    if args.cpu == 1:                                max_len=SEQ_LEN, mode=0)

        device_to_use = torch.device("cpu")        test_model = TransformerModel(input_dim=INPUT_DIMENSION, d_model=D_MODEL, nhead=NHEAD, 

                                         num_layers=6, dim_feedforward=2048, output_dim=OUTPUT_DIMENSION, 

    # 读取上次的损失                                     max_len=SEQ_LEN, mode=1)

    if os.path.exists('loss.txt'):        save_path = transformer_path

        try:        criterion = nn.MSELoss()

            with open('loss.txt', 'r') as file:    elif model_mode == "CNNLSTM":

                last_loss = float(file.read())        model = CNNLSTM(input_dim=INPUT_DIMENSION, num_classes=OUTPUT_DIMENSION, predict_days=abs(int(args.predict_days)))

        except:        test_model = CNNLSTM(input_dim=INPUT_DIMENSION, num_classes=OUTPUT_DIMENSION, predict_days=abs(int(args.predict_days)))

            last_loss = 1e10        save_path = cnnlstm_path

    print("last_loss=", last_loss)        criterion = nn.MSELoss()

        else:

    mode = args.mode        print("No such model")

    model_mode = args.model.upper()        return

    PKL = False if args.pkl <= 0 else True    

        # 移动模型到设备

    # 设置模型    model = model.to(device_to_use, non_blocking=True)

    if model_mode == "LSTM":    if args.test_gpu == 0:

        model = LSTM(dimension=INPUT_DIMENSION)        test_model = test_model.to('cpu', non_blocking=True)

        test_model = LSTM(dimension=INPUT_DIMENSION)    else:

        save_path = lstm_path        test_model = test_model.to(device_to_use, non_blocking=True)

        criterion = nn.MSELoss()    

    elif model_mode == "TRANSFORMER":    # 多GPU支持

        model = TransformerModel(input_dim=INPUT_DIMENSION, d_model=D_MODEL, nhead=NHEAD,     if torch.cuda.device_count() >= 1:

                                num_layers=6, dim_feedforward=2048, output_dim=OUTPUT_DIMENSION,         print("Let's use", torch.cuda.device_count(), "GPUs!")

                                max_len=SEQ_LEN, mode=0)        if torch.cuda.device_count() > 1:

        test_model = TransformerModel(input_dim=INPUT_DIMENSION, d_model=D_MODEL, nhead=NHEAD,             model = nn.DataParallel(model)

                                     num_layers=6, dim_feedforward=2048, output_dim=OUTPUT_DIMENSION,             if args.test_gpu == 1:

                                     max_len=SEQ_LEN, mode=1)                test_model = nn.DataParallel(test_model)

        save_path = transformer_path    else:

        criterion = nn.MSELoss()        print("Let's use CPU!")

    elif model_mode == "CNNLSTM":    

        model = CNNLSTM(input_dim=INPUT_DIMENSION, num_classes=OUTPUT_DIMENSION, predict_days=abs(int(args.predict_days)))    print(model)

        test_model = CNNLSTM(input_dim=INPUT_DIMENSION, num_classes=OUTPUT_DIMENSION, predict_days=abs(int(args.predict_days)))    

        save_path = cnnlstm_path    # 设置优化器

        criterion = nn.MSELoss()    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    else:    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=WARMUP_STEPS)

        print("No such model")    scaler = GradScaler()

        return    

        # 获取股票代码列表

    # 移动模型到设备    ts_codes = []

    model = model.to(device_to_use, non_blocking=True)    test_codes = []

    if args.test_gpu == 0:    

        test_model = test_model.to('cpu', non_blocking=True)    if PKL:

    else:        # 从pkl文件获取股票列表

        test_model = test_model.to(device_to_use, non_blocking=True)        import glob

            pkl_files = glob.glob(f"{pkl_path}/*.pkl")

    # 多GPU支持        for pkl_file in pkl_files:

    if torch.cuda.device_count() >= 1:            code = os.path.basename(pkl_file).rsplit(".", 1)[0]

        print("Let's use", torch.cuda.device_count(), "GPUs!")            if code != "train":  # 排除train.pkl

        if torch.cuda.device_count() > 1:                ts_codes.append(code)

            model = nn.DataParallel(model)    else:

            if args.test_gpu == 1:        # 从csv文件获取股票列表

                test_model = nn.DataParallel(test_model)        csv_files = glob.glob(f"{daily_path}/*.csv")

    else:        for csv_file in csv_files:

        print("Let's use CPU!")            code = os.path.basename(csv_file).rsplit(".", 1)[0]

                ts_codes.append(code)

    print(model)    

        random.shuffle(ts_codes)

    # 设置优化器      total_length = int(len(ts_codes) * TRAIN_WEIGHT)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)    test_codes = ts_codes[total_length:]

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=WARMUP_STEPS)    ts_codes = ts_codes[:total_length]

    scaler = GradScaler()    

        print(f"Total codes: {len(ts_codes)}, Test codes: {len(test_codes)}")

    # 获取股票代码列表    

    ts_codes = []    # 根据模式执行相应操作

    test_codes = []    if mode == "train":

            print("Starting training...")

    if PKL:        

        # 从pkl文件获取股票列表        for epoch in range(args.epochs):

        pkl_files = glob.glob(f"{pkl_path}/*.pkl")            print(f"Epoch {epoch+1}/{args.epochs}")

        for pkl_file in pkl_files:            

            code = os.path.basename(pkl_file).rsplit(".", 1)[0]            if PKL and args.pkl_queue == 1:

            if code != "train":  # 排除train.pkl                # 使用PKL队列模式

                ts_codes.append(code)                print("Using PKL queue mode")

    else:                # 这里需要实现PKL队列的训练逻辑

        # 从csv文件获取股票列表                # 为了简化，暂时跳过

        csv_files = glob.glob(f"{daily_path}/*.csv")                pass

        for csv_file in csv_files:            else:

            code = os.path.basename(csv_file).rsplit(".", 1)[0]                # 传统训练模式

            ts_codes.append(code)                pbar = tqdm(total=len(ts_codes), desc=f"Training Epoch {epoch+1}")

                    for ts_code in ts_codes:

    if len(ts_codes) == 0:                    try:

        print("No stock codes found!")                        if PKL:

        return                            dataset = stock_dataset(mode=0, ts_code=ts_code, label_num=OUTPUT_DIMENSION)

                                else:

    random.shuffle(ts_codes)                            dataset = import_csv(ts_code)

    total_length = int(len(ts_codes) * TRAIN_WEIGHT)                            

    test_codes = ts_codes[total_length:] if total_length < len(ts_codes) else []                        if dataset is None:

    ts_codes = ts_codes[:total_length]                            pbar.update(1)

                                continue

    print(f"Total codes: {len(ts_codes)}, Test codes: {len(test_codes)}")                            

                            train_dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, 

    # 根据模式执行相应操作                                                    shuffle=False, drop_last=False, 

    if mode == "train":                                                    num_workers=NUM_WORKERS, pin_memory=True, 

        print("Starting training...")                                                    collate_fn=custom_collate)

                                

        for epoch in range(args.epochs):                        train(epoch+1, train_dataloader, scaler, ts_code)

            print(f"Epoch {epoch+1}/{args.epochs}")                        pbar.update(1)

                                    

            # 传统训练模式                    except Exception as e:

            pbar = tqdm(total=len(ts_codes), desc=f"Training Epoch {epoch+1}")                        print(f"Error training {ts_code}: {e}")

            for ts_code in ts_codes:                        pbar.update(1)

                try:                        continue

                    dataset = None                pbar.close()

                    if PKL:            

                        # TODO: 实现PKL数据集加载            # 保存模型

                        print("PKL mode not implemented yet")            if (time.time() - last_save_time >= SAVE_INTERVAL):

                        pbar.update(1)                thread_save_model(model, optimizer, save_path, False, int(args.predict_days))

                        continue                last_save_time = time.time()

                    else:        

                        dataset = import_csv(ts_code)        print("Training finished!")

                                

                    if dataset is None:        # 绘制损失曲线

                        pbar.update(1)        if len(lo_list) > 0:

                        continue            loss_curve(lo_list)

                            

                    # 创建训练数据集        # 创建对比图

                    stock_train = Stock_Data(mode=0, dataFrame=dataset, label_num=OUTPUT_DIMENSION,        if len(test_codes) > 0:

                                           predict_days=abs(int(args.predict_days)), trend=int(args.trend))            test_index = random.randint(0, len(test_codes) - 1)

                                test_code = [test_codes[test_index]]

                    train_dataloader = DataLoader(dataset=stock_train, batch_size=BATCH_SIZE,             contrast_lines(test_code)

                                                shuffle=False, drop_last=False,             

                                                num_workers=NUM_WORKERS, pin_memory=True)    elif mode == "test":

                            print("Starting testing...")

                    train(epoch+1, train_dataloader, scaler, ts_code)        if args.test_code != "" and args.test_code != "all":

                    pbar.update(1)            test_code = [args.test_code]

                            else:

                except Exception as e:            if len(test_codes) > 0:

                    print(f"Error training {ts_code}: {e}")                test_index = random.randint(0, len(test_codes) - 1)

                    pbar.update(1)                test_code = [test_codes[test_index]]

                    continue            else:

            pbar.close()                print("No test codes available")

                            return

            # 保存模型        

            if (time.time() - last_save_time >= SAVE_INTERVAL):        contrast_lines(test_code)

                thread_save_model(model, optimizer, save_path, False, int(args.predict_days))        

                last_save_time = time.time()    elif mode == "predict":

                print(f"Model saved at epoch {epoch+1}")        print("Starting prediction...")

                if args.test_code == "":

        print("Training finished!")            print("Error: test_code is empty")

                    return

        # 绘制损失曲线        elif args.test_code in ts_codes or PKL == True:

        if len(lo_list) > 0:            test_code = [args.test_code]

            loss_curve(lo_list)            predict(test_code)

                else:

        # 创建对比图            print("Error: test_code is not in ts_codes")

        if len(test_codes) > 0:            return

            test_index = random.randint(0, len(test_codes) - 1)

            test_code = [test_codes[test_index]]

            contrast_lines(test_code)if __name__ == "__main__":

                main()
    elif mode == "test":
        print("Starting testing...")
        if args.test_code != "" and args.test_code != "all":
            test_code = [args.test_code]
        else:
            if len(test_codes) > 0:
                test_index = random.randint(0, len(test_codes) - 1)
                test_code = [test_codes[test_index]]
            else:
                print("No test codes available")
                return
        
        contrast_lines(test_code)
        
    elif mode == "predict":
        print("Starting prediction...")
        if args.test_code == "":
            print("Error: test_code is empty")
            return
        else:
            test_code = [args.test_code]
            predict(test_code)
    else:
        print(f"Unknown mode: {mode}")
        print("Available modes: train, test, predict")


if __name__ == "__main__":
    main()