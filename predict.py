#!/usr/bin/env python
# coding: utf-8
import random
import argparse
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from common import *

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default="train", type=str, help="select running mode: train, test, predict")
parser.add_argument('--model', default="transformer", type=str, help="lstm or transformer")
parser.add_argument('--begin_code', default="", type=str, help="begin code")
parser.add_argument('--cpu', default=0, type=int, help="only use cpu")
parser.add_argument('--pkl', default=1, type=int, help="use pkl file instead of csv file")
parser.add_argument('--pkl_queue', default=1, type=int, help="use pkl queue instead of csv file")
parser.add_argument('--test_code', default="", type=str, help="test code")
parser.add_argument('--test_gpu', default=1, type=int, help="test method use gpu or not")
parser.add_argument('--predict_days', default=0, type=int, help="number of the predict days,Positive numbers use interval prediction algorithm, 0 and negative numbers use date prediction algorithm")
parser.add_argument('--api', default="akshare", type=str, help="api-interface, tushare, akshare or yfinance")
args = parser.parse_args()
last_save_time = 0

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

def train(epoch, dataloader, scaler, ts_code="", data_queue=None):
    global loss, last_save_time, loss_list, iteration, lo_list, batch_none, data_none, last_loss, lr_scheduler
    model.train()
    subbar = tqdm(total=len(dataloader), leave=False, ncols=TQDM_NCOLS)
    test_iner = len(dataloader) // TEST_INTERVAL
    safe_save = False
    for batch in dataloader:
        try:
            safe_save = False
            iteration += 1
            if batch is None:
                batch_none += 1
                subbar.set_description(f"{ts_code}, e:{epoch}, bn:{batch_none}, loss:{loss.item():.2e}")
                subbar.update(1)
                continue
            data, label = batch
            if data is None or label is None:
                tqdm.write(f"code: {ts_code}, train error: data is None or label is None")
                subbar.update(1)
                continue
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)
            with autocast():
                if args.model == 'transformer':
                    data = pad_input(data)
                outputs = model.forward(data, label, int(args.predict_days))
                if outputs.shape == label.shape:
                    loss = criterion(outputs, label)
                else:
                    _label = label.reshape(outputs.shape)
                    if outputs.shape == _label.shape:
                        loss = criterion(outputs, _label)
                    else:
                        tqdm.write(f"code: {ts_code}, train error: outputs.shape != label.shape")
                        subbar.update(1)
                        continue
                
            optimizer.zero_grad()
            if device.type == "cuda":
                scaler.scale(loss).backward()
                lr_scheduler.step()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                lr_scheduler.step()
                optimizer.step()
            if is_number(str(loss.item())):
                loss_list.append(loss.item())
                lo_list.append(loss.item())

            subbar.set_description(f"{ts_code}, e:{epoch}, bn:{batch_none}, loss:{loss.item():.2e}")
            subbar.update(1)
            safe_save = True
        except Exception as e:
            print(f"code: {ts_code}, train error: {e}")
            safe_save = False
            subbar.update(1)
            continue
        if (TEST_INTERVAL > 0 and iteration % test_iner == 0):
            testmodel = copy.deepcopy(model)
            test_loss, predict_list, _ = test(data_queue, testmodel, dataloader_mode=1)
            if last_loss > test_loss:
                last_loss = test_loss
                thread_save_model(model, optimizer, save_path, True, int(args.predict_days))
                with open('loss.txt', 'w') as file:
                    file.write(str(last_loss))

        if (iteration % SAVE_NUM_ITER == 0 and time.time() - last_save_time >= SAVE_INTERVAL)  and safe_save == True:
            thread_save_model(model, optimizer, save_path, False, int(args.predict_days))
            last_save_time = time.time()
        

    if (epoch % SAVE_NUM_EPOCH == 0 or epoch == EPOCH) and time.time() - last_save_time >= SAVE_INTERVAL and safe_save == True:
        thread_save_model(model, optimizer, save_path, False, int(args.predict_days))
        last_save_time = time.time()
    testmodel = copy.deepcopy(model)
    test_loss, predict_list, _ = test(data_queue, testmodel, dataloader_mode=1)
    if last_loss > test_loss:
        last_loss = test_loss
        thread_save_model(model, optimizer, save_path, True, int(args.predict_days))
        with open('loss.txt', 'w') as file:
            file.write(str(last_loss))

    subbar.close()


def test(dataset, testmodel=None, dataloader_mode=0):
    global test_model
    predict_list = []
    accuracy_list = []
    if dataloader_mode in [0, 2]:
        stock_predict = Stock_Data(mode=dataloader_mode, dataFrame=dataset, label_num=OUTPUT_DIMENSION,predict_days=int(args.predict_days))
        dataloader = DataLoader(dataset=stock_predict, batch_size=BATCH_SIZE, shuffle=False, drop_last=drop_last, num_workers=NUM_WORKERS, pin_memory=True)
    elif dataloader_mode in [1]:
        _stock_test_data_queue = deep_copy_queue(dataset)
        stock_test = stock_queue_dataset(mode=1, data_queue=_stock_test_data_queue, label_num=OUTPUT_DIMENSION, buffer_size=BUFFER_SIZE, total_length=total_test_length,predict_days=int(args.predict_days))
        dataloader=DataLoader(dataset=stock_test,batch_size=BATCH_SIZE,shuffle=False,drop_last=drop_last, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=custom_collate)
    elif dataloader_mode in [3]:
        stock_predict = Stock_Data(mode=1, dataFrame=dataset, label_num=OUTPUT_DIMENSION,predict_days=int(args.predict_days))
        dataloader = DataLoader(dataset=stock_predict, batch_size=BATCH_SIZE, shuffle=False, drop_last=drop_last, num_workers=NUM_WORKERS, pin_memory=True)

    if testmodel is None:
        if int(args.predict_days) > 0:
            if os.path.exists(save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_pre" + str(args.predict_days) + "_Model.pkl") and os.path.exists(save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_pre" + str(args.predict_days) + "_Optimizer.pkl"):
                test_model.load_state_dict(torch.load(save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_pre" + str(args.predict_days) + "_Model.pkl"))
            else:
                tqdm.write("No model found")
                return -1, -1, -1
        else:
            if os.path.exists(save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_Model.pkl") and os.path.exists(save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_Optimizer.pkl"):
                test_model.load_state_dict(torch.load(save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_Model.pkl"))
            else:
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
                data, label = batch
                if data is None or label is None:
                    # tqdm.write(f"test error: data is None or label is None")
                    pbar.update(1)
                    continue
                if args.test_gpu == 1:
                    data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)
                else:
                    data, label = data.to("cpu", non_blocking=True), label.to("cpu", non_blocking=True)
                # test_optimizer.zero_grad()
                if args.model == 'transformer':
                    data = pad_input(data)
                predict = test_model.forward(data, label, int(args.predict_days))
                predict_list.append(predict)
                if(predict.shape == label.shape):
                    accuracy = accuracy_fn(predict, label)
                    accuracy_list.append(accuracy.item())
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
            data_queue = dill.load(f)
        while data_queue.empty() == False:
            try:
                item = data_queue.get(timeout=30)
                if str(item['ts_code'][0]).zfill(6) in test_codes:
                    _data = item
                    break
            except queue.Empty:
                break
        data_queue = queue.Queue()
        data = copy.deepcopy(_data)

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

    predict_data = copy.deepcopy(data)
    spliced_data = copy.deepcopy(data)
    show_days = 7
    real_list = []
    prediction_list = []
    if predict_data.empty or predict_data is None:
        print("Error: Train_data or Test_data is None")
        return
    current_date = predict_data["Date"][0]
    if int(args.predict_days) <= 0:
        predict_days = abs(int(args.predict_days))
        pbar = tqdm(total=predict_days, leave=False, ncols=TQDM_NCOLS)
        while predict_days > 0:
            lastdate = predict_data["Date"][0].strftime("%Y%m%d")
            if args.api == "tushare":
                lastclose = predict_data["Close"][0]
            predict_data.drop(['ts_code', 'Date'], axis=1, inplace=True)
            # predict_data = predict_data.dropna()
            predict_data = predict_data.fillna(-0.0)
            accuracy_list, predict_list = [], []
            test_loss, predict_list, _ = test(predict_data,dataloader_mode=2)
            if test_loss == -1 and predict_list == -1:
                return
            _tmp = []
            prediction_list = []
            for items in predict_list:
                items=items.to("cpu", non_blocking=True)
                for idxs in items:
                    _tmp = []
                    for index, item in enumerate(idxs):
                        if use_list[index] == 1:
                            _tmp.append((item*std_list[index]+mean_list[index]).detach().numpy())
            date_str = lastdate
            date_obj = datetime.strptime(date_str, "%Y%m%d")
            new_date_obj = date_obj + timedelta(days=1)
            # date_string = new_date_obj.strftime("%Y%m%d")
            _tmpdata = [test_codes[0], new_date_obj]
            _tmpdata = _tmpdata + copy.deepcopy(_tmp)
            _splice_data = copy.deepcopy(spliced_data).drop(['ts_code', 'Date'], axis=1)
            df_mean = _splice_data.mean().tolist()
            if args.api == "tushare":
                for index in range(len(_tmpdata) - 2, len(df_mean)-1):
                    _tmpdata.append(df_mean[index])
                _tmpdata.append(lastclose)
            elif args.api == "akshare" or args.api == "yfinance":
                for index in range(len(_tmpdata) - 2, len(df_mean)):
                    _tmpdata.append(-0.0)
            _tmpdata = pd.DataFrame(_tmpdata).T
            _tmpdata.columns = spliced_data.columns
            predict_data = pd.concat([_tmpdata, spliced_data], axis=0, ignore_index=True)
            spliced_data = copy.deepcopy(predict_data)
            predict_data['Date'] = pd.to_datetime(predict_data['Date'])

            if args.api == "akshare" or args.api == "yfinance":
                ## use akshare data or yfinance data
                predict_data[['Open', 'High', 'Low', 'Close', 'change', 'pct_change', 'Volume', 'amount', 'amplitude', 'exchange_rate']] = predict_data[['Open', 'High', 'Low', 'Close', 'change', 'pct_change', 'Volume', 'amount', 'amplitude', 'exchange_rate']].astype('float64')
                predict_data['Date'] = predict_data['Date'].dt.strftime('%Y%m%d')
                predict_data.rename(
                    columns={
                        'Date': 'trade_date', 'Open': 'open',
                        'High': 'high', 'Low': 'low',
                        'Close': 'close', 'Volume': 'vol'},
                    inplace=True)
                predict_data = predict_data.loc[:,["ts_code",
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
                    "exchange_rate"]]
            elif args.api == "tushare":
                ## Use tushare data
                predict_data['Date'] = predict_data['Date'].dt.strftime('%Y%m%d')
                predict_data = predict_data.loc[:,["ts_code","Date","Open","Close","High","Low","Volume","amount","amplitude","pct_change","change","exchange_rate"]]
                predict_data.rename(
                    columns={
                        'Date': 'trade_date', 'Open': 'open',
                        'High': 'high', 'Low': 'low',
                        'Close': 'close', 'Volume': 'vol'},
                    inplace=True)

            predict_data.to_csv(test_path,sep=',',index=False,header=True)
            load_data([test_codes[0]],None,test_path,data_queue=data_queue)
            while data_queue.empty() == False:
                try: 
                    predict_data = data_queue.get(timeout=30)
                    break
                except queue.Empty:
                    break

            predict_days -= 1
            pbar.update(1)
        pbar.close()

        datalist = predict_data.iloc[:, 2:2+OUTPUT_DIMENSION].values.tolist()[::-1]
        real_list = datalist[len(datalist)-abs(int(args.predict_days))-show_days:len(datalist)-abs(int(args.predict_days))]
        prediction_list = datalist[len(datalist)-abs(int(args.predict_days))-1:]
    else:
        predict_data.drop(['ts_code', 'Date'], axis=1, inplace=True)
        # predict_data = predict_data.dropna()
        predict_data = predict_data.fillna(-0.0)
        accuracy_list, predict_list = [], []
        test_loss, predict_list, dataloader = test(predict_data,dataloader_mode=2)
        
        for items in predict_list:
            items=items.to("cpu", non_blocking=True)
            for idxs in items:
                for idx in idxs:
                    _tmp = []
                    for index, item in enumerate(idx):
                        if show_list[index] == 1:
                            _tmp.append(item*std_list[index]+mean_list[index])
                    prediction_list.append(np.array(_tmp))
        
        _data_real =  predict_data.head(show_days).sort_values(by=['Date'], ascending=True).values.tolist()
        for idx in range(len(_data_real)):
            _tmp = []
            for index in range(len(show_list)):
                if show_list[index] == 1:
                    # _tmp.append(_data_real[idx][index]*std_list[index]+mean_list[index])
                    _tmp.append(_data_real[idx][index])
            real_list.append(np.array(_tmp))
        prediction_list = [real_list[-1]] + prediction_list
        # for i,(_,label) in enumerate(dataloader):
        #     for idx in range(label.shape[0]):
        #         _tmp = []
        #         for index in range(OUTPUT_DIMENSION):
        #             if use_list[index] == 1:
        #                 _tmp.append(label[idx][0][index]*std_list[index]+mean_list[index])
        #         real_list.append(np.array(_tmp))
        # real_list = real_list[len(real_list) - show_days:]
    # compounding_factor = cal_compounding_factor(test_codes[0])
    # real_list = np.array(real_list) * compounding_factor
    # prediction_list = np.array(prediction_list) * compounding_factor
    pbar = tqdm(total=sum(show_list), leave=False, ncols=TQDM_NCOLS)
    for i in range(sum(show_list)):
        _real_list = np.transpose(real_list)[i]
        _prediction_list = np.transpose(prediction_list)[i]
        assert len(_real_list) >= show_days, "The length of real_list is less than show_days"
        plt.figure()
        x1 = np.linspace(len(_real_list) - show_days, len(_real_list), show_days)
        x2 = np.linspace(len(_real_list), len(_real_list) + len(_prediction_list), len(_prediction_list))
        # x1 = generate_dates(current_date.strftime("%Y%m%d"), -1 * (show_days - 1))
        # x2 = np.concatenate((np.array([""]),generate_dates((current_date + timedelta(days=1)).strftime("%Y%m%d"), len(_prediction_list) - 2)),axis=0)
        plt.plot(x1, np.array(_real_list), label=current_date.strftime("%Y%m%d")+"_real_"+name_list[i])
        plt.plot(x2, np.array(_prediction_list), label=current_date.strftime("%Y%m%d")+"_prediction_"+name_list[i], linewidth=0.75, linestyle='--')
        for item in range(len(_real_list)):
            plt.text(item, _real_list[item], '%.2f' % _real_list[item], ha='center', va='bottom', fontsize=10)
        for item in range(1, len(_prediction_list)):
            plt.text(item + len(_real_list), _prediction_list[item], '%.2f' % _prediction_list[item], ha='center', va='bottom', fontsize=10)
        plt.legend()
        now = datetime.now()
        date_string = now.strftime("%Y%m%d%H%M%S")
        plt.savefig(png_path + "/predict/" + cnname + "_" + str(test_code[0]).split('.')[0] + "_" + model_mode + "_" + name_list[i] + "_" + date_string + "_Pre.png", dpi=600)
        pbar.update(1)
    pbar.close()

def loss_curve(loss_list):
    try:
        plt.figure()
        x=np.linspace(1,len(loss_list),len(loss_list))
        x=20*x
        plt.plot(x,np.array(loss_list),label="train_loss")
        plt.ylabel("MSELoss")
        plt.xlabel("iteration")
        now = datetime.now()
        date_string = now.strftime("%Y%m%d%H%M%S")
        plt.savefig(png_path + "/train_loss/"+cnname+"_"+model_mode+"_"+date_string+"_train_loss.png",dpi=600)
        plt.close()
    except Exception as e:
        print("Error: loss_curve", e)

def contrast_lines(test_codes):
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
            data_queue = dill.load(f)
        while data_queue.empty() == False:
            try:
                item = data_queue.get(timeout=30)
            except queue.Empty:
                break
            if str(item['ts_code'][0]).zfill(6) in test_codes:
                data = copy.deepcopy(item)
                break
        if data is NoneDataFrame:
            print("Error: data is None")
            return
        data_queue = queue.Queue()
        data.drop(['ts_code','Date'],axis=1,inplace = True)  
    
    # data = data.dropna()
    data = data.fillna(-0.0)
    print("test_code=", test_codes)
    if data.empty or (PKL is False and data["ts_code"][0] == "None"):
        print("Error: data is empty or ts_code is None")
        return -1

    if PKL is False:
        data.drop(['ts_code', 'Date'], axis=1, inplace=True)
    train_size = int(TRAIN_WEIGHT * (data.shape[0]))
    if train_size < SEQ_LEN or train_size + SEQ_LEN > data.shape[0]:
        print("Error: train_size is too small or too large")
        return -1

    Train_data = copy.deepcopy(data)
    Test_data = copy.deepcopy(data)
    if Train_data.empty or Test_data.empty or Train_data is None or Test_data is None:
        print("Error: Train_data or Test_data is None")
        return -1

    accuracy_list, predict_list = [], []
    test_loss, predict_list, dataloader = test(Test_data, dataloader_mode=3)
    if test_loss == -1 and predict_list == -1:
        print("Error: No model excist")
        exit(0)
    print("test_data MSELoss:(pred-real)/real=", test_loss)

    real_list = []
    prediction_list = []
    if int(args.predict_days) <= 0:
        for i,(_,label) in enumerate(dataloader):
            for idx in range(label.shape[0]):
                _tmp = []
                for index in range(len(show_list)):
                    if show_list[index] == 1:
                        _tmp.append(label[idx][index]*test_std_list[index]+test_mean_list[index])
                real_list.append(np.array(_tmp))

        for items in predict_list:
            items=items.to("cpu", non_blocking=True)
            for idxs in items:
                _tmp = []
                for index, item in enumerate(idxs):
                    if show_list[index] == 1:
                        _tmp.append(item*test_std_list[index]+test_mean_list[index])
                prediction_list.append(np.array(_tmp))
    else:
        for i,(_,label) in enumerate(dataloader):
            for idx in range(label.shape[0]):
                _tmp = []
                for index in range(len(show_list)):
                    if show_list[index] == 1:
                        _tmp.append(label[idx][0][index]*test_std_list[index]+test_mean_list[index])
                real_list.append(np.array(_tmp))

        for items in predict_list:
            items=items.to("cpu", non_blocking=True)
            for idxs in items:
                _tmp = []
                for index, item in enumerate(idxs[0]):
                    if show_list[index] == 1:
                        _tmp.append(item*test_std_list[index]+test_mean_list[index])
                prediction_list.append(np.array(_tmp))
    pbar = tqdm(total=sum(show_list), ncols=TQDM_NCOLS)
    for i in range(sum(show_list)):
        try:
            pbar.set_description(f"{name_list[i]}")
            _real_list = np.transpose(real_list)[i]
            _prediction_list = np.transpose(prediction_list)[i]
            plt.figure()
            x1 = np.linspace(0, len(_real_list), len(_real_list))
            x2 = np.linspace(0, len(_prediction_list), len(_prediction_list))
            plt.plot(x1, np.array(_real_list), label="real_"+name_list[i])
            plt.plot(x2, np.array(_prediction_list), label="prediction_"+name_list[i], linewidth=0.75, linestyle='--')
            plt.legend()
            now = datetime.now()
            date_string = now.strftime("%Y%m%d%H%M%S")
            plt.savefig(png_path + "/test/" + cnname + "_"  + str(test_code[0]).split('.')[0] + "_" + model_mode + "_" + name_list[i] + "_" + date_string + "_Pre.png", dpi=600)
            pbar.update(1)
        except Exception as e:
            print("Error: contrast_lines", e)
            pbar.update(1)
            continue
    pbar.close()
    plt.close()

if __name__=="__main__":
    global last_loss,test_model,model,total_test_length,lr_scheduler,drop_last
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

    if model_mode=="LSTM":
        model=LSTM(dimension=INPUT_DIMENSION)
        test_model=LSTM(dimension=INPUT_DIMENSION)
        save_path=lstm_path
        criterion=nn.MSELoss()
    elif model_mode=="TRANSFORMER":
        model=TransformerModel(input_dim=INPUT_DIMENSION, d_model=D_MODEL, nhead=NHEAD, num_layers=6, dim_feedforward=2048, output_dim=OUTPUT_DIMENSION, max_len=SEQ_LEN, mode=0)
        test_model=TransformerModel(input_dim=INPUT_DIMENSION, d_model=D_MODEL, nhead=NHEAD, num_layers=6, dim_feedforward=2048, output_dim=OUTPUT_DIMENSION, max_len=SEQ_LEN, mode=1)
        save_path=transformer_path
        criterion=nn.MSELoss()
    else:
        print("No such model")
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

    period = 100
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
    random.shuffle(ts_codes)
    random.shuffle(train_codes)
    random.shuffle(test_codes)

    if mode == 'train':
        lo_list=[]
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
                _data_queue = dill.load(f)
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
                    _data = _data.fillna(-0.0)
                    if _data.empty:
                        continue
                    _ts_code = str(_data['ts_code'][0]).zfill(6)
                    if args.api == "akshare":
                        _ts_code = _ts_code.zfill(6)
                    if _ts_code in train_codes:
                        data_queue.put(_data)
                        total_length += _data.shape[0] - SEQ_LEN
                    if _ts_code in test_codes:
                        test_queue.put(_data)
                        total_test_length += _data.shape[0] - SEQ_LEN
                    if _ts_code not in train_codes and _ts_code not in test_codes:
                        print("Error: %s not in train or test"%_ts_code)
                        continue
                    if _ts_code in train_codes and _ts_code in test_codes:
                        print("Error: %s in train and test"%_ts_code)
                        continue
                init_bar.close()
            codes_len = data_queue.qsize()
        print("total codes: %d, total length: %d"%(codes_len, total_length))
        print("total test codes: %d, total test length: %d"%(test_queue.qsize(), total_test_length))
        batch_none = 0
        data_none = 0
        scaler = GradScaler()
        pbar = tqdm(total=EPOCH, leave=False, ncols=TQDM_NCOLS)
        last_epoch = 0
        for epoch in range(0,EPOCH):
            if len(lo_list) == 0:
                    m_loss = 0
            else:
                m_loss = np.mean(lo_list)
            pbar.set_description("%d, %e"%(epoch+1,m_loss))
            if args.pkl_queue == 0:
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
                        data = data.fillna(-0.0)
                        if data.empty or data["ts_code"][0] == "None":
                            tqdm.write("data is empty or data has invalid col")
                            code_bar.update(1)
                            continue
                        ts_code = str(data['ts_code'][0]).zfill(6)
                        if args.begin_code != "":
                            if ts_code != args.begin_code:
                                code_bar.update(1)
                                continue
                            else:
                                args.begin_code = ""
                        data.drop(['ts_code','Date'],axis=1,inplace = True)    
                        train_size=int(TRAIN_WEIGHT*(data.shape[0]))
                        if train_size<SEQ_LEN or train_size+SEQ_LEN>data.shape[0]:
                            code_bar.update(1)
                            continue
                        Train_data=data[:train_size+SEQ_LEN]
                        # Test_data=data[train_size-SEQ_LEN:]
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
                tqdm.write("pkl_queue is enabled")
                ts_code = "data_queue"
                index = len(ts_codes) - 1
                tqdm.write("epoch: %d, data_queue size before deep copy: %d" % (epoch, data_queue.qsize()))
                _stock_data_queue = deep_copy_queue(data_queue)

                tqdm.write("epoch: %d, data_queue size after deep copy: %d" % (epoch, data_queue.qsize()))
                tqdm.write("epoch: %d, _stock_data_queue size: %d" % (epoch, _stock_data_queue.qsize()))
                
                stock_train = stock_queue_dataset(mode=0, data_queue=_stock_data_queue, label_num=OUTPUT_DIMENSION, buffer_size=BUFFER_SIZE, total_length=total_length,predict_days=int(args.predict_days))
            iteration=0
            loss_list=[]
            
            train_dataloader=DataLoader(dataset=stock_train,batch_size=BATCH_SIZE,shuffle=False,drop_last=drop_last, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=custom_collate)
            predict_list=[]
            accuracy_list=[]
            train(epoch+1, train_dataloader, scaler, ts_code, test_queue)
            if args.pkl_queue == 0:
                code_bar.update(1)
            if (time.time() - last_save_time >= SAVE_INTERVAL or index == len(ts_codes) - 1) and safe_save == True:
                thread_save_model(model, optimizer, save_path, False, int(args.predict_days))
                last_save_time = time.time()
            if args.pkl_queue == 0:
                code_bar.close()
            if len(lo_list) > 0:
                tqdm.write("Start create image for loss")
                loss_curve(lo_list)
            pbar.update(1)
            last_epoch = epoch
        pbar.close()
        print("Training finished!")
        if len(lo_list) > 0:
            print("Start create image for loss")
            loss_curve(lo_list)
        print("Start create image for pred-real")
        test_index = random.randint(0, len(test_codes) - 1)
        test_code = [test_codes[test_index]]
        while contrast_lines(test_code) == -1:
            test_index = random.randint(0, len(test_codes) - 1)
            test_code = [test_codes[test_index]]
        print("train epoch: %d" % (last_epoch))
    elif mode == "test":
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


