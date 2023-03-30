#!/usr/bin/env python
# coding: utf-8

import argparse
import copy
import glob
import random
import threading
import numpy as np
import matplotlib.pyplot as plt
import common
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import dill
from tqdm import tqdm
from datetime import datetime, timedelta
from torch.cuda.amp import autocast, GradScaler

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default="train", type=str, help="select running mode: train, test, predict")
parser.add_argument('--model', default="transformer", type=str, help="lstm or transformer")
parser.add_argument('--batch_size', default=8, type=int, help="Batch_size")
parser.add_argument('--begin_code', default="", type=str, help="begin code")
parser.add_argument('--epochs', default=1, type=int, help="epochs")
parser.add_argument('--seq_len', default=180, type=int, help="SEQ_LEN")
parser.add_argument('--lr', default=0.001, type=float, help="LEARNING_RATE")
parser.add_argument('--wd', default=0.0001, type=float, help="WEIGHT_DECAY")
parser.add_argument('--workers', default=1, type=int, help="num_workers")
parser.add_argument('--pkl', default=1, type=int, help="use pkl file instead of csv file")
parser.add_argument('--test_code', default="", type=str, help="test code")
parser.add_argument('--test_gpu', default=1, type=int, help="test method use gpu or not")
parser.add_argument('--predict_days', default=15, type=int, help="number of the predict days")
args = parser.parse_args()
last_save_time = 0

if common.device.type == "cuda":
    torch.backends.cudnn.benchmark = True

def train(epoch, dataloader, scaler, ts_code=""):
    global loss, last_save_time, loss_list, iteration, lo_list
    model.train()
    subbar = tqdm(total=len(dataloader), leave=False, ncols=common.TQDM_NCOLS)
    
    for i, (data, label) in enumerate(dataloader):
        try:
            iteration += 1
            data, label = data.to(common.device, non_blocking=True), label.to(common.device, non_blocking=True)
            with autocast():
                outputs = model.forward(data, label)
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
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if common.is_number(str(loss.item())):
                loss_list.append(loss.item())
                lo_list.append(loss.item())

            subbar.set_description(f"{ts_code}, {iteration}, {loss.item():.2e}")
            subbar.update(1)

        except Exception as e:
            print(f"code: {ts_code}, train error: {e}")
            subbar.update(1)
            continue

        if iteration % common.SAVE_NUM_ITER == 0 and time.time() - last_save_time >= common.SAVE_INTERVAL:
            torch.save(model.state_dict(), save_path + "_out" + str(common.OUTPUT_DIMENSION) + "_Model.pkl")
            torch.save(optimizer.state_dict(), save_path + "_out" + str(common.OUTPUT_DIMENSION) + "_Optimizer.pkl")
            last_save_time = time.time()

    if (epoch % common.SAVE_NUM_EPOCH == 0 or epoch == common.EPOCH) and time.time() - last_save_time >= common.SAVE_INTERVAL:
        torch.save(model.state_dict(), save_path + "_out" + str(common.OUTPUT_DIMENSION) +  "_Model.pkl")
        torch.save(optimizer.state_dict(), save_path + "_out" + str(common.OUTPUT_DIMENSION) +  "_Optimizer.pkl")
        last_save_time = time.time()

    subbar.close()


def test(dataloader):
    # global predict_list, test_loss, accuracy_list

    # if len(stock_test) < 4:
    #     return 0.00
    predict_list = []
    accuracy_list = []
    # test_optimizer=optim.Adam(test_model.parameters(),lr=common.LEARNING_RATE, weight_decay=common.WEIGHT_DECAY)
    if os.path.exists(save_path + "_out" + str(common.OUTPUT_DIMENSION) + "_Model.pkl") and os.path.exists(save_path + "_out" + str(common.OUTPUT_DIMENSION) + "_Optimizer.pkl"):
        test_model.load_state_dict(torch.load(save_path + "_out" + str(common.OUTPUT_DIMENSION) + "_Model.pkl"))
        # test_optimizer.load_state_dict(torch.load(save_path + "_out" + str(common.OUTPUT_DIMENSION) + "_Optimizer.pkl"))

    test_model.eval()
    accuracy_fn = nn.MSELoss()
    pbar = tqdm(total=len(dataloader), leave=False, ncols=common.TQDM_NCOLS)
    with torch.no_grad():
        for data, label in dataloader:
            if args.test_gpu == 1:
                data, label = data.to(common.device, non_blocking=True), label.to(common.device, non_blocking=True)
            else:
                data, label = data.to("cpu", non_blocking=True), label.to("cpu", non_blocking=True)
            # test_optimizer.zero_grad()
            predict = test_model.forward(data, label)
            predict_list.append(predict)
            if(predict.shape == label.shape):
                accuracy = accuracy_fn(predict, label)
                accuracy_list.append(accuracy.item())
                pbar.update(1)
            else:
                tqdm.write(f"test error: predict.shape != label.shape")
                pbar.update(1)
                continue
    pbar.close()
    if not accuracy_list:
        accuracy_list = [0]

    test_loss = np.mean(accuracy_list)
    return test_loss, predict_list


def predict(test_codes):
    print("test_code=", test_codes)
    if common.PKL == 0:
        common.load_data(test_codes)
        data = common.data_queue.get()
    else:
        _data = common.NoneDataFrame
        with open(common.train_pkl_path, 'rb') as f:
            common.data_queue = dill.load(f)
        while common.data_queue.empty() == False:
            item = common.data_queue.get()
            if item['ts_code'][0] in test_codes:
                _data = item
                break
        common.data_queue = common.queue.Queue()
        data = copy.deepcopy(_data)

    # data = data.dropna()
    # data.fillna(0, inplace=True)

    if data.empty or data["ts_code"][0] == "None":
        print("Error: data is empty or ts_code is None")
        return

    if data['ts_code'][0] != test_codes[0]:
        print("Error: ts_code is not match")
        return

    predict_size = int(data.shape[0])
    if predict_size < common.SEQ_LEN:
        print("Error: train_size is too small or too large")
        return

    # Train_data = data[:train_size + common.SEQ_LEN]
    # Test_data = data[train_size - common.SEQ_LEN:]
    predict_data = copy.deepcopy(data)
    spliced_data = copy.deepcopy(data)
    if predict_data.empty or predict_data is None:
        print("Error: Train_data or Test_data is None")
        return
    predict_days = int(args.predict_days)
    pbar = tqdm(total=predict_days, leave=False, ncols=common.TQDM_NCOLS)
    while predict_days > 0:
        lastdate = predict_data["Date"][0].strftime("%Y%m%d")
        predict_data.drop(['ts_code', 'Date'], axis=1, inplace=True)
        predict_data = predict_data.dropna()
        stock_predict = common.Stock_Data(mode=2, dataFrame=predict_data, label_num=common.OUTPUT_DIMENSION)
        dataloader = common.DataLoaderX(dataset=stock_predict, batch_size=1, shuffle=False, drop_last=True, num_workers=common.NUM_WORKERS, pin_memory=True)
        accuracy_list, predict_list = [], []
        test_loss, predict_list = test(dataloader)
        # print("test_data MSELoss:(pred-real)/real=", test_loss)
        _tmp = []
        prediction_list = []
        for index in range(common.OUTPUT_DIMENSION):
            if common.use_list[index] == 1:
                _tmp.append(round((predict_list[0][0][index]*common.std_list[index]+common.mean_list[index]).cpu().item(), 2))
        date_str = lastdate
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        new_date_obj = date_obj + timedelta(days=1)
        # date_string = new_date_obj.strftime("%Y%m%d")
        _tmpdata = [test_codes[0], new_date_obj]
        _tmpdata = _tmpdata + copy.deepcopy(_tmp)
        # while len(_tmpdata) < spliced_data.columns.size:
        #     _tmpdata.append(0)
        _splice_data = copy.deepcopy(spliced_data).drop(['ts_code', 'Date'], axis=1)
        df_mean = _splice_data.mean().tolist()
        for index in range(len(_tmpdata) - 2, len(df_mean)):
            _tmpdata.append(df_mean[index])
        _tmpdata = common.pd.DataFrame(_tmpdata).T
        _tmpdata.columns = spliced_data.columns
        predict_data = common.pd.concat([_tmpdata, spliced_data], axis=0, ignore_index=True)
        spliced_data = copy.deepcopy(predict_data)
        predict_data['Date'] = common.pd.to_datetime(predict_data['Date'])
        predict_data['Date'] = predict_data['Date'].dt.strftime('%Y%m%d')
        predict_data.drop(["macd_dif","macd_dea","macd_bar","k","d","j","boll_upper","boll_mid","boll_lower","cci","pdi","mdi","adx","adxr","taq_up","taq_mid","taq_down","trix","trma","atr"], axis=1, inplace=True)
        predict_data.rename(
            columns={
                'Date': 'trade_date', 'Open': 'open',
                'High': 'high', 'Low': 'low',
                'Close': 'close', 'Volume': 'vol'},
            inplace=True)
        predict_data.to_csv(common.test_path,sep=',',index=False,header=True)
        common.load_data([test_codes[0]],None,common.test_path)
        predict_data = common.data_queue.get()
        # predict_data.set_index(predict_data['Date'], inplace=True)
        # predict_data = common.add_target(predict_data)
        # predict_data = predict_data.dropna()
        predict_days -= 1
        pbar.update(1)
    pbar.close()

    # prediction_list.append(np.array(_tmp))
    # print("predict_list=", prediction_list)
    datalist = predict_data.iloc[:, 2:6].values.tolist()[::-1]
    real_list = datalist[:len(datalist)-int(args.predict_days)]
    prediction_list = datalist[len(datalist)-int(args.predict_days)-1:]
    pbar = tqdm(total=common.OUTPUT_DIMENSION, leave=False, ncols=common.TQDM_NCOLS)
    for i in range(common.OUTPUT_DIMENSION):
        _real_list = np.transpose(real_list)[i]
        _prediction_list = np.transpose(prediction_list)[i]
        plt.figure()
        x1 = np.linspace(0, len(_real_list), len(_real_list))
        x2 = np.linspace(len(_real_list), len(_real_list) + len(_prediction_list), len(_prediction_list))
        plt.plot(x1, np.array(_real_list), label="real_"+common.name_list[i])
        plt.plot(x2, np.array(_prediction_list), label="prediction_"+common.name_list[i])
        plt.legend()
        now = datetime.now()
        date_string = now.strftime("%Y%m%d%H%M%S")
        plt.savefig("./png/predict/" + cnname + "_" + str(test_code[0]).split('.')[0] + str(test_code[0]).split('.')[1] + "_" + model_mode + "_" + common.name_list[i] + "_" + date_string + "_Pre.png", dpi=600)
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
        plt.savefig("./png/train_loss/"+cnname+"_"+model_mode+"_"+date_string+"_train_loss.png",dpi=600)
        # plt.show()
        plt.close()
    except Exception as e:
        print("Error: loss_curve", e)

def contrast_lines(test_codes):
    # global test_loss, accuracy_list, predict_list
    # test_loss = 0.00


    if common.PKL is False:
        common.load_data(test_codes)
        data = common.data_queue.get()
    else:
        _data = common.NoneDataFrame
        with open(common.train_pkl_path, 'rb') as f:
            common.data_queue = dill.load(f)
        while common.data_queue.empty() == False:
            item = common.data_queue.get()
            if item['ts_code'][0] in test_codes:
                _data = item
                break
        common.data_queue = common.queue.Queue()
        data = copy.deepcopy(_data)
        data.drop(['ts_code','Date'],axis=1,inplace = True)  
    
    data = data.dropna()
    # data.fillna(0, inplace=True)
    print("test_code=", test_codes)
    if data.empty or (common.PKL is False and data["ts_code"][0] == "None"):
        print("Error: data is empty or ts_code is None")
        return -1

    # if data['ts_code'][0] != test_code[0]:
    #     print("Error: ts_code is not match")
    #     return
    if common.PKL is False:
        data.drop(['ts_code', 'Date'], axis=1, inplace=True)
    train_size = int(common.TRAIN_WEIGHT * (data.shape[0]))
    if train_size < common.SEQ_LEN or train_size + common.SEQ_LEN > data.shape[0]:
        print("Error: train_size is too small or too large")
        return -1

    Train_data = copy.deepcopy(data)
    Test_data = copy.deepcopy(data)
    if Train_data.empty or Test_data.empty or Train_data is None or Test_data is None:
        print("Error: Train_data or Test_data is None")
        return -1

    stock_train = common.Stock_Data(mode=0, dataFrame=Train_data, label_num=common.OUTPUT_DIMENSION)
    stock_test = common.Stock_Data(mode=1, dataFrame=Test_data, label_num=common.OUTPUT_DIMENSION)

    dataloader = common.DataLoaderX(dataset=stock_test, batch_size=common.BATCH_SIZE, shuffle=False, drop_last=True, num_workers=common.NUM_WORKERS, pin_memory=True)
    accuracy_list, predict_list = [], []
    test_loss, predict_list = test(dataloader)
    print("test_data MSELoss:(pred-real)/real=", test_loss)

    real_list = []
    prediction_list = []
    for i,(_,label) in enumerate(dataloader):
        for idx in range(common.BATCH_SIZE):
            _tmp = []
            for index in range(common.OUTPUT_DIMENSION):
                if common.use_list[index] == 1:
                    # real_list.append(np.array(label[idx]*common.std_list[0]+common.mean_list[0]))
                    _tmp.append(label[idx][index]*common.std_list[index]+common.mean_list[index])
            real_list.append(np.array(_tmp))
    # real_list = copy.deepcopy(stock_test.label.numpy())
    # for i in range(len(real_list[0])):
    #     real_list[:, i] = real_list[:, i] * (common.std_list[i] + 1e-8) + common.mean_list[i]

    for items in predict_list:
        items=items.to("cpu", non_blocking=True)
        for idxs in items:
            _tmp = []
            for index, item in enumerate(idxs):
                if common.use_list[index] == 1:
                    # prediction_list.append(np.array((item[idx]*common.std_list[0]+common.mean_list[0])))
                    _tmp.append(item*common.std_list[index]+common.mean_list[index])
            prediction_list.append(np.array(_tmp))
    # real_list = real_list[abs(len(real_list)-len(prediction_list)):]
    pbar = tqdm(total=common.OUTPUT_DIMENSION, ncols=common.TQDM_NCOLS)
    for i in range(common.OUTPUT_DIMENSION):
        try:
            pbar.set_description(f"{common.name_list[i]}")
            _real_list = np.transpose(real_list)[i]
            _prediction_list = np.transpose(prediction_list)[i]
            plt.figure()
            x1 = np.linspace(0, len(_real_list), len(_real_list))
            x2 = np.linspace(0, len(_prediction_list), len(_prediction_list))
            plt.plot(x1, np.array(_real_list), label="real_"+common.name_list[i])
            plt.plot(x2, np.array(_prediction_list), label="prediction_"+common.name_list[i])
            plt.legend()
            now = datetime.now()
            date_string = now.strftime("%Y%m%d%H%M%S")
            plt.savefig("./png/test/" + cnname + "_"  + str(test_code[0]).split('.')[0] + str(test_code[0]).split('.')[1] + "_" + model_mode + "_" + common.name_list[i] + "_" + date_string + "_Pre.png", dpi=600)
            pbar.update(1)
        except Exception as e:
            print("Error: contrast_lines", e)
            pbar.update(1)
            continue
    pbar.close()
    plt.close()

if __name__=="__main__":
    loss_list=[]
    data_list=[]
    mode = args.mode
    model_mode = args.model.upper()
    common.BATCH_SIZE = args.batch_size
    common.EPOCH = args.epochs
    common.SEQ_LEN = args.seq_len
    common.LEARNING_RATE = args.lr
    common.WEIGHT_DECAY = args.wd
    common.NUM_WORKERS = args.workers
    common.PKL = False if args.pkl <= 0 else True
    symbol = 'Generic.Data'
    # symbol = '000001.SZ'
    cnname = ""
    for item in symbol.split("."):
        cnname += item
    common.check_exist("./" + cnname)
    lstm_path="./"+cnname+"/LSTM"
    transformer_path="./"+cnname+"/TRANSFORMER"
    save_path=lstm_path

    if model_mode=="LSTM":
        model=common.LSTM(dimension=common.INPUT_DIMENSION)
        test_model=common.LSTM(dimension=common.INPUT_DIMENSION)
        save_path=lstm_path
        criterion=nn.MSELoss()
    elif model_mode=="TRANSFORMER":
        model=common.TransformerModel(input_dim=common.INPUT_DIMENSION, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, output_dim=common.OUTPUT_DIMENSION, max_len=common.SEQ_LEN)
        test_model=common.TransformerModel(input_dim=common.INPUT_DIMENSION, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, output_dim=common.OUTPUT_DIMENSION, max_len=common.SEQ_LEN)
        save_path=transformer_path
        criterion=nn.MSELoss()
    else:
        print("No such model")
        exit(0)

    model=model.to(common.device, non_blocking=True)
    if args.test_gpu == 0:
        test_model=test_model.to('cpu', non_blocking=True)
    else:
        test_model=test_model.to(common.device, non_blocking=True)
    if torch.cuda.device_count() >= 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            if args.test_gpu == 1:
                test_model = nn.DataParallel(test_model)
    else:
        print("Let's use CPU!")

    print(model)
    optimizer=optim.Adam(model.parameters(),lr=common.LEARNING_RATE, weight_decay=common.WEIGHT_DECAY)
    if os.path.exists(save_path + "_out" + str(common.OUTPUT_DIMENSION) + "_Model.pkl") and os.path.exists(save_path + "_out" + str(common.OUTPUT_DIMENSION) + "_Optimizer.pkl"):
        print("Load model and optimizer from file")
        model.load_state_dict(torch.load(save_path + "_out" + str(common.OUTPUT_DIMENSION) + "_Model.pkl"))
        optimizer.load_state_dict(torch.load(save_path + "_out" + str(common.OUTPUT_DIMENSION) + "_Optimizer.pkl"))
    else:
        print("No model and optimizer file, train from scratch")

    period = 100
    print("Clean the data...")
    if symbol == 'Generic.Data':
        # ts_codes = get_stock_list()
        csv_files = glob.glob("./stock_daily/*.csv")
        ts_codes =[]
        for csv_file in csv_files:
            ts_codes.append(os.path.basename(csv_file).rsplit(".", 1)[0])
    else:
        ts_codes = [symbol]
    random.shuffle(ts_codes)
    if mode == 'train':
        lo_list=[]
        data_len=0
        if common.PKL is False:
            data_thread = threading.Thread(target=common.load_data, args=(ts_codes,))
            data_thread.start()
            codes_len = len(ts_codes)
        else:
            with open(common.train_pkl_path, 'rb') as f:
                common.data_queue = dill.load(f)
            while common.data_queue.empty() == False:
                data_list += [common.data_queue.get()]
                data_len = max(data_len, common.data_queue.qsize())
            random.shuffle(data_list)
            codes_len = len(data_list)
        #data_thread.join()
        scaler = GradScaler()
        pbar = tqdm(total=common.EPOCH, leave=False, ncols=common.TQDM_NCOLS)
        for epoch in range(0,common.EPOCH):
            # if common.data_queue.empty() and data_thread.is_alive() == False:
            #     data_thread = threading.Thread(target=common.load_data, args=(ts_codes,))  
            #     data_thread.start()
            if len(lo_list) == 0:
                    m_loss = 0
            else:
                m_loss = np.mean(lo_list)
            pbar.set_description("%d, %e"%(epoch+1,m_loss))
            code_bar = tqdm(total=codes_len, ncols=common.TQDM_NCOLS)
            for index in range (codes_len):
                try:
                    # if common.GET_DATA:
                    #     dataFrame = get_stock_data(ts_code, False)
                    # data = import_csv(ts_code, dataFrame)
                    # if args.begin_code != "":
                    #     if ts_code != args.begin_code:
                    #         code_bar.update(1)
                    #         continue
                    #     else:
                    #         args.begin_code = ""
                    # lastFlag = 0
                    # data = common.data_queue.get()
                    if common.PKL is False:
                        while common.data_queue.empty() == False:
                            data_list += [common.data_queue.get()]
                            data_len = max(data_len, common.data_queue.qsize())
                        Err_nums = 5
                        while index >= len(data_list):
                            if common.data_queue.empty() == False:
                                data_list += [common.data_queue.get()]
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
                    data = data.dropna()
                    # data.fillna(0, inplace=True)
                    # data_len = len(data_list)
                    if data.empty or data["ts_code"][0] == "None":
                        tqdm.write("data is empty or data has invalid col")
                        code_bar.update(1)
                        continue
                    # if data['ts_code'][0] != ts_code:
                    #     tqdm.write("Error: ts_code is not match")
                    #     exit(0)
                    # df_draw=data[-period:]
                    # draw_Kline(df_draw,period,symbol)
                    ts_code = data['ts_code'][0]
                    if args.begin_code != "":
                        if ts_code != args.begin_code:
                            code_bar.update(1)
                            continue
                        else:
                            args.begin_code = ""
                    data.drop(['ts_code','Date'],axis=1,inplace = True)    
                    train_size=int(common.TRAIN_WEIGHT*(data.shape[0]))
                    # print("Split the data for trainning and testing...")
                    if train_size<common.SEQ_LEN or train_size+common.SEQ_LEN>data.shape[0]:
                        # tqdm.write(ts_code + ":train_size is too small or too large")
                        code_bar.update(1)
                        continue
                    Train_data=data[:train_size+common.SEQ_LEN]
                    # Test_data=data[train_size-common.SEQ_LEN:]
                    if Train_data.empty or Train_data is None:
                        tqdm.write(ts_code + ":Train_data is None")
                        code_bar.update(1)
                        continue
                    # Train_data.to_csv(common.train_path,sep=',',index=False,header=False)
                    # Test_data.to_csv(common.test_path,sep=',',index=False,header=False)
                    if len(loss_list) == 0:
                        m_loss = 0
                    else:
                        m_loss = np.mean(loss_list)
                    code_bar.set_description("%s, %d, %e" % (ts_code,data_len,m_loss))
                    stock_train=common.Stock_Data(mode=0, dataFrame=Train_data, label_num=common.OUTPUT_DIMENSION)
                    iteration=0
                    loss_list=[]
                except Exception as e:
                    print(ts_code,"main function ", e)
                    code_bar.update(1)
                    continue
                #开始训练神经网络
                # print("Start training the model...")
                train_dataloader=common.DataLoaderX(dataset=stock_train,batch_size=common.BATCH_SIZE,shuffle=False,drop_last=False, num_workers=common.NUM_WORKERS, pin_memory=True)
                predict_list=[]
                accuracy_list=[]
                train(epoch+1, train_dataloader, scaler, ts_code)
                code_bar.update(1)
                if time.time() - last_save_time >= common.SAVE_INTERVAL or index == len(ts_codes) - 1:
                    torch.save(model.state_dict(),save_path + "_out" + str(common.OUTPUT_DIMENSION) +  "_Model.pkl")
                    torch.save(optimizer.state_dict(),save_path + "_out" + str(common.OUTPUT_DIMENSION) +  "_Optimizer.pkl")
                    last_save_time = time.time()
            code_bar.close()
            pbar.update(1)
        pbar.close()
        print("Training finished!")
        if len(lo_list) > 0:
            print("Start create image for loss")
            loss_curve(lo_list)
        print("Start create image for pred-real")
        test_index = random.randint(0, len(ts_codes) - 1)
        test_code = [ts_codes[test_index]]
        while contrast_lines(test_code) == -1:
            test_index = random.randint(0, len(ts_codes) - 1)
            test_code = [ts_codes[test_index]]
    elif mode == "test":
        if args.test_code != "" or args.test_code == "all":
            test_code = [args.test_code]
        else:
            test_index = random.randint(0, len(ts_codes) - 1)
            test_code = [ts_codes[test_index]]
        while contrast_lines(test_code) == -1:
            test_index = random.randint(0, len(ts_codes) - 1)
            test_code = [ts_codes[test_index]]
    elif mode == "predict":
        if args.test_code == "":
            print("Error: test_code is empty")
            exit(0)
        elif args.test_code in ts_codes or common.PKL == True:
            test_code = [args.test_code]
            predict(test_code)
        else:
            print("Error: test_code is not in ts_codes")
            exit(0)



