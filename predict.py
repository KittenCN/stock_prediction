#!/usr/bin/env python
# coding: utf-8

import argparse
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
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default="train", type=str, help="select running mode")
parser.add_argument('--model', default="lstm", type=str, help="lstm or transformer")
parser.add_argument('--batch_size', default=32, type=int, help="Batch_size")
# parser.add_argument('--begin_code', default="", type=str, help="begin code")
parser.add_argument('--epochs', default=2, type=int, help="epochs")
parser.add_argument('--seq_len', default=179, type=int, help="SEQ_LEN")
parser.add_argument('--lr', default=0.001, type=float, help="LEARNING_RATE")
parser.add_argument('--wd', default=0.0001, type=float, help="WEIGHT_DECAY")
parser.add_argument('--workers', default=4, type=int, help="num_workers")
parser.add_argument('--pkl', default=0, type=int, help="use pkl file instead of csv file")
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
                outputs = model.forward(data)
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
            torch.save(model.state_dict(), save_path + "_Model.pkl")
            torch.save(optimizer.state_dict(), save_path + "_Optimizer.pkl")
            last_save_time = time.time()

    if (epoch % common.SAVE_NUM_EPOCH == 0 or epoch == common.EPOCH) and time.time() - last_save_time >= common.SAVE_INTERVAL:
        torch.save(model.state_dict(), save_path + "_Model.pkl")
        torch.save(optimizer.state_dict(), save_path + "_Optimizer.pkl")
        last_save_time = time.time()

    subbar.close()


def test(dataloader):
    global accuracy_list, predict_list, test_loss

    if len(stock_test) < 4:
        return 0.00
    
    test_optimizer=optim.Adam(test_model.parameters(),lr=common.LEARNING_RATE, weight_decay=common.WEIGHT_DECAY)
    if os.path.exists(save_path+"_Model.pkl") and os.path.exists(save_path+"_Optimizer.pkl"):
        test_model.load_state_dict(torch.load(save_path+"_Model.pkl"))
        test_optimizer.load_state_dict(torch.load(save_path+"_Optimizer.pkl"))

    test_model.eval()
    accuracy_fn = nn.MSELoss()

    with torch.no_grad():
        for data, label in dataloader:
            test_optimizer.zero_grad()
            predict = test_model.forward(data)
            predict_list.append(predict)
            if(predict.shape == label.shape):
                accuracy = accuracy_fn(predict, label)
                accuracy_list.append(accuracy.item())
            else:
                tqdm.write(f"test error: predict.shape != label.shape")
                continue

    if not accuracy_list:
        accuracy_list = [0]

    test_loss = np.mean(accuracy_list)


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
        plt.savefig("./png/train_loss/"+cnname+"_"+date_string+"_train_loss.png",dpi=3000)
        # plt.show()
        plt.close()
    except Exception as e:
        print("Error: loss_curve", e)

def contrast_lines(test_code):
    global stock_test, test_loss, accuracy_list, predict_list, loss_list, lo_list

    print("test_code=", test_code)
    common.load_data(test_code)
    data = common.data_queue.get()

    if data.empty or data["ts_code"][0] == "None":
        print("Error: data is empty or ts_code is None")
        return

    if data['ts_code'][0] != test_code[0]:
        print("Error: ts_code is not match")
        return

    data.drop(['ts_code', 'Date'], axis=1, inplace=True)
    train_size = int(common.TRAIN_WEIGHT * (data.shape[0]))
    if train_size < common.SEQ_LEN or train_size + common.SEQ_LEN > data.shape[0]:
        print("Error: train_size is too small or too large")
        return -1

    Train_data = data[:train_size + common.SEQ_LEN]
    Test_data = data[train_size - common.SEQ_LEN:]
    if Train_data is None or Test_data is None:
        print("Error: Train_data or Test_data is None")
        return

    stock_train = common.Stock_Data(train=True, dataFrame=Train_data, label_num=common.OUTPUT_DIMENSION)
    stock_test = common.Stock_Data(train=False, dataFrame=Test_data, label_num=common.OUTPUT_DIMENSION)

    dataloader = common.DataLoaderX(dataset=stock_test, batch_size=common.BATCH_SIZE, shuffle=False, drop_last=True, num_workers=common.NUM_WORKERS, pin_memory=True)
    accuracy_list, predict_list = [], []
    test(dataloader)
    print("test_data MSELoss:(pred-real)/real=", test_loss)

    real_list = []
    prediction_list = []
    for i,(data,label) in enumerate(dataloader):
        for idx in range(common.BATCH_SIZE):
            _tmp = []
            for index in range(common.OUTPUT_DIMENSION):
                if common.use_list[index] == 1:
                    # real_list.append(np.array(label[idx]*common.std_list[0]+common.mean_list[0]))
                    _tmp.append(label[idx][index]*common.std_list[index]+common.mean_list[index])
            real_list.append(np.array(_tmp))
    for item in predict_list:
        item=item.to("cpu", non_blocking=True)
        for idx in range(common.BATCH_SIZE):
            _tmp = []
            for index in range(common.OUTPUT_DIMENSION):
                if common.use_list[index] == 1:
                    # prediction_list.append(np.array((item[idx]*common.std_list[0]+common.mean_list[0])))
                    _tmp.append(item[idx][index]*common.std_list[index]+common.mean_list[index])
            prediction_list.append(np.array(_tmp))
    pbar = tqdm(total=common.OUTPUT_DIMENSION, ncols=common.TQDM_NCOLS)
    for i in range(common.OUTPUT_DIMENSION):
        try:
            pbar.set_description(f"{common.name_list[i]}")
            _real_list = np.transpose(real_list)[i]
            _prediction_list = np.transpose(prediction_list)[i]
            plt.figure()
            x = np.linspace(1, len(_real_list), len(_real_list))
            plt.plot(x, np.array(_real_list), label="real_"+common.name_list[i])
            plt.plot(x, np.array(_prediction_list), label="prediction_"+common.name_list[i])
            plt.legend()
            now = datetime.now()
            date_string = now.strftime("%Y%m%d%H%M%S")
            plt.savefig("./png/predict/" + cnname + "_" + common.name_list[i] + "_" + date_string + "_Pre.png", dpi=3000)
            pbar.update(1)
        except Exception as e:
            print("Error: contrast_lines", e)
            pbar.update(1)
            continue
    pbar.close()
    plt.close()

if __name__=="__main__":
    global test_loss
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
    test_loss = 0.00
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
    elif model_mode=="TRANSFORMER":
        model=common.TransAm(feature_size=common.INPUT_DIMENSION)
        test_model=common.TransAm(feature_size=common.INPUT_DIMENSION)
        save_path=transformer_path
    else:
        print("No such model")
        exit(0)

    model=model.to(common.device, non_blocking=True)
    test_model.to('cpu', non_blocking=True)
    print(model)
    criterion=nn.MSELoss()
    optimizer=optim.Adam(model.parameters(),lr=common.LEARNING_RATE, weight_decay=common.WEIGHT_DECAY)
    if os.path.exists(save_path+"_Model.pkl") and os.path.exists(save_path+"_Optimizer.pkl"):
        print("Load model and optimizer from file")
        model.load_state_dict(torch.load(save_path+"_Model.pkl"))
        optimizer.load_state_dict(torch.load(save_path+"_Optimizer.pkl"))
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
    test_index = random.randint(0, len(ts_codes) - 1)
    test_code = [ts_codes[test_index]]
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
            for index, ts_code in range(codes_len):
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
                    data = data_list[index].copy(deep=True)
                    data = data.dropna()
                    if common.PKL is False:
                        # data_len = len(data_list)
                        if data is None or data["ts_code"][0] == "None":
                            tqdm.write("data is empty or data has invalid col")
                            code_bar.update(1)
                            continue
                        if data['ts_code'][0] != ts_code:
                            tqdm.write("Error: ts_code is not match")
                            exit(0)
                        # df_draw=data[-period:]
                        # draw_Kline(df_draw,period,symbol)
                        data.drop(['ts_code','Date'],axis=1,inplace = True)    
                        train_size=int(common.TRAIN_WEIGHT*(data.shape[0]))
                        # print("Split the data for trainning and testing...")
                        if train_size<common.SEQ_LEN or train_size+common.SEQ_LEN>data.shape[0]:
                            # tqdm.write(ts_code + ":train_size is too small or too large")
                            code_bar.update(1)
                            continue
                        Train_data=data[:train_size+common.SEQ_LEN]
                        Test_data=data[train_size-common.SEQ_LEN:]
                        if Train_data is None or Test_data is None:
                            tqdm.write(ts_code + ":Train_data or Test_data is None")
                            code_bar.update(1)
                            continue
                        # Train_data.to_csv(common.train_path,sep=',',index=False,header=False)
                        # Test_data.to_csv(common.test_path,sep=',',index=False,header=False)
                    else:
                        Train_data = data
                        ts_code = "pkl_queue"
                    if len(loss_list) == 0:
                        m_loss = 0
                    else:
                        m_loss = np.mean(loss_list)
                    code_bar.set_description("%s, %d, %e" % (ts_code,data_len,m_loss))
                    stock_train=common.Stock_Data(train=True, dataFrame=Train_data, label_num=common.OUTPUT_DIMENSION)
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
                    torch.save(model.state_dict(),save_path+"_Model.pkl")
                    torch.save(optimizer.state_dict(),save_path+"_Optimizer.pkl")
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
        while contrast_lines(test_code) == -1:
            test_index = random.randint(0, len(ts_codes) - 1)
            test_code = [ts_codes[test_index]]



