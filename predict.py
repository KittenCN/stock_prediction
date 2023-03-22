#!/usr/bin/env python
# coding: utf-8

import argparse
import glob
import random
import threading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import matplotlib as mpl# 用于设置曲线参数
import common
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
from tqdm import tqdm
from cycler import cycler# 用于定制线条颜色
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default="train", type=str, help="select running mode")
parser.add_argument('--model', default="LSTM", type=str, help="LSTM or TRANSFORMER")
parser.add_argument('--batch_size', default=32, type=int, help="Batch_size")
parser.add_argument('--begin_code', default="", type=str, help="begin code")
parser.add_argument('--epochs', default=10, type=int, help="epochs")
parser.add_argument('--SEQ_LEN', default=179, type=int, help="SEQ_LEN")
parser.add_argument('--lr', default=0.001, type=float, help="LEARNING_RATE")
parser.add_argument('--wd', default=0.0001, type=float, help="WEIGHT_DECAY")
args = parser.parse_args()
last_save_time = 0

#数据清洗：丢弃行，或用上一行的值填充
def data_wash(dataset,keepTime=False):
    if keepTime:
        dataset.fillna(axis=1,method='ffill')
    else:
        dataset.dropna()
    return dataset

def import_csv(stock_code, dataFrame=None):
    #time设为index的同时是否保留时间列
    if os.path.exists('stock_daily/'+stock_code + '.csv') and dataFrame is None:
        df = pd.read_csv('stock_daily/'+stock_code + '.csv')
    elif os.path.exists('stock_daily/'+stock_code + '.csv') == False and dataFrame is None:
        # print('stock_daily/'+stock_code + '.csv'+' not exist')
        common.csv_queue.put(common.NoneDataFrame)
        return None
    elif dataFrame is not None:
        df = dataFrame
    #清洗数据
    df=data_wash(df,keepTime=False)
    df.rename(
            columns={
            'trade_date': 'Date', 'open': 'Open', 
            'high': 'High', 'low': 'Low', 
            'close': 'Close', 'vol': 'Volume'}, 
            inplace=True)
    df['Date'] = pd.to_datetime(df['Date'],format='%Y%m%d')    
    df.set_index(df['Date'], inplace=True)
    if df.empty:
        common.csv_queue.put(common.NoneDataFrame)
        return None
    common.csv_queue.put(df)
    return df

def draw_Kline(df,period,symbol):

    # 设置基本参数
    # type:#绘制图形的类型，有candle, renko, ohlc, line等
    # 此处选择candle,即K线图
    # mav(moving average):均线类型,此处设置7,30,60日线
    # volume:布尔类型，设置是否显示成交量，默认False
    # title:设置标题
    # y_label:设置纵轴主标题
    # y_label_lower:设置成交量图一栏的标题
    # figratio:设置图形纵横比
    # figscale:设置图形尺寸(数值越大图像质量越高)
    kwargs = dict(
        type='candle', 
        mav=(7, 30, 60), 
        volume=True, 
        title='\nA_stock %s candle_line' % (symbol),    
        ylabel='OHLC Candles', 
        ylabel_lower='Shares\nTraded Volume', 
        figratio=(15, 10), 
        figscale=2)

    # 设置marketcolors
    # up:设置K线线柱颜色，up意为收盘价大于等于开盘价
    # down:与up相反，这样设置与国内K线颜色标准相符
    # edge:K线线柱边缘颜色(i代表继承自up和down的颜色)，下同。详见官方文档)
    # wick:灯芯(上下影线)颜色
    # volume:成交量直方图的颜色
    # inherit:是否继承，选填
    mc = mpf.make_marketcolors(
        up='red', 
        down='green', 
        edge='i', 
        wick='i', 
        volume='in', 
        inherit=True)

    # 设置图形风格
    # gridaxis:设置网格线位置
    # gridstyle:设置网格线线型
    # y_on_right:设置y轴位置是否在右
    s = mpf.make_mpf_style(
        gridaxis='both', 
        gridstyle='-.', 
        y_on_right=False, 
        marketcolors=mc)

    # 设置均线颜色，配色表可见下图
    # 建议设置较深的颜色且与红色、绿色形成对比
    # 此处设置七条均线的颜色，也可应用默认设置
    mpl.rcParams['axes.prop_cycle'] = cycler(
        color=['dodgerblue', 'deeppink', 
        'navy', 'teal', 'maroon', 'darkorange', 
        'indigo'])
    
    # 设置线宽
    mpl.rcParams['lines.linewidth'] = .5

    # 图形绘制
    # show_nontrading:是否显示非交易日，默认False
    # savefig:导出图片，填写文件名及后缀
    mpf.plot(df, 
        **kwargs, 
        style=s, 
        show_nontrading=False,)
    mpf.plot(df, 
        **kwargs, 
        style=s, 
        show_nontrading=False,
        savefig='A_stock-%s %s_candle_line'
        %(symbol, period) + '.jpg')
    plt.show()

def train(epoch, dataloader, scaler):
    global loss, last_save_time, loss_list, iteration, lo_list
    model.train()
    subbar = tqdm(total=len(dataloader), leave=False, ncols=common.TQDM_NCOLS)
    for i,(data,label) in enumerate(dataloader):
        iteration=iteration+1
        data,label = data.to(common.device),label.to(common.device)
        # 开启autocast
        with autocast():
            # 前向传播
            outputs = model.forward(data)
            loss = criterion(outputs, label)
        optimizer.zero_grad()
        # output=model.forward(data)
        # loss=criterion(output,label)
        # loss.backward()        
        # optimizer.step()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        subbar.set_description("iter=%d,lo=%.8f"%(iteration,loss.item()))
        subbar.update(1)
        loss_list.append(loss.item())      
        lo_list.append(loss.item())  
        if iteration%common.SAVE_NUM_ITER==0 and time.time() - last_save_time >= common.SAVE_INTERVAL:
            torch.save(model.state_dict(),save_path+"_Model.pkl")
            torch.save(optimizer.state_dict(),save_path+"_Optimizer.pkl")
            last_save_time = time.time()
    if (epoch%common.SAVE_NUM_EPOCH==0  or epoch==common.EPOCH) and time.time() - last_save_time >= common.SAVE_INTERVAL:
        torch.save(model.state_dict(),save_path+"_Model.pkl")
        torch.save(optimizer.state_dict(),save_path+"_Optimizer.pkl")
        last_save_time = time.time()
    subbar.close()

def test(dataloader):
    global accuracy_list, predict_list, test_loss, loss

    lock = threading.Lock()
    with lock:
        test_criterion=nn.MSELoss()
        test_optimizer=optim.Adam(test_model.parameters(),lr=common.LEARNING_RATE, weight_decay=common.WEIGHT_DECAY)
        if os.path.exists(save_path+"_Model.pkl") and os.path.exists(save_path+"_Optimizer.pkl"):
            # print("Load model and optimizer from file")
            test_model.load_state_dict(torch.load(save_path+"_Model.pkl"))
            test_optimizer.load_state_dict(torch.load(save_path+"_Optimizer.pkl"))
        test_model.eval()
        test_optimizer.zero_grad()
        if len(stock_test) < 4:
            return 0.00
        for i,(data,label) in enumerate(dataloader):
            with torch.no_grad():            
                # data,label=data.to(common.device),label.to(common.device)
                test_optimizer.zero_grad()
                predict=test_model.forward(data)
                predict_list.append(predict)
                loss=test_criterion(predict,label)
                accuracy_fn=nn.MSELoss()
                accuracy=accuracy_fn(predict,label)
                accuracy_list.append(accuracy.item())
        # print("test_data MSELoss:(pred-real)/real=",np.mean(accuracy_list))
        if len(accuracy_list) == 0:
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
        print("Error: loss_curve",e)

def contrast_lines(test_code):
    global stock_test, test_loss

    real_list=[]
    prediction_list=[]
    predict_list=[]
    accuracy_list=[]

    print("test_code=",test_code)
    load_data(test_code)
    data = common.data_queue.get()
    print("data.shape=",data.shape)
    if data.empty or data["ts_code"][0] == "None":
        print("Error: data is empty or ts_code is None")
        return
    if data['ts_code'][0] != test_code[0]:
        print("Error: ts_code is not match")
        return
    data.drop(['ts_code','Date'],axis=1,inplace = True)    
    train_size=int(common.TRAIN_WEIGHT*(data.shape[0]))
    if train_size<common.SEQ_LEN or train_size+common.SEQ_LEN>data.shape[0]:
        print("Error: train_size is too small or too large")
        return -1
    Train_data=data[:train_size+common.SEQ_LEN]
    Test_data=data[train_size-common.SEQ_LEN:]
    if Train_data is None or Test_data is None:
        print("Error: Train_data or Test_data is None")
        return
    stock_train=common.Stock_Data(train=True, dataFrame=Train_data, label_num=common.OUTPU_DIMENSION)
    stock_test=common.Stock_Data(train=False, dataFrame=Test_data, label_num=common.OUTPU_DIMENSION)

    dataloader=common.DataLoaderX(dataset=stock_test,batch_size=common.BATCH_SIZE,shuffle=False,drop_last=True, num_workers=4, pin_memory=True)

    test_criterion=nn.MSELoss()
    test_optimizer=optim.Adam(test_model.parameters(),lr=common.LEARNING_RATE, weight_decay=common.WEIGHT_DECAY)
    if os.path.exists(save_path+"_Model.pkl") and os.path.exists(save_path+"_Optimizer.pkl"):
        print("Load model and optimizer from file")
        test_model.load_state_dict(torch.load(save_path+"_Model.pkl"))
        test_optimizer.load_state_dict(torch.load(save_path+"_Optimizer.pkl"))
    test_model.eval()
    test_optimizer.zero_grad()
    if len(stock_test) < common.BATCH_SIZE:
        print("Error: len(stock_test) < common.BATCH_SIZE")
        return -1
    test_bar = tqdm(total=len(dataloader), ncols=common.TQDM_NCOLS)
    for i,(data,label) in enumerate(dataloader):
        with torch.no_grad():            
            # data,label=data.to(common.device),label.to(common.device)
            test_optimizer.zero_grad()
            predict=test_model.forward(data)
            predict_list.append(predict)
            loss=test_criterion(predict,label)
            accuracy_fn=nn.MSELoss()
            accuracy=accuracy_fn(predict,label)
            accuracy_list.append(accuracy.item())
        test_bar.update(1)
    test_bar.close()
    if len(accuracy_list) == 0:
        accuracy_list = [0]
    test_loss = np.mean(accuracy_list)
    print("test_data MSELoss:(pred-real)/real=",test_loss)

    test_bar = tqdm(total=len(dataloader) * common.BATCH_SIZE * common.OUTPU_DIMENSION, ncols=common.TQDM_NCOLS)
    for i,(data,label) in enumerate(dataloader):
        for idx in range(common.BATCH_SIZE):
            _tmp = []
            for index in range(common.OUTPU_DIMENSION):
                if common.use_list[index] == 1:
                    # real_list.append(np.array(label[idx]*common.std_list[0]+common.mean_list[0]))
                    _tmp.append(label[idx][index]*common.std_list[index]+common.mean_list[index])
                test_bar.update(1)
            real_list.append(np.array(_tmp))
    test_bar.close()
    test_bar = tqdm(total=len(predict_list) * common.BATCH_SIZE * common.OUTPU_DIMENSION, ncols=common.TQDM_NCOLS)
    for item in predict_list:
        item=item.to("cpu")
        for idx in range(common.BATCH_SIZE):
            _tmp = []
            for index in range(common.OUTPU_DIMENSION):
                if common.use_list[index] == 1:
                    # prediction_list.append(np.array((item[idx]*common.std_list[0]+common.mean_list[0])))
                    _tmp.append(item[idx][index]*common.std_list[index]+common.mean_list[index])
                test_bar.update(1)
            prediction_list.append(np.array(_tmp))
    test_bar.close()
    pbar = tqdm(total=common.OUTPU_DIMENSION, ncols=common.TQDM_NCOLS)
    for i in range(common.OUTPU_DIMENSION):
        try:
            _real_list = np.transpose(real_list)[i]
            _prediction_list = np.transpose(prediction_list)[i]
            plt.figure()
            x=np.linspace(1,len(_real_list),len(_real_list))
            plt.plot(x,np.array(_real_list),label="real")
            plt.plot(x,np.array(_prediction_list),label="prediction")
            plt.legend()
            now = datetime.now()
            date_string = now.strftime("%Y%m%d%H%M%S")
            plt.savefig("./png/predict/"+cnname+"_"+common.name_list[i]+"_"+date_string+"_Pre.png",dpi=3000)
            pbar.update(1)
        except:
            pbar.update(1)
            continue
    plt.close()
    pbar.close()
    # plt.show()

def load_data(ts_codes):
    for ts_code in ts_codes:
        if common.data_queue.empty():
            print("data_queue is empty, loading data...")
        if common.GET_DATA:
            # get_stock_data(ts_code, False)
            # dataFrame = common.stock_data_queue.get()
            import_csv(ts_code, None)
            data = common.csv_queue.get()
            common.data_queue.put(data)
            # data_list.append(data)

if __name__=="__main__":
    global test_loss, data_list
    loss_list=[]
    data_list=[]
    mode = args.mode
    model_mode = args.model
    common.BATCH_SIZE = args.batch_size
    common.EPOCH = args.epochs
    common.SEQ_LEN = args.SEQ_LEN
    common.LEARNING_RATE = args.lr
    common.WEIGHT_DECAY = args.wd
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

    model=model.to(common.device)
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
    test_index = random.randint(0, len(ts_codes) - 1)
    test_code = [ts_codes[test_index]]
    if mode == 'train':
        data_thread = threading.Thread(target=load_data, args=(ts_codes,))
        data_thread.start()
        # data_thread.join()
        scaler = GradScaler()
        pbar = tqdm(total=common.EPOCH, leave=False, ncols=common.TQDM_NCOLS)
        lo_list=[]
        data_len=0
        for epoch in range(0,common.EPOCH):
            # if common.data_queue.empty() and data_thread.is_alive() == False:
            #     data_thread = threading.Thread(target=load_data, args=(ts_codes,))  
            #     data_thread.start()
            if len(lo_list) == 0:
                    m_loss = 0
            else:
                m_loss = np.mean(lo_list)
            pbar.set_description("epoch=%d,loss=%.8f"%(epoch+1,m_loss))
            code_bar = tqdm(total=len(ts_codes), ncols=common.TQDM_NCOLS)
            for index, ts_code in enumerate(ts_codes):
                try:
                    # if common.GET_DATA:
                    #     dataFrame = get_stock_data(ts_code, False)
                    # data = import_csv(ts_code, dataFrame)
                    if args.begin_code != "":
                        if ts_code != args.begin_code:
                            code_bar.update(1)
                            continue
                        else:
                            args.begin_code = ""
                    lastFlag = 0
                    # data = common.data_queue.get()
                    if common.data_queue.empty() == False:
                        data_list += [common.data_queue.get()]
                        data_len = max(data_len, common.data_queue.qsize())
                    Err_nums = 5
                    while index >= len(data_list):
                        if common.data_queue.empty() == False:
                            data_list += [common.data_queue.get()]
                        time.sleep(0.01)
                        Err_nums -= 1
                        if Err_nums == 0:
                            tqdm.write("Error: data_list is empty")
                            exit(0)
                    data = data_list[index].copy(deep=True)
                    # data_len = len(data_list)
                    if data is None or data["ts_code"][0] == "None":
                        tqdm.write("data is empty or data has invalid col")
                        code_bar.update(1)
                        continue
                    if data['ts_code'][0] != ts_code:
                        tqdm.write("Error: ts_code is not match")
                        exit(0)
                    if len(loss_list) == 0:
                        m_loss = 0
                    else:
                        m_loss = np.mean(loss_list)
                    code_bar.set_description("%s %d|%d %.8f" % (ts_code,index+1,data_len,m_loss))
                    # df_draw=data[-period:]
                    # draw_Kline(df_draw,period,symbol)
                    data.drop(['ts_code','Date'],axis=1,inplace = True)    
                    train_size=int(common.TRAIN_WEIGHT*(data.shape[0]))
                    # print("Split the data for trainning and testing...")
                    if train_size<common.SEQ_LEN or train_size+common.SEQ_LEN>data.shape[0]:
                        tqdm.write(ts_code + ":train_size is too small or too large")
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
                    stock_train=common.Stock_Data(train=True, dataFrame=Train_data, label_num=common.OUTPU_DIMENSION)
                    iteration=0
                    loss_list=[]
                except Exception as e:
                    print(e)
                    code_bar.update(1)
                    continue
                #开始训练神经网络
                # print("Start training the model...")
                train_dataloader=common.DataLoaderX(dataset=stock_train,batch_size=common.BATCH_SIZE,shuffle=False,drop_last=True, num_workers=4, pin_memory=True)
                predict_list=[]
                accuracy_list=[]
                train(epoch+1, train_dataloader, scaler)
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



