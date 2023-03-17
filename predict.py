#!/usr/bin/env python
# coding: utf-8

import glob
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
from getdata import get_stock_list, get_stock_data
from tqdm import tqdm
from cycler import cycler# 用于定制线条颜色

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

def train(epoch, dataloader):
    global loss
    model.train()
    global loss_list
    global iteration
    subbar = tqdm(total=len(dataloader), leave=False)
    for i,(data,label) in enumerate(dataloader):
        iteration=iteration+1
        data,label = data.to(common.device),label.to(common.device)
        optimizer.zero_grad()
        output=model.forward(data)
        loss=criterion(output,label)
        loss.backward()        
        optimizer.step()
        subbar.set_description("iter=%d,lo=%.4f"%(iteration,loss.item()))
        subbar.update(1)
        if i%20==0:
            loss_list.append(loss.item())
            # print("epoch=",epoch,"iteration=",iteration,"loss=",loss.item())
        if iteration%common.SAVE_NUM_ITER==0:
            torch.save(model.state_dict(),save_path+"_Model.pkl")
            torch.save(optimizer.state_dict(),save_path+"_Optimizer.pkl")
    if (epoch+1)%common.SAVE_NUM_EPOCH==0 or (epoch+1)==common.EPOCH:
        torch.save(model.state_dict(),save_path+"_Model.pkl")
        torch.save(optimizer.state_dict(),save_path+"_Optimizer.pkl")
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
           test_loss = 0.00
        else:
            test_loss = np.mean(accuracy_list)

def loss_curve(loss_list):
    x=np.linspace(1,len(loss_list),len(loss_list))
    x=20*x
    plt.plot(x,np.array(loss_list),label="train_loss")
    plt.ylabel("MSELoss")
    plt.xlabel("iteration")
    plt.savefig("./png/train_loss/"+cnname+"_train_loss.png",dpi=3000)
    # plt.show()

def contrast_lines(predict_list):
    real_list=[]
    prediction_list=[]
    dataloader=common.DataLoaderX(dataset=stock_test,batch_size=common.BATCH_SIZE,shuffle=False,drop_last=True, num_workers=4, pin_memory=True)
    for i,(data,label) in enumerate(dataloader):
        for idx in range(common.BATCH_SIZE):
            real_list.append(np.array(label[idx]*common.std_list[0]+common.mean_list[0]))
    for item in predict_list:
        item=item.to("cpu")
        for idx in range(common.BATCH_SIZE):
            prediction_list.append(np.array((item[idx]*common.std_list[0]+common.mean_list[0])))
    x=np.linspace(1,len(real_list),len(real_list))
    plt.plot(x,np.array(real_list),label="real")
    # plt.plot(x,np.array(prediction_list),label="prediction")
    plt.legend()
    plt.savefig("./png/predict/"+cnname+"_Pre.png",dpi=3000)
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

if __name__=="__main__":
    global test_loss
    test_loss = 0.00
    symbol = 'Generic.Data'
    # symbol = '000001.SZ'
    cnname = ""
    for item in symbol.split("."):
        cnname += item
    if os.path.exists("./" + cnname) is False:
        os.makedirs("./" + cnname)
    lstm_path="./"+cnname+"/LSTM"
    transformer_path="./"+cnname+"/TRANSFORMER"
    save_path=lstm_path

    #选择模型为LSTM或Transformer，注释掉一个
    model_mode="LSTM"
    if model_mode=="LSTM":
        model=common.LSTM(dimension=8)
        test_model=common.LSTM(dimension=8)
        save_path=lstm_path
    elif model_mode=="TRANSFORMER":
        model=common.TransAm(feature_size=8)
        test_model=common.TransAm(feature_size=8)
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
    data_thread = threading.Thread(target=load_data, args=(ts_codes,))
    data_thread.start()
    code_bar = tqdm(total=len(ts_codes))
    for index, ts_code in enumerate(ts_codes):
        try:
            # if common.GET_DATA:
            #     dataFrame = get_stock_data(ts_code, False)
            # data = import_csv(ts_code, dataFrame)
            data = common.data_queue.get()
            data_len = common.data_queue.qsize()
            if data.empty or data["ts_code"][0] == "None":
                code_bar.update(1)
                continue
            if data['ts_code'][0] != ts_code:
                print("Error: ts_code is not match")
                exit(0)
            code_bar.set_description("%s %d:%d" % (ts_code,index,data_len))
            df_draw=data[-period:]
            # draw_Kline(df_draw,period,symbol)
            data.drop(['ts_code','Date'],axis=1,inplace = True)    
            train_size=int(common.TRAIN_WEIGHT*(data.shape[0]))
            # print("Split the data for trainning and testing...")
            if train_size<common.SEQ_LEN or train_size+common.SEQ_LEN>data.shape[0]:
                code_bar.update(1)
                continue
            Train_data=data[:train_size+common.SEQ_LEN]
            Test_data=data[train_size-common.SEQ_LEN:]
            if Train_data is None or Test_data is None:
                code_bar.update(1)
                continue
            # Train_data.to_csv(common.train_path,sep=',',index=False,header=False)
            # Test_data.to_csv(common.test_path,sep=',',index=False,header=False)
            stock_train=common.Stock_Data(train=True, dataFrame=Train_data)
            stock_test=common.Stock_Data(train=False, dataFrame=Test_data)
            iteration=0
            loss_list=[]
        except Exception as e:
            print(e)
            code_bar.update(1)
            continue
        #开始训练神经网络
        # print("Start training the model...")
        train_dataloader=common.DataLoaderX(dataset=stock_train,batch_size=common.BATCH_SIZE,shuffle=False,drop_last=True, num_workers=4, pin_memory=True)
        test_dataloader=common.DataLoaderX(dataset=stock_test,batch_size=4,shuffle=False,drop_last=True, num_workers=4, pin_memory=True)
        pbar = tqdm(total=common.EPOCH, leave=False)
        for epoch in range(0,common.EPOCH):
            predict_list=[]
            accuracy_list=[]
            train(epoch+1, train_dataloader)
            # if (epoch+1)%common.TEST_NUM==0:
            #     # test(test_dataloader)
            #     test_thread = threading.Thread(target=test, args=(test_dataloader,))
            #     test_thread.start()
            # pbar.set_description("ep=%d,lo=%.4f,tl=%.4f"%(epoch+1,loss.item(),test_loss))
            pbar.set_description("ep=%d,lo=%.4f"%(epoch+1,loss.item()))
            pbar.update(1)
        pbar.close()
        code_bar.update(1)
    code_bar.close()
    print("Training finished!")

    # print("Create the png for loss")
    # #绘制损失函数下降曲线    
    # loss_curve(loss_list)
    # print("Create the png for pred-real")
    # #绘制测试集pred-real对比曲线
    # contrast_lines(predict_list)





