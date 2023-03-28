# 基于神经网络的通用股票预测模型 A general stock prediction model based on neural networks

## 1 项目介绍 Project Introduction

基本思路及创意来自于：https://github.com/MiaoChenglin125/stock_prediction-based-on-lstm-and-transformer
由于原作者貌似已经放弃更新，我将继续按照现有掌握的新的模型和技术，继续迭代这个程序

The basic idea and creativity come from: https://github.com/MiaoChenglin125/stock_prediction-based-on-lstm-and-transformer. As the original author seems to have abandoned the updates, I will continue to iterate this program based on the new models and technologies that I have mastered.

## New
* 20230327
* 1. 修改了部分运行逻辑，配合load pkl预处理文件，极大的提高了训练速度
*    Modified some running logic and used preprocessed pkl files to greatly improve training speed.
* 2. 修正了一个影响极大的关于数据流方向的bug
*    Corrected a bug that had a great impact on the direction of data flow.
* 3. 尝试使用新的模型
*    Tried a new model.
* 4. 增加了一个新的指标，用于评估模型的好坏
*    Added a new index to evaluate the quality of the model.
* 20230325
* 1. 增加数据预处理功能，并能将预处理好的queue保存为pkl文件，减少IO损耗
*    Add data preprocessing function and save the preprocessed queue as a pkl file to reduce IO loss.
* 2. 修改不必要的代码
*    Modify unnecessary code.
* 3. 简化逻辑，减少时间负责度，方向是以空间换时间
*    Simplify the logic and reduce the time burden. The direction is to trade space for time.
* 4. 增加常见的指标，增加预测精度
*    Add common indicators to increase prediction accuracy.
* 20230322
* 1. 增加输出内容控制，可以自行定义输出的内容和数量
*    Add output content control, you can define the content and quantity of the output yourself.
* 2. 修改读取数据源为本地csv文件
*    Modify the data source to local csv files.
* 3. 修改IO逻辑，使用多线程读取指定文件夹下的csv文件，并存储到内存中，反复训练，减少IO次数
*    Modify the IO logic to use multiple threads to read the csv files in the specified folder and store them in memory, repeatedly train, and reduce the number of IO operations.
* 4. 修改lstm, transformer模型
*    Modify the lstm and transformer models.
* 5. 增加下载数据功能，请使用自己的api token
*    Add download data function, please use your own api token.

## 获取下载数据的api token: Get the api token to download data:
* 1. 在https://tushare.pro/ 网站注册，并按要求获取足够的积分（到2023年3月为止，只需要修改下用户信息，就足够积分了，以后不能确定）
*    Register on https://tushare.pro/ website and obtain enough points as required (as of March 2023, only modifying user information is sufficient to obtain enough points, but it may not be guaranteed in the future).
* 2. 在https://tushare.pro/user/token 页面可以查看自己的api token
*    You can view your api token on https://tushare.pro/user/token page.
* 3. 在本项目根目录建立一个api.txt，并将获得的api token写入这个文件
*    Create an api.txt file in the root directory of this project and write the api token you obtained into this file.
* 4. 使用本项目getdata.py，即可自动下载日数据
*    Use the getdata.py of this project to automatically download the daily data.

股票行情是引导交易市场变化的一大重要因素，若能够掌握股票行情的走势，则对于个人和企业的投资都有巨大的帮助。然而，股票走势会受到多方因素的影响，因此难以从影响因素入手定量地进行衡量。但如今，借助于机器学习，可以通过搭建网络，学习一定规模的股票数据，通过网络训练，获取一个能够较为准确地预测股票行情的模型，很大程度地帮助我们掌握股票的走势。本项目便搭建了**LSTM（长短期记忆网络）**成功地预测了股票的走势。

Stock market trends are an important factor in guiding changes in the trading market. If we can master the trend of stock market trends, it would be of great help for personal and enterprise investments. However, stock market trends are influenced by multiple factors, making it difficult to measure them quantitatively from the perspective of influencing factors. Fortunately, with the help of machine learning, we can build a network, learn a certain scale of stock data, and obtain a model that can predict stock market trends more accurately through network training, which greatly helps us to master the trend of stocks. This project uses LSTM (Long Short-Term Memory Network) to successfully predict stock trends.

首先在**数据集**方面，我们选择上证000001号，中国平安股票（编号SZ_000001)数据集采用2016.01.01-2019.12.31股票数据，数据内容包括当天日期，开盘价，收盘价，最高价，最低价，交易量，换手率。数据集按照0.1比例分割产生测试集。训练过程以第T-99到T天数据作为训练输入，预测第T+1天该股票开盘价。(此处特别感谢**Tushare**提供的股票日数据集，欢迎大家多多支持)

First, in terms of the dataset, we chose the dataset of China Ping An stock (SZ_000001), which is the Shanghai Stock Exchange 000001 index. The stock data covers the period from January 1, 2016, to December 31, 2019, including the date, opening price, closing price, highest price, lowest price, trading volume, and turnover rate. The dataset is split into a test set and a training set in a 9:1 ratio. The training process takes data from T-99 to T days as input and predicts the opening price of the stock on day T+1. (Special thanks to Tushare for providing the daily stock dataset, and we encourage everyone to support them.)

在**训练模型及结果**方面，我们首先采用了LSTM（长短期记忆网络），它相比传统的神经网络能够保持上下文信息，更有利于股票预测模型基于原先的行情，预测未来的行情。LSTM网络帮助我们得到了很好的拟合结果，loss很快趋于0。之后，我们又采用比LSTM模型更新提出的Transformer Encoder部分进行测试。但发现，结果并没有LSTM优越，曲线拟合的误差较大，并且loss的下降较慢。因此本项目，重点介绍LSTM模型预测股票行情的实现思路。

In terms of training models and results, we first used LSTM (Long Short-Term Memory Network). Compared with traditional neural networks, LSTM can maintain context information, which is more conducive to predicting future market trends based on past trends. The LSTM network helped us achieve a good fitting result, and the loss quickly tended to 0. Then, we tested the Transformer Encoder part proposed by more updated models than LSTM. However, we found that the results were not superior to LSTM, and the fitting error was large, and the loss decreased slowly. Therefore, this project focuses on introducing the implementation of the LSTM model to predict stock market trends.

## 2 LSTM模型原理 Principles of LSTM model

### 2.1 时间序列模型 Time Series Model

 **时间序列模型**：时间序列预测分析就是利用过去一段时间内某事件时间的特征来预测未来一段时间内该事件的特征。这是一类相对比较复杂的预测建模问题，和回归分析模型的预测不同，时间序列模型是依赖于事件发生的先后顺序的，同样大小的值改变顺序后输入模型产生的结果是不同的。

  **Time series model**: Time series prediction analysis is to use the characteristics of the event time in the past period of time to predict the characteristics of the event in the future period of time. This is a relatively complex prediction modeling problem. Unlike the prediction of regression analysis models, time series models are dependent on the order of the occurrence of events. The same size of the value changes the order of input into the model to produce different results.

### 2.1 从RNN到LSTM From RNN to LSTM

**RNN：**递归神经网络RNN每一次隐含层的计算结果都与当前输入以及上一次的隐含层结果相关。通过这种方法，RNN的计算结果便具备了记忆之前几次结果的特点。其中，**ｘ**为输入层，ｏ为输出层，**s**为隐含层，而**t**指第几次的计算，**V,W,U**为权重，第ｔ次隐含层状态如下公式所示：

RNN (Recurrent Neural Network): The calculation result of each hidden layer of the recurrent neural network (RNN) is related to the current input and the previous hidden layer result. Through this method, the calculation result of RNN has the characteristic of memorizing the previous results. In the following formula, x is the input layer, o is the output layer, s is the hidden layer, and t indicates the calculation time, while V, W, U are weights. The formula for the t-th hidden layer state is as follows:

$$
St = f(U*Xt + W*St-1)　（１）
$$
<img src="readme\RNN.png" alt="RNN" style="zoom:50%;" />

可见，通过RNN模型想要当前隐含层状态与前ｎ次相关，需要增大计算量，复杂度呈指数级增长。然而采用LSTM网络可解决这一问题。

As we can see, if we want the current hidden layer state to be related to the previous n states through the RNN model, it requires a significant increase in computation, and the complexity grows exponentially. However, this problem can be solved by using the LSTM network.

**LSTM（长短期记忆网络）LSTM (Long Short-Term Memory Network)：**

LSTM是一种特殊的RNN，它主要是Eileen解决长序列训练过程中的梯度消失和梯度爆炸问题。相比RNN，LSTM更能够在长的序列中又更好的表现。

LSTM is a special RNN, which is mainly used to solve the problem of gradient disappearance and gradient explosion in the training process of long sequences. Compared with RNN, LSTM can perform better in long sequences.

<img src="readme\LSTM.png" alt="LSTM" style="zoom:67%;" />

LSTM拥有两个传输状态： ![[公式]](https://www.zhihu.com/equation?tex=c%5Et)在 （cell state）， ![[公式]](https://www.zhihu.com/equation?tex=h%5Et)（hidden state），其中 ![[公式]](https://www.zhihu.com/equation?tex=c%5Et) 的改变往往很慢，而 ![[公式]](https://www.zhihu.com/equation?tex=h%5Et)在不同的节点下会有很大的区别。

LSTM has two transmission states: ![[公式]](https://www.zhihu.com/equation?tex=c%5Et) in （cell state）， ![[公式]](https://www.zhihu.com/equation?tex=h%5Et)（hidden state），where the change of ![[公式]](https://www.zhihu.com/equation?tex=c%5Et) is often slow, while the hidden state ![[公式]](https://www.zhihu.com/equation?tex=h%5Et)  can vary greatly at different nodes.

- 首先，使用LSTM的当前输入 ![[公式]](https://www.zhihu.com/equation?tex=x%5Et)和上一个状态传递下来的![[公式]](https://www.zhihu.com/equation?tex=h%5E%7Bt-1%7D)得到四个状态：![[公式]](https://www.zhihu.com/equation?tex=z%5Ef+) ， ![[公式]](https://www.zhihu.com/equation?tex=z%5Ei) ，![[公式]](https://www.zhihu.com/equation?tex=z%5Eo)，![[公式]](https://www.zhihu.com/equation?tex=z)，前三者为拼接向量乘以权重矩阵后使用sigmoid函数得到0-1之间的值作为门控状态，后者为通过tanh函数得到（-1）-1之间的值。

First, using the current input ![[公式]](https://www.zhihu.com/equation?tex=x%5Et)  and the previous state ![[公式]](https://www.zhihu.com/equation?tex=h%5E%7Bt-1%7D)  passed down by LSTM, four states are obtained: ![[公式]](https://www.zhihu.com/equation?tex=z%5Ef+) ， ![[公式]](https://www.zhihu.com/equation?tex=z%5Ei) ，![[公式]](https://www.zhihu.com/equation?tex=z%5Eo)，![[公式]](https://www.zhihu.com/equation?tex=z)，, where the first three are gate states obtained by using sigmoid function on the concatenated vector multiplied by the weight matrix, and the last one is obtained by using the tanh function to get values between -1 and 1.

  <img src="readme\LSTM2.png" alt="LSTM2" style="zoom:67%;" />

- LSTM内部有三个阶段：**忘记阶段、选择记忆阶段、输出阶段**  LSTM has three stages: the forget stage, the select memory stage, and the output stage.

  - **忘记阶段：**通过计算 ![[公式]](https://www.zhihu.com/equation?tex=z%5Ef)来作为门控，控制上一个状态的 ![[公式]](https://www.zhihu.com/equation?tex=c%5E%7Bt-1%7D)需要遗忘的内容。

  - **Forget stage: **By computing ![[公式]](https://www.zhihu.com/equation?tex=z%5Ef) as a gate control, the content of the previous state ![[公式]](https://www.zhihu.com/equation?tex=c%5E%7Bt-1%7D) needs to be forgotten.

  - **选择记忆阶段：**对输入![[公式]](https://www.zhihu.com/equation?tex=x%5Et)进行选择记忆，门控信号由 ![[公式]](https://www.zhihu.com/equation?tex=z%5Ei)进行控制，输入内容由![[公式]](https://www.zhihu.com/equation?tex=z+)进行表示。

  - **Select memory stage: **Select memory for input ![[公式]](https://www.zhihu.com/equation?tex=x%5Et), gate control signal controlled by ![[公式]](https://www.zhihu.com/equation?tex=z%5Ei), and input content represented by ![[公式]](https://www.zhihu.com/equation?tex=z+).

  - **输出阶段：**决定当前状态输出的内容，通过 ![[公式]](https://www.zhihu.com/equation?tex=z%5Eo)控制，并且还对上一阶段得到的 ![[公式]](https://www.zhihu.com/equation?tex=c%5Et)进行放缩。

  - **Output stage: **Decide the content of the current state output, controlled by ![[公式]](https://www.zhihu.com/equation?tex=z%5Eo), and also scale the previous stage ![[公式]](https://www.zhihu.com/equation?tex=c%5Et).

    <img src="readme\LSTM3.png" alt="LSTM3" style="zoom:67%;" />



## ３LSTM预测股票模型实现 LSTM prediction model implementation for stock forecasting

**１、数据集准备**

**1. Data set preparation**

- 数据集分割：数据集按照0.1比例分割产生测试集。训练过程以第T-99到T天数据作为训练输入，预测第T+1天该股票开盘价。

- Data Set Splitting: The data set is split into training and testing sets in a 0.1 ratio, with the testing set being 10% of the data. During training, the input for predicting the opening price of the stock on day T+1 is the stock market data from day T-99 to day T. Therefore, the training set comprises data from the first day to day T-99, and the testing set comprises data from day T to the end of the data set. This methodology ensures that the LSTM model is effectively trained on historical stock market data to make predictions for future stock market trends.

- 对数据进行标准化：训练集与测试集都需要按列除以极差。在训练完成后需要进行逆处理来获得结果。

- Data normalization: Both the training set and the testing set need to be divided by the range of values along each column. After training is complete, the results need to be processed in reverse to obtain the results.

$$
train([:,i])=(train([:,i]))-min(train[:,i])/(max(train[:,i])-min(train[:,i])) （2）
$$

$$
test([:,i])=(test([:,i]))-min(train[:,i])/(max(train[:,i])-min(train[:,i])) （3）
$$



**２、模型搭建** 

**2. Model construction**

使用pytorch框架搭建LSTM模型，torch.nn.LSTM()当中包含的**参数设置**：

When building an LSTM model using the PyTorch framework with the torch.nn.LSTM() module：

- 输入特征的维数: input_size=dimension(dimension=8)

- The dimensionality of input features: input_size=dimension(dimension=8)

- LSTM中隐层的维度: hidden_size=128

- The dimension of the hidden layer in LSTM: hidden_size=128

- 循环神经网络的层数：num_layers=3

- The number of layers of the recurrent neural network: num_layers=3

- batch_first: TRUE

- 偏置：bias默认使用

- Bias: bias is used by default

**全连接层参数**设置：

The parameter settings for the fully connected layer include:

- 第一层：in_features=128, out_featrues=16

- The first layer: in_features=128, out_featrues=16

- 第二层：in_features=16, out_features=1 (映射到一个值)

- The second layer: in_features=16, out_features=1 (mapped to a value)

**３、模型训练**

**3. Model training**

- 经过调试，确定学习率lr=0.00001

- After debugging, a learning rate of lr=0.00001 was determined.

- 优化函数：批量梯度下降(SGD)

- Optimization function: Stochastic Gradient Descent (SGD)

- 批大小batch_size=4

- Batch size:batch_size=4

- 训练代数epoch=100

- Train epochs: epoch=100

- 损失函数：MSELoss均方损失函数，最终训练模型得到MSELoss下降为0.8左右。

- Loss function: MSELoss mean square loss function, the final training model obtained MSELoss down to about 0.8.

  <img src="readme\MSE.png" alt="MSE" style="zoom:80%;" />

**４、模型预测**

**4. Model prediction**

测试集使用已训练的模型进行验证，与真实数据不叫得到平均绝对百分比误差（MAPELoss）为0.04，可以得到测试集的准确率为96%。

The trained model was used to validate the test set, and the average absolute percentage error (MAPELoss) was obtained as 0.04, indicating a testing accuracy of 96% compared to the ground truth data.

<img src="readme\MAPE.png" alt="MAPE" style="zoom: 80%;" />

**５、模型成果**

**5. Model results**

下图是对整体数据集最后一百天的K线展示：当日开盘价低于收盘价则为红色，当日开盘价高于收盘价为绿色。图中还现实了当日交易量以及均线等信息。

The following figure shows the K-line display for the last hundred days of the entire dataset: red indicates that the opening price is lower than the closing price, while green indicates that the opening price is higher than the closing price. The figure also displays daily trading volume and moving average information.

<img src="readme\candleline.png" alt="candleline" style="zoom: 50%;" />

LSTM模型进行预测的测试集结果与真实结果对比图，可见LSTM模型预测的结果和现实股票的走势十分接近，因此具有很大的参考价值。

The following figure shows the comparison between the test results predicted by the LSTM model and the real results. It can be seen that the predicted results of the LSTM model are very close to the actual trend of the stock, indicating that it has great reference value.

<img src="readme\prediction.png" alt="prediction" style="zoom:50%;" />

LSTM模型训练过程中MSELoss的变化，可以看到随着训练代数的增加，此模型的MSELoss逐渐趋于0。

The change of MSELoss during the training process of the LSTM model can be seen, and it can be observed that with the increase of training iterations, the MSELoss of the model gradually approaches 0.

<img src="readme\loss.png" alt="loss" style="zoom: 67%;" />



## ４结语

## 4 Conclusion

本项目使用机器学习方法解决了股票市场预测的问题。项目采用开源股票数据中心的上证000001号，中国平安股票(编号SZ_000001)，使用更加适合进行长时间序列预测的LSTM(长短期记忆神经网络)进行训练，通过对训练集序列的训练，在测试集上预测开盘价，最终得到准确率为96%的LSTM股票预测模型，较为精准地实现解决了股票市场预测的问题。

This project uses machine learning methods to solve the problem of stock market prediction. The project uses the Shanghai Stock Exchange 000001, China Ping An stock (code SZ_000001) from an open-source stock data center and trains it using LSTM (Long Short-Term Memory Neural Network) which is more suitable for long-term sequence prediction. By training on the training set sequence and predicting the opening price on the test set, we obtained an LSTM stock prediction model with an accuracy rate of 96%, which effectively solves the problem of stock market prediction with high precision.

在项目开展过程当中，也采用过比LSTM更加新提出的Transformer模型，但对测试集的预测效果并不好，后期分析认为可能是由于在一般Transformer模型中由encoder和对应的decoder层，但在本项目的模型中使用了全连接层代替decoder，所以导致效果不佳。在后序的研究中，可以进一步改进，或许可以得到比LSTM更加优化的结果。

During the project, we also tried using the Transformer model, which is more recently proposed than LSTM. However, the prediction performance on the test set was not good. Further analysis revealed that this might be due to the fact that in the general Transformer model, there are encoder and corresponding decoder layers, but in our model, we used fully connected layers instead of the decoder. Therefore, the performance was not as good as expected. In future research, we can further improve the model or explore other approaches to potentially obtain better results than LSTM.

