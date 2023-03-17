# 基于LSTM与Transfomer的股票预测模型

## 1 项目介绍

股票行情是引导交易市场变化的一大重要因素，若能够掌握股票行情的走势，则对于个人和企业的投资都有巨大的帮助。然而，股票走势会受到多方因素的影响，因此难以从影响因素入手定量地进行衡量。但如今，借助于机器学习，可以通过搭建网络，学习一定规模的股票数据，通过网络训练，获取一个能够较为准确地预测股票行情的模型，很大程度地帮助我们掌握股票的走势。本项目便搭建了**LSTM（长短期记忆网络）**成功地预测了股票的走势。

首先在**数据集**方面，我们选择上证000001号，中国平安股票（编号SZ_000001)数据集采用2016.01.01-2019.12.31股票数据，数据内容包括当天日期，开盘价，收盘价，最高价，最低价，交易量，换手率。数据集按照0.1比例分割产生测试集。训练过程以第T-99到T天数据作为训练输入，预测第T+1天该股票开盘价。(此处特别感谢**Tushare**提供的股票日数据集，欢迎大家多多支持)

在**训练模型及结果**方面，我们首先采用了LSTM（长短期记忆网络），它相比传统的神经网络能够保持上下文信息，更有利于股票预测模型基于原先的行情，预测未来的行情。LSTM网络帮助我们得到了很好的拟合结果，loss很快趋于0。之后，我们又采用比LSTM模型更新提出的Transformer Encoder部分进行测试。但发现，结果并没有LSTM优越，曲线拟合的误差较大，并且loss的下降较慢。因此本项目，重点介绍LSTM模型预测股票行情的实现思路。

## 2 LSTM模型原理

### 2.1 时间序列模型

 **时间序列模型**：时间序列预测分析就是利用过去一段时间内某事件时间的特征来预测未来一段时间内该事件的特征。这是一类相对比较复杂的预测建模问题，和回归分析模型的预测不同，时间序列模型是依赖于事件发生的先后顺序的，同样大小的值改变顺序后输入模型产生的结果是不同的。

### 2.1 从RNN到LSTM

**RNN：**递归神经网络RNN每一次隐含层的计算结果都与当前输入以及上一次的隐含层结果相关。通过这种方法，RNN的计算结果便具备了记忆之前几次结果的特点。其中，**ｘ**为输入层，ｏ为输出层，**s**为隐含层，而**t**指第几次的计算，**V,W,U**为权重，第ｔ次隐含层状态如下公式所示：
$$
St = f(U*Xt + W*St-1)　（１）
$$
<img src="readme\RNN.png" alt="RNN" style="zoom:50%;" />

可见，通过RNN模型想要当前隐含层状态与前ｎ次相关，需要增大计算量，复杂度呈指数级增长。然而采用LSTM网络可解决这一问题。

**LSTM（长短期记忆网络）：**

LSTM是一种特殊的RNN，它主要是Eileen解决长序列训练过程中的梯度消失和梯度爆炸问题。相比RNN，LSTM更能够在长的序列中又更好的表现。

<img src="readme\LSTM.png" alt="LSTM" style="zoom:67%;" />

LSTM拥有两个传输状态： ![[公式]](https://www.zhihu.com/equation?tex=c%5Et)在 （cell state）， ![[公式]](https://www.zhihu.com/equation?tex=h%5Et)（hidden state），其中 ![[公式]](https://www.zhihu.com/equation?tex=c%5Et) 的改变往往很慢，而 ![[公式]](https://www.zhihu.com/equation?tex=h%5Et)在不同的节点下会有很大的区别。

- 首先，使用LSTM的当前输入 ![[公式]](https://www.zhihu.com/equation?tex=x%5Et)和上一个状态传递下来的![[公式]](https://www.zhihu.com/equation?tex=h%5E%7Bt-1%7D)得到四个状态：![[公式]](https://www.zhihu.com/equation?tex=z%5Ef+) ， ![[公式]](https://www.zhihu.com/equation?tex=z%5Ei) ，![[公式]](https://www.zhihu.com/equation?tex=z%5Eo)，![[公式]](https://www.zhihu.com/equation?tex=z)，前三者为拼接向量乘以权重矩阵后使用sigmoid函数得到0-1之间的值作为门控状态，后者为通过tanh函数得到（-1）-1之间的值。

  <img src="readme\LSTM2.png" alt="LSTM2" style="zoom:67%;" />

- LSTM内部有三个阶段：**忘记阶段、选择记忆阶段、输出阶段**

  - **忘记阶段：**通过计算 ![[公式]](https://www.zhihu.com/equation?tex=z%5Ef)来作为门控，控制上一个状态的 ![[公式]](https://www.zhihu.com/equation?tex=c%5E%7Bt-1%7D)需要遗忘的内容。

  - **选择记忆阶段：**对输入![[公式]](https://www.zhihu.com/equation?tex=x%5Et)进行选择记忆，门控信号由 ![[公式]](https://www.zhihu.com/equation?tex=z%5Ei)进行控制，输入内容由![[公式]](https://www.zhihu.com/equation?tex=z+)进行表示。

  - **输出阶段：**决定当前状态输出的内容，通过 ![[公式]](https://www.zhihu.com/equation?tex=z%5Eo)控制，并且还对上一阶段得到的 ![[公式]](https://www.zhihu.com/equation?tex=c%5Et)进行放缩。

    <img src="readme\LSTM3.png" alt="LSTM3" style="zoom:67%;" />



## ３LSTM预测股票模型实现

**１、数据集准备**

- 数据集分割：数据集按照0.1比例分割产生测试集。训练过程以第T-99到T天数据作为训练输入，预测第T+1天该股票开盘价。
- 对数据进行标准化：训练集与测试集都需要按列除以极差。在训练完成后需要进行逆处理来获得结果。

$$
train([:,i])=(train([:,i]))-min(train[:,i])/(max(train[:,i])-min(train[:,i])) （2）
$$

$$
test([:,i])=(test([:,i]))-min(train[:,i])/(max(train[:,i])-min(train[:,i])) （3）
$$



**２、模型搭建**

使用pytorch框架搭建LSTM模型，torch.nn.LSTM()当中包含的**参数设置**：

- 输入特征的维数: input_size=dimension(dimension=8)
- LSTM中隐层的维度: hidden_size=128
- 循环神经网络的层数：num_layers=3
- batch_first: TRUE

- 偏置：bias默认使用

**全连接层参数**设置：

- 第一层：in_features=128, out_featrues=16
- 第二层：in_features=16, out_features=1 (映射到一个值)

**３、模型训练**

- 经过调试，确定学习率lr=0.00001

- 优化函数：批量梯度下降(SGD)

- 批大小batch_size=4

- 训练代数epoch=100

- 损失函数：MSELoss均方损失函数，最终训练模型得到MSELoss下降为0.8左右。

  <img src="readme\MSE.png" alt="MSE" style="zoom:80%;" />

**４、模型预测**

测试集使用已训练的模型进行验证，与真实数据不叫得到平均绝对百分比误差（MAPELoss）为0.04，可以得到测试集的准确率为96%。

<img src="readme\MAPE.png" alt="MAPE" style="zoom: 80%;" />

**５、模型成果**

下图是对整体数据集最后一百天的K线展示：当日开盘价低于收盘价则为红色，当日开盘价高于收盘价为绿色。图中还现实了当日交易量以及均线等信息。

<img src="readme\candleline.png" alt="candleline" style="zoom: 50%;" />

LSTM模型进行预测的测试集结果与真实结果对比图，可见LSTM模型预测的结果和现实股票的走势十分接近，因此具有很大的参考价值。

<img src="readme\prediction.png" alt="prediction" style="zoom:50%;" />

LSTM模型训练过程中MSELoss的变化，可以看到随着训练代数的增加，此模型的MSELoss逐渐趋于0。

<img src="readme\loss.png" alt="loss" style="zoom: 67%;" />



## ４结语

本项目使用机器学习方法解决了股票市场预测的问题。项目采用开源股票数据中心的上证000001号，中国平安股票（编号SZ_000001)，使用更加适合进行长时间序列预测的LSTM(长短期记忆神经网络)进行训练，通过对训练集序列的训练，在测试集上预测开盘价，最终得到准确率为96%的LSTM股票预测模型，较为精准地实现解决了股票市场预测的问题。

在项目开展过程当中，也采用过比LSTM更加新提出的Transformer模型，但对测试集的预测效果并不好，后期分析认为可能是由于在一般Transformer模型中由encoder和对应的decoder层，但在本项目的模型中使用了全连接层代替decoder，所以导致效果不佳。在后序的研究中，可以进一步改进，或许可以得到比LSTM更加优化的结果。


