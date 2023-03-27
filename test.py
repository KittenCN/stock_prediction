from matplotlib import pyplot as plt
import numpy as np
import common
import copy

common.SEQ_LEN=1
mean_list=[]
std_list=[]
input=[]
predict=[]
common.load_data(["test.1"])
data = common.data_queue.get()
data = data.dropna()
data.drop(['ts_code', 'Date'], axis=1, inplace=True)
ori_data = data.copy(deep=True).values[:, 0:common.INPUT_DIMENSION]
pre_data = data.copy(deep=True)
stock_train = common.Stock_Data(mode=0, dataFrame=pre_data, label_num=common.OUTPUT_DIMENSION)
# stock_test = common.Stock_Data(mode=1, dataFrame=ori_data, label_num=common.OUTPUT_DIMENSION)
# dataloader = common.DataLoaderX(dataset=stock_test, batch_size=common.BATCH_SIZE, shuffle=False, drop_last=True, num_workers=common.NUM_WORKERS, pin_memory=True)

# for i in range(len(ori_data[0])):
#     mean_list.append(np.mean(ori_data[:, i]))
#     std_list.append(np.std(ori_data[:, i]))
#     ori_data[:, i] = (ori_data[:, i] - mean_list[i]) / (std_list[i] + 1e-8)
# for i in range(len(ori_data[0])):
#     pre_data[:, i] = ori_data[:, i] * (std_list[i] + 1e-8) + mean_list[i]

if __name__=='__main__':
    label = stock_train.label.numpy()
    pre_data = copy.deepcopy(label)
    for i in range(len(pre_data[0])):
        pre_data[:, i] = pre_data[:, i] * (common.std_list[i] + 1e-8) + common.mean_list[i]

print(np.abs(data.copy(deep=True).values[:, 0:common.OUTPUT_DIMENSION] - pre_data).sum())
_real_list = np.transpose(np.flip(data.copy(deep=True).values[:, 0:common.INPUT_DIMENSION], 0))[0]
min_lin = min(min(_real_list) - 5, 0)
max_lin = max(_real_list) + 5
plt.figure()
x = np.linspace(min_lin, max_lin, len(_real_list))
plt.plot(x, np.array(_real_list), label="real_"+common.name_list[0])
plt.legend()
plt.show()