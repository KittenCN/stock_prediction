import torch
import torch.utils.data as Data
from transformers import BertModel
from datasets import load_from_disk
from transformers import BertTokenizer
from transformers import AdamW
from init import *

def get_train_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch_size',type=int,default=1,help = '每批数据的数量')
    parser.add_argument('--nepoch',type=int,default=3,help = '训练的轮次')
    parser.add_argument('--lr',type=float,default=5e-4,help = '学习率')
    # parser.add_argument('--num_workers',type=int,default=NUM_WORKERS,help='dataloader使用的线程数量')
    parser.add_argument('--num_labels',type=int,default=2,help='分类类数')
    opt=parser.parse_args()
    print(opt)
    return opt

def main(opt):
    pretrained_model = BertModel.from_pretrained('bert-base-chinese', cache_dir=bert_data_path+'/model/')  # 加载预训练模型
    model = Model(pretrained_model, opt)  # 构建自己的模型
    if os.path.exists(bert_data_path+'/model/bert_model.pth'):
        model.load_state_dict(torch.load(bert_data_path+'/model/bert_model.pth'))
    # 如果有 gpu, 就用 gpu
    if torch.cuda.is_available():
        model.to(device)
    train_data = load_from_disk(bert_data_path+'/data/ChnSentiCorp/')['train']  # 加载训练数据
    test_data = load_from_disk(bert_data_path+'/data/ChnSentiCorp/')['test']  # 加载测试数据
    optimizer = AdamW(model.parameters(), lr=opt.lr)  # 优化器
    criterion = torch.nn.CrossEntropyLoss()  # 损失函数
    epochs = opt.nepoch  # 训练次数
    # 训练模型
    epoch_bar = tqdm(total=epochs, ncols=TQDM_NCOLS, leave=False)
    for i in range(epochs):
        # print("--------------- >>>> epoch : {} <<<< -----------------".format(i))
        train(model, train_data, criterion, optimizer, opt)
        test(model, test_data, opt)
        torch.save(model.state_dict(),bert_data_path+'/model/bert_model.pth')
        epoch_bar.update(1)
    epoch_bar.close()


# 定义下游任务模型
class Model(torch.nn.Module):
    def __init__(self, pretrained_model, opt):
        super().__init__()
        self.pretrain_model = pretrained_model
        self.fc = torch.nn.Linear(768, opt.num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():  # 上游的模型不进行梯度更新
            output = self.pretrain_model(input_ids=input_ids,  # input_ids: 编码之后的数字(即token)
                                         attention_mask=attention_mask,  # attention_mask: 其中 pad 的位置是 0 , 其他位置是 1
                                         # token_type_ids: 第一个句子和特殊符号的位置是 0 , 第二个句子的位置是 1
                                         token_type_ids=token_type_ids)
        output = self.fc(output[0][:, 0])  # 取出每个 batch 的第一列作为 CLS, 即 (16, 786)
        output = output.softmax(dim=1)  # 通过 softmax 函数, 并使其在 1 的维度上进行缩放，使元素位于[0,1] 范围内，总和为 1
        return output


def train(model, dataset, criterion, optimizer, opt):
    global test_acc, last_save_time
    loader_train = Data.DataLoader(dataset=dataset,
                                   batch_size=opt.batch_size,
                                   collate_fn=collate_fn,
                                   shuffle=True,  # 顺序打乱
                                   drop_last=True)  # 设置为'True'时，如果数据集大小不能被批处理大小整除，则删除最后一个不完整的批次
    model.train()
    total_acc_num = 0
    train_num = 0
    iter_bar = tqdm(total=len(loader_train), ncols=TQDM_NCOLS, leave=False)
    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader_train):
        output = model(input_ids=input_ids,  # input_ids: 编码之后的数字(即token)
                       attention_mask=attention_mask,  # attention_mask: 其中 pad 的位置是 0 , 其他位置是 1
                       token_type_ids=token_type_ids)  # token_type_ids: 第一个句子和特殊符号的位置是 0 , 第二个句子
        # 计算 loss, 反向传播, 梯度清零
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # 算 acc
        output = output.argmax(dim=1)  # 取出所有在维度 1 上的最大值的下标
        accuracy_num = (output == labels).sum().item()
        total_acc_num += accuracy_num
        train_num += loader_train.batch_size
        iter_bar.update(1)
        iter_bar.set_description("loss: %.2e acc: %.2e test: %.2e" % (loss.item(), total_acc_num / train_num, test_acc))
        if i % (len(loader_train) / 10) == 0 and time.time() - last_save_time > SAVE_INTERVAL:
            torch.save(model.state_dict(),bert_data_path+'/model/bert_model.pth')
            last_save_time = time.time()
            # print("train_schedule: [{}/{}] train_loss: {} train_acc: {}".format(i, len(loader_train),
            #                                                                     loss.item(), total_acc_num / train_num))
    iter_bar.close()
    print("total train_acc: {}".format(total_acc_num / train_num))


def test(model, dataset, opt):
    global test_acc
    correct_num = 0
    test_num = 0
    loader_test = Data.DataLoader(dataset=dataset,
                                  batch_size=opt.batch_size,
                                  collate_fn=collate_fn,
                                  shuffle=True,
                                  drop_last=True)
    model.eval()
    for t, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader_test):
        with torch.no_grad():
            output = model(input_ids=input_ids,  # input_ids: 编码之后的数字(即token)
                           attention_mask=attention_mask,  # attention_mask: 其中 pad 的位置是 0 , 其他位置是 1
                           token_type_ids=token_type_ids)  # token_type_ids: 第一个句子和特殊符号的位置是 0 , 第二个句子
        output = output.argmax(dim=1)
        correct_num += (output == labels).sum().item()
        test_num += loader_test.batch_size
        # if t % 10 == 0:
        #     print("schedule: [{}/{}] acc: {}".format(t, len(loader_test), correct_num / test_num))
    test_acc = correct_num / test_num
    # print("total test_acc: {}".format(correct_num / test_num))


def collate_fn(data):
    # 将数据中的文本和标签分别提取出来
    sentences = [tuple_x['text'] for tuple_x in data]
    labels = [tuple_x['label'] for tuple_x in data]
    # 加载字典和分词工具
    token = BertTokenizer.from_pretrained('bert-base-chinese', cache_dir='./my_vocab')
    # 对数据进行编码
    data = token.batch_encode_plus(batch_text_or_text_pairs=sentences,
                                   truncation=True,
                                   max_length=500,
                                   padding='max_length',
                                   return_tensors='pt',
                                   return_length=True)
    input_ids = data['input_ids']  # input_ids: 编码之后的数字(即token)
    attention_mask = data['attention_mask']  # attention_mask: 其中 pad 的位置是 0 , 其他位置是 1
    token_type_ids = data['token_type_ids']  # token_type_ids: 第一个句子和特殊符号的位置是 0 , 第二个句子的位置是 1
    labels = torch.LongTensor(labels)
    if torch.cuda.is_available():  # 如果有 gpu, 就用 gpu
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        labels = labels.to(device)
    return input_ids, attention_mask, token_type_ids, labels


if __name__ == '__main__':
    global test_acc, last_save_time
    last_save_time = 0
    test_acc = 0
    opt = get_train_args()
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 全局变量
    print('Use: ', device)
    main(opt)

