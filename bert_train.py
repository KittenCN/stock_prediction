import torch
import torch.utils.data as Data
from transformers import BertModel
from datasets import load_from_disk
from transformers import BertTokenizer,get_linear_schedule_with_warmup
# from transformers import AdamW
from torch.optim import AdamW
from common import *
import os
os.environ['NO_PROXY'] = 'huggingface.co'

def get_train_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch_size',type=int,default=1,help = 'batch_size')
    parser.add_argument('--nepoch',type=int,default=3,help = 'nepoch')
    parser.add_argument('--lr',type=float,default=5e-4,help = 'lr')
    # parser.add_argument('--num_workers',type=int,default=NUM_WORKERS,help='dnum_workers')
    parser.add_argument('--num_labels',type=int,default=2,help='num_labels')
    parser.add_argument('--no_grad',type=int,default=1,help='no_grad')
    opt=parser.parse_args()
    print(opt)
    return opt

def main(opt):
    global train_acc, train_best_acc, test_best_acc
    pretrained_model = BertModel.from_pretrained(bert_data_path+'/base_model/bert-base-chinese', cache_dir=bert_data_path+'/model/')  
    model = Bert_Model(pretrained_model, opt)  # 构建自己的模型
    if os.path.exists(bert_data_path+'/model/bert_model.pth'):
        model.load_state_dict(torch.load(bert_data_path+'/model/bert_model.pth'))
    # 如果有 gpu, 就用 gpu
    if torch.cuda.is_available():
        model.to(device)

    csv_file_path = bert_data_path+'/data/train'
    train_dataset = csvToDataset(csv_file_path)
    csv_file_path = bert_data_path+'/data/test'
    test_dataset = csvToDataset(csv_file_path)

    total_steps = len(train_dataset) * opt.nepoch  
    warmup_steps = total_steps * 0.1
    optimizer = AdamW(model.parameters(), lr=opt.lr)  
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    criterion = torch.nn.CrossEntropyLoss()  # 损失函数
    epochs = opt.nepoch  # 训练次数
    # 训练模型
    epoch_bar = tqdm(total=epochs, ncols=TQDM_NCOLS, leave=False)
    for i in range(epochs):
        epoch_bar.set_description("train acc: %.2f%% test acc: %.2f%%" % (train_acc * 100, test_acc * 100))
        train(model, train_dataset, criterion, optimizer, opt, scheduler)
        test(model, test_dataset, opt)
        torch.save(model.state_dict(),bert_data_path+'/model/bert_model.pth')
        if train_acc > train_best_acc:
            train_best_acc = train_acc
            torch.save(model.state_dict(),bert_data_path+'/model/bert_model_train_best.pth')
        if test_acc > test_best_acc:
            test_best_acc = test_acc
            torch.save(model.state_dict(),bert_data_path+'/model/bert_model_test_best.pth')
        epoch_bar.update(1)
    epoch_bar.close()

def train(model, dataset, criterion, optimizer, opt, scheduler):
    global test_acc, train_acc, train_best_acc, test_best_acc
    loader_train = Data.DataLoader(dataset=dataset,
                                   batch_size=opt.batch_size,
                                #    num_workers=opt.num_workers,
                                   collate_fn=collate_fn,
                                   shuffle=True,  
                                   drop_last=True)  
    model.train()
    total_acc_num = 0
    train_num = 0
    total_loss_num = 0
    iter_bar = tqdm(total=len(loader_train), ncols=TQDM_NCOLS, leave=False)
    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader_train):
        output = model(input_ids=input_ids, 
                       attention_mask=attention_mask, 
                       token_type_ids=token_type_ids)  
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        output = output.argmax(dim=1)  
        accuracy_num = (output == labels).sum().item()
        loss_num = abs((output - labels).sum().item())
        total_loss_num += loss_num
        total_acc_num += accuracy_num
        train_num += loader_train.batch_size
        iter_bar.update(1)
        iter_bar.set_description("loss: %.2e mean: %.2f acc: %.2f%%" % (loss.item(), total_loss_num / train_num, total_acc_num / train_num * 100))
        scheduler.step()
        if i % (len(loader_train) / 10) == 0 :
            torch.save(model.state_dict(),bert_data_path+'/model/bert_model.pth')
        if total_acc_num / train_num > train_best_acc:
            train_best_acc = total_acc_num / train_num
            torch.save(model.state_dict(),bert_data_path+'/model/bert_model_train_best.pth')
    iter_bar.close()
    train_acc = total_acc_num / train_num

def test(model, dataset, opt):
    global test_acc, train_best_acc, test_best_acc
    correct_num = 0
    test_num = 0
    loader_test = Data.DataLoader(dataset=dataset,
                                  batch_size=opt.batch_size,
                                #   num_workers=opt.num_workers,
                                  collate_fn=collate_fn,
                                  shuffle=True,
                                  drop_last=True)
    model.eval()
    for t, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader_test):
        with torch.no_grad():
            output = model(input_ids=input_ids,  
                           attention_mask=attention_mask,  
                           token_type_ids=token_type_ids)  
        output = output.argmax(dim=1)
        correct_num += (output == labels).sum().item()
        test_num += loader_test.batch_size
    test_acc = correct_num / test_num
    if test_acc > test_best_acc:
        test_best_acc = test_acc
        torch.save(model.state_dict(),bert_data_path+'/model/bert_model_test_best.pth')

def collate_fn(data):
    sentences = [tuple_x['text'] for tuple_x in data]
    labels = [int(tuple_x['label']) for tuple_x in data]
    token = BertTokenizer.from_pretrained(bert_data_path+'/base_model/bert-base-chinese', cache_dir=bert_data_path+'/model/')
    data = token.batch_encode_plus(batch_text_or_text_pairs=sentences,
                                   truncation=True,
                                   max_length=max_length,
                                   padding='max_length',
                                   return_tensors='pt',
                                   return_length=True)
    input_ids = data['input_ids'] 
    attention_mask = data['attention_mask'] 
    token_type_ids = data['token_type_ids'] 
    labels = torch.LongTensor(labels)
    if torch.cuda.is_available(): 
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        labels = labels.to(device)
    return input_ids, attention_mask, token_type_ids, labels


if __name__ == '__main__':
    global test_acc, train_acc, train_best_acc, test_best_acc
    max_length = 500
    test_acc = 0
    train_acc = 0
    train_best_acc = 0
    test_best_acc = 0
    opt = get_train_args()
    print('Use: ', device)
    main(opt)

