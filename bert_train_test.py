from common import *

def get_train_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch_size',type=int,default=1,help = '每批数据的数量')
    parser.add_argument('--nepoch',type=int,default=3,help = '训练的轮次')
    parser.add_argument('--lr',type=float,default=LEARNING_RATE,help = '学习率')
    # parser.add_argument('--gpu',type=bool,default=True,help = '是否使用gpu')
    parser.add_argument('--num_workers',type=int,default=NUM_WORKERS,help='dataloader使用的线程数量')
    parser.add_argument('--num_labels',type=int,default=3,help='分类类数')
    parser.add_argument('--data_path',type=str,default=bert_data_path,help='数据路径')
    opt=parser.parse_args()
    print(opt)
    return opt

def get_model(opt):
    model = BertForSequenceClassification.from_pretrained(checkpoint,num_labels=opt.num_labels)
    return model

def get_data(opt):
    trainset = BertDataSet(opt.data_path,is_train = 1)
    trainloader=torch.utils.data.DataLoader(trainset,batch_size=opt.batch_size,shuffle=True,num_workers=opt.num_workers)
    testset = BertDataSet(opt.data_path,is_train = 0)
    testloader=torch.utils.data.DataLoader(testset,batch_size=opt.batch_size,shuffle=False,num_workers=opt.num_workers)
    return trainloader,testloader

def train(epoch,model,trainloader,testloader,optimizer,opt):
    global last_save_time
    model.train()
    start_time = time.time()
    print_step = int(len(trainloader)/10)
    pbar = tqdm(total=len(trainloader),ncols=TQDM_NCOLS,leave=False)
    for batch_idx,(sue,label,posi) in enumerate(trainloader):
        if device != 'cpu':
            sue = sue.cuda()
            posi = posi.cuda()
            label = label.unsqueeze(1).cuda()
        
        optimizer.zero_grad()
        outputs = model(sue, position_ids=posi,labels = label)

        loss, logits = outputs[0],outputs[1]
        loss.backward()
        optimizer.step()
        pbar.update(1)
        pbar.set_description("loss:%.2e" % loss.mean())
        if (batch_idx % print_step == 0 and time.time() - last_save_time >= SAVE_INTERVAL):
            last_save_time = time.time()
            torch.save(model.state_dict(),bert_data_path+'/model/bert_model.pth')
    pbar.close()


def test(epoch,model,trainloader,testloader,opt):
    model.eval()
    total=0
    correct=0
    with torch.no_grad():
        for batch_idx,(sue,label,posi) in enumerate(testloader):
            if device != 'cpu':
                sue = sue.cuda()
                posi = posi.cuda()
                labels = label.unsqueeze(1).cuda()
                label = label.cuda()
            else:
                labels = label.unsqueeze(1)
            
            outputs = model(sue, labels=labels)
            loss, logits = outputs[:2]
            _,predicted=torch.max(logits.data,1)

            total+=sue.size(0)
            correct+=predicted.data.eq(label.data).cpu().sum()
    
    s = ("Acc:%.3f" %((1.0*correct.numpy())/total))
    print(s)


if __name__=='__main__':
    global last_save_time
    last_save_time = 0
    opt = get_train_args()
    model = get_model(opt)
    trainloader,testloader = get_data(opt)
    
    if device != 'cpu':
        model.cuda()
    
    optimizer=torch.optim.AdamW(model.parameters(),lr=opt.lr)
    
    if os.path.exists(bert_data_path+'/model/bert_model.pth'):
        model.load_state_dict(torch.load(bert_data_path+'/model/bert_model.pth'))
    epoch_bar = tqdm(total=opt.nepoch,ncols=TQDM_NCOLS)
    for epoch in range(opt.nepoch):
        train(epoch,model,trainloader,testloader,optimizer,opt)
        epoch_bar.update(1)
    epoch_bar.close()
    torch.save(model.state_dict(),bert_data_path+'/model/bert_model.pth')
    test(epoch,model,trainloader,testloader,opt)