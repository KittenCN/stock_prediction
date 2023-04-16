from common import *
train_first_line = True
x_list = []
y_list = []

## ----type I
texts = pd.read_csv(open(bert_data_path+'/data'+'/Train_DataSet.csv',encoding='UTF-8'))
labels = pd.read_csv(open(bert_data_path+'/data'+'/Train_DataSet_Label.csv',encoding='UTF-8'))

pbar = tqdm(total=len(texts), leave=False)
for id in texts['id']:
    pbar.update(1)
    if labels[labels['id'] == id].empty:
        continue
    label = labels[labels['id'] == id]['label'].values[0]
    if not pd.isna(texts[texts['id'] == id]['title'].values[0]) and pd.isna(texts[texts['id'] == id]['content'].values[0]):
        text = texts[texts['id'] == id]['title'].values[0]
    elif pd.isna(texts[texts['id'] == id]['title'].values[0]) and not pd.isna(texts[texts['id'] == id]['content'].values[0]):
        text = texts[texts['id'] == id]['content'].values[0]
    elif not pd.isna(texts[texts['id'] == id]['content'].values[0]) and not pd.isna(texts[texts['id'] == id]['content'].values[0]):
        text = texts[texts['id'] == id]['title'].values[0]+','+texts[texts['id'] == id]['content'].values[0]
    else:
        continue
    text.replace("'",'').replace('"','')
    x_list.append(text)
    y_list.append(label*0.5)
pbar.close()

data = pd.DataFrame({'label':y_list,'text':x_list})
data.to_csv(bert_data_path+'/data'+'/Train2.csv',index=False,sep=',',encoding='utf-8')

##----type II
negative = open(bert_data_path+'/data'+'/negative.txt',encoding='UTF-8').readlines()
positive = open(bert_data_path+'/data'+'/positive.txt',encoding='UTF-8').readlines()
negative = map(lambda x: x.strip(), negative)
positive = map(lambda x: x.strip(), positive)
df_neg = pd.DataFrame({'label':0,'text':negative})
df_pos = pd.DataFrame({'label':1,'text':positive})
df = pd.concat([df_pos,df_neg],axis=0)
df.to_csv(bert_data_path+'/data'+'/Train3.csv',index=False,sep=',',encoding='utf-8')

