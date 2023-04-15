from common import *
train_first_line = True
x_list = []
y_list = []

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
