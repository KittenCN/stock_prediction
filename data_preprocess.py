import queue
import threading
import time
import pandas as pd
import common
import glob
import os
import dill
from tqdm import tqdm

if __name__ == ("__main__"):
    csv_files = glob.glob("./stock_daily/*.csv")
    data_list = []
    ts_codes =[]
    Train_data = pd.DataFrame()
    data_len = 0
    dump_queue=queue.Queue()
    for csv_file in csv_files:
        ts_codes.append(os.path.basename(csv_file).rsplit(".", 1)[0])
    data_thread = threading.Thread(target=common.load_data, args=(ts_codes,))
    data_thread.start()
    pbar = tqdm(total=len(ts_codes), leave=False)
    for index, ts_code in enumerate(ts_codes):
        try:
            # tqdm.set_description("Loading data: %s" % (ts_code))
            while common.data_queue.empty() == False:
                data_list += [common.data_queue.get()]
                data_len = max(data_len, common.data_queue.qsize())
            Err_nums = 5
            while index >= len(data_list):
                if common.data_queue.empty() == False:
                    data_list += [common.data_queue.get()]
                time.sleep(5)
                Err_nums -= 1
                if Err_nums == 0:
                    tqdm.write("Error: data_list is empty")
                    exit(0)
            data = data_list[index].copy(deep=True)
            data = data.dropna()
            if data is None or data["ts_code"][0] == "None":
                tqdm.write("data is empty or data has invalid col")
                pbar.update(1)
                continue
            if data['ts_code'][0] != ts_code:
                tqdm.write("Error: ts_code is not match")
                exit(0)
            data.drop(['ts_code','Date'],axis=1,inplace = True)    
            train_size=int(common.TRAIN_WEIGHT*(data.shape[0]))
            if train_size<common.SEQ_LEN or train_size+common.SEQ_LEN>data.shape[0]:
                # tqdm.write(ts_code + ":train_size is too small or too large")
                pbar.update(1)
                continue
            dump_queue.put(data[:train_size+common.SEQ_LEN])
            pbar.update(1)
        except Exception as e:
            print(ts_code, e)
            pbar.update(1)
            continue
    with open(common.train_pkl_path, "wb") as f:
        dill.dump(dump_queue, f)
    pbar.close()