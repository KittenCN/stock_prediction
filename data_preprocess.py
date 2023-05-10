import random
import argparse
from common import *

parser = argparse.ArgumentParser()
parser.add_argument('--pklname', default="train.pkl", type=str, help="code")
args = parser.parse_args()

if __name__ == ("__main__"):
    csv_files = glob.glob(daily_path+"/*.csv")
    # data_list = []
    ts_codes =[]
    Train_data = pd.DataFrame()
    data_len = 0
    dump_queue=queue.Queue()
    for csv_file in csv_files:
        ts_codes.append(os.path.basename(csv_file).rsplit(".", 1)[0])
    random.shuffle(ts_codes)
    # data_thread = threading.Thread(target=load_data, args=(ts_codes,))
    # data_thread.start()
    load_data(ts_codes, True)
    pbar = tqdm(total=len(ts_codes), leave=False, ncols=TQDM_NCOLS)
    while data_queue.empty() == False:
        try:
            data = data_queue.get(timeout=1)
            # data = data.dropna()
            # data.fillna(0, inplace=True)
            if data.empty or data["ts_code"][0] == "None":
                tqdm.write("data is empty or data has invalid col")
                pbar.update(1)
                continue
            ts_code = data["ts_code"][0]
            dump_queue.put(data)
            pbar.update(1)
        except Exception as e:
            print(ts_code, e)
            pbar.update(1)
            continue
    with open(pkl_path+"/"+args.pklname, "wb") as f:
        dill.dump(dump_queue, f)
    pbar.close()
    print("dump_queue size: ", dump_queue.qsize())