import random
import argparse
import queue
import os

# 处理相对导入问题
try:
    from .init import daily_path, pkl_path, tqdm, TQDM_NCOLS, data_queue, pd, dill, glob
    from .common import load_data
except ImportError:
    # 如果直接运行此文件，使用绝对导入
    import sys
    from pathlib import Path
    current_dir = Path(__file__).resolve().parent
    root_dir = current_dir.parent.parent
    src_dir = root_dir / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from stock_prediction.init import daily_path, pkl_path, tqdm, TQDM_NCOLS, data_queue, pd, dill, glob
    from stock_prediction.common import load_data


def preprocess_data(pkl_name="train.pkl"):
    """数据预处理主函数"""
    csv_files = glob.glob(daily_path+"/*.csv")
    ts_codes = []
    dump_queue = queue.Queue()
    
    for csv_file in csv_files:
        ts_codes.append(os.path.basename(csv_file).rsplit(".", 1)[0])
    random.shuffle(ts_codes)
    
    load_data(ts_codes, True)
    pbar = tqdm(total=len(ts_codes), leave=False, ncols=TQDM_NCOLS)
    
    while not data_queue.empty():
        try:
            data = data_queue.get(timeout=1)
            if data.empty or data["ts_code"].iloc[0] == "None":
                tqdm.write("data is empty or data has invalid col")
                pbar.update(1)
                continue
            ts_code = data["ts_code"].iloc[0]
            dump_queue.put(data)
            pbar.update(1)
        except Exception as e:
            print(f"Error processing {ts_code}: {e}")
            pbar.update(1)
            continue
    
    with open(f"{pkl_path}/{pkl_name}", "wb") as f:
        dill.dump(dump_queue, f)
    pbar.close()
    print("dump_queue size: ", dump_queue.qsize())
    return dump_queue.qsize()


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--pklname', default="train.pkl", type=str, help="pkl file name")
    args = parser.parse_args()
    
    preprocess_data(args.pklname)


if __name__ == "__main__":
    main()