"""Utilities for converting daily CSV data into serialized queues."""
from __future__ import annotations

import argparse
import os
import queue
import random
from pathlib import Path

try:
    from .common import load_data
    from .init import TQDM_NCOLS, data_queue, daily_path, dill, glob, pd, pkl_path, tqdm
except ImportError:  # pragma: no cover
    import sys

    current_dir = Path(__file__).resolve().parent
    root_dir = current_dir.parent.parent
    src_dir = root_dir / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from stock_prediction.common import load_data
    from stock_prediction.init import TQDM_NCOLS, data_queue, daily_path, dill, glob, pd, pkl_path, tqdm


def preprocess_data(pkl_name: str = "train.pkl") -> int:
    """Convert the prepared daily CSV files into a serialized queue for training."""

    daily_dir = Path(daily_path)
    pkl_dir = Path(pkl_path)

    csv_files = [str(path) for path in daily_dir.glob("*.csv")]
    ts_codes: list[str] = []
    dump_queue: queue.Queue[pd.DataFrame] = queue.Queue()

    for csv_file in csv_files:
        ts_codes.append(os.path.basename(csv_file).rsplit(".", 1)[0])
    random.shuffle(ts_codes)

    load_data(ts_codes, True)
    pbar = tqdm(total=len(ts_codes), leave=False, ncols=TQDM_NCOLS)

    while not data_queue.empty():
        try:
            data = data_queue.get(timeout=1)
            if data.empty or data["ts_code"].iloc[0] == "None":
                tqdm.write("data is empty or has invalid columns")
                pbar.update(1)
                continue
            ts_code = data["ts_code"].iloc[0]
            dump_queue.put(data)
            pbar.update(1)
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"Error processing {ts_code}: {exc}")
            pbar.update(1)
            continue

    with open(pkl_dir / pkl_name, "wb") as fh:
        dill.dump(dump_queue, fh)
    pbar.close()
    print("dump_queue size:", dump_queue.qsize())
    return dump_queue.qsize()


def main() -> None:
    """Command-line entrypoint."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--pklname", default="train.pkl", type=str, help="output pkl filename")
    args = parser.parse_args()

    preprocess_data(args.pklname)


if __name__ == "__main__":
    main()
