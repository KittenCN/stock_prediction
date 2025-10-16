"""Utility helpers for persisting logs and converting JSON payloads."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

try:
    from .init import data_path, daily_path
except ImportError:  # pragma: no cover
    import sys

    current_dir = Path(__file__).resolve().parent
    root_dir = current_dir.parent.parent
    src_dir = root_dir / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from stock_prediction.init import data_path, daily_path


def _ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_page(stock_id: str, logstr: str, log_path: str | Path | None = None) -> None:
    """Write page progress markers to disk."""

    target_dir = _ensure_dir(log_path or f"{data_path}/log")
    (target_dir / stock_id).write_text(logstr, encoding="utf-8")


def read_page(stock_id: str, log_path: str | Path | None = None) -> int:
    """Read page progress markers from disk."""

    target_dir = _ensure_dir(log_path or f"{data_path}/log")
    file_path = target_dir / stock_id
    if not file_path.exists():
        return 0
    line = file_path.read_text(encoding="utf-8").splitlines()[0]
    return int(line)


def write_log(stock_id: str, logstr: str, log_path: str | Path | None = None) -> None:
    """Append a log entry for the given stock id."""

    target_dir = _ensure_dir(log_path or f"{data_path}/log")
    with (target_dir / stock_id).open('a', encoding='utf-8') as fh:
        fh.write(logstr + '\n')


def read_log(stock_id: str, log_path: str | Path | None = None) -> dict:
    """Read log entries stored as JSON lines."""

    target_dir = _ensure_dir(log_path or f"{data_path}/log")
    file_path = target_dir / stock_id
    if not file_path.exists():
        return {}
    result: dict[str, dict] = {}
    with file_path.open('r', encoding='utf-8') as fh:
        for line in fh:
            record = json.loads(line.rstrip('\n'))
            result[record['comment_url']] = record
    return result


def write_url(stock_id: str, logstr: str, log_path: str | Path | None = None) -> None:
    """Append a URL entry for the given stock id."""

    target_dir = _ensure_dir(log_path or f"{data_path}/log")
    with (target_dir / stock_id).open('a', encoding='utf-8') as fh:
        fh.write(logstr + '\n')


def read_url(stock_id: str, log_path: str | Path | None = None) -> set[str]:
    """Read distinct URL entries from disk."""

    target_dir = _ensure_dir(log_path or f"{data_path}/log")
    file_path = target_dir / stock_id
    if not file_path.exists():
        return set()
    with file_path.open('r', encoding='utf-8') as fh:
        return {line.rstrip('\n') for line in fh}


def json2csv(path: str | Path, save_path: str | Path) -> None:
    """Convert a folder containing JSON documents to a CSV file."""

    path = Path(path)
    frames = []
    for file in path.iterdir():
        if file.is_file():
            with file.open('r', encoding='utf-8') as fh:
                item = json.load(fh)
                frames.append(pd.DataFrame(item, index=[0]))
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True).set_index('time')
    df.drop(['read', 'subcomments', 'comment_url', 'comment_id'], axis=1, inplace=True)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path)


def cal_compounding_factor(ts_code: str = ""):
    """Calculate a compounding factor using adjusted and raw price data."""

    import glob
    import numpy as np
    import random

    try:
        from .getdata import get_stock_data, set_adjust
    except ImportError:  # pragma: no cover
        from stock_prediction.getdata import get_stock_data, set_adjust

    if ts_code == "":
        csv_files = glob.glob(f"{daily_path}/*.csv")
        ts_codes = [Path(csv).stem for csv in csv_files]
        ts_code = random.sample(ts_codes, 1)[0]

    data_file_path = Path(data_path) / f"{ts_code}.csv"
    daily_file_path = Path(daily_path) / f"{ts_code}.csv"
    set_adjust("")
    get_stock_data(ts_code, save=True, save_path=data_path)

    if not (daily_file_path.exists() and data_file_path.exists()):
        return None

    df_daily = pd.read_csv(daily_file_path)
    df_daily['trade_date'] = pd.to_datetime(df_daily['trade_date'], format='%Y%m%d')
    df_daily = df_daily.sort_values(by='trade_date', ascending=False)
    daily_open = df_daily['open'].iloc[0]
    daily_close = df_daily['close'].iloc[0]
    daily_high = df_daily['high'].iloc[0]
    daily_low = df_daily['low'].iloc[0]
    latest_date = df_daily['trade_date'].iloc[0]

    df_data = pd.read_csv(data_file_path)
    df_data['trade_date'] = pd.to_datetime(df_data['trade_date'], format='%Y%m%d')
    df_data = df_data.sort_values(by='trade_date', ascending=False)

    matched = df_data.loc[df_data['trade_date'] == latest_date]
    if matched.empty:
        return None

    data_open = matched['open'].iloc[0]
    data_close = matched['close'].iloc[0]
    data_high = matched['high'].iloc[0]
    data_low = matched['low'].iloc[0]

    factors = [
        data_open / daily_open,
        data_close / daily_close,
        data_high / daily_high,
        data_low / daily_low,
    ]
    return float(np.mean(factors))
