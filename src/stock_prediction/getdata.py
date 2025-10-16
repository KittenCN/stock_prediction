"""Data acquisition helpers for the stock_prediction package."""
from __future__ import annotations

import argparse
import datetime
import random
from pathlib import Path
from typing import Iterable, Sequence

try:
    from .init import (
        TQDM_NCOLS,
        NoneDataFrame,
        daily_path,
        pd,
        stock_data_queue,
        stock_list_queue,
        threading,
        time,
        tqdm,
    )
except ImportError:  # pragma: no cover
    import sys

    current_dir = Path(__file__).resolve().parent
    root_dir = current_dir.parent.parent
    src_dir = root_dir / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from stock_prediction.init import (
        TQDM_NCOLS,
        NoneDataFrame,
        daily_path,
        pd,
        stock_data_queue,
        stock_list_queue,
        threading,
        time,
        tqdm,
    )

try:
    import tushare as ts
except ImportError:
    ts = None

try:
    import akshare as ak
except ImportError:
    ak = None

try:
    import yfinance as yf
except ImportError:
    yf = None


class DataConfig:
    """Runtime switches controlling which upstream API is used."""

    def __init__(self) -> None:
        self.api = "akshare"
        self.adjust = "hfq"
        self.code = ""


config = DataConfig()


def set_adjust(adjust: str) -> None:
    """Update the adjustment flag used for downstream fetch operations."""

    config.adjust = adjust


def _rename_first_column(frame: pd.DataFrame, target_name: str) -> pd.DataFrame:
    """Rename the first column without referencing locale specific headers."""

    if frame.columns.empty:
        return frame
    first = frame.columns[0]
    if first != target_name:
        frame = frame.rename(columns={first: target_name})
    return frame


def get_stock_list() -> Sequence[str]:
    """Return a list of stock codes based on the configured API provider."""

    if config.api == "tushare":
        if ts is None:
            raise ImportError("tushare not installed")
        df = ts.pro_api().stock_basic(fields=["ts_code"])
        stock_list = df["ts_code"].tolist()
        stock_list_queue.put(stock_list)
        return stock_list

    if config.api == "akshare":
        if ak is None:
            raise ImportError("akshare not installed")
        stock_frame = ak.stock_zh_a_spot_em()
        stock_frame = _rename_first_column(stock_frame, "code")
        stock_list = stock_frame["code"].astype(str).tolist()
        stock_list_queue.put(stock_list)
        return stock_list

    raise ValueError(f"Unsupported api provider: {config.api}")


def _iterable_from_code(ts_code: str | Sequence[str]) -> Iterable[str]:
    if isinstance(ts_code, str):
        if ts_code:
            return [ts_code]
        return []
    return ts_code


def get_stock_data(ts_code: Sequence[str] | str = "", save: bool = True, start_code: str = "", save_path: Path | str = "", datediff: int = -1):
    """Download historical bar data for the provided symbols."""

    if isinstance(save_path, str):
        save_path = Path(save_path)

    if config.api == "tushare":
        if ts is None:
            raise ImportError("tushare not installed")

        pro = ts.pro_api()
        stock_list = list(_iterable_from_code(ts_code)) or get_stock_list()
        if start_code:
            stock_list = stock_list[stock_list.index(start_code):]
        pbar = tqdm(total=len(stock_list), leave=False, ncols=TQDM_NCOLS) if save else None
        lock = threading.Lock()

        with lock:
            adjust_suffix = f"_{config.adjust}" if config.adjust else ""
            for code in stock_list:
                try:
                    if config.adjust:
                        fields = [
                            "ts_code",
                            "trade_date",
                            f"open{adjust_suffix}",
                            f"high{adjust_suffix}",
                            f"low{adjust_suffix}",
                            f"close{adjust_suffix}",
                            f"pre_close{adjust_suffix}",
                            "change",
                            "pct_change",
                            "vol",
                            "amount",
                        ]
                        df = pro.stk_factor(ts_code=code, fields=fields)
                        df.columns = [
                            "ts_code",
                            "trade_date",
                            "open",
                            "high",
                            "low",
                            "close",
                            "pre_close",
                            "change",
                            "pct_change",
                            "vol",
                            "amount",
                        ]
                    else:
                        df = pro.daily(ts_code=code, fields=[
                            "ts_code",
                            "trade_date",
                            "open",
                            "high",
                            "low",
                            "close",
                            "pre_close",
                            "change",
                            "pct_chg",
                            "vol",
                            "amount",
                        ])
                        df = df.rename(columns={"pct_chg": "pct_change"})
                    df = df.reindex(columns=[
                        "ts_code",
                        "trade_date",
                        "open",
                        "high",
                        "low",
                        "close",
                        "change",
                        "pct_change",
                        "vol",
                        "amount",
                        "pre_close",
                    ])
                except Exception as exc:  # pragma: no cover
                    message = f"{code} {exc}"
                    if save and pbar is not None:
                        tqdm.write(message)
                        pbar.update(1)
                    else:
                        print(message)
                    continue

                time.sleep(random.uniform(0.1, 0.9))
                if save:
                    save_path.mkdir(parents=True, exist_ok=True)
                    df.to_csv(save_path / f"{code}.csv", index=False)
                    if pbar is not None:
                        pbar.update(1)
                else:
                    stock_data_queue.put(df if not df.empty else NoneDataFrame)
                    return df if not df.empty else None

        if pbar is not None:
            pbar.close()
        return None

    if config.api == "akshare":
        if ak is None:
            raise ImportError("akshare not installed")

        stock_list = list(_iterable_from_code(ts_code)) or get_stock_list()
        if start_code:
            stock_list = stock_list[stock_list.index(start_code):]
        pbar = tqdm(total=len(stock_list), leave=False, ncols=TQDM_NCOLS) if save else None
        lock = threading.Lock()

        with lock:
            end_date = (datetime.datetime.now() + datetime.timedelta(days=datediff)).strftime("%Y%m%d")
            for code in stock_list:
                try:
                    df = ak.stock_zh_a_hist(symbol=code, period="daily", end_date=end_date, adjust=config.adjust)
                    df.columns = [
                        "trade_date",
                        "ts_code",
                        "open",
                        "close",
                        "high",
                        "low",
                        "vol",
                        "amount",
                        "amplitude",
                        "pct_change",
                        "change",
                        "exchange_rate",
                    ]
                    columns = list(df.columns)
                    columns[0], columns[1] = columns[1], columns[0]
                    df = df[columns]
                    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y%m%d")
                    df.sort_values(by=["trade_date"], ascending=False, inplace=True)
                    df = df.reindex(columns=[
                        "ts_code",
                        "trade_date",
                        "open",
                        "high",
                        "low",
                        "close",
                        "change",
                        "pct_change",
                        "vol",
                        "amount",
                        "amplitude",
                        "exchange_rate",
                    ])
                except Exception as exc:  # pragma: no cover
                    message = f"{code} {exc}"
                    if save and pbar is not None:
                        tqdm.write(message)
                        pbar.update(1)
                    else:
                        print(message)
                    if getattr(exc, "args", []) and isinstance(exc.args[0], Exception):
                        inner = exc.args[0]
                        text = str(inner)
                        if "Connection aborted" in text or "Remote end closed connection" in text:
                            break
                    continue

                time.sleep(random.uniform(0.1, 0.9))
                if save:
                    save_path.mkdir(parents=True, exist_ok=True)
                    df.to_csv(save_path / f"{code}.csv", index=False)
                    if pbar is not None:
                        pbar.update(1)
                else:
                    stock_data_queue.put(df if not df.empty else NoneDataFrame)
                    return df if not df.empty else None

        if pbar is not None:
            pbar.close()
        return None

    if config.api == "yfinance":
        if yf is None:
            raise ImportError("yfinance not installed")

        auto_adjust = back_adjust = False
        if config.adjust == "qfq":
            auto_adjust = True
        elif config.adjust == "hfq":
            auto_adjust = True
            back_adjust = True

        stock_list = list(_iterable_from_code(ts_code))
        pbar = tqdm(total=len(stock_list), leave=False, ncols=TQDM_NCOLS) if save else None
        lock = threading.Lock()

        with lock:
            for code in stock_list:
                try:
                    df = yf.download(code, auto_adjust=auto_adjust, back_adjust=back_adjust)
                    df.reset_index(inplace=True)
                    df.insert(0, "ts_code", code)
                    df.columns = [
                        "ts_code",
                        "trade_date",
                        "open",
                        "high",
                        "low",
                        "close",
                        "vol",
                    ]
                    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y%m%d")
                    df.sort_values(by=["trade_date"], ascending=False, inplace=True)
                    df = df.reindex(columns=[
                        "ts_code",
                        "trade_date",
                        "open",
                        "high",
                        "low",
                        "close",
                        "vol",
                    ])
                except Exception as exc:  # pragma: no cover
                    message = f"{code} {exc}"
                    if save and pbar is not None:
                        tqdm.write(message)
                        pbar.update(1)
                    else:
                        print(message)
                    continue

                if save:
                    save_path.mkdir(parents=True, exist_ok=True)
                    df.to_csv(save_path / f"{code}.csv", index=False)
                    if pbar is not None:
                        pbar.update(1)
                else:
                    stock_data_queue.put(df if not df.empty else NoneDataFrame)
                    return df if not df.empty else None

        if pbar is not None:
            pbar.close()
        return None

    raise ValueError(f"Unsupported api provider: {config.api}")


def main() -> None:
    """Command-line entry point for fetching quote data."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--code", default="", type=str, help="single stock code or ticker")
    parser.add_argument("--api", default="akshare", type=str, help="api-provider: tushare, akshare or yfinance")
    parser.add_argument("--adjust", default="hfq", type=str, help="adjustment: none, qfq, or hfq")
    args = parser.parse_args()

    config.api = args.api
    config.adjust = args.adjust
    config.code = args.code

    if args.api == "yfinance":
        tickers = ["DAX", "IBM"]
        if not tickers:
            raise ValueError("Please provide at least one ticker when using yfinance")
        get_stock_data(tickers, save=True, save_path=daily_path)
        return

    if args.code:
        get_stock_data(args.code, save=True, save_path=daily_path)
    else:
        get_stock_data("", save=True, save_path=daily_path, datediff=-1)


if __name__ == "__main__":
    main()
