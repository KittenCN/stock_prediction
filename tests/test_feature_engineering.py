import sys
from pathlib import Path

import pandas as pd
import torch

# 添加 src 到路径，便于本地/CI 均可导入
root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from stock_prediction.app_config import FeatureSettings, SlidingWindowConfig, ExternalFeatureConfig  # noqa: E402
from stock_prediction.feature_engineering import FeatureEngineer  # noqa: E402


def _sample_dataframe() -> pd.DataFrame:
    data = {
        "ts_code": ["000001.SZ"] * 6 + ["000002.SZ"] * 6,
        "trade_date": [20240101 + i for i in range(6)] + [20240101 + i for i in range(6)],
        "open": [10 + 0.1 * i for i in range(12)],
        "close": [10.2 + 0.12 * i for i in range(12)],
        "pct_change": [0.01 * ((-1) ** i) for i in range(12)],
        "change": [0.2 * ((-1) ** i) for i in range(12)],
        "vol": [1000 + 10 * i for i in range(12)],
    }
    return pd.DataFrame(data)


def test_feature_engineer_generates_returns_and_windows(tmp_path):
    settings = FeatureSettings(
        price_columns=["close"],
        difference_columns=["close"],
        volatility_columns=["pct_change"],
        sliding_windows=[SlidingWindowConfig(size=3, stride=1, agg="mean")],
        external_sources=[
            ExternalFeatureConfig(
                name="macro",
                path=str(root_dir / "config" / "external" / "macro_sample.csv"),
                join_on="trade_date",
                forward_fill=True,
                domain="macro",
            )
        ],
        multi_stock=True,
    )

    engineer = FeatureEngineer(settings=settings)
    df = _sample_dataframe()
    result = engineer.transform(df)

    assert "close_ret_1" in result.columns
    assert "close_diff_1" in result.columns
    assert "pct_change_win3_mean" in result.columns

    macro_cols = [c for c in result.columns if c.startswith("macro_")]
    assert macro_cols, "宏观特征应合并到结果中"

    assert not result.isna().any().any(), "特征工程后的数据不应存在 NaN"
    # 保证张量转换无报错
    tensor_data = torch.tensor(result.select_dtypes(include="number").values, dtype=torch.float32)
    assert tensor_data.ndim == 2 and tensor_data.shape[1] >= 3


def test_symbol_normalization():
    settings = FeatureSettings(
        target_mode="difference",
        difference_order=0,
        price_columns=[],
        difference_columns=[],
        volatility_columns=[],
        enable_symbol_normalization=True,
        multi_stock=True,
    )
    engineer = FeatureEngineer(settings=settings)
    df = pd.DataFrame(
        {
            "ts_code": ["AAA"] * 5 + ["BBB"] * 5,
            "trade_date": [20240101 + i for i in range(5)] * 2,
            "open": list(range(5)) + [10 + i for i in range(5)],
            "close": list(range(5, 10)) + [20 + i for i in range(5)],
        }
    )
    result = engineer.transform(df)
    stats = engineer.get_symbol_stats()
    assert "AAA" in stats and "BBB" in stats
    for symbol in ("AAA", "BBB"):
        subset = result[result["ts_code"] == symbol]
        for col in ["open", "close"]:
            mean_val = subset[col].mean()
            assert abs(mean_val) < 1e-6
