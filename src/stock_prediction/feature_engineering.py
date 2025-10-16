"""
Feature engineering utilities enabling PTFT+VSSM 改进计划中提出的收益率建模、外生特征融合与滑动窗口统计。

该模块围绕 FeatureSettings 提供标准化接口，用于：
1. 基于收盘价等价格字段生成对数收益率或差分序列；
2. 按照配置引入宏观、行业、舆情等外生特征；
3. 在多股票场景下支持按股票分组的滑动窗口统计量；
4. 输出包含缺失值处理和排序后的 DataFrame，供下游 Dataset 使用。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from .app_config import AppConfig, ExternalFeatureConfig, FeatureSettings, SlidingWindowConfig


@dataclass
class FeatureEngineerCache:
    """简单的缓存结构，避免多次读取相同外生特征文件。"""

    external_frames: Dict[str, pd.DataFrame]

    def __init__(self) -> None:
        self.external_frames = {}

    def get(self, key: str) -> Optional[pd.DataFrame]:
        return self.external_frames.get(key)

    def set(self, key: str, value: pd.DataFrame) -> None:
        self.external_frames[key] = value


class FeatureEngineer:
    """负责根据 FeatureSettings 对原始行情数据做增强处理。"""

    def __init__(self, settings: FeatureSettings, cache: Optional[FeatureEngineerCache] = None) -> None:
        self.settings = settings
        self.cache = cache or FeatureEngineerCache()

    @classmethod
    def from_app_config(cls, cfg: AppConfig) -> "FeatureEngineer":
        return cls(cfg.features if hasattr(cfg, "features") else FeatureSettings())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """对输入 DataFrame 进行特征工程，返回增强后的 DataFrame。

        要求：
        - DataFrame 至少包含 trade_date；若存在 ts_code，则默认按股票分组处理；
        - 返回结果按照 trade_date 升序排列，去除前置 burn-in 行并填充缺失值；
        """
        if df.empty:
            return df

        if "trade_date" not in df.columns:
            if "Date" in df.columns:
                df = df.rename(columns={"Date": "trade_date"})
            else:
                # 缺少日期列时直接返回原始数据，避免中断训练流程
                return df

        df = df.copy()
        df["trade_date"] = self._normalise_trade_date(df["trade_date"])

        if self.settings.multi_stock and "ts_code" in df.columns:
            processed_frames = []
            for ts_code, group in df.groupby("ts_code"):
                processed = self._process_single_stock(group.copy(), symbol=str(ts_code))
                processed_frames.append(processed)
            if not processed_frames:
                return df
            result = pd.concat(processed_frames, ignore_index=True)
        else:
            result = self._process_single_stock(df, symbol=str(df.get("ts_code", "UNKNOWN").iloc[0]) if "ts_code" in df.columns else None)

        result = result.replace([np.inf, -np.inf], np.nan)
        if self.settings.align_holiday:
            result = result.sort_values("trade_date")
            if "ts_code" in result.columns:
                grouped_frames = []
                for _, group in result.groupby("ts_code", as_index=False, group_keys=False):
                    grouped_frames.append(group.ffill().bfill())
                result = pd.concat(grouped_frames, ignore_index=True) if grouped_frames else result
            else:
                result = result.ffill().bfill()

        burn_in = max(self.settings.return_lag, self.settings.difference_order)
        if burn_in > 0 and len(result) > burn_in:
            result = result.iloc[burn_in:].reset_index(drop=True)
        else:
            result = result.reset_index(drop=True)

        result = result.ffill().bfill().fillna(0.0)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _process_single_stock(self, df: pd.DataFrame, symbol: Optional[str]) -> pd.DataFrame:
        df = df.sort_values("trade_date").reset_index(drop=True)
        df = self._compute_returns(df)
        df = self._compute_differences(df)
        df = self._append_sliding_windows(df)
        df = self._merge_external_features(df, symbol)
        return df

    def _compute_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        mode = self.settings.target_mode
        if mode not in {"log_return", "pct_return", "hybrid"}:
            return df

        columns = [col for col in self.settings.price_columns if col in df.columns]
        if not columns:
            return df

        lag = self.settings.return_lag
        for col in columns:
            shifted = df[col].shift(lag)
            if self.settings.return_kind == "log":
                safe_current = df[col].clip(lower=1e-6)
                safe_shifted = shifted.clip(lower=1e-6)
                returns = np.log(safe_current) - np.log(safe_shifted)
            else:
                returns = (df[col] - shifted) / (shifted.abs() + 1e-6)
            df[f"{col}_ret_{lag}"] = returns
            if mode == "hybrid":
                df[f"{col}_pct_{lag}"] = (df[col] - shifted) / (shifted.abs() + 1e-6)

        return df

    def _compute_differences(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.settings.difference_order <= 0:
            return df

        columns = [col for col in self.settings.difference_columns if col in df.columns]
        if not columns:
            return df

        order = self.settings.difference_order
        for col in columns:
            diff_series = df[col].diff(periods=order)
            df[f"{col}_diff_{order}"] = diff_series
        return df

    def _append_sliding_windows(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.settings.sliding_windows:
            return df

        target_columns = [col for col in self.settings.volatility_columns if col in df.columns]
        if not target_columns:
            return df

        for window in self.settings.sliding_windows:
            for col in target_columns:
                rolling = df[col].rolling(window.size, min_periods=window.size)
                agg_value = self._apply_agg(rolling, window.agg)
                df[f"{col}_win{window.size}_{window.agg}"] = agg_value
        return df

    def _apply_agg(self, rolling_obj: pd.core.window.Rolling, agg: str) -> pd.Series:
        agg = agg.lower()
        if agg == "mean":
            return rolling_obj.mean()
        if agg == "std":
            return rolling_obj.std()
        if agg == "max":
            return rolling_obj.max()
        if agg == "min":
            return rolling_obj.min()
        if agg == "median":
            return rolling_obj.median()
        if agg == "sum":
            return rolling_obj.sum()
        raise ValueError(f"Unsupported aggregation {agg}")

    def _merge_external_features(self, df: pd.DataFrame, symbol: Optional[str]) -> pd.DataFrame:
        if not self.settings.external_sources:
            return df

        df = df.copy()
        for source in self.settings.external_sources:
            ext_df = self._load_external(source)
            if ext_df is None or ext_df.empty:
                continue

            filtered = ext_df.copy()
            if source.join_on in filtered.columns:
                filtered[source.join_on] = self._normalise_trade_date(filtered[source.join_on])

            if source.ts_code_column and symbol is not None and source.ts_code_column in filtered.columns:
                filtered = filtered[filtered[source.ts_code_column] == symbol]

            if filtered.empty:
                continue

            merge_key_left = source.join_on if source.join_on in df.columns else "trade_date"
            merge_key_right = source.join_on

            if merge_key_left in df.columns:
                df[merge_key_left] = self._normalise_trade_date(df[merge_key_left])

            rename_map = {
                col: f"{source.name}_{col}"
                for col in filtered.columns
                if col not in {merge_key_right, source.ts_code_column}
            }
            filtered = filtered.rename(columns=rename_map)
            filtered = filtered.drop_duplicates(subset=[merge_key_right])

            df = pd.merge(df, filtered, how="left", left_on=merge_key_left, right_on=merge_key_right)
            if merge_key_right != merge_key_left:
                df = df.drop(columns=[merge_key_right], errors="ignore")
            if source.ts_code_column and source.ts_code_column in df.columns:
                df = df.drop(columns=[source.ts_code_column], errors="ignore")

            feature_cols = list(rename_map.values())
            if feature_cols:
                df[feature_cols] = df[feature_cols].astype(float, errors="ignore")
                if source.forward_fill:
                    df[feature_cols] = df[feature_cols].ffill()

        return df

    def _load_external(self, config: ExternalFeatureConfig) -> Optional[pd.DataFrame]:
        cached = self.cache.get(config.path)
        if cached is not None:
            return cached

        path = Path(config.path)
        if not path.exists():
            return None

        try:
            frame = pd.read_csv(path)
        except Exception:
            return None

        frame = frame.replace([np.inf, -np.inf], np.nan)
        self.cache.set(config.path, frame)
        return frame

    @staticmethod
    def _normalise_trade_date(series: pd.Series) -> pd.Series:
        """将 trade_date 转换为统一的字符串格式 YYYYMMDD，兼容 int/str/datetime。"""

        if series.empty:
            return series

        if pd.api.types.is_datetime64_any_dtype(series):
            normalised = series.dt.strftime("%Y%m%d")
        else:
            normalised = pd.to_datetime(series.astype(str), errors="coerce").dt.strftime("%Y%m%d")

        # 若全部转换失败，退回原始字符串表示
        if normalised.isna().all():
            return series.astype(str)
        return normalised.ffill().bfill()


__all__ = ["FeatureEngineer", "FeatureEngineerCache"]
