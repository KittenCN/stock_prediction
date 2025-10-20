"""
diagnostics.py | Prediction diagnostics helpers.
Provides distribution checks, bias warnings, and bias-correction utilities.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Mapping, MutableMapping

import pandas as pd

from .metrics import distribution_report, metrics_report

STD_RATIO_WARNING: float = 0.8
BIAS_WARNING: float = 0.5
STD_FLOOR: float = 1e-6
_BIAS_DIR = Path("output")


def _is_number(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def evaluate_feature_metrics(
    feature_name: str,
    history_series: pd.Series,
    prediction_series: pd.Series,
    regression_bucket: MutableMapping[str, Dict[str, float]],
    distribution_bucket: MutableMapping[str, Dict[str, float]],
) -> None:
    """Compute metrics for a single feature and emit warnings when deviations appear."""
    if history_series is None or prediction_series is None:
        return
    aligned_length = min(len(history_series), len(prediction_series))
    if aligned_length <= 0:
        return
    hist = history_series.iloc[-aligned_length:]
    pred = prediction_series.iloc[-aligned_length:]
    if aligned_length == 0:
        return

    metrics_data = metrics_report(hist.values, pred.values)
    regression_bucket[feature_name] = metrics_data
    dist_data = distribution_report(hist.values, pred.values)
    distribution_bucket[feature_name] = dist_data

    std_ratio = dist_data.get("std_ratio")
    pred_std = dist_data.get("pred_std")
    bias = dist_data.get("bias")

    if _is_number(pred_std) and float(pred_std) <= STD_FLOOR:
        print(f"[WARN] {feature_name} 预测标准差≈0，模型可能只输出常数值。")
    if _is_number(std_ratio) and float(std_ratio) < STD_RATIO_WARNING:
        print(f"[WARN] {feature_name} 振幅偏低（std_ratio={float(std_ratio):.3f}），建议提升波动约束或切换收益目标。")
    if _is_number(bias) and abs(float(bias)) > BIAS_WARNING:
        print(f"[WARN] {feature_name} 预测均值偏移 {float(bias):+.3f}，请检查归一化统计或损失权重。")


def _bias_file(symbol: str, model: str) -> Path:
    sanitized_symbol = symbol.replace("/", "_")
    sanitized_model = model.replace("/", "_")
    return _BIAS_DIR / f"bias_{sanitized_symbol}_{sanitized_model}.json"


def load_bias_corrections(symbol: str, model: str) -> Dict[str, float]:
    """Load previously saved bias corrections for a symbol/model pair."""
    path = _bias_file(symbol, model)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {k: float(v) for k, v in data.items()}
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] Failed to load bias corrections from {path}: {exc}")
    return {}


def save_bias_corrections(
    symbol: str,
    model: str,
    distribution_metrics: Mapping[str, Mapping[str, float]],
    smoothing: float = 0.0,
) -> None:
    """Persist latest bias measurements; optional smoothing keeps some previous correction."""
    smoothing = min(max(smoothing, 0.0), 0.99)
    latest_bias = {
        feature: float(values.get("bias", 0.0))
        for feature, values in distribution_metrics.items()
        if isinstance(values, Mapping)
    }
    if not latest_bias:
        return

    _BIAS_DIR.mkdir(parents=True, exist_ok=True)
    path = _bias_file(symbol, model)
    if path.exists() and smoothing > 0:
        try:
            prev = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            prev = {}
        if isinstance(prev, dict):
            for key, value in prev.items():
                if key in latest_bias:
                    latest_bias[key] = smoothing * float(value) + (1.0 - smoothing) * latest_bias[key]

    path.write_text(json.dumps(latest_bias, ensure_ascii=False, indent=2), encoding="utf-8")


def apply_bias_corrections_to_dataframe(df: pd.DataFrame, corrections: Mapping[str, float]) -> bool:
    """Shift dataframe columns by stored bias values; returns True if any column adjusted."""
    applied = False
    for col, corr in corrections.items():
        if col in df.columns and _is_number(corr) and float(corr) != 0.0:
            df[col] = df[col] - float(corr)
            applied = True
    return applied


__all__ = [
    "evaluate_feature_metrics",
    "STD_RATIO_WARNING",
    "BIAS_WARNING",
    "STD_FLOOR",
    "load_bias_corrections",
    "save_bias_corrections",
    "apply_bias_corrections_to_dataframe",
]
