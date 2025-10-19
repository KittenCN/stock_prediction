"""
diagnostics.py | 预测结果诊断工具
提供分布一致性检查及统一的振幅/偏差告警。
"""

from __future__ import annotations

import math
from typing import Dict, MutableMapping

import pandas as pd

from .metrics import metrics_report, distribution_report

STD_RATIO_WARNING: float = 0.8
BIAS_WARNING: float = 0.5
STD_FLOOR: float = 1e-6


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
    """计算单个特征的指标，并在发现异常时输出告警。"""
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


__all__ = [
    "evaluate_feature_metrics",
    "STD_RATIO_WARNING",
    "BIAS_WARNING",
    "STD_FLOOR",
]
