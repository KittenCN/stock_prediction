"""
metrics.py | 评估指标采集模块
支持 RMSE、MAPE、分位覆盖率、VaR、CVaR 以及分布差异诊断。
"""

from __future__ import annotations

import numpy as np


def rmse(y_true, y_pred):
    """均方根误差 Root Mean Squared Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mape(y_true, y_pred):
    """平均绝对百分比误差 Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def quantile_coverage(y_true, y_pred_q, q):
    """分位覆盖率：y_pred_q 为分位预测值，q 为分位数(如 0.05、0.5、0.95)"""
    y_true, y_pred_q = np.array(y_true), np.array(y_pred_q)
    return np.mean(y_true <= y_pred_q) if q < 0.5 else np.mean(y_true >= y_pred_q)


def var(y_true, alpha=0.05):
    """VaR: Value at Risk, 置信水平 alpha 下的损失分位数"""
    y_true = np.array(y_true)
    return np.percentile(y_true, alpha * 100)


def cvar(y_true, alpha=0.05):
    """CVaR: Conditional Value at Risk, 超过 VaR 部分的均值"""
    y_true = np.array(y_true)
    v = var(y_true, alpha)
    return y_true[y_true <= v].mean() if alpha < 0.5 else y_true[y_true >= v].mean()


def metrics_report(y_true, y_pred, y_pred_qs=None, quantiles=(0.05, 0.5, 0.95)):
    """综合指标报告，支持分位预测输出"""
    report = {
        "rmse": float(rmse(y_true, y_pred)),
        "mape": float(mape(y_true, y_pred)),
        "var_5": float(var(y_true, 0.05)),
        "cvar_5": float(cvar(y_true, 0.05)),
    }
    if y_pred_qs is not None:
        for i, q in enumerate(quantiles):
            report[f"quantile_coverage_{q}"] = float(quantile_coverage(y_true, y_pred_qs[i], q))
    return report


def _safe_std(array: np.ndarray) -> float:
    """返回稳定的标准差，自动剔除 NaN/Inf。"""
    if array.size == 0:
        return float("nan")
    cleaned = np.nan_to_num(array.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.std(cleaned))


def distribution_report(y_true, y_pred):
    """比较预测与真实的分布形态，便于捕获振幅塌缩或均值偏移。"""
    true_arr = np.array(y_true, dtype=float)
    pred_arr = np.array(y_pred, dtype=float)
    if true_arr.size == 0 or pred_arr.size == 0:
        return {
            "bias": float("nan"),
            "true_mean": float("nan"),
            "pred_mean": float("nan"),
            "true_std": float("nan"),
            "pred_std": float("nan"),
            "std_ratio": float("nan"),
            "corr": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
        }

    true_mean = float(np.mean(true_arr))
    pred_mean = float(np.mean(pred_arr))
    bias = pred_mean - true_mean
    true_std = _safe_std(true_arr)
    pred_std = _safe_std(pred_arr)
    std_ratio = float(pred_std / true_std) if true_std > 1e-8 else float("nan")
    mae_value = float(np.mean(np.abs(pred_arr - true_arr)))
    rmse_value = float(np.sqrt(np.mean((pred_arr - true_arr) ** 2)))
    if true_std <= 1e-8 or pred_std <= 1e-8:
        corr = 0.0
    else:
        corr_matrix = np.corrcoef(true_arr, pred_arr)
        corr = float(corr_matrix[0, 1]) if corr_matrix.size == 4 else 0.0

    return {
        "bias": float(bias),
        "true_mean": true_mean,
        "pred_mean": pred_mean,
        "true_std": true_std,
        "pred_std": pred_std,
        "std_ratio": std_ratio,
        "corr": corr,
        "mae": mae_value,
        "rmse": rmse_value,
    }


__all__ = [
    "rmse",
    "mape",
    "quantile_coverage",
    "var",
    "cvar",
    "metrics_report",
    "distribution_report",
]
