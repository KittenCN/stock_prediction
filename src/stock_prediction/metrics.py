"""
metrics.py | 评估指标采集模块
支持 RMSE、MAPE、分位覆盖率、VaR、CVaR 等常用金融回测与回归指标。
"""
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
    """分位覆盖率: y_pred_q 为分位预测值，q 为分位数(如0.05,0.5,0.95)"""
    y_true, y_pred_q = np.array(y_true), np.array(y_pred_q)
    return np.mean(y_true <= y_pred_q) if q < 0.5 else np.mean(y_true >= y_pred_q)


def var(y_true, alpha=0.05):
    """VaR: Value at Risk, 置信度 alpha 下的损失分位数"""
    y_true = np.array(y_true)
    return np.percentile(y_true, alpha * 100)


def cvar(y_true, alpha=0.05):
    """CVaR: Conditional Value at Risk, 超过 VaR 部分的均值"""
    y_true = np.array(y_true)
    v = var(y_true, alpha)
    return y_true[y_true <= v].mean() if alpha < 0.5 else y_true[y_true >= v].mean()


def metrics_report(y_true, y_pred, y_pred_qs=None, quantiles=(0.05,0.5,0.95)):
    """综合指标报告，支持分位预测输入"""
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
