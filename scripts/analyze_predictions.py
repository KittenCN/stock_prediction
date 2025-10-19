#!/usr/bin/env python
# coding: utf-8
"""analyze_predictions.py | 批量分析 png/test 或 png/predict 下的预测 CSV 文件。

示例：
    python scripts/analyze_predictions.py --folder png/test --std-threshold 0.8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Ensure src/ is importable when script is executed directly
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stock_prediction.metrics import distribution_report, metrics_report


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"Date", "Actual", "Forecast"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} 缺少列: {', '.join(sorted(missing))}")
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def analyze_file(path: Path, std_threshold: float, bias_threshold: float) -> Dict[str, object]:
    df = load_csv(path)
    actual = df["Actual"].astype(float).values
    forecast = df["Forecast"].astype(float).values
    reg_metrics = metrics_report(actual, forecast)
    dist_metrics = distribution_report(actual, forecast)

    flags: List[str] = []
    std_ratio = dist_metrics.get("std_ratio")
    bias = dist_metrics.get("bias")
    if isinstance(std_ratio, float) and std_ratio < std_threshold:
        flags.append(f"振幅偏低(std_ratio={std_ratio:.3f})")
    if isinstance(bias, float) and abs(bias) > bias_threshold:
        flags.append(f"均值偏移(bias={bias:+.3f})")
    if isinstance(dist_metrics.get("pred_std"), float) and dist_metrics["pred_std"] < 1e-6:
        flags.append("预测标准差≈0")

    return {
        "file": path.name,
        "rows": int(len(df)),
        "regression": reg_metrics,
        "distribution": dist_metrics,
        "flags": flags,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="分析预测 CSV，输出 RMSE、偏差、振幅等指标，并标记潜在风险。"
    )
    parser.add_argument("--folder", default="png/test", help="CSV 所在目录，默认 png/test")
    parser.add_argument("--pattern", default="*.csv", help="匹配的文件模式，默认 *.csv")
    parser.add_argument("--std-threshold", type=float, default=0.8, help="振幅比预警阈值，默认 0.8")
    parser.add_argument("--bias-threshold", type=float, default=0.5, help="均值偏移预警阈值，默认 0.5")
    parser.add_argument("--export-json", type=Path, help="可选，输出诊断结果到指定 JSON 文件")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        raise FileNotFoundError(f"目录不存在: {folder}")

    summaries: List[Dict[str, object]] = []
    for csv_path in sorted(folder.glob(args.pattern)):
        try:
            summary = analyze_file(csv_path, args.std_threshold, args.bias_threshold)
            summaries.append(summary)
        except Exception as exc:
            summaries.append({"file": csv_path.name, "error": str(exc), "flags": ["解析失败"]})

    if not summaries:
        print(f"[WARN] 在 {folder} 未找到匹配 {args.pattern} 的 CSV。")
        return

    print(
        f"[INFO] 共分析 {len(summaries)} 个文件（阈值: std_ratio<{args.std_threshold}, |bias|>{args.bias_threshold}）。"
    )
    for item in summaries:
        flags = ", ".join(item.get("flags", [])) or "正常"
        reg = item.get("regression", {})
        dist = item.get("distribution", {})
        rmse = reg.get("rmse", float("nan"))
        std_ratio = dist.get("std_ratio", float("nan"))
        bias = dist.get("bias", float("nan"))
        print(
            f"- {item['file']}: rows={item.get('rows', 'n/a')}, "
            f"RMSE={rmse:.3f} StdRatio={std_ratio:.3f} Bias={bias:.3f} Flags={flags}"
        )

    if args.export_json:
        payload = {
            "folder": str(folder),
            "std_threshold": args.std_threshold,
            "bias_threshold": args.bias_threshold,
            "items": summaries,
        }
        args.export_json.parent.mkdir(parents=True, exist_ok=True)
        args.export_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[INFO] 结果已导出至 {args.export_json}")


if __name__ == "__main__":
    main()
