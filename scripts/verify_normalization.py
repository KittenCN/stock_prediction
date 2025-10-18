#!/usr/bin/env python
# coding: utf-8
"""
Normalization Consistency Verification Script

Verifies that data normalization is consistent between training and test modes,
and optionally compares against persisted *_norm_params*.json entries.

Usage:
    python scripts/verify_normalization.py --stock_code 000019 --norm_file models/GenericData/HYBRID/HYBRID_out4_time5_norm_params.json
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# 添加项目根目录到路径
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from stock_prediction.common import (
    Stock_Data,
    mean_list,
    std_list,
    test_mean_list,
    test_std_list,
    canonical_symbol,
)


def _load_persisted_norm(norm_file: Path, symbol_key: str):
    if not norm_file.exists():
        print(f"[WARN] Norm file not found: {norm_file}")
        return None
    try:
        with open(norm_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        print(f"[WARN] Failed to load norm file {norm_file}: {exc}")
        return None

    per_symbol = payload.get("per_symbol") or {}
    stats = per_symbol.get(symbol_key)
    if stats is None:
        print(f"[WARN] No per-symbol stats for {symbol_key} in {norm_file.name}; falling back to global entries.")
        stats = {
            "mean_list": payload.get("mean_list", []),
            "std_list": payload.get("std_list", []),
        }
    return stats


def verify_normalization(stock_code: str, csv_path: str = None, norm_file: str = None) -> bool:
    symbol_key = canonical_symbol(stock_code)
    if symbol_key is None:
        raise ValueError(f"Invalid stock code: {stock_code}")

    print("=" * 60)
    print(f"Normalization Consistency Verification - Stock Code: {symbol_key}")
    print("=" * 60 + "\n")

    csv_path = Path(csv_path) if csv_path else root_dir / "stock_daily" / f"{symbol_key}.csv"
    if not csv_path.exists():
        print(f"[ERROR] File not found: {csv_path}")
        return False

    print(f"📂 Loading data: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        print("[ERROR] Data is empty")
        return False
    print(f"✅ Data loaded successfully, {len(df)} rows\n")

    # Training mode dataset
    print("🔄 Creating training mode dataset (mode=0)...")
    train_dataset = Stock_Data(mode=0, dataFrame=df, label_num=4, norm_symbol=symbol_key)
    if len(train_dataset) == 0:
        print("[ERROR] Training dataset is empty")
        return False
    train_input = train_dataset[0][0].numpy()
    train_mean = np.array(mean_list[:train_input.shape[-1]])
    train_std = np.array(std_list[:train_input.shape[-1]])
    print(f"✅ Training dataset ready: samples={len(train_dataset)}, input_shape={train_input.shape}")
    print(f"   Normalization parameters: mean={len(train_mean)}, std={len(train_std)}\n")

    # Test mode dataset
    print("🔄 Creating test mode dataset (mode=1)...")
    test_dataset = Stock_Data(mode=1, dataFrame=df, label_num=4, norm_symbol=symbol_key)
    if len(test_dataset) == 0:
        print("[ERROR] Test dataset is empty")
        return False
    test_input = test_dataset[0][0].numpy()
    test_mean = np.array(test_mean_list[:test_input.shape[-1]])
    test_std = np.array(test_std_list[:test_input.shape[-1]])
    print(f"✅ Test dataset ready: samples={len(test_dataset)}, input_shape={test_input.shape}")
    print(f"   Normalization parameters: mean={len(test_mean)}, std={len(test_std)}\n")

    # Compare train/test statistics
    mean_diff = np.max(np.abs(train_mean - test_mean))
    std_diff = np.max(np.abs(train_std - test_std))
    input_diff = np.max(np.abs(train_input - test_input))
    input_mean_diff = np.mean(np.abs(train_input - test_input))

    print("=" * 60)
    print("📊 Normalization Parameter Comparison")
    print("=" * 60 + "\n")
    print(f"Mean max difference: {mean_diff:.6e}")
    print(f"Std max difference: {std_diff:.6e}\n")

    print("First 5 feature comparison (Mean):")
    print(f"{'Idx':<5}{'Train':>14}{'Test':>14}{'Diff':>14}")
    for i in range(min(5, len(train_mean))):
        diff = float(abs(train_mean[i] - test_mean[i]))
        print(f"{i:<5}{train_mean[i]:>14.6f}{test_mean[i]:>14.6f}{diff:>14.6e}")

    print("\nFirst 5 feature comparison (Std):")
    print(f"{'Idx':<5}{'Train':>14}{'Test':>14}{'Diff':>14}")
    for i in range(min(5, len(train_std))):
        diff = float(abs(train_std[i] - test_std[i]))
        print(f"{i:<5}{train_std[i]:>14.6f}{test_std[i]:>14.6f}{diff:>14.6e}")

    print("\n" + "=" * 60)
    print("📊 Input Data Comparison")
    print("=" * 60 + "\n")
    print(f"Input max difference: {input_diff:.6e}")
    print(f"Input average difference: {input_mean_diff:.6e}\n")

    passed = True
    threshold = 1e-6
    if mean_diff >= threshold:
        print(f"[WARN] Mean parameters inconsistent (diff={mean_diff:.6e} >= {threshold})")
        passed = False
    else:
        print(f"[OK] Mean parameters consistent (diff < {threshold})")
    if std_diff >= threshold:
        print(f"[WARN] Std parameters inconsistent (diff={std_diff:.6e} >= {threshold})")
        passed = False
    else:
        print(f"[OK] Std parameters consistent (diff < {threshold})")
    if input_diff >= threshold:
        print(f"[WARN] Input tensors inconsistent (diff={input_diff:.6e} >= {threshold})")
        passed = False
    else:
        print(f"[OK] Input tensors consistent (diff < {threshold})")

    # Compare with persisted norm params if provided
    if norm_file:
        stats = _load_persisted_norm(Path(norm_file), symbol_key)
        if stats:
            saved_mean = np.asarray(stats.get("mean_list", []), dtype=float)
            saved_std = np.asarray(stats.get("std_list", []), dtype=float)
            if saved_mean.size and len(train_mean) >= saved_mean.size:
                diff_mean_saved = float(np.max(np.abs(saved_mean - train_mean[:saved_mean.size])))
                print(f"Persisted mean diff vs training: {diff_mean_saved:.6e}")
                if diff_mean_saved >= threshold:
                    passed = False
            if saved_std.size and len(train_std) >= saved_std.size:
                diff_std_saved = float(np.max(np.abs(saved_std - train_std[:saved_std.size])))
                print(f"Persisted std diff vs training: {diff_std_saved:.6e}")
                if diff_std_saved >= threshold:
                    passed = False

    print("\n" + "=" * 60)
    if passed:
        print("✅ Verification passed! Normalization is consistent.")
    else:
        print("❌ Verification failed. Please inspect normalization pipeline.")
        print("   Suggestions:")
        print("   1. Ensure enable_symbol_normalization=false (config.yaml).")
        print("   2. Confirm per-symbol stats are persisted in *_norm_params*.json.")
        print("   3. Re-run training to rebuild normalization stats if necessary.")
    print("=" * 60 + "\n")

    return passed


def main():
    parser = argparse.ArgumentParser(description="Verify normalization consistency between training and test modes")
    parser.add_argument(
        "--stock_code",
        "--ts_code",
        dest="stock_code",
        type=str,
        default=None,
        help="Stock code, e.g., 000019 or 000019.SZ",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="Custom CSV path (optional)",
    )
    parser.add_argument(
        "--norm_file",
        type=str,
        default=None,
        help="Path to *_norm_params*.json for comparison (optional)",
    )

    args = parser.parse_args()
    stock_code = args.stock_code or "000001.SZ"

    try:
        passed = verify_normalization(stock_code, args.csv_path, args.norm_file)
        sys.exit(0 if passed else 1)
    except Exception as exc:
        print(f"[ERROR] Verification failed: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
