#!/usr/bin/env python
# coding: utf-8
"""
Normalization Consistency Verification Script

Verifies that data normalization is consistent between training and test modes,
ensuring the system uses the same normalization parameters for training set inference.

Usage:
    python scripts/verify_normalization.py --stock_code 000001.SZ
"""

import argparse
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

from stock_prediction.common import Stock_Data, mean_list, std_list, test_mean_list, test_std_list


def verify_normalization(stock_code: str = "000001.SZ", csv_path: str = None):
    """
    Verify normalization consistency between training and test modes
    
    Args:
        stock_code: Stock code
        csv_path: CSV file path (optional)
    
    Returns:
        bool: Whether verification passed
    """
    print(f"{'='*60}")
    print(f"Normalization Consistency Verification - Stock Code: {stock_code}")
    print(f"{'='*60}\n")
    
    # Load data
    if csv_path is None:
        csv_path = root_dir / "stock_daily" / f"{stock_code.split('.')[0]}.csv"
    
    if not Path(csv_path).exists():
        print(f"❌ Error: File not found {csv_path}")
        return False
    
    print(f"📂 Loading data: {csv_path}")
    df = pd.read_csv(csv_path)
    
    if df.empty:
        print("❌ Error: Data is empty")
        return False
    
    print(f"✅ Data loaded successfully, {len(df)} rows\n")
    
    # Training mode
    print("🔄 Creating training mode dataset (mode=0)...")
    train_dataset = Stock_Data(mode=0, dataFrame=df, label_num=4)
    
    if len(train_dataset) == 0:
        print("❌ Error: Training dataset is empty")
        return False
    
    train_input = train_dataset[0][0].numpy()  # Input of first sample
    train_mean = np.array(mean_list[:train_input.shape[-1]])
    train_std = np.array(std_list[:train_input.shape[-1]])
    
    print(f"✅ Training dataset created successfully, {len(train_dataset)} samples")
    print(f"   Input shape: {train_input.shape}")
    print(f"   Normalization parameters: mean={len(train_mean)}, std={len(train_std)}\n")
    
    # Test mode
    print("🔄 Creating test mode dataset (mode=1)...")
    test_dataset = Stock_Data(mode=1, dataFrame=df, label_num=4)
    
    if len(test_dataset) == 0:
        print("❌ Error: Test dataset is empty")
        return False
    
    test_input = test_dataset[0][0].numpy()
    test_mean = np.array(test_mean_list[:test_input.shape[-1]])
    test_std = np.array(test_std_list[:test_input.shape[-1]])
    
    print(f"✅ Test dataset created successfully, {len(test_dataset)} samples")
    print(f"   Input shape: {test_input.shape}")
    print(f"   Normalization parameters: mean={len(test_mean)}, std={len(test_std)}\n")
    
    # Verify normalization parameters
    print(f"{'='*60}")
    print("📊 Normalization Parameter Comparison")
    print(f"{'='*60}\n")
    
    # Calculate differences
    mean_diff = np.abs(train_mean - test_mean)
    std_diff = np.abs(train_std - test_std)
    
    mean_max_diff = mean_diff.max()
    std_max_diff = std_diff.max()
    
    print(f"Mean maximum difference: {mean_max_diff:.6e}")
    print(f"Std maximum difference: {std_max_diff:.6e}\n")
    
    # Display comparison for first 5 features
    print("Normalization parameter comparison for first 5 features:")
    print(f"{'Feature Idx':<13} {'Train Mean':<15} {'Test Mean':<15} {'Difference':<15}")
    print("-" * 60)
    for i in range(min(5, len(train_mean))):
        diff = abs(train_mean[i] - test_mean[i])
        print(f"{i:<13} {train_mean[i]:<15.6f} {test_mean[i]:<15.6f} {diff:<15.6e}")
    
    print(f"\n{'Feature Idx':<13} {'Train Std':<15} {'Test Std':<15} {'Difference':<15}")
    print("-" * 60)
    for i in range(min(5, len(train_std))):
        diff = abs(train_std[i] - test_std[i])
        print(f"{i:<13} {train_std[i]:<15.6f} {test_std[i]:<15.6f} {diff:<15.6e}")
    
    # Verify input data
    print(f"\n{'='*60}")
    print("📊 Input Data Comparison")
    print(f"{'='*60}\n")
    
    input_diff = np.abs(train_input - test_input)
    input_max_diff = input_diff.max()
    input_mean_diff = input_diff.mean()
    
    print(f"Input data maximum difference: {input_max_diff:.6e}")
    print(f"Input data average difference: {input_mean_diff:.6e}\n")
    
    # Evaluate results
    print(f"{'='*60}")
    print("🎯 Verification Results")
    print(f"{'='*60}\n")
    
    threshold = 1e-6
    passed = True
    
    if mean_max_diff < threshold:
        print(f"✅ Mean parameters consistent (difference < {threshold})")
    else:
        print(f"❌ Mean parameters inconsistent (difference = {mean_max_diff:.6e} >= {threshold})")
        passed = False
    
    if std_max_diff < threshold:
        print(f"✅ Std parameters consistent (difference < {threshold})")
    else:
        print(f"❌ Std parameters inconsistent (difference = {std_max_diff:.6e} >= {threshold})")
        passed = False
    
    if input_max_diff < threshold:
        print(f"✅ Input data consistent (difference < {threshold})")
    else:
        print(f"❌ Input data inconsistent (difference = {input_max_diff:.6e} >= {threshold})")
        passed = False
    
    print(f"\n{'='*60}")
    if passed:
        print("✅ Verification passed! Normalization is consistent between training and test modes.")
    else:
        print("❌ Verification failed! Please check normalization implementation.")
        print("\n💡 Suggestions:")
        print("   1. Ensure enable_symbol_normalization=false in config.yaml")
        print("   2. Ensure contrast_lines function uses mean_list/std_list for denormalization")
        print("   3. Check the normalize_data method in Stock_Data class")
    print(f"{'='*60}\n")
    
    return passed


def main():
    parser = argparse.ArgumentParser(description="Verify normalization consistency between training and test modes")
    parser.add_argument(
        "--stock_code",
        type=str,
        default="000001.SZ",
        help="Stock code, e.g., 000001.SZ"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="CSV file path (optional)"
    )
    
    args = parser.parse_args()
    
    try:
        passed = verify_normalization(args.stock_code, args.csv_path)
        sys.exit(0 if passed else 1)
    except Exception as e:
        print(f"❌ Error during verification: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
