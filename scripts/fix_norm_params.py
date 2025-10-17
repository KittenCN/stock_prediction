#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
工具脚本：修复空的归一化参数文件
从PKL文件或CSV文件中重新计算归一化参数
"""

import json
import numpy as np
import pandas as pd
import dill
import queue
from pathlib import Path
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stock_prediction.init import *
from stock_prediction.common import normalize_date_column

def fix_normalization_params_from_pkl(pkl_path="pkl_handle/train.pkl"):
    """从PKL文件重新计算归一化参数"""
    print(f"[INFO] 从 {pkl_path} 加载数据...")
    
    try:
        with open(pkl_path, 'rb') as f:
            data_queue = dill.load(f)
        
        # 确保队列兼容性
        if isinstance(data_queue, queue.Queue) and not hasattr(data_queue, "is_shutdown"):
            data_queue.is_shutdown = False
        
        # 收集所有数据
        all_data = []
        while not data_queue.empty():
            try:
                item = data_queue.get_nowait()
                all_data.append(item)
            except queue.Empty:
                break
        
        print(f"[INFO] 加载了 {len(all_data)} 个数据项")
        
        if not all_data:
            print("[ERROR] PKL文件为空")
            return None, None
        
        # 合并所有数据并计算归一化参数
        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data = normalize_date_column(combined_data)
        feature_data = combined_data.drop(columns=['ts_code', 'Date'], errors='ignore')
        feature_data = feature_data.fillna(feature_data.median(numeric_only=True))
        
        data_array = feature_data.values
        
        # 计算归一化参数
        mean_vals = []
        std_vals = []
        
        for i in range(data_array.shape[1]):
            col_data = data_array[:, i]
            col_data = np.nan_to_num(col_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            mean_val = np.mean(col_data)
            std_val = np.std(col_data)
            
            if np.isnan(mean_val) or np.isinf(mean_val):
                mean_val = 0.0
            if np.isnan(std_val) or np.isinf(std_val) or std_val < 1e-8:
                std_val = 1.0
            
            mean_vals.append(float(mean_val))
            std_vals.append(float(std_val))
        
        print(f"[INFO] 计算完成：{len(mean_vals)} 个特征")
        return mean_vals, std_vals
        
    except Exception as e:
        print(f"[ERROR] 处理PKL文件失败: {e}")
        return None, None

def update_norm_params_files(models_dir="models", mean_vals=None, std_vals=None):
    """更新所有空的归一化参数文件"""
    models_path = Path(models_dir)
    
    if not models_path.exists():
        print(f"[ERROR] 目录不存在: {models_dir}")
        return
    
    # 查找所有归一化参数文件
    norm_files = list(models_path.rglob("*norm_params*.json"))
    
    print(f"\n[INFO] 找到 {len(norm_files)} 个归一化参数文件")
    
    updated_count = 0
    for norm_file in norm_files:
        try:
            with open(norm_file, 'r', encoding='utf-8') as f:
                params = json.load(f)
            
            # 检查是否为空
            if not params.get('mean_list') or not params.get('std_list'):
                print(f"\n[FIX] 修复空文件: {norm_file.name}")
                
                # 更新参数
                params['mean_list'] = mean_vals
                params['std_list'] = std_vals
                
                # 保存回文件
                with open(norm_file, 'w', encoding='utf-8') as f:
                    json.dump(params, f, ensure_ascii=False, indent=2)
                
                updated_count += 1
                print(f"      ✓ 已更新：mean_list和std_list各{len(mean_vals)}个值")
            else:
                print(f"\n[SKIP] 文件已有数据: {norm_file.name}")
                print(f"       mean_list长度: {len(params['mean_list'])}")
                print(f"       std_list长度: {len(params['std_list'])}")
        
        except Exception as e:
            print(f"\n[ERROR] 处理文件失败 {norm_file.name}: {e}")
    
    print(f"\n[DONE] 共更新了 {updated_count} 个文件")

def main():
    print("=" * 70)
    print("归一化参数文件修复工具")
    print("=" * 70)
    
    # 从PKL文件计算归一化参数
    mean_vals, std_vals = fix_normalization_params_from_pkl()
    
    if mean_vals is None or std_vals is None:
        print("\n[ERROR] 无法计算归一化参数，退出")
        return 1
    
    # 更新所有空的归一化参数文件
    update_norm_params_files(mean_vals=mean_vals, std_vals=std_vals)
    
    print("\n" + "=" * 70)
    print("修复完成！")
    print("=" * 70)
    print("\n提示：现在可以运行测试模式:")
    print("  python scripts\\train.py --model Hybrid --mode test --test_code 000019")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
