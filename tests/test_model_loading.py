#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试脚本：验证模型参数和归一化参数的保存与加载
用于确保测试模式下能够正确恢复训练时的配置
"""

import os
import json
import torch
from pathlib import Path

def test_model_files():
    """检查模型文件是否完整"""
    models_dir = Path("models")
    
    if not models_dir.exists():
        print("[INFO] models/ 目录不存在，跳过测试")
        return
    
    # 查找所有模型文件
    model_files = list(models_dir.rglob("*_Model.pkl")) + list(models_dir.rglob("*_Model_best.pkl"))
    
    print(f"[INFO] 找到 {len(model_files)} 个模型文件")
    
    for model_file in model_files[:5]:  # 只检查前5个
        print(f"\n[CHECK] {model_file.name}")
        
        # 检查是否有对应的参数文件
        args_file = str(model_file).replace("_Model.pkl", "_Model_args.json").replace("_Model_best.pkl", "_Model_best_args.json")
        norm_file = str(model_file).replace("_Model.pkl", "_norm_params.json").replace("_Model_best.pkl", "_norm_params_best.json")
        
        has_args = os.path.exists(args_file)
        has_norm = os.path.exists(norm_file)
        
        print(f"  ├─ 参数文件: {'✓' if has_args else '✗'} {Path(args_file).name if has_args else '缺失'}")
        print(f"  └─ 归一化文件: {'✓' if has_norm else '✗'} {Path(norm_file).name if has_norm else '缺失'}")
        
        # 如果存在，检查内容
        if has_args:
            try:
                with open(args_file, 'r', encoding='utf-8') as f:
                    args = json.load(f)
                print(f"     模型参数: {list(args.keys())}")
            except Exception as e:
                print(f"     [ERROR] 读取参数文件失败: {e}")
        
        if has_norm:
            try:
                with open(norm_file, 'r', encoding='utf-8') as f:
                    norm = json.load(f)
                print(f"     归一化参数: mean_list长度={len(norm.get('mean_list', []))}, std_list长度={len(norm.get('std_list', []))}")
            except Exception as e:
                print(f"     [ERROR] 读取归一化文件失败: {e}")

def test_save_and_load():
    """测试保存和加载功能"""
    print("\n[TEST] 测试参数保存与加载...")
    
    # 创建临时测试文件
    test_dir = Path("output/test_temp")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # 测试数据
    test_args = {"input_dim": 30, "hidden_dim": 128, "output_dim": 4}
    test_norm = {
        "mean_list": [0.1, 0.2, 0.3],
        "std_list": [1.0, 1.5, 2.0],
        "show_list": [1, 1, 1],
        "name_list": ["open", "close", "high"]
    }
    
    args_file = test_dir / "test_args.json"
    norm_file = test_dir / "test_norm.json"
    
    # 保存
    try:
        with open(args_file, 'w', encoding='utf-8') as f:
            json.dump(test_args, f, ensure_ascii=False, indent=2)
        with open(norm_file, 'w', encoding='utf-8') as f:
            json.dump(test_norm, f, ensure_ascii=False, indent=2)
        print("  ✓ 保存成功")
    except Exception as e:
        print(f"  ✗ 保存失败: {e}")
        return
    
    # 加载
    try:
        with open(args_file, 'r', encoding='utf-8') as f:
            loaded_args = json.load(f)
        with open(norm_file, 'r', encoding='utf-8') as f:
            loaded_norm = json.load(f)
        
        assert loaded_args == test_args, "模型参数不匹配"
        assert loaded_norm == test_norm, "归一化参数不匹配"
        print("  ✓ 加载验证成功")
    except Exception as e:
        print(f"  ✗ 加载验证失败: {e}")
    finally:
        # 清理
        if args_file.exists():
            args_file.unlink()
        if norm_file.exists():
            norm_file.unlink()
        if test_dir.exists():
            test_dir.rmdir()

if __name__ == "__main__":
    print("=" * 60)
    print("模型文件完整性检查")
    print("=" * 60)
    
    test_model_files()
    test_save_and_load()
    
    print("\n" + "=" * 60)
    print("检查完成")
    print("=" * 60)
