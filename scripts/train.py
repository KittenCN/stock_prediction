#!/usr/bin/env python
# coding: utf-8
"""
股票预测训练脚本
支持 LSTM、Transformer、CNNLSTM 等模型
"""
import sys
from pathlib import Path

# 添加 src 路径
src_path = Path(__file__).resolve().parents[1] / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from stock_prediction.train import main

if __name__ == '__main__':
    main()
