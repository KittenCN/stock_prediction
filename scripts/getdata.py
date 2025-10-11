#!/usr/bin/env python
# coding: utf-8
"""
数据获取脚本
用于从各种数据源获取股票数据
"""
import sys
from pathlib import Path

# 添加 src 路径
src_path = Path(__file__).resolve().parents[1] / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from stock_prediction.getdata import main

if __name__ == "__main__":
    main()