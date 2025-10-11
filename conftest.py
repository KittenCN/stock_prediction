import os
import sys
from pathlib import Path

# 确保 src 在 sys.path 中，便于导入 stock_prediction 包
ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# 可选：强制在 CI 或本地使用特定的 conda 环境名提示
os.environ.setdefault('PROJECT_ENV', 'stock_prediction')
