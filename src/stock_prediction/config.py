"""
统一配置管理模块
集中管理所有路径、参数和配置
"""
import os
from pathlib import Path


class Config:
    """项目配置类"""
    
    def __init__(self, root_path=None):
        # 根路径设置
        if root_path is None:
            self.root_path = Path(__file__).resolve().parents[2]  # 从 src/stock_prediction 回到项目根目录
        else:
            self.root_path = Path(root_path)
            
        # 确保根路径存在
        self.root_path.mkdir(parents=True, exist_ok=True)
        
        # 数据路径
        self.data_path = self.root_path / "stock_data"
        self.daily_path = self.root_path / "stock_daily"
        self.handle_path = self.root_path / "stock_handle"
        self.pkl_path = self.root_path / "pkl_handle"
        self.bert_data_path = self.root_path / "bert_data"
        
        # 输出路径
        self.png_path = self.root_path / "png"
        self.output_path = self.root_path / "output"
        
        # 特定文件路径
        self.train_path = self.handle_path / "stock_train.csv"
        self.test_path = self.handle_path / "stock_test.csv"
        self.train_pkl_path = self.pkl_path / "train.pkl"
        
        # 模型保存路径
        self.models_path = self.root_path / "models"
        self.lstm_path = self.models_path / "LSTM"
        self.transformer_path = self.models_path / "TRANSFORMER" 
        self.cnnlstm_path = self.models_path / "CNNLSTM"
        
        # 创建必要的目录
        self._create_directories()
        
    def _create_directories(self):
        """创建必要的目录结构"""
        directories = [
            self.data_path,
            self.daily_path,
            self.handle_path,
            self.pkl_path,
            self.bert_data_path,
            self.png_path,
            self.output_path,
            self.models_path,
            self.png_path / "train_loss",
            self.png_path / "predict", 
            self.png_path / "test",
            self.bert_data_path / "model",
            self.bert_data_path / "data",
            self.data_path / "log"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    def get_model_path(self, model_type, symbol="Generic.Data"):
        """获取特定模型的保存路径"""
        symbol_clean = symbol.replace(".", "")
        model_dir = self.models_path / symbol_clean / model_type.upper()
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir / model_type.upper()
        
    def __str__(self):
        return f"Config(root_path={self.root_path})"


# 全局配置实例
config = Config()

# 为了向后兼容，提供字符串路径
root_path = str(config.root_path)
train_path = str(config.train_path)
test_path = str(config.test_path)
train_pkl_path = str(config.train_pkl_path)
png_path = str(config.png_path)
daily_path = str(config.daily_path)
handle_path = str(config.handle_path)
pkl_path = str(config.pkl_path)
bert_data_path = str(config.bert_data_path)
data_path = str(config.data_path)
lstm_path = str(config.lstm_path)
transformer_path = str(config.transformer_path)
cnnlstm_path = str(config.cnnlstm_path)