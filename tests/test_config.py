"""测试配置模块"""
import pytest
from stock_prediction.config import Config, config


def test_config_initialization():
    """测试配置初始化"""
    test_config = Config()
    assert test_config.root_path.exists()
    assert test_config.data_path.name == "stock_data"
    assert test_config.daily_path.name == "stock_daily"


def test_config_directories_created():
    """测试必要目录是否创建"""
    test_config = Config()
    assert test_config.png_path.exists()
    assert (test_config.png_path / "train_loss").exists()
    assert (test_config.png_path / "predict").exists()
    assert test_config.bert_data_path.exists()


def test_model_path_generation():
    """测试模型路径生成"""
    test_config = Config()
    lstm_path = test_config.get_model_path("LSTM", "000001.SZ")
    assert "000001SZ" in str(lstm_path)
    assert "LSTM" in str(lstm_path)


def test_global_config():
    """测试全局配置实例"""
    assert config is not None
    assert config.root_path.exists()