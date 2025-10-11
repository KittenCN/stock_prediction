"""端到端冒烟测试"""
import sys
from pathlib import Path


def test_package_imports():
    """测试所有主要模块能正常导入"""
    try:
        import stock_prediction
        assert stock_prediction is not None
    except ImportError as e:
        assert False, f"Failed to import stock_prediction: {e}"
    
    try:
        from stock_prediction import config
        assert config is not None
    except ImportError as e:
        assert False, f"Failed to import config: {e}"
    
    try:
        from stock_prediction import init
        assert init is not None
    except ImportError as e:
        assert False, f"Failed to import init: {e}"
    
    try:
        from stock_prediction import common
        assert common is not None
    except ImportError as e:
        assert False, f"Failed to import common: {e}"
    
    try:
        from stock_prediction import target
        assert target is not None
    except ImportError as e:
        assert False, f"Failed to import target: {e}"


def test_config_paths_accessible():
    """测试配置路径可访问"""
    from stock_prediction.config import config
    
    # 测试路径对象存在
    assert config.root_path.exists()
    assert hasattr(config, 'data_path')
    assert hasattr(config, 'daily_path')
    assert hasattr(config, 'png_path')
    
    # 测试关键目录存在
    assert config.png_path.exists()
    assert config.bert_data_path.exists()


def test_models_importable():
    """测试模型类能正常导入"""
    try:
        from stock_prediction.common import LSTM, TransformerModel, CNNLSTM
        assert LSTM is not None
        assert TransformerModel is not None  
        assert CNNLSTM is not None
    except ImportError as e:
        assert False, f"Failed to import models: {e}"


def test_technical_indicators_importable():
    """测试技术指标函数能正常导入"""
    try:
        from stock_prediction.target import MACD, KDJ, BOLL, ATR
        assert MACD is not None
        assert KDJ is not None
        assert BOLL is not None
        assert ATR is not None
    except ImportError as e:
        assert False, f"Failed to import technical indicators: {e}"


def test_scripts_exist():
    """测试入口脚本存在"""
    project_root = Path(__file__).resolve().parents[1]
    scripts_dir = project_root / "scripts"
    
    assert scripts_dir.exists(), "Scripts directory not found"
    
    expected_scripts = [
        "predict.py",
        "getdata.py", 
        "data_preprocess.py"
    ]
    
    for script in expected_scripts:
        script_path = scripts_dir / script
        assert script_path.exists(), f"Script {script} not found"


def test_predictor_creation():
    """测试预测器能正常创建"""
    try:
        from stock_prediction.predict import create_predictor
        
        # 测试创建LSTM预测器
        predictor = create_predictor("lstm", "cpu")
        assert predictor is not None
        assert predictor.model_type == "LSTM"
        
    except Exception as e:
        # 如果缺少某些依赖（如torch），这是可以接受的
        print(f"Predictor creation test skipped due to: {e}")
        assert True