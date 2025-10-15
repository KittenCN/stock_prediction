def test_import_package():
    import importlib
    pkg = importlib.import_module('stock_prediction')
    assert pkg is not None


def test_import_predict_main():
    """测试能否成功导入 predict 模块的 main 函数"""
    from stock_prediction.predict import main
    assert main is not None
    assert callable(main)
