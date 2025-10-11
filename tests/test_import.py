def test_import_package():
    import importlib
    pkg = importlib.import_module('stock_prediction')
    assert pkg is not None
