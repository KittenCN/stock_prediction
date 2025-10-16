import math
from stock_prediction.trainer import EarlyStopping, EarlyStoppingConfig


def test_early_stopping_improves_and_stops():
    cfg = EarlyStoppingConfig(patience=2, min_delta=0.1, mode="min")
    es = EarlyStopping(cfg)

    # Initial set
    assert es.step(1.0) is False
    assert math.isclose(es.best, 1.0)

    # Improves by > min_delta
    assert es.step(0.8) is False
    assert math.isclose(es.best, 0.8)

    # No improvement 1
    assert es.step(0.79) is False  # improvement smaller than min_delta
    # No improvement 2 -> should stop
    assert es.step(0.78) is True