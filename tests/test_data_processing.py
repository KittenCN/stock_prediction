"""测试数据处理功能"""
import pandas as pd
import numpy as np
from stock_prediction.common import is_number, data_replace, cmp_append


def test_is_number():
    """测试数字判断函数"""
    # 测试有效数字
    assert is_number("123") == True
    assert is_number("123.45") == True
    assert is_number("-123.45") == True
    assert is_number("+123.45") == True
    assert is_number("0") == True
    assert is_number("0.0") == True
    
    # 测试无效输入
    assert is_number("abc") == False
    assert is_number("") == False
    assert is_number("12.34.56") == False


def test_data_replace():
    """测试数据替换函数"""
    # 测试正常情况
    result = data_replace("123.456789")
    assert result == 123.45
    
    result = data_replace(123.456789)
    assert result == 123.45
    
    # 测试边界情况
    result = data_replace("1.2")
    assert result == 1.2


def test_cmp_append():
    """测试数组补齐函数"""
    # 测试需要补齐的情况
    short_list = [1, 2, 3]
    long_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
    
    result = cmp_append(short_list, long_df)
    assert len(result) == len(long_df)
    assert result == [1, 2, 3, 0, 0]
    
    # 测试不需要补齐的情况
    equal_list = [1, 2, 3, 4, 5]
    result = cmp_append(equal_list, long_df)
    assert result == equal_list
    
    # 测试长度已经够的情况
    longer_list = [1, 2, 3, 4, 5, 6, 7]
    result = cmp_append(longer_list, long_df)
    assert result == longer_list


def test_stock_data_basic():
    """测试股票数据基本功能"""
    # 创建测试数据
    test_data = {
        'ts_code': ['000001.SZ'] * 10,
        'trade_date': [f'2023010{i}' for i in range(1, 11)],
        'open': [10 + i for i in range(10)],
        'high': [11 + i for i in range(10)],
        'low': [9 + i for i in range(10)],
        'close': [10.5 + i for i in range(10)],
        'vol': [1000 + i * 100 for i in range(10)],
        'amount': [10000 + i * 1000 for i in range(10)]
    }
    
    df = pd.DataFrame(test_data)
    
    # 基本验证
    assert not df.empty
    assert len(df) == 10
    assert 'ts_code' in df.columns
    assert 'close' in df.columns


def test_dataframe_operations():
    """测试DataFrame操作"""
    # 创建测试DataFrame
    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [1.1, 2.2, 3.3, np.nan, 5.5],
        'C': ['a', 'b', 'c', 'd', 'e']
    })
    
    # 测试fillna操作
    df_filled = df.fillna(df.median(numeric_only=True))
    
    # 验证数值列的NaN被填充
    assert not df_filled['A'].isna().any()
    assert not df_filled['B'].isna().any()
    
    # 验证非数值列保持不变
    assert df_filled['C'].isna().sum() == df['C'].isna().sum()