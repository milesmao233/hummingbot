import numpy as np


def signal(*args):
    # VolumeStd
    df = args[0]
    column_name = args[1]
    base_price_period = args[2]
    factor_name = args[3]

    condition = (df['close'].shift(base_price_period) / df['index_capital_curve'].shift(base_price_period)) < (
            df['close'] / df['index_capital_curve'])
    df['base_price'] = np.where(condition, df['close'].shift(base_price_period), df['close'])
    df['base_price_index'] = np.where(
        condition,
        df['index_capital_curve'].shift(base_price_period),
        df['index_capital_curve']
    )
    df[factor_name] = np.where(
        condition,
        df['close'].shift(base_price_period) / df['index_capital_curve'].shift(base_price_period),
        df['close'] / df['index_capital_curve']
    )
    return df
