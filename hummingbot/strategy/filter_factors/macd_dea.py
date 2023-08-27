import numpy as np

from hummingbot.strategy.filter_factors.utils.calc_period_macd import calc_period_macd


def signal(*args):
    # VolumeStd
    df = args[0]
    column_name = args[1]
    macd_period = args[2][0]
    macd_params = args[2][1]
    factor_name = args[3]

    df = calc_period_macd(df, column_name, macd_params, macd_period)

    time_period_min = macd_period / 60
    dea_name = f'dea_{time_period_min}_{column_name}'

    df[factor_name] = df[dea_name]

    return df
