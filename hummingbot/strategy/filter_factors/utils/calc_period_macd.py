import numpy as np
import pandas as pd


def calculate_ema(df, price_name, span=3, time_interval='1H', ema_name='ema'):
    df.set_index('candle_begin_time', inplace=True)

    hourly_df = df.resample(time_interval).last()

    df[ema_name] = np.nan

    alpha = 2 / (span + 1)

    for i in range(span - 1, len(hourly_df)):
        closes_hourly = hourly_df.iloc[i - span + 1:i][price_name].values

        current_hour_df = df.loc[hourly_df.index[i]:hourly_df.index[min(i + 1, len(hourly_df) - 1)]]

        # Use apply function to replace inner loop
        def _calculate_ema(row):
            closes = np.append(closes_hourly, row[price_name])
            ema = closes[0]
            for close in closes[1:]:
                ema = (1 - alpha) * ema + alpha * close
            return ema

        df.loc[current_hour_df.index, ema_name] = current_hour_df.apply(_calculate_ema, axis=1)

    df.reset_index(inplace=True)
    return df


def calc_period_macd(df, price_name, params_list, period):
    short_windows = int(params_list[0])
    long_windows = int(params_list[1])
    dea_windows = int(params_list[2])

    time_period_min = period / 60
    time_interval = f'{time_period_min}H'

    # 计算 EMA_Short 和 EMA_Long
    ema_short_name = f'ema{short_windows}_{time_period_min}_{price_name}'
    ema_long_name = f'ema{long_windows}_{time_period_min}_{price_name}'

    df = calculate_ema(df, price_name, short_windows, time_interval, ema_short_name)
    df = calculate_ema(df, price_name, long_windows, time_interval, ema_long_name)

    # 计算 DIF
    diff_name = f'dif_{time_period_min}_{price_name}'
    df[diff_name] = df[ema_short_name] - df[ema_long_name]

    # 计算 DEA
    dea_name = f'dea_{time_period_min}_{price_name}'
    df = calculate_ema(df, diff_name, dea_windows, time_interval, dea_name)

    return df
