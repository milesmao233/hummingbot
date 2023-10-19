from datetime import timedelta

import numpy as np
import pandas as pd


def closest_hour_interval(dt, interval=4):
    # 取余数
    remainder = dt.hour % interval
    # 如果余数为0，则返回时间本身
    if remainder == 0:
        return dt.replace(minute=0, second=0, microsecond=0)
    # 否则返回上一个4小时整点
    else:
        hours_to_subtract = timedelta(hours=remainder)
        closest_time = dt - hours_to_subtract
        return closest_time.replace(minute=0, second=0, microsecond=0)


def calculate_recursive_ema(df, price_name, span, ema_name='ema'):
    alpha = 2 / (span + 1)
    # df = df.copy()

    # Initialize the first EMA value
    df = df.copy()

    # Start the recursive EMA calculation from the second row
    for i in range(1, len(df)):
        previous_ema = df[ema_name].iloc[i - 1]
        current_price = df[price_name].iloc[i]
        current_ema = (1 - alpha) * previous_ema + alpha * current_price
        df.at[df.index[i], ema_name] = current_ema

    return df


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


def calculate_remaining_ema(df, price_name, windows, interval, ema_name='ema'):
    last_time = df.iloc[-1]['candle_begin_time']
    closest_time = closest_hour_interval(last_time, interval)
    # 找到 closest_time 是 symbol_df 的第几行
    closest_time_index = df[df['candle_begin_time'] == closest_time].index[0]
    # 获取 closest_time_index 之后的数据
    symbol_df_after = df.iloc[closest_time_index:].copy()
    symbol_df_after.reset_index(drop=True, inplace=True)
    symbol_df_after = calculate_recursive_ema(symbol_df_after, price_name, windows, ema_name)

    # # 删除 closest_time_index 之后的数据
    df_before = df.drop(df.index[closest_time_index:], inplace=False)
    #
    # # 合并 df 和 symbol_df_after, 以 candle_begin_time 为 key
    df = pd.concat([df_before, symbol_df_after], ignore_index=True)
    df.drop_duplicates(subset=['candle_begin_time'], keep='last', inplace=True)
    df.sort_values('candle_begin_time', inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def calc_period_macd(df, price_name, params_list, period):
    short_windows = int(params_list[0])
    long_windows = int(params_list[1])
    dea_windows = int(params_list[2])

    time_period_min = int(period / 60)
    time_interval = f'{time_period_min}H'

    # 计算 EMA_Short 和 EMA_Long
    ema_short_name = f'ema{short_windows}_{time_period_min}_{price_name}'
    ema_long_name = f'ema{long_windows}_{time_period_min}_{price_name}'

    df = calculate_ema(df, price_name, short_windows, time_interval, ema_short_name)
    df = calculate_remaining_ema(df, price_name, short_windows, time_period_min, ema_short_name)

    df = calculate_ema(df, price_name, long_windows, time_interval, ema_long_name)
    df = calculate_remaining_ema(df, price_name, long_windows, time_period_min, ema_long_name)

    # 计算 DIF
    diff_name = f'dif_{time_period_min}_{price_name}'
    df[diff_name] = df[ema_short_name] - df[ema_long_name]

    # 计算 DEA
    dea_name = f'dea_{time_period_min}_{price_name}'
    df = calculate_ema(df, diff_name, dea_windows, time_interval, dea_name)
    df = calculate_remaining_ema(df, diff_name, dea_windows, time_period_min, dea_name)

    return df
