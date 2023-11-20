import pandas as pd


def signal(*args):
    df = args[0]
    column_name = args[1]
    quote_volume_calc_period = args[2][0]  # 60
    quote_volume_history_period = args[2][1]  # 7days
    quote_volume_x_times = args[2][2]  # 3
    factor_name = args[3]

    df['quote_volume_calc_period'] = df[column_name].rolling(quote_volume_calc_period).sum()

    df_4h = df.resample(rule='4H', on='candle_begin_time').agg({
        column_name: 'sum',
    })

    df_4h['quote_volume_history_period'] = (df_4h['quote_volume'].shift(1)
                                            .rolling(int(quote_volume_history_period * 24 / 4))
                                            .mean()) * (quote_volume_calc_period / (4 * 60))
    # 删除 df_4h 中 quote_volume 列的数据
    df_4h.drop(columns=['quote_volume'], inplace=True)

    # 合并 df 和 df_4h，按照 df 的时间顺序，如果 quote_volume_history_cycle 为 nan，则用上一个 quote_volume_history_cycle 的值填充
    df = pd.merge(df, df_4h, on='candle_begin_time', how='left')
    df['quote_volume_history_period'].fillna(method='ffill', inplace=True)

    df[factor_name] = df['quote_volume_history_period'] * quote_volume_x_times

    del df['quote_volume_calc_period'], df['quote_volume_history_period']

    return df
