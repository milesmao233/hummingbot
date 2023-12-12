from hummingbot.strategy.filter_factors.utils.calc_std_index import calc_std_for_index, calculate_rolling_std


def signal(*args):
    # 表示使用波动率来判断
    df = args[0]
    column_name = args[1]
    std_period = args[2][0]
    std_interval = args[2][1]
    relative_vol_threshold_period = args[2][2]
    relative_vol_threshold_update_period = args[2][3]
    relative_vol_threshold_coefficient = args[2][4]

    factor_name = args[3]

    period = int(std_period / std_interval)

    df[f'{column_name}_pct_change'] = df[column_name].pct_change(std_interval)

    # 计算标准差，60,5 的参数，表示计算最近60分钟，每5分钟的标准差
    df[f'{column_name}_pct_change_std'] = calculate_rolling_std(df[column_name], period, std_interval)

    # df[f'{column_name}_pct_change_std'] = df[f'{column_name}_pct_change'].rolling(period).std(ddof=0)

    df[factor_name] = df.index.to_series().apply(
        lambda x: calc_std_for_index(
            x, df, f'{column_name}_pct_change_std',
            relative_vol_threshold_period,
            relative_vol_threshold_update_period,
            relative_vol_threshold_coefficient
        )
    )

    del df[f'{column_name}_pct_change'], df[f'{column_name}_pct_change_std']

    return df
