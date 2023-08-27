from icecream import ic

from hummingbot.strategy.filter_factors.utils.calc_std_index import calc_std_for_index


def signal(*args):
    # 表示使用波动率的变化来判断
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
    df[f'{column_name}_pct_change_std'] = df[f'{column_name}_pct_change'].rolling(period).std(ddof=0)
    df[f'{column_name}_pct_change_std_last'] = df[f'{column_name}_pct_change'].shift().rolling(period, min_periods=1).std(ddof=0)
    df[f'{column_name}_volatility_change'] = df[f'{column_name}_pct_change_std'] / df[f'{column_name}_pct_change_std_last'] - 1

    df[factor_name] = df.index.to_series().apply(
        lambda x: calc_std_for_index(
            x, df, f'{column_name}_volatility_change',
            relative_vol_threshold_period,
            relative_vol_threshold_update_period,
            relative_vol_threshold_coefficient
        )
    )

    del df[f'{column_name}_pct_change'], df[f'{column_name}_pct_change_std']
    del df[f'{column_name}_pct_change_std_last'], df[f'{column_name}_volatility_change']

    return df
