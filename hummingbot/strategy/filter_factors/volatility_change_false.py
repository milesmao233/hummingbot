from hummingbot.strategy.filter_factors.utils.calc_std_index import calculate_rolling_std


def signal(*args):
    df = args[0]
    column_name = args[1]
    std_period = args[2][0]  # 60
    std_interval = args[2][1]  # 5
    factor_name = args[3]

    # df[f'{column_name}_pct_change'] = df[column_name].pct_change(std_interval)
    # df[f'{column_name}_pct_change_std'] = df[f'{column_name}_pct_change'].rolling(std_period).std(ddof=0)
    # df[f'{column_name}_pct_change_std_last'] = df[f'{column_name}_pct_change'].shift().rolling(std_period, min_periods=1).std(ddof=0)
    # df[factor_name] = df[f'{column_name}_pct_change_std'] / df[f'{column_name}_pct_change_std_last'] - 1

    period = int(std_period / std_interval)

    df[f'{column_name}_pct_change'] = df[column_name].pct_change(std_interval)
    df[factor_name] = calculate_rolling_std(df[f'{column_name}_pct_change'], period, std_interval)

    del df[f'{column_name}_pct_change']

    return df
