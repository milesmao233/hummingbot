def signal(*args):
    df = args[0]
    column_name = args[1]
    std_period = args[2][0]
    std_interval = args[2][1]
    factor_name = args[3]

    df[f'{column_name}_pct_change'] = df[column_name].pct_change(std_interval)
    df[f'{column_name}_pct_change_std'] = df[f'{column_name}_pct_change'].rolling(std_period).std(ddof=0)
    df[f'{column_name}_pct_change_std_last'] = df[f'{column_name}_pct_change'].shift().rolling(std_period, min_periods=1).std(ddof=0)
    df[factor_name] = df[f'{column_name}_pct_change_std'] / df[f'{column_name}_pct_change_std_last'] - 1

    del df[f'{column_name}_pct_change'], df[f'{column_name}_pct_change_std']
    del df[f'{column_name}_pct_change_std_last']

    return df
