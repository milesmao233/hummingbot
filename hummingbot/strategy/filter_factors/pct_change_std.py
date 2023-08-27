def signal(*args):
    df = args[0]
    column_name = args[1]
    std_period = args[2][0]
    std_interval = args[2][1]
    factor_name = args[3]

    period = int(std_period / std_interval)

    df[f'{column_name}_pct_change'] = df[column_name].pct_change(std_interval)
    df[factor_name] = df[f'{column_name}_pct_change'].rolling(period).std(ddof=0)

    return df
