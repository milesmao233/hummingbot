def signal(*args):
    df = args[0]
    column_name = args[1]
    n = args[2]
    factor_name = args[3]

    df[factor_name] = df[column_name].rolling(n).mean()

    return df
