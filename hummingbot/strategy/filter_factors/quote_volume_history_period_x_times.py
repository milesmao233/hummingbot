def signal(*args):
    df = args[0]
    column_name = args[1]
    quote_volume_calc_period = args[2][0]
    quote_volume_history_period = args[2][1]
    quote_volume_x_times = args[2][2]
    factor_name = args[3]

    df['quote_volume_calc_period'] = df[column_name].rolling(quote_volume_calc_period).sum()
    # df['sum_amount'] = df['symbol_totalamount'].rolling(window=calc_period).sum()
    # df['moving_average'] = df['sum_amount'].shift(calc_period).rolling(window=history_period).mean()
    df['quote_volume_history_period'] = \
        df['quote_volume_calc_period'].shift(quote_volume_calc_period).rolling(quote_volume_history_period).mean()
    df[factor_name] = df['quote_volume_history_period'] * quote_volume_x_times

    del df['quote_volume_calc_period'], df['quote_volume_history_period']

    return df



# macd_diff_240_12_26_9  macd_diff_240_6_13_5