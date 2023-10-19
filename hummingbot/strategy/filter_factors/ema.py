from hummingbot.strategy.filter_factors.utils.calc_period_macd import calculate_ema, calculate_remaining_ema


def signal(*args):
    # VolumeStd
    df = args[0]
    column_name = args[1]
    ema_params = args[2]
    factor_name = args[3]

    period = ema_params[0]
    ema_window = ema_params[1]

    time_period_min = int(period / 60)
    time_interval = f'{time_period_min}H'

    # df = calc_period_macd(df, column_name, macd_params, macd_period)

    df = calculate_ema(df, column_name, ema_window, time_interval)
    df = calculate_remaining_ema(df, column_name, ema_window, time_period_min)

    df[factor_name] = df['ema']

    return df
