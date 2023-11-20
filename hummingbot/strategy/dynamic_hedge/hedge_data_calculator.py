import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from hummingbot.strategy.dynamic_hedge.constant import (
    current_candles_num,
    head_columns,
    last_candles_num,
    strategy_conf,
)
from hummingbot.strategy.dynamic_hedge.hedge_common_method import HedgeCommonMethod


class HedgeDataCalculator:
    def __init__(self, symbol_list, index_list, history_row_candles_dict):
        self.symbol_list = symbol_list
        self.index_list = index_list
        self.history_row_candles_dict = history_row_candles_dict
        self.multiply_process = True
        self._new_candles_dict = {}
        self._four_hours_candles_dict = {}
        self._last_settle_df_dict = {}
        self._last_index_row_candles = {}
        self._combined_index_data = None

    @property
    def last_settle_df_dict(self):
        return self._last_settle_df_dict

    def update_last_settle_df_dict(self, last_settle_df_dict):
        self._last_settle_df_dict = last_settle_df_dict

    def update_last_index_row_candles(self, last_index_row_candles):
        self._last_index_row_candles = last_index_row_candles

    def update_new_candles(self, new_candles_dict):
        self._new_candles_dict = new_candles_dict

    def update_four_hours_candles(self, four_hours_candles_dict):
        self._four_hours_candles_dict = four_hours_candles_dict

    def update_combined_index_data(self):
        df_list = []
        for df in self._last_index_row_candles.values():
            df_list.append(df)

        all_coin_data = pd.concat(df_list, ignore_index=True)
        all_coin_data.sort_values('candle_begin_time', inplace=True)
        all_coin_data.reset_index(inplace=True, drop=True)

        index_df = HedgeCommonMethod.generate_index_df(all_coin_data)

        self._combined_index_data = index_df

        return index_df

    def get_processed_df(self):
        index_coin_data = HedgeCommonMethod.concat_all_coin_data(self.index_list, self.history_row_candles_dict)
        index_df = HedgeCommonMethod.generate_index_df(index_coin_data)
        symbol_coin_data = HedgeCommonMethod.concat_all_coin_data(self.symbol_list, self.history_row_candles_dict)
        merged_df = pd.merge(symbol_coin_data, index_df, on='candle_begin_time', how='left')
        merged_df['symbol_relative'] = merged_df['close'] / merged_df['index_capital_curve']

        grouped = merged_df.groupby('symbol', group_keys=False)
        processed_dict = {}
        for symbol_name, coin_df in grouped:
            coin_df.reset_index(drop=True, inplace=True)
            processed_dict[symbol_name] = coin_df

        return processed_dict

    def calc_factor(self, df_with_index):
        factor_filter_dict = {}
        factor_filter_list = []
        if self.multiply_process:
            factor_filter_list = Parallel(n_jobs=max(os.cpu_count() - 1, 1))(
                delayed(HedgeCommonMethod.calc_one_df_filter)(symbol_name, coin_df, factor_filter_dict, strategy_conf, head_columns)
                for symbol_name, coin_df in df_with_index.items()
            )
        else:
            for symbol_name, coin_df in df_with_index.items():
                d = HedgeCommonMethod.calc_one_df_filter(symbol_name, coin_df, factor_filter_dict, strategy_conf, head_columns)
                factor_filter_list.append(d)

        factor_filter_dict_result = {}
        for item in factor_filter_list:
            factor_filter_dict_result.update(item)

        return factor_filter_dict_result

    def get_last_index_candles(self, num_rows):
        last_index_row_candles = {}
        for symbol_name in self.index_list:
            last_index_row_candles[symbol_name] = self.history_row_candles_dict[symbol_name].tail(num_rows)

        return last_index_row_candles

    def get_last_concat_new_df(self, num):
        before_config_dict = {}
        for symbol_name, settle_df in self._last_settle_df_dict.items():
            new_df = self._new_candles_dict[symbol_name]

            df = pd.concat([settle_df, new_df], ignore_index=True)
            df.drop_duplicates(subset=['candle_begin_time'], keep='last', inplace=True)
            df.sort_values('candle_begin_time', inplace=True)

            # 获取最后 last_candles_num 行 数据
            df = df.tail(num)
            df.reset_index(drop=True, inplace=True)

            before_config_dict[symbol_name] = df

        return before_config_dict

    def calc_updated_last_index_candles(self):
        updated_last_index_candles = {}
        for symbol_name, index_df in self._last_index_row_candles.items():
            new_df = self._new_candles_dict[symbol_name]
            df = HedgeCommonMethod.concat_two_df_keep_last(index_df, new_df)

            # self_log
            df_new = df.tail(current_candles_num + 1)
            df_new.reset_index(drop=True, inplace=True)
            df_new = HedgeCommonMethod.calculated_last_candles(df_new)

            df = HedgeCommonMethod.concat_two_df_keep_last(index_df, df_new)

            # df 只保留最后 last_candles_num 行
            df = df.tail(last_candles_num)
            df.reset_index(drop=True, inplace=True)
            updated_last_index_candles[symbol_name] = df

        return updated_last_index_candles

    def calc_last_config_data(self, before_config_dict):
        after_config_dict = {}
        # if self.save_config_data_time != current_time.strftime('%Y%m%d%H'):
        #     save_tag = True
        #     self.save_config_data_time = current_time.strftime('%Y%m%d%H')
        # else:
        #     save_tag = False

        for symbol, df in before_config_dict.items():
            # 循环 df
            # 填充最后3行新的数据
            after_df = self.calc_last_df(symbol, df)
            after_df = after_df.tail(last_candles_num)
            after_df.reset_index(drop=True, inplace=True)

            # if save_tag:
            #     self.save_log(after_df, f'{symbol}_after_config_data')

            after_config_dict[symbol] = after_df

        return after_config_dict

    def calc_last_df(self, symbol, df):
        index_row_candles = self._combined_index_data.copy().set_index('candle_begin_time')
        df = self.update_base_data(df, symbol, index_row_candles)
        df = self.update_first_cycle_data(df)
        df = self.update_second_cycle_data(df, symbol)
        df = self.update_third_cycle_data(df)

        return df

    def update_base_data(self, df, symbol, index_row_candles):
        # # 将 last_index_row_candles 变成以 candle_begin_time 为 index
        for i in range(len(df) - (current_candles_num + 2), len(df)):
            current_row = df.iloc[i]
            last_row = df.iloc[i - 1]
            last_close = last_row['close']
            last_close_capital_curve = last_row['close_capital_curve']
            last_open = last_row['open']
            last_open_capital_curve = last_row['open_capital_curve']

            current_close_pct_change = current_row['close'] / last_close - 1
            current_close_capital_curve = (current_close_pct_change + 1) * last_close_capital_curve

            current_open_pct_change = current_row['open'] / last_open - 1
            current_open_capital_curve = (current_open_pct_change + 1) * last_open_capital_curve

            df.loc[i, 'symbol'] = symbol
            df.loc[i, 'close_pct_change'] = current_close_pct_change
            df.loc[i, 'close_capital_curve'] = current_close_capital_curve
            df.loc[i, 'open_pct_change'] = current_open_pct_change
            df.loc[i, 'open_capital_curve'] = current_open_capital_curve

            # 获取 candle_time
            candle_time = current_row['candle_begin_time']
            # ic(candle_time)
            current_index_row = index_row_candles.loc[candle_time]
            df.loc[i, 'index_coins'] = current_index_row['index_coins']
            df.loc[i, 'index_coins_size'] = current_index_row['index_coins_size']
            df.loc[i, 'index_capital_curve'] = current_index_row['index_capital_curve']
            df.loc[i, 'open_index_capital_curve'] = current_index_row['open_index_capital_curve']
            df.loc[i, 'index_cumulative_pct_change'] = current_index_row['index_cumulative_pct_change']
            df.loc[i, 'index_pct_change'] = current_index_row['index_pct_change']
            df.loc[i, 'index_total_quote_amount'] = current_index_row['index_total_quote_amount']
            df.loc[i, 'symbol_relative'] = current_row['close'] / current_index_row['index_capital_curve']

        return df

    def update_first_cycle_data(self, df):
        for i in range(len(df) - (current_candles_num + 2), len(df)):
            current_row = df.iloc[i]
            last_row = df.iloc[i - 1]
            candle_time = current_row['candle_begin_time']

            for filter_factor in strategy_conf['filter_factors']:
                filter_name = filter_factor['filter']
                column_name = filter_factor['column_name']
                params_list = filter_factor['params_list']

                if filter_name == 'pct_change':
                    for params in params_list:
                        period = int(params)
                        previous_close = df.iloc[i - period][column_name]
                        if i - period < 0:
                            current_pct_change = np.nan
                        else:
                            current_pct_change = current_row[column_name] / previous_close - 1

                        df.loc[i, f'{filter_name}_pl_{column_name}_fl_{params}'] = current_pct_change

                elif filter_name == 'rolling_mean':
                    for params in params_list:
                        period = int(params)
                        start_i = int(i - period + 1)
                        if start_i < 1:
                            current_mean = np.nan
                        else:
                            # ic(df.iloc[start_i:i+1][column_name])
                            current_mean = df.iloc[start_i:i + 1][column_name].mean()

                        df.loc[i, f'{filter_name}_pl_{column_name}_fl_{params}'] = current_mean

                elif filter_name == 'rolling_sum':
                    for params in params_list:
                        period = int(params)
                        start_i = int(i - period + 1)
                        if start_i < 1:
                            current_sum = np.nan
                        else:
                            current_sum = df.iloc[start_i:i + 1][column_name].sum()

                        df.loc[i, f'{filter_name}_pl_{column_name}_fl_{params}'] = current_sum

                elif filter_name == 'ema':
                    for params in params_list:
                        period = int(params[0])
                        span = int(params[1])
                        previous_ema = last_row[f'ema_pl_{column_name}_fl_{[period, span]}']

                        time_period_min = int(period / 60)
                        if candle_time.hour % time_period_min == 0 and candle_time.minute == 0:
                            current_ema = current_row[column_name]
                        else:
                            alpha = 2 / (span + 1)
                            current_ema = (1 - alpha) * previous_ema + alpha * current_row[column_name]

                        df.loc[i, f'ema_pl_{column_name}_fl_{params}'] = current_ema

                elif filter_name == 'macd_diff':
                    for params in params_list:
                        period = int(params[0])
                        short_span = int(params[1][0])
                        long_span = int(params[1][1])
                        previous_short_ema = last_row[f'ema_pl_{column_name}_fl_{[period, short_span]}']
                        previous_long_ema = last_row[f'ema_pl_{column_name}_fl_{[period, long_span]}']

                        time_period_min = int(period / 60)
                        if candle_time.hour % time_period_min == 0 and candle_time.minute == 0:
                            current_short_ema = current_row[column_name]
                            current_long_ema = current_row[column_name]
                        else:
                            short_alpha = 2 / (short_span + 1)
                            long_alpha = 2 / (long_span + 1)
                            current_short_ema \
                                = (1 - short_alpha) * previous_short_ema + short_alpha * current_row[column_name]
                            current_long_ema \
                                = (1 - long_alpha) * previous_long_ema + long_alpha * current_row[column_name]

                        current_diff = current_short_ema - current_long_ema
                        df.loc[i, f'{filter_name}_pl_{column_name}_fl_{params}'] = current_diff

        return df

    def update_second_cycle_data(self, df, symbol):
        for i in range(len(df) - (current_candles_num + 2), len(df)):
            current_row = df.iloc[i]
            last_row = df.iloc[i - 1]
            candle_time = current_row['candle_begin_time']

            for filter_factor in strategy_conf['filter_factors']:
                filter_name = filter_factor['filter']
                column_name = filter_factor['column_name']
                params_list = filter_factor['params_list']

                if filter_name == 'pct_change_std':
                    pass
                    for params in params_list:
                        std_period = int(params[0])
                        pct_span = int(params[1])
                        period = std_period / pct_span

                        if int(i - period + 1) < 1:
                            current_pct_change_std = np.nan
                        else:
                            previous_pct_change_list = df.iloc[int(i - period + 1):(i + 1)][
                                f'pct_change_pl_{column_name}_fl_{pct_span}'].values
                            current_pct_change_std = np.std(previous_pct_change_list)

                        df.loc[i, f'{filter_name}_pl_{column_name}_fl_{params}'] = current_pct_change_std

                elif filter_name == 'quote_volume_history_period_x_times':
                    for params in params_list:
                        quote_volume_calc_period = int(params[0])  # 5
                        quote_volume_history_period = int(params[1])  # 7
                        quote_volume_x_times = float(params[2])  # 5

                        n = int(quote_volume_history_period * 24 / 4)
                        four_hour_df = self._four_hours_candles_dict[symbol]
                        # doge_four_hour_df 获取最后 n 个 quote_volume 的均值，不包括最后一个

                        mean_of_last_n = four_hour_df['quote_volume'].iloc[(-n - 1):-1].mean()
                        quote_volume_history_period = mean_of_last_n * (quote_volume_calc_period / (4 * 60))
                        df.loc[
                            i, f'{filter_name}_pl_{column_name}_fl_{params}'] = (
                                quote_volume_history_period * quote_volume_x_times)

                elif filter_name == 'volatility_change':
                    for params in params_list:
                        std_period = int(params[0])
                        std_interval = int(params[1])

                        if int(i - std_period + 1) < 1:
                            vol_change = np.nan
                        else:
                            previous_pct_change_list = df.iloc[int(i - std_period + 1):(i + 1)][
                                f'pct_change_pl_{column_name}_fl_{std_interval}'].values
                            current_pct_change_std = np.std(previous_pct_change_list)

                            shift_pct_change_list = df.iloc[int(i - std_period):i][
                                f'pct_change_pl_{column_name}_fl_{std_interval}'].values
                            shift_pct_change_std = np.std(shift_pct_change_list)

                            vol_change = current_pct_change_std / shift_pct_change_std - 1

                        df.loc[i, f'{filter_name}_pl_{column_name}_fl_{params}'] = vol_change

                elif filter_name == 'macd_dea':
                    for params in params_list:
                        period = int(params[0])

                        dea_span = int(params[1][2])
                        previous_dea = last_row[f'{filter_name}_pl_{column_name}_fl_{params}']
                        current_diff = current_row[f'macd_diff_pl_{column_name}_fl_{params}']

                        time_period_min = int(period / 60)
                        if candle_time.hour % time_period_min == 0 and candle_time.minute == 0:
                            current_dea = current_diff
                        else:
                            dea_alpha = 2 / (dea_span + 1)
                            current_dea \
                                = (1 - dea_alpha) * previous_dea + dea_alpha * current_diff

                        df.loc[i, f'{filter_name}_pl_{column_name}_fl_{params}'] = current_dea

        return df

    def update_third_cycle_data(self, df):
        for i in range(len(df) - (current_candles_num + 2), len(df)):
            current_row = df.iloc[i]
            last_row = df.iloc[i - 1]
            candle_time = current_row['candle_begin_time']

            for filter_factor in strategy_conf['filter_factors']:
                filter_name = filter_factor['filter']
                column_name = filter_factor['column_name']
                params_list = filter_factor['params_list']

                if filter_name == 'vol_change_threshold_false':
                    for params in params_list:
                        std_period = int(params[0])
                        std_interval = int(params[1])
                        relative_vol_threshold_period = int(params[2])
                        relative_vol_threshold_update_period = int(params[3])
                        relative_vol_threshold_coefficient = int(params[4])

                        # 获取 f'{filter_name}_pl_{column_name}_fl_{params}' 这一列的所有值
                        vol_change_threshold_values = df[f'{filter_name}_pl_{column_name}_fl_{params}'].values
                        last_index = 0
                        sustain_value = 0
                        for index, value in enumerate(vol_change_threshold_values):
                            if value != sustain_value and not np.isnan(value):
                                last_index = index
                                sustain_value = value
                        sustain_num = i - last_index

                        if sustain_num < relative_vol_threshold_update_period:
                            vol_change_threshold_value = sustain_value
                        else:
                            previous_vol_change_threshold_values \
                                = df.iloc[i - relative_vol_threshold_period + 1:i + 1][
                                f'pct_change_std_pl_{column_name}_fl_{[std_period, std_interval]}'].values
                            vol_change_threshold_value = np.std(
                                previous_vol_change_threshold_values) * relative_vol_threshold_coefficient

                        df.loc[i, f'{filter_name}_pl_{column_name}_fl_{params}'] = vol_change_threshold_value

                elif filter_name == 'vol_change_threshold_true':
                    for params in params_list:
                        std_period = int(params[0])
                        std_interval = int(params[1])
                        relative_vol_threshold_period = int(params[2])
                        relative_vol_threshold_update_period = int(params[3])
                        relative_vol_threshold_coefficient = int(params[4])

                        # 获取 f'{filter_name}_pl_{column_name}_fl_{params}' 这一列的所有值
                        vol_change_threshold_values = df[f'{filter_name}_pl_{column_name}_fl_{params}'].values
                        last_index = 0
                        sustain_value = 0
                        for index, value in enumerate(vol_change_threshold_values):
                            if value != sustain_value and not np.isnan(value):
                                last_index = index
                                sustain_value = value
                        sustain_num = i - last_index
                        if sustain_num < relative_vol_threshold_update_period:
                            vol_change_threshold_value = sustain_value
                        else:
                            previous_vol_change_threshold_values \
                                = df.iloc[i - relative_vol_threshold_period + 1:i + 1][
                                f'volatility_change_pl_{column_name}_fl_{[std_period, std_interval]}'].values
                            vol_change_threshold_value = np.std(
                                previous_vol_change_threshold_values) * relative_vol_threshold_coefficient

                        df.loc[i, f'{filter_name}_pl_{column_name}_fl_{params}'] = vol_change_threshold_value

        return df



