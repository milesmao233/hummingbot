import numpy as np
import pandas as pd

from hummingbot.strategy.dynamic_hedge.constant import last_candles_num


class HedgeCommonMethod:
    @classmethod
    def concat_all_coin_data(cls, _symbol_list, row_candles):
        df_list = []
        for symbol in _symbol_list:
            df_list.append(row_candles[symbol])

        all_coin_data = pd.concat(df_list, ignore_index=True)
        all_coin_data.sort_values('candle_begin_time', inplace=True)
        all_coin_data.reset_index(inplace=True, drop=True)
        return all_coin_data

    @classmethod
    def concat_two_df_keep_last(cls, df1, df2):
        df = pd.concat([df1, df2], ignore_index=True)
        df.drop_duplicates(subset=['candle_begin_time'], keep='last', inplace=True)
        df.sort_values('candle_begin_time', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    @classmethod
    def generate_index_df(cls, all_coin_data):
        all_coin_data['index_coins'] = all_coin_data['symbol'] + ' '

        group = all_coin_data.groupby('candle_begin_time')

        index_df = pd.DataFrame()
        index_df['index_coins'] = group['index_coins'].sum()
        index_df['index_coins_size'] = group['index_coins'].size()

        # capital_curve: 资金曲线
        index_df['index_capital_curve'] = group['close_capital_curve'].apply(lambda x: np.mean(x, axis=0))
        index_df['open_index_capital_curve'] = group['open_capital_curve'].apply(lambda x: np.mean(x, axis=0))

        # cumulative_pct_change: 累积涨跌幅
        index_df['index_cumulative_pct_change'] = index_df['index_capital_curve'].apply(lambda x: x - 1)
        index_df['index_pct_change'] = index_df['index_capital_curve'].pct_change()
        index_df['index_pct_change'].fillna(value=0, inplace=True)

        index_df['index_total_quote_amount'] = group['quote_volume'].sum()

        # merge_df['index_coins] = group['选币'].sum()
        index_df.sort_values('candle_begin_time', inplace=True)
        index_df.reset_index(inplace=True)

        return index_df

    @classmethod
    def cal_one_filter(cls, coin_df, filter_config, factor_filter_dict, head_columns):
        df = coin_df.copy()
        class_name = filter_config['filter']
        params_list = filter_config['params_list']
        column_name = filter_config['column_name']
        filter_list = []
        # =====技术指标
        _cls = __import__('hummingbot.strategy.filter_factors.%s' % class_name, fromlist=('',))
        for params in params_list:
            str_params = str(params)
            filter_name = f'{class_name}_pl_{column_name}_fl_{str_params}'
            filter_list.append(filter_name)
            # 计算因子
            df = getattr(_cls, 'signal')(df, column_name, params, filter_name)

        filter_list.sort()
        df = df[head_columns + filter_list]
        df.sort_values(by=['candle_begin_time', ], inplace=True)
        df.reset_index(drop=True, inplace=True)

        factor_filter_dict[f'coin_factor_{class_name}_pl_{column_name}'] = df

    @classmethod
    def calc_one_df_filter(cls, symbol_name, coin_df, factor_filter_dict, strategy_conf, head_columns):
        factor_filter_dict[symbol_name] = {}
        for filter_factor in strategy_conf['filter_factors']:
            cls.cal_one_filter(coin_df, filter_factor, factor_filter_dict[symbol_name], head_columns)

        return factor_filter_dict

    @classmethod
    def settle_config_data(cls, df_with_index, df_filter_factor):
        config_data_last_df = {}
        for symbol_name, df in df_with_index.items():
            for factor_df in df_filter_factor[symbol_name].values():
                for f in factor_df.columns:
                    df[f] = factor_df[f]

                config_data_last_df[symbol_name] = df.tail(last_candles_num)

        return config_data_last_df

    @classmethod
    def calculated_last_candles(cls, df):
        # 获取 df 第一行数据
        first_row = df.iloc[0]
        symbol = first_row['symbol']
        last_close = first_row['close']
        last_close_capital_curve = first_row['close_capital_curve']
        last_open = first_row['open']
        last_open_capital_curve = first_row['open_capital_curve']

        # 循环从第二行开始
        for i in range(1, len(df)):
            # 获取当前行数据
            current_row = df.iloc[i]
            current_close_pct_change = current_row['close'] / last_close - 1
            current_close_capital_curve = (current_close_pct_change + 1) * last_close_capital_curve

            current_open_pct_change = current_row['open'] / last_open - 1
            current_open_capital_curve = (current_open_pct_change + 1) * last_open_capital_curve

            df.loc[i, 'symbol'] = symbol
            df.loc[i, 'close_pct_change'] = current_close_pct_change
            df.loc[i, 'close_capital_curve'] = current_close_capital_curve
            df.loc[i, 'open_pct_change'] = current_open_pct_change
            df.loc[i, 'open_capital_curve'] = current_open_capital_curve

            # 更新初始变量
            last_close = current_row['close']
            last_close_capital_curve = current_close_capital_curve
            last_open = current_row['open']
            last_open_capital_curve = current_open_capital_curve

        return df

