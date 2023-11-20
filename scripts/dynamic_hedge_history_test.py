import os
import time
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from hummingbot import data_path
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.data_feed.candles_feed.candles_factory import CandlesConfig, CandlesFactory
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


def ensure_datetime_format(date_str):
    # 尝试按照日期时间格式解析字符串
    try:
        # 如果字符串已经是完整的日期时间格式，则直接返回
        dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        # 如果字符串不是完整的日期时间格式，那么假定它是日期格式，并添加午夜时间
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%Y-%m-%d 00:00:00")


def calc_one_df_filter(symbol_name, coin_df, factor_filter_dict, strategy_conf, head_columns):
    factor_filter_dict[symbol_name] = {}
    for filter_factor in strategy_conf['filter_factors']:
        cal_one_filter(coin_df, filter_factor, factor_filter_dict[symbol_name], head_columns)

    return factor_filter_dict


def cal_one_filter(coin_df, filter_config, factor_filter_dict, head_columns):
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


class DynamicHedgeHistoryTest(ScriptStrategyBase):
    exchange: str = "binance"
    trading_pairs: str = {
        "DOGE-USDT", "ETH-USDT", "BTC-USDT",
    }
    markets = {exchange: trading_pairs}
    interval = "1m"
    days_to_download = 6
    current_candles_num = 5
    last_candles_num = 3600
    begin_history_data_path = data_path() + "/begin_history_data"

    symbol_list = ["DOGE-USDT"]
    cover_list = ["ETH-USDT", "BTC-USDT"]
    index_list = ["ETH-USDT", "BTC-USDT"]
    in_waiting_list = []
    in_trading_list = []
    trading_list_before_calc = []
    current_loading_candles_symbol = []
    candles_data_path = data_path() + "/candles_data"
    fetched_history_data = False
    calculated_history_data = False
    multiply_process = True
    save_config_data_time = datetime.now().strftime("%Y%m%d%H")
    history_row_candles_dict = {}
    last_settle_df_dict = {}
    last_index_row_candles = {}
    date_str = "2023-11-01"

    head_columns = [
        'candle_begin_time',
        'symbol',
        'open',
        'close',
        'high',
        'low',
        'volume',
        'quote_volume',
        'close_pct_change',
        'close_capital_curve',
        'open_pct_change',
        'open_capital_curve',
        'index_capital_curve',
        'open_index_capital_curve',
        'index_total_quote_amount',
        'symbol_relative'
    ]

    strategy_conf = {
        "filter_factors": [
            {
                'filter': 'pct_change',
                'column_name': 'symbol_relative',
                'params_list': [1, 5, 10]
            },
            {
                'filter': 'pct_change_std',
                'column_name': 'symbol_relative',
                'params_list': [[60, 5]],

            },
            {
                'filter': 'volatility_change',
                'column_name': 'symbol_relative',
                'params_list': [[60, 5]],
            },
            {
                'filter': 'vol_change_threshold_true',
                'column_name': 'symbol_relative',
                'params_list': [[60, 5, 2880, 480, 4]],
            },
            {
                'filter': 'vol_change_threshold_false',
                'column_name': 'symbol_relative',
                'params_list': [[60, 5, 2880, 480, 4]],
            },
            {
                'filter': 'rolling_mean',
                'column_name': 'close',
                'params_list': [5],
            },
            {
                'filter': 'rolling_sum',
                'column_name': 'quote_volume',
                'params_list': [1, 5, 60]
            },
            {
                'filter': 'quote_volume_history_period_x_times',
                'column_name': 'quote_volume',
                'params_list': [
                    [
                        5,
                        480,
                        5
                    ],
                    [
                        1,
                        480,
                        10
                    ],
                    [
                        60,
                        480,
                        3
                    ]
                ]
            },
            {
                'filter': 'ema',
                'column_name': 'close',
                'params_list': [
                    [240, 12],
                    [240, 26],
                    [60, 12],
                    [60, 26],
                ]
            },
            {
                'filter': 'ema',
                'column_name': 'symbol_relative',
                'params_list': [
                    [240, 12],
                    [240, 26],
                    [60, 12],
                    [60, 26],
                ]
            },
            {
                'filter': 'macd_diff',
                'column_name': 'close',
                'params_list': [
                    [240, [12, 26, 9]],
                    [60, [12, 26, 9]],
                ]
            },
            {
                'filter': 'macd_diff',
                'column_name': 'symbol_relative',
                'params_list': [
                    [240, [12, 26, 9]],
                    [60, [12, 26, 9]],
                ]
            },
            {
                'filter': 'macd_dea',
                'column_name': 'close',
                'params_list': [
                    [240, [12, 26, 9]],
                    [60, [12, 26, 9]],
                ]
            },
            {
                'filter': 'macd_dea',
                'column_name': 'symbol_relative',
                'params_list': [
                    [240, [12, 26, 9]],
                    [60, [12, 26, 9]],
                ]
            },
        ],
        "conf": {
            "g_std_period": 60,
            "g_std_interval": 5,
            "g_relative_vol_threshold_period": 2880,
            "g_relative_vol_threshold_coefficient": 4,
            "g_relative_vol_threshold_update_period": 480,
            "g_relative_vol_change_threshold_period": 2880,
            "g_relative_vol_change_threshold_coefficient": 4,
            "g_relative_vol_change_threshold_update_period": 480,
            "g_totalamount_check_params": [
                [
                    5,
                    480,
                    5
                ],
                [
                    1,
                    480,
                    10
                ],
                [
                    60,
                    480,
                    3
                ]
            ],
            "g_price_change_threshold": 0.0001,
            "g_relative_price_change_threshold": 0.0001,
            "g_accu_relative_price_change_threshold": 0.003,
            "g_accu_relative_price_change_period": 10,
            "g_base_price_period_before_waiting_list": 1,
            "g_accu_change_threshold": 0.02,
            "g_accu_change_discard_threshold": 0.08,
            "g_observed_timeout": 7200,
            "g_macd_period": [
                240,
                60
            ],
            "g_macd_params": [
                12,
                26,
                9
            ],
            "g_take_profit_relative_change_trigger_threshold": 0.02,
            "g_take_profit_relative_change_down_threshold": 0.02,
            "g_take_profit_relative_percent_threshold": 0,
            "g_stop_loss_relative_change_threshold": -0.03,
            "g_opened_timeout": 240,
            "g_total_fund": 10000,
            "g_leverage": 6,
            "g_max_open_positions": 3,
            "g_posions_percent": [
                0.4,
                0.3,
                0.3
            ],
            "g_name": "g_1"
        },
        "waiting_list_switch": {
            # check_relative_vol_threshold_True 表示使用波动率的变化来判断
            # check_relative_vol_threshold_False 表示使用波动率来判断
            "check_relative_vol_threshold_True": False,
            "check_relative_vol_threshold_False": False,
            # 2-3-11
            "check_price_change_threshold": True,
            # 2-3-12
            "check_relative_price_change_threshold": True,
            # 2-3-13
            "check_accu_relative_price_change_threshold": True,
            # 2-3-14
            "check_totalamount_condition_0": True,
        },
        "trading_list_switch": {
            # check_relative_vol_threshold_True 表示使用波动率的变化来判断
            # 2-5-V1
            "check_relative_vol_threshold_True": False,
            # check_relative_vol_threshold_False 表示使用波动率来判断
            # 2-5-V2
            "check_relative_vol_threshold_False": True,
            # 2-5
            "check_accu_change_threshold": True,
            # 2-5-11
            "check_price_change_threshold": True,
            # 2-5-12
            "check_relative_price_change_threshold": True,
            # 2-5-13A
            "check_macd_condition": True,
            # 2-5-13B
            "check_macd_condition_relative": False,
            # 2-5-14A
            "check_totalamount_condition_0": False,
            # 2-5-14B
            "check_totalamount_condition_1": False,
            # 2-5-14C
            "check_totalamount_condition_2": True,

        }
    }

    @property
    def all_candles_ready(self):
        """
        Checks if the candlesticks are full.
        :return:
        """

        return all([
            self.candles[symbol][process_type].is_ready for symbol, process_type in self.current_loading_candles_symbol
        ])

    @staticmethod
    def get_max_records(days_to_download: int, interval: str) -> int:
        conversion = {"m": 1, "h": 60, "d": 1440}
        unit = interval[-1]
        quantity = int(interval[:-1])
        return int(days_to_download * 24 * 60 * quantity / conversion[unit])

    @staticmethod
    def to_offset_candles(df):
        df = df.copy()
        columns = {
            'timestamp': 'candle_begin_time',
            'quote_asset_volume': 'quote_volume',
            'n_trades': 'trade_num',
            'taker_buy_base_volume': 'taker_buy_base_asset_volume',
            'taker_buy_quote_volume': 'taker_buy_quote_asset_volume'
        }
        df.rename(columns=columns, inplace=True)

        df['candle_begin_time'] = pd.to_datetime(df['candle_begin_time'], unit='ms')
        # # =====获取当前服务器时区，距离UTC 0点的偏差
        utc_offset = int(time.localtime().tm_gmtoff / 60 / 60)
        # df = df[df['candle_begin_time'] + pd.Timedelta(hours=utc_offset) < self.run_time]
        df['candle_begin_time'] = df['candle_begin_time'] + pd.Timedelta(hours=utc_offset)
        df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def to_calculated_candles(df, symbol):
        df['symbol'] = symbol
        df['close'].fillna(method='ffill', inplace=True)
        df['open'].fillna(value=df['close'], inplace=True)
        df['high'].fillna(value=df['close'], inplace=True)
        df['low'].fillna(value=df['close'], inplace=True)
        df['close_pct_change'] = df['close'].pct_change()
        df['close_pct_change'].fillna(value=0, inplace=True)
        df['close_capital_curve'] = (df['close_pct_change'] + 1).cumprod()

        df['open_pct_change'] = df['open'].pct_change()
        df['open_pct_change'].fillna(value=0, inplace=True)
        df['open_capital_curve'] = (df['open_pct_change'] + 1).cumprod()
        df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def calculate_minute_klines(start_date_str):
        # 转换起始日期字符串为datetime对象
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d %H:%M:%S")
        # 获取当前日期和时间
        now = datetime.now()
        # 计算两个日期之间的时间差
        delta = now - start_date
        # 时间差转换为分钟
        minutes_diff = delta.total_seconds() / 60
        # 由于每个1分钟K线对应一分钟，所以直接取整就是1分钟K线的数量
        k_line_count = int(minutes_diff)
        # 打印结果
        print(f"从 {start_date_str} 到现在一共有 {k_line_count} 根1分钟K线。")

        return k_line_count

    @staticmethod
    def calc_all_coin_data(_symbol_list, row_candles):
        # if self.multiply_process:
        #     df_list = Parallel(n_jobs=max(os.cpu_count() - 1, 1))(
        #         delayed(calc_coin_data)(row_candles, symbol) for symbol in _symbol_list
        #     )
        # else:
        df_list = []
        for symbol in _symbol_list:
            df_list.append(row_candles[symbol])

        all_coin_data = pd.concat(df_list, ignore_index=True)
        all_coin_data.sort_values('candle_begin_time', inplace=True)
        all_coin_data.reset_index(inplace=True, drop=True)
        return all_coin_data

    @staticmethod
    def generate_index_df(all_coin_data):
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

    @staticmethod
    def concat_two_df_keep_last(df1, df2):
        df = pd.concat([df1, df2], ignore_index=True)
        df.drop_duplicates(subset=['candle_begin_time'], keep='last', inplace=True)
        df.sort_values('candle_begin_time', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def _calculated_last_candles(df):
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


    def __init__(self, connectors: Dict[str, ConnectorBase]):
        # Is necessary to start the Candles Feed.
        super().__init__(connectors)
        self.connector = self.connectors[self.exchange]
        self.candles = {f"{trading_pair}": {} for trading_pair in self.trading_pairs}

        self._start_date = ensure_datetime_format(self.date_str)
        num_to_download = self.calculate_minute_klines(self._start_date)

        for trading_pair in self.trading_pairs:
            history_candle_config = CandlesConfig(
                connector=self.exchange,
                trading_pair=trading_pair,
                interval=self.interval,
                max_records=num_to_download,
            )
            candle_history = CandlesFactory.get_candle(history_candle_config)
            self.candles[trading_pair]["candle_history"] = candle_history

            current_candle_config = CandlesConfig(
                connector=self.exchange,
                trading_pair=trading_pair,
                interval=self.interval,
                max_records=self.current_candles_num,
            )
            current_candles = CandlesFactory.get_candle(current_candle_config)
            current_candles.start()
            self.candles[trading_pair]["current_candles"] = current_candles

            symbol_candles_path = self.candles_data_path + f"/{trading_pair}_candles.csv"
            process_type = 'candle_history'  # 默认为完整历史数据处理
            if os.path.exists(symbol_candles_path):
                df = pd.read_csv(symbol_candles_path, parse_dates=['candle_begin_time'])
                last_time = df.iloc[-1]['candle_begin_time']

                now = datetime.now()
                delta = now - last_time
                minutes_diff = delta.total_seconds() / 60
                time_diff = int(minutes_diff)

                if time_diff <= 1440:  # 1440 一天
                    # full_history_required = False
                    process_type = 'increment_candles'

                    increment_candle_config = CandlesConfig(
                        connector=self.exchange,
                        trading_pair=trading_pair,
                        interval=self.interval,
                        max_records=time_diff,
                    )

                    increment_candles = CandlesFactory.get_candle(increment_candle_config)
                    self.candles[trading_pair]["increment_candles"] = increment_candles

            self._start_data_retrieval(trading_pair, process_type)

    def _start_data_retrieval(self, symbol, process_type):
        """根据处理类型开始数据检索过程，并记录相应的日志。"""

        process_label = '完整历史数据' if process_type == 'candle_history' else '增量历史数据'
        self.logger().info(f'即将获取{process_label}')

        # 启动相应的数据检索过程
        self.candles[symbol][process_type].start()
        self.current_loading_candles_symbol.append((symbol, process_type))

    def all_current_loading_candles_stop(self):
        for symbol, process_type in self.current_loading_candles_symbol:
            self.candles[symbol][process_type].stop()

    def on_tick(self):
        if not self.fetched_history_data and self.all_candles_ready:
            self.history_row_candles_dict = self._start_fetch_history_data()
            self.all_current_loading_candles_stop()

        elif not self.calculated_history_data:
            self.logger().info('开始计算历史数据')
            # new_candles_dict = self.get_new_last_df()
            df_with_index = self.get_processed_df(self.history_row_candles_dict)
            df_filter_factor = self.calc_factor(df_with_index)
            self.last_settle_df_dict = self.settle_config_data(df_with_index, df_filter_factor)
            self.last_index_row_candles = self.get_last_index_candles(self.history_row_candles_dict)

            self.logger().info('计算历史数据完成, 获取计算后的最后两行数据')

            self.calculated_history_data = True
        else:
            self.trading_list_before_calc = self.in_trading_list.copy()
            current_time = datetime.now()
            print(f'当前时间 {current_time}，开始获取最新数据并进行计算')
            new_candles_dict = self.get_new_last_df()
            before_config_dict = self.set_new_last_df_factor(new_candles_dict, self.last_settle_df_dict)
            self.last_index_row_candles \
                = self.update_last_index_candles(new_candles_dict, self.last_index_row_candles)
            update_index_candles_data = self.update_index_candles_data(self.last_index_row_candles)

            # self.last_settle_df_dict = self.settle_last_data(before_config_dict, update_index_candles_data,
            #                                                  current_time)

            # self.calc_trading_position()

            print('tick')

    def on_stop(self):
        for trading_pair in self.trading_pairs:
            self.candles[trading_pair]["current_candles"].stop()

    def settle_last_data(self, before_config_dict, update_index_candles_data, current_time):
        after_config_dict = {}
        if self.save_config_data_time != current_time.strftime('%Y%m%d%H'):
            save_tag = True
            self.save_config_data_time = current_time.strftime('%Y%m%d%H')
        else:
            save_tag = False

        for symbol, df in before_config_dict.items():
            # 循环 df
            # 填充最后3行新的数据
            after_df = self.calc_last_df(symbol, df, update_index_candles_data)
            after_df = after_df.tail(self.last_candles_num)
            after_df.reset_index(drop=True, inplace=True)

            if save_tag:
                self.save_log(after_df, f'{symbol}_after_config_data')

            after_config_dict[symbol] = after_df

        return after_config_dict

    def update_index_candles_data(self, last_index_row_candles):
        df_list = []
        for df in last_index_row_candles.values():
            df_list.append(df)

        all_coin_data = pd.concat(df_list, ignore_index=True)
        all_coin_data.sort_values('candle_begin_time', inplace=True)
        all_coin_data.reset_index(inplace=True, drop=True)
        print('all_coin_data\n', all_coin_data)

        index_df = self.generate_index_df(all_coin_data)
        print('index_df\n', index_df)

        return index_df

    def update_last_index_candles(self, new_candles_dict, last_index_row_candles):
        updated_last_index_candles = {}
        for symbol_name, index_df in last_index_row_candles.items():
            new_df = new_candles_dict[symbol_name]
            df = self.concat_two_df_keep_last(index_df, new_df)

            # self_log
            df_new = df.tail(self.current_candles_num + 1)
            df_new.reset_index(drop=True, inplace=True)
            df_new = self._calculated_last_candles(df_new)

            df = self.concat_two_df_keep_last(index_df, df_new)

            # df 只保留最后 last_candles_num 行
            df = df.tail(self.last_candles_num)
            df.reset_index(drop=True, inplace=True)
            updated_last_index_candles[symbol_name] = df

        return updated_last_index_candles

    def set_new_last_df_factor(self, new_candles_dict, last_settle_df_dict):
        before_config_dict = {}
        for symbol_name, settle_df in last_settle_df_dict.items():
            new_df = new_candles_dict[symbol_name]

            df = pd.concat([settle_df, new_df], ignore_index=True)
            df.drop_duplicates(subset=['candle_begin_time'], keep='last', inplace=True)
            df.sort_values('candle_begin_time', inplace=True)

            # 获取最后 last_candles_num 行 数据
            df = df.tail(self.last_candles_num)
            df.reset_index(drop=True, inplace=True)

            before_config_dict[symbol_name] = df

        return before_config_dict

    def get_last_index_candles(self, history_row_candles):
        last_index_row_candles = {}
        for symbol_name in self.index_list:
            last_index_row_candles[symbol_name] = history_row_candles[symbol_name].tail(self.last_candles_num)

        return last_index_row_candles

    def settle_config_data(self, df_with_index, df_filter_factor):
        config_data_last_df = {}
        for symbol_name, df in df_with_index.items():
            for factor_df in df_filter_factor[symbol_name].values():
                for f in factor_df.columns:
                    df[f] = factor_df[f]

                config_data_last_df[symbol_name] = df.tail(self.last_candles_num)

        return config_data_last_df

    def calc_factor(self, df_with_index):
        factor_filter_dict = {}
        factor_filter_list = []
        if self.multiply_process:
            factor_filter_list = Parallel(n_jobs=max(os.cpu_count() - 1, 1))(
                delayed(calc_one_df_filter)(symbol_name, coin_df, factor_filter_dict, self.strategy_conf, self.head_columns)
                for symbol_name, coin_df in df_with_index.items()
            )
        else:
            for symbol_name, coin_df in df_with_index.items():
                d = calc_one_df_filter(symbol_name, coin_df, factor_filter_dict, self.strategy_conf, self.head_columns)
                factor_filter_list.append(d)

        factor_filter_dict_result = {}
        for item in factor_filter_list:
            factor_filter_dict_result.update(item)

        return factor_filter_dict_result

    def get_new_last_df(self):
        new_candles_dict = {}
        for trading_pair, candles_info in self.candles.items():
            if not candles_info["current_candles"].is_ready:
                self.logger().info(
                    f"current candles not ready yet for {trading_pair}! Missing {candles_info['current_candles']._candles.maxlen - len(candles_info['current_candles']._candles)}")
            else:
                symbol_name = trading_pair.split('_')[0]
                df = candles_info["current_candles"].candles_df
                df = self.to_offset_candles(df)
                # df = to_calculated_candles(df, symbol_name)
                new_candles_dict[symbol_name] = df
        return new_candles_dict

    def get_processed_df(self, row_candles):
        index_coin_data = self.calc_all_coin_data(self.index_list, row_candles)

        index_df = self.generate_index_df(index_coin_data)

        symbol_coin_data = self.calc_all_coin_data(self.symbol_list, row_candles)

        merged_df = pd.merge(symbol_coin_data, index_df, on='candle_begin_time', how='left')
        merged_df['symbol_relative'] = merged_df['close'] / merged_df['index_capital_curve']

        grouped = merged_df.groupby('symbol', group_keys=False)
        processed_dict = {}
        for symbol_name, coin_df in grouped:
            coin_df.reset_index(drop=True, inplace=True)
            processed_dict[symbol_name] = coin_df

        return processed_dict


    def _start_fetch_history_data(self):
        history_row_candles_dict = {}
        for symbol, process_type in self.current_loading_candles_symbol:
            candles = self.candles[symbol][process_type].candles_df
            # 删除 最后一行最新的数据
            candles.drop(index=candles.index[-1], inplace=True)
            df = self.to_offset_candles(candles)
            # df = self.to_calculated_candles(df, symbol)

            # print('df \n', df)

            if process_type == 'candle_history':
                print(f'save history {symbol}')
                df = self.to_calculated_candles(df, symbol)
                df.to_csv(self.candles_data_path + f"/{symbol}_candles.csv", index=False)
                history_row_candles_dict[symbol] = df
            else:
                print(f'save append {symbol}')
                df_history = pd.read_csv(self.candles_data_path + f"/{symbol}_candles.csv", parse_dates=['candle_begin_time'])
                # 通过 df_history 的最后一行，计算增量数据
                df_history_last_row = df_history.tail(1)
                # print('df_history_last_row \n', df_history_last_row)

                df_append = self.calculate_append_df(df, symbol, df_history_last_row)
                # 计算 df_append 有几行，删除 df_history 最初的这几行
                df_history.drop(index=range(len(df_append)), inplace=True)
                # 将 df_append 与 df_history 合并
                df = pd.concat([df_history, df_append], ignore_index=True)
                # 去重 排序
                df.drop_duplicates(subset=['candle_begin_time'], inplace=True, keep='last')
                df.sort_values(by=['candle_begin_time'], inplace=True)
                df.reset_index(drop=True, inplace=True)
                df.to_csv(self.candles_data_path + f"/{symbol}_candles.csv", index=False)
                history_row_candles_dict[symbol] = df

        self.fetched_history_data = True
        return history_row_candles_dict

    def calculate_append_df(self, append_df, symbol, df_history_last_row):
        df = pd.concat([df_history_last_row, append_df], ignore_index=True)
        df.sort_values(by=['candle_begin_time'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        return self._calculated_last_candles(df)
