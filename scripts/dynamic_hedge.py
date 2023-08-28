import os
import time
import numpy as np
import pandas as pd

from typing import Dict, List
from datetime import datetime
from decimal import Decimal
from joblib import Parallel, delayed

from hummingbot import data_path
from hummingbot.connector.constants import s_decimal_0
from hummingbot.connector.derivative.position import Position
from hummingbot.connector.trading_rule import TradingRule
from hummingbot.core.data_type.common import OrderType, TradeType, PositionAction
from hummingbot.core.data_type.order_candidate import OrderCandidate, PerpetualOrderCandidate
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory


# from hummingbot.strategy.filter_factors

# pd.set_option('display.max_rows', 1000)
# pd.set_option('expand_frame_repr', False)  # 当列太多时不换行

def get_twap_symbol_info_list(symbol_info, order_amount):
    """
    对超额订单进行拆分,并进行调整,尽可能每批中子订单、每批订单让多空平衡
    :param symbol_info 原始下单信息
    :param order_amount:单次下单最大金额
    """

    # 对下单资金进行拆单
    symbol_info['拆单金额'] = symbol_info['实际下单资金'].apply(
        lambda x: [x] if abs(x) < order_amount else [(1 if x > 0 else -1) * order_amount] * int(
            abs(x) / order_amount) + [x % (order_amount if x > 0 else -order_amount)])
    symbol_info['拆单金额'] = symbol_info['拆单金额'].apply(np.array)  # 将list转成numpy的array
    # 计算拆单金额对应多少的下单量
    symbol_info['实际下单量'] = symbol_info['实际下单量'] / symbol_info['实际下单资金'] * symbol_info['拆单金额']
    symbol_info.reset_index(inplace=True)
    del symbol_info['拆单金额']

    # 将拆单量进行数据进行展开
    symbol_info = symbol_info.explode('实际下单量')

    # 定义拆单list
    twap_symbol_info_list = []
    # 分组
    group = symbol_info.groupby(by='index')
    # 获取分组里面最大的长度
    max_group_len = group['index'].size().max()
    # 批量构建拆单数据
    for i in range(max_group_len):
        twap_symbol_info_list.append(group.nth(i))

    return twap_symbol_info_list


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


def calc_macd_condition(df, period, price_name):
    dif_condition1 = df[f'dif_0_{price_name}'] - df[f'dea_0_{price_name}'] > df[f'dif_0_{price_name}'].shift(period) - \
                     df[
                         f'dea_0_{price_name}'].shift(period)
    dif_condition2 = df[f'dif_0_{price_name}'] - df[f'dea_0_{price_name}'] > df[f'dif_0_{price_name}'].shift(
        period * 2) - df[
                         f'dea_0_{price_name}'].shift(period * 2)
    condition_macd_1 = (df[f'dif_0_{price_name}'] >= 0) & dif_condition1
    condition_macd_2 = (df[f'dif_0_{price_name}'] >= 0) & dif_condition2
    condition_macd_3 = (df[f'dif_0_{price_name}'] < 0) & dif_condition1 & dif_condition2

    return condition_macd_1, condition_macd_2, condition_macd_3


def process_one_line_open_position(df, conf, trading_list_switch):
    g_take_profit_relative_change_trigger_threshold = conf['g_take_profit_relative_change_trigger_threshold']
    g_take_profit_relative_change_down_threshold = conf['g_take_profit_relative_change_down_threshold']
    g_stop_loss_relative_change_threshold = conf['g_stop_loss_relative_change_threshold']
    g_opened_timeout = conf['g_opened_timeout']

    total_check = 0
    for value in trading_list_switch.values():
        if value:
            total_check += 1

    trading_status = dict(
        in_open_position=0,
        back_waiting_list=0,
        take_profit_trigger=False,
        open_symbol_relative=0,
        open_k_lines=0,
        max_symbol_relative_pct_change=0,
        base_price=0,
        base_price_relative=0,
        stop_loss=0,
        stop_trigger=False,
    )

    df['in_open_position'] = np.nan
    df['accu_price_change'] = np.nan
    # df['base_price'] = np.nan
    # df['base_price_relative'] = np.nan
    base_price_period = int(conf["g_base_price_period_before_waiting_list"])
    g_accu_change_threshold = conf["g_accu_change_threshold"]
    check_accu_change_threshold = trading_list_switch["check_accu_change_threshold"]
    observed_timeout = int(conf["g_observed_timeout"]/60)

    for i, row in df.iterrows():
        # 如果已经止损了，但还在 in_waiting_list 标记的，就把 in_waiting_list 标记去掉
        if trading_status['stop_loss'] == 1 and row['in_waiting_list'] == 1:
            df.at[i, 'in_waiting_list'] = 0
            trading_status['stop_loss'] = 0

        if trading_status['back_waiting_list'] > 0:
            trading_status['back_waiting_list'] += 1
        if trading_status['back_waiting_list'] > observed_timeout:
            trading_status['back_waiting_list'] = 0

        # 有一种情况，在 in_open_position 止盈或止损后，waiting_list = 1 那个标记应该变为 0，直到 enter_waiting_list = 1 的出现
        if row['enter_waiting_list'] == 1:
            trading_status['stop_trigger'] = False
        if trading_status['stop_trigger']:
            df.at[i, 'in_waiting_list'] = 0

        if trading_status['in_open_position'] == 0:
            df.at[i, 'in_open_position'] = 0

            # 先更新 base_price 和 base_price_relative
            if row['in_waiting_list'] == 1 or trading_status['back_waiting_list'] > 0:
                # 如果 base_price  == 0， 更新一下 base_price 价格
                if trading_status['base_price'] == 0:
                    close_previous = df.loc[i - base_price_period, 'close']
                    index_previous = df.loc[i - base_price_period, 'index_capital_curve']
                    if (close_previous / index_previous) < (row['close'] / row['index_capital_curve']):
                        trading_status['base_price'] = close_previous
                        trading_status['base_price_relative'] = close_previous / index_previous
                    else:
                        trading_status['base_price'] = row['close']
                        trading_status['base_price_relative'] = row['close'] / row['index_capital_curve']

                else:
                    df.at[i, 'accu_price_change'] = (row['close'] / trading_status['base_price']) - 1

                    if (row['symbol_relative'] / trading_status['base_price_relative'] - 1) < 0:
                        trading_status['base_price'] = row['close']
                        trading_status['base_price_relative'] = row['close'] / row['index_capital_curve']

                df.at[i, '2-5'] = 1 if (df.at[
                                            i, 'accu_price_change'] > g_accu_change_threshold) & check_accu_change_threshold else 0
                # open_check 包含判断 base_price 的逻辑
                if row['open_check_sum'] + df.at[i, '2-5'] == total_check:
                    trading_status['in_open_position'] = 1
                    trading_status['back_waiting_list'] = 0
                    trading_status['open_symbol_relative'] = row['symbol_relative']

            else:
                trading_status['base_price'] = 0
                trading_status['base_price_relative'] = 0

        else:
            df.at[i, 'in_open_position'] = 1
            trading_status['base_price'] = 0
            trading_status['base_price_relative'] = 0

            # 如果 symbol_relative_pct_change < 0.3，说明价格下跌，就平仓
            symbol_relative_pct_change = row['symbol_relative'] / trading_status['open_symbol_relative'] - 1
            # df.at[i, 'symbol_relative_pct_change'] = symbol_relative_pct_change
            # df.at[i, 'max_symbol_relative_pct_change'] = trading_status['max_symbol_relative_pct_change']
            if trading_status['take_profit_trigger']:
                trading_status['open_k_lines'] += 1

            # 止损平仓 或 超时平仓，平仓后要从 waiting_list 移除
            if symbol_relative_pct_change < g_stop_loss_relative_change_threshold\
                    or trading_status['open_k_lines'] > g_opened_timeout:
                df.at[i, 'in_open_position'] = 0
                df.at[i, 'in_waiting_list'] = 0
                df.at[i, 'stop_loss'] = 1
                trading_status['stop_trigger'] = True
                trading_status['stop_loss'] = 1
                trading_status['in_open_position'] = 0
                trading_status['back_waiting_list'] = 0
                trading_status['take_profit_trigger'] = False
                trading_status['open_symbol_relative'] = 0
                trading_status['open_k_lines'] = 0
                trading_status['max_symbol_relative_pct_change'] = 0

            # 追踪止盈
            # 记录累积的最大相对涨幅
            if symbol_relative_pct_change > trading_status['max_symbol_relative_pct_change']:
                trading_status['max_symbol_relative_pct_change'] = symbol_relative_pct_change
            # 触发追踪止盈的 trigger
                if trading_status['max_symbol_relative_pct_change'] > g_take_profit_relative_change_trigger_threshold:
                    trading_status['take_profit_trigger'] = True

            # 如果触发后，回落了 0.02，就追踪止盈，放入 waiting_list 中
            if trading_status['take_profit_trigger'] \
                    and ((trading_status['max_symbol_relative_pct_change'] - symbol_relative_pct_change) > g_take_profit_relative_change_down_threshold):
                df.at[i, 'in_open_position'] = 0
                df.at[i, 'stop_win'] = 1
                trading_status['stop_trigger'] = True
                trading_status['in_open_position'] = 0
                trading_status['back_waiting_list'] = 1
                trading_status['take_profit_trigger'] = False
                trading_status['open_symbol_relative'] = 0
                trading_status['open_k_lines'] = 0
                trading_status['max_symbol_relative_pct_change'] = 0

    df['in_open_position'].fillna(value=0, inplace=True)

    return df


def calc_coin_data(row_candles, symbol):
    # print('self.row_candles', self.row_candles)
    df = row_candles[symbol]
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
        getattr(_cls, 'signal')(df, column_name, params, filter_name)

    filter_list.sort()
    df = df[head_columns + filter_list]
    df.sort_values(by=['candle_begin_time', ], inplace=True)
    df.reset_index(drop=True, inplace=True)

    factor_filter_dict[f'coin_factor_{class_name}_pl_{column_name}'] = df


class DynamicHedge(ScriptStrategyBase):
    # symbol_list = ["DOGE-USDT", "SOL-USDT"]
    # cover_list = ["ETH-USDT", "BNB-USDT"]
    # index_list = ["ETH-USDT", "BNB-USDT"]
    #
    # total_list = []
    # pairs = {}
    # for currency in (symbol_list + cover_list + index_list):
    #     if currency not in total_list:
    #         total_list.append(currency)
    #
    #

    # exchange: str = "binance_perpetual"
    # # base_currencies = ["BTC", "ETH", "MATIC", "XRP", "BNB", "ADA", "DOT", "LTC", "DOGE", "SOL"]
    # # pairs = {f"{currency}-USDT" for currency in base_currencies}

    # markets = {exchange: {"ETH-USDT", "BTC-USDT"}}
    symbol_list = [
        # "DYDX-USDT",
        "DOGE-USDT", "SOL-USDT", "DYDX-USDT", "1000PEPE-USDT", "LPT-USDT"
        # "ENS-USDT", "APE-USDT", "MATIC-USDT", "ADA-USDT", "ATOM-USDT", "FIL-USDT",
        # "AR-USDT", "PEOPLE-USDT", "AAVE-USDT", "UNI-USDT", "DYDX-USDT",
        # "1000SHIB-USDT", "XRP-USDT", "FTM-USDT", "APT-USDT", "CRV-USDT", "NEAR-USDT",
        # "ICP-USDT"
    ]
    cover_list = ["ETH-USDT", "BNB-USDT"]
    index_list = ["ETH-USDT", "BNB-USDT"]

    trading_pairs: str = {
        # "DYDX-USDT", "ETH-USDT", "BNB-USDT",
        "DOGE-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT", "DYDX-USDT", "1000PEPE-USDT", "LPT-USDT"
        # "ENS-USDT", "APE-USDT", "MATIC-USDT", "ADA-USDT", "ATOM-USDT", "FIL-USDT",
        # "AR-USDT", "PEOPLE-USDT", "AAVE-USDT", "UNI-USDT", "DYDX-USDT",
        # "1000SHIB-USDT", "XRP-USDT", "FTM-USDT", "APT-USDT", "CRV-USDT", "NEAR-USDT",
        # "ICP-USDT"
    }
    intervals = ["1m"]
    days_to_download = 6
    exchange: str = "binance_perpetual"
    max_one_order_amount = 500

    markets = {exchange: trading_pairs}
    calc_refresh_time = 10
    create_timestamp = 0
    row_candles = {}
    df_with_index = {}
    df_filter_factor = {}
    config_data = {}
    trading_result = {}
    download_kline_data = True
    run_time = None
    multiply_process = True
    log_data_path = data_path() + f"/{exchange}_log_data"
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

    # config
    strategy_conf = {
        "filter_factors": [
            {
                'filter': 'pct_change',
                'column_name': 'symbol_relative',
                'params_list': [1, 10]
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

    @staticmethod
    def get_max_records(days_to_download: int, interval: str) -> int:
        conversion = {"m": 1, "h": 60, "d": 1440}
        unit = interval[-1]
        quantity = int(interval[:-1])
        return int(days_to_download * 24 * 60 * quantity / conversion[unit])

    def __init__(self, connectors: Dict[str, ConnectorBase]):
        super().__init__(connectors)
        combinations = [(trading_pair, interval) for trading_pair in self.trading_pairs for interval in self.intervals]
        self.candles = {f"{combinations[0]}_{combinations[1]}": {} for combinations in combinations}
        for combination in combinations:
            candle_history = CandlesFactory.get_candle(connector=self.exchange, trading_pair=combination[0],
                                                       interval=combination[1],
                                                       max_records=self.get_max_records(self.days_to_download,
                                                                                        combination[1]))
            self.candles[f"{combination[0]}_{combination[1]}"]["candle_history"] = candle_history
            self.candles[f"{combination[0]}_{combination[1]}"][
                "csv_path"] = data_path() + f"/candles_{self.exchange}_{combination[0]}_{combination[1]}.csv"

            current_candles = CandlesFactory.get_candle(connector=self.exchange, trading_pair=combination[0],
                                                        interval=combination[1],
                                                        max_records=10)
            current_candles.start()
            self.candles[f"{combination[0]}_{combination[1]}"]["current_candles"] = current_candles
            self.candles[f"{combination[0]}_{combination[1]}"]["download_kline_data"] = False
            self.candles[f"{combination[0]}_{combination[1]}"]["start_kline_data"] = False

        self.connector = self.connectors[self.exchange]

    @property
    def all_history_candles_download(self):
        is_download_list = [candles_info["download_kline_data"] for candles_info in self.candles.values()]
        return all(is_download_list)

    @property
    def all_history_candles_start(self):
        is_download_list = [candles_info["start_kline_data"] for candles_info in self.candles.values()]
        return all(is_download_list)

    @property
    def active_positions(self) -> Dict[str, Position]:

        return self.connector.account_positions

    def active_positions_df(self) -> pd.DataFrame:
        columns = ["Symbol", "Type", "Entry Price", "Amount", "Leverage", "Unrealized PnL"]
        data = []
        # market, trading_pair = self._market_info.market, self._market_info.trading_pair
        for idx in self.active_positions.values():
            # is_buy = True if idx.amount > 0 else False
            # unrealized_profit = ((market.get_price(trading_pair, is_buy) - idx.entry_price) * idx.amount)
            data.append([
                idx.trading_pair,
                idx.position_side.name,
                float(idx.entry_price),
                float(idx.amount),
                idx.leverage,
                float(idx.unrealized_pnl)
            ])

        df = pd.DataFrame(data=data, columns=columns)
        df.set_index('Symbol', inplace=True)
        return df


    def on_tick(self):
        # self.testt()
        # 如果所有 k 线都没有开始，就开始每一个 tick 下载
        # 如果所有 k 线都已经开始，就每个 tick 检查是否下载完成
        #     如果下载完成，就保存在本地
        #     如果没有下载完成，就继续等下一个 tick 检查
        if not self.all_history_candles_start:
            for trading_pair, candles_info in self.candles.items():
                # 如果没有下载过，就进行下载 candles，这一个 tick 就下载一次
                if not candles_info['start_kline_data']:
                    print(f"开始加载 {trading_pair}")
                    candles_info['candle_history'].start()
                    candles_info['start_kline_data'] = True
                    break
        else:
            if not self.all_history_candles_download:
                for trading_pair, candles_info in self.candles.items():
                    if not candles_info["candle_history"].is_ready:
                        self.logger().info(
                            f"Candles not ready yet for {trading_pair}! Missing {candles_info['candle_history']._candles.maxlen - len(candles_info['candle_history']._candles)}")
                    else:
                        if candles_info["download_kline_data"]:
                            continue
                        df = candles_info["candle_history"].candles_df
                        df = to_offset_candles(df)
                        df.to_csv(candles_info["csv_path"], index=False)
                        print(f'downloaded {trading_pair}')
                        candles_info["candle_history"].stop()
                        candles_info["download_kline_data"] = True
            else:
                current_time = datetime.now()
                print(f'当前时间 {current_time}，开始获取最新数据并进行计算')
                start_time = datetime.now()
                self.increment_kline()
                self.row_candles = self.get_row_df()
                self.df_with_index = self.get_processed_df(self.row_candles)
                self.df_filter_factor = self.calc_factor()
                self.config_data = self.settle_config_data()
                self.trading_result = self.calc_filter_data()

                process_coin = self.get_process_coin()
                balance_df = self.get_balance_df()

                exchange_balance_df = balance_df.loc[balance_df["Exchange"] == self.exchange]
                usdt_balance = exchange_balance_df.loc[exchange_balance_df['Asset'] == 'USDT']["Total Balance"].values[
                    0]
                equity = usdt_balance
                position_df = self.active_positions_df()
                print('process_coin\n', process_coin)
                symbol_order = self.cal_order_amount(process_coin, equity, position_df, leverage=self.strategy_conf['conf']['g_leverage'])

                # process_coin = pd.concat([last_row_eth, last_row_btc, last_row_btc, last_row_doge, last_row_sol])
                if not symbol_order.empty:
                    twap_symbol_info_list = get_twap_symbol_info_list(symbol_order, self.max_one_order_amount)
                    print('拆单信息：\n', twap_symbol_info_list)

                    # =====遍历下单
                    proposal = []
                    for i in range(len(twap_symbol_info_list)):
                        # print(f"twap {i} \n", twap_symbol_info_list[i])
                        orders = self.create_active_order(twap_symbol_info_list[i])
                        proposal.extend(orders)

                    if len(proposal) > 0:
                        adjusted_proposal = self.connector.budget_checker.adjust_candidates(
                            proposal, all_or_none=True)
                        print('adjusted_proposal \n', adjusted_proposal)
                        for order in adjusted_proposal:
                            order_close = PositionAction.CLOSE if order.position_close else PositionAction.OPEN
                            if order.order_side == TradeType.BUY:
                                self.buy(self.exchange, order.trading_pair, order.amount,
                                         order.order_type, Decimal(order.price), position_action=order_close)
                            elif order.order_side == TradeType.SELL:
                                self.sell(self.exchange, order.trading_pair, order.amount,
                                          order.order_type, Decimal(order.price), position_action=order_close)
                else:
                    print('没有需要下单的信息...')

                print("用时：", datetime.now() - start_time)

    def create_active_order(self, symbol_order):
        orders: List[OrderCandidate] = []
        # print('symbol_order\n', symbol_order)
        for symbol, row in symbol_order.iterrows():
            quantity = row['实际下单量']
            if quantity > 0:
                is_bid = True
            else:
                is_bid = False
            # 根据最小递进下单量，进行下单量的调整，取小数
            quantity = self.connector.quantize_order_amount(symbol, Decimal(abs(quantity)))

            mid_price = self.connector.get_mid_price(symbol)
            bid_spread = Decimal(.1)
            ask_spread = Decimal(.1)
            bid_price = mid_price + mid_price * bid_spread * Decimal(.01)
            ask_price = mid_price - mid_price * ask_spread * Decimal(.01)

            price = bid_price if is_bid else ask_price
            price = self.connector.quantize_order_price(symbol, Decimal(price))
            reduce_only = True if row['交易模式'] == '清仓' else False

            trading_rule: TradingRule = self.connector.trading_rules[symbol]
            # 清仓状态不跳过
            if not reduce_only:
                if quantity == s_decimal_0:
                    print(symbol, '交易 quantity 为 0')
                    continue
                elif quantity * price < trading_rule.min_notional_size:
                    print(symbol, '交易金额是小于最小下单金额，跳过该笔交易')
                    print('下单量：', quantity, '价格：', price)
                    continue

            order = PerpetualOrderCandidate(
                trading_pair=symbol,
                is_maker=False,
                order_type=OrderType.LIMIT,
                order_side=TradeType.BUY if is_bid else TradeType.SELL,
                amount=quantity,
                price=price,
                position_close=reduce_only,
                leverage=Decimal(self.strategy_conf['conf']['g_leverage'])
            )
            orders.append(order)
        return orders

    def cal_order_amount(self, process_coin, equity, position_df, leverage):
        strategy_trade_usdt = int(equity * leverage / 3)
        # strategy_trade_usdt = 1000
        process_coin['方向选币数量'] = process_coin.groupby(['candle_begin_time', '方向'])['symbol'].transform(
            'size')  # 计算每个方向的选币数量
        process_coin['目标持仓量'] = strategy_trade_usdt / process_coin['方向选币数量'] / process_coin['close'] * process_coin['方向']  # 计算每个币种的目标持仓量

        # print('process_coin2 \n', process_coin)

        symbol_order = pd.DataFrame(index=list(set(process_coin['symbol']) | set(position_df.index)),
                                    columns=['当前持仓量'])
        symbol_order['当前持仓量'] = position_df['Amount']
        symbol_order['当前持仓量'].fillna(value=0, inplace=True)

        # =目前持仓量当中，可能可以多空合并
        symbol_order['目标持仓量'] = process_coin.groupby('symbol')[['目标持仓量']].sum()
        symbol_order['目标持仓量'].fillna(value=0, inplace=True)

        # print(symbol_order)

        # ===计算实际下单量和实际下单资金
        symbol_order['实际下单量'] = symbol_order['目标持仓量'] - symbol_order['当前持仓量']

        # ===计算下单的模式，清仓、建仓、调仓等
        symbol_order = symbol_order[symbol_order['实际下单量'] != 0]  # 过滤掉实际下当量为0的数据
        # 判断下单数据是否为空
        if symbol_order.empty:  # 如果实际下单数据为空，直接返回空df
            return pd.DataFrame()

        symbol_order.loc[symbol_order['目标持仓量'] == 0, '交易模式'] = '清仓'
        symbol_order.loc[symbol_order['当前持仓量'] == 0, '交易模式'] = '建仓'
        symbol_order['交易模式'].fillna(value='调仓', inplace=True)  # 增加或者减少原有的持仓，不会降为0

        process_coin.sort_values('candle_begin_time', inplace=True)
        symbol_order['close'] = process_coin.groupby('symbol')[['close']].last()
        symbol_order['实际下单资金'] = symbol_order['实际下单量'] * symbol_order['close']
        del symbol_order['close']

        # 补全历史持仓的最新价格信息
        if symbol_order['实际下单资金'].isnull().any():
            nan_symbol = symbol_order.loc[symbol_order['实际下单资金'].isnull(), '实际下单资金'].index
            symbol_last_price = self.get_ticker_data(nan_symbol)
            symbol_order.loc[nan_symbol, '实际下单资金'] = (
                    symbol_order.loc[nan_symbol, '实际下单量'] * symbol_last_price[nan_symbol])

        return symbol_order

    def get_process_coin(self):
        process_coin = pd.DataFrame()
        for symbol, symbol_df in self.trading_result.items():
            tail_df = symbol_df.tail(1)
            # 选取最后一行 in_open_position 的值
            in_open_position = tail_df['in_open_position'].values[0]
            if in_open_position == 1:
                # process_coin 增加一行 eth 和 btc 的最后一行数据
                tail_df.loc[:, '方向'] = 1
                process_coin = pd.concat([process_coin, tail_df])
                for index_symbol in self.index_list:
                    df_symbol = self.row_candles[index_symbol]
                    last_row = df_symbol.tail(1)
                    last_row.loc[:, '方向'] = -1
                    last_row.loc[:, 'symbol'] = index_symbol
                    process_coin = pd.concat([process_coin, last_row])

            elif in_open_position == 0:
                # process_coin 增加一行 eth 和 btc 的最后一行数据
                tail_df.loc[:, '方向'] = 0
                process_coin = pd.concat([process_coin, tail_df])

                for index_symbol in self.index_list:
                    df_symbol = self.row_candles[index_symbol]
                    last_row = df_symbol.tail(1)
                    last_row.loc[:, '方向'] = 0
                    last_row.loc[:, 'symbol'] = index_symbol
                    process_coin = pd.concat([process_coin, last_row])

        # process_coin.drop(columns=['in_open_position'], inplace=True)
        process_coin = process_coin[['candle_begin_time', 'symbol', 'open', 'close', '方向']]
        process_coin.reset_index(drop=True, inplace=True)

        return process_coin

    def increment_kline(self):
        for trading_pair, candles_info in self.candles.items():
            if not candles_info["current_candles"].is_ready:
                self.logger().info(
                    f"current candles not ready yet for {trading_pair}! Missing {candles_info['current_candles']._candles.maxlen - len(candles_info['current_candles']._candles)}")
            else:
                df = candles_info["current_candles"].candles_df
                df = to_offset_candles(df)
                df_hist = pd.read_csv(candles_info["csv_path"], parse_dates=['candle_begin_time'])
                df_full = pd.concat([df, df_hist], ignore_index=True)
                df_full.drop_duplicates(subset=['candle_begin_time'], keep='last', inplace=True)
                df_full.sort_values('candle_begin_time', inplace=True)
                candles_info["candles"] = df_full
                df_full.to_csv(candles_info["csv_path"], mode='w', index=False)
                # print("increment")

    def calc_filter_data(self):
        trading_result_dict = {}
        for symbol_name, conf_data in self.config_data.items():
            waiting_result = self.calc_filter_waiting_data(
                conf_data['waiting'],
                self.strategy_conf['conf'],
                self.strategy_conf['waiting_list_switch']
            )
            trading_result = self.calc_filter_trading_data(
                conf_data['trading'],
                self.strategy_conf['conf'],
                self.strategy_conf['trading_list_switch'],
                waiting_result
            )

            trading_result_dict[symbol_name] = trading_result
        return trading_result_dict

    def calc_filter_waiting_data(self, df, conf, waiting_list_switch):
        check_relative_vol_threshold_True = waiting_list_switch["check_relative_vol_threshold_True"]
        check_relative_vol_threshold_False = waiting_list_switch["check_relative_vol_threshold_False"]
        check_price_change_threshold = waiting_list_switch["check_price_change_threshold"]
        check_relative_price_change_threshold = waiting_list_switch["check_relative_price_change_threshold"]
        check_accu_relative_price_change_threshold = waiting_list_switch[
            "check_accu_relative_price_change_threshold"]
        check_totalamount_condition_0 = waiting_list_switch["check_totalamount_condition_0"]

        g_std_interval = conf["g_std_interval"]
        g_std_period = conf["g_std_period"]
        std_period = int(g_std_period / g_std_interval)
        std_interval = g_std_interval
        g_price_change_threshold = conf["g_price_change_threshold"]
        g_relative_price_change_threshold = conf["g_relative_price_change_threshold"]
        g_accu_relative_price_change_threshold = conf["g_accu_relative_price_change_threshold"]

        df['2-3-X1'] = np.where((df['vol_change'] > df['vol_change_threshold_true'])
                                & check_relative_vol_threshold_True, 1, 0)
        df['2-3-X2'] = np.where((df[f'pct_relative_{std_interval}_std_{std_period}'] > df['vol_change_threshold_false'])
                                & check_relative_vol_threshold_False, 1, 0)
        df['2-3-11'] = np.where(((df['close'] / df['avg_price_5min']) - 1 > g_price_change_threshold)
                                & check_price_change_threshold, 1, 0)
        df['2-3-12'] = np.where((df['pct_change_relative_before'] > g_relative_price_change_threshold)
                                & check_relative_price_change_threshold, 1, 0)
        df['2-3-13A'] = np.where((df['pct_change_relative_accu'] > g_accu_relative_price_change_threshold)
                                 & check_accu_relative_price_change_threshold, 1, 0)
        df['2-3-14'] = np.where((df['quote_volume_calc_period'] > df['quote_volume_history_period_x_times'])
                                & check_totalamount_condition_0, 1, 0)

        total_check = 0
        for value in waiting_list_switch.values():
            if value:
                total_check += 1

        df['wait_check_sum'] = df['2-3-X1'] + df['2-3-X2'] + df['2-3-11'] + df['2-3-12'] + df['2-3-13A'] + df['2-3-14']

        df['enter_waiting_list'] = np.where(df['wait_check_sum'] == total_check, 1, 0)

        df = df[['candle_begin_time', 'symbol', 'conf_name', 'open', 'close', 'quote_volume',
                 'index_capital_curve', 'index_total_quote_amount', 'symbol_relative',
                 '2-3-X1', '2-3-X2', '2-3-11', '2-3-12', '2-3-13A', '2-3-14',
                 'enter_waiting_list',
                 ]]

        df['in_waiting_list'] = df['enter_waiting_list'].shift(0)
        df['in_waiting_list'].replace(0, np.nan, inplace=True)

        g_observed_timeout = int(conf["g_observed_timeout"] / 60)

        df['in_waiting_list'].ffill(limit=g_observed_timeout, inplace=True)
        df['in_waiting_list'].replace(np.nan, 0, inplace=True)

        return df

    def calc_filter_trading_data(self, df, conf, trading_list_switch, waiting_result):
        waiting_result = waiting_result[['candle_begin_time', 'symbol', 'enter_waiting_list', 'in_waiting_list']]
        df = pd.merge(df, waiting_result, on=['candle_begin_time', 'symbol'], how='right')
        check_relative_vol_threshold_True = trading_list_switch["check_relative_vol_threshold_True"]
        check_relative_vol_threshold_False = trading_list_switch["check_relative_vol_threshold_False"]
        check_price_change_threshold = trading_list_switch["check_price_change_threshold"]
        check_relative_price_change_threshold = trading_list_switch["check_relative_price_change_threshold"]
        check_macd_condition = trading_list_switch["check_macd_condition"]
        check_macd_condition_relative = trading_list_switch["check_macd_condition_relative"]
        check_totalamount_condition_0 = trading_list_switch["check_totalamount_condition_0"]
        check_totalamount_condition_1 = trading_list_switch["check_totalamount_condition_1"]
        check_totalamount_condition_2 = trading_list_switch["check_totalamount_condition_2"]

        df['2-5-V1'] = np.where((df['vol_change'] > df['vol_change_threshold_true'])
                                & check_relative_vol_threshold_True, 1, 0)

        g_std_interval = conf["g_std_interval"]
        g_std_period = conf["g_std_period"]
        std_period = int(g_std_period / g_std_interval)
        std_interval = g_std_interval
        df['2-5-V2'] = np.where((df[f'pct_relative_{std_interval}_std_{std_period}']
                                 > df['vol_change_threshold_false'])
                                & check_relative_vol_threshold_False, 1, 0)

        g_price_change_threshold = conf["g_price_change_threshold"]
        df['2-5-11'] = np.where(((df['close'] / df['avg_price_5min']) - 1 > g_price_change_threshold)
                                & check_price_change_threshold, 1, 0)

        g_relative_price_change_threshold = conf["g_relative_price_change_threshold"]
        df['2-5-12'] = np.where((df['pct_change_relative_before'] > g_relative_price_change_threshold)
                                & check_relative_price_change_threshold, 1, 0)

        g_macd_period = conf["g_macd_period"]
        time_period = g_macd_period[0]
        condition_macd_close_1, condition_macd_close_2, condition_macd_close_3 = \
            calc_macd_condition(df, time_period, 'close')
        condition_macd_relative_1, condition_macd_relative_2, condition_macd_relative_3 = \
            calc_macd_condition(df, time_period, 'symbol_relative')

        df['2-5-13A'] = np.where(
            (condition_macd_close_1 | condition_macd_close_2 | condition_macd_close_3)
            & check_macd_condition, 1, 0
        )
        df['2-5-13B'] = np.where(
            (condition_macd_relative_1 | condition_macd_relative_2 | condition_macd_relative_3)
            & check_macd_condition_relative, 1, 0
        )

        params = conf["g_totalamount_check_params"]
        count = 0
        for idx, param in enumerate(params):
            quote_volume_calc_period = param[0]
            quote_volume_x_times = param[2]

            check_totalamount_condition_list = [
                check_totalamount_condition_0, check_totalamount_condition_1, check_totalamount_condition_2
            ]

            df[f'2-5-14_{idx}'] = np.where(
                (df[f'quote_volume_calc_period_{quote_volume_calc_period}'] > df[
                    f'quote_volume_history_period_{quote_volume_x_times}_times'])
                & check_totalamount_condition_list[idx], 1, 0
            )
            count += 1

        df['open_check_sum'] = df['2-5-V1'] + df['2-5-V2'] + df['2-5-11'] + df['2-5-12'] + df['2-5-13A'] + df['2-5-13B']
        for count in range(count):
            df['open_check_sum'] += df[f'2-5-14_{count}']

        df = process_one_line_open_position(df, conf, trading_list_switch)

        df = df[['candle_begin_time', 'symbol', 'open', 'close', 'quote_volume',
                 'index_capital_curve', 'index_total_quote_amount', 'symbol_relative',
                 'conf_name',
                 'enter_waiting_list', 'in_waiting_list', 'in_open_position'
                 ]]
        return df

    def settle_config_data(self):
        config_data = {}
        for symbol_name, df in self.df_with_index.items():
            config_data[symbol_name] = {}
            for factor_df in self.df_filter_factor[symbol_name].values():
                for f in factor_df.columns:
                    df[f] = factor_df[f]

            df_waiting = self.settle_waiting_data(df, self.strategy_conf['conf'])
            df_trading = self.settle_trading_data(df, self.strategy_conf['conf'])
            config_data[symbol_name] = {
                'waiting': df_waiting,
                'trading': df_trading
            }

            self.save_log(df_waiting, f'{symbol_name}_waiting')
            self.save_log(df_trading, f'{symbol_name}_trading')

        return config_data

    def settle_trading_data(self, df, conf):
        conf_df = pd.DataFrame()
        conf_df[self.head_columns] = df[self.head_columns]
        conf_df['symbol'] = df['symbol']
        g_name = conf["g_name"]
        conf_df['conf_name'] = g_name

        g_std_period = conf["g_std_period"]
        g_std_interval = conf["g_std_interval"]
        std_interval = g_std_interval

        g_relative_vol_threshold_period = conf["g_relative_vol_threshold_period"]
        g_relative_vol_threshold_update_period = conf["g_relative_vol_threshold_update_period"]
        g_relative_vol_threshold_coefficient = conf["g_relative_vol_threshold_coefficient"]

        conf_df['vol_change'] = df[f'volatility_change_pl_symbol_relative_fl_[{g_std_period}, {std_interval}]']
        conf_df['vol_change_threshold_true'] = \
            df[
                f'vol_change_threshold_true_pl_symbol_relative_fl_[{g_std_period}, {std_interval}, {g_relative_vol_threshold_period}, {g_relative_vol_threshold_update_period}, {g_relative_vol_threshold_coefficient}]']
        conf_df['vol_change_threshold_false'] = \
            df[
                f'vol_change_threshold_false_pl_symbol_relative_fl_[{g_std_period}, {std_interval}, {g_relative_vol_threshold_period}, {g_relative_vol_threshold_update_period}, {g_relative_vol_threshold_coefficient}]']
        conf_df[f'pct_relative_{std_interval}_std_{int(g_std_period / g_std_interval)}'] = df[
            f'pct_change_std_pl_symbol_relative_fl_[{g_std_period}, {std_interval}]']
        conf_df['avg_price_5min'] = df['rolling_mean_pl_close_fl_5']
        conf_df['pct_change_relative_before'] = df['pct_change_pl_symbol_relative_fl_1']

        g_macd_period = conf["g_macd_period"]
        time_period = g_macd_period[0]
        g_macd_params = conf["g_macd_params"]

        conf_df[f'dif_0_close'] = df[
            f'macd_diff_pl_close_fl_[{time_period}, [{g_macd_params[0]}, {g_macd_params[1]}, {g_macd_params[2]}]]']
        conf_df[f'dif_0_symbol_relative'] = df[
            f'macd_diff_pl_symbol_relative_fl_[{time_period}, [{g_macd_params[0]}, {g_macd_params[1]}, {g_macd_params[2]}]]']

        conf_df[f'dea_0_close'] = df[
            f'macd_dea_pl_close_fl_[{time_period}, [{g_macd_params[0]}, {g_macd_params[1]}, {g_macd_params[2]}]]']
        conf_df[f'dea_0_symbol_relative'] = df[
            f'macd_dea_pl_symbol_relative_fl_[{time_period}, [{g_macd_params[0]}, {g_macd_params[1]}, {g_macd_params[2]}]]']

        params = conf["g_totalamount_check_params"]
        for idx, param in enumerate(params):
            quote_volume_calc_period = param[0]
            quote_volume_history_period = param[1]
            quote_volume_x_times = param[2]

            conf_df[f'quote_volume_calc_period_{quote_volume_calc_period}'] = df[
                f'rolling_sum_pl_quote_volume_fl_{quote_volume_calc_period}']
            conf_df[f'quote_volume_history_period_{quote_volume_x_times}_times'] = df[
                f'quote_volume_history_period_x_times_pl_quote_volume_fl_[{quote_volume_calc_period}, {quote_volume_history_period}, {quote_volume_x_times}]']

        return conf_df

    def settle_waiting_data(self, df, conf):
        # 根据 config 进行数据整理
        conf_df = pd.DataFrame()
        conf_df[self.head_columns] = df[self.head_columns]
        conf_df['symbol'] = df['symbol']
        g_name = conf["g_name"]
        conf_df['conf_name'] = g_name

        g_std_period = conf["g_std_period"]
        g_std_interval = conf["g_std_interval"]
        g_relative_vol_threshold_period = conf["g_relative_vol_threshold_period"]
        g_relative_vol_threshold_update_period = conf["g_relative_vol_threshold_update_period"]
        g_relative_vol_threshold_coefficient = conf["g_relative_vol_threshold_coefficient"]
        g_accu_relative_price_change_period = conf["g_accu_relative_price_change_period"]

        # period = int(g_std_period / g_std_interval)
        std_interval = g_std_interval

        # conf_df[f'pct_relative_{std_interval}'] = df[f'pct_change_pl_symbol_relative_fl_{std_interval}']
        conf_df['vol_change'] = df[f'volatility_change_pl_symbol_relative_fl_[{g_std_period}, {std_interval}]']
        conf_df['vol_change_threshold_true'] = \
            df[
                f'vol_change_threshold_true_pl_symbol_relative_fl_[{g_std_period}, {std_interval}, {g_relative_vol_threshold_period}, {g_relative_vol_threshold_update_period}, {g_relative_vol_threshold_coefficient}]']
        conf_df['vol_change_threshold_false'] = \
            df[
                f'vol_change_threshold_false_pl_symbol_relative_fl_[{g_std_period}, {std_interval}, {g_relative_vol_threshold_period}, {g_relative_vol_threshold_update_period}, {g_relative_vol_threshold_coefficient}]']
        conf_df[f'pct_relative_{std_interval}_std_{int(g_std_period / g_std_interval)}'] = df[
            f'pct_change_std_pl_symbol_relative_fl_[{g_std_period}, {std_interval}]']
        conf_df['avg_price_5min'] = df['rolling_mean_pl_close_fl_5']
        conf_df['pct_change_relative_before'] = df['pct_change_pl_symbol_relative_fl_1']
        conf_df['pct_change_relative_accu'] = df[
            f'pct_change_pl_symbol_relative_fl_{g_accu_relative_price_change_period}']

        params = conf["g_totalamount_check_params"][0]
        quote_volume_calc_period = params[0]
        quote_volume_history_period = params[1]
        quote_volume_x_times = params[2]
        conf_df['quote_volume_calc_period'] = df[f'rolling_sum_pl_quote_volume_fl_{quote_volume_calc_period}']
        conf_df['quote_volume_history_period_x_times'] = df[
            f'quote_volume_history_period_x_times_pl_quote_volume_fl_[{quote_volume_calc_period}, {quote_volume_history_period}, {quote_volume_x_times}]']

        return conf_df

    def calc_factor(self):
        factor_filter_dict = {}
        # for symbol_name, coin_df in self.df_with_index.items():
        #     factor_filter_dict[symbol_name] = {}
        #     if self.multiply_process:
        #         Parallel(n_jobs=9, backend='threading')(
        #             delayed(self.cal_one_filter)(coin_df, filter_factor, factor_filter_dict[symbol_name]) for
        #             filter_factor in self.strategy_conf['filter_factors']
        #         )
        #     else:
        #         for filter_factor in self.strategy_conf['filter_factors']:
        #             self.cal_one_filter(coin_df, filter_factor, factor_filter_dict[symbol_name])

        factor_filter_list = []
        if self.multiply_process:
            factor_filter_list = Parallel(n_jobs=max(os.cpu_count() - 1, 1))(
                delayed(calc_one_df_filter)(symbol_name, coin_df, factor_filter_dict, self.strategy_conf, self.head_columns)
                for symbol_name, coin_df in self.df_with_index.items()
            )
        else:
            for symbol_name, coin_df in self.df_with_index.items():
                d = calc_one_df_filter(symbol_name, coin_df, factor_filter_dict, self.strategy_conf, self.head_columns)
                factor_filter_list.append(d)

        factor_filter_dict = {}
        for item in factor_filter_list:
            factor_filter_dict.update(item)

        return factor_filter_dict

        # return df, class_name, column_name

    def get_processed_df(self, row_candles):
        # print('before calc_all_coin_data')

        index_coin_data = self.calc_all_coin_data(self.index_list, row_candles)
        # print('after calc_all_coin_data')
        index_df = generate_index_df(index_coin_data)

        symbol_coin_data = self.calc_all_coin_data(self.symbol_list, row_candles)
        # print('after calc_all_coin_data')

        merged_df = pd.merge(symbol_coin_data, index_df, on='candle_begin_time', how='left')
        merged_df['symbol_relative'] = merged_df['close'] / merged_df['index_capital_curve']

        grouped = merged_df.groupby('symbol', group_keys=False)
        processed_dict = {}
        for symbol_name, coin_df in grouped:
            coin_df.reset_index(drop=True, inplace=True)
            processed_dict[symbol_name] = coin_df

        return processed_dict

    def calc_all_coin_data(self, _symbol_list, row_candles):

        if self.multiply_process:
            # df_list = []
            # with concurrent.futures.ThreadPoolExecutor(max_workers=max(os.cpu_count() - 1, 1)) as executor:
            #     futures = {executor.submit(self.calc_coin_data, symbol) for symbol in _symbol_list}
            #     for future in concurrent.futures.as_completed(futures):
            #         df_list.append(future.result())
            #         # print(future.result())

            df_list = Parallel(n_jobs=max(os.cpu_count() - 1, 1))(
                delayed(calc_coin_data)(row_candles, symbol) for symbol in _symbol_list
            )
        else:
            df_list = []
            for symbol in _symbol_list:
                df_list.append(calc_coin_data(row_candles, symbol))
        #

        # df_list = []
        # for symbol in _symbol_list:
        #     df_list.append(self.calc_coin_data(symbol))

        all_coin_data = pd.concat(df_list, ignore_index=True)
        all_coin_data.sort_values('candle_begin_time', inplace=True)
        all_coin_data.reset_index(inplace=True, drop=True)
        return all_coin_data

    def get_row_df(self):
        result = {}
        for trading_pair, candles_info in self.candles.items():
            result[trading_pair.split('_')[0]] = candles_info["candles"]
        # if self.all_candles_ready:
        #     for candle in self.candles:
        #         candle_name = candle.name.split("_")[-1]
        #         # print(candle)
        #         df = candle.candles_df.copy()
        #         columns = {
        #             'timestamp': 'candle_begin_time',
        #             'quote_asset_volume': 'quote_volume',
        #             'n_trades': 'trade_num',
        #             'taker_buy_base_volume': 'taker_buy_base_asset_volume',
        #             'taker_buy_quote_volume': 'taker_buy_quote_asset_volume'
        #         }
        #         df.rename(columns=columns, inplace=True)
        #
        #         df['candle_begin_time'] = pd.to_datetime(df['candle_begin_time'], unit='ms')
        #         # # =====获取当前服务器时区，距离UTC 0点的偏差
        #         utc_offset = int(time.localtime().tm_gmtoff / 60 / 60)
        #         # df = df[df['candle_begin_time'] + pd.Timedelta(hours=utc_offset) < self.run_time]
        #         df['candle_begin_time'] = df['candle_begin_time'] + pd.Timedelta(hours=utc_offset)
        #         df.reset_index(drop=True, inplace=True)
        #         result[candle_name] = df

        return result

    def save_log(self, data, name):
        data.to_csv(self.log_data_path + f"_{name}.csv", index=False)

    def get_ticker_data(self, nan_symbol):
        nan_symbol_values = nan_symbol.values
        symbol_price_dict = {
            "symbol": [],
            "price": []
        }
        for i in range(len(nan_symbol_values)):
            symbol_name = nan_symbol_values[i]
            symbol_price_dict["symbol"].append(symbol_name)
            symbol_price = self.connectors[self.exchange].get_mid_price(symbol_name)
            symbol_price_dict["price"].append(symbol_price)

        tickers = pd.DataFrame(symbol_price_dict, dtype=float)
        tickers.set_index('symbol', inplace=True)
        return tickers['price']
