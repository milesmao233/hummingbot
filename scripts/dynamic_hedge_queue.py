import os
import base64
import hashlib
import hmac
import json
import time
import requests

import numpy as np
import pandas as pd

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict
from urllib import parse
from joblib import Parallel, delayed

from hummingbot import data_path
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.connector.derivative.position import Position
from hummingbot.core.data_type.common import OrderType, PriceType
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory, CandlesConfig
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase

pd.set_option('expand_frame_repr', False)  # 当列太多时不换行

# 钉钉api
dingding_api = {
    'robot_id': '27f1b4a51df681a65056b700fb2adc7d3df968887c20efd76719bfe759c540b3',
    'secret': 'SEC4c59245a55e3bcdb6842969f0a85aaa71fe3df12de77f92fe531038e84d5bdab',
}


# ===发送钉钉相关函数
# 计算钉钉时间戳
def cal_timestamp_sign(secret):
    # 根据钉钉开发文档，修改推送消息的安全设置https://ding-doc.dingtalk.com/doc#/serverapi2/qf2nxq
    # 也就是根据这个方法，不只是要有robot_id，还要有secret
    # 当前时间戳，单位是毫秒，与请求调用时间误差不能超过1小时
    # python3用int取整
    timestamp = int(round(time.time() * 1000))
    # 密钥，机器人安全设置页面，加签一栏下面显示的SEC开头的字符串
    secret_enc = bytes(secret.encode('utf-8'))
    string_to_sign = '{}\n{}'.format(timestamp, secret)
    string_to_sign_enc = bytes(string_to_sign.encode('utf-8'))
    hmac_code = hmac.new(secret_enc, string_to_sign_enc, digestmod=hashlib.sha256).digest()
    # 得到最终的签名值
    sign = parse.quote_plus(base64.b64encode(hmac_code))
    return str(timestamp), str(sign)


def send_dingding_msg(content, dingding_api):
    """
    :param content:
    :param robot_id:  你的access_token，即webhook地址中那段access_token。
                        例如如下地址：https://oapi.dingtalk.com/robot/send?access_token=81a0e96814b4c8c3132445f529fbffd4bcce66
    :param secret: 你的secret，即安全设置加签当中的那个密钥
    :return:
    """

    robot_id = dingding_api['robot_id']
    secret = dingding_api['secret']

    try:
        msg = {
            "msgtype": "text",
            "text": {"content": content + '\n' + datetime.now().strftime("%m-%d %H:%M:%S")}}
        headers = {"Content-Type": "application/json;charset=utf-8"}
        # https://oapi.dingtalk.com/robot/send?access_token=XXXXXX&timestamp=XXX&sign=XXX
        timestamp, sign_str = cal_timestamp_sign(secret)
        url = 'https://oapi.dingtalk.com/robot/send?access_token=' + robot_id + \
              '&timestamp=' + timestamp + '&sign=' + sign_str
        body = json.dumps(msg)
        requests.post(url, data=body, headers=headers, timeout=10)
        print('成功发送钉钉', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    except Exception as e:
        print("发送钉钉失败:", e, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


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


# def calc_coin_data(row_candles, symbol):
#     # print('self.row_candles', self.row_candles)
#     df = row_candles[symbol]
#     df['symbol'] = symbol
#     df['close'].fillna(method='ffill', inplace=True)
#     df['open'].fillna(value=df['close'], inplace=True)
#     df['high'].fillna(value=df['close'], inplace=True)
#     df['low'].fillna(value=df['close'], inplace=True)
#     df['close_pct_change'] = df['close'].pct_change()
#     df['close_pct_change'].fillna(value=0, inplace=True)
#     df['close_capital_curve'] = (df['close_pct_change'] + 1).cumprod()
#
#     df['open_pct_change'] = df['open'].pct_change()
#     df['open_pct_change'].fillna(value=0, inplace=True)
#     df['open_capital_curve'] = (df['open_pct_change'] + 1).cumprod()
#     df.reset_index(drop=True, inplace=True)
#     return df


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


def calc_one_df_filter(symbol_name, coin_df, factor_filter_dict, strategy_conf, head_columns):
    factor_filter_dict[symbol_name] = {}
    for filter_factor in strategy_conf['filter_factors']:
        cal_one_filter(coin_df, filter_factor, factor_filter_dict[symbol_name], head_columns)

    return factor_filter_dict


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


def concat_two_df_keep_last(df1, df2):
    df = pd.concat([df1, df2], ignore_index=True)
    df.drop_duplicates(subset=['candle_begin_time'], keep='last', inplace=True)
    df.sort_values('candle_begin_time', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


class DynamicHedgeQueue(ScriptStrategyBase):
    symbol_list = [
        'UNFI-USDT', 'GLM-USDT', 'SPELL-USDT',
        'PENDLE-USDT', 'BSW-USDT', 'TWT-USDT',
        'CVC-USDT', 'JOE-USDT', 'GLMR-USDT',
        'HIFI-USDT', 'OXT-USDT', 'CHR-USDT',
        'API3-USDT', 'ATA-USDT', 'KNC-USDT',
        'RUNE-USDT', 'TRX-USDT', 'STMX-USDT',
        'OGN-USDT', 'BLZ-USDT', 'AKRO-USDT',
        'APT-USDT', 'QNT-USDT', 'ZRX-USDT',
        'LPT-USDT', 'XVS-USDT', 'LINA-USDT',
        'LQTY-USDT', 'ENJ-USDT'
    ]
    # symbol_list = ["DOGE-USDT"]

    cover_list = ["ETH-USDT", "BTC-USDT"]
    index_list = ["ETH-USDT", "BTC-USDT"]

    trading_pairs: str = {
        'UNFI-USDT', 'GLM-USDT', 'SPELL-USDT', 'PENDLE-USDT',
        'BSW-USDT', 'TWT-USDT', 'CVC-USDT', 'JOE-USDT',
        'GLMR-USDT', 'HIFI-USDT', 'OXT-USDT', 'CHR-USDT',
        'API3-USDT', 'ATA-USDT', 'KNC-USDT', 'RUNE-USDT',
        'TRX-USDT', 'STMX-USDT', 'OGN-USDT', 'BLZ-USDT',
        'AKRO-USDT', 'APT-USDT', 'QNT-USDT', 'ZRX-USDT',
        'LPT-USDT', 'XVS-USDT', 'LINA-USDT', 'LQTY-USDT',
        'ENJ-USDT', "ETH-USDT", "BTC-USDT",
    }
    # trading_pairs: str = {
    #     "DOGE-USDT", "ETH-USDT", "BTC-USDT",
    # }

    intervals = ["1m"]
    days_to_download = 6
    current_candles_num = 10
    last_candles_num = 3600
    exchange: str = "binance"
    markets = {exchange: trading_pairs}
    log_data_path = data_path() + f"/{exchange}_log_data"
    multiply_process = True
    last_settle_df_dict = {}
    last_index_row_candles = {}
    in_waiting_list = []
    in_trading_list = []
    trading_list_before_calc = []
    observed_symbol_param = {}
    trading_symbol_param = {}
    total_amount = 100
    every_amount = 20
    max_one_order_amount = 500
    price_source = PriceType.MidPrice
    check_position = False

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
    def all_history_candles_start(self):
        is_start_list = [candles_info["start_kline_data"] for candles_info in self.candles.values()]
        return all(is_start_list)

    @property
    def all_history_candles_download(self):
        is_download_list = [candles_info["download_kline_data"] for candles_info in self.candles.values()]
        return all(is_download_list)

    @staticmethod
    def get_max_records(days_to_download: int, interval: str) -> int:
        conversion = {"m": 1, "h": 60, "d": 1440}
        unit = interval[-1]
        quantity = int(interval[:-1])
        return int(days_to_download * 24 * 60 * quantity / conversion[unit])

    def __init__(self, connectors: Dict[str, ConnectorBase]):
        super().__init__(connectors)
        self.connector = self.connectors[self.exchange]
        self.calculated_history_data = False
        combinations = [(trading_pair, interval) for trading_pair in self.trading_pairs for interval in self.intervals]
        self.candles = {f"{combinations[0]}_{combinations[1]}": {} for combinations in combinations}
        for combination in combinations:
            candle_config = CandlesConfig(
                connector=self.exchange,
                trading_pair=combination[0],
                interval=combination[1],
                max_records=self.get_max_records(self.days_to_download, combination[1])
            )
            candle_history = CandlesFactory.get_candle(candle_config)
            self.candles[f"{combination[0]}_{combination[1]}"]["candle_history"] = candle_history
            self.candles[f"{combination[0]}_{combination[1]}"][
                "csv_path"] = data_path() + f"/candles_{self.exchange}_{combination[0]}_{combination[1]}.csv"

            current_candle_config = CandlesConfig(
                connector=self.exchange,
                trading_pair=combination[0],
                interval=combination[1],
                max_records=self.current_candles_num
            )
            current_candles = CandlesFactory.get_candle(current_candle_config)
            current_candles.start()
            self.candles[f"{combination[0]}_{combination[1]}"]["current_candles"] = current_candles
            self.candles[f"{combination[0]}_{combination[1]}"]["download_kline_data"] = False
            self.candles[f"{combination[0]}_{combination[1]}"]["start_kline_data"] = False

    def save_log(self, data, name):
        data.to_csv(self.log_data_path + f"_{name}.csv", index=False)

    def on_tick(self):
        # 如果有历史数据加载没有开始，那么开始加载
        if not self.all_history_candles_start:
            for trading_pair, candles_info in self.candles.items():
                # 如果没有下载过，就进行下载 candles，这一个 tick 就下载一次
                if not candles_info['start_kline_data']:
                    print(f"开始加载 {trading_pair}")
                    candles_info['candle_history'].start()
                    candles_info['start_kline_data'] = True
                    break
        else:
            # 如果有历史数据没有下载完，那么就下载
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
                        df = to_calculated_candles(df, trading_pair.split('_')[0])
                        df.to_csv(candles_info["csv_path"], index=False)
                        print(f'downloaded {trading_pair}')
                        candles_info["candle_history"].stop()
                        candles_info["download_kline_data"] = True
            elif not self.calculated_history_data:
                self.logger().info('开始计算历史数据')
                # 增加容错，将最新的k 线和历史k线进行合并
                new_candles_dict = self.get_new_last_df()
                history_row_candles = self.get_row_history_df(new_candles_dict)

                # calc history data
                df_with_index = self.get_processed_df(history_row_candles)
                df_filter_factor = self.calc_factor(df_with_index)
                self.last_settle_df_dict = self.settle_config_data(df_with_index, df_filter_factor)
                self.last_index_row_candles = self.get_last_index_candles(history_row_candles)

                self.logger().info('计算历史数据完成, 获取计算后的最后两行数据')
                self.calculated_history_data = True
            # 初始化资金仓位
            elif not self.check_position:
                balance_df = self.get_balance_df()

                # lambda x 如果是 USDT的话，直接返回1，
                # 如果不是的话，self.connectors[self.exchange].get_price_by_type(x['Asset'] + '-USDT', self.price_source)
                balance_df['current_price'] = balance_df.apply(
                    lambda x: float(1) if x['Asset'] == 'USDT' else float(
                        self.connectors[self.exchange].get_price_by_type(x['Asset'] + '-USDT', self.price_source)),
                    axis=1)
                balance_df['usdt_value'] = balance_df['current_price'] * balance_df['Total Balance']

                # 找到 usdt_value 大于0 的行的 Asset
                balance_has_value = balance_df[balance_df['usdt_value'] > 10]
                balance_has_value.reset_index(drop=True, inplace=True)
                for asset in balance_has_value.loc[:, 'Asset'].values:
                    if asset == 'USDT':
                        continue
                    asset_name = asset + '-USDT'
                    if asset_name not in self.in_trading_list:
                        self.in_trading_list.append(asset_name)

                self.check_position = True
            else:
                self.trading_list_before_calc = self.in_trading_list.copy()
                current_time = datetime.now()
                print(f'当前时间 {current_time}，开始获取最新数据并进行计算')

                for waiting_symbol in self.in_waiting_list:
                    symbol = self.observed_symbol_param[waiting_symbol]['symbol']
                    enter_time = self.observed_symbol_param[waiting_symbol]['enter_time']
                    if current_time - enter_time > timedelta(minutes=(self.strategy_conf['conf']["g_observed_timeout"] / 60)):
                        self.in_waiting_list.remove(waiting_symbol)
                        # self.observed_symbol_param {} 移除 symbol
                        self.observed_symbol_param.pop(symbol)
                        self.logger().info(f'reason: waiting timeout, {symbol} remove from in_waiting_list, time: {current_time}')

                new_candles_dict = self.get_new_last_df()

                # 新 k 线获取失败的列表
                fetch_failed_list = []
                # 如果 new_candles_dict 中的 key 不在 last_settle_df_dict 中，那么保存进一个 list
                for symbol_name in new_candles_dict.keys():
                    if symbol_name not in self.last_settle_df_dict.keys():
                        fetch_failed_list.append(symbol_name)

                before_config_dict = self.set_new_last_df_factor(new_candles_dict, self.last_settle_df_dict)

                self.last_index_row_candles \
                    = self.update_last_index_candles(new_candles_dict, self.last_index_row_candles)
                update_index_candles_data = self.update_index_candles_data(self.last_index_row_candles)

                # before_config_dict 与 update_index_candles_data 行数相同
                start_time = datetime.now()
                self.last_settle_df_dict = self.settle_last_data(before_config_dict, update_index_candles_data)
                self.calc_trading_position()
                # for symbol, df in self.last_settle_df_dict.items():
                #     self.save_log(df, f'{symbol}_last_data')
                end_time = datetime.now()
                self.logger().info(f'waiting_list: {self.in_waiting_list}')
                self.logger().info(f'trading_list: {self.in_trading_list}')
                print(f'计算最新数据耗时 {end_time - start_time}')

                # self.trading_list_before_calc 和 self.in_trading_list 比较
                # 如果存在于 self.trading_list_before_calc 但是不存在于 self.in_trading_list，那么就是需要卖出的
                # 如果存在于 self.in_trading_list 但是不存在于 self.trading_list_before_calc，那么就是需要买入的
                # 如果存在于 self.in_trading_list 且存在于 self.trading_list_before_calc，那么就是不需要操作的
                if len(self.trading_list_before_calc) > 0:
                    for trading_pair in self.trading_list_before_calc:
                        if trading_pair not in self.in_trading_list:
                            # 卖出
                            self.logger().info(f"sell {trading_pair}")
                            self.sell_with_twap_type(trading_pair, OrderType.MARKET)

                if len(self.in_trading_list) > 0:
                    for trading_pair in self.in_trading_list:
                        if trading_pair not in self.trading_list_before_calc:
                            # 买入
                            self.logger().info(f"buy {trading_pair}")
                            self.buy_with_twap_type(trading_pair, self.every_amount, OrderType.MARKET)

    def sell_with_twap_type(self, trading_pair, order_type):
        # get_balance_trading_pair
        amount = float(self.connectors[self.exchange].get_available_balance(trading_pair.split('-')[0]))
        new_price = float(self.connectors[self.exchange].get_price_by_type(trading_pair, self.price_source))
        balance_usdt_value = amount * new_price
        twap_symbol_amount_list = self.get_twap_symbol_info_list(balance_usdt_value, self.max_one_order_amount, new_price)

        # twap_symbol_amount_list 总数和 amount 进行对比，
        # 如果小于 amount，那么最后一单就加上 amount-twap_symbol_amount_list的和
        # 如果大于 amount，那么最后一单就减去 twap_symbol_amount_list-amount的和
        # twap_symbol_amount_list 的和
        twap_symbol_amount_list_sum = np.sum(twap_symbol_amount_list)
        if twap_symbol_amount_list_sum < amount:
            twap_symbol_amount_list[-1] += amount - twap_symbol_amount_list_sum
        elif twap_symbol_amount_list_sum > amount:
            twap_symbol_amount_list[-1] -= twap_symbol_amount_list_sum - amount

        for amount in twap_symbol_amount_list:
            self.sell(
                connector_name=self.exchange,
                trading_pair=trading_pair,
                amount=Decimal(amount),
                order_type=order_type,
            )
            self.logger().info(f'{amount} {trading_pair} selled')

    def buy_with_twap_type(self, trading_pair, amount, order_type):
        new_price = self.connectors[self.exchange].get_price_by_type(trading_pair, self.price_source)
        twap_symbol_amount_list = self.get_twap_symbol_info_list(amount, self.max_one_order_amount, new_price)
        for amount in twap_symbol_amount_list:
            self.buy(
                connector_name=self.exchange,
                trading_pair=trading_pair,
                amount=Decimal(amount),
                order_type=order_type,
            )
            # 开仓成功 logger
            self.logger().info(f'{amount} {trading_pair} bought')

    @staticmethod
    def get_twap_symbol_info_list(amount, order_amount, new_price):
        if amount > 0:
            sign = 1
        else:
            sign = -1

        # 第二步: 创建一个列表，其中包含多个'order_amount'，数量为'amount'除以'order_amount'的绝对值
        repeated_amounts = [sign * order_amount] * int(abs(amount) / order_amount)

        # 第三步: 计算'amount'除以'order_amount'的余数，并确保余数计算的正确性
        if amount > 0:
            remainder = amount % order_amount
        else:
            remainder = amount % (-order_amount)

        # 第四步: 将余数添加到列表中
        result_list = repeated_amounts + [remainder]

        # 第五步: 如果最后一单小于 20 刀，合并到最后第二单中
        if abs(result_list[-1]) < 20:
            result_list[-2] += result_list[-1]
            result_list.pop()

        # 第六步: 计算每个订单的价格
        result_list = [x / new_price for x in result_list]

        return np.array(result_list)

    @property
    def active_positions(self) -> Dict[str, Position]:

        return self.connector.account_positions

    def active_positions_df(self) -> pd.DataFrame:
        columns = ["Symbol", "Type", "Entry Price", "Amount", "Leverage", "Unrealized PnL"]
        data = []
        self.logger().info(f'active_positions: {self.active_positions.values()}')
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

    def get_row_history_df(self, new_candles_dict):
        result = {}
        for trading_pair, candles_info in self.candles.items():
            symbol_name = trading_pair.split('_')[0]
            df_hist = pd.read_csv(candles_info["csv_path"], parse_dates=['candle_begin_time'])
            new_df = new_candles_dict[symbol_name]
            # concat df_hist and new_df
            df_hist = pd.concat([df_hist, new_df], ignore_index=True)

            # 计算 nan 的数据
            df_hist = self.calculated_last_candles(df_hist)
            # drop duplicates
            df_hist.drop_duplicates(subset=['candle_begin_time'], keep='last', inplace=True)
            df_hist.sort_values('candle_begin_time', inplace=True)
            df_hist.reset_index(drop=True, inplace=True)
            # save df_hist
            df_hist.to_csv(candles_info["csv_path"], index=False)

            result[symbol_name] = df_hist

        return result

    def get_processed_df(self, row_candles):
        index_coin_data = calc_all_coin_data(self.index_list, row_candles)

        index_df = generate_index_df(index_coin_data)

        symbol_coin_data = calc_all_coin_data(self.symbol_list, row_candles)

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

    def settle_config_data(self, df_with_index, df_filter_factor):
        config_data_last_df = {}
        for symbol_name, df in df_with_index.items():
            for factor_df in df_filter_factor[symbol_name].values():
                for f in factor_df.columns:
                    df[f] = factor_df[f]

                config_data_last_df[symbol_name] = df.tail(self.last_candles_num)

                self.save_log(df, f'{symbol_name}_config_data')

        return config_data_last_df

    def get_last_index_candles(self, history_row_candles):
        last_index_row_candles = {}
        for symbol_name in self.index_list:
            last_index_row_candles[symbol_name] = history_row_candles[symbol_name].tail(self.last_candles_num)

        return last_index_row_candles

    def calculated_last_candles(self, df_hist):
        tail_df = df_hist.tail(self.current_candles_num + 1)
        tail_df.reset_index(drop=True, inplace=True)
        tail_df = _calculated_last_candles(tail_df)

        # 合并 df_hist 和 tail_df
        df_hist = pd.concat([df_hist, tail_df], ignore_index=True)
        df_hist.drop_duplicates(subset=['candle_begin_time'], keep='last', inplace=True)
        df_hist.sort_values('candle_begin_time', inplace=True)
        df_hist.reset_index(drop=True, inplace=True)

        return df_hist

    def get_new_last_df(self):
        new_candles_dict = {}
        for trading_pair, candles_info in self.candles.items():
            if not candles_info["current_candles"].is_ready:
                self.logger().info(
                    f"current candles not ready yet for {trading_pair}! Missing {candles_info['current_candles']._candles.maxlen - len(candles_info['current_candles']._candles)}")
            else:
                symbol_name = trading_pair.split('_')[0]
                df = candles_info["current_candles"].candles_df
                df = to_offset_candles(df)
                # df = to_calculated_candles(df, symbol_name)
                new_candles_dict[symbol_name] = df
        return new_candles_dict

    def update_last_index_candles(self, new_candles_dict, last_index_row_candles):
        updated_last_index_candles = {}
        for symbol_name, index_df in last_index_row_candles.items():
            new_df = new_candles_dict[symbol_name]
            df = concat_two_df_keep_last(index_df, new_df)

            # self_log
            df_new = df.tail(self.current_candles_num + 1)
            df_new.reset_index(drop=True, inplace=True)
            df_new = _calculated_last_candles(df_new)

            df = concat_two_df_keep_last(index_df, df_new)

            # df 只保留最后 last_candles_num 行
            df = df.tail(self.last_candles_num)
            df.reset_index(drop=True, inplace=True)
            updated_last_index_candles[symbol_name] = df

        return updated_last_index_candles

    @staticmethod
    def update_index_candles_data(last_index_row_candles):
        df_list = []
        for df in last_index_row_candles.values():
            df_list.append(df)

        all_coin_data = pd.concat(df_list, ignore_index=True)
        all_coin_data.sort_values('candle_begin_time', inplace=True)
        all_coin_data.reset_index(inplace=True, drop=True)

        index_df = generate_index_df(all_coin_data)

        return index_df

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

    def calc_trading_position(self):
        conf = self.strategy_conf['conf']
        trading_list_switch = self.strategy_conf["trading_list_switch"]
        for symbol, df in self.last_settle_df_dict.items():
            # after_df 最后几行的数据
            last_row = df.tail(1)
            last_two_row2 = df.tail(2)
            last_period_rows = df.tail(500)
            waiting_df = self.settle_waiting_data(last_two_row2)
            waiting_result = self.calc_filter_waiting_data(waiting_df)

            if waiting_result.iloc[-1]['enter_waiting_list'] == 1 and symbol not in self.in_waiting_list:
                enter_time = waiting_result.iloc[-1]['candle_begin_time']
                self.in_waiting_list.append(symbol)
                self.observed_list_append(symbol, enter_time, last_two_row2)
                send_dingding_msg(
                    f"进入 waiting_list 信号: {symbol}  \n  当前 waiting_list: {self.in_waiting_list} \n  当前 trading_list: {self.in_trading_list}",
                    dingding_api)

            if symbol in self.in_waiting_list:
                # 获取 symbol_relative 的值
                symbol_relative = last_row.iloc[-1]['symbol_relative']
                current_close = last_row.iloc[-1]['close']
                # if (row['symbol_relative'] / trading_status['base_price_relative'] - 1) < 0:
                if symbol_relative / self.observed_symbol_param[symbol]['base_price_relative'] - 1 < 0:
                    # 更新 base_price 的值
                    self.observed_symbol_param[symbol]['base_price'] = last_row.iloc[-1]['close']
                    self.observed_symbol_param[symbol]['base_index'] = last_row.iloc[-1]['index_capital_curve']
                    self.observed_symbol_param[symbol]['base_price_relative'] = last_row.iloc[-1]['close'] / \
                                                                           last_row.iloc[-1]['index_capital_curve']

                trading_df = self.settle_trading_data(last_two_row2)
                trading_df_macd = self.settle_trading_macd_data(last_period_rows)
                trading_result_before_stop_loss = self.calc_filter_trading_data(trading_df, trading_df_macd)
                open_check_sum = trading_result_before_stop_loss.iloc[-1]['open_check_sum']

                accu_price_change = current_close / self.observed_symbol_param[symbol]['base_price'] - 1
                g_accu_change_threshold = conf["g_accu_change_threshold"]
                check_accu_change_threshold = trading_list_switch["check_accu_change_threshold"]

                if accu_price_change > g_accu_change_threshold and check_accu_change_threshold:
                    open_check_sum += 1

                total_check = 0
                for value in trading_list_switch.values():
                    if value:
                        total_check += 1

                if open_check_sum == total_check:
                    max_trading_symbol = int(self.total_amount / self.every_amount)
                    if len(self.in_trading_list) < max_trading_symbol:
                        self.in_trading_list.append(symbol)
                        self.trading_symbol_param[symbol] = {}
                        self.trading_symbol_param[symbol]['symbol'] = symbol
                        self.trading_symbol_param[symbol]['enter_time'] = last_row.iloc[-1]['candle_begin_time']
                        self.trading_symbol_param[symbol]['open_symbol_relative'] = last_row.iloc[-1]['symbol_relative']
                        # 移除 in_waiting_list
                        self.in_waiting_list.remove(symbol)
                        self.observed_symbol_param.pop(symbol)

                        send_dingding_msg(
                            f"进入 trading_list 信号: {symbol}  \n  当前 waiting_list: {self.in_waiting_list} \n  当前 trading_list: {self.in_trading_list}",
                            dingding_api)
                    else:
                        self.logger().info(f'超过最大持仓数量 {max_trading_symbol}, 未加入 symbol: {symbol}')

                # 如果在 waiting_list 超时，移除 symbol
                current_time = waiting_result.iloc[-1]['candle_begin_time']
                enter_time = self.observed_symbol_param[symbol]['enter_time']
                if current_time - enter_time > timedelta(minutes=(conf["g_observed_timeout"] / 60)):
                    self.in_waiting_list.remove(symbol)
                    self.observed_symbol_param.pop(symbol)
                    self.logger().info(f'超时 {symbol} remove from in_waiting_list')

            if symbol in self.in_trading_list:
                # 获取 symbol_relative 的值
                current_time = last_row.iloc[-1]['candle_begin_time']
                symbol_relative = last_row.iloc[-1]['symbol_relative']
                symbol_relative_pct_change = symbol_relative / self.trading_symbol_param[symbol]['open_symbol_relative'] - 1

                g_stop_loss_relative_change_threshold = conf['g_stop_loss_relative_change_threshold']
                g_opened_timeout = conf['g_opened_timeout']

                # 止损平仓 或 超时平仓，平仓后要从 waiting_list 移除
                if symbol_relative_pct_change < g_stop_loss_relative_change_threshold \
                        or current_time - self.trading_symbol_param[symbol]['enter_time'] > timedelta(
                    minutes=g_opened_timeout):
                    self.in_trading_list.remove(symbol)
                    self.logger().info(f'{symbol} remove from in_trading_list')
                    self.in_waiting_list.remove(symbol)
                    self.observed_symbol_param.pop(symbol)
                    self.logger().info(f'{symbol} remove from in_waiting_list')

                    send_dingding_msg(
                        f"移除 trading_list 信号: {symbol}  \n  止损平仓 或 超时平仓 \n 当前 waiting_list: {self.in_waiting_list} \n  当前 trading_list: {self.in_trading_list}",
                        dingding_api)

                # 追踪止盈
                # 记录累积的最大相对涨幅
                g_take_profit_relative_change_trigger_threshold = conf[
                    'g_take_profit_relative_change_trigger_threshold']
                g_take_profit_relative_change_down_threshold = conf['g_take_profit_relative_change_down_threshold']

                if symbol_relative_pct_change > self.trading_symbol_param[symbol]['max_symbol_relative_pct_change']:
                    self.trading_symbol_param[symbol]['max_symbol_relative_pct_change'] = symbol_relative_pct_change

                    # 触发追踪止盈的 trigger
                    if self.trading_symbol_param[symbol]['max_symbol_relative_pct_change'] > g_take_profit_relative_change_trigger_threshold:
                        self.trading_symbol_param[symbol]['take_profit_trigger'] = True

                # 如果触发后，回落了 0.02，就追踪止盈，放入 waiting_list 中
                if self.trading_symbol_param[symbol]['take_profit_trigger'] and ((self.trading_symbol_param[symbol][
                                                                                 'max_symbol_relative_pct_change'] - symbol_relative_pct_change) > g_take_profit_relative_change_down_threshold):
                    self.in_trading_list.remove(symbol)
                    self.in_waiting_list.append(symbol)
                    enter_time = last_row.iloc[-1]['candle_begin_time']
                    self.observed_list_append(symbol, enter_time, last_two_row2)

                    send_dingding_msg(
                        f"移除 trading_list 信号: {symbol}  \n  追踪止盈 放入 waiting_list \n 当前 waiting_list: {self.in_waiting_list} \n  当前 trading_list: {self.in_trading_list}",
                        dingding_api)

        # return in_trading_list

    def observed_list_append(self, symbol, enter_time, last_two_row2):
        self.observed_symbol_param[symbol] = {}
        self.observed_symbol_param[symbol]['symbol'] = symbol
        self.observed_symbol_param[symbol]['enter_time'] = enter_time
        conf = self.strategy_conf['conf']

        base_price_period = int(conf["g_base_price_period_before_waiting_list"])
        # 获得 -1 - base_price_period 的close数据，
        base_close_price_list = last_two_row2.iloc[-1 - base_price_period:]['close'].values
        base_index_price_list = last_two_row2.iloc[-1 - base_price_period:]['index_capital_curve'].values

        close_previous = base_close_price_list[0]
        index_previous = base_index_price_list[0]
        close_current = base_close_price_list[1]
        index_current = base_index_price_list[1]
        if (close_previous / index_previous) < (close_current / index_current):
            self.observed_symbol_param[symbol]['base_price'] = close_previous
            self.observed_symbol_param[symbol]['base_index'] = index_previous
            self.observed_symbol_param[symbol]['base_price_relative'] = close_previous / index_previous
        else:
            self.observed_symbol_param[symbol]['base_price'] = close_current
            self.observed_symbol_param[symbol]['base_index'] = index_current
            self.observed_symbol_param[symbol]['base_price_relative'] = close_current / index_current

    def settle_waiting_data(self, df):
        conf = self.strategy_conf['conf']
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

    def calc_filter_waiting_data(self, df):
        conf = self.strategy_conf['conf']
        waiting_list_switch = self.strategy_conf["waiting_list_switch"]
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

        return df

    def settle_trading_data(self, df):
        conf = self.strategy_conf['conf']
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

    def settle_trading_macd_data(self, df):
        conf = self.strategy_conf['conf']
        conf_df = pd.DataFrame()
        conf_df[self.head_columns] = df[self.head_columns]
        conf_df['symbol'] = df['symbol']
        g_name = conf["g_name"]
        conf_df['conf_name'] = g_name

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

        return conf_df

    def calc_filter_trading_data(self, df, macd_df):
        conf = self.strategy_conf['conf']
        trading_list_switch = self.strategy_conf["trading_list_switch"]
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
            self.calc_macd_condition(macd_df, time_period, 'close')
        condition_macd_relative_1, condition_macd_relative_2, condition_macd_relative_3 = \
            self.calc_macd_condition(macd_df, time_period, 'symbol_relative')

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

        return df

    @staticmethod
    def calc_macd_condition(df, period, price_name):
        last_row = df.iloc[-1]
        previous_row = df.iloc[-(1 + period)]
        previous_row_2 = df.iloc[-(1 + period * 2)]

        # 计算条件
        dif_condition1_last = (last_row[f'dif_0_{price_name}'] - last_row[f'dea_0_{price_name}']) > (
                    previous_row[f'dif_0_{price_name}'] - previous_row[f'dea_0_{price_name}'])
        dif_condition2_last = (last_row[f'dif_0_{price_name}'] - last_row[f'dea_0_{price_name}']) > (
                    previous_row_2[f'dif_0_{price_name}'] - previous_row_2[f'dea_0_{price_name}'])

        condition_macd_1_last = (last_row[f'dif_0_{price_name}'] >= 0) & dif_condition1_last
        condition_macd_2_last = (last_row[f'dif_0_{price_name}'] >= 0) & dif_condition2_last
        condition_macd_3_last = (last_row[f'dif_0_{price_name}'] < 0) & dif_condition1_last & dif_condition2_last

        return condition_macd_1_last, condition_macd_2_last, condition_macd_3_last

    def settle_last_data(self, before_config_dict, update_index_candles_data):
        # show_df = before_config_dict['DOGE-USDT']
        # self.logger().info(f'before_config_dict: \n{show_df}')
        # # save log
        # self.save_log(show_df, 'doge_before_config_data')
        #
        # self.logger().info(f'update_index_candles_data: \n{update_index_candles_data}')
        # self.save_log(update_index_candles_data, 'update_index_candles_data')

        after_config_dict = {}
        for symbol, df in before_config_dict.items():
            # 循环 df
            # 填充最后3行新的数据
            after_df = self.calc_last_df(symbol, df, update_index_candles_data)
            after_df = after_df.tail(self.last_candles_num)
            after_df.reset_index(drop=True, inplace=True)

            after_config_dict[symbol] = after_df

        return after_config_dict

    def calc_last_df(self, symbol, df, _update_index_candles_data):
        index_row_candles = _update_index_candles_data.copy().set_index('candle_begin_time')
        df = self.update_base_data(df, symbol, index_row_candles)
        df = self.update_first_cycle_data(df)
        df = self.update_second_cycle_data(df)
        df = self.update_third_cycle_data(df)

        return df

    def update_base_data(self, df, symbol, index_row_candles):
        # # 将 last_index_row_candles 变成以 candle_begin_time 为 index
        # index_row_candles = _update_index_candles_data.copy().set_index('candle_begin_time')

        for i in range(len(df) - (self.current_candles_num + 2), len(df)):
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
        for i in range(len(df) - 5, len(df)):
            current_row = df.iloc[i]
            last_row = df.iloc[i - 1]
            candle_time = current_row['candle_begin_time']

            for filter_factor in self.strategy_conf['filter_factors']:
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

    def update_second_cycle_data(self, df):
        for i in range(len(df) - 5, len(df)):
            current_row = df.iloc[i]
            last_row = df.iloc[i - 1]
            candle_time = current_row['candle_begin_time']

            for filter_factor in self.strategy_conf['filter_factors']:
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
                        quote_volume_calc_period = int(params[0])
                        quote_volume_history_period = int(params[1])
                        quote_volume_x_times = int(params[2])

                        previous_sum_mean = df.iloc[
                                            i - quote_volume_calc_period - quote_volume_history_period + 1:i - quote_volume_calc_period + 1][
                            f'rolling_sum_pl_{column_name}_fl_{quote_volume_calc_period}'].mean()
                        previous_sum_mean_x_times = previous_sum_mean * quote_volume_x_times

                        df.loc[i, f'{filter_name}_pl_{column_name}_fl_{params}'] = previous_sum_mean_x_times

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
        for i in range(len(df) - 5, len(df)):
            current_row = df.iloc[i]
            last_row = df.iloc[i - 1]
            candle_time = current_row['candle_begin_time']

            for filter_factor in self.strategy_conf['filter_factors']:
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
