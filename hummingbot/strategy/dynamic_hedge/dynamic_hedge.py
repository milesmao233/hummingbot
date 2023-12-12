import logging
import os
import time
from datetime import datetime
from decimal import Decimal
from typing import List

import numpy as np
import pandas as pd
import pytz

from hummingbot import data_path
from hummingbot.core.clock import Clock
from hummingbot.core.data_type.common import OrderType, PositionMode, PositionAction, TradeType
from hummingbot.core.event.events import BuyOrderCompletedEvent, SellOrderCompletedEvent
from hummingbot.data_feed.candles_feed.candles_factory import CandlesConfig, CandlesFactory
from hummingbot.logger import HummingbotLogger
from hummingbot.strategy.dynamic_hedge.constant import last_candles_num, strategy_conf
from hummingbot.strategy.dynamic_hedge.dynamic_hedge_dingding import HedgeDataDingdingRobot
from hummingbot.strategy.dynamic_hedge.hedge_common_method import HedgeCommonMethod
from hummingbot.strategy.dynamic_hedge.hedge_data_calculator import HedgeDataCalculator
from hummingbot.strategy.dynamic_hedge.hedge_order import HedgeOrder
from hummingbot.strategy.dynamic_hedge.hedge_waiting_trading_calc import HedgeWaitingTradingCalc
from hummingbot.strategy.market_trading_pair_tuple import MarketTradingPairTuple
from hummingbot.strategy.strategy_py_base import StrategyPyBase

hws_logger = None


class DynamicHedge(StrategyPyBase):
    candles_data_path = data_path() + "/candles_data"
    # exchange: str = "binance"

    # max_one_order_amount = 500
    # every_amount = 200

    @classmethod
    def logger(cls) -> HummingbotLogger:
        global hws_logger
        if hws_logger is None:
            hws_logger = logging.getLogger(__name__)
        return hws_logger

    def __init__(self):
        super().__init__()
        self.log_data_path = None
        self.current_loading_candles_symbol = []
        self.history_row_candles_dict = {}

        self._market = None
        self._account_name = None
        self.candles = None
        self._trading_pair_list = None
        self._symbol_list = None
        self._cover_list = None
        self._index_list = None
        self._market_trading_pair_tuples = None
        self._total_amount = 0
        self._every_order_amount = 0
        self._max_one_order_amount = 0
        self._trade_type = None
        self._cover_trigger = None

        self._start_date = ""
        self.interval = "1m"
        self.current_candles_num = 5
        self.fetched_history_data = False
        self.calculated_history_data = False
        self.hedge_data_calculator = None
        self.hedge_waiting_trading_calc = None
        self.last_settle_df_dict = {}
        self.last_index_row_candles = {}
        self.hedge_notify_robot = None
        self.dynamic_hedge_order = None

        # self.in_waiting_list = []
        # self.trading_list_before_calc = []
        # self.in_trading_list = []

    def init_params(self,
                    market_trading_pair_tuples: List[MarketTradingPairTuple],
                    account_name: str,
                    trading_pair_list: List[str],
                    symbol_list: List[str],
                    cover_list: List[str],
                    index_list: List[str],
                    start_date: str,
                    total_amount: int,
                    every_order_amount: int,
                    max_one_order_amount: int,
                    trade_type: str,
                    cover_trigger: bool,
                    dingding_robot_id: str,
                    dingding_secret: str,
                    dingding_waiting_robot_id: str,
                    dingding_waiting_secret: str,
                    ):
        self._market_trading_pair_tuples = market_trading_pair_tuples
        self._account_name = account_name
        self._trading_pair_list = trading_pair_list
        self._symbol_list = symbol_list
        self._cover_list = cover_list
        self._index_list = index_list
        self._start_date = start_date
        self._total_amount = total_amount
        self._every_order_amount = every_order_amount
        self._cover_trigger = cover_trigger,
        self._max_one_order_amount = max_one_order_amount
        self._trade_type = trade_type

        self._market = self._market_trading_pair_tuples[0].market

        self.add_markets([self._market])

        self.hedge_notify_robot = HedgeDataDingdingRobot(
            dingding_robot_id,
            dingding_secret,
            dingding_waiting_robot_id,
            dingding_waiting_secret
        )
        self.dynamic_hedge_order = HedgeOrder(
            self._market,
            self._trade_type,
            self._market_trading_pair_tuples,
            self._account_name,
            self._max_one_order_amount,
            self._every_order_amount,
            self._cover_trigger,
            self.hedge_notify_robot,
        )

        self.log_data_path = data_path() + f"/{self._market.name}_log_data"

    @property
    def account_name(self):
        return self._account_name

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
    def calculate_append_df(append_df, df_history_last_row):
        df = pd.concat([df_history_last_row, append_df], ignore_index=True)
        df.sort_values(by=['candle_begin_time'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        df = HedgeCommonMethod.calculated_last_candles(df)

        return df

    def _start_data_retrieval(self, symbol, process_type):
        """根据处理类型开始数据检索过程，并记录相应的日志。"""

        process_label = '完整历史数据' if process_type == 'candle_history' else '增量历史数据'
        self.logger().info(f'即将获取{process_label}')

        # 启动相应的数据检索过程
        self.candles[symbol][process_type].start()
        self.current_loading_candles_symbol.append((symbol, process_type))

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

                df_append = self.calculate_append_df(df, df_history_last_row)
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

    def all_current_loading_candles_stop(self):
        for symbol, process_type in self.current_loading_candles_symbol:
            self.candles[symbol][process_type].stop()

    def start(self, clock: Clock, timestamp: float) -> None:
        self.candles = {f"{trading_pair}": {} for trading_pair in self._trading_pair_list}
        num_to_download = self.calculate_minute_klines(self._start_date)
        if not os.path.exists(self.candles_data_path):
            os.makedirs(self.candles_data_path)

        for trading_pair in self._trading_pair_list:
            history_candle_config = CandlesConfig(
                connector=self._market.name,
                trading_pair=trading_pair,
                interval=self.interval,
                max_records=num_to_download,
            )
            candle_history = CandlesFactory.get_candle(history_candle_config)
            self.candles[trading_pair]["candle_history"] = candle_history

            four_hour_candle_config = CandlesConfig(
                connector=self._market.name,
                trading_pair=trading_pair,
                interval='4h',
                max_records=60,
            )
            candles_four_hours = CandlesFactory.get_candle(four_hour_candle_config)
            candles_four_hours.start()
            self.candles[trading_pair]["candles_four_hours"] = candles_four_hours

            current_candle_config = CandlesConfig(
                connector=self._market.name,
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
                        connector=self._market.name,
                        trading_pair=trading_pair,
                        interval=self.interval,
                        max_records=time_diff,
                    )

                    increment_candles = CandlesFactory.get_candle(increment_candle_config)
                    self.candles[trading_pair]["increment_candles"] = increment_candles

            self._start_data_retrieval(trading_pair, process_type)
        # self._start_fetch_history_data()

    def tick(self, timestamp: float):
        if self.all_candles_ready and not self.fetched_history_data:
            self.history_row_candles_dict = self._start_fetch_history_data()
            self.all_current_loading_candles_stop()

        elif not self.calculated_history_data and self.fetched_history_data:
            self.logger().info('开始计算历史数据')
            start_time = datetime.now()
            self.hedge_data_calculator = HedgeDataCalculator(symbol_list=self._symbol_list, index_list=self._index_list,
                                                             history_row_candles_dict=self.history_row_candles_dict)
            df_with_index = self.hedge_data_calculator.get_processed_df()
            df_filter_factor = self.hedge_data_calculator.calc_factor(df_with_index)
            last_settle_df_dict = HedgeCommonMethod.settle_config_data(df_with_index, df_filter_factor)
            self.hedge_data_calculator.update_last_settle_df_dict(last_settle_df_dict)

            last_index_row_candles = self.hedge_data_calculator.get_last_index_candles(last_candles_num)
            self.hedge_data_calculator.update_last_index_row_candles(last_index_row_candles)

            for symbol_name, df in last_settle_df_dict.items():
                self.save_log(df, f'{symbol_name}_before_config_data')

            self.hedge_waiting_trading_calc = HedgeWaitingTradingCalc(
                self.last_settle_df_dict,
                self._total_amount,
                self._every_order_amount,
                self.hedge_notify_robot,
                self.account_name,
            )

            end_time = datetime.now()
            self.logger().info(f'计算历史数据完成, 初始化类完成, 耗时 {end_time - start_time}')
            self.calculated_history_data = True

            self._market.set_position_mode(PositionMode.ONEWAY)

        elif self.calculated_history_data and self.fetched_history_data:
            # self.trading_list_before_calc = self.in_trading_list.copy()
            start_time = datetime.now()
            print(f'当前时间 {start_time}，开始获取最新数据并进行计算')
            new_candles_dict = self.get_new_last_df()
            four_hours_candles_dict = self.get_four_hours_df()
            for symbol_name, df in four_hours_candles_dict.items():
                self.save_log(df, f'{symbol_name}_four_hour_data')
            self.hedge_data_calculator.update_new_candles(new_candles_dict)
            self.hedge_data_calculator.update_four_hours_candles(four_hours_candles_dict)

            before_config_dict = self.hedge_data_calculator.get_last_concat_new_df(last_candles_num)
            updated_last_index_candles = self.hedge_data_calculator.calc_updated_last_index_candles()
            self.hedge_data_calculator.update_last_index_row_candles(updated_last_index_candles)
            combined_index_data = self.hedge_data_calculator.update_combined_index_data()
            self.save_log(combined_index_data, 'combined_index_data')

            last_settle_df = self.hedge_data_calculator.calc_last_config_data(before_config_dict)
            self.hedge_data_calculator.update_last_settle_df_dict(last_settle_df)

            self.hedge_waiting_trading_calc.in_trading_list.copy()
            self.hedge_waiting_trading_calc.update_last_settle_df_dict(last_settle_df)
            self.hedge_waiting_trading_calc.calc_trading_position()

            in_waiting_list = self.hedge_waiting_trading_calc.in_waiting_list
            in_trading_list = self.hedge_waiting_trading_calc.in_trading_list

            end_time = datetime.now()
            self.logger().info(f'waiting_list: {in_waiting_list}')
            self.logger().info(f'trading_list: {in_trading_list}')

            self.dynamic_hedge_order.update_in_waiting_list(in_waiting_list)
            self.dynamic_hedge_order.update_in_trading_list(in_trading_list)
            print(f'计算最新数据耗时 {end_time - start_time}')

            proposal = self.dynamic_hedge_order.sell_or_buy()

            if len(proposal) > 0:
                adjusted_proposal = self._market.budget_checker.adjust_candidates(
                    proposal, all_or_none=True)
                self.logger().info(f'adjusted_proposal \n {adjusted_proposal}')
                for order in adjusted_proposal:
                    order_close = PositionAction.CLOSE if order.position_close else PositionAction.OPEN
                    market_pair = self.find_trading_pair_tuple(order.trading_pair)
                    if order.amount * order.price < 10:
                        continue
                    if order.order_side == TradeType.BUY:
                        order_res = self.buy_with_specific_market(market_pair, order.amount, order.order_type,
                                                                  position_action=order_close)

                        print('order_res: \n', order_res)
                    elif order.order_side == TradeType.SELL:
                        order_res = self.sell_with_specific_market(market_pair, order.amount, order.order_type,
                                                                   position_action=order_close)
                        print('order_res: \n', order_res)

            # self.trading_list_before_calc 和 self.in_trading_list 比较
            # 如果存在于 self.trading_list_before_calc 但是不存在于 self.in_trading_list，那么就是需要卖出的
            # 如果存在于 self.in_trading_list 但是不存在于 self.trading_list_before_calc，那么就是需要买入的
            # 如果存在于 self.in_trading_list 且存在于 self.trading_list_before_calc，那么就是不需要操作的

    def get_new_last_df(self):
        new_candles_dict = {}
        for trading_pair, candles_info in self.candles.items():
            symbol_name = trading_pair.split('_')[0]
            if not candles_info["current_candles"].is_ready:
                self.logger().info(
                    f"current candles not ready yet for {trading_pair}! Missing {candles_info['current_candles']._candles.maxlen - len(candles_info['current_candles']._candles)}")
                new_candles_dict[symbol_name] = pd.DataFrame()
            else:
                df = candles_info["current_candles"].candles_df
                df = self.to_offset_candles(df)

                new_candles_dict[symbol_name] = df
        return new_candles_dict

    def get_four_hours_df(self):
        four_hours_candles_dict = {}
        for trading_pair, candles_info in self.candles.items():
            if not candles_info["candles_four_hours"].is_ready:
                self.logger().info(
                    f"four hours candles not ready yet for {trading_pair}! Missing {candles_info['candles_four_hours']._candles.maxlen - len(candles_info['candles_four_hours']._candles)}")
            else:
                symbol_name = trading_pair.split('_')[0]
                df = candles_info["candles_four_hours"].candles_df
                df = self.to_offset_candles(df)

                four_hours_candles_dict[symbol_name] = df
        return four_hours_candles_dict

    def save_log(self, data, name):
        data.to_csv(self.log_data_path + f"_{name}.csv", index=False)

    def find_trading_pair_tuple(self, trading_pair):
        for x in self._market_trading_pair_tuples:
            if x.trading_pair == trading_pair:
                return x

    def did_complete_sell_order(self, order_completed_event: SellOrderCompletedEvent):
        self.logger().info(f'卖单完成：\n {order_completed_event}')
        self.hedge_notify_robot.send_dingding_msg(
            f"account: miles_test \n 拆单卖出完成信号  \n"
            f" 订单编号: {order_completed_event.order_id} \n"
            f" 币种: {order_completed_event.base_asset} - {order_completed_event.quote_asset} \n"
            f" 订单数量: base_asset: {order_completed_event.base_asset_amount}, quote_asset: {order_completed_event.quote_asset_amount} \n"
            f" 订单价格: {order_completed_event.quote_asset_amount / order_completed_event.base_asset_amount} \n",
        )

    def did_complete_buy_order(self, order_completed_event: BuyOrderCompletedEvent):
        self.logger().info(f'买单完成：\n {order_completed_event}')
        self.hedge_notify_robot.send_dingding_msg(
            f"account: miles_test \n 拆单买入完成信号  \n"
            f" 订单编号: {order_completed_event.order_id} \n"
            f" 币种: {order_completed_event.base_asset} - {order_completed_event.quote_asset} \n"
            f" 订单数量: base_asset: {order_completed_event.base_asset_amount}, quote_asset: {order_completed_event.quote_asset_amount} \n"
            f" 订单价格: {order_completed_event.quote_asset_amount / order_completed_event.base_asset_amount} \n",
        )





