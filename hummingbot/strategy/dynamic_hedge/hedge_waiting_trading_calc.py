import logging
from datetime import timedelta
from logging import Logger

import numpy as np
import pandas as pd

from hummingbot.logger import HummingbotLogger
from hummingbot.strategy.dynamic_hedge.constant import head_columns, strategy_conf

hws_logger = None


def settle_waiting_data(df):
    conf = strategy_conf['conf']
    conf_df = pd.DataFrame()
    conf_df[head_columns] = df[head_columns]
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


def calc_filter_waiting_data(df):
    conf = strategy_conf['conf']
    waiting_list_switch = strategy_conf["waiting_list_switch"]
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


def settle_trading_data(df):
    conf = strategy_conf['conf']
    conf_df = pd.DataFrame()
    conf_df[head_columns] = df[head_columns]
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


def settle_trading_macd_data(df):
    conf = strategy_conf['conf']
    conf_df = pd.DataFrame()
    conf_df[head_columns] = df[head_columns]
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


def calc_filter_trading_data(df, macd_df):
    conf = strategy_conf['conf']
    trading_list_switch = strategy_conf["trading_list_switch"]
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
        calc_macd_condition(macd_df, time_period, 'close')
    condition_macd_relative_1, condition_macd_relative_2, condition_macd_relative_3 = \
        calc_macd_condition(macd_df, time_period, 'symbol_relative')

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


class HedgeWaitingTradingCalc:
    _in_waiting_list = []
    _in_trading_list = ['MTL-USDT']
    observed_symbol_param = {}
    trading_symbol_param = {}

    # total_amount = 1000
    # every_amount = 200

    @classmethod
    def logger(cls) -> HummingbotLogger:
        global hws_logger
        if hws_logger is None:
            hws_logger = logging.getLogger(__name__)
        return hws_logger

    def __init__(self, last_settle_df_dict, total_amount, every_order_amount, hedge_notify_robot, account_name):
        self._last_settle_df_dict = last_settle_df_dict
        self._total_amount = total_amount
        self._every_order_amount = every_order_amount
        self.hedge_notify_robot = hedge_notify_robot
        self.account_name = account_name
        self.waiting_result_dict = {}

    @property
    def in_trading_list(self):
        return self._in_trading_list

    @property
    def in_waiting_list(self):
        return self._in_waiting_list

    def update_last_settle_df_dict(self, new_df_dict):
        self._last_settle_df_dict = new_df_dict

    def calc_trading_position(self):
        conf = strategy_conf['conf']
        trading_list_switch = strategy_conf["trading_list_switch"]

        for symbol, df in self._last_settle_df_dict.items():
            # after_df 最后几行的数据
            last_row = df.tail(1)
            last_two_row2 = df.tail(2)
            last_period_rows = df.tail(500)
            waiting_df = settle_waiting_data(last_two_row2)
            waiting_result = calc_filter_waiting_data(waiting_df)

            if (waiting_result.iloc[-1][
                'enter_waiting_list'] == 1 and symbol not in self._in_waiting_list and symbol not
                    in self._in_trading_list):
                enter_time = waiting_result.iloc[-1]['candle_begin_time']
                self._in_waiting_list.append(symbol)
                self.observed_list_append(symbol, enter_time, last_two_row2)
                self.waiting_result_dict[symbol] = {}

                self.hedge_notify_robot.send_dingding_waiting_msg(
                    f'account: {self.account_name} \n '
                    f'进入 waiting_list 信号: {symbol}  \n'
                    f'当前 waiting_list: {self.in_waiting_list} \n'
                    f'当前 trading_list: {self.in_trading_list}'
                )

            if symbol in self._in_waiting_list:
                self.waiting_result_dict[symbol]['waiting_result'] = waiting_result
                self.waiting_result_dict[symbol]['last_row'] = last_row
                self.waiting_result_dict[symbol]['last_two_row2'] = last_two_row2
                self.waiting_result_dict[symbol]['last_period_rows'] = last_period_rows

                # self.logger().info(
                #     f'account: {self.account_name} \n 进入 waiting_list 信号: {symbol}  \n 当前 waiting_list: {self.in_waiting_list} \n 当前 trading_list: {self.in_trading_list}')

        for symbol in self._in_waiting_list:
            waiting_result = self.waiting_result_dict[symbol]['waiting_result']
            last_row = self.waiting_result_dict[symbol]['last_row']
            last_two_row2 = self.waiting_result_dict[symbol]['last_two_row2']
            last_period_rows = self.waiting_result_dict[symbol]['last_period_rows']
            current_time = waiting_result.iloc[-1]['candle_begin_time']
            enter_time = self.observed_symbol_param[symbol]['enter_time']

            if current_time - enter_time > timedelta(minutes=(conf["g_observed_timeout"] / 60)):
                self._in_waiting_list.remove(symbol)
                self.observed_symbol_param.pop(symbol)
                self.waiting_result_dict.pop(symbol)
                # self.logger().info(f'超时 {symbol} remove from in_waiting_list')
                self.hedge_notify_robot.send_dingding_waiting_msg(
                    f"account: {self.account_name} \n "
                    f"超时离开 waiting_list 信号: {symbol}  \n"
                    f"当前 waiting_list: {self.in_waiting_list} \n"
                    f"当前 trading_list: {self.in_trading_list}",
                )
                continue

            symbol_relative = last_row.iloc[-1]['symbol_relative']
            current_close = last_row.iloc[-1]['close']
            if symbol_relative / self.observed_symbol_param[symbol]['base_price_relative'] - 1 < 0:
                # 更新 base_price 的值
                self.observed_symbol_param[symbol]['base_price'] = last_row.iloc[-1]['close']
                self.observed_symbol_param[symbol]['base_index'] = last_row.iloc[-1]['index_capital_curve']
                self.observed_symbol_param[symbol]['base_price_relative'] = (
                        last_row.iloc[-1]['close'] / last_row.iloc[-1]['index_capital_curve'])

            trading_df = settle_trading_data(last_two_row2)
            trading_df_macd = settle_trading_macd_data(last_period_rows)
            trading_result_before_stop_loss = calc_filter_trading_data(trading_df, trading_df_macd)
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
                max_trading_symbol = int(self._total_amount / self._every_order_amount)
                if len(self._in_trading_list) < max_trading_symbol:
                    self._in_trading_list.append(symbol)
                    self.trading_symbol_param[symbol] = {}
                    self.trading_symbol_param[symbol]['symbol'] = symbol
                    self.trading_symbol_param[symbol]['enter_time'] = last_row.iloc[-1]['candle_begin_time']
                    self.trading_symbol_param[symbol]['open_symbol_relative'] = last_row.iloc[-1]['symbol_relative']
                    self.trading_symbol_param[symbol]['max_symbol_relative_pct_change'] = 0
                    self.trading_symbol_param[symbol]['take_profit_trigger'] = False
                    self.trading_symbol_param[symbol]['take_profit_trigger_time'] = None
                    # 移除 in_waiting_list
                    self._in_waiting_list.remove(symbol)
                    self.observed_symbol_param.pop(symbol)

                    self.hedge_notify_robot.send_dingding_msg(
                        f"account: {self.account_name} \n"
                        f"进入 trading_list 信号: {symbol} \n "
                        f"当前 waiting_list: {self.in_waiting_list} \n "
                        f"当前 trading_list: {self.in_trading_list}",
                    )
                else:
                    self.logger().info(f'超过最大持仓数量 {max_trading_symbol}, 未加入 symbol: {symbol}')

        for symbol in self._in_trading_list:
            last_row = self.waiting_result_dict[symbol]['last_row']
            last_two_row2 = self.waiting_result_dict[symbol]['last_two_row2']

            # 获取 symbol_relative 的值
            current_time = last_row.iloc[-1]['candle_begin_time']
            symbol_relative = last_row.iloc[-1]['symbol_relative']
            symbol_relative_pct_change = symbol_relative / self.trading_symbol_param[symbol][
                'open_symbol_relative'] - 1

            g_stop_loss_relative_change_threshold = conf['g_stop_loss_relative_change_threshold']
            g_opened_timeout = conf['g_opened_timeout']

            self.logger().info(f'{symbol} trading_list 数据: ')
            self.logger().info(f'{symbol} last_row: {last_row}')
            self.logger().info(f'{symbol} self.trading_symbol_param[symbol]: {self.trading_symbol_param[symbol]}')
            self.logger().info(f'{symbol} symbol_relative_pct_change: {symbol_relative_pct_change}')

            # 止损平仓
            if symbol_relative_pct_change < g_stop_loss_relative_change_threshold:
                self._in_trading_list.remove(symbol)
                self.waiting_result_dict.pop(symbol)

                self.logger().info(f'{symbol} remove from in_trading_list 止损平仓')
                # self.in_waiting_list.remove(symbol)
                # self.observed_symbol_param.pop(symbol)
                # self.logger().info(f'{symbol} remove from in_waiting_list')

                self.hedge_notify_robot.send_dingding_msg(
                    f"account: {self.account_name} \n"
                    f"移除 trading_list 信号: {symbol} 止损平仓 \n "
                    f"当前 waiting_list: {self.in_waiting_list} \n "
                    f"当前 trading_list: {self.in_trading_list}",
                )

            # 超时平仓
            if (self.trading_symbol_param[symbol]['take_profit_trigger_time'] is not None and
                    current_time - self.trading_symbol_param[symbol]['take_profit_trigger_time'] > timedelta(
                        minutes=g_opened_timeout)):
                self._in_trading_list.remove(symbol)
                self.waiting_result_dict.pop(symbol)

                self.logger().info(f'{symbol} remove from in_trading_list 触发止盈标记后超时平仓')
                self.hedge_notify_robot.send_dingding_msg(
                    f"account: {self.account_name} \n"
                    f"移除 trading_list 信号: {symbol} 触发止盈标记后超时平仓 \n "
                    f"当前 waiting_list: {self.in_waiting_list} \n "
                    f"当前 trading_list: {self.in_trading_list}",
                )

            # 追踪止盈
            # 记录累积的最大相对涨幅
            g_take_profit_relative_change_trigger_threshold = conf[
                'g_take_profit_relative_change_trigger_threshold']
            g_take_profit_relative_change_down_threshold = conf['g_take_profit_relative_change_down_threshold']

            if symbol_relative_pct_change > self.trading_symbol_param[symbol]['max_symbol_relative_pct_change']:
                self.trading_symbol_param[symbol]['max_symbol_relative_pct_change'] = symbol_relative_pct_change

                # 触发追踪止盈的 trigger
                if self.trading_symbol_param[symbol][
                    'max_symbol_relative_pct_change'] > g_take_profit_relative_change_trigger_threshold:
                    self.trading_symbol_param[symbol]['take_profit_trigger'] = True
                    self.trading_symbol_param[symbol]['take_profit_trigger_time'] = last_row.iloc[-1][
                        'candle_begin_time']

            # 如果触发后，回落了 0.02，就追踪止盈，放入 waiting_list 中
            if (self.trading_symbol_param[symbol]['take_profit_trigger'] and
                    ((self.trading_symbol_param[symbol]['max_symbol_relative_pct_change'] - symbol_relative_pct_change) > g_take_profit_relative_change_down_threshold)):
                self._in_trading_list.remove(symbol)
                self._in_waiting_list.append(symbol)
                enter_time = last_row.iloc[-1]['candle_begin_time']
                self.observed_list_append(symbol, enter_time, last_two_row2)
                self.waiting_result_dict[symbol] = {}

                self.hedge_notify_robot.send_dingding_waiting_msg(
                    f"account: {self.account_name} \n "
                    f"追踪止盈 放入 waiting_list : {symbol} \n "
                    f"当前 waiting_list: {self.in_waiting_list} \n  "
                    f"当前 trading_list: {self.in_trading_list}",
                )

                self.hedge_notify_robot.send_dingding_msg(
                    f"account: {self.account_name} \n"
                    f"移除 trading_list 信号: {symbol} \n "
                    f"追踪止盈 放入 waiting_list \n "
                    f"当前 waiting_list: {self.in_waiting_list} \n "
                    f"当前 trading_list: {self.in_trading_list}",
                )

    def observed_list_append(self, symbol, enter_time, last_two_row2):
        self.observed_symbol_param[symbol] = {}
        self.observed_symbol_param[symbol]['symbol'] = symbol
        self.observed_symbol_param[symbol]['enter_time'] = enter_time
        conf = strategy_conf['conf']

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
