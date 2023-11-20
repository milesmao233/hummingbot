import logging
from datetime import datetime
from decimal import Decimal
from typing import Dict, List

import numpy as np
import pandas as pd

from hummingbot.connector.constants import s_decimal_0
from hummingbot.connector.derivative.position import Position
from hummingbot.connector.trading_rule import TradingRule
from hummingbot.core.data_type.common import OrderType, PositionAction, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate, PerpetualOrderCandidate
from hummingbot.logger import HummingbotLogger

hws_logger = None


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


def get_twap_symbol_info_list_swap(symbol_info, order_amount):
    """
    对超额订单进行拆分,并进行调整,尽可能每批中子订单、每批订单让多空平衡
    :param symbol_info 原始下单信息
    :param order_amount:单次下单最大金额
    """

    # 对下单资金进行拆单
    symbol_info['拆单金额'] = symbol_info['实际下单资金'].apply(
        lambda x: [x] if abs(x) < order_amount else [(1 if x > 0 else -1) * order_amount] * int(
            abs(x) / order_amount) + [x % (order_amount if x > 0 else -order_amount)])
    # 使得最后一个元素，如果小于15，就加到最后第二个元素中
    symbol_info['拆单金额'] = symbol_info['拆单金额'].apply(
        lambda lst: lst[:-2] + [lst[-2] + lst[-1]] if lst[-1] < 15 and len(lst) > 1 else lst
    )
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



class HedgeOrder:
    in_waiting_list = []
    trading_list_before_calc = []
    in_trading_list = []

    @classmethod
    def logger(cls) -> HummingbotLogger:
        global hws_logger
        if hws_logger is None:
            hws_logger = logging.getLogger(__name__)
        return hws_logger

    @property
    def active_positions(self) -> Dict[str, Position]:
        return self._market.account_positions

    def __init__(
            self,
            market,
            trade_type,
            market_trading_pair_tuples,
            account_name,
            max_one_order_amount,
            every_order_amount,
            cover_trigger,
            hedge_notify_robot
    ):
        self._market = market
        self._trade_type = trade_type
        self._market_trading_pair_tuples = market_trading_pair_tuples
        self._account_name = account_name
        self._max_one_order_amount = max_one_order_amount
        self._every_order_amount = every_order_amount
        self._cover_trigger = cover_trigger
        self.hedge_notify_robot = hedge_notify_robot

    def update_in_waiting_list(self, in_waiting_list):
        self.in_waiting_list = in_waiting_list

    def update_in_trading_list(self, in_trading_list):
        self.trading_list_before_calc = self.in_trading_list.copy()
        self.in_trading_list = in_trading_list

    def sell_or_buy(self):
        if self._trade_type == 'spot':
            if len(self.trading_list_before_calc) > 0:
                for trading_pair in self.trading_list_before_calc:
                    if trading_pair not in self.in_trading_list:
                        # 卖出
                        self.logger().info(f"sell {trading_pair}")
                        # self.sell_with_twap_type(trading_pair)
                        self.sell_with_twap_type(trading_pair)

            if len(self.in_trading_list) > 0:
                for trading_pair in self.in_trading_list:
                    if trading_pair not in self.trading_list_before_calc:
                        # 买入
                        self.logger().info(f"buy {trading_pair}")
                        self.buy_with_twap_type(trading_pair)

        elif self._trade_type == 'swap':
            process_coin = self.get_process_coin()
            position_df = self.active_positions_df()
            # usdt_balance = self.get_balance('USDT') / 5
            symbol_order = self.cal_order_amount(process_coin, position_df, self._every_order_amount)
            for symbol, row in symbol_order.iterrows():
                amount = row['实际下单量']
                quantity = row['实际下单资金']
                if amount > 0:
                    is_bid = True
                else:
                    is_bid = False
                schedule_price = round(float(quantity / amount), 3)
                # bid_or_not =
                self.hedge_notify_robot.send_dingding_msg(
                    f'计划 {"买入" if is_bid else "卖出"}：\n'
                    f'币种：{symbol} \n'
                    f'数量：{abs(amount)} \n'
                    f'价格：{schedule_price} \n'
                    f'生成拆分订单......'
                )

            if not symbol_order.empty:
                twap_symbol_info_list = get_twap_symbol_info_list_swap(symbol_order, self._max_one_order_amount)
                self.logger().info(f'拆单信息：\n {twap_symbol_info_list}', )

                # =====遍历下单
                proposal = []
                for i in range(len(twap_symbol_info_list)):
                    # print(f"twap {i} \n", twap_symbol_info_list[i])
                    orders = self.create_active_order(twap_symbol_info_list[i])
                    # print('orders\n', orders)
                    proposal.extend(orders)

                if len(proposal) > 0:
                    adjusted_proposal = self._market.budget_checker.adjust_candidates(
                        proposal, all_or_none=True)
                    self.logger().info(f'adjusted_proposal \n {adjusted_proposal}')
                    for order in adjusted_proposal:
                        order_close = PositionAction.CLOSE if order.position_close else PositionAction.OPEN
                        market_pair = self.find_trading_pair_tuple(order.trading_pair)
                        if order.order_side == TradeType.BUY:
                            order_res = self.buy_with_specific_market(market_pair, order.amount, order.order_type,
                                                          position_action=order_close)

                            print('order_res: \n', order_res)
                        elif order.order_side == TradeType.SELL:
                            order_res = self.sell_with_specific_market(market_pair, order.amount, order.order_type,
                                                           position_action=order_close)
                            print('order_res: \n', order_res)

    def sell(self):
        pass

    def buy(self):
        pass

    def sell_with_twap_type(self, trading_pair):
        if self._trade_type == 'swap':
            self.sell_with_twap_type_swap(trading_pair)
        elif self._trade_type == 'spot':
            self.sell_with_twap_type_spot(trading_pair)

    def buy_with_twap_type(self, trading_pair):
        if self._trade_type == 'swap':
            self.buy_with_twap_type_swap(trading_pair)
        elif self._trade_type == 'spot':
            self.buy_with_twap_type_spot(trading_pair)

    def sell_with_twap_type_swap(self, trading_pair):
        pass

    def buy_with_twap_type_swap(self, trading_pair):
        pass

    def sell_with_twap_type_spot(self, trading_pair):
        trading_pair_tuple = self.find_trading_pair_tuple(trading_pair)
        amount = trading_pair_tuple.base_balance
        new_price = trading_pair_tuple.get_mid_price()
        balance_usdt_value = amount * new_price
        twap_symbol_amount_list = get_twap_symbol_info_list(
            balance_usdt_value,
            self._max_one_order_amount,
            new_price,
        )

        twap_symbol_amount_list_sum = np.sum(twap_symbol_amount_list)
        if twap_symbol_amount_list_sum < amount:
            twap_symbol_amount_list[-1] += amount - twap_symbol_amount_list_sum
        elif twap_symbol_amount_list_sum > amount:
            twap_symbol_amount_list[-1] -= twap_symbol_amount_list_sum - amount

        for twap_amount in twap_symbol_amount_list:
            trading_pair_tuple.market.sell(
                trading_pair=trading_pair_tuple.trading_pair,
                amount=twap_amount,
                order_type=OrderType.MARKET,
            )
            self.logger().info(f'{twap_amount} {trading_pair_tuple.trading_pair} selled')

        self.hedge_notify_robot.send_dingding_msg(
            f"account: {self._account_name} \n "
            f"卖出信号: {trading_pair}  \n"
            f"当前 waiting_list: {self.in_waiting_list} \n"
            f"当前 trading_list: {self.in_trading_list} \n"
            f"时间: {datetime.now()} \n"
            f"价格: {new_price} \n"
            f"卖出数量: {np.sum(twap_symbol_amount_list)}"
        )

    def buy_with_twap_type_spot(self, trading_pair):
        trading_pair_tuple = self.find_trading_pair_tuple(trading_pair)
        new_price = trading_pair_tuple.get_mid_price()
        twap_symbol_amount_list = get_twap_symbol_info_list(
            self._every_order_amount, self._max_one_order_amount, new_price
        )
        for twap_amount in twap_symbol_amount_list:
            trading_pair_tuple.market.buy(
                trading_pair=trading_pair_tuple.trading_pair,
                amount=Decimal(twap_amount),
                order_type=OrderType.MARKET,
            )
            self.logger().info(f'{twap_amount} {trading_pair_tuple.trading_pair} bought')

    def find_trading_pair_tuple(self, trading_pair):
        for x in self._market_trading_pair_tuples:
            if x.trading_pair == trading_pair:
                return x

    def active_positions_df(self) -> pd.DataFrame:
        columns = ["symbol", "type", "entry_price", "amount", "leverage", "unrealized_pnl"]
        data = []
        # market, trading_pair = self._market_info.market, self._market_info.trading_pair
        for idx in self.active_positions.values():
            # is_buy = True if idx.amount > 0 else False
            # unrealized_profit = ((market.get_price(trading_pair, is_buy) - idx.entry_price) * idx.amount)
            data.append([
                idx.trading_pair,
                idx.position_side.name,
                Decimal(idx.entry_price),
                Decimal(idx.amount),
                idx.leverage,
                Decimal(idx.unrealized_pnl)
            ])

        df = pd.DataFrame(data=data, columns=columns)
        df.set_index('symbol', inplace=True)
        return df

    def get_process_coin(self):
        process_coin = pd.DataFrame()
        process_coin['symbol'] = self.in_trading_list
        current_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
        process_coin['candle_begin_time'] = current_time

        symbol_price = {}
        for symbol in self.in_trading_list:
            trading_pair_tuple = self.find_trading_pair_tuple(symbol)
            new_price = trading_pair_tuple.get_mid_price()
            symbol_price[symbol] = Decimal(new_price)
        process_coin['close'] = process_coin['symbol'].map(symbol_price)
        process_coin['方向'] = 1

        return process_coin

    def cal_order_amount(self, process_coin, position_df, usdt_balance):
        # process_coin['方向选币数量'] = process_coin.groupby(['candle_begin_time', '方向'])['symbol'].transform(
        #     'size')
        process_coin['目标持仓量'] = usdt_balance / process_coin['close'] * process_coin['方向']  # 计算每个币种的目标持仓量

        symbol_order = pd.DataFrame(index=list(set(process_coin['symbol']) | set(position_df.index)),
                                    columns=['当前持仓量'])
        symbol_order['当前持仓量'] = position_df['amount']
        symbol_order['当前持仓量'].fillna(value=0, inplace=True)
        symbol_order['当前持仓量'] = symbol_order['当前持仓量'].apply(Decimal)

        # =目前持仓量当中，可能可以多空合并
        symbol_order['目标持仓量'] = process_coin.groupby('symbol')[['目标持仓量']].sum()
        symbol_order['目标持仓量'].fillna(value=0, inplace=True)
        symbol_order['目标持仓量'] = symbol_order['目标持仓量'].apply(Decimal)
        # symbol_order['目标持仓量'].apply(Decimal)

        # print('目标持仓量\n', symbol_order['目标持仓量'])
        # print('当前持仓量\n', symbol_order['当前持仓量'])

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

    def get_ticker_data(self, nan_symbol):
        nan_symbol_values = nan_symbol.values
        symbol_price_dict = {
            "symbol": [],
            "price": []
        }
        for i in range(len(nan_symbol_values)):
            symbol_name = nan_symbol_values[i]
            symbol_price_dict["symbol"].append(symbol_name)
            trading_pair_tuple = self.find_trading_pair_tuple(symbol_name)
            symbol_price = trading_pair_tuple.get_mid_price()
            symbol_price_dict["price"].append(symbol_price)

        tickers = pd.DataFrame(symbol_price_dict)
        tickers.set_index('symbol', inplace=True)
        return tickers['price']

    def create_active_order(self, symbol_order):
        orders: List[OrderCandidate] = []
        for symbol, row in symbol_order.iterrows():
            trading_rule: TradingRule = self._market.trading_rules[symbol]
            quantity = row['实际下单量']
            if quantity > 0:
                is_bid = True
            else:
                is_bid = False
            # 根据最小递进下单量，进行下单量的调整，取小数
            # min_base_amount_increment = Decimal(trading_rule.min_base_amount_increment)
            # quantity = (Decimal(abs(quantity)) // min_base_amount_increment) * min_base_amount_increment
            quantity = Decimal(abs(quantity))

            mid_price = self._market.get_mid_price(symbol)
            bid_spread = Decimal(.1)
            ask_spread = Decimal(.1)
            bid_price = mid_price + mid_price * bid_spread * Decimal(.01)
            ask_price = mid_price - mid_price * ask_spread * Decimal(.01)

            price = bid_price if is_bid else ask_price
            price = self._market.quantize_order_price(symbol, Decimal(price))
            reduce_only = True if row['交易模式'] == '清仓' else False

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
                order_type=OrderType.MARKET,
                order_side=TradeType.BUY if is_bid else TradeType.SELL,
                amount=quantity,
                price=price,
                position_close=reduce_only,
                leverage=Decimal(10)
            )

            # print('order: ', order)
            orders.append(order)
        return orders

