from hummingbot.strategy.limit_order import LimitOrder
from hummingbot.strategy.limit_order.limit_order_config_map import limit_order_config_map as c_map
from hummingbot.strategy.market_trading_pair_tuple import MarketTradingPairTuple


def convert_to_list(input_str):
    # 去除空格
    stripped_str = input_str.replace(" ", "")
    # 根据逗号分割字符串
    items_list = stripped_str.split(',')
    # 返回列表
    return items_list


def start(self):
    connector = c_map.get("connector").value.lower()
    # market = c_map.get("market").value
    symbol_list = convert_to_list(c_map.get("symbol_list").value)

    trading_pair_list = list(set(symbol_list))

    self._initialize_markets([(connector, trading_pair_list)])
    for trading_pair in trading_pair_list:
        base, quote = trading_pair.split("-")
        market_trading_pair_info = MarketTradingPairTuple(self.markets[connector], trading_pair, base, quote)
        self.market_trading_pair_tuples.append(market_trading_pair_info)

    # base, quote = market.split("-")
    # market_info = MarketTradingPairTuple(self.markets[connector], market, base, quote)
    # self.market_trading_pair_tuples = [market_info]

    self.strategy = LimitOrder(self.market_trading_pair_tuples)
