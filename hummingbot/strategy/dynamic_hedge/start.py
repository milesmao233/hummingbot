from datetime import datetime

from hummingbot.strategy.dynamic_hedge.dynamic_hedge import DynamicHedge
from hummingbot.strategy.dynamic_hedge.dynamic_hedge_config_map import dynamic_hedge_config_map as d_map
from hummingbot.strategy.market_trading_pair_tuple import MarketTradingPairTuple


def convert_to_list(input_str):
    # 去除空格
    stripped_str = input_str.replace(" ", "")
    # 根据逗号分割字符串
    items_list = stripped_str.split(',')
    # 返回列表
    return items_list


def convert_to_bool(input_str):
    if input_str == "True" or input_str == "true":
        return True
    elif input_str == "False" or input_str == "false":
        return False
    else:
        raise ValueError("input_str must be True or False")


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


def start(self):
    connector = d_map.get("connector").value.lower()
    account_name = d_map.get("account_name").value
    symbol_list = convert_to_list(d_map.get("symbol_list").value)
    cover_list = convert_to_list(d_map.get("cover_list").value)
    index_list = convert_to_list(d_map.get("index_list").value)
    start_date = ensure_datetime_format(d_map.get("start_date").value)
    total_amount = int(d_map.get("total_amount").value)
    every_order_amount = int(d_map.get("every_order_amount").value)
    max_one_order_amount = int(d_map.get("max_one_order_amount").value)
    trade_type = d_map.get("trade_type").value
    cover_trigger = convert_to_bool(d_map.get("cover_trigger").value)
    dingding_robot_id = d_map.get("dingding_robot_id").value
    dingding_secret = d_map.get("dingding_secret").value
    dingding_waiting_robot_id = d_map.get("dingding_waiting_robot_id").value
    dingding_waiting_secret = d_map.get("dingding_waiting_secret").value

    trading_pair_list = list(set(symbol_list + cover_list + index_list))
    self._initialize_markets([(connector, trading_pair_list)])

    self.trading_pair_tuples = []
    for trading_pair in trading_pair_list:
        base, quote = trading_pair.split("-")
        market_trading_pair_info = MarketTradingPairTuple(self.markets[connector], trading_pair, base, quote)
        self.market_trading_pair_tuples.append(market_trading_pair_info)

    self.strategy = DynamicHedge()
    self.strategy.init_params(
        market_trading_pair_tuples=self.market_trading_pair_tuples,
        trading_pair_list=trading_pair_list,
        account_name=account_name,
        symbol_list=symbol_list,
        cover_list=cover_list,
        index_list=index_list,
        start_date=start_date,
        total_amount=total_amount,
        every_order_amount=every_order_amount,
        max_one_order_amount=max_one_order_amount,
        trade_type=trade_type,
        cover_trigger=cover_trigger,
        dingding_robot_id=dingding_robot_id,
        dingding_secret=dingding_secret,
        dingding_waiting_robot_id=dingding_waiting_robot_id,
        dingding_waiting_secret=dingding_waiting_secret,
    )
