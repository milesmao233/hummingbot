from typing import Optional

from hummingbot.client.config.config_var import ConfigVar
from hummingbot.client.settings import required_exchanges


def exchange_on_validated(value: str) -> None:
    required_exchanges.add(value)


def validate_connector(value: str) -> Optional[str]:
    """
    Restrict valid derivatives to the connector file names
    """
    from hummingbot.client import settings
    from hummingbot.client.settings import AllConnectorSettings
    if (value not in AllConnectorSettings.get_connector_settings()
            and value not in settings.PAPER_TRADE_EXCHANGES):
        return f"Invalid connector, please choose value from {AllConnectorSettings.get_connector_settings().keys()}"


dynamic_hedge_config_map = {
    "strategy":
        ConfigVar(
            key="strategy",
            prompt="",
            default="dynamic_hedge",
        ),
    "connector":
        ConfigVar(
            key="connector",
            prompt="Enter a spot connector (Exchange) >>> ",
            prompt_on_new=True,
            validator=validate_connector,
            on_validated=exchange_on_validated
        ),
    "account_name":
        ConfigVar(
            key="account_name",
            prompt="Enter the symbol list >>> ",
            prompt_on_new=True,
        ),
    "symbol_list":
        ConfigVar(
            key="symbol_list",
            prompt="Enter the symbol list >>> ",
            prompt_on_new=True,
        ),
    "cover_list":
        ConfigVar(
            key="cover_list",
            prompt="Enter the cover list >>> ",
            prompt_on_new=True,
        ),
    "index_list":
        ConfigVar(
            key="index_list",
            prompt="Enter the cover list >>> ",
            prompt_on_new=True,
        ),
    "start_date":
        ConfigVar(
            key="start_date",
            prompt="Enter the start date >>> ",
            prompt_on_new=True,
        ),
    "total_amount":
        ConfigVar(
            key="total_amount",
            prompt="Enter the total amount >>> ",
            prompt_on_new=True,
        ),
    "every_order_amount":
        ConfigVar(
            key="every_order_amount",
            prompt="Enter the every order amount >>> ",
            prompt_on_new=True,
        ),
    "max_one_order_amount":
        ConfigVar(
            key="max_one_order_amount",
            prompt="Enter the max one order amount >>> ",
            prompt_on_new=True,
        ),
    "trade_type":
        ConfigVar(
            key="trade_type",
            prompt="Enter the trade type >>> ",
            prompt_on_new=True,
        ),
    "cover_trigger":
        ConfigVar(
            key="cover_trigger",
            prompt="Enter the cover trigger >>> ",
            default=False,
        ),
    "dingding_robot_id":
        ConfigVar(
            key="dingding_robot_id",
            prompt="Enter the dingding robot id >>> ",
            prompt_on_new=True,
        ),
    "dingding_secret":
        ConfigVar(
            key="dingding_secret",
            prompt="Enter the dingding secret >>> ",
            prompt_on_new=True,
        ),
    "dingding_waiting_robot_id":
        ConfigVar(
            key="dingding_waiting_robot_id",
            prompt="Enter the dingding waiting robot id >>> ",
            prompt_on_new=True,
        ),
    "dingding_waiting_secret":
        ConfigVar(
            key="dingding_waiting_secret",
            prompt="Enter the dingding waiting secret >>> ",
            prompt_on_new=True,
        ),
}

