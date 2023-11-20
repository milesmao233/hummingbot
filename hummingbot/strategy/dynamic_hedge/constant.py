last_candles_num = 3600
current_candles_num = 5

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
                    7,
                    5
                ],
                [
                    1,
                    7,
                    10,
                ],
                [
                    60,
                    7,
                    1.2,
                ],
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
                7,
                5
            ],
            [
                1,
                7,
                10,
            ],
            [
                60,
                7,
                1.2,
            ],
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