import numpy as np


def calc_std_for_index(x, df, column_name, relative_vol_threshold_offset, relative_vol_threshold_update_period,
                       relative_vol_threshold_coefficient):
    """
    计算指定周期内的标准差
    :param x: 第几行
    :param df: 整个 dataframe
    :param column_name: close
    :param relative_vol_threshold_update_period: 参数
    :param relative_vol_threshold_offset: 参数
    :param relative_vol_threshold_coefficient: 参数
    :return:

    g_relative_vol_threshold_period: 4    更新的数据是4个周期的数据（包括当前行）
    g_relative_vol_threshold_update_period: 3   3分钟更新一次

    index为0的是nan
    index为1的是nan
    index为2的是nan
    index为3的是index: 0,1,2,3的 symbol_relative std
    index为4的是index: 0,1,2,3的 symbol_relative std
    index为5的是index: 0,1,2,3的 symbol_relative std
    index为6的是index: 3,4,5,6的 symbol_relative std
    index为7的是index: 3,4,5,6的 symbol_relative std
    index为8的是index: 3,4,5,6的 symbol_relative std
    index为9的是index: 6,7,8,9的 symbol_relative std
    index为10的是index: 6,7,8,9的 symbol_relative std
    index为11的是index: 6,7,8,9的 symbol_relative std
    """

    group_start_index = (x // relative_vol_threshold_update_period) * relative_vol_threshold_update_period
    if group_start_index < relative_vol_threshold_offset:
        return np.nan
    else:
        return df.loc[(group_start_index - relative_vol_threshold_offset + 1): group_start_index, column_name].std(
            ddof=0) * relative_vol_threshold_coefficient


def calculate_rolling_std(df, window_size, interval):
    std_values = []
    for i in range(len(df)):
        # 确定窗口的起始和结束位置
        start = max(0, i - (window_size - 1) * interval)
        end = i + 1  # 因为Python切片不包括结束索引

        # 计算标准差
        window_std = df.iloc[start:end:interval].std(ddof=0)
        std_values.append(window_std)

    return std_values
