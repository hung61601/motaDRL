def condition_clamp(n: int, largest: int, smallest: int) -> int:
    """
    限制數字在給定範圍內，該效能比使用 min() max() 速度還快，效能測試見 performance_test.py。
    :param n: 輸入值。
    :param largest: 最大值。
    :param smallest: 最小值。
    :return: 限制範圍後的輸入值。
    """
    return smallest if smallest > n else n if largest > n else largest
