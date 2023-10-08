import torch


def condition_clamp(n: int, largest: int, smallest: int) -> int:
    """
    限制數字在給定範圍內，該效能比使用 min() max() 速度還快，效能測試見 performance_test.py。
    :param n: 輸入值。
    :param largest: 最大值。
    :param smallest: 最小值。
    :return: 限制範圍後的輸入值。
    """
    return smallest if smallest > n else n if largest > n else largest


def get_device():
    """
    選擇 torch 可用的設備。
    :return: 如果無可用的 CUDA，則切換成 CPU 運行。
    """
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
    return device
