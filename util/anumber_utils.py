import math


def anumber_func(hiddendim):
    if hiddendim not in [256, 512, 1024, 2048]:
        return None
    else:
        return int(math.log2(hiddendim) - 8)