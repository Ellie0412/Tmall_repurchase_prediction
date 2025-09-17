import time
import numpy as np
import pandas as pd

def time_cost(func):
    """耗时装饰器"""
    def wrapper(*args, **kw):
        start = time.time()
        res = func(*args, **kw)
        print(f"Function: {func.__name__}, Cost: {time.time()-start:.3f} sec")
        return res
    return wrapper


def data_clean(data: pd.DataFrame, fea: str, sigma: int = 3):
    """3σ 离群点标记"""
    mean, std = data[fea].mean(), data[fea].std(ddof=1)
    delta = sigma * std
    low, high = mean - delta, mean + delta
    data[fea + "_outlier"] = data[fea].apply(lambda x: "T" if x < low or x > high else "F")
    return data