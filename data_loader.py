import pandas as pd
from config import FILENAME, TESTNAME
from utils import time_cost
from feature_builder import (
    add_total_logs, add_item_count, add_cat_count,
    add_action_stats, add_browse_days,
    add_bought_rate, add_sold_rate
)

@time_cost
def load_data(filename: dict = None):
    if filename is None:
        filename = FILENAME

    print("Loading samples...")
    train = pd.read_csv(filename["train"])
    user_info = pd.read_csv(filename["user_info"])
    user_log = pd.read_csv(filename["user_log"]).drop(columns=["brand_id"])
    print("Filling NaN...")
    user_info["age_range"] = user_info["age_range"].fillna(user_info["age_range"].mode()[0])
    user_info["gender"] = user_info["gender"].fillna(2)

    print("Building features...")
    train = add_total_logs(train, user_log)
    train = add_item_count(train, user_log)
    train = add_cat_count(train, user_log)
    train = add_action_stats(train, user_log)
    train = add_browse_days(train, user_log)
    train = add_bought_rate(train, user_log)
    train = add_sold_rate(train, user_log)

    label = train["label"]
    train = train.drop(columns=["user_id", "merchant_id", "label"])
    print("Dataset shape:", train.shape)
    return train, label


@time_cost
def load_test(testname: str = TESTNAME):
    print("Loading test...")
    test = pd.read_csv(testname)
    user_info = pd.read_csv(FILENAME["user_info"])
    user_log = pd.read_csv(FILENAME["user_log"]).drop(columns=["brand_id"])
    user_info["age_range"] = user_info["age_range"].fillna(user_info["age_range"].mode()[0])
    user_info["gender"] = user_info["gender"].fillna(2)

    test = add_total_logs(test, user_log)
    test = add_item_count(test, user_log)
    test = add_cat_count(test, user_log)
    test = add_action_stats(test, user_log)
    test = add_browse_days(test, user_log)
    test = add_bought_rate(test, user_log)
    test = add_sold_rate(test, user_log)

    # 保留 ID 用于提交
    test["user_id"] = test["user_id"].astype(str)
    test["merchant_id"] = test["merchant_id"].astype(str)
    info = test[["user_id", "merchant_id"]].to_numpy()
    test = test.drop(columns=["user_id", "merchant_id"])
    return test, info