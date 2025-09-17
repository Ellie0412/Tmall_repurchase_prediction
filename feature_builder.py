import pandas as pd
from typing import Tuple

def add_total_logs(df: pd.DataFrame, user_log: pd.DataFrame) -> pd.DataFrame:
    tmp = (user_log.groupby(["user_id", "seller_id"])
           .size()
           .reset_index(name="total_logs")
           .rename(columns={"seller_id": "merchant_id"}))
    return df.merge(tmp, on=["user_id", "merchant_id"], how="left")


def add_item_count(df: pd.DataFrame, user_log: pd.DataFrame) -> pd.DataFrame:
    tmp = (user_log.groupby(["user_id", "seller_id", "item_id"])
           .size()
           .reset_index()
           .groupby(["user_id", "seller_id"])
           .size()
           .reset_index(name="item_count")
           .rename(columns={"seller_id": "merchant_id"}))
    return df.merge(tmp, on=["user_id", "merchant_id"], how="left")


def add_cat_count(df: pd.DataFrame, user_log: pd.DataFrame) -> pd.DataFrame:
    tmp = (user_log.groupby(["user_id", "seller_id", "cat_id"])
           .size()
           .reset_index()
           .groupby(["user_id", "seller_id"])
           .size()
           .reset_index(name="cat_count")
           .rename(columns={"seller_id": "merchant_id"}))
    return df.merge(tmp, on=["user_id", "merchant_id"], how="left")


def add_action_stats(df: pd.DataFrame, user_log: pd.DataFrame) -> pd.DataFrame:
    # 0-click 1-cart 2-buy 3-fav
    dummy = pd.get_dummies(user_log, columns=["action_type"])
    agg_cols = {f"action_type_{i}": "sum" for i in range(4)}
    tmp = (dummy.groupby(["user_id", "seller_id"])
           .agg(agg_cols)
           .reset_index()
           .rename(columns={"seller_id": "merchant_id",
                            "action_type_0": "click_on",
                            "action_type_1": "add_cart",
                            "action_type_2": "buy_up",
                            "action_type_3": "mark_down"}))
    return df.merge(tmp, on=["user_id", "merchant_id"], how="left")


def add_browse_days(df: pd.DataFrame, user_log: pd.DataFrame) -> pd.DataFrame:
    tmp = (user_log.groupby(["user_id", "seller_id", "time_stamp"])
           .size()
           .reset_index()[["user_id", "seller_id", "time_stamp"]]
           .groupby(["user_id", "seller_id"])
           .size()
           .reset_index(name="browse_days")
           .rename(columns={"seller_id": "merchant_id"}))
    return df.merge(tmp, on=["user_id", "merchant_id"], how="left")


def add_bought_rate(df: pd.DataFrame, user_log: pd.DataFrame) -> pd.DataFrame:
    dummy = pd.get_dummies(user_log, columns=["action_type"])
    tmp = (dummy.groupby("user_id")[["action_type_0", "action_type_1", "action_type_2", "action_type_3"]]
           .sum()
           .reset_index())
    tmp["bought_rate"] = tmp["action_type_2"] / (tmp.iloc[:, 1:].sum(axis=1) + 1e-8)
    return df.merge(tmp[["user_id", "bought_rate"]], on="user_id", how="left")


def add_sold_rate(df: pd.DataFrame, user_log: pd.DataFrame) -> pd.DataFrame:
    dummy = pd.get_dummies(user_log, columns=["action_type"])
    tmp = (dummy.groupby("seller_id")[["action_type_0", "action_type_1", "action_type_2", "action_type_3"]]
           .sum()
           .reset_index())
    tmp["sold_rate"] = tmp["action_type_2"] / (tmp.iloc[:, 1:].sum(axis=1) + 1e-8)
    tmp = tmp.rename(columns={"seller_id": "merchant_id"})[["merchant_id", "sold_rate"]]
    return df.merge(tmp, on="merchant_id", how="left")