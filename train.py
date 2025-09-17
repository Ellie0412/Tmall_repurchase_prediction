# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
import pandas as pd
from data_loader import load_data, load_test


def train():
    # 1. 数据
    X, y = load_data()          # 全部是数值/布尔，无需 cat_features
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=2020, stratify=y)

    # 2. 模型
    model = CatBoostClassifier(
        iterations=20000,
        od_type='Iter',
        depth=5,
        learning_rate=1e-3,
        l2_leaf_reg=5,
        loss_function='Logloss',
        logging_level='Verbose',
        subsample=0.80,
        random_seed=2020,
        thread_count=-1,
        eval_metric='AUC'
    )

    # 3. 训练
    model.fit(X_train, y_train,
              eval_set=(X_val, y_val),
              early_stopping_rounds=300,   # 自动停
              verbose=500)

    # 4. 验证
    val_pred = model.predict_proba(X_val)[:, 1]
    print("Validation AUC:", roc_auc_score(y_val, val_pred))

    # 5. 保存
    model.save_model("/Users/ellie/Documents/Assignments/university-python/天猫/model/catboost_model.cbm")
    print("Model saved to ./catboost_model.cbm")


if __name__ == '__main__':
    train()
