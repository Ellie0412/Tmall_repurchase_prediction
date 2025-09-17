# main.py
from data_loader import load_data, load_test
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd
import os

def train():
    """训练 + 保存模型"""
    X, y = load_data()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    model = CatBoostClassifier(
        iterations=20000,
        depth=6,
        learning_rate=1e-3,
        l2_leaf_reg=5,
        subsample=0.8,
        random_seed=42,
        early_stopping_rounds=300,
        verbose=500
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    print("Val AUC:", roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))
    model.save_model("/Users/ellie/Documents/Assignments/university-python/天猫/model/catboost_model.cbm")

def predict_submit():
    """预测 + 生成 prediction.csv"""
    X_test, info = load_test()
    model = CatBoostClassifier().load_model("/Users/ellie/Documents/Assignments/university-python/天猫/model/catboost_model.cbm")
    prob = model.predict_proba(X_test)[:, 1]

    sub = pd.DataFrame(info, columns=["user_id", "merchant_id"])
    sub["prob"] = prob
    pre_dir = "/Users/ellie/Documents/Assignments/university-python/天猫/data/prediction"
    os.makedirs(pre_dir, exist_ok=True)
    sub.to_csv(os.path.join(pre_dir, "prediction.csv"), index=False, sep=",")
    print("prediction.csv saved.")

if __name__ == '__main__':
    train()          # 先训练
    predict_submit() # 再生成提交文件