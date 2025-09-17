# -*- coding: utf-8 -*-
from catboost import CatBoostClassifier
from data_loader import load_test   # 我们刚拆好的接口
import pandas as pd
import numpy as np
import time
import os

# 1. 读测试集
X_test, info_arr = load_test()      # info_arr: ndarray, shape=(N,2)
print('X_test shape:', X_test.shape)

# 2. 加载模型
model_path = '/Users/ellie/Documents/Assignments/university-python/天猫/model/catboost_model.cbm'
model = CatBoostClassifier()
model.load_model(model_path)
print('Model loaded.')

# 3. 预测
y_prob = model.predict_proba(X_test)[:, 1]

# 4. 拼接提交格式
sub = pd.DataFrame(info_arr, columns=['user_id', 'merchant_id'])
sub['prob'] = y_prob

# 5. 保存
res_dir = '/Users/ellie/Documents/Assignments/university-python/天猫/result'
os.makedirs(res_dir, exist_ok=True)
save_name = f'ans_{time.strftime("%Y%m%d_%H%M%S", time.localtime())}.csv'
save_path = os.path.join(res_dir, save_name)
sub.to_csv(save_path, sep=',', header=True, index=False)
print(f'Result saved -> {save_path}')