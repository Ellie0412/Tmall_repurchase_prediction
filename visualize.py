# visualize_only.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1. 路径配置
BASE = "/Users/ellie/Documents/Assignments/university-python/Tmall/data/原数据集"
train_df = pd.read_csv(f"{BASE}/train_format1.csv")
user_info_df = pd.read_csv(f"{BASE}/user_info_format1.csv")
user_log_df = pd.read_csv(f"{BASE}/user_log_format1.csv")

SAVE_DIR = Path('/Users/ellie/Documents/Assignments/university-python/Tmall/pictures')
SAVE_DIR.mkdir(exist_ok=True)

# ---- 1. 用户画像 vs 重复购买率 ----
def plot_user_profile():
    df = train_df.merge(user_info_df, on='user_id', how='left')
    # 年龄 vs 复购率
    age = df.groupby('age_range')['label'].agg(['mean', 'count']).reset_index()
    age['mean'].plot(kind='bar', title='复购率 ~ 年龄')
    plt.savefig(SAVE_DIR / 'profile_age.png', dpi=120)
    plt.close()
    # 性别 vs 复购率
    gender = df.groupby('gender')['label'].agg(['mean', 'count']).reset_index()
    gender['mean'].plot(kind='bar', title='复购率 ~ 性别')
    plt.savefig(SAVE_DIR / 'profile_gender.png', dpi=120)
    plt.close()

# ---- 2. 用户行为 vs 重复购买率 ----
def plot_behavior():
    # 合并行为
    df = train_df.merge(user_log_df, on='user_id', how='left')
    # 行为类型 vs 复购率
    beh = df.groupby('action_type')['label'].mean()
    beh.plot(kind='bar', title='复购率 ~ 行为类型')
    plt.savefig(SAVE_DIR / 'behavior_action.png', dpi=120)
    plt.close()

# ---- 3. 商户表现 vs 用户行为 ----
def plot_merchant():
    # 商户维度行为汇总
    merc = user_log_df.groupby('seller_id').agg(
        clicks=('action_type', lambda x: (x==0).sum()),
        carts=('action_type', lambda x: (x==1).sum()),
        buys=('action_type', lambda x: (x==2).sum()),
        favs=('action_type', lambda x: (x==3).sum())
    ).reset_index()
    # 散点：购买数 vs 点击数
    sns.scatterplot(data=merc, x='clicks', y='buys')
    plt.title('商户：点击量 vs 购买量')
    plt.savefig(SAVE_DIR / 'merchant_clicks_buys.png', dpi=120)
    plt.close()



if __name__ == '__main__':
    plot_user_profile()
    plot_behavior()
    plot_merchant()
    print("3 张图已保存到 ./pictures/")