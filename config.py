import os

# 如果后面想自动下载或换路径，只改这里
BASE = "/Users/ellie/Documents/Assignments/university-python/天猫/data/原数据集"

FILENAME = {
    "train": os.path.join(BASE, "train_format1.csv"),
    "user_log": os.path.join(BASE, "user_log_format1.csv"),
    "user_info": os.path.join(BASE, "user_info_format1.csv"),
}
TESTNAME = os.path.join(BASE, "test_format1.csv")