import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 设置随机种子以确保结果的可复现性
np.random.seed(0)

# 生成数据
rows = 1515
cols = 17
data = np.zeros((rows, cols))

# 分类赋值
for i in range(rows):
    if i % 3 == 0:
        value = np.random.uniform(0, 5)  # 生成第一类数据
        data[i, :] = value
    elif i % 3 == 1:
        value = np.random.uniform(5, 10)  # 生成第二类数据
        data[i, :] = value
    else:
        value = np.random.uniform(10, 15)  # 生成第三类数据
        data[i, :] = value

# 创建DataFrame
df = pd.DataFrame(data, columns=[f"Col{num+1}" for num in range(cols)])

# 添加DElabel列
df['DElabel'] = [i % 3 for i in range(rows)]

# Data split
train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42)

# Save files
train_df.to_csv('../../results/demo_train.csv', index=False)
val_df.to_csv('../../results/demo_val.csv', index=False)
test_df.to_csv('../../results/demo_test.csv', index=False)