
import pandas as pd
import numpy as np

in_path = "/Users/admin/Desktop/史蒂芬大春/办公桌面/编程/sklearn/RNN时间序列预测走势/19-23.8.22流水.csv"
df = pd.read_csv(in_path,index_col='订单服务日期',parse_dates=['订单服务日期'])

# 假设你的二维数组为 arr
arr = [[1], [2], [3]]

# 使用 numpy.ravel() 函数将二维数组转换为一维数组
arr_flat = np.ravel(arr)

# 打印转换后的一维数组
print(df.size)