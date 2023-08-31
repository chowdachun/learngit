

import pandas as pd
import matplotlib.pyplot as plt

path = "/Users/admin/Desktop/史蒂芬大春/办公桌面/编程/python/周报/数据库/rides.csv"
A = pd.read_csv(path)
df = A.query("serving_at >= '2023-04-01' and serving_at < '2023-04-15'")
s_price = df[["serving_at","operation_center_id","payments_price"]]
s_price = s_price.groupby(["serving_at","operation_center_id"]).agg(price=("payments_price",lambda x:x.sum()/100))
s_price = s_price.reset_index()

"""
作图区域
"""
x = s_price["serving_at"].unique()
y_93 = s_price.query("operation_center_id == 93")["price"]
y_2 = s_price.query("operation_center_id == 2")["price"]

fig = plt.figure()      # 创建空白画布
ax = fig.add_axes([0.1,0.1,0.8,0.8])    # 分别对应图形的左侧、底部、宽度、高度，每个数字在0-1之间

ax.set_xlabel("serving")        # 设置X轴名称
ax.set_ylabel("price")        # 设置y轴名称
ax.set_title('ABC')    # 设置图表名称
ax.plot(x,y_93,'r')
ax.plot(x,y_2,'y')
plt.show()              # 展示图表


