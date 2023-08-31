
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["font.sans-serif"]=['Arial Unicode MS']   # 设置字体
plt.rcParams["axes.unicode_minus"]=False     # 正常显示负号

path = "/Users/admin/Desktop/史蒂芬大春/办公桌面/编程/python/周报/数据库/rides.csv"
A = pd.read_csv(path)
df = A.query("serving_at >= '2023-04-01' and serving_at < '2023-04-15'")
s_price = df[["serving_at","operation_center_id","payments_price","id"]]
s_price = s_price.groupby(["serving_at","operation_center_id"]).agg(
    price=("payments_price",lambda x:x.sum()/100),
    order_count=("id","count")
)
s_price = s_price.reset_index()
# s_price.set_index("serving_at")

"""
作图区域
"""
"""
饼图plt.pie()
X：数组序列，数组元素对应扇形区域的数量大小。
labels：列表字符串序列，为每个扇形区域备注一个标签名字。
color：为每个扇形区域设置颜色，默认按照颜色周期自动设置。
autopct：格式化字符串"fmt%pct"，使用百分比的格式设置每个扇形区的标签，并将其放置在扇形区内。
"""
x = s_price.loc[s_price["serving_at"]=="2023-04-01","operation_center_id"].tolist()
y = s_price.loc[s_price["serving_at"]=="2023-04-01","price"].tolist()

fig = plt.figure()
ax = fig.add_axes([0.1,0.1,0.8,0.8])

len = ["平和","龙海","漳厦","滨州"]
# date = [y_2.iloc[0],y_93.iloc[0],y_228.iloc[0],y_359.iloc[0]]

ax.pie(y,labels=x,autopct='%1.1f%%')
ax.set_title("流水比例")
plt.show()

