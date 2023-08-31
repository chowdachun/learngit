
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

"""
作图区域
"""
""""
散点图plt.scatter()
    x：表示 x 轴数据点的集合，可以是数组、列表或者是一个标量，x和y必须要相同长度
    y：表示 y 轴数据点的集合，可以是数组、列表或者是一个标量。
    s：表示散点的尺寸，默认值为 rcParams['lines.markersize'] ** 2，即默认值为 36。
    c：表示散点的颜色，可以是字符串、数字、数组或者 colormap 的名称，例如 c='r'、c=0.5、c=[0, 1, 2] 或 c=plt.cm.cool。
    marker：表示散点的标记形状，如 . 表示圆形、o 表示实心圆、s 表示正方形等，还可以使用 Unicode 字符来显示别的特殊符号。
    alpha：表示散点的透明度，取值范围从 0（完全透明）到 1（完全不透明）
"""
x = np.arange(1,250,5)
y_93 = (df.loc[df["operation_center_id"]==93,"payments_price"]/100).tolist()
y_228 = (df.loc[df["operation_center_id"]==228,"payments_price"]/100).tolist()

fig = plt.figure()
ax = fig.add_axes([0.1,0.1,0.8,0.8])

hist_93,bin_93 = np.histogram(y_93,x)
hist_228,bin_228 = np.histogram(y_228,x)


ax.scatter(x[:len(hist_93)],hist_93,color='b',label="龙海")
ax.scatter(x[:len(hist_228)],hist_228,color='g',label="漳厦")
ax.set_title("龙海/漳厦订单价格散点图")
ax.set_ylabel("金额")
ax.set_xlabel("价格区间")
ax.set_xticks(x)
plt.legend()
plt.show()

