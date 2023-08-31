
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=['Arial Unicode MS']   # 设置字体
plt.rcParams["axes.unicode_minus"]=False     # 正常显示负号

# path = "/Users/admin/Desktop/史蒂芬大春/办公桌面/编程/python/周报/数据库/rides.csv"
# MacBook path
path = "/Users/shuuomousakai/Desktop/史蒂芬大春/办公桌面/编程/python/周报/数据库/rides.csv"

A = pd.read_csv(path)
df = A.query("serving_at >= '2023-04-01' and serving_at < '2023-04-15'")
s_price = df[["serving_at","operation_center_id","payments_price"]]
s_price = s_price.groupby(["serving_at","operation_center_id"]).agg(price=("payments_price",lambda x:x.sum()/100))
s_price = s_price.reset_index()

"""
作图区域
"""

xx = s_price["serving_at"].unique()
x = []
for i in xx:
    x.append(i[5:])

y_93 = s_price.query("operation_center_id == 93")["price"]
y_2 = s_price.query("operation_center_id == 2")["price"]
y_228 = s_price.query("operation_center_id == 228")["price"]
y_359 = s_price.query("operation_center_id == 359")["price"]

"""
grid(color='b', ls = '-.', lw = 0.25)网格线设置
color：表示网格线的颜色；
ls：表示网格线的样式；
lw：表示网格线的宽度
"""

fig,a= plt.subplots(1,3,figsize=(12,4))

a[0].plot(x,y_2)
a[0].set_title("平和")
a[0].set_ylim(0,20000)   # 设置y轴的范围值
a[0].grid()              # 展示网格线

a[1].plot(x,y_93)
a[1].set_yscale('log')  # 设置y轴的缩放形式
a[1].spines['left'].set_color('red')    # 给左轴加颜色
a[1].spines['bottom'].set_color('blue')    # 给底轴加颜色
a[1].set_title("龙海")
a[1].grid(color='g',ls='-',lw=0.25)     # 设置网格线颜色

a[2].plot(x,y_228)
a[2].set_title("漳厦")
a[2].set_xticks([0,2,4,6])  # 设置X轴标签
a[2].set_xticklabels(['zero','two','four','six'])   # 设置X轴刻度标签

plt.tight_layout()
plt.show()