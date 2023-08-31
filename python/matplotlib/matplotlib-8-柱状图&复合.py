
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

xx = s_price["serving_at"].unique()
x_day = []
for i in xx:
    x_day.append(i[5:])

y_2 = s_price.query("operation_center_id == 2")["price"]
y_93 = s_price.query("operation_center_id == 93")["price"]
y_228 = s_price.query("operation_center_id == 228")["price"]
y_359 = s_price.query("operation_center_id == 359")["price"]
y_93_order = s_price.query("operation_center_id == 93")["order_count"]

"""
条形图ax.bar(x, height, width, bottom, align)
x：设置位置，一个标量序列，代表柱状图的x坐标，默认x取值是每个柱状图所在的中点位置，或者也可以是柱状图左侧边缘位置。
height：一个标量或者是标量序列，代表柱状图的高度。
width：可选参数，标量或类数组，柱状图的默认宽度值为 0.8。
bottom：可选参数，标量或类数组，柱状图的y坐标默认为None。
algin：有两个可选项 {"center","edge"}，默认为 'center'，该参数决定 x 值位于柱状图的位置。
"""

# 创建一个画布对象
fig = plt.figure()
ax = fig.add_axes([0.1,0.1,0.8,0.8])
# 选取数据
x = ["平和","龙海","漳厦","滨州"]
y = [y_2.iloc[0],y_93.iloc[0],y_228.iloc[0],y_359.iloc[0]]
# 绘制条形图
plt.bar(x,y)


# 绘制多条形图
fig = plt.figure()
ax_mulite = fig.add_axes([0.1,0.1,0.8,0.8])
date = [
    y_2.iloc[0:4].tolist(),y_93.iloc[0:4].tolist(),y_228.iloc[0:4].tolist(),y_359.iloc[0:4].tolist()
]
x = ["4月1日","4月2日","4月3日","4月4日"]
i = np.arange(4)
# 条形图x参数要确定条形图位置
ax_mulite.bar(i+0.20,date[0],color='r',width=0.20)  # i+0.2 定位值i[1,2,3,4]+0.2=[1.2,2.2,3.2,4.2]
ax_mulite.bar(i+0.40,date[1],color='b',width=0.20)  # i+0.4 定位值i[1,2,3,4]+0.2=[1.4,2.4,3.4,4.4]
ax_mulite.bar(i+0.60,date[2],color='y',width=0.20)  # i+0.6 定位值i[1,2,3,4]+0.2=[1.6,2.6,3.6,4.6]
ax_mulite.bar(i+0.80,date[3],color='g',width=0.20)  # i+0.8 定位值i[1,2,3,4]+0.2=[1.8,2.8,3.8,4.8]
fig.legend(labels=("平和","龙海","漳厦","滨州"),loc='upper right')  # 图例是在fig对象上设置
ax_mulite.set_xticks([0.5,1.5,2.5,3.5],x)       # 设置X轴标签，第1个参数是位置参数
ax_mulite.set_title("流水对比图")



# 绘制多条形图折线图
fig = plt.figure()
ax_mulite = fig.add_axes([0.1,0.1,0.8,0.8])
date = [
    y_2.iloc[0:4].tolist(),y_93.iloc[0:4].tolist(),y_228.iloc[0:4].tolist(),y_359.iloc[0:4].tolist()
]
x = ["4月1日","4月2日","4月3日","4月4日"]
i = np.arange(len(x))
# 条形图x参数要确定条形图位置
ax_mulite.bar(i+0.20,date[0],color='r',width=0.20)  # i+0.2 定位值i[1,2,3,4]+0.2=[1.2,2.2,3.2,4.2]
ax_mulite.bar(i+0.40,date[1],color='b',width=0.20)  # i+0.4 定位值i[1,2,3,4]+0.2=[1.4,2.4,3.4,4.4]
ax_mulite.bar(i+0.60,date[2],color='y',width=0.20)  # i+0.6 定位值i[1,2,3,4]+0.2=[1.6,2.6,3.6,4.6]
ax_mulite.bar(i+0.80,date[3],color='g',width=0.20)  # i+0.8 定位值i[1,2,3,4]+0.2=[1.8,2.8,3.8,4.8]
fig.legend(labels=("平和","龙海","漳厦","滨州"),loc='upper right')  # 图例是在fig对象上设置
ax_mulite.set_xticks([0.5,1.5,2.5,3.5],x)       # 设置X轴标签，第1个参数是位置参数
ax_mulite.set_ylabel("流水")
ax_mulite.set_title("流水对比图")
# 绘制复合折线图
a2 = ax_mulite.twinx()
y_plot = y_93_order.iloc[0:4]
a2.plot([0.5,1.5,2.5,3.5],y_plot,"ro-")
a2.set_ylabel("订单量")
a2.set_ylim(0,1500)


# 绘制堆叠柱状图
fig = plt.figure(figsize=(10,5))    #(figsize=(10,5)设置图形的长宽
ax_over = fig.add_axes([0.1,0.1,0.8,0.8])
i = np.arange(len(x_day))
ax_over.bar(x_day,y_2.tolist(),color='#CD853F',label="平和")
ax_over.bar(x_day,y_93.tolist(),color='silver',bottom=y_2,label="龙海")
y = [a+b for a,b in zip(y_2.tolist(),y_93.tolist())]
ax_over.bar(x_day,y_228.tolist(),color='gold',bottom=y,label="漳厦")
# fig.legend(loc="upper right")
ax_over.set_title("流水堆叠柱状图")
ax_over.set_ylabel("流水金额")
z = [round(a+b+c) for  a,b,c in zip(y_2.tolist(),y_93.tolist(),y_228.tolist())]
for a, b in zip(i,z):
    plt.text(a, b+300, b, ha='center')

# 绘制折线组合图
a2 = ax_over.twinx()
a2.plot(x_day,y_93_order,'ro-',label='订单量')
a2.set_ylabel("订单量")
a2.set_ylim(0,1500)
for a, b in zip(i,y_93_order):
    plt.text(a, b-100, b, ha='center')      # 展示文本
fig.legend(loc="upper right")
plt.savefig("/Users/admin/Desktop/1.jpg")   # 保存图片
plt.show()

