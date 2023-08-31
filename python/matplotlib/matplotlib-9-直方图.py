
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["font.sans-serif"]=['Arial Unicode MS']   # 设置字体
plt.rcParams["axes.unicode_minus"]=False     # 正常显示负号

path = "/Users/admin/Desktop/史蒂芬大春/办公桌面/编程/python/周报/数据库/rides.csv"
A = pd.read_csv(path)
df = A.query("serving_at >= '2023-04-01' and serving_at < '2023-04-15'")
s_price = df[["serving_at","operation_center_id","payments_price","id"]]


s_price = s_price.reset_index()

"""
作图区域，绘制订单价格分布直方图
"""
"""
构建直方图，遵循以下步骤：1、将整个值范围划分为一系列区间。2、区间值（bin）的取值，不可遗漏数据；3、计算每个区间中有多少个值。
plt.hist()
x：必填参数，数组或者数组序列。
    bins：可选参数，整数或者序列，bins 表示每一个间隔的边缘（起点和终点）默认会生成10个间隔。
    range：指定全局间隔的下限与上限值 (min,max)，元组类型，默认值为 None。
    density：如果为 True，返回概率密度直方图；默认为 False，返回相应区间元素的个数的直方图。
    histtype：要绘制的直方图类型，默认值为“bar”，可选值有 barstacked(堆叠条形图)、step(未填充的阶梯图)、stepfilled(已填充的阶梯图)。
"""
# 选择数据
date = s_price["payments_price"]/100
# 价格区间
x = [0,15,30,45,60,75,90,105,120,135,150,175,200]
# 绘制1行2列的画布
# fig,a = plt.subplots(2,1)
# 绘制第一张直方图
fig = plt.figure()
ax = fig.add_axes([0.1,0.1,0.8,0.8])
ax.hist(date,bins=x)
ax.set_xticks(x)
ax.set_title("单均价格直方图")
ax.set_ylabel("订单数量")
# 标示数据
hist,bin = np.histogram(date,x)     # np.histogram(date,bins)求出区间bins范围内date出现的个数
print(hist)
print(bin)
for a, b in zip(bin,hist):          # 设置数据标签
    plt.text(a+8, b+80, b, ha='center')
plt.show()

