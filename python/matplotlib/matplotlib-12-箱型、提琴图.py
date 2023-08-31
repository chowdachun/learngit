
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["font.sans-serif"]=['Arial Unicode MS']   # 设置字体
plt.rcParams["axes.unicode_minus"]=False     # 正常显示负号

path = "/Users/admin/Desktop/史蒂芬大春/办公桌面/编程/python/周报/数据库/rides_2023.csv"
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
"""
箱型图：plt.boxplot()
    x：表示要绘制箱线图的数据。可以是一个数组、一个列表或者一个pandas的Series/DataFrame类型的对象。
    positions：表示每个箱线图在x轴上的位置。可以是一个数组或列表，其中每个元素都是一个数字，表示对应箱线图的x轴位置。如果未指定位置，则默认为每个箱线图在x轴上等距分布。
    vert：表示箱线图是否应该垂直绘制。默认为True，即垂直绘制箱线图。
    showfliers=False：去除异常值，默认是True，不去除异常值
    异常值判断：在数组正态分布中，默认超出3δ（即在正态分布中超出3个单位区间，3δ以外的概率<=0.003）
"""

# 需求：运营中心订单价格箱型图
# 准备数据
date_2 = df.loc[df["operation_center_id"]==2,"payments_price"]/100
date_93 = df.loc[df["operation_center_id"]==93,"payments_price"]/100
date_228 = df.loc[df["operation_center_id"]==228,"payments_price"]/100
date_359 = df.loc[df["operation_center_id"]==359,"payments_price"]/100
date = [date_2,date_93,date_228,date_359]

# 绘制箱型图
fig = plt.figure(figsize=(12,6))
ax = fig.add_axes([0.1,0.1,0.8,0.8])
bp = ax.boxplot(date,vert=False,showfliers=False)    # showfliers=False异常值不展示，vert=False横向箱型图
ax.set_yticklabels(["平和","龙海","漳厦","滨州"])
ax.set_title("订单金额箱型图")
ax.set_ylabel("服务中心")
ax.set_xlabel("订单金额区间")


# 绘制提琴图
fig_vio = plt.figure(figsize=(12,6))
ax_vio = fig_vio.add_axes([0.1,0.1,0.8,0.8])
vio = ax_vio.violinplot(date,vert=False,showmeans=True,showextrema=True,showmedians=True)
ax_vio.set_yticklabels(["平和","龙海","漳厦","滨州"])
ax_vio.set_yticks([1,2,3,4])
ax_vio.set_title("订单金额提琴图")
ax_vio.set_ylabel("服务中心")
ax_vio.set_xlabel("订单金额区间")
# 设置均值线颜色
mean_line = vio['cmeans']    # 抓取vio中的均值线
plt.setp(mean_line, color='r', linewidth=1)
# 设置图表数值
for a,b in zip([1,2,3,4],date):
      ax_vio.text(b.max(),a+0.2,int(b.max()),ha='center')   #最大值
      ax_vio.text(b.min(),a+0.2,int(b.min()),ha='center')   #最小值
      ax_vio.text(b.mean(),a+0.2,int(b.mean()),ha='center')  #均值
      ax_vio.text(b.median(),a-0.2,("均值",int(b.median())),ha='center')  #中位数
      ax_vio.text(b.quantile(0.25),a+0.2,int(b.quantile(0.25)),ha='center')  #25%四分位
      ax_vio.text(b.quantile(0.75),a+0.2,int(b.quantile(0.75)),ha='center')  #75%四分位
plt.show()
