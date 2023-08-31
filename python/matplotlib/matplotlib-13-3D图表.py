
from mpl_toolkits import mplot3d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"]=['Arial Unicode MS']   # 设置字体
plt.rcParams["axes.unicode_minus"]=False     # 正常显示负号

path = "/Users/admin/Desktop/史蒂芬大春/办公桌面/编程/python/周报/数据库/rides.csv"
A = pd.read_csv(path)
df = A.query("serving_at >= '2023-04-01' and serving_at < '2023-04-15'")
s_price = df[["serving_at","operation_center_id","payments_price","id","payments_completed_at"]]
s_price = s_price.groupby(["serving_at","operation_center_id"]).agg(
    price=("payments_price",lambda x:x.sum()/100),
    order_count=("id","count"),
    order_completed=("payments_completed_at","count")
)
s_price["completed_percent"] = s_price["order_completed"]/s_price["order_count"]
s_price = s_price.reset_index()

xx = s_price["serving_at"].unique()
# x_day = []
# for i in xx:
#     x_day.append(i[5:])
"""
作图区域
"""
# 绘制天、下单量、完成率的【3D折线图表】
fig = plt.figure()
ax = plt.axes(projection='3d')
z = np.arange(len(xx))
x = s_price.loc[s_price["operation_center_id"]==93,"order_count"]
y = s_price.loc[s_price["operation_center_id"]==93,"completed_percent"]
ax.plot3D(x,y,z)
ax.set_title('3D 龙海 line plot')



# 绘制天、下单量、完成率的【3D散点图表】
fig_scat = plt.figure()
ax_scat = plt.axes(projection='3d')

z_93 = df.loc[df["operation_center_id"]==93,"rider_number"]
y_93 = df.loc[df["operation_center_id"]==93,"payments_price"]/100
x_93 = df.loc[df["operation_center_id"]==93,"payments_paid"]/100

# 初始化颜色数组
colors = np.linspace(min(z_93), max(z_93), 1)

ax_scat.scatter3D(x_93,y_93,z_93,c=z_93, cmap='coolwarm')
ax_scat.set_title('3D 龙海 scatter')
ax_scat.set_xlabel("实际支付金额")
ax_scat.set_ylabel("订单金额")
ax_scat.set_zlabel("订单人数")

plt.show()
