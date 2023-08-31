
import pandas as pd
import matplotlib.pyplot as plt
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
x = []
for i in xx:
    x.append(i[5:])

y_93_A = s_price.query("operation_center_id == 93")["price"]
y_93_B = s_price.query("operation_center_id == 93")["order_count"]
y_2_B = s_price.query("operation_center_id == 2")["order_count"]

# 创建fig实例
fig = plt.figure()
# 添加子图区域
a1 = fig.add_axes([0.1,0.1,0.8,0.8])
a1.plot(x,y_93_A)
a1.set_ylabel("流水")
a1.set_title("流水与订单量")
a1.set_ylim(0,250000)
# 添加双轴
a2 = a1.twinx()         # 在a1中绘制另外坐标轴
a2.plot(x,y_93_B,"ro-")
a2.set_ylabel("订单量")
a2.set_ylim(0,2500)

a3 = a1.twinx()         # 在a1中绘制另外坐标轴
a3.plot(x,y_2_B,"b")
a3.set_ylabel("订单量")
a3.set_ylim(0,2500)

# 绘制图例，对应set_ylabel()参数
fig.legend(labels=("龙海流水","龙海订单量","平和订单量"),loc='upper right')
plt.show()




