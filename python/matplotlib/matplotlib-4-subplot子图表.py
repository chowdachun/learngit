
import pandas as pd
import matplotlib.pyplot as plt

path = "/Users/admin/Desktop/史蒂芬大春/办公桌面/编程/python/周报/数据库/rides_2023.csv"
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

plt.plot([1,2,3])

# 创建一个子图，他表示有2行1列的网格图顶部
# 但是创建的图会与第一个图plt.plot([1,2,3])重叠，所以之前创建的会被删除
# plt.subplot(nrows,cols,index),如plt.subplot(211)表示创建2行1列位置在第1的图
plt.subplot(311)
plt.plot(x,y_93)
plt.ylabel("price")
plt.title("flow line")
# 创建带有黄色背景的第2个子图
plt.subplot(312,facecolor='y')  # 确定第2个位置，背景黄色
plt.plot(x,y_2)
plt.xlabel("serving_date")
plt.ylabel("price")
      # 此时生成上下排版的2幅图形

"""
在图表中增加详细图的add_subplot()函数,实现图中图
"""
plt.subplot(313)
fig = plt.figure()
sub_3 = fig.add_subplot(111)
sub_3.plot(x,y_2)
sub_3_B = fig.add_subplot(222)
sub_3_B.plot([1,3,2,4])


"""
用axes实现图中图效果，（注意主图和子图的顺序要对应）
"""
fig = plt.figure()

axes_main = fig.add_axes([0.1,0.1,0.8,0.8])
axes_sub = fig.add_axes([0.55,0.55,0.3,0.3])

axes_main.plot(x,y_2,'b')
axes_sub.plot(x,y_93,'r')

axes_main.set_title("longhai")
axes_sub.set_title("pinghe")
axes_main.set_xlabel("serving_date")
axes_main.set_ylabel("price")
axes_sub.set_xlabel("serving_date")
axes_sub.set_ylabel("price")

plt.show()
