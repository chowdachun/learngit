
import pandas as pd
import matplotlib.pyplot as plt
# macmini path
# panth = "/Users/admin/Desktop/史蒂芬大春/办公桌面/编程/python/周报/数据库/rides.csv"
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
fig,ax = plt.subplots(nrows,ncols)可以在画布上画出多个表格
"""
fig,a = plt.subplots(2,2)   # 定义图表个数
a[0][0].plot(x,y_93)
a[0][0].set_title("93")

a[0][1].plot(x,y_2)
a[0][1].set_title("2")

a[1][0].plot(x,y_228)
a[1][0].set_title("228")

a[1][1].plot(x,y_359)
a[1][1].set_title("359")


"""
plt.subplot2grid(shape, location, rowspan, colspan)可以在画布上画出多个不同形状的表格
    shape：把该参数值规定的网格区域作为绘图区域；
    location：在给定的位置绘制图形，初始位置 (0,0) 表示第1行第1列；
    rowsapan/colspan：这两个参数用来设置让子区跨越几行几列。
"""
line_2 = plt.subplot2grid((3,3),(0,0),colspan=2) # (3,3)在格子3*3区域内，(0,0)第1行1列位置，colspan=2占用2列单元格
line_93 = plt.subplot2grid((3,3),(1,0),colspan=2,rowspan=2)
line_228 = plt.subplot2grid((3,3),(0,2),rowspan=3,colspan=1)    # colspan/rowspan参数默认=1

line_2.plot(x,y_2,'r')
line_2.set_title("2")

line_93.plot(x,y_93,'r')
line_93.set_title("93")

line_228.plot(x,y_228,'r')
line_228.set_title("228")

plt.tight_layout()  # 自动调整图形
plt.show()
