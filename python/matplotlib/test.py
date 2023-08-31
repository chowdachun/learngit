import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path = "/Users/admin/Desktop/史蒂芬大春/办公桌面/编程/python/周报/数据库/passengers.csv"
df = pd.read_csv(path)

path_id = "/Users/admin/Desktop/ID.xlsx"
id = pd.read_excel(path_id)

id_list = id["运营中心ID"].to_list()

rides = df.query(f"first_order_operation_center_id in @id_list")["id"]

print(rides.shape)
print(rides)
# print(df.head(20))
path_to = '/Users/admin/Desktop/IDxx.csv'
rides.to_csv(path_to)