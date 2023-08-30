
import pandas as pd


df_1 = pd.read_pickle("/Users/admin/Desktop/史蒂芬大春/办公桌面/编程/python/周报/数据库/rides_2022.pkl")
df_2 = pd.read_pickle("/Users/admin/Desktop/史蒂芬大春/办公桌面/编程/python/周报/数据库/rides_2023.pkl")

df = pd.concat([df_1,df_2],axis=0,join='outer')
print(df_1)
print(df_2)
print(df)

df.to_pickle("/Users/admin/Desktop/days_60/data.pkl")