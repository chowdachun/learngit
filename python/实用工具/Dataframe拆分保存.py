
import pandas as pd
# import datetime as  dt
import time

start_time = time.time()
path = "/周报/数据库/rides.csv"
df = pd.read_csv(path)

# 获取时间字段list[]
datetime_list = ['created_at','payments_completed_at','serving_at', 'pickup_from', 'pickup_to','actual_pickup_at', 'actual_dropoff_at','accepted_at',
                 'canceled_at','penalty_paid_at','reminded_at']

for name in datetime_list:
    df[name] = df[name].str.replace("\+08", "")


df['created_at'] = pd.to_datetime(df['created_at'],format='%Y-%m-%d',errors='coerce')

for name in datetime_list:
    df[name] = pd.to_datetime(df[name],format='%Y-%m-%d %H:%M:%S',errors='coerce')

stat_2017 = pd.to_datetime('2017-01-01', format='%Y-%m-%d')
stat_2018 = pd.to_datetime('2018-01-01', format='%Y-%m-%d')
stat_2019 = pd.to_datetime('2019-01-01', format='%Y-%m-%d')
stat_2020 = pd.to_datetime('2020-01-01', format='%Y-%m-%d')
stat_2021 = pd.to_datetime('2021-01-01', format='%Y-%m-%d')
stat_2022= pd.to_datetime('2022-01-01', format='%Y-%m-%d')
stat_2023= pd.to_datetime('2023-01-01', format='%Y-%m-%d')
stat_2024= pd.to_datetime('2024-01-01', format='%Y-%m-%d')

df_2017 = df.query(f"serving_at >= @stat_2017 and serving_at < @stat_2018 ")
print("2017",df_2017.shape)
df_2017.to_csv("/Users/admin/Desktop/史蒂芬大春/办公桌面/编程/python/周报/数据库/rides_2017.csv")

df_2018 = df.query(f"serving_at >= @stat_2018 and serving_at < @stat_2019 ")
print("2018",df_2018.shape)
df_2018.to_csv("/Users/admin/Desktop/史蒂芬大春/办公桌面/编程/python/周报/数据库/rides_2018.csv")

df_2019 = df.query(f"serving_at >= @stat_2019 and serving_at < @stat_2020 ")
print("2019",df_2019.shape)
df_2019.to_csv("/Users/admin/Desktop/史蒂芬大春/办公桌面/编程/python/周报/数据库/rides_2019.csv")

df_2020 = df.query(f"serving_at >= @stat_2020 and serving_at < @stat_2021 ")
print("2020",df_2020.shape)
df_2020.to_csv("/Users/admin/Desktop/史蒂芬大春/办公桌面/编程/python/周报/数据库/rides_2020.csv")

df_2021 = df.query(f"serving_at >= @stat_2021 and serving_at < @stat_2022 ")
print("2021",df_2021.shape)
df_2021.to_csv("/Users/admin/Desktop/史蒂芬大春/办公桌面/编程/python/周报/数据库/rides_2021.csv")

df_2022 = df.query(f"serving_at >= @stat_2022 and serving_at < @stat_2023 ")
print("2022",df_2022.shape)
df_2022.to_csv("/Users/admin/Desktop/史蒂芬大春/办公桌面/编程/python/周报/数据库/rides_2022.csv")

df_2023 = df.query(f"serving_at >= @stat_2023 and serving_at < @stat_2024 ")
print("2023",df_2023.shape)
df_2023.to_csv("/Users/admin/Desktop/史蒂芬大春/办公桌面/编程/python/周报/数据库/rides_2023.csv")

end_time = time.time()
execution_time = end_time - start_time
print("fit_timeflow: ", execution_time/60, "minutes")