
import pandas as pd
import datetime as dt

path = "/Users/admin/Desktop/driver.csv"
df = pd.read_csv(path)
day=30

df = df[["driver_id","serving_at","payments_price"]]
df["payments_price"] = df["payments_price"]/100.0
df['serving_at'] = pd.to_datetime(df['serving_at'], format='%Y-%m-%d', errors='coerce')


def count_d(group):
    group_min = group["serving_at"].min()
    group_30 = group_min + dt.timedelta(days=day)
    sum_driver = group.query(f"serving_at >= @group_min and  serving_at <= @group_30")["payments_price"].sum()
    return sum_driver
data = df.groupby("driver_id").apply(count_d)
print(data)

# data.to_csv("/Users/admin/Desktop/dddddd.csv")

group_min = df.groupby("driver_id")["serving_at"].transform("min")
print(group_min)


