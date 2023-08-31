
import pandas as pd
import datetime as dt

data_path = "/Users/admin/Desktop/史蒂芬大春/办公桌面/编程/python/周报/数据库/rides_2023.pkl"
data = pd.read_pickle(data_path)

star = pd.to_datetime("2023-7-24")
end = pd.to_datetime("2023-8-25")       #">"
data["serving_at"] = pd.to_datetime(data["serving_at"])
data = data.query(f"serving_at >= @star and @end > serving_at")
print(data.shape)

df = data.groupby("rider_id").agg(下单量 = ("id","count"),
                                  完单量=("actual_dropoff_at","count"),
                                  流水=("payments_price", lambda x: x.sum() / 100.0))

print(df.shape)

active_uid = pd.read_csv("/Users/admin/Desktop/平台激活用户.csv")  #第一列是用户ID
active_uid = active_uid[["用户ID"]]
active_uid.rename(columns={"用户ID":"rider_id"},inplace=True)
print(active_uid)

df_concat = pd.merge(active_uid,df,on="rider_id",how="left")
print(df_concat)

df_concat.to_csv("/Users/admin/Desktop/激活用户下单量.csv")