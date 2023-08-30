
import os
import pandas as pd
import numpy as np
import liner_UserLtvModel as c

df_panding_path = os.path.abspath(os.path.join(os.getcwd(), "data/pandingdata.csv"))
df_panding = pd.read_csv(df_panding_path)

df = df_panding[["serving_at","rider_id","总订单","服务订单","服务行程","流水"]]

"""
数据整理清洗
"""
df["流水"] = df["流水"].fillna(0)
df["serving_at"] = pd.to_datetime(df["serving_at"],errors='coerce')

#选择要预测的数据标签
star_time = '2023-02-01'
end_time = '2023-06-01'     #"<"
pd.to_datetime([star_time,end_time])
df_3m = df.query(f"serving_at >= @star_time and serving_at < @end_time")
print(df_3m)

#计算每个用户的特征
df_uesr = pd.DataFrame(df_3m["rider_id"].unique())
df_uesr.columns=["用户ID"]

df_value = df_3m.groupby("rider_id")["总订单"].sum().reset_index()
df_value.columns = ["用户ID","总订单",]
df_uesr = pd.merge(df_uesr,df_value[["用户ID","总订单"]],on="用户ID")

df_value = df_3m.groupby("rider_id")["服务订单"].sum().reset_index()
df_value.columns = ["用户ID","服务订单",]
df_uesr = pd.merge(df_uesr,df_value[["用户ID","服务订单"]],on="用户ID")

df_value = df_3m.groupby("rider_id")["流水"].sum().reset_index()
df_value.columns = ["用户ID","流水",]
df_uesr = pd.merge(df_uesr,df_value[["用户ID","流水"]],on="用户ID")

#数值越大，最近活跃程度越高
df_value = df_3m.groupby("rider_id")["serving_at"].max().reset_index()
df_value.columns = ["用户ID","最近服务时间",]
df_value["最近乘车天数"] = ( df_value['最近服务时间'] -df_value['最近服务时间'].min() ).dt.days
df_uesr = pd.merge(df_uesr,df_value[["用户ID","最近乘车天数"]],on="用户ID")


"""
把特征筛选出来
"""
x_pred = df_uesr.drop(["用户ID",],axis=1)

#把预测值放到df_user中
df_uesr["model_pred"] = c.model_lr.predict(x_pred)

df_uesr["pred"] = df_uesr["model_pred"] -df_uesr["流水"]

print(df_uesr["pred"].sum())
# print(df_uesr)

