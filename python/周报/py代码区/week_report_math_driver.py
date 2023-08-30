
import pandas as pd
import datetime as dt
import week_center as wc

import numpy as np

print("******week_report_math_driver开始运行******")
math_star = wc.end - dt.timedelta(days=7)
end = wc.end

"""
开始计算public.rides
"""


# 选取上周数据表
rides_df_7 = wc.rides_df.query(f"serving_at >= @math_star and serving_at < @wc.end")  # 上周数据表，用作计算


AA = rides_df_7.query("status == 'completed'").groupby(["driver_id","driver_name","capacity_supplier_id","capacity_supplier_name",]).agg(
    pay_price = ("payments_price",lambda x:x.sum()/100.0),  #支付流水
    order_serving = ("id","count"),                         #完单量
    rider_number_serving=("rider_number", "sum"),           #服务人次
    group_driver = ("group_id",lambda x:x.nunique()),       #服务趟次
    serving_passage=("rider_id", lambda x: x.nunique()),    #司机服务乘客人数（去重）
    order_route_completed=("route_id", lambda x: x.nunique())  #完单线路数
)


BB = rides_df_7.groupby(["driver_id","driver_name","capacity_supplier_id","capacity_supplier_name",]).agg(
    order_count = ("id","count"),  #接单量
    order_driver_reassign=("driver_reassign_count", "count"),  #驾驶员改派量
    driver_exceed = ("serving_at",lambda x:x.nunique()),    #出勤天数
    rating_bad = ("rating",lambda x:(3>=x).sum()),           #差评量
    at_dropoff = ("status",lambda x:(x=='at_dropoff').sum()),  #待支付订单量
    order_driver_canceled=("status", lambda x: (x == "driver_canceled").sum()),  #司机取消订单
    order_qrcode=("by_qrcode", lambda x: (x == 1).sum()),       #扫码创建订单
)

# 订单组人数
CC = rides_df_7.groupby(["driver_id","driver_name","capacity_supplier_id","capacity_supplier_name", "group_id"]).agg(
    group_id_number=("rider_number", "sum"), ).reset_index(). \
    groupby(["driver_id","driver_name","capacity_supplier_id","capacity_supplier_name",]).agg(
    group_1=("group_id_number", lambda x: (x == 1).sum()),  # 65 【订单组1人】
    group_2=("group_id_number", lambda x: (x == 2).sum()),  # 66 【订单组2人】
    group_3=("group_id_number", lambda x: (x == 3).sum()),  # 67 【订单组3人】
    group_4=("group_id_number", lambda x: (x == 4).sum()),  # 68 【订单组4人】
    group_5=("group_id_number", lambda x: (x == 5).sum()),  # 69 【订单组5人】
    group_6=("group_id_number", lambda x: (x == 6).sum()),  # 70 【订单组6人】
)
week_driver_report = pd.concat([AA,BB,CC], join="outer", axis=1)

# 完成率
week_driver_report["order_serving_percent"] = week_driver_report["order_serving"] / week_driver_report["order_count"]

# 接驾时长
rides_df_7["time_A"] = rides_df_7.query("actual_dropoff_at.notnull()")["actual_pickup_at"] - rides_df_7.query("actual_dropoff_at.notnull()")["pickup_from"]
rides_df_7["time_late"] = rides_df_7.query("actual_dropoff_at.notnull()")["actual_pickup_at"] - rides_df_7.query("actual_dropoff_at.notnull()")["pickup_to"]
rides_df_7["time_B"] = rides_df_7.query("actual_dropoff_at.notnull()")["actual_dropoff_at"] - rides_df_7.query("actual_dropoff_at.notnull()")["actual_pickup_at"]

rides_df_7["time_A"] = rides_df_7["time_A"].apply(lambda x: x.total_seconds() / 3600)
rides_df_7["time_late"] = rides_df_7["time_late"].apply(lambda x: x.total_seconds() / 3600)
rides_df_7["time_B"] = rides_df_7["time_B"].apply(lambda x: x.total_seconds() / 3600)

time_total_A = rides_df_7.query(f"actual_dropoff_at.notnull() and manual_assigned_count == 0").\
            groupby(["driver_id","driver_name","capacity_supplier_id","capacity_supplier_name",]).agg(
    time_A=("time_A", lambda x: x[rides_df_7["time_A"] >= 0].sum()),
    time_A_count=("time_A", lambda x: x[rides_df_7["time_A"] > -10000].count()),
    order_late= ("time_late",lambda x: x[rides_df_7["time_late"] > 0].count()),
    time_B = ("time_B", lambda x: x[rides_df_7["time_B"] >= 0].sum()),
    time_B_count = ("time_B", lambda x: x[rides_df_7["time_B"] > -10000].count()),
    )
week_driver_report["pickup_time"] = time_total_A["time_A"]*60/time_total_A["time_A_count"]   #【接驾时长（均次）】
week_driver_report["order_late"] = time_total_A["order_late"]/time_total_A["time_A_count"]   #【接驾迟到率】
week_driver_report["distance_time"]  = time_total_A["time_B"]*60/time_total_A["time_B_count"]  #【行程时长（均次）】

# 单均价格
week_driver_report["order_average"] = week_driver_report["pay_price"] / week_driver_report["order_serving"]

# 更换中文target
arg = pd.read_excel(wc.report_target_path)  # 导入字段中文
arg_dict = {}
for a, b in zip(arg["name"], arg["c_name"]):
    arg_dict[a] = b
week_driver_report.rename(columns=arg_dict, inplace=True)
# print(week_driver_report)
week_driver_report.to_csv(wc.week_driver_report_path)

print("******week_report_math_driver运行结束******")