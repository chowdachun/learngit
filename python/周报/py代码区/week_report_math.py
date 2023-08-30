import pandas as pd
import datetime as dt
import week_center as wc
import numpy as np

"""
字段计数文档，只需要在week_center设置end时间
"""
print("******week_report_math开始运行******")
math_star = wc.end - dt.timedelta(days=7)
end = wc.end

"""# @wc.end  @math_star
开始计算public.passengers
"""

df = pd.read_csv(wc.passengers_path)
datetime_list = ['first_order_at', 'last_order_at', 'created_at', 'updated_at', 'first_sign_in_at', 'last_sign_in_at']
for name in datetime_list:
    df[name] = pd.to_datetime(df[name], format='%Y-%m-%d %H:%M:%S', errors='coerce')
total_riders = df.query(f"first_order_at >= '2018-01-01' and first_order_at < @end"). \
    groupby("first_order_operation_center_id"). \
    agg(total_rider=("id", "count")).fillna(0)  # 【总用户】
new_week_rider = df.query(f"first_order_at >= @math_star and first_order_at < @end"). \
    groupby("first_order_operation_center_id"). \
    agg(new_week_rider=("id", "count")).fillna(0)   # 【本周新增用户】

"""
开始计算public.rides
"""
# 导入表头
week_center_report = pd.read_csv(wc.week_center_arg_path, index_col=0)

# 选取上周数据表
rides_df_7 = wc.rides_df.query(f"serving_at >= @math_star and serving_at < @wc.end")  # 上周数据表，用作计算

# 没有状态筛选
null_status = rides_df_7.groupby("operation_center_id").agg(
    order_count=("id", "count"),  # 4 【下单量】
    order_expired=("status", lambda x: (x == "expired").sum()),  # 10 【过期订单】
    order_passenger_canceled=("status", lambda x: (x == "passenger_canceled").sum()),  # 12 【乘客取消订单】
    order_driver_canceled=("status", lambda x: (x == "driver_canceled").sum()),  # 13 【司机取消订单】
    order_console_canceled=("status", lambda x: (x == "console_canceled").sum()),  # 14 【后台取消订单】
    order_console=("client", lambda x: (x == "backend").sum()),  # 18 【后台创建订单】
    order_qrcode=("by_qrcode", lambda x: (x == 1).sum()),  # 19 【扫码创建订单】
    order_assigned=("manual_assigned_count", lambda x: (x > 0).sum()),  # 35 【人工调单量】
    order_driver_reassign=("driver_reassign_count", "sum"),  # 37 【驾驶员改派量】
    order_console_reassign=("admin_reassign_count", "sum"),  # 39 【后台改派量】
    rider_order=("rider_id", lambda x: x.nunique()),  # 43 【下单总用户】
    rating_1=("rating", lambda x: (x == 1).sum()),  # 48 【1星评价】
    rating_2=("rating", lambda x: (x == 2).sum()),  # 49 【2星评价】
    rating_3=("rating", lambda x: (x == 3).sum()),  # 50 【3星评价】
    rating_4=("rating", lambda x: (x == 4).sum()),  # 51 【4星评价】
    rating_5=("rating", lambda x: (x == 5).sum()),  # 52 【5星评价】
    order_app = ("client",lambda x:(x=='app').sum()),     #【App下单量】
    order_prepaid = ("is_prepaid",lambda x:(x==1).sum()),       #【预付订单量】
    driver=("driver_id", lambda x: x.nunique()),  # 55  【出勤驾驶员】
    accepted_order=("id", lambda x: x[rides_df_7["accepted_at"].notnull()].count()),  # 4 【已接订单】部分人工调单没有接单时间
)
null_status["accepted_order_A"]=rides_df_7.query("manual_assigned_count >0 and accepted_at.isnull()").groupby("operation_center_id").\
    agg(accepted_order_A=("id", "count"))
null_status["accepted_order"] = null_status["accepted_order_A"].fillna(0) + null_status["accepted_order"]  # 【已接订单(完整)】
null_status.drop(columns=["accepted_order_A"], inplace=True)


#   完成订单状态
completed_status = rides_df_7.query("actual_dropoff_at.notnull()").groupby("operation_center_id").agg(
    pay_price=("payments_price", "sum"),            # 1 【流水】
    order_serving=("actual_dropoff_at", "count"),  # 2 【完单量】
    rider_number_serving=("rider_number", "sum"),  # 3 【服务人次】
    order_console_completed=("client", lambda x: (x == "backend").sum()),  # 20 【后台创建完单量】
    order_qrcode_completed=("by_qrcode", lambda x: (x == 1).sum()),  # 21 【扫码创建完单量】
    rider_order_completed=("rider_id", lambda x: x.nunique()),  # 44  【完单用户数】
    driver_completed=("driver_id", lambda x: x.nunique()),  # 56  【完单驾驶员】
    order_prepaid_completed=("is_prepaid", lambda x: (x == 1).sum()),  # 【预付完单量】
)
completed_status["pay_price"] = completed_status["pay_price"] / 100.0

# 9 【订单平均支付时长(h)】,apply()后需要产生新的列
rides_df_7["order_paytime"] = rides_df_7.loc[rides_df_7["status"] == "completed", :]["payments_completed_at"] \
                              - rides_df_7.loc[rides_df_7["status"] == "completed", :]["actual_pickup_at"]
rides_df_7["order_paytime_1"] = rides_df_7["order_paytime"].apply(lambda x: x.total_seconds() / 3600)  # 设置为小时H
print("####" * 15)
order_paytime_mean = rides_df_7.groupby("operation_center_id").agg(
    order_paytime_sum_1=("order_paytime", "sum"),
    order_serving_2=("payments_completed_at", "count"))
order_paytime_mean["order_paytime_mean"] = order_paytime_mean["order_paytime_sum_1"] / order_paytime_mean[
    "order_serving_2"]
order_paytime_mean["order_paytime_mean"] = order_paytime_mean["order_paytime_mean"].apply(
    lambda x: x.total_seconds() / 3600)
# 删除order_paytime_mean后面列删除，只保留"order_paytime_mean"
order_paytime_mean.drop(columns=["order_paytime_sum_1", "order_serving_2"], inplace=True)

# 26 【预约订单量】
order_booking_df = rides_df_7.loc[((rides_df_7["pickup_from"] - rides_df_7["created_at"]).
                                   apply(lambda x: x.total_seconds() / 3600)) > 1]. \
    groupby("operation_center_id").agg(
    order_booking=("id", "count")
)
# 29 【预约订单完成量】
order_booking_completed_df = rides_df_7.loc[(((rides_df_7["pickup_from"] - rides_df_7["created_at"]).
                                              apply(lambda x: x.total_seconds() / 3600)) > 1)
                                            & (rides_df_7["status"] == "completed")]. \
    groupby("operation_center_id").agg(
    order_booking_completed=("id", "count")
)

# 33 【响应时长min】
rides_df_7["respond_time"] = (rides_df_7["accepted_at"] - rides_df_7["created_at"]).dt.total_seconds()/60 # 转换为min
    # apply(lambda x: x.total_seconds() / 60)
respond_time = rides_df_7.loc[rides_df_7["accepted_at"].notnull()].groupby("operation_center_id"). \
    agg(respond_time=("respond_time", "sum"),
        respond_time_count=("respond_time", "count"))
respond_time["respond_time_mean"] = respond_time["respond_time"] / respond_time["respond_time_count"]
respond_time.drop(columns=["respond_time", "respond_time_count"], inplace=True)

# 34 【接驾】
rides_df_7["order_pickup"] = (rides_df_7["pickup_to"] - rides_df_7["actual_pickup_at"]).dt.total_seconds()/3600
# rides_df_7["order_pickup"] = rides_df_7["order_pickup"].apply(lambda x: x.total_seconds() / 3600)       # 设置实际上车时间

pickup_punctual = rides_df_7.\
            query(f"actual_dropoff_at.notnull() and manual_assigned_count == 0").\
            groupby("operation_center_id").agg(pickup_punctual=("order_pickup",lambda x:(x >= 0).sum()))      #34 【接驾准时量】
pickup_punctual["os_actual"] = rides_df_7. \
    query(f"actual_dropoff_at.notnull() and manual_assigned_count == 0"). \
    groupby("operation_center_id").agg(os_actual=("actual_dropoff_at", "count"))
pickup_punctual["pickup_punctual_percent"] = pickup_punctual["pickup_punctual"] / pickup_punctual["os_actual"]     #34 【接驾准时率%】
pickup_punctual.drop(columns=["os_actual"], inplace=True)

# 45 【复乘用户数】
rider_mulite_1 = rides_df_7.loc[rides_df_7["status"] == "completed"]. \
    groupby(["operation_center_id", "rider_id"]). \
    agg(completed_count=("id", "count")).reset_index()
rider_mulite = rider_mulite_1.groupby("operation_center_id"). \
    agg(rider_mulite=("completed_count", lambda x: (x >= 2).sum()))

# 司机服务状态
driver_exceed_3_status = rides_df_7.groupby(["operation_center_id", "driver_id"]).agg(
    driver_exceed_3=("serving_at", lambda x: x.nunique()),  # 计算司机的服务天数
    rider_number_completed=("rider_number", lambda x: x[rides_df_7["status"] == "completed"].sum()),  # 计算订单完成状态的服务人数
    rider_group_completed=("group_id", lambda x: x.nunique()),       # 计算去重的订单组
    driver_2300=("payments_price",lambda x: x[rides_df_7["status"] == "completed"].sum()/100)
    ).reset_index()
driver_exceed_3 = driver_exceed_3_status.groupby(["operation_center_id"]).agg(
    driver_exceed_3=("driver_exceed_3", lambda x: (x > 3).sum()),  # 57 【出勤司机>3天】
    rider_number_completed=("rider_number_completed", lambda x: x[driver_exceed_3_status["driver_exceed_3"] > 3].sum()),
    # 58  【出勤司机>3天的服务人次】
    rider_group_completed=("rider_group_completed", lambda x: x[driver_exceed_3_status["driver_exceed_3"] > 3].sum()),
    # xx  【司机流水>=2300】
    driver_2300=("driver_2300", lambda x: (x >= 2300).sum())
)

# 订单组人数
group_number = rides_df_7.groupby(["operation_center_id", "group_id"]).agg(
    group_id_number=("rider_number", "sum"), ).reset_index(). \
    groupby("operation_center_id").agg(
    group_1=("group_id_number", lambda x: (x == 1).sum()),  # 65 【订单组1人】
    group_2=("group_id_number", lambda x: (x == 2).sum()),  # 66 【订单组2人】
    group_3=("group_id_number", lambda x: (x == 3).sum()),  # 67 【订单组3人】
    group_4=("group_id_number", lambda x: (x == 4).sum()),  # 68 【订单组4人】
    group_5=("group_id_number", lambda x: (x == 5).sum()),  # 69 【订单组5人】
    group_6=("group_id_number", lambda x: (x == 6).sum()),  # 70 【订单组6人】
)

# 接单状态
accepted_status = rides_df_7.query("accepted_at.notnull()").groupby("operation_center_id").\
    agg(accepted_order_backend=("client", lambda x:(x == "backend").sum()),         #【后台订单接单量】人工调单接单时间为空
        accepted_order_qrcode=("by_qrcode", lambda x:(x == 1).sum()),          #【扫码订单接单量】
)
accepted_status["accepted_order_backend_A"]=rides_df_7.query("client=='backend' and manual_assigned_count >0 and accepted_at.isnull()").groupby("operation_center_id").\
    agg(accepted_order_backend_A=("id", "count"))
accepted_status["accepted_order_backend"]=accepted_status["accepted_order_backend_A"].fillna(0)+accepted_status["accepted_order_backend"]
accepted_status.drop(columns=["accepted_order_backend_A"], inplace=True)

"""
在此处做Dataframe合并
"""
columns_list = [null_status, completed_status, order_paytime_mean, order_booking_df, group_number,
                order_booking_completed_df, respond_time, pickup_punctual, rider_mulite, driver_exceed_3, total_riders,
                new_week_rider, accepted_status]
# 合并多个数据表
for df in columns_list:
    week_center_report = pd.concat([week_center_report, df], join="outer", axis=1)  # 把每个表的index都设置相同

"""
合并后的数据表计算
"""
# 5 【接单比例% 】(响应比例)
week_center_report["accepted_order_percent"] = week_center_report["accepted_order"] / week_center_report["order_count"]
# 5 【后台创建接单比例% 】
week_center_report["accepted_order_percent_backend"] = \
    week_center_report["accepted_order_backend"] / week_center_report["order_console"]


# 6 【接驾订单量】（取消时间>时间头 + 总完成订单，还要把已接单的筛出来）
week_center_report[["pickup_count","pickup_backend","pickup_qrcode"]] = rides_df_7. \
    query("canceled_at >= pickup_from and accepted_at.notnull() and actual_pickup_at.isnull()"). \
    groupby("operation_center_id"). \
    agg(pickup_count=("id", "count"),
        pickup_backend=("client",lambda x:(x == "backend").sum()),
        pickup_qrcode=("by_qrcode",lambda x:(x == 1).sum()))
week_center_report["pickup_count"] = week_center_report["pickup_count"].fillna(0) + week_center_report["order_serving"]
week_center_report["pickup_backend"] = week_center_report["pickup_backend"].fillna(0) + week_center_report["order_console_completed"]  # 后台创建订单接驾量
week_center_report["pickup_qrcode"] = week_center_report["pickup_qrcode"].fillna(0) + week_center_report["order_qrcode_completed"]   # 扫码订单接驾量


# 7 【接驾比例%】
week_center_report["pickup_count_percent"] = week_center_report["pickup_count"] / week_center_report["order_count"]

# 8 【完成率】
week_center_report["order_serving_percent"] = week_center_report["order_serving"] / week_center_report["order_count"]

# 11 【过期比例%】
week_center_report["order_expired_present"] = week_center_report["order_expired"] / week_center_report["order_count"]

# 15 【乘客取消比例%】
week_center_report["order_passenger_canceled_present"] = \
    week_center_report["order_passenger_canceled"] / week_center_report["order_count"]
# 16 【司机取消比例%】
week_center_report["order_driver_canceled_present"] = \
    week_center_report["order_driver_canceled"] / week_center_report["order_count"]
# 17 【后台取消比例%】
week_center_report["order_console_canceled_present"] = \
    week_center_report["order_console_canceled"] / week_center_report["order_count"]

# 22 后台创建比例
week_center_report["order_console_present"] = \
    week_center_report["order_console"] / week_center_report["order_count"]
# 23 后台创建完成率
week_center_report["order_console_completed_present"] = \
    week_center_report["order_console_completed"] / week_center_report["order_console"]
# 24 扫码创建比例
week_center_report["order_qrcode_present"] = \
    week_center_report["order_qrcode"] / week_center_report["order_count"]
# 25 扫码创建完成率
week_center_report["order_qrcode_completed_present"] = \
    week_center_report["order_qrcode_completed"] / week_center_report["order_qrcode"]

# 27 【实时订单量】
week_center_report["order_right"] = week_center_report["order_count"] - week_center_report["order_booking"]
# 27 【实时订单比例%】
week_center_report["order_right_percent"] = week_center_report["order_right"] / week_center_report["order_count"]
# 28 【预约订单比例%】
week_center_report["order_booking_percent"] = week_center_report["order_booking"] / week_center_report["order_count"]
# 30 【实时完成订单量】
week_center_report["order_right_completed"] = week_center_report["order_serving"] - week_center_report[
    "order_booking_completed"]
# 31 【实时订单完成比例%】
week_center_report["order_right_completed_percent"] = week_center_report["order_right_completed"] / \
                                                      week_center_report["order_right"]
# 32 【预约订单完成比例%】
week_center_report["order_booking_completed_percent"] = week_center_report["order_booking_completed"] / \
                                                        week_center_report["order_booking"]

# 36 【人工调单量比例%】
week_center_report["order_assigned_percent"] = week_center_report["order_assigned"] / week_center_report["order_count"]
# 38 【驾驶员改派比例%】
week_center_report["order_driver_reassign_percent"] = week_center_report["order_driver_reassign"] / week_center_report[
    "order_count"]
# 40 【后台改派率%】
week_center_report["order_console_reassign_percent"] = week_center_report["order_console_reassign"] / \
                                                       week_center_report["order_count"]
# 41 【总调度量】
week_center_report["total_operate"] = \
    week_center_report["order_assigned"] + week_center_report["order_driver_reassign"] + week_center_report[
        "order_console_reassign"]
# 42 【总调度率%】
week_center_report["total_operate_percent"] = week_center_report["total_operate"] / week_center_report["order_count"]

# 45 【完单用户比例%】乘车率
week_center_report["rider_order_completed_percent"] = \
    week_center_report["rider_order_completed"] / week_center_report["rider_order"]
# 47 【复乘用户比例%】复乘率
week_center_report["rider_mulite_percent"] = week_center_report["rider_mulite"] / week_center_report[
    "rider_order_completed"]

# 53 【评价总数】
week_center_report = week_center_report. \
    assign(rating_total=lambda x: x["rating_1"] + x["rating_2"] + x["rating_3"] + x["rating_4"] + x["rating_5"])
# 54 【评价率%】
week_center_report["rating_percent"] = week_center_report["rating_total"] / week_center_report["order_serving"]
# 54 【订单好评率】
week_center_report["rating_best_percent"] = \
    (week_center_report["rating_1"] + week_center_report["rating_2"] + week_center_report["rating_3"]) / \
    week_center_report["rating_total"]

# 59 【上座数/趟】
week_center_report["group_number"] = week_center_report["rider_number_completed"] / week_center_report[
    "rider_group_completed"]
# 60 【每天服务趟数/车】
week_center_report["group_driver_day"] = week_center_report["rider_group_completed"] / week_center_report[
    "driver_exceed_3"] / 7

# 61 【单均价格】
week_center_report["order_average"] = week_center_report["pay_price"] / week_center_report["order_serving"]
# 62 【人均价格】
week_center_report["order_number_average"] = week_center_report["pay_price"] / week_center_report[
    "rider_number_serving"]
# 63 【用户人均价格】
week_center_report["rider_order_average"] = week_center_report["pay_price"] / week_center_report[
    "rider_order_completed"]

# 【接驾时长&行程时长】
rides_df_7["time_A"] = rides_df_7.query("actual_dropoff_at.notnull()")["actual_pickup_at"] - rides_df_7.query("actual_dropoff_at.notnull()")["pickup_from"]
rides_df_7["time_B"] = rides_df_7.query("actual_dropoff_at.notnull()")["actual_dropoff_at"] - rides_df_7.query("actual_dropoff_at.notnull()")["actual_pickup_at"]
rides_df_7["time_A"] = rides_df_7["time_A"].apply(lambda x: x.total_seconds() / 3600)
rides_df_7["time_B"] = rides_df_7["time_B"].apply(lambda x: x.total_seconds() / 3600)

time_total_A = rides_df_7.query(f"actual_dropoff_at.notnull() and manual_assigned_count == 0").\
            groupby("operation_center_id").agg(
    time_A = ("time_A",lambda x:x[rides_df_7["time_A"] >= 0].sum()),
    time_A_count = ("time_A",lambda x:x[rides_df_7["time_A"]>-10000].count()),
    time_B = ("time_B",lambda x:x[rides_df_7["time_B"] >= 0].sum()),
    time_B_count = ("time_B",lambda x:x[rides_df_7["time_B"]>-10000].count()),
)
week_center_report["pickup_time"] = time_total_A["time_A"]*60/time_total_A["time_A_count"]   #【接驾时长（均次）】
week_center_report["distance_time"]  = time_total_A["time_B"]*60/time_total_A["time_B_count"]  #【行程时长（均次）】

# 【预付订单未支付】
week_center_report["not_prepaid"] = rides_df_7.query("is_prepaid ==1 and prepayment_completed == 0 ").groupby("operation_center_id").agg(
    not_prepaid = ("id","count"),
)

# 更换中文target
arg = pd.read_excel(wc.report_target_path)  # 导入字段中文
arg_dict = {}
for a, b in zip(arg["name"], arg["c_name"]):
    arg_dict[a] = b
week_center_report.rename(columns=arg_dict, inplace=True)


week_center_report.to_csv(wc.week_center_report_path)


print("rides_df_7起始时间：", rides_df_7["serving_at"].min())
print("rides_df_7结束时间：", rides_df_7["serving_at"].max())
print("rides_df_7总行数：", rides_df_7.shape[0])
print("******week_report_math运行结束******")
