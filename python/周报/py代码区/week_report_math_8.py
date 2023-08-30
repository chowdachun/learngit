
import pandas as pd
import datetime as dt
import week_center as wc
import numpy as np

"""
近8周趋势计数文档，只需要在week_center设置end时间
"""

end = wc.end            # week_center文档end时间
end = pd.to_datetime(end,format='%Y-%m-%d')
star = wc.star
print("******week_report_math_8开始运行******")

# 导入字段中文
arg = pd.read_excel(wc.report_target_path)

# 导入表头  week_center_report = pd.read_csv(wc.week_center_arg_path, index_col=0)
week_center_report_week_8 = pd.read_csv(wc.week_center_arg_path, index_col=0)
# week_center_report = week_center_report.reset_index()

# 选取上周数据表
rides_df_8week = wc.rides_df.query(f"serving_at >= @wc.star and serving_at < @wc.end")       # 上周数据表，用作计算

week_number = 7
week_list_end = []
while True:
    if week_number > -1:
        week =  end - dt.timedelta(days=7*week_number)
        week_list_end.append(week)
        week_number = week_number-1
    else:
        break
week_list_start = []
for i in week_list_end:
    i = i - dt.timedelta(days=7)
    week_list_start.append(i)
print(week_list_start)
print(week_list_end)
print("时间设置开始时间：",star)
print("时间设置结束时间：",end)
print("----"*15)
print("数据表开始时间：",rides_df_8week["serving_at"].min())
print("数据表结束时间：",rides_df_8week["serving_at"].max())

#   8周下单量
week_number = 7
os_add =0
while True:
    if week_number > -1 :
        os_add += 1
        os = "order_" + str(os_add)                         # 设置列名称
        st = list(reversed(week_list_start))[week_number]           # 截取当周起始时间
        et = list(reversed(week_list_end))[week_number]             # 截取当周结束时间
        week_center_report_week_8[os] = rides_df_8week.query(f"serving_at >= @st and serving_at < @et").\
            groupby("operation_center_id").\
            agg(os=("id","count"))
        week_number = week_number - 1
        # print("os:",os)
        # print("st:",st)
        # print("et:",et)
        # print("week_number:",week_number)         # 字段检查
    else:
        print("8周下单量")
        break


#   8周完单量
week_number = 7
os_add =0
while True:
    if week_number > -1 :
        os_add += 1
        os = "order_serving_" + str(os_add)                         # 设置列名称
        st = list(reversed(week_list_start))[week_number]           # 截取当周起始时间
        et = list(reversed(week_list_end))[week_number]             # 截取当周结束时间
        week_center_report_week_8[os] = rides_df_8week.query(f"serving_at >= @st and serving_at < @et").\
            groupby("operation_center_id").\
            agg(os=("actual_dropoff_at","count"))
        week_number = week_number - 1
        # print("os:",os)
        # print("st:",st)
        # print("et:",et)
        # print("week_number:",week_number)         # 字段检查
    else:
        print("8周完单量")
        break

#   8周完成率
week_number = 7
os_add =0
while True:
    if week_number > -1 :
        os_add += 1
        os_completed = "order_serving_" + str(os_add)
        os_count = "order_count_" + str(os_add)
        os = "serving_percent_" + str(os_add)                         # 设置列名称
        st = list(reversed(week_list_start))[week_number]           # 截取当周起始时间
        et = list(reversed(week_list_end))[week_number]             # 截取当周结束时间
        week_center_report_week_8[os_count] = rides_df_8week.query(f"serving_at >= @st and serving_at < @et").\
            groupby("operation_center_id").\
            agg(os=("id","count"))
        week_center_report_week_8[os] = week_center_report_week_8[os_completed] / week_center_report_week_8[os_count]
        week_center_report_week_8.drop(columns=os_count,inplace=True)
        week_number = week_number - 1
        # print("os:",os)
        # print("os_completed:",os_completed)
        # print("st:",st)
        # print("et:",et)
        # print("week_number:",week_number)         # 字段检查
    else:
        print("8周完成率")
        break

#   8周响应时长(接单时间不为空)
week_number = 7
os_add =0
rides_df_8week["respond_time"] = (rides_df_8week["accepted_at"] - rides_df_8week["created_at"]). \
            apply(lambda x: x.total_seconds() / 60)         # 转换为时间数值
while True:
    if week_number > -1 :
        os_add += 1
        os = "respond_time_" + str(os_add)                         # 设置列名称
        st = list(reversed(week_list_start))[week_number]           # 截取当周起始时间
        et = list(reversed(week_list_end))[week_number]             # 截取当周结束时间
        week_center_report_week_8[os] = rides_df_8week.query(f"serving_at >= @st and serving_at < @et and accepted_at.notnull()").\
            groupby("operation_center_id").\
            agg(os=("respond_time","mean"))
        week_number = week_number - 1
    else:
        print("8周响应时长(接单时间不为空)")
        break

#   8周接单量(接单时间不为空)
week_number = 7
os_add =0
while True:
    if week_number > -1 :
        os_add += 1
        os_1 = "accepted_order_" + str(os_add)                         # 设置接单列名称
        st = list(reversed(week_list_start))[week_number]           # 截取当周起始时间
        et = list(reversed(week_list_end))[week_number]             # 截取当周结束时间
        week_center_report_week_8[os_1] = rides_df_8week.query(f"serving_at >= @st and serving_at < @et and accepted_at.notnull()").\
            groupby("operation_center_id").\
            agg(accepted_order = ("respond_time","count"))
        week_number = week_number - 1
    else:
        print("8周接单量(接单时间不为空)")
        break

#   8周接驾准时率（剔除人工派单）
rides_df_8week["order_pickup"] = rides_df_8week["pickup_to"] - rides_df_8week["actual_pickup_at"]
rides_df_8week["order_pickup"] = rides_df_8week["order_pickup"].apply(lambda x: x.total_seconds() / 3600)       # 设置实际上车时间

week_number = 7
os_add =0
while True:
    if week_number > -1 :
        os_add += 1
        os_actual = "order_actual_" + str(os_add)
        os_pickup = "order_pickup_" + str(os_add)
        os = "pickup_punctual_percent_" + str(os_add)                         # 设置列名称
        st = list(reversed(week_list_start))[week_number]           # 截取当周起始时间
        et = list(reversed(week_list_end))[week_number]             # 截取当周结束时间
        week_center_report_week_8[os_pickup] = rides_df_8week.\
            query(f"serving_at >= @st and serving_at <= @et and actual_dropoff_at.notnull() and manual_assigned_count == 0").\
            groupby("operation_center_id").agg(os_pickup=("order_pickup",lambda x:(x >= 0).sum()))
        week_center_report_week_8[os_actual] = rides_df_8week.\
            query(f"serving_at >= @st and serving_at <= @et and actual_dropoff_at.notnull() and manual_assigned_count == 0").\
            groupby("operation_center_id").agg(os_actual=("actual_dropoff_at","count"))

        # week_center_report_week_8[os] = week_center_report_week_8[os_pickup] / week_center_report_week_8[os_actual]

        # week_center_report_week_8.drop(columns=[os_pickup,os_actual], inplace=True)
        week_number = week_number - 1
        # print("os:",os)
        # print("st:",st)
        # print("et:",et)
        # print("week_number:",week_number)         # 字段检查
    else:
        print("8周接驾准时率（剔除人工派单）")
        break

# 上周1-5星评价情况
lastrating_star = end - dt.timedelta(days=7*2)
lastrating_end = end - dt.timedelta(days=7*1)
week_center_report_week_8[["rating_1","rating_2","rating_3","rating_4","rating_5",]]= \
    wc.rides_df.query(f"serving_at >= @lastrating_star and serving_at < @lastrating_end").groupby("operation_center_id").agg(
    rating_1=("rating", lambda x: (x == 1).sum()),  # 48 【1星评价】
    rating_2=("rating", lambda x: (x == 2).sum()),  # 49 【2星评价】
    rating_3=("rating", lambda x: (x == 3).sum()),  # 50 【3星评价】
    rating_4=("rating", lambda x: (x == 4).sum()),  # 51 【4星评价】
    rating_5=("rating", lambda x: (x == 5).sum()),  # 52 【5星评价】
)

# 更换中文target
arg_dict = {}
for a,b in zip(arg["name"],arg["c_name"]):
    arg_dict[a] = b
week_center_report_week_8.rename(columns=arg_dict,inplace=True)



print("####"*15)
week_center_report_week_8.to_csv(wc.week_8_path)
print("******week_report_math_8运行结束******")