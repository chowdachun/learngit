import pandas as pd
import datetime as dt
import os

"""
运营中心表头提取出
"""
print("******week_center开始运行******")

# input_end = 计算结束天+1
input_end = "2023-08-27"

"""
选取文件路径
"""
#参数路径
args_center_path = os.path.abspath(os.path.join(os.getcwd(),os.pardir,"参数/args_center.xlsx"))
report_target_path = os.path.abspath(os.path.join(os.getcwd(),os.pardir,"参数/report_target.xlsx"))

#数据库路径
passengers_path= os.path.abspath(os.path.join(os.getcwd(),os.pardir,"数据库/passengers.csv"))
rides_path= os.path.abspath(os.path.join(os.getcwd(),os.pardir,"数据库/rides_2023.pkl"))
total_rides_path= os.path.abspath(os.path.join(os.getcwd(),os.pardir,"数据库/rides.pkl"))

#文件输出路径
week_center_arg_path = os.path.abspath(os.path.join(os.getcwd(),os.pardir,"输出文件/week_center_arg.csv"))
week_center_report_path = os.path.abspath(os.path.join(os.getcwd(),os.pardir,"输出文件/week_center_report.csv"))
week_driver_report_path = os.path.abspath(os.path.join(os.getcwd(),os.pardir,"输出文件/week_driver_report.csv"))
week_8_path = os.path.abspath(os.path.join(os.getcwd(),os.pardir,"输出文件/week_8.csv"))
week_route_report_path = os.path.abspath(os.path.join(os.getcwd(),os.pardir,"输出文件/week_route_report.csv"))

#合并文件路径
new_file_passagers_path = os.path.abspath(os.path.join(os.getcwd(),os.pardir,"合并文件/new_file_passagers.csv"))
new_file_rides_path = os.path.abspath(os.path.join(os.getcwd(),os.pardir,"合并文件/new_file_rides.csv"))

#起始、结束时间设置
input_end = pd.to_datetime(input_end, format='%Y-%m-%d')
end = input_end + dt.timedelta(days=1)
end = pd.to_datetime(end, format='%Y-%m-%d')
star = end - dt.timedelta(days=7 * 8)
star = pd.to_datetime(star, format='%Y-%m-%d')

# 读取rides.csv
rides_df = pd.read_pickle(rides_path)
# 把时间设置为pandas时间格式
datetime_list = ['created_at', 'payments_completed_at', 'serving_at', 'pickup_from', 'pickup_to', 'actual_pickup_at',
                 'actual_dropoff_at', 'accepted_at',
                 'canceled_at', 'penalty_paid_at', 'reminded_at']
rides_df['created_at'] = pd.to_datetime(rides_df['created_at'], format='%Y-%m-%d', errors='coerce')
for name in datetime_list:
    rides_df[name] = pd.to_datetime(rides_df[name], format='%Y-%m-%d %H:%M:%S', errors='coerce')

# 新建week_center_report
week_center_report = rides_df.query(f"serving_at >= @star and serving_at < @end"  # 要用@转义
                                    ).groupby("operation_center_id")["id"].count()

# 提取center_name
center_name = pd.read_excel(args_center_path)
center_name = center_name.set_index("id", drop=True)

center_name_dict = {}
for id, center_name in center_name.items():
    center_name_dict[id] = center_name
# print(center_name_dict)
center_name_dict_df = pd.DataFrame(center_name_dict)

# 把center_name合并进week_center_report
week_center_report = pd.merge(week_center_report, center_name_dict_df,
                              how="left",
                              left_index=True, right_index=True)
week_center_report.drop(columns=["id"], inplace=True)
# print(week_center_report.head())                                    # 到此步把运营中心ID=index，名字，省份提取出来

# 检查是否有重复行，有重复行删除
week_center_report = week_center_report.drop_duplicates(subset=["center_name"])

print("week_roport_center：运行结束")
print("----" * 15)

week_center_report.to_csv(week_center_arg_path)

print("数据源起始时间：", rides_df["serving_at"].min())
print("数据源结束时间：", rides_df["serving_at"].max())

print("******week_center运行结束******")
