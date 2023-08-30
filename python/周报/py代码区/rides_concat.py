import week_center as wc
import pandas as pd
import os
import time


print("******week_roport_file_concat开始运行******")

# 新一周的public.rides文件加进来
new_file = pd.read_csv(wc.new_file_rides_path)  #合并文件名new_file_rides.csv

# 获取时间字段list[]
datetime_list = ['created_at','payments_completed_at','serving_at', 'pickup_from', 'pickup_to','actual_pickup_at', 'actual_dropoff_at','accepted_at',
                 'canceled_at','penalty_paid_at','reminded_at']
print("datetime_list个数：",len(datetime_list))

print("created_at",new_file['created_at'].dtype)
print("serving_at",new_file['serving_at'].dtype)

# 把+08:00删除,注意！+号属于正则表达式，需要在前面加上"\"来表示不需要转义
for name in datetime_list:
    new_file[name] = new_file[name].astype(str).str.replace("\+08:00", "")

# 把Datafram的时间设置为pandas时间格式
new_file['created_at'] = pd.to_datetime(new_file['created_at'],format='%Y-%m-%d',errors='coerce')

for name in datetime_list:
    new_file[name] = pd.to_datetime(new_file[name],format='%Y-%m-%d %H:%M:%S',errors='coerce')
# print(new_rides_df[datetime_list].dtypes)


# 有rides后常规合并
new_rides_df_1 = pd.read_pickle(wc.rides_path)
new_rides_df = pd.concat([new_rides_df_1,new_file],axis=0,ignore_index=True)

# 验证，合并成功会删除文件
new_line = new_rides_df.shape[0]
check_line = new_rides_df_1.shape[0] + new_file.shape[0]
print(new_rides_df.shape[0])
if  new_line - check_line ==0 :
    print("行合并数量相同,合并成功")
    os.remove(wc.new_file_rides_path)
    print("new_file.csv文件删除成功")
else:
    print("合并有误")


# 检查是否有重复行，有重复行删除
new_rides_df = new_rides_df.drop_duplicates(subset=["id"],keep='last')


# 删除重复行验证
new_line = new_rides_df.shape[0]
check_line = new_rides_df_1.shape[0] + new_file.shape[0]
print(new_rides_df.shape[0])
if  new_line - check_line ==0 :
    print("没有重复行")
else:
    print("删除了重复行",new_line - check_line)


# 按订单ID排序
new_rides_df["id"].sort_values()


# 把new_rides_df保存为pkl格式
new_rides_df.to_pickle(wc.rides_path)
print("rides总行数：",new_rides_df.shape)

"""----把新增df同步保存在rides.pkl----"""
t1 = time.time()
#读取总rides表
rides_pkl = pd.read_pickle(wc.total_rides_path)
t3 = time.time()
print("读取总rides表",(t3-t1)/60,"min")

#合并总rides表
rides_pkl = pd.concat([rides_pkl,new_file],axis=0,ignore_index=True)
t4 = time.time()
print("合并总rides表",(t4-t1)/60,"min")

# 检查是否有重复行，有重复行删除
rides_pkl = rides_pkl.drop_duplicates(subset=["id"],keep='last')
t6 = time.time()
print("检查是否有重复行，有重复行删除",(t6-t1)/60,"min")
print(rides_pkl.shape)

rides_pkl.to_pickle(wc.total_rides_path,protocol=4)

t7 = time.time()
print("pkl形式保存总表速度为",(t7-t1)/60,"min")

print("******week_roport_file_concat运行结束******")


