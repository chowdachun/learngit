import week_center as wc
import pandas as pd
import os

print("****passengers_test开始运行！****")


passagers = pd.read_csv(wc.passengers_path)
passagers_add = pd.read_csv(wc.new_file_passagers_path)     #文件名new_file_passagers.csv

# 新、旧passenger表合并
passagers_new = pd.concat([passagers,passagers_add],axis=0,ignore_index=True)

# 检验增加新表的行数合计
new_line = passagers_new.shape[0]
check_line = passagers.shape[0] + passagers_add.shape[0]
# print(new_rides_df.shape[0])
if  new_line - check_line == 0 :
    print("行合并数量相同,合并成功")
    os.remove(wc.new_file_passagers_path)
    print("new_file.csv文件删除成功")
else:
    print("合并有误")

# 检查是否有重复行，有重复行删除
passagers_new.sort_values(["id","first_order_at"],inplace=True)
passagers_new.drop_duplicates(subset=["id"],keep='first',inplace=True)

# 删除重复行验证
# new_line = passagers_new.shape[0]
# check_line = passagers.shape[0] + passagers_add.shape[0]
# print(passagers_new.shape[0])
# if  new_line - check_line ==0 :
#     print("没有重复行")
# else:
#     print("删除了重复行",new_line - check_line)

# 获取时间字段list[]
datetime_list = ['first_order_at','last_order_at','created_at','updated_at','first_sign_in_at','last_sign_in_at']

# 把+08:00删除,注意！+号属于正则表达式，需要在前面加上"\"来表示不需要转义
for name in datetime_list:
    passagers_new[name] = passagers_new[name].str.replace("\+08:00", "",regex=True)
# print(passagers_new[datetime_list].head())

# 把Datafram的时间设置为pandas时间格式
for name in datetime_list:
    passagers_new[name] = pd.to_datetime(passagers_new[name],format='%Y-%m-%d %H:%M:%S',errors='coerce')
# print(passagers_new[datetime_list].dtypes)

passagers_new.to_csv(wc.passengers_path,index=False)
print("最大id为:",passagers_new["id"].max())

print("passagers_concat 运行结束！")