
import  pandas as pd
import os
from collections import Counter

star = pd.to_datetime("2023-07-24")
end = pd.to_datetime("2023-07-31")

df = pd.read_pickle("/Users/admin/Desktop/经营诊断data.pkl")
df["服务日期"] = pd.to_datetime(df["服务日期"])
df = df.query(f"服务日期 >= @star and @end > 服务日期")

data = df.groupby("operation_center_id").agg(下单用户list=("下单乘客list",lambda x: sum(x, []))  ,
                                                     完单乘客list=("完单乘客list",lambda x: sum(x, [])),
                                                     接单司机list=("接单司机list",lambda x: sum(x, [])),
                                                     完单司机list=("完单司机list",lambda x: sum(x, [])),
                                                     接单x天司机趟数=("司机趟次_dict_list",lambda x: sum(x, [])),)

data["下单用户list_bom"] = data["下单用户list"].apply(lambda x:set(x))
data["完单乘客list_bom"] = data["完单乘客list"].apply(lambda x:set(x))
data["接单司机list_bom"] = data["接单司机list"].apply(lambda x:set(x))
data["完单司机list_bom"] = data["完单司机list"].apply(lambda x:set(x))

data["下单用户list_len"] = data["下单用户list_bom"].apply(lambda x:len(x))
data["完单乘客list_len"] = data["完单乘客list_bom"].apply(lambda x:len(x))
data["接单司机list_len"] = data["接单司机list_bom"].apply(lambda x:len(x))
data["完单司机list_len"] = data["完单司机list_bom"].apply(lambda x:len(x))
data.to_csv("/Users/admin/Desktop/经营诊断test3333.csv")
def count_3(x):
    ls = Counter(x)
    con = 0
    for a , b in ls.items():
        if b >3 :
            con +=1
    return con

data["接单>x天司机list_len"] = data["接单司机list"].apply(count_3)

# # 接单>x天司机趟数
# def driver_x(x):
#     dfs = []
#     for i in x:
#         dfff = pd.DataFrame.from_dict(i, orient='index').T
#         dfs.append(dfff)
#     if dfs:
#         df = pd.concat(dfs,axis=0).reset_index()
#         # print(df)
#         data_bridge = df.groupby("driver_id").agg(接单天数=("当日趟次","count"),趟数=("当日趟次","sum"))
#
#         data = data_bridge.loc[data_bridge["接单天数"]>= 4]["趟数"].sum()
#         print(data)
#         return data
#     else:
#         return 0
# data["接单>x天司机趟数"] = df["司机趟次_dict_list"].apply(driver_x)

# 接单>x天司机趟数
def driver_x(x):
    df = pd.DataFrame(x)
    print("dfs",df)
    if len(df)>0:
        data_bridge = df.groupby("driver_id").agg(接单天数=("当日趟次","count"),趟数=("当日趟次","sum"))

        data = data_bridge.loc[data_bridge["接单天数"]>= 4]["趟数"].sum()
        # print(data)
        return data
    else:
        return 0
data["接单>x天司机趟数"] = data["接单x天司机趟数"].apply(driver_x)

print(data)
data.to_csv("/Users/admin/Desktop/经营诊断test.csv")

