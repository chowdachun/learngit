
import  pandas as  pd
import datetime as dt
import numpy as np

df = pd.read_pickle("/Users/shuuomousakai/Desktop/史蒂芬大春/办公桌面/编程/python/周报/数据库/rides_2023.pkl")

end_time = "2023-07-10"
days = 7
star_time = pd.to_datetime(end_time) - dt.timedelta(days=days)
#accepted_at.notnull()
df_1 = df.query(f"serving_at>=@star_time and @end_time>serving_at and canceled_at.notnull() and accepted_at.notnull()")
df_2 = df.query(f"serving_at>=@star_time and @end_time>serving_at and "
                f"accepted_at.isnull() and manual_assigned_count>0 and canceled_at.notnull()")
df = pd.concat([df_1,df_2],axis=0 ,join='outer')
df["开始接驾后取消"] = np.where(df["canceled_at"] >= df["pickup_from"],1,0)
print(df.shape)

df["星期"] = df["created_at"].dt.weekday + 1
df["小时"] = df["created_at"].dt.hour
bins = [-1, 5, 8, 12, 15, 19, 22, 24]
labels = ["凌晨", "早高峰", "中午", "下午", "晚高峰", "晚上", "凌晨"]
df["时间段分段"] = pd.cut(df["小时"], bins=bins, labels=labels, ordered=False)
df.to_csv("/Users/shuuomousakai/Desktop/接单后取消模型111.csv")

data = df.groupby(["operation_center_id", "星期", "时间段分段", "status"]).\
    agg(接驾前取消=("开始接驾后取消",lambda x:(x==0).sum()),
        接驾后取消=("开始接驾后取消",lambda x:(x==1).sum()),)

data.to_csv("/Users/shuuomousakai/Desktop/接单后取消模型222.csv")

# print(df[["pickup_from","canceled_at","开始接驾后取消"]].head(30))