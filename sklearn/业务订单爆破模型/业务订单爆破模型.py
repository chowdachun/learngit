
import pandas as pd
import datetime as dt
import numpy as np


class FlowBom:
    def __init__(self,end_time,days,in_path,to_path):
        self.end_time = end_time
        self.days =days
        self.in_apth =in_path
        self.to_path = to_path
        self.df = pd.read_pickle(in_path)
        self.star_time = None


    def weixiangying(self):
        self.star_time = pd.to_datetime(self.end_time) - dt.timedelta(days=self.days)

        self.df.loc[((self.df["is_prepaid"]==True) & (self.df["prepayment_completed"]==False)),"预付未付款"] =1
        df = self.df.query(f"serving_at >= @self.star_time and @self.end_time > serving_at and "
                      f" accepted_at.isnull() and 预付未付款.isnull() and manual_assigned_count==0 ")

        df["星期"]  = df["created_at"].dt.weekday+1
        df["小时"] = df["created_at"].dt.hour
        bins = [-1, 5, 8, 12 ,15, 19, 22, 24]
        labels = ["凌晨", "早高峰", "中午", "下午", "晚高峰", "晚上", "凌晨"]
        df["时间段分段"] = pd.cut(df["小时"],bins=bins,labels=labels,ordered=False)
        df.loc[df["canceled_at"].isnull(),"canceled_at"] = df["pickup_to"]
        df["等待时长"] = (df["canceled_at"] - df["created_at"]).dt.total_seconds()/60


        data = df.groupby(["operation_center_id","星期","时间段分段","status"]).\
            agg(订单数量 = ("rider_id","count"),
                等待时长sum=("等待时长","sum"),
                等待时长mean=("等待时长","mean"),
                等待时长med=("等待时长","median"),)#.reset_index()

        print(self.star_time,self.end_time)
        return data

    def trips(self):
        df = self.df.query(f"serving_at >= @self.star_time and @self.end_time > serving_at ")
        data = df.groupby(["operation_center_id", "group_id", "status"]).\
            agg(订单组时间头=("pickup_from", "min"),
                订单组人数=("rider_number", "sum"),
                车辆可载人数=("vehicle_seat_number", "max"),)
        data["车辆可载人数"] = data["车辆可载人数"]-1

        data["星期"] = data["订单组时间头"].dt.weekday + 1
        data["小时"] = data["订单组时间头"].dt.hour
        bins = [-1, 5, 8, 12 ,15, 19, 22, 24]
        labels = ["凌晨", "早高峰", "中午", "下午", "晚高峰", "晚上", "凌晨"]
        data["时间段分段"] = pd.cut(data["小时"], bins=bins, labels=labels, ordered=False)
        data = data.reset_index()

        df_yunli = data.groupby(["operation_center_id", "星期", "时间段分段", "status"]). \
            agg(出车趟次=("group_id", "count"),
                载客量=("订单组人数", "sum"),
                可载客量=("车辆可载人数", "sum"), )
        return df_yunli

    def accept_cancel(self):
        df_1 = self.df.query(
            f"serving_at>=@self.star_time and @self.end_time>serving_at and canceled_at.notnull() and accepted_at.notnull()")
        df_2 = self.df.query(f"serving_at>=@self.star_time and @self.end_time>serving_at and "
                        f"accepted_at.isnull() and manual_assigned_count>0 and canceled_at.notnull()")
        df = pd.concat([df_1, df_2], axis=0, join='outer')
        df["开始接驾后取消"] = np.where(df["canceled_at"] >= df["pickup_from"], 1, 0)
        print(df.shape)

        df["星期"] = df["created_at"].dt.weekday + 1
        df["小时"] = df["created_at"].dt.hour
        bins = [-1, 5, 8, 12, 15, 19, 22, 24]
        labels = ["凌晨", "早高峰", "中午", "下午", "晚高峰", "晚上", "凌晨"]
        df["时间段分段"] = pd.cut(df["小时"], bins=bins, labels=labels, ordered=False)

        data = df.groupby(["operation_center_id", "星期", "时间段分段", "status"]). \
            agg(接驾前取消=("开始接驾后取消", lambda x: (x == 0).sum()),
                接驾后取消=("开始接驾后取消", lambda x: (x == 1).sum()), )
        return data


    def df_concat(self):
        a = self.weixiangying()
        b = self.trips()
        c = self.accept_cancel()
        df = pd.concat([a,b,c],axis=1,join='outer')
        df.to_csv(self.to_path)
        print(df.head(30))
        return df



if __name__ == '__main__':
    in_path = "/Users/admin/Desktop/史蒂芬大春/办公桌面/编程/python/周报/数据库/rides_2023.pkl"
    to_path = "/Users/admin/Desktop/业务流程爆破.csv"
    end_time = "2023-08-28"
    days = 7

    case = FlowBom(days=days,end_time=end_time,in_path=in_path,to_path=to_path)
    df = case.df_concat()