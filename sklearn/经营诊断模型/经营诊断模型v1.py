
import pandas as pd
import os
import numpy as np
import datetime as dt

class BusinessDiagnosticModelCount:
    def __init__(self,star,end,df_path):
        self.star = pd.to_datetime(star, format='%Y-%m-%d')
        self.end = pd.to_datetime(end, format='%Y-%m-%d')
        self.group = ["operation_center_id"]    #"serving_at",
        self.df = pd.read_pickle(df_path).query(f"serving_at >= @self.star and serving_at < @self.end")
        self.rider_tiem_value_df = None


    #供需侧
    def demand(self,df):
        data = df.groupby(self.group).agg(完单量=("actual_dropoff_at", "count"),
                                               接单量=("id", lambda x: x[df["accepted_at"].notnull()].count()),
                                               下单量=("id","count"),
                                               app下单量 = ("client",lambda x:(x=='app').sum()),
                                               后台下单量 = ("client", lambda x: (x == "backend").sum()),
                                               扫码下单量 =("by_qrcode", lambda x: (x == 1).sum()),
                                               )
        data["流水"] = df.query("actual_dropoff_at.notnull()").groupby(self.group).agg(流水 = ("payments_price",lambda x:(x/100.0).sum()))
        data["小程序下单量"] = data["下单量"] - data["app下单量"] - data["后台下单量"] - data["扫码下单量"]
        # 接驾订单量（取消时间>时间头 + 总完成订单，还要把已接单的筛出来）
        data["接驾订单量"] =df.query("canceled_at >= pickup_from and accepted_at.notnull() and actual_pickup_at.isnull()"). \
            groupby(self.group). \
            agg(接驾订单量=("id", "count"),)
        data["接驾订单量"] = data["接驾订单量"].fillna(0) + data["完单量"]

        # 完整已接订单（部分人工派单没人接单时间）
        data["accepted_order_A"] =df.query("manual_assigned_count >0 and accepted_at.isnull()").\
            groupby(self.group). \
            agg(accepted_order_A=("id", "count"))
        data["接单量"] = data["accepted_order_A"].fillna(0) + data["接单量"]  # 【已接订单(完整)】
        data.drop(columns=["accepted_order_A"], inplace=True)

        # 未响应订单量
        df.loc[((df["is_prepaid"]==True) & (df["prepayment_completed"]==False)),"预付未付款"] =1
        df_weixiangying = df.query("accepted_at.isnull() and 预付未付款.isnull() and manual_assigned_count==0")
        data["未响应订单量"] = df_weixiangying.groupby(self.group).agg(未响应订单量 = ("id","count")).fillna(0)

        data["未响应订单_后台取消"]=df_weixiangying.loc[df_weixiangying["status"]=="console_canceled"].\
            groupby(self.group).agg(未响应订单_后台取消 = ("id","count"))

        # 接驾准时量/率 (剔除人工派单)
        df["order_pickup"] = (df["pickup_to"] - df["actual_pickup_at"]).dt.total_seconds() / 3600
        df_jiejiazhunshi = df.query("actual_dropoff_at.notnull() and manual_assigned_count == 0"). \
            groupby(self.group).agg(接驾准时量=("order_pickup", lambda x: (x >= 0).sum()))
        df_jiejiazhunshi["总接驾量"] = df.query("canceled_at >= pickup_from and accepted_at.notnull() and actual_pickup_at.isnull()"). \
            groupby(self.group).agg(os_actual=("id", "count"))
        df_jiejiazhunshi["总接驾量"] = df_jiejiazhunshi["总接驾量"].fillna(0) + data["完单量"]
        df_jiejiazhunshi["接驾准时率"]=df_jiejiazhunshi["接驾准时量"]/df_jiejiazhunshi["总接驾量"]  #【接驾准时率%】
        # df_jiejiazhunshi.drop(columns=["总接驾量"], inplace=True)
        # data = pd.concat([data,df_jiejiazhunshi],join='outer',axis=1)   # 合并

        # 响应时长(min)
        df["respond_time"]=(df["accepted_at"]-df["created_at"]).dt.total_seconds()/60  # 转换为min

        respond_time = df.loc[df["accepted_at"].notnull()].groupby(self.group). \
            agg(总响应时长=("respond_time", "sum"),
                响应订单量=("respond_time", "count"))
        respond_time["响应时长"] = respond_time["总响应时长"] / respond_time["响应订单量"]
        # respond_time.drop(columns=["总响应时长", "响应订单量"], inplace=True)

        # 接驾时长&行程时长
        df["接驾时间"] = (df.query("actual_dropoff_at.notnull()")["actual_pickup_at"]
                -df.query("actual_dropoff_at.notnull()")["pickup_from"]).dt.total_seconds()/60
        df["行程时间"] = (df.query("actual_dropoff_at.notnull()")["actual_dropoff_at"]
                -df.query("actual_dropoff_at.notnull()")["actual_pickup_at"]).dt.total_seconds()/60
        df_pick_trip =df.query(f"actual_dropoff_at.notnull() and manual_assigned_count == 0"). \
            groupby("operation_center_id").agg(
            接驾总时长=("接驾时间", lambda x: x[df["接驾时间"] >= 0].sum()),
            接驾总单量=("接驾时间", lambda x: x[df["接驾时间"] > -10000].count()),
            行程总时长=("行程时间", lambda x: x[df["行程时间"] >= 0].sum()),
            行程总单量=("行程时间", lambda x: x[df["行程时间"] > -10000].count()),
        )

        # 接驾前取消量&接驾后取消量
        df_jiejiaquxiao_1 = df.query(f"canceled_at.notnull() and accepted_at.notnull()")  # 筛选出取消时间为空&截单时间不为空
        df_jiejiaquxiao_2 = df.query(f"accepted_at.isnull() and manual_assigned_count>0 and canceled_at.notnull()")  # 筛选出人工调单没有接单时间
        df_jiejiaquxiao = pd.concat([df_jiejiaquxiao_1, df_jiejiaquxiao_2], axis=0, join='outer')
        df_jiejiaquxiao["开始接驾后取消"] = np.where(df_jiejiaquxiao["canceled_at"] >= df_jiejiaquxiao["pickup_from"], 1, 0)

        df_jjqx = df_jiejiaquxiao.groupby("operation_center_id").agg(接驾前取消=("开始接驾后取消", lambda x: (x == 0).sum()),
                                                                    接驾后取消=("开始接驾后取消", lambda x: (x == 1).sum()))

        data = pd.concat([data, respond_time,df_jiejiazhunshi,df_pick_trip,df_jjqx], join='outer', axis=1)  # 合并
        return data

    #运力测
    def capacity(self,df):
        data = df.groupby(self.group).agg(趟数=("group_id",lambda x:x.nunique()),)

        df_shangzuolv=df.query(f"actual_dropoff_at.notnull()").\
            groupby(self.group+["group_id"]).agg(订单组人数=("rider_number", "sum"),
                                                荷载人数=("vehicle_seat_number", lambda x:x.max()-1))
        shangzuolv= df_shangzuolv.groupby(self.group).agg(总人数=("订单组人数","sum"),
                                                        总荷载人数=("荷载人数","sum"))
        data["上座率"] = shangzuolv["总人数"] / shangzuolv["总荷载人数"]
        data["总人数"] = shangzuolv["总人数"]
        data["总荷载人数"] = shangzuolv["总荷载人数"]

        return data

    # 经营侧
    def profit(self,df):
        # 需求满足率
        df_xuqiumanzulv = df.groupby(["operation_center_id","route_id","rider_id"])\
            .agg(当天完单量=("actual_dropoff_at",lambda x:x[df["actual_dropoff_at"].notnull()].count())).reset_index()
        data = df_xuqiumanzulv.groupby(self.group).agg(需求下单用户=("rider_id","count"),
                                                       需求未满足用户=("当天完单量",lambda x:(x==0).sum()))

        return data

    # 通过总订单数据获取用户触发时间
    def rider_tiem_value(self):
        parent_folder = os.path.dirname(os.getcwd())
        grandparent_folder = os.path.dirname(parent_folder)
        df = pd.read_pickle(os.path.abspath(os.path.join(grandparent_folder,"python/周报/数据库/rides.pkl")))
        df = df[["operation_center_id","rider_id","created_at","actual_pickup_at","payments_promotion"]]

        datetime_list = ['created_at', 'actual_pickup_at',]

        df['created_at'] = pd.to_datetime(df['created_at'], format='%Y-%m-%d', errors='coerce')
        for name in datetime_list:
            df[name] = pd.to_datetime(df[name], format='%Y-%m-%d %H:%M:%S', errors='coerce')

        data_A = df.groupby(["operation_center_id","rider_id"]).agg(最早下单时间=("created_at", "min"),
                                                                    最早上车时间=("actual_pickup_at", "min"),
                                                                    最近下单时间=("created_at", "max"),)
        data_B =  df.query("payments_promotion.notnull()").groupby(["operation_center_id","rider_id"]).\
                                                                 agg(最早用券时间=("actual_pickup_at","min"))
        data_concat = pd.concat([data_A,data_B], join="outer", axis=1,)
        data_concat["首单用券"] = np.where(data_concat["最早用券时间"] == data_concat["最早上车时间"], 1, 0)

        self.rider_tiem_value_df = data_concat.reset_index()
        return data_concat

    # 基础数据整合
    def basedata_concat(self,df):
        a = self.demand(df)
        b = self.capacity(df)
        c = self.profit(df)
        df = pd.concat([a,b,c],join='outer',axis=1)

        return df

    # 遍历serving_at
    def x(self):
        serving_at_ls = self.df["serving_at"].astype(str).unique().tolist()
        d = self.rider_tiem_value()

        big_data = pd.DataFrame()
        # 遍历日期
        for a in serving_at_ls:
            a = pd.to_datetime(a)
            print(a)
            df = self.df.query(f"serving_at == @a ")
            print(df.shape)
            data = self.basedata_concat(df)
            data["服务日期"] = a
            b = a+dt.timedelta(days=1)
            data["新增新用户"] = self.rider_tiem_value_df.query(f"最早下单时间>=@a and @b>最早下单时间 ").\
                groupby("operation_center_id").agg(新增新用户=("最早下单时间","count"))

            # 遍历服务中心
            op_ls = df["operation_center_id"].unique().tolist()
            op_ls.sort()        # 获取当天服务中心list
            op_rider_ls = []
            op_driver_ls = []
            op_driver_compl_ls = []
            op_rider_compl_ls = []
            df_dict_ls = []
            for i in op_ls:
                # 下单乘客list
                op_rider = list(df.loc[df["operation_center_id"] == i]["rider_id"])
                # 完单乘客list
                op_rider_compl = \
                    list(df.loc[(df["operation_center_id"] == i)&((df["actual_dropoff_at"].notnull()))]["rider_id"])
                # 接单司机list
                op_driver = list(df.loc[(df["operation_center_id"]==i)&((df["driver_id"].notnull()))]["driver_id"])
                # 完单司机list
                op_driver_compl = \
                    list(df.loc[(df["operation_center_id"] == i)&((df["actual_dropoff_at"].notnull()))]["driver_id"])
                # 接单司机list去重
                op_driver = list(set(op_driver))
                # 集合形式显示（司机id：趟数）
                driver_trips = df.loc[(df["operation_center_id"] == i)].groupby("driver_id").\
                    agg(当日趟次=("group_id",lambda x:x.nunique())).reset_index()
                df_dict = driver_trips[["driver_id","当日趟次"]].to_dict(orient="records")

                op_rider_ls.append(op_rider)
                op_rider_compl_ls.append(op_rider_compl)
                op_driver_ls.append(op_driver)
                op_driver_compl_ls.append(op_driver_compl)
                df_dict_ls.append(df_dict)

            data["下单乘客list"] = op_rider_ls
            data["完单乘客list"] = op_rider_compl_ls
            data["接单司机list"] = op_driver_ls
            data["完单司机list"] = op_driver_compl_ls
            data["司机趟次_dict_list"] = df_dict_ls

            big_data = pd.concat([big_data, data], join='outer')

        return big_data


if __name__ == '__main__':
    star = '2023-07-01'
    end = '2023-08-28'  #"<"
    df_path = "/Users/admin/Desktop/史蒂芬大春/办公桌面/编程/python/周报/数据库/rides_2023.pkl"
    # df_path = "/Users/shuuomousakai/Desktop/史蒂芬大春/办公桌面/编程/python/周报/数据库/rides_2023.pkl"
    count_day = BusinessDiagnosticModelCount(star=star,end=end,df_path=df_path)
    data = count_day.x()
    print(data)
    # print(len(data))
    # print(data.head(30))
    data.to_pickle("/Users/admin/Desktop/经营诊断data.pkl")