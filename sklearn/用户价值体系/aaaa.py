import pandas as pd
import datetime as dt
import os
import numpy as np
import time
import swifter



class RiderValue:
    def __init__(self, end_time, days, operation_center,route_id=None,in_path = None,
                 to_path=os.path.abspath(os.path.join(os.getcwd(),"data/rider_life_values.csv")),
                 x_last_days=0,x_last_orders=0, lossdays=90,df_set=None):
        """
        :param end_time: 开始倒数的日期
        :param days: 由结束时间开始往前倒数x天
        :param operation_center: [True,False] 是否按运营中心分组
        :param route_id: 按线路读取数据，默认None
        :param df_set: 直接传入计算数据集，默认None
        :param in_path: 数据读取路径.pkl，默认None
        :param to_path: 数据保存路径.csv，默认值：data/rider_life_values.csv
        :param x_last_days: 以乘客最近订单往前x天的数据集
        :param x_last_orders: 以乘客最近订单往前x单的数据集
        :param lossdays: 乘客流失天数值
        """
        self.end_time = end_time
        self.days = days
        self.operation_center = operation_center
        self.in_path = in_path
        self.data = None
        self.arg_holiday_path = os.path.abspath(os.path.join(os.getcwd(), "data/arg_holiday.xlsx"))
        self.fitdf = None
        self.to_path = to_path
        self.x_last_days = x_last_days
        self.x_last_orders = x_last_orders
        self.route_id = route_id
        self.lossdays = lossdays
        self.df_set = df_set

    def data_time(self):
        star_time_dt = time.time()
        if self.df_set is not None:
            data = self.df_set
        elif self.in_path == None:
            parent_folder = os.path.dirname(os.getcwd())
            grandparent_folder = os.path.dirname(parent_folder)
            data = pd.read_pickle(os.path.abspath(os.path.join(grandparent_folder,"python/周报/数据库/rides.pkl")))
        else:
            print(self.in_path)
            data = pd.read_pickle(self.in_path)
        print("读取数据集大小：",data.shape)

        if self.route_id == None:
            pass
        else:
            data = data.query(f"route_id == @self.route_id")

        datetime_list = ['created_at', 'payments_completed_at', 'serving_at', 'pickup_from', 'pickup_to',
                         'actual_pickup_at',
                         'actual_dropoff_at', 'accepted_at',
                         'canceled_at', 'penalty_paid_at', 'reminded_at']
        data['created_at'] = pd.to_datetime(data['created_at'], format='%Y-%m-%d', errors='coerce')
        for name in datetime_list:
            data[name] = pd.to_datetime(data[name], format='%Y-%m-%d %H:%M:%S', errors='coerce')

        #选择对应时间的数据
        end_time = pd.to_datetime(self.end_time, format='%Y-%m-%d')
        star_time = end_time - dt.timedelta(days=self.days)
        star_time = pd.to_datetime(star_time, format='%Y-%m-%d')
        print("数据集筛选-开始时间：",star_time)
        data = data.query(f"serving_at >= @star_time and serving_at < @end_time")
        target_ls = ["rider_id","operation_center_id","serving_at","created_at","display_id","by_qrcode","rating","route_id",
                     "actual_pickup_at","status","client","manual_assigned_count","pickup_address","dropoff_address",
                     "actual_dropoff_at","payments_price","rider_number","payments_promotion","pickup_from","id","accepted_at",
                     ]
        data = data[target_ls]
        print("数据集筛选-数据集大小：",data.shape)

        end_time_dt = time.time()
        execution_time = end_time_dt - star_time_dt
        print("数据集准备用时: ", execution_time / 60, "minutes")


        if self.x_last_days > 0:       #选择以用户最后订单前x天
            ls = ["rider_id"]
            if self.operation_center == False:
                ls = ["rider_id"]
            else:
                ls.append("operation_center_id")

            x_last_days_star = end_time - dt.timedelta(days=2*self.x_last_days)     #days=2*self.x_last_days
            print("x_last_day数据集筛选-数据集起始时间：",x_last_days_star,end_time)
            data = data.query(f"serving_at >= @x_last_days_star and serving_at < @end_time")

            df_max_index = data.groupby(ls)["serving_at"].max() - pd.Timedelta(days=self.x_last_days)
            df = pd.merge(data, df_max_index, on=ls, how='left')
            df.loc[df["serving_at_x"] > df["serving_at_y"], "serving_at_y"] = 1
            df = df.rename(columns={"serving_at_x": "serving_at", "serving_at_y": "筛选", })
            df = df.query("筛选 == 1")
            df = df.drop(columns=["筛选"])
            print(f"x_last_day数据集筛选-数据集大小：",df.shape)
            self.data = df
            return df

        elif self.x_last_orders > 0:        #选择以用户最后订单前x个订单
            star_time_dt = time.time()
            ls = ["rider_id"]
            if self.operation_center == False:
                ls = ["rider_id"]
            else:
                ls.append("operation_center_id")

            def sort_by_createdtime(group):
                group_sort = group.sort_values("created_at", ascending=True)
                ascend_order = group_sort["id"]
                if len(ascend_order) >= self.x_last_orders:
                    x_order_group = ascend_order.head(self.x_last_orders)
                    return x_order_group
                else:
                    x_order_group = ascend_order
                    return x_order_group
            print("x_last_orders-apply开始计算")
            sort_df_createdtime = data.groupby(ls).parallel_apply(sort_by_createdtime).reset_index()
            print("x_last_orders-apply结束计算")
            sort_df_createdtime["筛选"] = 1
            df = pd.merge(data, sort_df_createdtime, on="id", how="left")#.fillna(0)
            df.drop(columns=['rider_id_y', 'operation_center_id_y', 'level_2'], inplace=True)
            print("x_last_orders-数据集merge后大小：", df.shape)
            df = df.query("筛选 == 1")
            print("x_last_orders-数据集-query后大小：", df.shape)
            df.drop(columns=['筛选'], inplace=True)
            df = df.rename(columns={"rider_id_x": "rider_id","operation_center_id_x":"operation_center_id"})
            end_time_dt = time.time()
            execution_time = end_time_dt - star_time_dt
            print("x_last_orders-数据集计算时间:", execution_time / 60, "minutes")
            print("x_last_orders-数据集大小:", df.shape)
            self.data = df
            return df

        else:
            self.data = data
            return data


    def rider_order(self):
        ls = ["rider_id"]
        if self.operation_center == False:
            ls = ["rider_id"]
        else:
            ls.append("operation_center_id")
        data_A = self.data.groupby(ls).agg(下单量=("display_id", "count"),
                                           扫码下单量=("by_qrcode",  lambda x: (x ==1).sum()),
                                           评价数量=("rating", lambda x: (x > 0).sum()),
                                           差评数量=("rating", lambda x: (x < 4).sum()),
                                           覆盖线路数量=("route_id", lambda x: x.nunique()),
                                           最早下单时间=("created_at", "min"),
                                           最早上车时间=("actual_pickup_at","min"),
                                           最近下单时间=("created_at", "max"),
                                           过期订单量=("status", lambda x: (x == "expired").sum()),
                                           App下单量=("client",lambda x:(x=='app').sum()),
                                           后台下单量=("client",lambda x:(x=='backend').sum()),
                                           小程序下单量=("client",lambda x:(x=='mini_program').sum()),
                                           人工调单量=("manual_assigned_count", lambda x: (x > 0).sum()),
                                           起点地去重=("pickup_address", lambda x: x.nunique()),
                                           终点地去重=("dropoff_address", lambda x: x.nunique()),)
        data_A["总活跃天数"] = (data_A["最近下单时间"] - data_A["最早下单时间"]).dt.days

        data_A["过期率"] = data_A["过期订单量"] / data_A["下单量"]
        data_A["App下单率"] = data_A["App下单量"] / data_A["下单量"]
        data_A["后台下单率"] = data_A["后台下单量"] / data_A["下单量"]
        data_A["小程序下单率"] = data_A["小程序下单量"] / data_A["下单量"]
        data_A["人工调单率"] = data_A["人工调单量"] / data_A["下单量"]

        data_B = self.data.query("actual_dropoff_at.notnull()").groupby(ls).agg(
            完单量=("rider_id", "count"), 流水=("payments_price", lambda x: x.sum() / 100.0),
            完单上座数=("rider_number", "sum"),优惠订单量=("payments_promotion",lambda x: (x>0).sum()),)
        data_B["完成率"] = data_B["完单量"] / data_A["下单量"]
        data_B["平均完单上座数"] = data_B["完单上座数"] / data_B["完单量"]
        data_B["优惠订单率"] = data_B["优惠订单量"] / data_B["完单量"]

        data_C = self.data.query("payments_promotion.notnull()").groupby(ls).\
            agg(最早用券时间=("actual_pickup_at","min"))

        # 预约订单量
        order_booking = self.data.loc[
            ((self.data["pickup_from"] - self.data["created_at"]).swifter.apply(lambda x: x.total_seconds() / 3600)) > 1]. \
            groupby(ls).agg(预约订单量=("id", "count"))
        order_booking["预约率"] = (order_booking["预约订单量"] / data_A["下单量"]).fillna(0)
        data_concat = pd.concat([data_A, data_B, order_booking,data_C], join="outer", axis=1, ).fillna(0)
        data_concat["首单用券"] = np.where(data_concat["最早用券时间"]==data_concat["最早上车时间"],1,0)
        return data_concat

    def rider_life(self):
        ls = ["rider_id"]
        if self.operation_center == False:
            ls = ["rider_id"]
        else:
            ls.append("operation_center_id")
        data_A = self.data.groupby(ls).agg(最近下单时间=("serving_at", lambda x: x.max()))
        data_A["最近下单天数"] = (data_A["最近下单时间"].max() - data_A["最近下单时间"]).dt.days
        data = data_A[["最近下单天数"]]
        data["是否流失"] = np.where(data["最近下单天数"] > self.lossdays, 1, 0)

        def count_time_dfii(group):
            group_sort = group.sort_values("created_at", ascending=True)
            time_diff = group_sort["created_at"].diff()
            counts = (time_diff >= pd.Timedelta(days=self.lossdays)).sum()
            return counts

        groups = self.data.groupby(ls).parallel_apply(count_time_dfii)
        result = pd.concat([data, groups], axis=1, join='outer')
        ls_add = ["最近下单天数", "是否流失", "自活次数"]
        result.columns = ls_add
        return result

    def rider_work_hoildays(self):  # holidays_arg,
        ls = ["rider_id"]
        if self.operation_center == False:
            ls = ["rider_id"]
        else:
            ls.append("operation_center_id")
        # 区分周末及工作日
        data = self.data[["rider_id", "operation_center_id", "serving_at"]]
        data["weekday"] = self.data.loc[:, "serving_at"].swifter.apply(
            lambda x: 1 if pd.to_datetime(x).weekday() in [4, 5, 6] else 0)
        data["workday"] = self.data.loc[:, "serving_at"].swifter.apply(
            lambda x: 1 if pd.to_datetime(x).weekday() in [0, 1, 2, 3] else 0)
        # 添加节假日及前一日参数
        # self.arg_holiday_path = os.path.abspath(os.path.join(os.getcwd(), "data/arg_holiday.xlsx"))
        arg_holiday = pd.read_excel(self.arg_holiday_path)
        arg_holiday["serving_at"] = pd.to_datetime(arg_holiday["serving_at"], format='%Y-%m-%d %H:%M:%S',
                                                   errors='coerce')
        data = pd.merge(data, arg_holiday, on="serving_at", how='left').fillna(0)
        # 把工作日的节假日变为非工作日
        data.loc[data["holiday"] == 1, "workday"] = 0
        # 把补班变为工作日
        data.loc[data["holiday"] == -1, "workday"] = 1
        # 把节假日补班变为非周末
        data.loc[data["holiday"] == -1, "weekday"] = 0
        # 因为把补班都设置为工作日且非周末，所以节假日补班的-1设为正常0
        data.loc[data["holiday"] == -1, "holiday"] = 0
        data_A = data.groupby(ls).agg(工作日次数1_4=("workday", "sum"), 周末次数5_7=("weekday", "sum"),
                                      节假日次数=("holiday", "sum"), )
        return data_A

    def rider_client(self):
        ls = ["rider_id"]
        if self.operation_center == False:
            ls = ["rider_id"]
        else:
            ls.append("operation_center_id")

        # 设置最小创建时间对应client的apply函数
        def get_min_created_at_client(data):
            min_created_at_idx = data["created_at"].idxmin()
            return data.loc[min_created_at_idx, "client"]

        # 使用上述函数
        result_min = self.data.groupby(ls).parallel_apply(get_min_created_at_client)  # .reset_index()

        # 设置最大创建时间对应client的apply函数
        def get_max_created_at_client(data):
            min_created_at_idx = data["created_at"].idxmax()
            return data.loc[min_created_at_idx, "client"]

        # 使用上述函数
        result_max = self.data.groupby(ls).parallel_apply(get_max_created_at_client)  # .reset_index()
        # 把两个数据表合并
        # result = pd.merge(result_min, result_max, on=ls) #.reset_index(drop=True)
        result = pd.concat([result_min, result_max], axis=1, join='outer')
        ls_add = ["时间段首次下单渠道", "时间段最近下单渠道"]
        # list_total = ls + ls_add
        result.columns = ls_add  # list_total
        # result = result.drop('index',axis=1)
        return result

    def rider_badratinglife(self, badratinglossday=30):
        """
        :param badratinglossday:差评后x天没坐车
        :return: Dataframe
        """
        ls = ["rider_id"]
        if self.operation_center == False:
            ls = ["rider_id"]
        else:
            ls.append("operation_center_id")
        data = self.data.loc[self.data["rating"] < 4, :]
        data = data.groupby(ls).agg(最新差评时间=("created_at", "max"))
        data_new = self.data.groupby(ls).agg(最新订单时间=("created_at", "max"))
        result = pd.merge(data, data_new, how='left', on=ls)
        result["差评流失"] = (result["最新订单时间"] - result["最新差评时间"]).dt.days
        result["差评流失"] = np.where(result["差评流失"] >= badratinglossday, 1, -1)
        result = result[["差评流失"]]
        return result


    def df_concat(self, df_list):
        result = pd.concat(df_list, axis=1, join='outer').reset_index(drop=True)
        result = result.fillna(0)
        return result


    def riderlifedays(self,df):
        """
        需要rider_life_values.csv，计算用户的生命周期均值&中位数
        :df: abcde合并后的Dataframe
        :return: Dataframe
        """
        ls = ["rider_id"]
        if self.operation_center == False:
            df_A = df.query("是否流失 == 1 and 自活次数 == 0 and 总活跃天数 > 1")  # 获取一个完整的生命周期
            data = df_A["总活跃天数"]
            # operation_center ==None
            df["生命周期均值"] = data.mean()
            df["生命周期中位数"] = data.median()
            df = df.set_index(ls)
            return df
        else:
            ls.append("operation_center_id")
            df_A = df.query("是否流失 == 1 and 自活次数 == 0 and 总活跃天数 > 1")  # 获取一个完整的生命周期
            print("筛选完整的生命周期用户-数据集大小：",df_A.shape)
            data = df_A[ls + ["总活跃天数"]]
            # operation_center ==True
            df_opertion_mean = data.groupby("operation_center_id").mean()
            df_opertion_mean = df_opertion_mean[["总活跃天数"]]
            df_opertion_median = data.groupby("operation_center_id").median()
            df_opertion_median = df_opertion_median[["总活跃天数"]]
            df_opertion = pd.concat([df_opertion_mean, df_opertion_median], axis=1, join='outer').reset_index()
            df_opertion.columns = ["operation_center_id", "生命周期均值", "生命周期中位数"]
            print("生命周期均值中位数-数据集大小：",df_opertion.shape)
            # merge
            df = pd.merge(df, df_opertion, on='operation_center_id',how='left')
            df = df.set_index(ls)
            return df


    def riderlifefilter(self,fitdf):
        fitdf["导入期"] = np.where((fitdf["下单量"] >= 1) & (fitdf["完单量"] == 0), 1, 0)
        fitdf["体验期"] = np.where((fitdf["完单量"] == 1), 1, 0)

        fitdf.loc[
            (fitdf["完单量"] > 1) & (fitdf["生命周期中位数"] >= fitdf["总活跃天数"]), "成长期"] = 1
        fitdf["成长期"] = fitdf["成长期"].fillna(0)

        fitdf.loc[
            (fitdf["完单量"] > 1) & (fitdf["总活跃天数"] > fitdf["生命周期中位数"]), "成熟期"] = 1
        fitdf["成熟期"] = fitdf["成熟期"].fillna(0)
        return fitdf


    def fit(self, badratinglossday=60,save=True):
        """
        声明周期出现空值意味是新开城运营中心，没有完整的生命周期用户
        :param badratinglossday: 差评流失天数
        :param save: 是否保存 [True,False]
        :return: Dataframe
        """
        print("fit-初始化Dataframe")
        start_time = time.time()
        data_time_df = self.data_time()
        A = self.rider_order()
        print("A", A.shape)
        B = self.rider_life()
        print("B", B.shape)
        C = self.rider_work_hoildays()
        print("C", C.shape)
        D = self.rider_client()
        print("D", D.shape)
        E = self.rider_badratinglife(badratinglossday=badratinglossday)
        print("E", E.shape)
        result = pd.concat([A, B, C, D, E], axis=1, join='outer')  # .reset_index(drop=True)
        print("ABCDE数据concat后-数据集大小：", result.shape)
        result = result.fillna(0).reset_index()
        fitdf = self.riderlifedays(result)
        print("用户生命周期转换-数据集大小：", fitdf.shape)
        df = self.riderlifefilter(fitdf)
        df = df[~((df['下单量'] == 0))]
        print("fit-计算完成-数据集大小：", df.shape)
        if save == True:
            df.to_csv(self.to_path)
            self.fitdf = df
            end_time = time.time()
            execution_time = end_time - start_time
            print("----fit-计算结束时长----: ", execution_time/60, "minutes")
            return self.fitdf
        else:
            self.fitdf = df
            end_time = time.time()
            execution_time = end_time - start_time
            print("----fit-计算结束时长----: ", execution_time/60, "minutes")
            return self.fitdf

    def fit_timeflow(self,save=True,method='mean'):
        """
        响应-接驾-行程 的时间统计，提供均值与中位数计算
        :param save: 是否保存
        :param method: 计算方法，mean：均值，median：中位数，默认(method='mean')
        :return: Dataframe
        """
        start_time = time.time()
        ls = ["rider_id"]
        if self.operation_center == False:
            ls = ["rider_id"]
        else:
            ls.append("operation_center_id")

        self.data["响应时长min"] = (self.data["accepted_at"] - self.data["created_at"]).dt.total_seconds() / 60
        self.data["接驾时长min"] = (self.data["actual_pickup_at"] - self.data["pickup_from"]).dt.total_seconds() / 60
        self.data["接驾时长min"] = np.where(self.data["接驾时长min"] < 0, 0, self.data["接驾时长min"])
        self.data["行程时长min"] = (self.data["actual_dropoff_at"] - self.data["actual_pickup_at"]).dt.total_seconds() / 60

        if method == 'median':
            time_mean = self.data.groupby(ls).agg(
                响应时长min_median=("响应时长min", lambda x: x.median()),
                接驾时长min_median=("接驾时长min", lambda x: x.median()),
                行程时长min_median=("行程时长min", lambda x: x.median()),)
        else:
            time_mean = self.data.groupby(ls).agg(
                响应时长min_mean=("响应时长min", lambda x: x.sum() / len(x)),
                接驾时长min_mean=("接驾时长min", lambda x: x.sum() / len(x)),
                行程时长min_mean=("行程时长min", lambda x: x.sum() / len(x)),)

        df = pd.concat([self.fitdf,time_mean], axis=1, join='outer')
        df = df[~((df['下单量'] == 0))]
        # print("fit_timeflow_self.data", self.data.shape)
        print("fit_timeflow-数据集大小：",df.shape)
        if save == True:
            df.to_csv(self.to_path)
            end_time = time.time()
            execution_time = end_time - start_time
            print("----fit_timeflow-计算时长----：", execution_time/60, "minutes")
            return df
        else:
            end_time = time.time()
            execution_time = end_time - start_time
            print("----fit_timeflow-计算时长----: ", execution_time/60, "minutes")
            return df


    def data_group_test(self):
        """
        用作Dataframe的index行数验证
        :return: shape
        """
        ls = ["rider_id"]
        if self.operation_center == False:
            ls = ["rider_id"]
        else:
            ls.append("operation_center_id")
        df = self.data[ls]
        df["tset"] = df["rider_id"] + df ["operation_center_id"] +999
        df = df.drop_duplicates(keep='last')
        print(df.shape)


