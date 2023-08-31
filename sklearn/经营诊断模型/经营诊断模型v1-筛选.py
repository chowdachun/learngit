
import pandas as pd
from collections import Counter

class BusinessDiagnosticModel:
    def __init__(self,front_star,front_end,after_star,after_end,in_path,op,driver_employee_day=4):
        self.front_star = pd.to_datetime(front_star)
        self.front_end = pd.to_datetime(front_end)
        self.after_star = pd.to_datetime(after_star)
        self.after_end = pd.to_datetime(after_end)
        self.driver_employee_day = driver_employee_day
        self.op = op
        self.df = pd.read_pickle(in_path)
        self.df["服务日期"] = pd.to_datetime(self.df["服务日期"])

    def sum_base(self,star,end):
        star = pd.to_datetime(star)
        end = pd.to_datetime(end)
        df = self.df.query(f"服务日期 >= @star and @end > 服务日期")
        data = df.groupby("operation_center_id").agg(完单量=("完单量","sum"),
                                                     接驾订单量=("接驾订单量", "sum"),
                                                     接单量=("接单量","sum"),
                                                     下单量=("下单量","sum"),
                                                     接驾后取消=("接驾后取消","sum"),
                                                     接驾准时量=("接驾准时量","sum"),
                                                     总接驾量=("总接驾量","sum"),
                                                     接驾前取消=("接驾前取消","sum"),
                                                     未响应订单量=("未响应订单量","sum"),
                                                     未响应订单_后台取消=("未响应订单_后台取消","sum"),
                                                     总响应时长=("总响应时长","sum"),
                                                     响应订单量=("响应订单量","sum"),
                                                     app下单量=("app下单量","sum"),
                                                     后台下单量=("后台下单量","sum"),
                                                     小程序下单量=("小程序下单量","sum"),
                                                     扫码下单量 = ("扫码下单量","sum"),
                                                     趟数=("趟数","sum"),
                                                     总人数=("总人数","sum"),
                                                     总荷载人数=("总荷载人数","sum"),
                                                     接驾总时长 = ("接驾总时长","sum"),
                                                     接驾总单量 = ("接驾总单量","sum"),
                                                     行程总时长= ("行程总时长","sum"),
                                                     行程总单量 = ("行程总单量","sum"),
                                                     需求下单用户 = ("需求下单用户","sum"),
                                                     需求未满足用户 = ("需求未满足用户","sum"),
                                                     流水= ("流水","sum"),
                                                     新增新用户=("新增新用户","sum"),
                                                     下单用户list=("下单乘客list", lambda x: sum(x, [])),
                                                     完单乘客list=("完单乘客list", lambda x: sum(x, [])),
                                                     接单司机list=("接单司机list", lambda x: sum(x, [])),
                                                     完单司机list=("完单司机list", lambda x: sum(x, [])),
                                                     接单x天司机趟数=("司机趟次_dict_list", lambda x: sum(x, [])),
                                                     )

        data["完成率"] = data["完单量"] / data["下单量"]
        data["接驾准时率"] = data["接驾准时量"] / data["总接驾量"]
        data["上座率"] = data["总人数"] / data["总荷载人数"]
        data["需求满足率"] = 1- (data["需求未满足用户"] / data["需求下单用户"])
        data["响应时长min"] = data["总响应时长"] / data["响应订单量"]
        data["单均价格"] = data["流水"] / data["完单量"]
        data["单均接驾时长"] = data["接驾总时长"] / data["接驾总单量"]
        data["单均行程时长"] = data["行程总时长"] / data["行程总单量"]
        data["人均价格"] = data["流水"] / data["总人数"]

        # 把列表列转换为值
        data["下单用户list_bom"] = data["下单用户list"].apply(lambda x: set(x))
        data["完单乘客list_bom"] = data["完单乘客list"].apply(lambda x: set(x))
        data["接单司机list_bom"] = data["接单司机list"].apply(lambda x: set(x))
        data["完单司机list_bom"] = data["完单司机list"].apply(lambda x: set(x))

        data["下单用户list_len"] = data["下单用户list_bom"].apply(lambda x: len(x))
        data["完单乘客list_len"] = data["完单乘客list_bom"].apply(lambda x: len(x))
        data["接单司机list_len"] = data["接单司机list_bom"].apply(lambda x: len(x))
        data["完单司机list_len"] = data["完单司机list_bom"].apply(lambda x: len(x))

        def count_3(x):
            ls = Counter(x)
            con = 0
            for a, b in ls.items():
                if b >= self.driver_employee_day:
                    con += 1
            return con
        data["接单>x天司机list_len"] = data["接单司机list"].apply(count_3)

        # 接单>x天司机趟数
        def driver_x(x):
            df = pd.DataFrame(x)    # 把集合转换为df
            if len(df) > 0:
                data_bridge = df.groupby("driver_id").agg(接单天数=("当日趟次", "count"), 趟数=("当日趟次", "sum"))
                data = data_bridge.loc[data_bridge["接单天数"] >= self.driver_employee_day]["趟数"].sum()
                return data
            else:
                return 0
        data["接单>x天司机趟数"] = data["接单x天司机趟数"].apply(driver_x)

        data["接单>x天司机日均趟数"]=\
            data["接单>x天司机趟数"]/data["接单>x天司机list_len"]/((self.after_end-self.after_star).days)

        data["日均时长"] = (data["单均接驾时长"]+data["单均行程时长"])*data["接单>x天司机日均趟数"]/60

        # 运力满载流水预测
        bins_avgrider = [-1, 30, 50, 70, 100, 150, 1000]
        labels_daytime = [6.5, 7.5, 8.7, 9.7, 10.7, 12]
        labels_riderate = [0.92, 0.9, 0.88, 0.85, 0.85, 0.85]
        data["匹配日均时长"] = pd.cut(data["人均价格"], bins=bins_avgrider, labels=labels_daytime, ordered=False)
        data["匹配上座率"] = pd.cut(data["人均价格"], bins=bins_avgrider, labels=labels_riderate, ordered=False)

        data["预估日均时长增长%"]=(data["匹配日均时长"].astype(float)-data["日均时长"].astype(float))/data["日均时长"].astype(float)
        data.loc[data["预估日均时长增长%"]<0,"预估日均时长增长%"] = 0

        data["预估上座率增长%"] = (data["匹配上座率"].astype(float) - data["上座率"].astype(float))/data["上座率"].astype(float)
        data.loc[data["预估上座率增长%"] < 0,"预估上座率增长%"] = 0
        data["预测增长max%"] = data["预估日均时长增长%"] + data["预估上座率增长%"]

        data["预测流水max"] = data["流水"] * (1+data["预测增长max%"])
        data["日均时长预估差值"] = data["日均时长"]*(1+data["预估日均时长增长%"])-data["日均时长"]
        data["峰值达成"] = data["流水"]/data["预测流水max"]
        data["接驾量-完单量转化%"] = data["完单量"]/data["总接驾量"]
        data["接单量-接驾量转化%"] = data["总接驾量"]/data["接单量"]
        data["下单量-接单量转化%"] = data["接单量"]/data["下单量"]
        data["老新用户比例"] = (data["下单用户list_len"] - data["新增新用户"])/data["新增新用户"]

        data.drop(["下单用户list","完单乘客list","接单司机list","完单司机list",
                   "下单用户list_bom","完单乘客list_bom","接单司机list_bom",
                   "完单司机list_bom","接单x天司机趟数","匹配日均时长","匹配上座率"],axis=1,inplace=True)

        return data

    def loop(self):
        data_front = self.sum_base(star=self.front_star,end=self.front_end)
        data_after = self.sum_base(star=self.after_star,end=self.after_end)
        # print(data_after)

        df = data_after - data_front
        df = df.reset_index()
        data_after = data_after.reset_index()

        # return df.loc[df["operation_center_id"]==self.op] , data_after.loc[df["operation_center_id"]==self.op]
        return df , data_after

if __name__ == '__main__':
    front_star = '2023-08-14'
    front_end = '2023-08-21'  #"<"
    after_star = '2023-08-21'
    after_end = '2023-08-28'
    op = None
    driver_employee_day=4
    in_apth = "/Users/admin/Desktop/经营诊断data.pkl"
    k = BusinessDiagnosticModel(front_star=front_star,front_end=front_end,after_star=after_star,after_end=after_end,
                                in_path=in_apth,op=op,driver_employee_day=driver_employee_day)
    # data = k.sum_base(star=after_star,end=after_end)
    data_diff,data_after = k.loop()
    data_diff.to_csv("/Users/admin/Desktop/经营诊断data_diff.csv")
    data_after.to_csv("/Users/admin/Desktop/经营诊断data_after.csv")

