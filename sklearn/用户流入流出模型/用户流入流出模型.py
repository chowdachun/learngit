
import pandas as pd
import datetime as dt

class RiderInOut:
    def __init__(self,end_time,df_path,df_rider_life_path,interval_days=60,days=7,loop=0,to_path=None):
        self.end_time = end_time
        self.df_path =df_path
        self.df_rider_life_path = df_rider_life_path
        self.interval_days= interval_days
        self.days = days
        self.loop = loop
        self.to_path = to_path

    def count_rider_of_operatioan(self,end_time):

        df = pd.read_pickle(self.df_path)
        df_rider_life = pd.read_csv(self.df_rider_life_path)

        end_time = pd.to_datetime(end_time)
        star_time = end_time - dt.timedelta(days=self.interval_days)    #5/11
        ls_rider = ["rider_id","operation_center_id"]

        df['serving_at'] = pd.to_datetime(df['serving_at'], format='%Y-%m-%d', errors='coerce')
        df_rider_life['最早下单时间'] = pd.to_datetime(df_rider_life['最早下单时间'], format='%Y-%m-%d', errors='coerce')
        df_rider_life['最近下单时间'] = pd.to_datetime(df_rider_life['最近下单时间'], format='%Y-%m-%d', errors='coerce')

        #当段下单用户
        df_now_rider = df.query(f"serving_at >= @star_time and @end_time > serving_at")
        now_rider = df_now_rider.groupby(ls_rider).agg(本段用户数下单量=("rider_id","count"),)

        #前段乘车用户
        before_end_time = end_time -dt.timedelta(days=self.days)        #7/3
        before_star_time = before_end_time - dt.timedelta(days=self.interval_days)      #5/4
        df_before_rider = df.query(f"serving_at >= @before_star_time and @before_end_time > serving_at")
        before_rider = df_before_rider.groupby(ls_rider).agg(前段用户数下单量=("rider_id","count"))

        #rider_life_values_operation 获取7/3-7/9日新增用户
        df_now_riderlife = df_rider_life.query(f"最早下单时间 >= @before_end_time and @end_time > 最早下单时间").\
            groupby(ls_rider).agg(本段新增新用户=("下单量",lambda x:(x>0).sum()),
                                  本段新增新用户_用券=("优惠订单量",lambda x:(x>0).sum()))
        #rider_life_values_operation 获取5/4-7/3日新增用户
        df_before_riderlife = df_rider_life.query(f"最早下单时间 >= @before_star_time and @before_end_time > 最早下单时间").\
            groupby(ls_rider).agg(前段新增新用户=("下单量",lambda x:(x>0).sum()),
                                  前段新增新用户_用券=("优惠订单量",lambda x:(x>0).sum()))
        #rider_life_values_operation 获取5/4-7/3日新增用户


        df_concat= pd.concat([df_now_riderlife,df_before_riderlife,now_rider,before_rider],axis=1, join='outer')

        #新指标转换
        df_concat.loc[(df_concat["前段用户数下单量"] > 0), "前段下单用户count"] = 1
        df_concat.loc[(df_concat["本段用户数下单量"]>0),"本段下单用户count"] = 1
        df_concat.loc[((df_concat["本段用户数下单量"].isnull())&(df_concat["前段用户数下单量"]>0)),"流失用户count"] = 1
        df_concat.loc[((df_concat["本段用户数下单量"].isnull()) & (df_concat["前段新增新用户"] > 0)), "流失新用户count"] = 1
        df_concat.loc[((df_concat["本段用户数下单量"].isnull()) & (df_concat["前段新增新用户"].isnull()) & (
                df_concat["前段用户数下单量"] > 0)), "流失老用户count"] = 1
        df_concat.loc[((df_concat["本段新增新用户"].isnull())&(df_concat["前段用户数下单量"].isnull())&(
                    df_concat["本段用户数下单量"]>0)),"新增老用户count"] = 1
        df_concat.loc[((df_concat["本段用户数下单量"]>0)&(df_concat["本段新增新用户"]>0)),"新增新用户count"] = 1
        df_concat.loc[((df_concat["本段用户数下单量"]>0) & (df_concat["本段新增新用户"]>0) & (
                    df_concat["本段新增新用户_用券"] > 0)), "新增新用户_用券count"] = 1



        #计算每个服务中心的流失情况
        df_operation = df_concat.groupby("operation_center_id").agg(前段下单用户=("前段下单用户count","sum"),
                                                                    本段下单用户=("本段下单用户count", "sum"),
                                                                    流失新用户 = ("流失新用户count", "sum"),
                                                                    流失老用户 = ("流失老用户count", "sum"),
                                                                    新增新用户 = ("新增新用户count", "sum"),
                                                                    新增老用户 = ("新增老用户count", "sum"),
                                                                    新增新用户_优惠=("新增新用户_用券count","sum"))
        df_operation["净流入"] = df_operation["本段下单用户"] - df_operation["前段下单用户"]
        df_operation["新用户净流入"] = df_operation["新增新用户"] - df_operation["流失新用户"]
        df_operation["老用户净流入"] = df_operation["新增老用户"] - df_operation["流失老用户"]

        # df_operation.to_csv("/Users/admin/Desktop/abc115.csv")
        return df_operation


    def loop_count(self):
        df_ls = []
        merged_df = pd.DataFrame()
        for i in range(self.loop):
            i = self.loop - i
            loop_end_time = pd.to_datetime(self.end_time) -dt.timedelta(days=(i-1) * self.days)
            df = self.count_rider_of_operatioan(end_time=loop_end_time)
            df = df[["净流入","新用户净流入","老用户净流入","新增新用户_优惠","本段下单用户"]]
            df.rename(columns={"净流入":f"净流入_{i}",
                               "新用户净流入":f"新用户净流入_{i}",
                               "老用户净流入":f"老用户净流入_{i}",
                               "新增新用户_优惠":f"新增新用户_优惠_{i}",
                               "本段下单用户":f"本段下单用户_{i}",},inplace=True)
            print(i)
            print(df.shape)
            print(loop_end_time,self.end_time)
            df_ls.append(df)

        for a in df_ls:
            merged_df = pd.concat([merged_df,a],axis=1,join="outer")
        print("merged_df",merged_df.shape)
        merged_df.to_csv(self.to_path)


# 此模型依赖全量用户价值表
if __name__ == "__main__":
    # df_path = "/Users/admin/Desktop/days_60/data.pkl"
    df_path = "/Users/admin/Desktop/史蒂芬大春/办公桌面/编程/python/周报/数据库/rides_2023.pkl"
    df_rider_life_path = "/Users/admin/Desktop/史蒂芬大春/办公桌面/编程/sklearn/用户价值体系/data/全量用户特征画像opr.csv"
    end_time = "2023-08-28"
    interval_day = 8*7
    days = 7
    to_path = "/Users/admin/Desktop/用户流入流出模型.csv"

    rit = RiderInOut(end_time=end_time,
                                   df_path=df_path,
                                   df_rider_life_path=df_rider_life_path,
                                   interval_days=interval_day,
                                   days=days,
                                   loop=8,
                                   to_path = to_path)
    df = rit.loop_count()


