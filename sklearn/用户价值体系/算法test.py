import pickle
import time
import pandas as pd
import datetime as dt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from pandarallel import pandarallel
import 二元分类机器学习算法集 as skfunc
import os

import aaaa as f
pandarallel.initialize(progress_bar=False, nb_workers=2)


class Youhua_logic_model:
    def __init__(self, end_time, lossdays, x_last_days, operation_center, in_path, to_path=None, back=0, df_set=None,
                 route_id=None):
        self.end_time = end_time
        self.lossdays = lossdays
        self.x_last_days = x_last_days
        self.operation_center = operation_center
        self.in_path = in_path
        self.to_path = to_path
        self.back = back
        self.a = None
        self.df_set = df_set
        self.route_id = route_id
        self.data = None

        # 数据清洗+特征转换

    def add_feature_clean(self, df):  # 增加特征
        df.loc[
            ((df["时间段首次下单渠道"] == "app") & (
                    df["时间段最近下单渠道"] == "mini_program")), "渠道转换"] = "app转mini"
        df.loc[
            ((df["时间段首次下单渠道"] == "app") & (df["时间段最近下单渠道"] == "backend")), "渠道转换"] = "app转backend"
        df.loc[
            ((df["时间段首次下单渠道"] == "mini_program") & (
                    df["时间段最近下单渠道"] == "app")), "渠道转换"] = "mini转app"
        df.loc[((df["时间段首次下单渠道"] == "mini_program") & (
                df["时间段最近下单渠道"] == "backend")), "渠道转换"] = "mini转backend"
        df.loc[
            ((df["时间段首次下单渠道"] == "backend") & (df["时间段最近下单渠道"] == "app")), "渠道转换"] = "backend转app"
        df.loc[((df["时间段首次下单渠道"] == "backend") & (
                df["时间段最近下单渠道"] == "mini_program")), "渠道转换"] = "backend转mini"

        df["工作日%"] = df["工作日次数1_4"] / df["下单量"]
        df["周末%"] = df["周末次数5_7"] / df["下单量"]
        df["节假日%"] = df["节假日次数"] / df["下单量"]

        dummies_col = ["时间段首次下单渠道", "时间段最近下单渠道"]
        df = pd.get_dummies(df, drop_first=True, columns=dummies_col)
        df.drop(columns=["渠道转换", "最近下单天数", "最早下单时间", "最近下单时间", "最早上车时间", "最早用券时间"],
                inplace=True)
        df.fillna(0, inplace=True)
        # print(df.isnull().sum())
        return df

    def split_train_pred(self, df_train, df_pred):
        self.x_pred_first = df_pred
        x_train = df_train.drop(["是否流失"], axis=1)
        y_train = df_train["是否流失"]
        x_pred = df_pred.drop(["是否流失"], axis=1)
        y_pred = df_pred["是否流失"]

        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.8, random_state=42)
        return x_train, x_test, x_pred, y_train, y_test, y_pred

    def logi_regression(self, x_train, y_train, x_test, y_test, x_pred, x_pred_first):
        print("x_train", x_train.shape)
        print("y_train", y_train.shape)
        print("x_test", x_test.shape)
        print("y_test", y_test.shape)
        print("x_pred", x_pred.shape)

        lgr = LogisticRegression()
        lgr.fit(x_train, y_train)
        print("logi的x_test拟合效果", lgr.score(x_test, y_test))
        y_pred = lgr.predict(x_pred)
        x_pred_first["预测值"] = y_pred  # 训练模型
        return x_pred_first

    def back_merge(self, back_test_1_pred=None, pred_0=None, a=0,select_startime=None, select_endtime=None,):
        # 预测结果与预测集合
        # result_ls = []
        ls = ["rider_id"]
        if self.operation_center == False:
            ls = ["rider_id"]
        else:
            ls.append("operation_center_id")
        back_test_1_pred = pd.merge(back_test_1_pred, pred_0[["下单量_x"]],  # 预测集/实际集
                                    on=ls, how="left").fillna(-1)

        back_test_1_pred = back_test_1_pred.rename(columns={"下单量_x": "实际流失"})
        back_test_1_pred.loc[back_test_1_pred["实际流失"] > 0, "实际流失"] = 0
        back_test_1_pred.loc[back_test_1_pred["实际流失"] == -1, "实际流失"] = 1

        back_test_1_pred["最近下单时间"] = pd.to_datetime(back_test_1_pred["最近下单时间"], format='%Y-%m-%d %H:%M:%S',
                                                          errors='coerce')
        print("内函数时间：", back_test_1_pred["最近下单时间"].min(), back_test_1_pred["最近下单时间"].max())
        back_test_1_pred = back_test_1_pred.query(
            f"最近下单时间 >= @select_startime and @select_endtime > 最近下单时间 ")

        loss_dui = back_test_1_pred.query("预测值 == 1 and 实际流失 == 1 ")["下单量"].count()
        loss_zong = back_test_1_pred.query("预测值 == 1 ")["下单量"].count()

        keep_dui = back_test_1_pred.query("预测值 == 0 and 实际流失 == 0 ")["下单量"].count()
        keep_zong = back_test_1_pred.query("预测值 == 0 ")["下单量"].count()

        total_dui = back_test_1_pred["预测值"] == back_test_1_pred["实际流失"]
        total_dui = total_dui.sum()
        total_zong = keep_zong + loss_zong
        back_ls = ["回测：{}".format(a), loss_dui / loss_zong, keep_dui / keep_zong, total_dui / total_zong]
        # result_ls.append(back_ls)
        print("流失准确率：", loss_dui / loss_zong)
        print("留存准确率：", keep_dui / keep_zong)
        print("总准确率：", total_dui / total_zong)
        return back_ls

    def train_merge(self, train, pred):

        train = train.drop(columns=["是否流失"])
        ls = ["rider_id"]
        if self.operation_center == False:
            ls = ["rider_id"]
        else:
            ls.append("operation_center_id")
        train = pd.merge(train, pred[["下单量"]], on=ls, how="left").fillna(-1)
        train = train.rename(columns={"下单量_x": "下单量", "下单量_y": "是否流失", })
        train.loc[train["是否流失"] > 0, "是否流失"] = 0
        train.loc[train["是否流失"] == -1, "是否流失"] = 1

        return train

    def fit(self):

        df_yuan_ls = []
        df_ls = []
        for i in range(self.back + 2):
            end_time = pd.to_datetime(self.end_time, format='%Y-%m-%d')
            back_time_end = end_time - dt.timedelta(days=i * self.lossdays)

            days = self.x_last_days * 2.5
            print("days:", days)
            data_pred = f.RiderValue(end_time=back_time_end,
                                     days=days,
                                     in_path=self.in_path,
                                     route_id=self.route_id,
                                     x_last_days=self.x_last_days,
                                     operation_center=self.operation_center,
                                     lossdays=self.lossdays,
                                     df_set=self.df_set, )
            data_pred.fit(save=False)
            pred = data_pred.fit_timeflow(save=False)
            df_yuan_ls.append(pred)
            print("pred:", pred.shape)
            pred = self.add_feature_clean(pred)
            df_ls.append(pred)

        with open(os.path.join(os.getcwd(),"流失预测缓存过度/df_ls.pkl"),'wb') as g:
            pickle.dump(df_ls,g)
        with open(os.path.join(os.getcwd(),"流失预测缓存过度/df_yuan_ls.pkl"),'wb') as j:
            pickle.dump(df_yuan_ls,j)

        # -----------------------------------------------------------------------------------
        pred_ls = []
        df_ls = pd.read_pickle(os.path.join(os.getcwd(),"流失预测缓存过度/df_ls.pkl"))
        df_yuan_ls = pd.read_pickle(os.path.join(os.getcwd(),"流失预测缓存过度/df_yuan_ls.pkl"))

        data = pd.read_csv(
            "/Users/admin/Desktop/史蒂芬大春/办公桌面/编程/sklearn/用户价值体系/data/全量用户特征画像opr.csv")

        for i in range(self.back + 1):  # 储存预测结果back+1个df

            train = self.train_merge(train=df_ls[i + 1], pred=df_ls[i])

            def add_feature(data,df):
                data = data.set_index(["rider_id", "operation_center_id"])
                data = data.drop(columns=["最早下单时间", "最早上车时间", "最近下单时间", "是否流失", "最早用券时间","最近下单天数"])
                dummies_col = ["时间段首次下单渠道", "时间段最近下单渠道"]
                data = pd.get_dummies(data, drop_first=True, columns=dummies_col)

                df = pd.merge(df, data, on=["rider_id", "operation_center_id"], how="left").fillna(0)
                return df

            df_ls[i] = add_feature(data=data,df=df_ls[i])
            train = add_feature(data=data,df=train)

            x_train, x_test, x_pred, y_train, y_test, y_pred = self.split_train_pred(df_train=train, df_pred=df_ls[i])
            print("y_train:", y_train.value_counts())

            #DNN神经网络yuc
            df_pred_first = skfunc.xgboost_xgb(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, x_pred=x_pred,
                 x_pred_first=df_yuan_ls[i])

            # 预测结果列表
            pred_ls.append(df_pred_first)

        result_ls = []
        for i in range(self.back):
            # 时间过滤
            select_endtime = pd.to_datetime(self.end_time, format='%Y-%m-%d') - dt.timedelta(
                days=(i + 1) * self.lossdays)
            select_startime = select_endtime - dt.timedelta(days=self.x_last_days)
            print("外函数时间：", select_startime, select_endtime)
            back_ls = self.back_merge(back_test_1_pred=pred_ls[i + 1],
                                      pred_0=df_ls[i],
                                      a=i + 1,
                                      select_startime=select_startime,
                                      select_endtime=select_endtime)
            result_ls.append(back_ls)
        print("流失准确率", "留存准确率", "总准确率")
        print("预测结果：", result_ls)

        shuchu_end = pd.to_datetime(self.end_time, format='%Y-%m-%d')
        shuchu_star = shuchu_end - dt.timedelta(days=self.x_last_days)
        pred_ls[0].query(f"最近下单时间 >= @shuchu_star and @shuchu_end > 最近下单时间").to_csv(self.to_path)


if __name__ == '__main__':
    # in_path = "/Users/shuuomousakai/Desktop/史蒂芬大春/办公桌面/编程/python/周报/数据库/rides_2023.pkl"
    in_path = "/Users/admin/Desktop/史蒂芬大春/办公桌面/编程/python/周报/数据库/rides_2023.pkl"
    # in_path = "/Users/admin/Desktop/days_60/data.pkl"
    end_time = "2023-08-28"
    lossdays = 90
    back = 2
    x_last_days = 30
    operation_center = True
    route_id = None
    to_path = "/Users/admin/Desktop/近30天用户流失预测_20230828.csv"

    t1 = time.time()
    df_set = pd.read_pickle(in_path)

    begin = Youhua_logic_model(end_time=end_time,
                               lossdays=lossdays,
                               back=back,
                               x_last_days=x_last_days,
                               operation_center=operation_center,
                               in_path=in_path,
                               to_path=to_path,
                               df_set=df_set,
                               route_id=route_id)
    begin.fit()
    t2 = time.time()
    print("计算时间：", (t2 - t1) / 60, "min")
