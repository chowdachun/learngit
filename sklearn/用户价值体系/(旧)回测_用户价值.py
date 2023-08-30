import time
import pandas as pd
import os
import datetime as dt
import func_fit用户价值 as t
import aaaa as f
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=False,nb_workers=1) #,nb_workers=1
# import psutil
# print("电脑核数：",psutil.cpu_count(logical=False))




parent_folder = os.path.dirname(os.getcwd())
grandparent_folder = os.path.dirname(parent_folder)
path = "/Users/shuuomousakai/Desktop/史蒂芬大春/办公桌面/编程/python/周报/数据库/rides_2023.pkl"

#用户价值类+逻辑回归类 聚合模型
class log_fit_agg_model:
    def __init__(self,end_time,lossdays=90,x_last_days=30,operation_center=True,in_path=None,to_path=None,df_set=None):
        self.end_time = end_time
        self.lossdays = lossdays
        self.x_last_days = x_last_days
        self.operation_center = operation_center
        self.in_path = in_path
        self.to_path = to_path
        self.df_set = df_set

    def pred_Logic_Pred_Rider_Loss(self):
        pred_end_time = pd.to_datetime(self.end_time, format='%Y-%m-%d')
        days = self.x_last_days * 2.5
        data_pred = f.RiderValue(end_time=self.end_time,
                            days=days,
                            in_path=self.in_path,
                            # route_id=[614,615],
                            x_last_days=self.x_last_days,
                            operation_center=self.operation_center,
                            lossdays=self.lossdays,
                            df_set=self.df_set)
        data_pred.fit(save=False)
        pred = data_pred.fit_timeflow(save=False)
        print("pred:",pred.shape)
        # pred.to_csv("/Users/admin/Desktop/days_60/pred.csv")
        return pred


    def train_Logic_Pred_Rider_Loss(self,pred,):
        end_time = pd.to_datetime(self.end_time, format='%Y-%m-%d')
        train_end_time = end_time  - dt.timedelta(days=self.lossdays)
        days = self.x_last_days * 2.5
        data_train = f.RiderValue(end_time=train_end_time,
                            days=days,
                            in_path=self.in_path,
                            # route_id=[614,615],
                            x_last_days=self.x_last_days,
                            operation_center=self.operation_center,
                            lossdays=self.lossdays,
                            df_set=self.df_set)
        data_train.fit(save=False)
        train = data_train.fit_timeflow(save=False)
        print("train:", train.shape)
        # train.to_csv("/Users/admin/Desktop/days_60/train.csv")

        train = train.drop(columns=["是否流失"])
        ls = ["rider_id"]
        if self.operation_center == False:
            ls = ["rider_id"]
        else:
            ls.append("operation_center_id")
        train = pd.merge(train, pred[["下单量"]],on=ls ,how="left" ).fillna(-1)
        train = train.rename(columns={"下单量_x": "下单量", "下单量_y": "是否流失", })
        train.loc[train["是否流失"]>0,"是否流失"] = 0
        train.loc[train["是否流失"]==-1,"是否流失"] = 1
        # print(train["是否流失"].value_counts())
        # train.to_csv("/Users/admin/Desktop/days_60/train.csv")
        return train


    def select_train_pred(self):
        pred_df = self.pred_Logic_Pred_Rider_Loss()
        train_df = self.train_Logic_Pred_Rider_Loss(pred_df)
        return pred_df,train_df

    #训练模型整合函数
    def logic_fit(self):
        pred_df,train_df = self.select_train_pred() #生成训练集和预测集

        lgr = t.Logic_Pred_Rider_Loss()

        train_df = lgr.add_feature_clean(df=train_df)  # 把数据清洗完毕
        pred_df = lgr.add_feature_clean(df=pred_df)  # 把数据清洗完毕

        x_train, x_test,x_pred, y_train, y_test,y_pred = lgr.split_train_pred(df_train=train_df,df_pred=pred_df)    #划分训练数据

        df_pred_first = lgr.logi_regression(x_train=x_train,
                             y_train=y_train,
                             x_test=x_test,
                             y_test=y_test,
                             x_pred=x_pred)     #训练模型

        if self.to_path == None:
            pass
        else:
            df_pred_first.to_csv(path=self.to_path)
        return df_pred_first




class Logic_Train_back:
    def __init__(self,end_time,lossdays,x_last_days,operation_center,in_path,to_path=None,back=0,df_set=None):
        self.end_time = end_time
        self.lossdays = lossdays
        self.x_last_days = x_last_days
        self.operation_center = operation_center
        self.in_path = in_path
        self.to_path = to_path
        self.back = back
        self.a = None
        self.df_set = df_set

    def back_merge(self, back_test_1_pred=None, pred_0=None,a=0):
        # 预测结果与预测集合
        # result_ls = []
        ls = ["rider_id"]
        if self.operation_center == False:
            ls = ["rider_id"]
        else:
            ls.append("operation_center_id")
        back_test_1_pred = pd.merge(back_test_1_pred, pred_0[["下单量"]],
                                    on=ls, how="left").fillna(-1)
        back_test_1_pred = back_test_1_pred.rename(columns={"下单量_x": "下单量", "下单量_y": "实际流失"})
        back_test_1_pred.loc[back_test_1_pred["实际流失"] > 0, "实际流失"] = 0
        back_test_1_pred.loc[back_test_1_pred["实际流失"] == -1, "实际流失"] = 1

        loss_dui = back_test_1_pred.query("预测值 == 1 and 实际流失 == 1 ")["下单量"].count()
        loss_zong = back_test_1_pred.query("预测值 == 1 ")["下单量"].count()

        keep_dui = back_test_1_pred.query("预测值 == 0 and 实际流失 == 0 ")["下单量"].count()
        keep_zong = back_test_1_pred.query("预测值 == 0 ")["下单量"].count()

        total_dui = back_test_1_pred["预测值"] == back_test_1_pred["实际流失"]
        total_dui = total_dui.sum()
        total_zong = keep_zong + loss_zong
        back_ls = ["回测：{}".format(self.a), loss_dui / loss_zong, keep_dui / keep_zong, total_dui / total_zong]
        # result_ls.append(back_ls)
        print("流失准确率：", loss_dui / loss_zong)
        print("留存准确率：", keep_dui / keep_zong)
        print("总准确率：", total_dui / total_zong)
        return back_ls

    def fit(self):
        #预测函数
        fit_agg = log_fit_agg_model(end_time=self.end_time,
                                    lossdays=self.lossdays,
                                    x_last_days=self.x_last_days,
                                    operation_center=self.operation_center,
                                    in_path=self.in_path,
                                    to_path=None,
                                    df_set=self.df_set)
        pred = fit_agg.logic_fit()     #输出预测函数预测结果
        pred.to_csv(self.to_path)

        #开始回测:
        if self.back == 0:
            pass
        else:
            result_ls =[]
            for i in range(self.back):
                a = i+1
                self.a = a
                #
                end_time = pd.to_datetime(self.end_time, format='%Y-%m-%d')
                back_time_end = end_time - dt.timedelta(days= a *lossdays)

                back_test_1 = log_fit_agg_model(end_time=back_time_end,
                                                lossdays=self.lossdays,
                                                x_last_days=self.x_last_days,
                                                operation_center=self.operation_center,
                                                in_path=self.in_path,
                                                to_path=None,
                                                df_set=self.df_set)

                back_test_1_pred = back_test_1.logic_fit()  #回测test1预测结果
                globals()["perd"] = back_test_1_pred
                ls = self.back_merge(back_test_1_pred=back_test_1_pred,pred_0=pred,a=a)
                pred = back_test_1_pred
                result_ls.append(ls)
            print("预测结果：",result_ls)


if __name__ == '__main__':
    path="/Users/shuuomousakai/Desktop/史蒂芬大春/办公桌面/编程/python/周报/数据库/rides_2023.pkl"
    # path = "/Users/shuuomousakai/Desktop/days_60/data.pkl"
    t1 = time.time()
    end_time = "2023-07-03"
    lossdays = 9
    back = 0
    x_last_days = 3
    operation_center = True
    in_path = path
    to_path = "/Users/shuuomousakai/Desktop/days_60/df_pred_first_xx.csv"

    df_set = pd.read_pickle(path)

    begin = Logic_Train_back(end_time=end_time,lossdays=lossdays,back=back,x_last_days=x_last_days,
                             operation_center=operation_center,in_path=in_path,to_path=to_path,df_set=df_set)
    begin.fit()

    t2= time.time()
    print("计算时间：",(t2-t1)/60,"min")
