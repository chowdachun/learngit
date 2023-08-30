
import pandas as pd
# import multiprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#逻辑回归计算模型
class Logic_Pred_Rider_Loss:
    def __init__(self):
        self.x_pred_first =None

    #数据清洗+特征转换
    def add_feature_clean(self,df):  #增加特征
        df.loc[
            ((df["时间段首次下单渠道"] == "app") & (df["时间段最近下单渠道"] == "mini_program")), "渠道转换"] = "app转mini"
        df.loc[((df["时间段首次下单渠道"] == "app") & (df["时间段最近下单渠道"] == "backend")), "渠道转换"] = "app转backend"
        df.loc[
            ((df["时间段首次下单渠道"] == "mini_program") & (df["时间段最近下单渠道"] == "app")), "渠道转换"] = "mini转app"
        df.loc[((df["时间段首次下单渠道"] == "mini_program") & (
                    df["时间段最近下单渠道"] == "backend")), "渠道转换"] = "mini转backend"
        df.loc[((df["时间段首次下单渠道"] == "backend") & (df["时间段最近下单渠道"] == "app")), "渠道转换"] = "backend转app"
        df.loc[((df["时间段首次下单渠道"] == "backend") & (
                    df["时间段最近下单渠道"] == "mini_program")), "渠道转换"] = "backend转mini"

        df["工作日%"] = df["工作日次数1_4"] / df["下单量"]
        df["周末%"] = df["周末次数5_7"] / df["下单量"]
        df["节假日%"] = df["节假日次数"] / df["下单量"]

        dummies_col = ["时间段首次下单渠道", "时间段最近下单渠道"]
        df = pd.get_dummies(df, drop_first=True, columns=dummies_col)
        df.drop(columns=["渠道转换", "最近下单天数","最早下单时间", "最近下单时间"], inplace=True)
        df.fillna(0, inplace=True)
        # print(df.isnull().sum())
        return df

    # 拆分训练集、测试集和生成预测集
    def split_train_pred(self,df_train,df_pred):
        self.x_pred_first = df_pred
        x_train = df_train.drop(["是否流失"], axis=1)
        y_train = df_train["是否流失"]
        x_pred = df_pred.drop(["是否流失"], axis=1)
        y_pred = df_pred["是否流失"]

        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.8, random_state=42)
        return x_train,x_test,x_pred , y_train,y_test ,y_pred

    #用逻辑回归二元分类
    def logi_regression(self,x_train,y_train,x_test,y_test,x_pred):
        print("x_train",x_train.shape)
        print("y_train",y_train.shape)
        print("x_test",x_test.shape)
        print("y_test",y_test.shape)
        print("x_pred",x_pred.shape)
        lgr = LogisticRegression()
        lgr.fit(x_train, y_train)
        print("logi的x_test拟合效果", lgr.score(x_test, y_test))
        y_pred = lgr.predict(x_pred)
        self.x_pred_first["预测值"] = y_pred
        return self.x_pred_first
