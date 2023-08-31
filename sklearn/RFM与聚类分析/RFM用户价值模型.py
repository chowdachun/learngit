
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class RFMModel:
    def __init__(self,in_path,star,end,op=None,to_path=None,):
        self.in_path = in_path
        self.to_path = to_path
        self.end = pd.to_datetime(end)
        self.star = pd.to_datetime(star)
        if op==None:
            self.df = pd.read_pickle(in_path).\
                query(f"serving_at >=@self.star and @self.end > serving_at ")
        else:
            self.df = pd.read_pickle(in_path).\
                query(f"serving_at >=@self.star and @self.end > serving_at & @op == operation_center_id")

    def data_filter(self):
        data = self.df.query("actual_dropoff_at.notnull()").groupby(["rider_id","operation_center_id"]).agg(
            F值=("id","count"),
            M值=("payments_price", lambda x: x.sum() / 100.0),
            最近下单时间=("serving_at","max"))

        data["R值"] = ((data["最近下单时间"] -self.star)/7).dt.days
        data = data[["R值","F值","M值"]].reset_index()

        # # 归一化缩放
        # def minmax(data,column,name):
        #     scaler =  MinMaxScaler()
        #     col_scaler = scaler.fit_transform(column)
        #     col_scaler = col_scaler.flatten()
        #     col = pd.Series(col_scaler)
        #     data = pd.concat([data,col],axis=1)
        #     data = data.rename(columns={0: name})
        #     return data
        #
        # data = minmax(data,data[["R值"]],"r_scaler")
        # data = minmax(data,data[["F值"]],"f_scaler")
        # data = minmax(data,data[["M值"]],"m_scaler")

        # 根据分位给列赋值
        def fenwei(data,column,name):
            cut_100 = column.quantile(1)
            data.loc[column<= cut_100, name] = 5
            cut_80 = column.quantile(0.8)
            data.loc[column <= cut_80, name] = 4
            cut_60 = column.quantile(0.6)
            data.loc[column <= cut_60, name] = 3
            cut_40 = column.quantile(0.4)
            data.loc[column <= cut_40, name] = 2
            cut_20 = column.quantile(0.2)
            data.loc[column <= cut_20, name] = 1
            return data

        data = fenwei(data,data["R值"],"r_cut")
        data = fenwei(data,data["F值"],"f_cut")
        data = fenwei(data,data["M值"],"m_cut")

        # 根据分为给赋值评分
        r_mean = data["r_cut"].mean()
        f_mean = data["f_cut"].mean()
        m_mean = data["m_cut"].mean()
        data["r_评分"] = np.where(data["r_cut"]>r_mean,1,0)
        data["f_评分"] = np.where(data["f_cut"]>f_mean,1,0)
        data["m_评分"] = np.where(data["m_cut"]>m_mean,1,0)

        # 赋予标签
        data.loc[((data["r_评分"]==1) & (data["f_评分"]==1) & (data["m_评分"]==1)),"级别"] = '重要价值用户'
        data.loc[((data["r_评分"]==0) & (data["f_评分"]==1) & (data["m_评分"]==1)),"级别"] = '重要保持用户'
        data.loc[((data["r_评分"]==1) & (data["f_评分"]==0) & (data["m_评分"]==1)),"级别"] = '重要发展用户'
        data.loc[((data["r_评分"]==0) & (data["f_评分"]==0) & (data["m_评分"]==1)),"级别"] = '重要挽留用户'
        data.loc[((data["r_评分"]==1) & (data["f_评分"]==1) & (data["m_评分"]==0)),"级别"] = '一般价值用户'
        data.loc[((data["r_评分"]==0) & (data["f_评分"]==1) & (data["m_评分"]==0)),"级别"] = '一般保持用户'
        data.loc[((data["r_评分"]==1) & (data["f_评分"]==0) & (data["m_评分"]==0)),"级别"] = '一般发展用户'
        data.loc[((data["r_评分"]==0) & (data["f_评分"]==0) & (data["m_评分"]==0)),"级别"] = '一般挽留用户'

        print(data["级别"].value_counts())
        print(data)
        return data



if __name__ == '__main__':
    end = '2023-08-09'
    star = '2023-01-01'
    op = 93
    in_path = '/Users/admin/Desktop/史蒂芬大春/办公桌面/编程/python/周报/数据库/rides_2023.pkl'

    rfm = RFMModel(end=end,star=star,in_path=in_path,op=op)
    rfm.data_filter()
