
import pandas as pd


class BusinessJudge:
    def __init__(self,in_path,capacity_limit=10):
        self.capacity_limit = capacity_limit
        self.data_A = pd.read_csv(in_path).fillna(0)
        self.data_A = self.data_A.astype(float)
        self.data = self.data_A.query(f"完单司机list_len >= @self.capacity_limit")
        self.data.drop("Unnamed: 0",axis=1,inplace=True)

    def operati(self):
        self.data["峰值达成"] = self.data["流水"] /self.data["预测流水max"]
        self.data["app和小程序"] = (self.data["app下单量"]+self.data["小程序下单量"])/self.data["下单量"]

        bins_1 = [-1, 0.45, 0.6,0.7, 0.85,10]
        bins_23 = [-1, 0.5, 0.65,0.75, 0.85,10]
        bins_4 = [-1, 0.5, 0.65,0.8, 0.9,10]
        bins_5 = [-1, 0.2, 0.4,0.6, 0.8,10]
        labels_daytime = ['E','D','C','B','A']
        self.data["峰值达成cut"] = pd.cut(self.data["峰值达成"], bins=bins_1, labels=labels_daytime, ordered=False)
        self.data["完成率cut"] = pd.cut(self.data["完成率"], bins=bins_23, labels=labels_daytime, ordered=False)
        self.data["需求满足率cut"] = pd.cut(self.data["需求满足率"], bins=bins_23, labels=labels_daytime, ordered=False)
        self.data["上座率cut"] = pd.cut(self.data["上座率"], bins=bins_4, labels=labels_daytime, ordered=False)
        self.data["app和小程序cut"] = pd.cut(self.data["app和小程序"], bins=bins_5, labels=labels_daytime, ordered=False)

        columns = ['峰值达成cut', '完成率cut','需求满足率cut','上座率cut','app和小程序cut',]
        self.data["cut"] = self.data[columns].apply(lambda row: ''.join(row), axis=1)
        self.data.to_csv("/Users/admin/Desktop/hhhh.csv")
        print(self.data)



        return

if __name__ == '__main__' :
    in_path = "/Users/admin/Desktop/经营诊断data_after.csv"
    capacity_limit = 10
    pred = BusinessJudge(in_path=in_path,capacity_limit=capacity_limit)
    df = pred.operati()