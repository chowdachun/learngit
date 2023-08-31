
import pandas as pd


class BusinessJudge:
    def __init__(self,in_path,capacity_limit=10):
        self.capacity_limit = capacity_limit
        self.data_A = pd.read_csv(in_path).fillna(0)
        self.data_A = self.data_A.astype(float)
        self.data = self.data_A.query(f"完单司机list_len >= @self.capacity_limit")
        self.data.drop("Unnamed: 0",axis=1,inplace=True)
        self.data.set_index("operation_center_id",inplace=True)

    def operati(self):
        A = "峰值达成"
        B = "完成率"
        C = "需求满足率"
        A_A = "接驾量-完单量转化%"
        A_B = "接单量-接驾量转化%"
        A_C = "下单量-接单量转化%"
        B_A = "用户活跃率"
        B_B = "日均时长预估差值"
        B_C = "老新用户比例"

        for i in range(self.data.shape[0]):
            print("-------"*10)
            print("运营中心ID：",self.data.index[i])
            if self.data.iloc[i][A] < 0.7 :

                if ((self.data.iloc[i][B] < 0.7) and (self.data.iloc[i][C] < 0.8)):
                    print(B,self.data.iloc[i][B],C,self.data.iloc[i][C])
                    if self.data.iloc[i][A_A] < 0.7 :
                        print(self.data.iloc[i][A_A],"【接驾量-完单量】转化问题严重。")
                    elif self.data.iloc[i][A_A] < 0.8 :
                        print(self.data.iloc[i][A_A],"【接驾量-完单量】转化存在问题。")
                    else:
                        pass

                    if self.data.iloc[i][A_B] < 0.7 :
                        print(self.data.iloc[i][A_B],"【接单量-接驾量】转化问题严重。")
                    elif self.data.iloc[i][A_B] < 0.8 :
                        print(self.data.iloc[i][A_B],"【接单量-接驾量】转化存在问题。")
                    else:
                        pass

                    if self.data.iloc[i][A_C] < 0.7 :
                        print(self.data.iloc[i][A_C],"【下单量-接单量】转化问题严重。")
                    elif self.data.iloc[i][A_C] < 0.8 :
                        print(self.data.iloc[i][A_C],"【下单量-接单量】转化存在问题。")
                    else:
                        pass

                else:
                    print("下单量少，运力运载不满足")
                    self.data[B_A] = (self.data["下单用户list_len"]/self.data["总用户"] )*100
                    if self.data.iloc[i][B_A] < 4 :
                        print(self.data.iloc[i][B_A],B_A,"--老用户较多，优先考虑用户激活。")
                    else:
                        print(self.data.iloc[i][B_A],B_A, "--运营初期，增加拉新力度。")

                    if self.data.iloc[i][B_B] < 2 :
                        print(self.data.iloc[i][B_B],B_B,"--运力趟数成本高，急需增加上座率。")
                    else:
                        pass

                    if self.data.iloc[i][B_C] < 2 :
                        print(self.data.iloc[i][B_C],B_C,"--用户量少，需要持续拉新。")
                    else:
                        pass


            else:
                print("经营状态良好。")






        return  print(self.data)

if __name__ == '__main__' :
    in_path = "/Users/admin/Desktop/经营诊断data_after.csv"
    capacity_limit = 10
    pred = BusinessJudge(in_path=in_path,capacity_limit=capacity_limit)
    df = pred.operati()