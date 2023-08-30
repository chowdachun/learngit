
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

df_panding_path = os.path.abspath(os.path.join(os.getcwd(), "data/pandingdata.csv"))
df_panding = pd.read_csv(df_panding_path)

"""
需求：通过前4个月预测后1个月的ltv
"""
"""
数据预处理
"""
#选择特征数据

df = df_panding[["serving_at","rider_id","总订单","服务订单","服务行程","流水"]]

#数据整理清洗
df["流水"] = df["流水"].fillna(0)
df["serving_at"] = pd.to_datetime(df["serving_at"],errors='coerce')

# print(df.info())
# print(df.describe())
# print(df)

#选择前三个月数据
star_time = '2023-01-01'
end_time = '2023-05-01'     #"<"
pd.to_datetime([star_time,end_time])
df_3m = df.query(f"serving_at >= @star_time and serving_at < @end_time")
print(df_3m)

#计算每个用户的特征
df_uesr = pd.DataFrame(df_3m["rider_id"].unique())
df_uesr.columns=["用户ID"]

df_value = df_3m.groupby("rider_id")["总订单"].sum().reset_index()
df_value.columns = ["用户ID","总订单",]
df_uesr = pd.merge(df_uesr,df_value[["用户ID","总订单"]],on="用户ID")

df_value = df_3m.groupby("rider_id")["服务订单"].sum().reset_index()
df_value.columns = ["用户ID","服务订单",]
df_uesr = pd.merge(df_uesr,df_value[["用户ID","服务订单"]],on="用户ID")

df_value = df_3m.groupby("rider_id")["流水"].sum().reset_index()
df_value.columns = ["用户ID","流水",]
df_uesr = pd.merge(df_uesr,df_value[["用户ID","流水"]],on="用户ID")

#数值越大，最近活跃程度越高
df_value = df_3m.groupby("rider_id")["serving_at"].max().reset_index()
df_value.columns = ["用户ID","最近服务时间",]
df_value["最近乘车天数"] = ( df_value['最近服务时间'] -df_value['最近服务时间'].min() ).dt.days
df_uesr = pd.merge(df_uesr,df_value[["用户ID","最近乘车天数"]],on="用户ID")


"""
算出整段时间的流水，再把流水放到df_user作为标签
"""
#计算整个周期的流水值

df_value = df.groupby("rider_id")["流水"].sum().reset_index()
df_value.columns = ["用户ID","周期总流水"]
df_uesr = pd.merge(df_uesr,df_value[["用户ID","周期总流水"]],on="用户ID")
print(df_uesr)

"""
创建特征集和标签集
"""
x = df_uesr.drop(["用户ID","周期总流水"],axis=1)
# print(x)
y = df_uesr["周期总流水"]
# print(y)

"""
拆分训练集、验证集和测试集
"""
#先把总体数据分为训练集和其他
x_train ,x_other ,y_train ,y_other = train_test_split(x,y,train_size=0.3,random_state=42)
#再把其他集拆分为测试集和验证集
x_test , x_vaild ,y_test ,y_vaild = train_test_split(x_other,y_other,train_size=0.5,random_state=42)

print(x_train.shape)
print(x_test.shape)
print(x_vaild.shape)
print(y_train.shape)
print(y_test.shape)
print(y_vaild.shape)


"""
选择合适的训练模型，常见的回归算法有：
    线性回归、贝叶斯回归、SVM回归、决策树、随机森林、AdaBoost和XGBoost等梯度提升算法、神经网络算法
在解决问题时可以选择多种贴合的模型算法，相互比较之后再确定比较合适的模型
"""
#选择线性回归、决策树、随机森林这三个模型进行比较
model_lr=LinearRegression()
model_dtr=DecisionTreeRegressor()
model_rdr=RandomForestRegressor()

#训练模型
model_lr.fit(x_train,y_train)
model_dtr.fit(x_train,y_train)
model_rdr.fit(x_train,y_train)

#模型评估，对验证集分别进行预测
y_valid_preds_lr = model_lr.predict(x_vaild)
y_valid_preds_dtr = model_dtr.predict(x_vaild)
y_valid_preds_rdr = model_rdr.predict(x_vaild)

#选择R方检验，用测试集vaild来评测模型效果
print("线性回归R2",r2_score(y_vaild,model_lr.predict(x_vaild)))  #（真值y，验证集）
print("决策树R2",r2_score(y_vaild,model_dtr.predict(x_vaild)))
print("随机森林R2",r2_score(y_vaild,model_rdr.predict(x_vaild)))


"""
至此模型训练完成
"""
#根据模型分数，线性回归R2评分更高
y_test_preds_lr = model_lr.predict(x_test)
print("线性回归测试集R2",r2_score(y_test,model_lr.predict(x_test)))