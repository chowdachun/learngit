
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge ,Lasso
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
import seaborn as sns
from matplotlib import pyplot as plt

plt.rcParams["font.sans-serif"]=['Arial Unicode MS']   # 设置字体
plt.rcParams["axes.unicode_minus"]=False     # 正常显示负号

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
类别特征转换，pandas的get_dummies函数把类别字段的每一个特征都变成0、1
"""

df_dummies = df_panding[["serving_at","rider_id","总订单","服务订单","服务行程","流水","client"]]
#数据整理清洗
df_dummies["流水"] =df_dummies["流水"].fillna(0)
df_dummies["serving_at"] = pd.to_datetime(df_dummies["serving_at"],errors='coerce')
df_3m_dummies = df_dummies.query(f"serving_at >= @star_time and serving_at < @end_time")
df_value_dummies = df_3m_dummies.groupby("rider_id").agg(A=("client",lambda x:x.max())).reset_index()
df_value_dummies.columns = ["用户ID","client",]
#merge合并
df_user_dummies = pd.merge(df_uesr,df_value_dummies[["用户ID","client",]],on="用户ID")
#新增了"client"字段，用于做类别特征转换
print(df_user_dummies)

# 把多分类字段转换为二分类虚拟变量
client_features = ["client"] #要转换的特征列表
df_user_dummies = pd.get_dummies(df_user_dummies,drop_first=True,columns=client_features) #创建哑变量
print(df_user_dummies)

"""
创建分类特征集和标签集
"""
x_dummies = df_user_dummies.drop(["用户ID","周期总流水"],axis=1)
# print(x)
y_dummies = df_user_dummies["周期总流水"]
# print(y)

#先把总体数据分为训练集和其他
x_dummies_train ,x_dummies_other ,y_dummies_train ,y_dummies_other = train_test_split(x_dummies,y_dummies,train_size=0.3,random_state=42)

x_dummies_test,x_dummies_vaild,y_dummies_test,y_dummies_vaild = \
    train_test_split(x_dummies_other,y_dummies_other,train_size=0.5,random_state=42)


print("-------------------"*2)
"""
通过cross_val_score进行k拆分，交叉验证
"""
model_lr = LinearRegression()
scores = cross_val_score(model_lr,x_dummies_train,y_dummies_train,cv=7,scoring='r2')
scores_avg = sum(scores)/len(scores)
print("k拆分的交叉验证分数：",scores)
print("交叉验证的r2平均分数：",scores_avg)

"""
通过网格化搜索找到最优的调参
"""
print("---------------随机森林网格化调参-------------")

#------调参开始-------
model_rfr = RandomForestRegressor()
model_rfr_param = {
    "n_estimators":[30,50,100],    #森林中树的数量，默认是100
    "criterion": ["squared_error"],        #树的分裂准则
    "max_depth":[3,5,7,9],      #决策数子节点深度
    "min_samples_split":[2,3,4],    #数内部节点划分的最小样本数
    "min_samples_leaf":[1,2,3],     #叶子节点的最小样本数
    "bootstrap":[False,True],        #是否抽取放回，默认True
}
model_rfr_gird = GridSearchCV(model_rfr,param_grid=model_rfr_param,cv=5,
                              scoring='r2',n_jobs=10,verbose=1,error_score=0)   #n_jobs=10搜索过程中并行的运算数量
model_rfr_gird.fit(x_dummies_train,y_dummies_train)
print("最佳参数组合：",model_rfr_gird.best_params_)
#-------调参结束------

model_rfr = RandomForestRegressor()

#训练模型
# model_rfr_bestparam.fit(x_dummies_train,y_dummies_train)
model_rfr.fit(x_dummies_train,y_dummies_train)

#其训练集得分和测试集得分
print("训练集-最优参随机森林：",r2_score(y_dummies_train,model_rfr_gird.predict(x_dummies_train)))
print("训练集-随机森林：",r2_score(y_dummies_train,model_rfr.predict(x_dummies_train)))
print("验证集-最优参随机森林：",r2_score(y_dummies_vaild,model_rfr_gird.predict(x_dummies_vaild)))
print("验证集-随机森林：",r2_score(y_dummies_vaild,model_rfr.predict(x_dummies_vaild)))
print("测试集-最优参随机森林：",r2_score(y_dummies_test,model_rfr_gird.predict(x_dummies_test)))
print("测试集-随机森林：",r2_score(y_dummies_test,model_rfr.predict(x_dummies_test)))
#结果：在测试集上有略微提升

print("---------------决策树网格化调参-------------")

#------调参开始-------
model_dtr = DecisionTreeRegressor()
model_dtr_param = {
    "splitter":['best','random'],    #森林中树的数量，默认是100
    "criterion": ["squared_error","mae","friedman_mse"],        #树的分裂准则
    "max_depth":[3,5,7,9],      #决策数子节点深度
    "min_samples_split":[2,3,4],    #数内部节点划分的最小样本数
    "min_samples_leaf":[1,2,3],     #叶子节点的最小样本数
}
model_dtr_gird = GridSearchCV(model_dtr,param_grid=model_dtr_param,cv=5,
                              scoring='r2',n_jobs=10,verbose=1,error_score=0)   #n_jobs=10搜索过程中并行的运算数量
model_dtr_gird.fit(x_dummies_train,y_dummies_train)
print("最佳参数组合：",model_dtr_gird.best_params_)
#-------调参结束------

model_dtr_2 =DecisionTreeRegressor()
model_dtr_2.fit(x_dummies_train,y_dummies_train)

#其训练集得分和测试集得分
print("训练集-最优参决策树：",r2_score(y_dummies_train,model_dtr_gird.predict(x_dummies_train)))
print("训练集-决策树：",r2_score(y_dummies_train,model_dtr_2.predict(x_dummies_train)))
print("验证集-最优参决策树：",r2_score(y_dummies_vaild,model_dtr_gird.predict(x_dummies_vaild)))
print("验证集-决策树：",r2_score(y_dummies_vaild,model_dtr_2.predict(x_dummies_vaild)))
print("测试集-最优参决策树：",r2_score(y_dummies_test,model_dtr_gird.predict(x_dummies_test)))
print("测试集-决策树：",r2_score(y_dummies_test,model_dtr_2.predict(x_dummies_test)))
#结果：