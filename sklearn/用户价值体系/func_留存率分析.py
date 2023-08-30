
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas.core.dtypes.common import is_numeric_dtype
import lifelines

plt.rcParams["font.sans-serif" ] =['Arial Unicode MS']   # 设置字体
plt.rcParams["axes.unicode_minus" ] =False     # 正常显示负号


value_path = "/Users/admin/Desktop/史蒂芬大春/办公桌面/编程/sklearn/用户价值体系/data/rider_life_values_operation.csv"
df = pd.read_csv(value_path)
data_dummies = df
# print(df.describe())
print(is_numeric_dtype(df["时间段首次下单渠道"]))  # 判断列是否为数值

df["响应时长min_mean"] = pd.to_numeric(df["响应时长min_mean"] ,errors='coerce')  # 把字段转换为数值类型，errors='coerce'表示无法转换的变为NaN
df["响应时长min_mean"].fillna(0 ,inplace=True)

df.loc[((df["时间段首次下单渠道"] == "app") & (df["时间段最近下单渠道"] == "mini_program")),"渠道转换"] = "app转mini"
df.loc[((df["时间段首次下单渠道"] == "app") & (df["时间段最近下单渠道"] == "backend")),"渠道转换"] ="app转backend"
df.loc[((df["时间段首次下单渠道"] == "mini_program") & (df["时间段最近下单渠道"] == "app")),"渠道转换"] = "mini转app"
df.loc[((df["时间段首次下单渠道"] == "mini_program") & (df["时间段最近下单渠道"] == "backend")),"渠道转换"] = "mini转backend"
df.loc[((df["时间段首次下单渠道"] == "backend") & (df["时间段最近下单渠道"] == "app")),"渠道转换"] = "backend转app"
df.loc[((df["时间段首次下单渠道"] == "backend") & (df["时间段最近下单渠道"] == "mini_program")),"渠道转换"] = "backend转mini"

df["工作日%"] = df["工作日次数1_4"] /df["下单量"]
df["周末%"] = df["周末次数5_7"] /df["下单量"]
df["节假日%"] = df["节假日次数"] /df["下单量"]



def pie_charts(ls=None,df=df,operation=None):
    if ls is None:
        ls = ["时间段首次下单渠道", "时间段最近下单渠道","渠道转换","是否流失"]
    if operation is None:
        pass
    else:
        df = df.query(f"operation_center_id == @operation")
    plt.figure(figsize=(16 ,5))
    plt.subplot(1, 4, 1)  # 创建一个具有 2 行 2 列的子图网格，并选择第一个子图作为当前活动子图
    ax = df.groupby(ls[0]).count()["rider_id"].plot.pie(autopct='%1.0f%%')
    plt.title("首次下单渠道")
    plt.subplot(1, 4, 2)
    ax = df.groupby(ls[1]).count()["rider_id"].plot.pie(autopct='%1.0f%%')
    plt.title("最近下单渠道")
    plt.subplot(1, 4, 3)
    ax = df.query("渠道转换.notnull()").groupby(ls[2]).count()["rider_id"].plot.pie(autopct='%1.0f%%')
    plt.title("渠道转换")
    plt.subplot(1,4,4)
    ax = df.groupby(ls[3]).count()["rider_id"].plot.pie(autopct='%1.0f%%')
    plt.title("流失占比")
    plt.show()

# pie_charts(df=df,operation=369)


#数据清洗
# df["性别"].replace(to_replace='男',value=1,inplace=True)
# df["性别"].replace(to_replace='女',value=0,inplace=True)   #文本替换诶数值


"""
运用lifelines绘制用户生命周期曲线
"""
def lifelines_chart(df=df,operation=None,event_observed="总活跃天数"):
    if operation is None:
        pass
    else:
        df = df.query(f"operation_center_id == @operation")
    kmf = lifelines.KaplanMeierFitter()     #创建KMF模型
    kmf.fit(df["总活跃天数"],
            event_observed=df[event_observed],
            label="用户生命曲线")
    fig,ax = plt.subplots(figsize=(10,6))   #创建画布
    kmf.plot(ax=ax)     #画图
    ax.set_title('Kaplan-Meier留存曲线-用户生命周期') #图题
    ax.set_xlabel('总活跃天数') #X轴标签
    ax.set_ylabel('留存率(%)') #Y轴标签
    plt.show() #显示图片

# lifelines_chart(df=df,operation=[2,54,93,228,359])
# lifelines_chart(df=df,operation=[2,54,93,228,359],event_observed="完单量")


def life_by_cat(item='时间段首次下单渠道',x='总活跃天数',y='是否流失',df=df,ax=None,operation=None,cut=False):
    if operation is None:
        pass
    else:
        df = df.query(f"operation_center_id == @operation")

    if cut is False:
        pass
    else:
        bins = [-0.001,df[item].max()*0.2,df[item].max()*0.4,df[item].max()*0.6,df[item].max()*0.8,df[item].max()]
        # bins = [-0.001,df[item].max()*0.2,df[item].max()*0.4,df[item].max()*0.6,df[item].max()*0.8,df[item].max()]
        # labels = ["0-0.2","0.2-0.4","0.4-0.6","0.6-0.8","0.8-1"]
        labels = [df[item].max()*0.2,df[item].max()*0.4,df[item].max()*0.6,df[item].max()*0.8,df[item].max()]
        df[item] = pd.cut(df[item], bins=bins, labels=labels)
        print(df[item])
    for cat in df[item].unique():
        idx = df[item] == cat #当前类别
        kmf = lifelines.KaplanMeierFitter()     #创建类对象
        kmf.fit(df[idx][x],event_observed=df[idx][y],label=cat)  #拟合模型
        kmf.plot(ax=ax,label=cat)

fig,ax = plt.subplots(figsize=(10,6))   #创建画布
life_by_cat(item="覆盖线路数量",ax=ax,operation=[2,54,93,228,359],cut=True)
ax.set_title("对留存天数的影响")
plt.show()

"""各指标与用户活跃天数关系"""
#1、线路覆盖越多，对用户的留存越久
#2、首单与最后一个的下单渠道的留存率排名都一致（app>mini_program>backend）
#3、自活次数越多，留存时间越久
#4、用户完成率在0.6-0.8的活跃天数最久
#5、周末订单占比0.2-0.8的用户200天留存率达80%
#6、工作日订单占比0.2-0.8的用户200天留存率达75%
#7、节假日订单占比0.2-0.4的用户200天留存率达85%，0.6-0.8用户200天留存率75%
#8、差评流失的用户活跃天数远比正常用户高


"""Cox 危害系数模型：预测用户留存概率"""

#1、对文本字段进行哑变量处理
dummies_col = ["时间段首次下单渠道","时间段最近下单渠道"]
df_dummies = pd.get_dummies(data_dummies,drop_first=True,columns=dummies_col)
df_dummies.drop(columns=["最早下单时间","最近下单时间","渠道转换","扫码下单量"],inplace=True)
df_dummies.fillna(0,inplace=True)
print(df_dummies.isnull().sum())

def lifelines_cox(df=df,x_col='总活跃天数',y_col='是否流失',):
    print(df.isnull().sum())
    #2、创建并拟合模型
    cph = lifelines.CoxPHFitter(penalizer=0.0002)
    #训练模型
    cph.fit(df,duration_col=x_col,event_col=y_col,show_progress=True,)

    """Cox 危害系数模型：分析影响留存的因子"""
    fig_cox, ax = plt.subplots(figsize=(12,7))
    ax.set_title('各个特征的留存相关系数')
    cph.plot(ax=ax)

    """Cox：绘制模型的风险曲线 (plot_partial_hazard())"""
    cph.predict_survival_function(df.loc[3]).plot()
    plt.show()

lifelines_cox(df=df_dummies,)

