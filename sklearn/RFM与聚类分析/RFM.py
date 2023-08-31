
import pandas as pd
import os
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

plt.rcParams["font.sans-serif"]=['Arial Unicode MS']   # 设置字体
plt.rcParams["axes.unicode_minus"]=False     # 正常显示负号

"""
RFM分析模型：根据用户行为给用户分组画像
R：用户最近一次消费距离现在的天数
F：用户时间段内消费的频次
M：用户这段时间内的消费总额
"""

#导入源数据
data_path = os.path.abspath(os.path.join(os.getcwd(), "data/pendingdata.csv"))
df_pending = pd.read_csv(data_path)

#了解数据情况
print(df_pending)
print(df_pending.info())
print(df_pending.isnull().sum())

#修整数据
df_pending["流水"] = df_pending["流水"].fillna(0)
df_pending.drop(df_pending.loc[df_pending["服务订单"]==0].index,inplace=True)     #去除完单为0
df93 = df_pending.drop_duplicates()                                  #删除重复的数据行
df93["最近坐车时间"] = pd.to_datetime(df93["最近坐车时间"],format='%Y-%m-%d', errors='coerce')

df93["R值"] = (df93["最近坐车时间"].max() - df93["最近坐车时间"] ).dt.days     #距离最近消费天数

df = df93[["下单乘客ID","R值","服务订单","流水"]]
df.rename(columns={"下单乘客ID":"用户","R值":"R","服务订单":"F","流水":"M",},inplace=True)
print(df.info())
print(df.describe())        #再查看清洗后的数据是否存在异样(主要看min、count)


#查看RFM的直方图
# df['R'].plot(kind='hist', bins=20, title = '新进度分布直方图')
# df['F'].plot(kind='hist', bins=20, title = '消费频次分布直方图')
# df['M'].plot(kind='hist', bins=20, title = '消费总额分布直方图')
# plt.show()


"""
选取聚类算法中的 K-Means 算法，聚类算法是把空间位置相近的特征数据归为同一组。
"""
#手肘法选取 K 值，聚类算法的损失值曲线来直观确定簇的数量，当曲线区域平缓时，选择该K值作为蔟数，如果K过大会影响性能且分组的意义也越少

def show_elbow(df):
    distance_list=[]  #聚质心的距离（损失）
    K = range(1,9)    #K值范围
    for a in K:
        kmeans = KMeans(n_clusters=a,max_iter=100,n_init=10)  # 创建KMeans模型,n_clusters=k表示聚类簇数(聚类蔟=把数据分为多少个群组)；max_iter 分布不均匀、聚类间距离较大越大（迭代数越大性能要求越高）
        kmeans = kmeans.fit(df)                      #拟合模型，需要传入Dataframe数据类型
        distance_list.append(kmeans.inertia_)        #创建每个K值的损失值
    plt.plot(K,distance_list,'bx-')                  #绘图
    plt.xlabel("K蔟值")
    plt.ylabel("损失/距离均方误差")
    plt.title("k值手肘图")
    plt.show()

#通过计算kmeans.inertia_，损失值会随k(群数)增大而减少，所以选择曲线趋于平缓的拐点处值
show_elbow(df[["R"]])
show_elbow(df[["F"]])
show_elbow(df[["M"]])

#观察手肘图选择拐点，对应各图的K值
R,F,M  =  3,4,3

"""
创建和训练模型
"""
#创建对应的kmeans对象
kmeans_R = KMeans(n_clusters=3,n_init='auto')   #n_init=10是kmeans算法会多次初始化聚类中心，10是聚类算法执行10次
kmeans_F = KMeans(n_clusters=4,n_init='auto')
kmeans_M = KMeans(n_clusters=3,n_init='auto')

#训练模型
kmeans_R.fit(df[["R"]])
kmeans_F.fit(df[["F"]])
kmeans_M.fit(df[["M"]])

#给RFM值聚类,即产出结果
df["R值层级"] = kmeans_R.predict(df[["R"]])
df["F值层级"] = kmeans_F.predict(df[["F"]])
df["M值层级"] = kmeans_M.predict(df[["M"]])
print(df)

#创建一个方法，把聚合后的层级由乱序转换为按顺序排列
def change_cluster(cluster_name,targe,df,ascending=False):
    """
    :param cluster_name: 聚类后层级的列名
    :param targe: 训练列的名字
    :param df:需要处理的Dataframe
    :param ascending:排序方式（True or False）
    :return:返回一个Dataframe
    """
    df_new = df.groupby(cluster_name)[targe].mean().reset_index()       #把所有的层级展示出来
    df_new = df_new.sort_values(by=targe,ascending=ascending).reset_index(drop=True)      #然后给指标R排序
    df_new["index"] = df_new.index                                #把新排序后的顺序生成一列名为'index'列
    df_new = pd.merge(df,df_new[[cluster_name,"index"]],on=cluster_name)      #在通过把原来预测出来的R层级对应拼上新的'index'顺序
    df_new = df_new.drop([cluster_name],axis=1)                       #删除原来的聚类名称'R值层级'
    df_new = df_new.rename(columns={"index":cluster_name})        #然后将索引字段重命名为聚类名称字段
    return df_new


#使用排序方法，把层级指标重新排序
df = change_cluster("R值层级","R",df)
print("R",df.groupby("R值层级")["R"].mean())       #判断层级的大小与指标优劣是否一致

df = change_cluster("F值层级","F",df,ascending=True)
print("F",df.groupby("F值层级")["F"].mean())

df = change_cluster("M值层级","M",df,ascending=True)
print("M",df.groupby("M值层级")["M"].mean())

df["总分"] =df["R值层级"] + df["F值层级"] + df["M值层级"]
print(df)

#看看总分用户各占的数量
AA = df.groupby("总分").agg(AA=("用户","count"))
plt.plot(AA)
plt.show()

