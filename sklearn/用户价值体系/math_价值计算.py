
import aaaa as f
import os
# import seaborn as sns
# from matplotlib import pyplot as plt
import time
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=False,nb_workers=2)


# plt.rcParams["font.sans-serif"]=['Arial Unicode MS']   # 设置字体
# plt.rcParams["axes.unicode_minus"]=False     # 正常显示负号



#/Users/shuuomousakai/Desktop/史蒂芬大春/办公桌面/编程/sklearn/用户价值体系/data/arg_holiday.xlsx
#结束日期

# path = "/Users/admin/Desktop/史蒂芬大春/办公桌面/编程/python/周报/数据库/rides_2018.csv"
# path = "/Users/shuuomousakai/Desktop/史蒂芬大春/办公桌面/编程/python/周报/数据库/rides.csv"
# path = "/Users/shuuomousakai/Desktop/dftest.csv"
# path = "/Users/admin/Desktop/dftest.csv"
# path = "/Users/admin/Desktop/史蒂芬大春/办公桌面/编程/python/周报/数据库/rides_2023.pkl"
# path = "/Users/admin/Desktop/days_60/data.pkl"
if __name__ == "__main__":
    days = 3000
    end = "2023-08-28"  #"<"
    t1 = time.time()
    parent_folder = os.path.dirname(os.getcwd())
    grandparent_folder = os.path.dirname(parent_folder)
    path = "/Users/admin/Desktop/史蒂芬大春/办公桌面/编程/python/周报/数据库/rides.pkl"

    # 获取全数据集
    data_train = f.RiderValue(end_time=end,
                              days=3000,
                              in_path=path,
                              # route_id=[614,615],
                              # x_last_days=30,
                              # x_last_orders=3,
                              lossdays=90,
                              to_path="/Users/admin/Desktop/史蒂芬大春/办公桌面/编程/sklearn/用户价值体系/data/全量用户特征画像opr.csv",
                              # to_path="/Users/admin/Desktop/7-10-30下单用户.csv",
                              operation_center=True)
    data_train.fit(save=False)
    train = data_train.fit_timeflow(save=True)
    t2 = time.time()
    print("计算时间：", (t2 - t1) / 60, "mins")


#线性相关性展示
# sns.heatmap(df.corr(),cmap="YlGnBu", annot = True)
# plt.show()




