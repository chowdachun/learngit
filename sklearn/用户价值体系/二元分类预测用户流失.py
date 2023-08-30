import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

path = "/Users/shuuomousakai/Desktop/史蒂芬大春/办公桌面/编程/sklearn/用户价值体系/data/rider_life_values_operation.csv"
# path = "/Users/admin/Desktop/史蒂芬大春/办公桌面/编程/sklearn/用户价值体系/data/rider_life_values_operation.csv"
df = pd.read_csv(path)


df.loc[((df["时间段首次下单渠道"] == "app") & (df["时间段最近下单渠道"] == "mini_program")),"渠道转换"] = "app转mini"
df.loc[((df["时间段首次下单渠道"] == "app") & (df["时间段最近下单渠道"] == "backend")),"渠道转换"] ="app转backend"
df.loc[((df["时间段首次下单渠道"] == "mini_program") & (df["时间段最近下单渠道"] == "app")),"渠道转换"] = "mini转app"
df.loc[((df["时间段首次下单渠道"] == "mini_program") & (df["时间段最近下单渠道"] == "backend")),"渠道转换"] = "mini转backend"
df.loc[((df["时间段首次下单渠道"] == "backend") & (df["时间段最近下单渠道"] == "app")),"渠道转换"] = "backend转app"
df.loc[((df["时间段首次下单渠道"] == "backend") & (df["时间段最近下单渠道"] == "mini_program")),"渠道转换"] = "backend转mini"

df["工作日%"] = df["工作日次数1_4"] /df["下单量"]
df["周末%"] = df["周末次数5_7"] /df["下单量"]
df["节假日%"] = df["节假日次数"] /df["下单量"]

dummies_col = ["时间段首次下单渠道","时间段最近下单渠道"]
df_dummies = pd.get_dummies(df,drop_first=True,columns=dummies_col)
df_dummies.drop(columns=["最早下单时间","最近下单时间","渠道转换","最近下单天数"],inplace=True)
df_dummies.fillna(0,inplace=True)
print(df_dummies.isnull().sum())


""" 拆分数据集 """
x = df_dummies.drop(["是否流失"],axis=1)
print("x",x.shape)
y = df_dummies["是否流失"]
print("y",y.shape)

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=42)
print("x_train列数：",x_train.shape)

""" 选择逻辑回归模型 """
lgr = LogisticRegression()
lgr.fit(x_train,y_train)
print("logi拟合效果",lgr.score(x_test,y_test))

y_pred = lgr.predict(x_test)
print("逻辑回归第一个测试特征：",x_test.iloc[99999])
print("逻辑回归验证第一个测试结果：",y[99999])



""" 选择DNN神经网络模型 """
dnn = Sequential()
dnn.add(Dense(units=12,input_dim=48,activation='relu')) #添加输入层
dnn.add(Dense(units=24,activation='relu'))          #添加隐藏层
dnn.add(Dense(units=1,activation='sigmoid'))        #添加输出层
dnn.summary()       #显示网络模型
#编译神经网络，指定优化器、损失函数、以及评估标准
dnn.compile(optimizer='RMSProp',        #指定优化器
            loss='binary_crossentropy',  #损失函数   binary_crossentropy
            metrics=["acc"])            #评估标准

""" 把Dataframe转换为numpy张量，训练神经网络模型 """
x_train = np.asarray(x_train).astype(np.float32)
x_test = np.asarray(x_test).astype(np.float32)
history = dnn.fit(x_train,y_train,
                  epochs=30,                #指定训练轮次
                  batch_size=64,            #指定数据批量
                  validation_split=0.2)     #直接从训练集中拆分验证集


def show_history(history): # 显示训练过程中的学习曲线
    loss = history.history['loss'] #训练损失
    val_loss = history.history['val_loss'] #验证损失
    epochs = range(1, len(loss) + 1) #训练轮次
    plt.figure(figsize=(12,4)) # 图片大小
    plt.subplot(1, 2, 1) #子图1
    plt.plot(epochs, loss, 'bo', label='Training loss') #训练损失
    plt.plot(epochs, val_loss, 'b', label='Validation loss') #验证损失
    plt.title('Training and validation loss') #图题
    plt.xlabel('Epochs') #X轴文字
    plt.ylabel('Loss') #Y轴文字
    plt.legend() #图例
    acc = history.history['acc'] #训练准确率
    val_acc = history.history['val_acc'] #验证准确率
    plt.subplot(1, 2, 2) #子图2
    plt.plot(epochs, acc, 'bo', label='Training acc') #训练准确率
    plt.plot(epochs, val_acc, 'b', label='Validation acc') #验证准确率
    plt.title('Training and validation accuracy') #图题
    plt.xlabel('Epochs') #X轴文字
    plt.ylabel('Accuracy') #Y轴文字
    plt.legend() #图例
    plt.show() #绘图
show_history(history) # 调用这个函数

result = dnn.evaluate(x_test, y_test) #评估测试集上的准确率
print('DNN的测试准确率为',"{0:.2f}%".format(result[1]))


""" 归一化重新训练训练神经网络模型 """
scaler = MinMaxScaler()     #创建归一化缩放器
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

history_ = dnn.fit(x_train,y_train,
                  epochs=30,                #指定训练轮次
                  batch_size=64,            #指定数据批量
                  validation_split=0.2)     #直接从训练集中拆分验证集
show_history(history_)

result = dnn.evaluate(x_test, y_test) #评估测试集上的准确率
print('DNN（归一化之后）的测试准确率为',"{0:.2f}%".format(result[1])*100)

