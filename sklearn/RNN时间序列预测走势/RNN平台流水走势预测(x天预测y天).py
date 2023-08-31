import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from keras.models import Sequential #导入序贯模型
from keras.layers import Dense, LSTM , GRU #导入全连接层和LSTM层
from keras.src.layers import Dropout
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
plt.rcParams["font.sans-serif"]=['Arial Unicode MS']   # 设置字体
plt.rcParams["axes.unicode_minus"]=False     # 正常显示负号


class RnnPred:
    def __init__(self,in_path):
        self.in_path = in_path
        self.df = pd.read_csv(in_path,index_col='订单服务日期',parse_dates=['订单服务日期'])

    def data_clean(self,split_date_after,y_pred_day,x_train_day):
        split_date_after = pd.to_datetime(split_date_after)
        self.split_date_after = split_date_after
        split_date_diff = split_date_after - dt.timedelta(days=1)
        self.df = self.df[:split_date_diff]
        print("数据集的空值数量：",self.df.isna().sum())
        print("数据集的是否有负数：",(self.df[self.df.columns[0]]<0).values.any())
        split_date_front = split_date_after - dt.timedelta(days=y_pred_day)

        train = self.df[:split_date_front].iloc[:,0:1].values
        test = self.df[split_date_front:].iloc[:,0:1].values
        print('训练集的形状是：',train.shape)
        print('测试集的形状是：',test.shape)

        # 归一化缩放
        Scaler = MinMaxScaler(feature_range=(0,1))
        train = Scaler.fit_transform(train)

        print("-------"*15)
        # 构建特征集和标签集
        x_train = []
        y_train = []
        for i in range(x_train_day,train.size-y_pred_day-1):
            x_train.append(train[i-x_train_day:i,0])
            y_train.append(train[i+1:i+1+y_pred_day,0])
        x_train,y_train = np.array(x_train),np.array(y_train)
        x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
        print(x_train.shape)
        print(y_train.shape)

        # 构建测试集
        x_test= []
        y_test= []
        traintest = self.df['流水'][:]      # 获取整个数据集，在下一步获取长度位置
        inputs = traintest[len(traintest)-x_train_day-y_pred_day:len(traintest)-y_pred_day].values
        inputs = inputs.reshape(-1,1)           # 转换数组
        inputs = Scaler.fit_transform(inputs)   # 归一化
        print(inputs.size)
        x_test.append(inputs)
        x_test = np.array(x_test)
        x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
        print(x_test.shape)

        # 选择神经网络框架LSTM
        # RNN_LSTM = Sequential()     # 序贯模型
        # RNN_LSTM.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1],1))) #输入层LSTM,return_sequences返回输出序列
        # RNN_LSTM.add(Dropout(0.2))      # 随机丢弃部分神经元，避免过拟合
        # RNN_LSTM.add(LSTM(units=100,return_sequences=True))  # 中间1层LSTM
        # RNN_LSTM.add(Dropout(0.2))
        # RNN_LSTM.add(LSTM(units=100,return_sequences=True))  # 中间2层LSTM
        # RNN_LSTM.add(Dropout(0.2))
        # RNN_LSTM.add(LSTM(units=50))  # 中间3层LSTM,不需要返回输出序列
        # RNN_LSTM.add(Dropout(0.2))
        # RNN_LSTM.add(Dense(units=7))

        # GRU
        RNN_LSTM = Sequential()
        RNN_LSTM.add(LSTM(units=150,return_sequences=True,input_shape=(x_train.shape[1],1)))
        RNN_LSTM.add(LSTM(units=100,return_sequences=True))  # 中间1层LSTM
        RNN_LSTM.add(LSTM(units=100,return_sequences=True))  # 中间2层LSTM
        RNN_LSTM.add(LSTM(units=50))
        RNN_LSTM.add(Dense(units=7))

        RNN_LSTM.compile(loss='mean_squared_error', # 损失函数(多元分类：categorical_crossentropy，二元分类：binary_crossentropy，回归：mse)
                         optimizer= 'RMSprop',          # 优化器(分类：accuracy，RMSprop，Adam)
                         metrics=['accuracy'])  # 评估指标

        RNN_LSTM.summary()  # 输出神经网络结构信息

        print("y_train：",y_train)
        # 训练模型
        history= RNN_LSTM.fit(x_train,y_train,epochs=30,validation_split=0.2)

        # 利用模型进行预测
        pred_test = RNN_LSTM.predict(x_test)


        # 模型损失曲线
        def show_history(history):  # 显示训练过程中的学习曲线
            loss = history.history['loss']  # 训练损失
            val_loss = history.history['val_loss']  # 验证损失
            epochs = range(1, len(loss) + 1)  # 训练轮次
            plt.figure(figsize=(12, 4))  # 图片大小
            plt.subplot(1, 2, 1)  # 子图1
            plt.plot(epochs, loss, 'bo', label='Training loss')  # 训练损失
            plt.plot(epochs, val_loss, 'b', label='Validation loss')  # 验证损失
            plt.title('Training and validation loss')  # 图题
            plt.xlabel('Epochs')  # X轴文字
            plt.ylabel('Loss')  # Y轴文字
            plt.legend()  # 图例
            acc = history.history['accuracy']  # 训练准确率(与compile的metrics值相同)
            val_acc = history.history['val_accuracy']  # 验证准确率(与compile的metrics值相同)
            plt.subplot(1, 2, 2)  # 子图2
            plt.plot(epochs, acc, 'bo', label='Training acc')  # 训练准确率
            plt.plot(epochs, val_acc, 'b', label='Validation acc')  # 验证准确率
            plt.title('Training and validation accuracy')  # 图题
            plt.xlabel('Epochs')  # X轴文字
            plt.ylabel('Accuracy')  # Y轴文字
            plt.legend()  # 图例
            plt.show()  # 绘图
        show_history(history)  # 调用这个函数


        # 反归一化
        pred_test = Scaler.inverse_transform(pred_test)

        # 将多为数据展平为一维数组
        pred_test = np.ravel(pred_test)

        y_test = self.df["流水"][(len(traintest)-len(test)):].values
        print("y_test:",y_test)
        print("pred_test:",pred_test)
        print("MSE损失值 {}.".format(mean_squared_error(y_test, pred_test)))
        print("R方误差损失值 {}.".format(r2_score(y_test, pred_test)))

        pred_test = np.ravel(pred_test)
        def plot_prediction(y_test, pred_test):
            plt.plot(y_test, label='y_test')
            plt.plot(pred_test, label='pred')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.title('Prediction Results')
            plt.legend()
            plt.show()
        plot_prediction(y_test,pred_test)

        print("pred",pred_test)

        # 构建预测集
        x_pred= []
        traintest = self.df['流水'][:]      # 获取整个数据集，在下一步获取长度位置
        inputs = traintest[len(traintest)-x_train_day:len(traintest)].values
        inputs = inputs.reshape(-1,1)           # 转换数组
        inputs = Scaler.fit_transform(inputs)   # 归一化
        print(inputs.size)
        x_pred.append(inputs)
        x_pred = np.array(x_pred)
        x_pred = np.reshape(x_pred,(x_pred.shape[0],x_pred.shape[1],1))
        print(x_pred.shape)

        # 模型预测
        y_pred = RNN_LSTM.predict(x_pred)

        # 反归一化
        y_pred = Scaler.inverse_transform(y_pred)
        y_pred = np.ravel(y_pred)
        print("y_pred",y_pred)
        pred_s = pd.Series(y_pred)
        return  pred_s

    # def matplot_trend(self):
    #     plt.style.use('fivethirtyeight')    # 设定绘图风格
    #     self.df[self.df.columns[0]][:self.split_date_front].plot(figsize=(12,6),legend=True)
    #     self.df[self.df.columns[0]][self.split_date_after:].plot(figsize=(12,6),legend=True)
    #     plt.legend(['train拆分后分隔趋势','test拆分后分隔趋势'])
    #     plt.title(['分隔后的',self.df.columns[0],'趋势'])
    #     plt.show()


if __name__ == '__main__':
    in_path = "/Users/admin/Desktop/史蒂芬大春/办公桌面/编程/sklearn/RNN时间序列预测走势/19-23.8.22流水.csv"
    df = RnnPred(in_path=in_path)
    data = df.data_clean( split_date_after='2023-08-21', y_pred_day=7,x_train_day=60)


