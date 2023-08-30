import numpy as np
import pandas as pd
from keras import Sequential
from keras.src.layers import Dense
from matplotlib import pyplot as plt
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


#logic二元分类
def logi_regression( x_train, y_train, x_test, y_test, x_pred, x_pred_first):
    print("x_train", x_train.shape)
    print("y_train", y_train.shape)
    print("x_test", x_test.shape)
    print("y_test", y_test.shape)
    print("x_pred", x_pred.shape)

    lgr = LogisticRegression()
    lgr.fit(x_train, y_train)
    y_test_pred = lgr.predict(x_test)
    print("logi的x_test拟合效果", lgr.score(x_test, y_test))
    print("logi的x_test拟合效果_f1_score", f1_score(y_test, y_test_pred))

    ls = x_train.columns.tolist()
    df_coef = pd.DataFrame({"特征":ls,"相关权重":lgr.coef_[0]})
    df_coef.sort_values("相关权重",inplace=True)
    print(df_coef.value_counts())

    y_pred = lgr.predict(x_pred)
    x_pred_first["预测值"] = y_pred  # 训练模型
    return x_pred_first


#DNN归一化神经网络分类
def dnn_minmax( x_train, y_train, x_test, y_test, x_pred, x_pred_first):
    print("x_train", x_train.shape)
    print("y_train", y_train.shape)
    print("x_test", x_test.shape)
    print("y_test", y_test.shape)
    print("x_pred", x_pred.shape)
    print("DNN归一化神经网络分类")
    dnn = Sequential()
    dnn.add(Dense(units=100, input_dim=49, activation='relu'))  # 添加输入层
    dnn.add(Dense(units=70, activation='relu'))  # 添加隐藏层
    dnn.add(Dense(units=50, activation='relu'))  # 添加隐藏层
    dnn.add(Dense(units=1, activation='sigmoid'))  # 添加输出层

    dnn.compile(optimizer='RMSProp',  # 指定优化器
                loss='binary_crossentropy',  # 损失函数   binary_crossentropy
                metrics=["accuracy"])  # 评估标准

    # 把Dataframe转换为numpy张量，训练神经网络模型
    x_train = np.asarray(x_train).astype(np.float32)
    x_test = np.asarray(x_test).astype(np.float32)

    # 创建归一化缩放器
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    history = dnn.fit(x_train, y_train,
                       epochs=30,  # 指定训练轮次
                       batch_size=64,  # 指定数据批量
                       validation_split=0.2)  # 直接从训练集中拆分验证集
    # print("dnn的x_test拟合效果_score", dnn.score(x_test, y_test))  #DNN没有score

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


    result = dnn.evaluate(x_test, y_test)  # 评估测试集上的准确率
    print("dnn的x_test拟合效果_evaluate",result)

    y_pred = dnn.predict(x_pred)
    y_pred = [x for i in y_pred for x in i]
    x_pred_first["预测值"] = y_pred
    return x_pred_first

# AdaBoost 算法（Boosting降低偏差法）
def adaboost_gird(x_train, y_train, x_test, y_test, x_pred, x_pred_first):
    print("AdaBoost 算法（Boosting降低偏差法）")
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    x_pred = scaler.transform(x_pred)
    dt = DecisionTreeClassifier()   #选择决策树分类器作为AdaBoost的基准算法
    ada = AdaBoostClassifier(dt) # AdaBoost 模型

    param = {
        "n_estimators": [40,50,60],  # 弱分类器的数量，默认为 50。指定要构建的弱分类器数量。
        "learning_rate": [0.8,1,1.2],  # 默认为 1.0。控制每个弱分类器对最终预测的贡献程度。较小的学习率可以使模型更加保守。
        "algorithm": ["SAMME","SAMME.R"], } # 默认为3。控制每棵树的生长程度，避免过拟合。较小的值通常可以提升模型的鲁棒性，但可能会降低模型的预测性能。
    odel_ada_gird = GridSearchCV(ada, param_grid=param, cv=5,
                                scoring='r2', n_jobs=10, verbose=1, error_score=0)  # n_jobs=10搜索过程中并行的运算数量

    odel_ada_gird.fit(x_train,y_train)
    y_test_pred = odel_ada_gird.predict(x_test)
    y_pred = odel_ada_gird.predict(x_pred)
    print("adaboost的x_test拟合效果_score", odel_ada_gird.score(x_test, y_test))
    print("adaboost的x_test拟合效果_f1_score", f1_score(y_test, y_test_pred))

    x_pred_first["预测值"] = y_pred
    return x_pred_first


# AdaBoost 算法（Boosting降低偏差法）
def adaboost(x_train, y_train, x_test, y_test, x_pred, x_pred_first):
    print("AdaBoost 算法（Boosting降低偏差法）")
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    x_pred = scaler.transform(x_pred)
    dt = DecisionTreeClassifier()   #选择决策树分类器作为AdaBoost的基准算法
    ada = AdaBoostClassifier(dt) # AdaBoost 模型
    ada.fit(x_train,y_train)

    y_test_pred = ada.predict(x_test)

    y_pred = ada.predict(x_pred)
    print("adaboost的x_test拟合效果_score", ada.score(x_test, y_test))
    print("adaboost的x_test拟合效果_f1_score", f1_score(y_test, y_test_pred))
    x_pred_first["预测值"] = y_pred
    return x_pred_first


# Granding Boosting 梯度提升决策树算法
def gbdt_gird(x_train, y_train, x_test, y_test, x_pred, x_pred_first):
    print("Granding Boosting 梯度提升决策树算法")
    gb = GradientBoostingClassifier() # 梯度提升模型

    param = {
        "n_estimators": [80,100,120],  # 森林中树的数量，默认是100
        "learning_rate": [0.1,0.08],  # 默认为 0.1。学习率控制每棵树的贡献程度。较小的值会使模型更加保守，但可能需要更多的树来达到较高的性能。
        "max_depth": [2, 3, 4],  # 默认为3。控制每棵树的生长程度，避免过拟合。较小的值通常可以提升模型的鲁棒性，但可能会降低模型的预测性能。
        "min_samples_split": [1,2,3],  # 默认为 2。如果一个内部节点的样本数少于该值，则不会再分裂
        "min_samples_leaf": [1, 2],  # 默认为 1。如果一个叶节点的样本数少于该值，则不会继续进行分裂。
        "subsample": [0.8,1],   # 默认为 1。控制每棵树所使用的训练样本的比例。较小的值可以防止过拟合。
        "loss": ["deviance", "exponential"], }  # 损失函数，默认为 'deviance'表示使用对数似然损失函数（分类）
    model_gb_gird = GridSearchCV(gb, param_grid=param, cv=5,
                                  scoring='r2', n_jobs=10, verbose=1, error_score=0)  # n_jobs=10搜索过程中并行的运算数量
    model_gb_gird.fit(x_train,y_train)
    y_test_pred = model_gb_gird.predict(x_test)
    y_pred = model_gb_gird.predict(x_pred)
    print("adaboost的x_test拟合效果_score", model_gb_gird.score(x_test, y_test))
    print("adaboost的x_test拟合效果_f1_score", f1_score(y_test, y_test_pred))

    x_pred_first["预测值"] = y_pred
    return x_pred_first

# Granding Boosting 梯度提升决策树算法
def gbdt(x_train, y_train, x_test, y_test, x_pred, x_pred_first):
    print("Granding Boosting 梯度提升决策树算法")
    gb = GradientBoostingClassifier() # 梯度提升模型
    gb.fit(x_train,y_train)
    y_test_pred = gb.predict(x_test)
    y_pred = gb.predict(x_pred)
    print("adaboost的x_test拟合效果_score", gb.score(x_test, y_test))
    print("adaboost的x_test拟合效果_f1_score", f1_score(y_test, y_test_pred))
    x_pred_first["预测值"] = y_pred
    return x_pred_first


# 集成学习XGBoost算法
def xgboost_xgb_gird( x_train, y_train, x_test, y_test, x_pred, x_pred_first):
    print("集成学习XGBoost算法")
    xgb = XGBClassifier()    #XGB模型

    param = {
        "n_estimators": [80,100,120],  # 森林中树的数量，默认是100
        "learning_rate": [0.03,0.02],  # 学习率控制每棵树的贡献程度。较小的值会使模型更加保守，但可能需要更多的树来达到较高的性能。
        "max_depth": [5, 7, 9],  # 控制每棵树的生长程度，避免过拟合。较小的值通常可以提升模型的鲁棒性，但可能会降低模型的预测性能。
        "subsample": [0.8,1],  # 随机采样训练样本的比例，默认为 1。控制每棵树所使用的训练样本的比例。较小的值可以防止过拟合。
        "colsample_bytree": [0.8, 1],  # 控制每棵树所使用的特征的比例。较小的值可以增加模型的多样性。
        "reg_alpha": [0,1],   # L1 正则化的参数，默认为 0。通过增加L1正则化惩罚项来控制模型的复杂度。
        "reg_lambda": [0, 1], }  #  L2 正则化的参数，默认为 1。通过增加L2正则化惩罚项来控制模型的复杂度
    model_xgb_gird = GridSearchCV(xgb, param_grid=param, cv=5,
                                 scoring='r2', n_jobs=10, verbose=1, error_score=0)  # n_jobs=10搜索过程中并行的运算数量
    model_xgb_gird.fit(x_train,y_train)
    y_test_pred = model_xgb_gird.predict(x_test)
    y_pred = model_xgb_gird.predict(x_pred)
    print("xgboost的x_test拟合效果_score", model_xgb_gird.score(x_test, y_test))
    print("xgboost的x_test拟合效果_f1_score", f1_score(y_test, y_test_pred))
    x_pred_first["预测值"] = y_pred
    return x_pred_first

# 集成学习XGBoost算法
def xgboost_xgb( x_train, y_train, x_test, y_test, x_pred, x_pred_first):
    print("集成学习XGBoost算法")
    xgb = XGBClassifier()    #XGB模型
    xgb.fit(x_train,y_train)
    y_test_pred = xgb.predict(x_test)
    y_pred = xgb.predict(x_pred)
    print("xgboost的x_test拟合效果_score", xgb.score(x_test, y_test))
    print("xgboost的x_test拟合效果_f1_score", f1_score(y_test, y_test_pred))

    x_pred_first["预测值"] = y_pred
    return x_pred_first


# 决策树Bagging算法
def bagging_tree_gird(x_train, y_train, x_test, y_test, x_pred, x_pred_first):
    print("决策树Bagging算法")
    bt = BaggingClassifier(DecisionTreeClassifier())

    param = {
        "n_estimators": [10, 15, 20],  # 森林中树的数量，默认是100
        "criterion": ["gini", "entropy"],  # 树的分裂准则
        "max_depth": [5, 7, 9],  # 决策数子节点深度
        "min_samples_split": [2, 3, 4],  # 数内部节点划分的最小样本数
        "min_samples_leaf": [2, 3],  # 叶子节点的最小样本数
        "bootstrap": [True], }  # 是否抽取放回，默认True
    model_bt_gird = GridSearchCV(bt, param_grid=param, cv=5,
                                  scoring='r2', n_jobs=10, verbose=1, error_score=0)  # n_jobs=10搜索过程中并行的运算数量
    model_bt_gird.fit(x_train,y_train)
    y_test_pred = model_bt_gird.predict(x_test)
    y_pred = model_bt_gird.predict(x_pred)
    print("Bagging_tree的x_test拟合效果_score", model_bt_gird.score(x_test, y_test))
    print("Bagging_tree的x_test拟合效果_f1_score", f1_score(y_test, y_test_pred))
    x_pred_first["预测值"] = y_pred
    return x_pred_first

# 决策树Bagging算法
def bagging_tree(x_train, y_train, x_test, y_test, x_pred, x_pred_first):
    print("决策树Bagging算法")
    bt = BaggingClassifier(DecisionTreeClassifier())
    bt.fit(x_train,y_train)
    y_test_pred = bt.predict(x_test)
    y_pred = bt.predict(x_pred)
    print("Bagging_tree的x_test拟合效果_score", bt.score(x_test, y_test))
    print("Bagging_tree的x_test拟合效果_f1_score", f1_score(y_test, y_test_pred))
    x_pred_first["预测值"] = y_pred
    return x_pred_first


# 极端随机森林算法
def extrees_gird(x_train, y_train, x_test, y_test, x_pred, x_pred_first):
    print("网格化调参：极端随机森林算法")
    ext = ExtraTreesClassifier()

    #网格化调参
    model_rfr_param = {
        "n_estimators": [30, 50, 100],  # 森林中树的数量，默认是100
        "criterion": ["gini","entropy"],  # 树的分裂准则
        "max_depth": [ 5, 7, 9],  # 决策数子节点深度
        "min_samples_split": [2, 3, 4],  # 数内部节点划分的最小样本数
        "min_samples_leaf": [ 2, 3],  # 叶子节点的最小样本数
        "bootstrap": [True],}  # 是否抽取放回，默认True
    model_ext_gird = GridSearchCV(ext, param_grid=model_rfr_param, cv=5,
                                  scoring='r2', n_jobs=10, verbose=1, error_score=0)  # n_jobs=10搜索过程中并行的运算数量
    model_ext_gird.fit(x_train,y_train)
    y_test_pred = model_ext_gird.predict(x_test)
    y_pred = model_ext_gird.predict(x_pred)
    print("extrees的x_test拟合效果_score", model_ext_gird.score(x_test, y_test))
    print("extrees的x_test拟合效果_f1_score", f1_score(y_test, y_test_pred))
    x_pred_first["预测值"] = y_pred
    return x_pred_first

# 极端随机森林算法
def extrees(x_train, y_train, x_test, y_test, x_pred, x_pred_first):
    print("极端随机森林算法")
    ext = ExtraTreesClassifier()
    ext.fit(x_train,y_train)
    y_test_pred = ext.predict(x_test)
    y_pred = ext.predict(x_pred)
    print("extrees的x_test拟合效果_score", ext.score(x_test, y_test))
    print("extrees的x_test拟合效果_f1_score", f1_score(y_test, y_test_pred))
    x_pred_first["预测值"] = y_pred
    return x_pred_first


# 随机森林算法
def forests_gird(x_train, y_train, x_test, y_test, x_pred, x_pred_first):
    print("网格化调参：随机森林算法")
    rf = RandomForestClassifier()

    #网格化调参
    model_rfr_param = {
        "n_estimators": [30, 50, 100],  # 森林中树的数量，默认是100
        "criterion": ["gini","entropy"],  # 树的分裂准则
        "max_depth": [ 5, 7, 9],  # 决策数子节点深度
        "min_samples_split": [2, 3, 4],  # 数内部节点划分的最小样本数
        "min_samples_leaf": [ 2, 3],  # 叶子节点的最小样本数
        "bootstrap": [ True],}  # 是否抽取放回，默认True
    model_rfr_gird = GridSearchCV(rf, param_grid=model_rfr_param, cv=5,
                                  scoring='r2', n_jobs=10, verbose=1, error_score=0)  # n_jobs=10搜索过程中并行的运算数量

    model_rfr_gird.fit(x_train,y_train)
    y_test_pred = model_rfr_gird.predict(x_test)
    y_pred = model_rfr_gird.predict(x_pred)
    print("extrees的x_test拟合效果_score", model_rfr_gird.score(x_test, y_test))
    print("extrees的x_test拟合效果_f1_score", f1_score(y_test, y_test_pred))
    x_pred_first["预测值"] = y_pred
    return x_pred_first


# 随机森林算法
def forests(x_train, y_train, x_test, y_test, x_pred, x_pred_first):
    print("随机森林算法")
    rf = RandomForestClassifier()

    rf.fit(x_train, y_train)
    y_test_pred = rf.predict(x_test)
    y_pred = rf.predict(x_pred)
    print("extrees的x_test拟合效果_score", rf.score(x_test, y_test))
    print("extrees的x_test拟合效果_f1_score", f1_score(y_test, y_test_pred))
    x_pred_first["预测值"] = y_pred
    return x_pred_first
