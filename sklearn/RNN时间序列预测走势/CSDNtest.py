
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Read the data
df = pd.read_csv('/Users/admin/Desktop/household_power_consumption.txt',
                 parse_dates={'dt' : ['Date', 'Time']},
                 sep=";", infer_datetime_format=True,
                 low_memory=False, na_values=['nan','?'], index_col='dt')
# The first five lines of df is shown below
df.head()
# we use "dataset_train_actual" for plotting in the end.
dataset_train_actual = df.copy()
# create "dataset_train for further processing
dataset_train = df.copy()


# Select features (columns) to be involved intro training and predictions
dataset_train = dataset_train.reset_index()
cols = list(dataset_train)[1:8]
# Extract dates (will be used in visualization)
datelist_train = list(dataset_train['dt'])
datelist_train = [date for date in datelist_train]
training_set = dataset_train.values


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
training_set_scaled = sc.fit_transform(training_set)
sc_predict = StandardScaler()
sc_predict.fit_transform(training_set[:, 0:1])



# Creating a data structure with 72 timestamps and 1 output
X_train = []
y_train = []
n_future = 30 # Number of days we want to predict into the future.
n_past = 72 # Number of past days we want to use to predict future.
for i in range(n_past, len(training_set_scaled) - n_future +1):
    X_train.append(training_set_scaled[i - n_past:i,
                   0:dataset_train.shape[1]])
    y_train.append(training_set_scaled[i+n_future-1:i+n_future, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

print('X_train shape == {}.'.format(X_train.shape))
print('y_train shape == {}.'.format(y_train.shape))

import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=32, kernel_size=3,
                           strides=1, padding="causal",
                           activation="relu",
                           input_shape=[None, 7]),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=False)),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 200)])
# lr_schedule = tf.keras.callbacks.LearningRateScheduler(
# lambda epoch: 1e-8 * 10**(epoch / 20))
optimizer = tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mse"])


# Perform predictions
predictions_future = model.predict(X_train[-n_future:])
# getting predictions for training data for plotting purpose
predictions_train = model.predict(X_train[n_past:])
y_pred_future = sc_predict.inverse_transform(predictions_future)
y_pred_train = sc_predict.inverse_transform(predictions_train)
# Construct two different dataframes for plotting.
# PREDICTIONS_FUTURE = pd.DataFrame(y_pred_future, columns=['Global_active_power']).set_index(pd.Series(datelist_future))
# PREDICTION_TRAIN = pd.DataFrame(y_pred_train, columns=['Global_active_power']).set_index(pd.Series(datelist_train[2 * n_past + n_future -1:]))


# # Set plot size
# plt.rcParams['figure.figsize'] = 14, 5
# # Plot parameters
# START_DATE_FOR_PLOTTING = '2009-06-07'
# # plot the target column in PREDICTIONS_FUTURE dataframe
# plt.plot(PREDICTIONS_FUTURE.index, PREDICTIONS_FUTURE['Global_active_power'], color='r', label='Predicted Global Active power')
# # plot the target column in PREDICTIONS_TRAIN dataframe
# plt.plot(PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:].index, PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING['Global_active_power'], color='orange', label='Training predictions')
# # plot the target column in input dataframe
# plt.plot(dataset_train_actual.loc[START_DATE_FOR_PLOTTING:].index, dataset_train_actual.loc[START_DATE_FOR_PLOTTING:]['Global_active_power'], color='b', label='Actual Global Active power')
# plt.axvline(x = min(PREDICTIONS_FUTURE.index), color='green', linewidth=2, linestyle='--')
# plt.grid(which='major', color='#cccccc', alpha=0.5)
# plt.legend(shadow=True)
# plt.title('Predcitions and Acutal Global Active power values', family='Arial', fontsize=12)
# plt.xlabel('Timeline', family='Arial', fontsize=10)
# plt.ylabel('Stock Price Value', family='Arial', fontsize=10)
