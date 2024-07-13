import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam

# 加载数据
# 假设你的数据在一个CSV文件中，其中日期时间为索引，有'status'，'high'，'low'，'current'列
data = pd.read_csv('intraday_data.csv', index_col='DateTime', parse_dates=True)

# 将数据按照每10分钟进行采样，使用平均值（或中位数等）
data_resampled = data.resample('10T').mean()

# 数据预处理
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_resampled)

# 定义时间窗口和预测步长
time_window = 3 * 7 * 24 * 6  # 三周的10分钟间隔数
prediction_step = 7 * 24 * 6   # 一周的10分钟间隔数

# 构建数据集
def create_dataset(dataset, look_back=time_window):
    X, Y = [], []
    for i in range(len(dataset) - look_back - prediction_step):
        X.append(dataset[i:(i + look_back)])
        Y.append(np.max(dataset[(i + look_back):(i + look_back + prediction_step), 3]))  # 预测未来一周'current'的最高点
    return np.array(X), np.array(Y)

X, y = create_dataset(scaled_data)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 重塑输入数据以适应LSTM网络
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 4))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 4))

# 构建LSTM模型
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer=Adam())

# 训练模型
model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=2)

# 预测未来一周的最高'current'值
y_pred = model.predict(X_test)

# 反向转换预测结果
y_pred_unscaled = scaler.inverse_transform(np.hstack((np.zeros((len(y_pred), 3)), y_pred.reshape(-1, 1))))[:, -1]

# 打印预测结果
print(f"预测未来一周中的最高'current'值: {y_pred_unscaled}")

# 可视化结果
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual')
plt.plot(y_pred_unscaled, label='Predicted')
plt.legend()
plt.show()
