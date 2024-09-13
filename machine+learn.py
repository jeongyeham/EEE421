import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 加载数据集
df = pd.read_csv('./Training data.csv')

# 选择特征和目标变量
features = df.drop('ENERGY_CONSUMPTION_CURRENT', axis=1)
target = df['ENERGY_CONSUMPTION_CURRENT']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# 创建模型
model = LinearRegression()

# 训练模型
model.fit(X_train, y_train)

# 预测测试集
predictions = model.predict(X_test)

# 计算偏差值
errors = y_test - predictions

# 计算均方根误差
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f'Root Mean Squared Error: {rmse}')

# 绘制回归图像
plt.scatter(y_test, predictions)
plt.plot(y_test, y_test, 'r--')  # 最佳拟合线
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Regression Plot')
plt.show()

# 绘制偏差值的直方图
plt.hist(errors, bins=20)
plt.xlabel('Prediction Error')
plt.ylabel('Count')
plt.title('Histogram of Prediction Errors')
plt.show()


# 加载新数据进行预测
new_data = pd.read_csv('./Training data.csv')
new_data.reset_index(drop=True, inplace=True)
# 确保新数据具有与训练数据相同的特征
# 如果有预处理步骤（如特征缩放），确保在这里也应用相同的步骤

# 使用模型进行预测
new_predictions = model.predict(new_data)

# 将预测结果添加到 new_data DataFrame 中
new_data['Predicted_ENERGY_CONSUMPTION_CURRENT'] = new_predictions

# 显示预测结果
print(new_data[['Predicted_ENERGY_CONSUMPTION_CURRENT']])

# 可选：将预测结果保存到 CSV 文件
new_data.to_csv('C:/Users/25436/Desktop/EEE421/project/data/predictions1.csv', index=False)
