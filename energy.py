import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. 读取数据
data = pd.read_csv('Training data.csv')

# 2. 数据预处理
# 2.1 删除Index特征
data = data.drop('Index', axis=1)

# 2.2 处理缺失值
data = data.fillna(data.median())

# 2.3 异常值处理 - 使用IQR法处理异常值
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

print(data.shape)

# 2.4 特征编码（Label Encoding）
categorical_features = data.select_dtypes(include=['object']).columns.tolist()
label_encoder = LabelEncoder()
for col in categorical_features:
    data[col] = label_encoder.fit_transform(data[col])

print(data)

# 2.5 特征归一化
X = data.drop('ENERGY_CONSUMPTION_CURRENT', axis=1)
y = data['ENERGY_CONSUMPTION_CURRENT']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. 特征选择（置换特征重要性）
# 基准模型训练
model = lgb.LGBMRegressor(random_state=42)
model.fit(X_train, y_train)
baseline_predictions = model.predict(X_test)
baseline_mse = mean_squared_error(y_test, baseline_predictions)
print(f'Baseline MSE: {baseline_mse}')

# 置换特征并记录MSE变化
feature_importance = {}
for i in range(X_train.shape[1]):
    X_test_permuted = X_test.copy()
    np.random.shuffle(X_test_permuted[:, i])
    permuted_predictions = model.predict(X_test_permuted)
    permuted_mse = mean_squared_error(y_test, permuted_predictions)
    mse_difference = permuted_mse - baseline_mse
    feature_importance[X.columns[i]] = mse_difference

# 特征重要性排序并选择重要特征
sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
important_features = [feature for feature, importance in sorted_features if importance > 0]
print(important_features)

# 5. 使用重要特征重新训练模型
X_train_selected = X_train[:, [X.columns.get_loc(f) for f in important_features]]
X_test_selected = X_test[:, [X.columns.get_loc(f) for f in important_features]]

# 使用LightGBM重新训练
model_selected = lgb.LGBMRegressor(random_state=42)
model_selected.fit(X_train_selected, y_train)

# 模型预测
y_pred = model_selected.predict(X_test_selected)

# 6. 模型评估
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R^2 Score: {r2}')

# 7. 可视化结果（真实值 vs 预测值）
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel('Actual ENERGY_CONSUMPTION_CURRENT')
plt.ylabel('Predicted ENERGY_CONSUMPTION_CURRENT')
plt.title('Actual vs Predicted Values')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # 参考线
plt.show()

['CURRENT_ENERGY_EFFICIENCY', 'MAINHEAT_ENERGY_EFF', 'TOTAL_FLOOR_AREA', 'POSTCODE', 'WALLS_ENERGY_EFF',
 'HOT_WATER_ENERGY_EFF', 'LOW_ENERGY_LIGHTING', 'MAINHEATC_ENERGY_EFF', 'PROPERTY_TYPE', 'ROOF_DESCRIPTION', 'TENURE',
 'CONSTRUCTION_AGE_BAND', 'GLAZED_TYPE', 'BUILT_FORM', 'WINDOWS_ENERGY_EFF', 'FLOOR_DESCRIPTION', 'TRANSACTION_TYPE']
