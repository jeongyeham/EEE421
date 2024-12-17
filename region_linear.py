import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt


def region_model_save(model_path, model):
    joblib.dump(model, model_path)

# 加载数据
data = pd.read_csv('./Training data.csv')
feature_columns = ['FLOOR_LEVEL', 'FLOOR_ENERGY_EFF', 'GLAZED_TYPE',
                   'WALLS_ENERGY_EFF', 'ROOF_ENERGY_EFF', 'MAINHEAT_ENERGY_EFF',
                   'MAINHEATC_ENERGY_EFF', 'LIGHTING_ENERGY_EFF']
target_column = 'ENERGY_CONSUMPTION_CURRENT'
X = data[feature_columns]
y = data[target_column]

#处理缺失值
X = X.dropna()
y = y.dropna()

#  训练集测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Bins用来划分数据集的域
# 随机森林和K-means不好时，所以只能依靠手动分域，但是没想到效果还可以
bins = np.percentile(y_train, [25, 50, 70, 85, 95, 100])
# 初始化
models = []
regions = []
linear_expressions = []
# 进行区域线性化（）
for i in range(len(bins)):

    if i == 0:
        region_mask = y_train <= bins[i]
    else:
        region_mask = (y_train > bins[i - 1]) & (y_train <= bins[i])

    # 分配数据
    X_train_region = X_train[region_mask]
    y_train_region = y_train[region_mask]

    # 每个区域进行线性化
    model = LinearRegression()
    model.fit(X_train_region, y_train_region)
    models.append(model)
    regions.append((bins[i - 1] if i > 0 else None, bins[i]))

    # 保存每部分的线性模型
    coefficients = model.coef_
    intercept = model.intercept_
    linear_expr = f"y = {intercept:.2f}"
    for j, coef in enumerate(coefficients, start=1):
        linear_expr += f" + {coef:.2f}*x{j}"
    linear_expressions.append(linear_expr)

# 验证
y_pred_test = np.zeros_like(y_test)
for i, model in enumerate(models):
    lower_bound = regions[i][0] if regions[i][0] is not None else -np.inf
    upper_bound = regions[i][1] if i < len(regions) - 1 else np.inf
    region_mask = (y_test >= lower_bound) & (y_test < upper_bound)
    y_pred_test[region_mask] = model.predict(X_test[region_mask])

# R^2
r_squared = r2_score(y_test, y_pred_test)

# 图表
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, color='blue', label='Predicted vs Actual')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', lw=2, label='Ideal Fit')
plt.xlabel('Actual Energy Consumption')
plt.ylabel('Predicted Energy Consumption')
plt.title('Fit of the Regional Linear Models')
plt.legend()
plt.show()
region_model_save('./model.joblib', model)

# 输出
print(f"R^2: {r_squared}")
for i, expr in enumerate(linear_expressions, start=1):
    print(f"Region {i}: {expr}")


def region_model_load(model_path):
    return joblib.load(model_path)

def region_model_predict(feature_names, model):
    return model.predict(feature_names)