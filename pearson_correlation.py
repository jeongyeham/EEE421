# this code include two parts:
# 1.the pearson correlation calculation.
# 2.the normal and simple linear regression and prediction
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load
df = pd.read_csv('./Training data.csv')

# selection
features = df.drop('ENERGY_CONSUMPTION_CURRENT', axis=1)
target = df['ENERGY_CONSUMPTION_CURRENT']

# calculate correlation
correlation = features.corrwith(target)
correlation_sorted = correlation.sort_values(ascending=False)
print(correlation_sorted)

# painting
correlation_sorted.plot(kind='bar')
plt.title('Feature Correlation with Target Variable')
plt.xlabel('Features')
plt.ylabel('Correlation Coefficient')
plt.xticks(rotation=45, fontsize=7, ha='right')  # 将X轴标签旋转45度
plt.tight_layout()  # 自动调整布局，避免标签重叠
plt.show()