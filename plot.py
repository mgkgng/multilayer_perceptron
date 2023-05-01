import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('../assets/data.csv', header=None)
cols = ['diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
df = df.drop(df.columns[0], axis=1)
df.columns = cols
print(df.head())

feature_cols = df.columns[1:]

fig, axes = plt.subplots(5, 6, figsize=(15, 9))

# Create a histogram of each feature and color-code based on M/B column
for i, col in enumerate(feature_cols):
    ax = axes[i // 6][i % 6]
    sns.histplot(data=df, x=col, hue=df.columns[0], ax=ax, element="poly", stat="density", common_norm=False)
    ax.set(xlabel=None, ylabel=None)
    ax.set_title(f'Feature {i + 1}')

plt.subplots_adjust(left=0.1, bottom=0.05, right=0.9, top=0.95)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

plt.show()

# X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
# print(X)
# print(y)