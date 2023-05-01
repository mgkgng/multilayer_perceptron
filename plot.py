import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('../assets/data.csv')
df = df.drop(df.columns[0],axis=1)
sns.pairplot(df, hue=df.columns[0])
X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
print(X)
print(y)