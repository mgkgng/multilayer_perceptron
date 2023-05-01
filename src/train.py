import pandas as pd
from sklearn.model_selection import train_test_split
from network import Network

df = pd.read_csv('../assets/data.csv', header=None)
df = df.drop(df.columns[0], axis=1)

cols = ['diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
df.columns = cols

X, y = df.iloc[:, 1:], df.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

network = Network([30, 15, 15, 1])