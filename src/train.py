import pandas as pd
from sklearn.model_selection import train_test_split
from network import Network

df = pd.read_csv('../assets/data.csv', header=None)
df = df.drop(df.columns[0], axis=1)

cols = ['Diagnosis'] + [f'Feature_{i}' for i in range(1, 31)]
df.columns = cols

# One-hot encoding for the diagnosis column
one_hot = pd.get_dummies(df['Diagnosis'], prefix='Diagnosed')
df = pd.concat([df, one_hot], axis=1)
df.drop(['Diagnosis'], axis=1, inplace=True)

X, y = df.iloc[:, :-2].values, df.iloc[:, -2:].values
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

network = Network([30, 16, 16, 2])
# network.SGD(X_train, y_train, X_val, y_val, epochs=1000, lr=0.03, batch_size=16)
network.NAG(X_train, y_train, X_val, y_val, epochs=1000, lr=0.03, batch_size=32, mu=0.4, early_stopping=True)