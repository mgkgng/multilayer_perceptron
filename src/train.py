import pandas as pd
from sklearn.model_selection import train_test_split
from network import Network
from preprocess import onehot_encode

df = pd.read_csv('../assets/data.csv', header=None)
df = df.drop(df.columns[0], axis=1)

cols = ['Diagnosis'] + [f'Feature_{i}' for i in range(1, 31)]
df.columns = cols

# One-hot encoding for the diagnosis column
one_hot = pd.get_dummies(df['Diagnosis'], prefix='Diagnosed')
df = pd.concat([df, one_hot], axis=1)
df.drop(['Diagnosis'], axis=1, inplace=True)

X, y = df.iloc[:, :-2], df.iloc[:, -2:]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

network = Network([30, 15, 15, 2])