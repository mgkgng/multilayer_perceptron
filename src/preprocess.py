import pandas as pd
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from k_fold import KFoldGenerator

def load_data(path):
    df = pd.read_csv(path, header=None)
    df = df.drop(df.columns[0], axis=1)

    cols = ['Diagnosis'] + [f'Feature_{i}' for i in range(1, 31)]
    df.columns = cols
    print('Data loading completed successfully')
    return df

def preprocess_data(df, k_fold=False):
    # Drop the features which have a p-value > 0.01 (Kolmogorov-Smirnov test)
    exam = {}
    for col in df.columns[1:]:
        res = stats.ks_2samp(df[df['Diagnosis'] == 'M'][col], df[df['Diagnosis'] == 'B'][col])
        exam[col] = res.pvalue
    to_drop = [k for k, v in exam.items() if v > 0.01]
    df = df.drop(to_drop, axis=1)

    # One-hot encoding for the diagnosis column
    one_hot = pd.get_dummies(df['Diagnosis'], prefix='Diagnosed')
    df = pd.concat([df, one_hot], axis=1)
    df.drop(['Diagnosis'], axis=1, inplace=True)

    # Normalize the data for the features which have too large values
    to_normalize = df.columns[df.max() > 1]
    normalize = lambda x: (x - x.min()) / (x.max() - x.min())
    df[to_normalize] = df[to_normalize].apply(normalize)

    X, y = df.iloc[:, :-2].values, df.iloc[:, -2:].values
    print('Data preprocessing completed successfully')
    if k_fold == True:
        return KFoldGenerator(X, y, n_splits=10, shuffle=True)
    return train_test_split(X, y, test_size=0.2, random_state=42)