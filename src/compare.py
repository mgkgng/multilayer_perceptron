from preprocess import load_data, preprocess_data
from network import Network
from predict import predict
from train import train
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    train_df = load_data('../assets/data.csv')
    X, y = preprocess_data(train_df, predict=True)
    y = np.argmax(y, axis=1)

    train(train_df, 'Default', compare=True)
    a = predict(X, y, compare=True)
    print('Default mode done')

    train(train_df, 'SGD', compare=True)
    b = predict(X, y, compare=True)
    print('SGD mode done')

    train(train_df, 'NAG', compare=True)
    c = predict(X, y, compare=True)
    print('NAG mode done')

    train(train_df, 'RMSProp', compare=True)
    d = predict(X, y, compare=True)
    print('RMSProp mode done')

    train(train_df, 'Adam', compare=True)
    e = predict(X, y, compare=True)
    print('Adam mode done')

    train(train_df, 'Adam_KFold', compare=True)
    f = predict(X, y, compare=True)
    print('Adam_KFold mode done')

    # create dataframe
    df = pd.DataFrame([a, b, c, d, e, f], columns=['Accuracy', 'Precision', 'Recall', 'F1 Score'], index=['Default', 'SGD', 'NAG', 'RMSProp', 'Adam', 'Adam_KFold'])

    # visualize
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    cols = df.columns
    for i, col in enumerate(cols):
        ax[i // 2, i % 2].bar(df.index, df[col])
        ax[i // 2, i % 2].set_title(col)

    plt.tight_layout()
    plt.show()

    plt.savefig('../assets/compare.png')