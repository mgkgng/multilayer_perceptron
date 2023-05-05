import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def histogram(df):
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

    plt.savefig('../assets/histogram.png')

def box_plot(df):
    feature_cols = df.columns[1:]
    fig, axes = plt.subplots(5, 6, figsize=(15, 9))

    # Create a histogram of each feature and color-code based on M/B column
    for i, col in enumerate(feature_cols):
        ax = axes[i // 6][i % 6]
        sns.boxplot(x='Diagnosis', y=col, hue='Diagnosis',data=df, ax=ax)
        ax.set(xlabel=None, ylabel=None)
        ax.set_title(f'Feature {i + 1}')

    plt.subplots_adjust(left=0.1, bottom=0.05, right=0.9, top=0.95)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    plt.savefig('../assets/box_plot.png')

def heatmap(df):
    fig, ax = plt.subplots(figsize=(15, 15))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(df.corr(), annot=True, cmap=cmap, ax=ax, linewidths=.5, fmt='.2f')
    plt.savefig('../assets/heatmap.png')

if __name__ == '__main__':
    df = pd.read_csv('../assets/data.csv', header=None)
    cols = ['Diagnosis'] + [f'Feature_{i}' for i in range(1, 31)]
    df = df.drop(df.columns[0], axis=1)
    df.columns = cols

    histogram(df)
    box_plot(df)
    heatmap(df)