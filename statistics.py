# -*- encoding: utf8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.preprocessing import LabelEncoder

def main():
    # Load the dataset
    dataset = load_wine()
    x = dataset.data
    y = dataset.target
    feature_names = dataset.feature_names

    feature_idxs = []
    print('Feature name')
    for idx, feature_name in enumerate(feature_names):
        feature_idxs.append(idx)
        print('{}: {}'.format(idx, feature_name))

    # Convert the data to a DataFrame
    df = pd.DataFrame(data=x, columns=feature_idxs)

    # Apply label encoding to the categorical feature
    # le = LabelEncoder()
    # df['categorical_feature_name'] = le.fit_transform(df['categorical_feature_name'])

    # Display basic statistics
    print('Basic Statistics of the Dataset:')
    print(df.describe())
    print()

    # Display the count of samples per class
    labels = pd.Series(y, name='target')
    print('\nCount of Samples per Class:')
    print(labels.value_counts())
    print()

    # Calculate the correlation matrix
    correlation_matrix = df.corr()

    # Display the correlation matrix
    print('Correlation Matrix:')
    print(correlation_matrix)

    # Create and display the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    plt.show()

if __name__ == '__main__':
    main()
