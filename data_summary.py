# -*- encoding: utf8 -*-

import numpy as np
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

    # Add 'target' to feature
    feature_names = list(feature_names) + ['target label']
    x = np.hstack((x, y.reshape(-1, 1)))

    feature_idxs = []
    print('\nFeature name')
    for idx, feature_name in enumerate(feature_names):
        feature_idxs.append(idx)
        print('{}: {}'.format(idx, feature_name))

    # Convert the data to a DataFrame
    df = pd.DataFrame(data=x, columns=feature_idxs)

    # Check for missing values
    print('\nMissing Values in the Dataset:')
    print(df.isnull().sum())

    '''
    # Apply label encoding to the categorical feature
    CATEGORICAL_COLUMNS = [0]
    label_encoders = {}
    for column in CATEGORICAL_COLUMNS:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le
    '''

    '''
    # Fill missing values
    COLUMNS_TO_FILL = [1]
    for column in COLUMNS_TO_FILL:
        # Fill missing values in a specific column with the mean
        df[column].fillna(df[column].mean(), inplace=True)
        # Fill missing values in a specific column with the median
        df[column].fillna(df[column].mean(), inplace=True)
        # Fill missing values in a specific column with the mode
        df[column].fillna(df[column].mode(), inplace=True)
    '''

    # Display the first 10 rows of the dataset
    print('\nDisplaying the first 10 rows of the dataset:')
    print(df.head(10))

    # Display basic statistics
    print('\nBasic Statistics of the Dataset:')
    print(df.describe(include='all'))

    # Display the count of samples per class
    labels = pd.Series(y, name='target')
    print('\nCount of Samples per Class:')
    print(labels.value_counts())

    # Calculate the correlation matrix
    correlation_matrix = df.corr()

    # Display the correlation matrix
    print('\nCorrelation Matrix:')
    print(correlation_matrix)

    # Create and display the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    plt.show()

    # Sort features by correlation with a specified idx
    TARGET_IDX = 0
    print(f'\nCorrelations with feature at index {TARGET_IDX} ({feature_names[TARGET_IDX]}):')
    sorted_correlations = correlation_matrix.iloc[:, TARGET_IDX].sort_values(ascending=False)
    for i, value in sorted_correlations.items():
        print(f'{i}: {feature_names[i]}, correlation: {value:.2f}')

if __name__ == '__main__':
    main()
