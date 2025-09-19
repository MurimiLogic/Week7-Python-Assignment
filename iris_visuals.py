
from typing import Tuple
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Configure logging for simple CLI feedback
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

#Not reading from a csv, reading directly from sklearn
def load_iris_df() -> pd.DataFrame:
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    return df

# Exploratory Data Analysis
def eda(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    logger.info("First 10 rows of the Iris dataset:")
    print(df.head(10))

    logger.info("Dataset info:")
    df.info()

    logger.info("Data types:")
    print(df.dtypes)

    logger.info("Missing values: (per column)")
    print(df.isnull().sum())

    if df.isnull().sum().any():
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mean())
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
        logger.info("Filled missing values where needed.")
    else:
        logger.info("No missing values found. Dataset is clean.")

    logger.info("Descriptive statistics for numerical columns:")
    print(df.describe())

 #median and standard deviation
    print("Median values for numerical columns:")
    print(df.select_dtypes(include=[np.number]).median())
    print("Standard deviation for numerical columns:")
    print(df.select_dtypes(include=[np.number]).std())

    logger.info("Mean values by species:")
    species_means = df.groupby('species').mean()
    print(species_means)

    logger.info("Mean sepal length by species:")
    sepal_length_means = df.groupby('species')['sepal length (cm)'].mean()
    print(sepal_length_means)

    logger.info("Key findings from analysis:")
    print("1. Setosa has the smallest petal measurements but largest sepal width")
    print("2. Virginica has the largest petal length and width")
    print("3. Versicolor has intermediate measurements across most features")
    print("4. Setosa appears to be the most distinct species in terms of measurements")
    print("5. Versicolor and Virginica have some overlapping characteristics")

    return species_means, sepal_length_means


def plot_visuals(df: pd.DataFrame, species_means: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Iris Dataset Feature Analysis', fontsize=16)

    features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    species = species_means.index

    for i, feature in enumerate(features):
        ax = axes[i // 2, i % 2]
        ax.plot(species, species_means[feature], marker='o', linewidth=2, markersize=8)
        ax.set_title(f'{feature} by Species')
        ax.set_xlabel('Species')
        ax.set_ylabel(feature)
        ax.grid(True)

    plt.tight_layout()
    plt.show()

    # Bar chart
    plt.figure(figsize=(10, 6))
    species_means['petal length (cm)'].plot(kind='bar', color=['skyblue', 'lightgreen', 'lightcoral'])
    plt.title('Average Petal Length by Species', fontsize=14)
    plt.xlabel('Species', fontsize=12)
    plt.ylabel('Petal Length (cm)', fontsize=12)
    plt.xticks(rotation=0)
    plt.show()

    # Histogram use
    plt.figure(figsize=(10, 6))
    df['sepal length (cm)'].hist(bins=15, color='skyblue', edgecolor='black')
    plt.title('Distribution of Sepal Length', fontsize=14)
    plt.xlabel('Sepal Length (cm)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(False)
    plt.show()

    # Scatter plot
    plt.figure(figsize=(10, 6))
    colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
    for sp in df['species'].unique():
        subset = df[df['species'] == sp]
        plt.scatter(subset['sepal length (cm)'], subset['petal length (cm)'],
                    color=colors[sp], label=sp, alpha=0.7)

    plt.title('Sepal Length vs Petal Length', fontsize=14)
    plt.xlabel('Sepal Length (cm)', fontsize=12)
    plt.ylabel('Petal Length (cm)', fontsize=12)
    plt.legend(title='Species')
    plt.grid(True)
    plt.show()

    # Seaborn scatter
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)',
                    hue='species', style='species', s=100)
    plt.title('Sepal vs Petal Length (Enhanced with Seaborn)', fontsize=14)
    plt.xlabel('Sepal Length (cm)', fontsize=12)
    plt.ylabel('Petal Length (cm)', fontsize=12)
    plt.show()

# Load the dataset and basic checks to show plots if needed.
def main(show_plots: bool = True) -> None:
    try:
        df = load_iris_df()

        if df.empty:
            raise ValueError("Loaded dataset is empty")

        required_columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'species']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column {col} not found in dataset")

        species_means, sepal_means = eda(df)

        if show_plots:
            plot_visuals(df, species_means)

        logger.info("Dataset loaded and analyzed successfully")

    except ValueError as ve:
        logger.error("Value error: %s", ve)
    except Exception as e:
        logger.exception("Unexpected error: %s", e)


if __name__ == '__main__':

    main(show_plots=True)