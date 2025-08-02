# 1. IMPORTING NECESSARY LIBRARIES

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Set some visualization styles
sns.set_style('whitegrid')
plt.style.use('fivethirtyeight')

# 2. LOADING THE DATASET
try: 
    # Load the dataset from local file
    data = pd.read_csv("insurance.csv")
    print("Dataset Loaded Successfully!")
    print("First 5 rows of the dataset:")
    print(data.head())
except Exception as e:
    print(f"Error Loading dataset: {e}")
    # for fallback in case its a dummy dataset
    data = pd.DataFrame({
        'age': [19, 33, 28, 62, 46],
        'sex': ['female', 'male', 'male', 'male', 'female'],
        'bmi': [27.9, 33.7, 28.9, 25.3, 30.0],
        'children': [0, 2, 3, 0, 1],
        'smoker': ['yes', 'no', 'no', 'no', 'yes'],
        'region': ['southwest', 'southeast', 'northwest', 'northeast', 'southwest'],
        'charges': [16884.92, 1725.55, 4449.46, 21984.47, 3866.86]
    })
    print('\nLoaded dummy dataset instead')

# 3. EXPLORATORY DATA ANALYSIS (EDA)
    print("\n---Exploratory Data Analysis---")
    print("\nDataset Information:")
    data.info()
    print("\nDescriptive Statistics:")
    print(data.describe())

    print.figure(figsize=(12, 6))
    sns.histplot(data['charges'], kde=True, bins=50)
    plt.title('Distribution of Medical Charges')
    plt.xlabel('Charges')
    plt.ylabel('Frequency')
    plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Categorical Features vs Charges', fontsize=20)

    sns.boxplot(ax=axes[0, 0], x='smoker', y='charges', data=data)
    axes[0, 0].set_title('Smoker vs. Charges'
                         )
    sns.boxplot(ax=axes[0, 0], x='sex', y='charges', data=data)
    axes[0, 0].set_title('Sex vs. Charges')

    sns.boxplot(ax=axes[1, 0], x='region', y='charges', data=data)
    axes[1, 0].set_title('Region vs. Charges')

    sns.boxplot(ax=axes[1, 1], x='children', y='charges', data=data)
    axes[1, 1].set_title('Children vs. Charges')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
