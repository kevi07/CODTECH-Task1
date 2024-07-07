import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

car_data = pd.read_csv('data.csv')
print("First few rows of the dataset:")
print(car_data.head())

print("\nStructure of the dataset:")
print(car_data.info())

print("\nSummary statistics for numerical features:")
print(car_data.describe())

print("\nMissing values in the dataset:")
print(car_data.isnull().sum())

numerical_features = car_data.select_dtypes(include=[np.number]).columns.tolist()

for feature in numerical_features:
    plt.figure(figsize=(10, 6))
    sns.histplot(car_data[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(car_data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

for feature in numerical_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=car_data[feature])
    plt.title(f'Boxplot of {feature}')
    plt.show()
