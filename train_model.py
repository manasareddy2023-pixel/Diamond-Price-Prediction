import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Load the Diamonds dataset
df = sns.load_dataset("diamonds")
df.to_csv("diamonds.csv", index=False)

# Display first 5 rows
print("First 5 rows:")
print(df.head())

# Check dataset shape
print("\nDataset Shape:")
print(df.shape)

# Check column names
print("\nColumn Names:")
print(df.columns)

# Check dataset information
print("\nDataset Information:")
df.info()

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Exploratory Data Analysis (EDA)

# Statistical Summary
print("\nStatistical Summary:")
print(df.describe())

# Categorical Columns Summary
print("\nCategorical Columns Summary:")
print(df.describe(include="category"))

# Check Duplicate Rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Distribution of Target Variable
plt.figure(figsize=(8, 5))
sns.histplot(df["price"], kde=True)
plt.title("Distribution of Diamond Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

# Categorical Feature Analysis

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="cut")
plt.title("Distribution of Diamond Cut")
plt.xlabel("Cut")
plt.ylabel("Count")
plt.show()


plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="color")
plt.title("Distribution of Diamond Color")
plt.xlabel("Color")
plt.ylabel("Count")
plt.show()


plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="clarity")
plt.title("Distribution of Diamond Clarity")
plt.xlabel("Clarity")
plt.ylabel("Count")
plt.show()
print("Categorical Feature Analysis completed!")


# Numerical Feature Analysis

numerical_columns = ["carat", "depth", "table", "price", "x", "y", "z"]

for col in numerical_columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x=col, kde=True)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

print("Numerical Feature Analysis Completed")


# Outlier Detection using Boxplots

numerical_columns = ["carat", "depth", "table", "price", "x", "y", "z"]

for col in numerical_columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.xlabel(col)
    plt.show()

print("Outlier Detection Completed")

# Correlation Analysis

plt.figure(figsize=(10, 7))

correlation = df.corr(numeric_only=True)

sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")

plt.title("Correlation Heatmap")
plt.show()

print("Correlation Analysis Completed")


# Feature vs Price Analysis

# Carat vs Price
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="carat", y="price")
plt.title("Carat vs Price")
plt.xlabel("Carat")
plt.ylabel("Price")
plt.show()

# Depth vs Price
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="depth", y="price")
plt.title("Depth vs Price")
plt.xlabel("Depth")
plt.ylabel("Price")
plt.show()

# Table vs Price
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="table", y="price")
plt.title("Table vs Price")
plt.xlabel("Table")
plt.ylabel("Price")
plt.show()

print("Feature vs Price Analysis Completed")


# Categorical Features vs Price Analysis

# Cut vs Price
plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x="cut", y="price")
plt.title("Cut vs Price")
plt.xlabel("Cut")
plt.ylabel("Price")
plt.show()

# Color vs Price
plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x="color", y="price")
plt.title("Color vs Price")
plt.xlabel("Color")
plt.ylabel("Price")
plt.show()

# Clarity vs Price
plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x="clarity", y="price")
plt.title("Clarity vs Price")
plt.xlabel("Clarity")
plt.ylabel("Price")
plt.show()

print("Categorical Features vs Price Analysis Completed")

# Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Check Duplicate Records
print("\nDuplicate Records Before Removal:")
print(df.duplicated().sum())

# Remove Duplicate Records
df = df.drop_duplicates()

# Verify Duplicate Removal
print("\nDuplicate Records After Removal:")
print(df.duplicated().sum())

print("\nDataset Shape After Removing Duplicates:")
print(df.shape)

print("Duplicate Removal Completed")


# Define Input Features and Target Variable

X = df.drop("price", axis=1)

y = df["price"]

print("\nInput Features (X):")
print(X.head())

print("\nTarget Variable (y):")
print(y.head())

print("\nX Shape:", X.shape)
print("y Shape:", y.shape)



# Train-Test Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

print("\nTraining and Testing Data Shapes:")

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


# Data Preprocessing

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Separate Numerical and Categorical Columns

numerical_columns = ["carat", "depth", "table", "x", "y", "z"]

categorical_columns = ["cut", "color", "clarity"]

# Apply Preprocessing

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_columns),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
    ]
)

# Fit on Training Data and Transform

X_train_processed = preprocessor.fit_transform(X_train)

# Transform Testing Data

X_test_processed = preprocessor.transform(X_test)

print("\nPreprocessing Completed!")

print("X_train_processed Shape:", X_train_processed.shape)
print("X_test_processed Shape:", X_test_processed.shape)


# KNN Regressor Model Training

from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline

# Create KNN Pipeline

knn_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", KNeighborsRegressor(n_neighbors=5))
])

# Train the Model

knn_pipeline.fit(X_train, y_train)

print("\nKNN Model Training Completed!")
print("Algorithm: KNN Regressor")
print("Number of Neighbors: 5")


# Model Evaluation

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Predict Prices

y_pred = knn_pipeline.predict(X_test)

# Calculate Evaluation Metrics

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)

# Display Results

print("\nModel Evaluation Results:")

print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

 #Save Trained Model Using Pickle

import pickle

with open("knn_pipeline.pkl", "wb") as file:
    pickle.dump(knn_pipeline, file)

print("\nModel Saved Successfully!")