# Customer Churn Prediction

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv("Customer_Churn_Prediction/dataset/Churn_Modelling.csv")

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# Dataset information
print("\nDataset Information:")
print(df.info())

# Dataset shape
print("\nDataset Shape:")
print(df.shape)

# Column names
print("\nColumns:")
print(df.columns)

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Remove unnecessary columns
df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

# Convert categorical columns into numerical values
encoder = LabelEncoder()
df["Gender"] = encoder.fit_transform(df["Gender"])
df["Geography"] = encoder.fit_transform(df["Geography"])

# Features and Target
X = df.drop("Exited", axis=1)
y = df["Exited"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

print("\nModel trained successfully!")

# Make predictions
prediction = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, prediction)
print("\nAccuracy:", accuracy)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, prediction))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, prediction))