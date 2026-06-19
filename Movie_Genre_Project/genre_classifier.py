import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("Loading dataset...")

# Dataset Path
current_folder = os.path.dirname(os.path.abspath(__file__))
train_file = os.path.join(current_folder, "train_data.txt")

# Load Dataset
data = []

with open(train_file, "r", encoding="utf-8") as file:
    for line in file:
        parts = line.strip().split(" ::: ")

        if len(parts) >= 4:
            genre = parts[2]
            description = " ::: ".join(parts[3:])

            data.append([genre, description])

# Create DataFrame
df = pd.DataFrame(data, columns=["Genre", "Description"])

print("Total Movies:", len(df))

# Features and Labels
X = df["Description"]
y = df["Genre"]

print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Converting text into TF-IDF features...")

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("Training Logistic Regression model...")

model = LogisticRegression(max_iter=1000)

model.fit(X_train_tfidf, y_train)

print("Making predictions...")

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL RESULTS ==========")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\n========== SAMPLE PREDICTIONS ==========")

for i in range(5):
    print(f"\nMovie {i+1}")
    print("Actual Genre    :", y_test.iloc[i])
    print("Predicted Genre :", y_pred[i])

print("\nProject Completed Successfully!")