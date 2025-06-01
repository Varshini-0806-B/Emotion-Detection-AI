from datasets import load_dataset
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Load the HuggingFace Emotion dataset
dataset = load_dataset("emotion")

# Convert to pandas DataFrame
train_df = pd.DataFrame(dataset["train"])
test_df = pd.DataFrame(dataset["test"])

# Features and labels
X = train_df["text"]
y = train_df["label"]

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split data (again for validation)
X_train, X_val, y_train, y_val = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on validation set
y_pred = model.predict(X_val)

# Accuracy
print("Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred, target_names=dataset["train"].features["label"].names))
