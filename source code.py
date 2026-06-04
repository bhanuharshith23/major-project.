# train_model.py

```python
import pandas as pd
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("iphone14_reviews.csv")

# Text preprocessing
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

df["Review"] = df["Review"].apply(clean_text)

# Features and labels
X = df["Review"]
y = df["Sentiment"]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)

X_tfidf = vectorizer.fit_transform(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42
)

# SVM Model
model = SVC(kernel="linear")

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
pickle.dump(model, open("svm_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model saved successfully")
```


# predict.py

```python
import pickle
import re

# Load model and vectorizer
model = pickle.load(open("svm_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

while True:
    review = input("\nEnter iPhone 14 Review: ")

    review = clean_text(review)

    review_vector = vectorizer.transform([review])

    prediction = model.predict(review_vector)

    print("Predicted Sentiment:", prediction[0])

    choice = input("Analyze another review? (y/n): ")

    if choice.lower() != "y":
        break
```
