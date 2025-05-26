# Sentiment Analysis Project

## Step 1: Install Required Libraries
# pip install pandas scikit-learn matplotlib seaborn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 2: Load Dataset
# Example dataset (you can replace this with a CSV load)
data = {
    "text": [
        "I love this product! It's amazing.",
        "Absolutely terrible, I hate it.",
        "Not bad, could be better.",
        "Excellent quality and great service.",
        "Worst experience ever.",
        "Pretty decent, not the best though.",
        "I will never buy this again.",
        "Highly recommend this!",
        "It's okay, I guess.",
        "Awful. Just awful."
    ],
    "label": [
        "positive",
        "negative",
        "neutral",
        "positive",
        "negative",
        "neutral",
        "negative",
        "positive",
        "neutral",
        "negative"
    ]
}

df = pd.DataFrame(data)

# Step 3: Preprocess Data
X = df['text']
y = df['label']

vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Step 4: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Step 5: Train Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 6: Make Predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate Model
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))

conf_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Step 8: Predict New Sentences
examples = [
    "I really enjoyed this!",
    "This is the worst thing ever.",
    "I'm not sure how I feel about this."
]
examples_vec = vectorizer.transform(examples)
example_preds = model.predict(examples_vec)

for text, pred in zip(examples, example_preds):
    print(f"Text: {text} => Sentiment: {pred}")

# Optional: Save the model and vectorizer using joblib
# from joblib import dump
# dump(model, 'sentiment_model.joblib')
# dump(vectorizer, 'vectorizer.joblib')
