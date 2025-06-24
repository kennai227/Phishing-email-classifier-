# Phishing Email Classifier using NLP and Logistic Regression
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

df = pd.read_csv("emails.csv")  # expects 'text' and 'label' columns
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['text'])
y = df['label']

model = LogisticRegression()
model.fit(X, y)
y_pred = model.predict(X)

print(classification_report(y, y_pred))