import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv("emails.csv")

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['label']

model = MultinomialNB()
model.fit(X, y)

text = input("Enter email: ")
vec = vectorizer.transform([text])
print("Prediction:", "Phishing" if model.predict(vec)[0] else "Safe")
