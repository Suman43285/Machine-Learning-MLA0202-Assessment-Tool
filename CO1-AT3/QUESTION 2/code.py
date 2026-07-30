import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
data = {
    "label": [
        "ham", "ham", "ham", "ham", "ham",
        "spam", "spam", "spam", "spam", "spam"
    ],
    "message": [
        "How are you?",
        "Let's meet tomorrow.",
        "Are you coming to class?",
        "Call me when you reach home.",
        "Can we have lunch today?",
        "Congratulations! You won a free lottery.",
        "Claim your cash prize now.",
        "Win a free mobile phone.",
        "Exclusive offer! Click here.",
        "You have won a $1000 gift card."
    ]
}
df = pd.DataFrame(data)
X = df["message"]
y = df["label"]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)
model = MultinomialNB()
model.fit(X, y)
test_message = ["Congratulations! You won a free ticket."]
test = vectorizer.transform(test_message)
print("Posterior Probabilities:")
print(model.predict_proba(test))
prediction = model.predict(test)
print("Prediction:", prediction[0])
