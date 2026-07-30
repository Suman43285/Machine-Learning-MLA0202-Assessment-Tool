from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
data = load_breast_cancer()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
malignant_prob = sum(y_train == 0) / len(y_train)
benign_prob = sum(y_train == 1) / len(y_train)
print("Prior Probability of Malignant:", malignant_prob)
print("Prior Probability of Benign:", benign_prob)
model = GaussianNB()
model.fit(X_train, y_train)
new_instance = X_test[0].reshape(1, -1)
prediction = model.predict(new_instance)
probability = model.predict_proba(new_instance)
print("\nPosterior Probabilities")
print("Malignant:", probability[0][0])
print("Benign:", probability[0][1])
if prediction[0] == 0:
    print("Predicted Class: Malignant")
else:
    print("Predicted Class: Benign")
