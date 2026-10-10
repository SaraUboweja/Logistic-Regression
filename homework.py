import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
X = np.arange(10).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
print("Classes:", np.unique(y))
print("Type: Binary Classification")
model = LogisticRegression()
model.fit(X, y)
y_pred = model.predict(X)
print("Actual y:", y)
print("Predicted y:", y_pred)
print("Accuracy:", accuracy_score(y, y_pred) * 100, "%")