import math
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class CustomLogisticRegression:

    def __init__(self, fit_intercept=True, l_rate=0.01, n_epoch=100):
        self.fit_intercept = fit_intercept
        self.l_rate = ...
        self.n_epoch = ...

    @staticmethod
    def sigmoid(t):
        return 1 / (1 + math.exp(-t))

    def predict_proba(self, row, coef_):
        t = (coef_[0] if self.fit_intercept else 0) +\
            np.dot(row, coef_[(1 if self.fit_intercept else 0):])
        return self.sigmoid(t)


data = load_breast_cancer(as_frame=True).frame
X, y = data[['worst concave points', 'worst perimeter']], data[['target']]
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=43)
model = CustomLogisticRegression(True)
print([model.predict_proba(x, [0.77001597, -2.12842434, -2.39305793]) for x in X_test[:10]])
