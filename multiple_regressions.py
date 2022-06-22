import pandas as pd
import numpy as np
from math import sqrt


class CustomLinearRegression:
    def __init__(self, *, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coefficient = ...
        self.intercept = ...

    def fit(self, X, y):
        if self.fit_intercept:
            ones = np.ones(shape=(X.shape[0], 1))
            X = np.hstack((ones, X))
        self.coefficient = np.linalg.inv(X.T @ X) @ X.T @ y
        self.intercept = self.coefficient[0]

    def predict(self, X):
        if self.fit_intercept:
            ones = np.ones(shape=(X.shape[0], 1))
            X = np.hstack((ones, X))
        return X @ self.coefficient

    def r2_score(self, y, yhat):
        y_mean = sum(y) / len(y)
        return 1 - sum([i ** 2 for i in list(yhat - y)]) / sum([i ** 2 for i in list(y - y_mean)])

    def rmse(self, y, yhat):
        return sqrt(sum([i ** 2 for i in list(yhat - y)]) / len(y))


data = pd.DataFrame({
    'Capacity': [0.9, 0.5, 1.75, 2.0, 1.4, 1.5, 3.0, 1.1, 2.6, 1.9],
    'Age': [11, 11, 9, 8, 7, 7, 6, 5, 5, 4],
    'Cost/ton': [21.95, 27.18, 16.9, 15.37, 16.03, 18.15, 14.22, 18.72, 15.4, 14.69],
})

X, y = data[['Capacity', 'Age']].values, data['Cost/ton'].values
obj = CustomLinearRegression(fit_intercept=True)
obj.fit(X, y)
y_pred = obj.predict(X)
R2, RMSE = obj.r2_score(y, y_pred), obj.rmse(y, y_pred)
print(np.array({'Intercept': obj.intercept, 'Coefficient': np.array(obj.coefficient[1:]),
       'R2': R2, 'RMSE': RMSE}))
