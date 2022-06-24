import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression


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
    'f1': [2.31, 7.07, 7.07, 2.18, 2.18, 2.18, 7.87, 7.87, 7.87, 7.87],
    'f2': [65.2, 78.9, 61.1, 45.8, 54.2, 58.7, 96.1, 100., 85.9, 94.3],
    'f3': [15.3, 17.8, 17.8, 18.7, 18.7, 18.7, 15.2, 15.2, 15.2, 15.2],
})
target = pd.Series([24., 21.6, 34.7, 33.4, 36.2, 28.7, 27.1, 16.5, 18.9, 15.])
X, y = data[['f1', 'f2', 'f3']].values, target.values
model, custom_model = LinearRegression(fit_intercept=True), CustomLinearRegression(fit_intercept=True)
model.fit(X, y)
custom_model.fit(X, y)
pred, custom_pred = model.predict(X), custom_model.predict(X)
custom_R2 = custom_model.r2_score(y, custom_pred)
R2 = r2_score(y, pred)
custom_RMSE = custom_model.rmse(y, yhat=custom_pred)
RMSE = sqrt(mean_squared_error(y, y_pred=pred))
print({'Intercept': abs(model.intercept_ - custom_model.intercept),
       'Coefficient': np.array(model.coef_) - np.array(custom_model.coefficient[1:]),
       'R2': abs(R2 - custom_R2),
       'RMSE': abs(RMSE - custom_RMSE)})
