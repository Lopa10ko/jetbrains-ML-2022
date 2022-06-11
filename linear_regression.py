import pandas as pd
import numpy as np


class CustomLinearRegression:

    def __init__(self, *, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coefficient = ...
        self.intercept = ...

    def fit(self, X, y):
        tmp_matrix = np.array((np.linalg.inv(X.T @ X) @ X.T) @ y)
        self.coefficient = np.array([tmp_matrix[1]])
        self.intercept = tmp_matrix[0]


matrix = pd.DataFrame({'x': [4., 4.5, 5, 5.5, 6., 6.5, 7.],
                       'y': [33, 42, 45, 51, 53, 61, 62]})
obj = CustomLinearRegression()
obj.fit(X=np.array([[1, i] for i in matrix['x'].values]), y=np.array(matrix['y'].values))
print({'Intercept': obj.intercept, 'Coefficient': obj.coefficient})
