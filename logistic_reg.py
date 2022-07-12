import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from sklearn.linear_model import LogisticRegression
import math


class CustomLogisticRegression:
    def __init__(self, fit_intercept=True, l_rate=0.01, n_epoch=1000):
        self.fit_intercept = fit_intercept
        self.l_rate = l_rate
        self.n_epoch = n_epoch
        self.mse_error_init = ...
        self.mse_error_end = ...
        self.logloss_error_init = ...
        self.logloss_error_end = ...

    @staticmethod
    def sigmoid(t):
        return float(1 / (1 + np.exp(-t)))

    def predict_proba(self, row, coef_):
        t = float(row.dot(coef_))
        return self.sigmoid(t)

    def fit_mse(self, X_train, y_train):
        self.coef_ = np.zeros(self.fit_intercept + X_train.shape[1])  # initialized weights
        for k in range(self.n_epoch):
            for i, row in enumerate(X_train.values):
                row = row.tolist()
                if self.fit_intercept:
                    row.insert(0, 1)
                row = np.array([row])
                y_hat = self.predict_proba(row, self.coef_)
                for j in range(row.shape[1]):
                    self.coef_[j] -= self.l_rate * (y_hat - y_train.values[i]) * y_hat * (1 - y_hat) * row[0, j]
            if k == 0:
                self.mse_error_init = [(y_hat - y_train.values[i]) ** 2 / len(y_train) for i in range(len(y_train))]
            elif k == self.n_epoch - 1:
                self.mse_error_end = [(y_hat - y_train.values[i]) ** 2 / len(y_train) for i in range(len(y_train))]

    def fit_log_loss(self, X_train, y_train):
        self.coef_ = np.zeros(self.fit_intercept + X_train.shape[1])  # initialized weights
        for k in range(self.n_epoch):
            for i, row in enumerate(X_train.values):
                row = row.tolist()
                if self.fit_intercept:
                    row.insert(0, 1)
                row = np.array([row])
                y_hat = self.predict_proba(row, self.coef_)
                for j in range(row.shape[1]):
                    self.coef_[j] -= self.l_rate * (y_hat - y_train.values[i]) * row[0, j] / (self.fit_intercept + X_train.shape[0])
            if k == 0:
                self.logloss_error_init = [-(y_train.values[i] * math.log(y_hat) + (1 - y_train.values[i]) * math.log(1 - y_hat)) / len(y_train) for i in range(len(y_train))]
            elif k == self.n_epoch - 1:
                self.logloss_error_end = [-(y_train.values[i] * math.log(y_hat) + (1 - y_train.values[i]) * math.log(1 - y_hat)) / len(y_train) for i in range(len(y_train))]

    def predict(self, X_test, cut_off=0.5):
        predictions = np.zeros(X_test.shape[0], dtype=int)
        for i, row in enumerate(X_test.values):
            row = row.tolist()
            if self.fit_intercept:
                row.insert(0, 1)
            row = np.array([row])
            y_hat = self.predict_proba(row, self.coef_)
            predictions[i] = int(y_hat >= cut_off)
        return predictions  # predictions are binary values - 0 or 1


X, y = datasets.load_breast_cancer(return_X_y=True, as_frame=True)
X = X[['worst concave points', 'worst perimeter', 'worst radius']].apply(zscore)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=43)
logloss_logreg, mse_logreg, std_logreg = CustomLogisticRegression(), CustomLogisticRegression(), LogisticRegression()

logloss_logreg.fit_log_loss(X_train, y_train)
mse_logreg.fit_mse(X_train, y_train)
std_logreg.fit(X_train, y_train)

mse_accuracy = np.sum(mse_logreg.predict(X_test) == y_test) / X_test.shape[0]
logloss_accuracy = np.sum(logloss_logreg.predict(X_test) == y_test) / X_test.shape[0]
sklern_accuracy = np.sum(std_logreg.predict(X_test) == y_test) / X_test.shape[0]

print({'mse_accuracy': mse_accuracy, 'logloss_accuracy': logloss_accuracy, 'sklearn_accuracy': sklern_accuracy,
       'mse_error_first': mse_logreg.mse_error_init, 'mse_error_last': mse_logreg.mse_error_end,
       'logloss_error_first': logloss_logreg.logloss_error_init, 'logloss_error_last': logloss_logreg.logloss_error_end})

# import numpy as np
# from sklearn import datasets
# from sklearn.model_selection import train_test_split
# from scipy.stats import zscore


# class CustomLogisticRegression:
#     def __init__(self, fit_intercept=True, l_rate=0.01, n_epoch=1000):
#         self.fit_intercept = fit_intercept
#         self.l_rate = l_rate
#         self.n_epoch = n_epoch

#     @staticmethod
#     def sigmoid(t):
#         return float(1 / (1 + np.exp(-t)))

#     def predict_proba(self, row, coef_):
#         t = float(row.dot(coef_))
#         return self.sigmoid(t)

#     def fit_mse(self, X_train, y_train):
#         self.coef_ = np.zeros(self.fit_intercept + X_train.shape[1])  # initialized weights
#         for _ in range(self.n_epoch):
#             for i, row in enumerate(X_train.values):
#                 row = row.tolist()
#                 if self.fit_intercept:
#                     row.insert(0, 1)
#                 row = np.array([row])
#                 y_hat = self.predict_proba(row, self.coef_)
#                 for j in range(row.shape[1]):
#                     self.coef_[j] -= self.l_rate * (y_hat - y_train.values[i]) * y_hat * (1 - y_hat) * row[0, j]

#     def predict(self, X_test, cut_off=0.5):
#         predictions = np.zeros(X_test.shape[0], dtype=int)
#         for i, row in enumerate(X_test.values):
#             row = row.tolist()
#             if self.fit_intercept:
#                 row.insert(0, 1)
#             row = np.array([row])
#             y_hat = self.predict_proba(row, self.coef_)
#             predictions[i] = int(y_hat >= cut_off)
#         return predictions  # predictions are binary values - 0 or 1


# X, y = datasets.load_breast_cancer(return_X_y=True, as_frame=True)
# X = X[['worst concave points', 'worst perimeter', 'worst radius']].apply(zscore)
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=43)
# my_logreq = CustomLogisticRegression()
# my_logreq.fit_mse(X_train, y_train)
# print({'coef_': my_logreq.coef_.tolist(), 'accuracy': np.sum(my_logreq.predict(X_test) == y_test) / X_test.shape[0]})

# import math
# import numpy as np
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler


# class CustomLogisticRegression:

#     def __init__(self, fit_intercept=True, l_rate=0.01, n_epoch=100):
#         self.fit_intercept = fit_intercept
#         self.l_rate = ...
#         self.n_epoch = ...

#     @staticmethod
#     def sigmoid(t):
#         return 1 / (1 + math.exp(-t))

#     def predict_proba(self, row, coef_):
#         t = (coef_[0] if self.fit_intercept else 0) +\
#             np.dot(row, coef_[(1 if self.fit_intercept else 0):])
#         return self.sigmoid(t)


# data = load_breast_cancer(as_frame=True).frame
# X, y = data[['worst concave points', 'worst perimeter']], data[['target']]
# X = StandardScaler().fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=43)
# model = CustomLogisticRegression(True)
# print([model.predict_proba(x, [0.77001597, -2.12842434, -2.39305793]) for x in X_test[:10]])
