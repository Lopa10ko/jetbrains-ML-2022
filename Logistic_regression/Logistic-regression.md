# Gradient descent with MSE
In the previous stage, we have provided you with the <code>coef_</code> values. In this stage, you need to estimate the coef_ values by gradient descent on the Mean squared error cost function. Gradient descent is an optimization technique for finding the local minimum of a cost function by first-order differentiating. To be precise, we're going to implement the Stochastic gradient descent.

The Mean squared error cost function can be expressed as:
$J(b_0, b_1, ...) = \frac{1}{n} \sum(\bar{y_i}-y_i)^2$ \
Where $i$ indexes the rows (observations), and: 
$\bar{y_i}=\frac{1}{1+e^-t}, \\ \\ t_i=b_0+b_1 x_{i_1}+b_2 x_{i_2}+...$ 
$\bar{y_i}$ is the predicted probability value for the $i^{th}$ row, while $y_i$ is its actual value. 

For learning purposes, we will use the entire training set to update weights sequentially. The number of the epoch <code>n_epoch</code> is the number of iterations over the training set. 

If a particular weight value is updated by large increments, it descents down the quadratic curve in an erratic way and may jump to the opposite side of the curve. In this case, we may miss the value of the weight that minimizes the loss function. The learning rate <code>l_rate</code> can tune the value for updating the weight to the step size that allows for gradual descent along the curve with every iteration  

The <code>predict</code> method calculates the values of <code>y_hat</code> for each row in the test set and returns a numpy array that contains these values. Since we are solving a **binary classification problem**, the predicted values can be only 0 or 1. The return of predict depends on the cut-off point. The <code>predict_proba</code> probabilities that are less than the cut-off point are rounded to 0, while those that are equal or bigger are rounded to 1. Set the default <code>cut-off</code> value to 0.5. To determine the prediction accuracy of your model, use <code>accuracy_score</code> from <code>sklearn.metrics</code>.

```
worst concave points,worst perimeter,worst radius,y
0.320904,0.230304,-0.17156,1.0
-1.743529,-0.954428,-0.899849,1.0
1.014627,0.780857,0.773975,0.0
1.43299,-0.132764,-0.123973,0.0
```

```py
lr = CustomLogisticRegression(fit_intercept=True, l_rate=0.01, n_epoch=100)
# {'coef_': [ 0.7219814 , -2.06824488, -1.44659819, -1.52869155], 'accuracy': 0.75}

lr = CustomLogisticRegression(fit_intercept=False, l_rate=0.01, n_epoch=100)
# {'coef_': [-1.86289827, -1.60283708, -1.69204615], 'accuracy': 0.75}
```

Solution:
```py
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from scipy.stats import zscore


class CustomLogisticRegression:
    def __init__(self, fit_intercept=True, l_rate=0.01, n_epoch=1000):
        self.fit_intercept = fit_intercept
        self.l_rate = l_rate
        self.n_epoch = n_epoch

    @staticmethod
    def sigmoid(t):
        return float(1 / (1 + np.exp(-t)))

    def predict_proba(self, row, coef_):
        t = float(row.dot(coef_))
        return self.sigmoid(t)

    def fit_mse(self, X_train, y_train):
        self.coef_ = np.zeros(self.fit_intercept + X_train.shape[1])  # initialized weights
        for _ in range(self.n_epoch):
            for i, row in enumerate(X_train.values):
                row = row.tolist()
                if self.fit_intercept:
                    row.insert(0, 1)
                row = np.array([row])
                y_hat = self.predict_proba(row, self.coef_)
                for j in range(row.shape[1]):
                    self.coef_[j] -= self.l_rate * (y_hat - y_train.values[i]) * y_hat * (1 - y_hat) * row[0, j]

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
my_logreq = CustomLogisticRegression()
my_logreq.fit_mse(X_train, y_train)
print({'coef_': my_logreq.coef_.tolist(), 'accuracy': np.sum(my_logreq.predict(X_test) == y_test) / X_test.shape[0]})
```

```py
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from scipy.stats import zscore


class CustomLogisticRegression:
    def __init__(self, fit_intercept=True, l_rate=0.01, n_epoch=1000):
        self.fit_intercept = fit_intercept
        self.l_rate = l_rate
        self.n_epoch = n_epoch

    @staticmethod
    def sigmoid(t):
        return float(1 / (1 + np.exp(-t)))

    def predict_proba(self, row, coef_):
        t = float(row.dot(coef_))
        return self.sigmoid(t)

    def fit_mse(self, X_train, y_train):
        self.coef_ = np.zeros(self.fit_intercept + X_train.shape[1])  # initialized weights
        for _ in range(self.n_epoch):
            for i, row in enumerate(X_train.values):
                row = row.tolist()
                if self.fit_intercept:
                    row.insert(0, 1)
                row = np.array([row])
                y_hat = self.predict_proba(row, self.coef_)
                for j in range(row.shape[1]):
                    self.coef_[j] -= self.l_rate * (y_hat - y_train.values[i]) * y_hat * (1 - y_hat) * row[0, j]

    def fit_log_loss(self, X_train, y_train):
        self.coef_ = np.zeros(self.fit_intercept + X_train.shape[1])  # initialized weights
        for _ in range(self.n_epoch):
            for i, row in enumerate(X_train.values):
                row = row.tolist()
                if self.fit_intercept:
                    row.insert(0, 1)
                row = np.array([row])
                y_hat = self.predict_proba(row, self.coef_)
                for j in range(row.shape[1]):
                    self.coef_[j] -= self.l_rate * (y_hat - y_train.values[i]) * row[0, j] / (self.fit_intercept + X_train.shape[0])

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
my_logreq = CustomLogisticRegression()
# my_logreq.fit_mse(X_train, y_train)
my_logreq.fit_log_loss(X_train, y_train)
print({'coef_': my_logreq.coef_.tolist(), 'accuracy': np.sum(my_logreq.predict(X_test) == y_test) / X_test.shape[0]})
```

```py
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
```
