# Stage 3/5: Train models with default settings

We are ready to train our models. In this stage, you need to find the best algorithm that can identify handwritten digits. 
Refer to the following algorithms: <code>K-nearest Neighbors</code>, <code>Decision Tree</code>, <code>Logistic Regression</code>, and <code>Random Forest</code>. \
In this stage, you need to train these four classifiers with default parameters. 
In the next stages, we will try to improve their performances. 

```py
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def fit_predict_eval(model, features_train, features_test, target_train, target_test):
    model.fit(features_train, target_train)
    print(f'Model: {model}\nAccuracy: {round(accuracy_score(model.predict(features_test), target_test), 4)}\n')


(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test, y_train, y_test = train_test_split(X_train[:6000], Y_train[:6000],
                                                    test_size=0.3, random_state=40)
# from matplotlib import pyplot
# for i in range(9):
#       pyplot.subplot(330 + 1 + i)
#       pyplot.imshow(X_train[i], cmap=pyplot.get_cmap('gray'))
# pyplot.show()
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

fit_predict_eval(model=KNeighborsClassifier(),
                 features_train=x_train, features_test=x_test,
                 target_train=y_train, target_test=y_test)
fit_predict_eval(model=DecisionTreeClassifier(random_state=40),
                 features_train=x_train, features_test=x_test,
                 target_train=y_train, target_test=y_test)
fit_predict_eval(model=LogisticRegression(random_state=40),
                 features_train=x_train, features_test=x_test,
                 target_train=y_train, target_test=y_test)
fit_predict_eval(model=RandomForestClassifier(random_state=40),
                 features_train=x_train, features_test=x_test,
                 target_train=y_train, target_test=y_test)
print(f"The answer to the question: RandomForestClassifier - {0.939}")
```

```
Model: KNeighborsClassifier()
Accuracy: 0.935

Model: DecisionTreeClassifier(random_state=40)
Accuracy: 0.7606

Model: LogisticRegression(random_state=40)
Accuracy: 0.8733

Model: RandomForestClassifier(random_state=40)
Accuracy: 0.9394

The answer to the question: RandomForestClassifier - 0.939
```
