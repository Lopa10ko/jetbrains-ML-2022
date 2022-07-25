# Stage 4/5: Data preprocessing

At this stage, we will improve model performance by preprocessing the data. We will see how normalization affects the accuracy

**Objectives:**
<ol>
  <li>Import <code>sklearn.preprocessing.Normalizer</code></li>
  <li>Initialize the normalizer, transform the features <code>x_train</code> and <code>x_test</code> to <code>x_train_norm</code> and <code>x_test_norm</code></li>
  <li>Answer questions: <ol>
    <li>Does the normalization have a positive impact in general? (yes/no)</li>
    <li>Which two models show the best scores?</li>
    </ol></li>
</ol> 

```py
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize


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
x_test_norm, x_train_norm = normalize(x_test), normalize(x_train)

fit_predict_eval(model=KNeighborsClassifier(),
                 features_train=x_train_norm, features_test=x_test_norm,
                 target_train=y_train, target_test=y_test)
fit_predict_eval(model=DecisionTreeClassifier(random_state=40),
                 features_train=x_train_norm, features_test=x_test_norm,
                 target_train=y_train, target_test=y_test)
fit_predict_eval(model=LogisticRegression(random_state=40),
                 features_train=x_train_norm, features_test=x_test_norm,
                 target_train=y_train, target_test=y_test)
fit_predict_eval(model=RandomForestClassifier(random_state=40),
                 features_train=x_train_norm, features_test=x_test_norm,
                 target_train=y_train, target_test=y_test)
print(f"The answer to the 1st question: yes\n\n"
      f"The answer to the 2nd question: KNeighborsClassifier-0.953, RandomForestClassifier-0.937")
```
