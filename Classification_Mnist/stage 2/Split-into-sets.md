# Stage 2/5: Split into sets

At this stage, you need to use <code>sklearn</code> to split your data into train and test sets. \
We will use only a portion of the [MNIST dataset](https://keras.io/api/datasets/mnist/) to process the training stage faster.

**Objectives:**
<ol>
  <li>Import a necessary tool from <code>sklearn</code>;</li>
  <li>Use the first 6000 rows of the dataset. Set the test set size as $0.3$ and the random seed of $40$ to make your output reproducible;</li>
  <li>Make sure that your set is balanced after splitting. Print new data shapes and the proportions of samples per class in the training set.</li>
</ol>


```py
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

def main():
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

  print(f"x_train shape: {x_train.shape}\n"
        f"x_test shape: {x_test.shape}\n"
        f"y_train shape: {y_train.shape}\n"
        f"y_test shape: {y_test.shape}\n"
        f"Proportion of samples per class in train set:\n"
        f"{pd.Series(y_train).value_counts(normalize=True).round(2)}")

if __name__ == '__main__':
    main()
```
```
x_train shape: (4200, 784)
x_test shape: (1800, 784)
y_train shape: (4200,)
y_test shape: (1800,)
Proportion of samples per class in train set:
1    0.11
7    0.11
4    0.11
6    0.10
0    0.10
9    0.10
3    0.10
2    0.10
8    0.09
5    0.09
dtype: float64
```
