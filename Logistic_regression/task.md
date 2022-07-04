# Gradient descent with MSE
In the previous stage, we have provided you with the coef_ values. In this stage, you need to estimate the coef_ values by gradient descent on the Mean squared error cost function. Gradient descent is an optimization technique for finding the local minimum of a cost function by first-order differentiating. To be precise, we're going to implement the Stochastic gradient descent.

The Mean squared error cost function can be expressed as:
$J(b_0, b_1, ...) = \frac{1}{n} \sum(\bar{y_i}-y_i)^2$ \
Where $i$ indexes the rows (observations), and: 
$\bar{y_i}=\frac{1}{1+e^-t}, \\ \\ t_i=b_0+b_1 x_{i_1}+b_2 x_{i_2}+...$ 
$\bar{y_i}$ is the predicted probability value for the $i^{th}$ row, while $y_i$ is its actual value. 

For learning purposes, we will use the entire training set to update weights sequentially. The number of the epoch $n_{epoch}$ is the number of iterations over the training set. 

If a particular weight value is updated by large increments, it descents down the quadratic curve in an erratic way and may jump to the opposite side of the curve. In this case, we may miss the value of the weight that minimizes the loss function. The learning rate $l_{rate}$ can tune the value for updating the weight to the step size that allows for gradual descent along the curve with every iteration  

The $predict$ method calculates the values of $y_{hat}$ for each row in the test set and returns a numpy array that contains these values. Since we are solving a **binary classification problem**, the predicted values can be only 0 or 1. The return of predict depends on the cut-off point. The predict_proba probabilities that are less than the cut-off point are rounded to 0, while those that are equal or bigger are rounded to 1. Set the default cut-off value to 0.5. To determine the prediction accuracy of your model, use **accuracy_score** from **sklearn.metrics**.

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