# Stage 1/5: The Keras dataset

We start by feeding data to our program. \
We will use the [MNIST digits dataset from Keras](https://keras.io/api/datasets/mnist/). \
Here is a sample of 28x28 images from the dataset:

<p><img alt="" src="https://ucarecdn.com/daa5381b-35fa-40bc-a45f-92898e06d1ad/-/crop/606x613/65,53/-/preview/"></p>

<p>To proceed, we need to figure out how a machine sees pictures. A computer senses an image as a 2D array of pixels. Each pixel has coordinates (x, y) and a value — from 0 (the darkest) to 255 (the brightest).</p>

<p><img alt="" src="https://ucarecdn.com/86dfc68f-d555-4628-859f-46a9f6324d84/"></p>

<p>In machine learning, we need to flatten (convert) an image into one dimension array. This means that a 28x28 image, which is initially a 2D array, transforms into a 1D array with 28x28 = 784 elements in it.</p>

**Objectives:** 

<ol>
	<li>Import <code>tensorflow</code> and <code>numpy</code> to your program. The first one loads the data, the second one transforms it;</li>
	<li>Load the data in your program. You need <code>x_train</code> and <code>y_train</code> only. Skip <code>x_test</code> and <code>y_test</code> in return of <code>load_data()</code>.<br>
	Sometimes <code>x_train</code> or <code>x_test</code> will be called <strong>the features array</strong> (because they contain brightnesses of the pixels, which are the images' features). <code>y_train</code> or <code>y_test</code> will be called <strong>the target array</strong> (because they contain classes, digits which we are going to predict);</li>
	<li>Reshape the features array to the 2D array with <span><span><span><span><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>n</mi></mrow><annotation encoding="application/x-tex">n</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.4306em;"></span><span class="mord mathnormal">n</span></span></span></span></span></span> rows (<span class="math-tex"><span><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>n</mi></mrow><annotation encoding="application/x-tex">n</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.4306em;"></span><span class="mord mathnormal">n</span></span></span></span></span></span> = number of images in the dataset) and <span class="math-tex"><span><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>m</mi></mrow><annotation encoding="application/x-tex">m</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.4306em;"></span><span class="mord mathnormal">m</span></span></span></span></span></span> columns (<span class="math-tex"><span><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>m</mi></mrow><annotation encoding="application/x-tex">m</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.4306em;"></span><span class="mord mathnormal">m</span></span></span></span></span></span> = number of pixels in each image);</li>
	<li>Provide the unique target classes' names, the shape of the features array, and the shape of the target variable in the following format: <code class="java">(the number of rows, the number of columns)</code>. Finally, print the minimum and maximum values of the features array.</li>
</ol>

<p>The input is the <a target="_blank" href="https://keras.io/api/datasets/mnist/">MNIST dataset</a>. The output contains flattened <code>features</code> and <code class="java">target</code> arrays. </p></div></div>

```py
import numpy as np
import tensorflow as tf


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
# assert X_train.shape == (60000, 28, 28)
# assert X_test.shape == (10000, 28, 28)
# assert y_train.shape == (60000,)
# assert y_test.shape == (10000,)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1] * X_train.shape[2]))

print(f"Classes: {np.unique(y_train)}\n"
      f"Features' shape: {X_train.shape}\n"
      f"Target's shape: {y_train.shape}\n"
      f"min: {np.amin(X_train)}, max: {np.amax(X_train)}")
```
```
Classes: [0 1 2 3 4 5 6 7 8 9]
Features' shape: (60000, 784)
Target's shape: (60000,)
min: 0, max: 255
```
