# Notes

## Loss

- Performed iteratively
  - A Machine Learning model is trained by starting with an initial guess for the weights and bias and iteratively adjusting those guesses until learning the weights and bias with the lowest possible loss.
- Gradient Descent
  - Convex problems have only one minimum; that is, only one place where the slope is exactly 0. That minimum is where the loss function converges.
  - Calculating the loss function for every conceivable value of the calculated slope.  Doing so over the entire data set would be an inefficient way of finding the convergence point. Let's examine a better mechanism—very popular in machine learning—called gradient descent.
  - The first stage in gradient descent is to pick a starting value (a starting point); the starting point doesn't matter much; therefore, many algorithms simply set to 0 or pick a random value.
  - The gradient descent algorithm then calculates the gradient of the loss curve at the starting point. Here in Figure 3, the gradient of loss is equal to the derivative (slope) of the curve, and tells you which way is "warmer" or "colder." When there are multiple weights, the gradient is a *vector of partial derivatives with respect to the weights*.
  - Note that a gradient is a *vector*, so it has both of the following characteristics:
    - a direction
    - a magnitude
  - The gradient always points in the direction of steepest increase in the loss function. The gradient descent algorithm takes a step in the direction of the negative gradient in order to reduce loss as quickly as possible.
  - As noted, the gradient vector has both a direction and a magnitude. Gradient descent algorithms multiply the gradient by a scalar known as the *learning rate* (also sometimes called step size) to determine the next point. For example, if the gradient magnitude is 2.5 and the learning rate is 0.01, then the gradient descent algorithm will pick the next point 0.025 away from the previous point.
  - Hyperparameters are the knobs that programmers tweak in machine learning algorithms. Most machine learning programmers spend a fair amount of time tuning the learning rate. If you pick a learning rate that is too small, learning will take too long:
    - There's a *Goldilocks* learning rate for every regression problem. The Goldilocks value is related to how flat the loss function is. If you know the gradient of the loss function is small then you can safely try a larger learning rate, which compensates for the small gradient and results in a larger step size.
  - In gradient descent, a *batch* is the total number of examples you use to calculate the gradient in a single iteration. So far, we've assumed that the batch has been the entire data set. When working at Google scale, data sets often contain billions or even hundreds of billions of examples. Furthermore, Google data sets often contain huge numbers of features. Consequently, a batch can be enormous. A very large batch may cause even a single iteration to take a very long time to compute.
  - What if we could get the right gradient on average for much less computation? By choosing examples at random from our data set, we could estimate (albeit, noisily) a big average from a much smaller one. *Stochastic gradient descent (SGD)* takes this idea to the extreme--it uses only a single example (a batch size of 1) per iteration. Given enough iterations, SGD works but is very noisy. The term "stochastic" indicates that the one example comprising each batch is chosen at random.
  - *Mini-batch stochastic gradient descent* (mini-batch SGD) is a compromise between full-batch iteration and SGD. A mini-batch is typically between 10 and 1,000 examples, chosen at random. Mini-batch SGD reduces the amount of noise in SGD but is still more efficient than full-batch.

## Tensorflow API

- High level to Low
  - TF Estimators, OOP API
  - tf.layers, tf.losses, tf.metrics
    - reusable libraries for common model components
  - Python Tensorflow, wraps C++ kernel
  - C++ Tensorflow
  - CPU/GPU/TPU - multiplatform usage

## Generic Estimator

```py
import tensorflow as tf

# Set up a linear classifier.
classifier = tf.estimator.LinearClassifier(feature_columns)

# Train the model on some example data.
classifier.train(input_fn=train_input_fn, steps=2000)

# Use it to predict.
predictions = classifier.predict(input_fn=predict_input_fn)
```

## The following three basic assumptions guide generalization

- We draw examples independently and identically (i.i.d) at random from the distribution. In other words, examples don't influence each other. (An alternate explanation: i.i.d. is a way of referring to the randomness of variables.)
- The distribution is stationary; that is the distribution doesn't change within the data set.
  - **Stationarity** is property of data in a data set, in which the data distribution stays constant across one or more dimensions. Most commonly, that dimension is time, meaning that data exhibiting stationarity doesn't change over time. For example, data that exhibits stationarity doesn't change from September to December. 
- We draw examples from partitions from the same distribution.
- In practice, we sometimes violate these assumptions. For example:
  - Consider a model that chooses ads to display. The i.i.d. assumption would be violated if the model bases its choice of ads, in part, on what ads the user has previously seen.
  - Consider a data set that contains retail sales information for a year. User's purchases change seasonally, which would violate stationarity.

- A Third Set
  - Having a test and training, and changing hyperparameters based on the test set may cause overfitting on the test set.
  - Use the *validation set* to evaluate results from the training set. Then, use the test set to double-check your evaluation after the model has "passed" the validation set. The following figure shows this new workflow:
  - Pick the model that does best on the validation set
  - Double-check that model against the test set
  - _This is a better workflow because it creates fewer exposures to the test set_
  - Test sets and validation sets "wear out" with repeated use. That is, the more you use the same data to make decisions about hyperparameter settings or other model improvements, the less confidence you'll have that these results actually generalize to new, unseen data. **Note that validation sets typically wear out more slowly than test sets.**
  - If possible, it's a good idea to collect more data to "refresh" the test set and validation set. Starting anew is a great reset.