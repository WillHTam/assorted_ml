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

## A Third Set

- Having a test and training, and changing hyperparameters based on the test set may cause overfitting on the test set.
- Use the *validation set* to evaluate results from the training set. Then, use the test set to double-check your evaluation after the model has "passed" the validation set. The following figure shows this new workflow:
- Pick the model that does best on the validation set
- Double-check that model against the test set
- _This is a better workflow because it creates fewer exposures to the test set_
- Test sets and validation sets "wear out" with repeated use. That is, the more you use the same data to make decisions about hyperparameter settings or other model improvements, the less confidence you'll have that these results actually generalize to new, unseen data. **Note that validation sets typically wear out more slowly than test sets.**
- If possible, it's a good idea to collect more data to "refresh" the test set and validation set. Starting anew is a great reset.

## Feature Engineering

- a feature vector, which is the set of floating-point values comprising the examples in your data set. Feature engineering means transforming raw data into a feature vector. Expect to spend significant time doing feature engineering.  Many machine learning models must represent the features as real-numbered vectors since the feature values must be multiplied by the model weights.
- Integer and floating-point data don't need a special encoding because they can be multiplied by a numeric weight.
- **Categorical features** have a discrete set of possible values. Since models cannot multiply strings by the learned weights, we use feature engineering to convert strings to numeric values. Instead of converting Cat1 to 1 and Cat2 to 2, instead use one-hot or multi-hot to encode so that there is no correlation between the numbers and the category.
- Instead of including all possible values, only include those that appear, _sparse representation_

## Qualities of Good Features

- Avoid rarely used discrete feature values
  - Good feature values should appear more than 5 or so times in a data set. Doing so enables a model to learn how this feature value relates to the label. That is, having many examples with the same discrete value gives the model a chance to see the feature in different settings, and in turn, determine when it's a good predictor for the label
  - Conversely, if a feature's value appears only once or very rarely, the model can't make predictions based on that feature. For example, unique_house_id is a bad feature because each value would be used only once, so the model couldn't learn anything from it
- Prefer clear and obvious meanings
  - I.e. prefer clear time measurements instead of say Unix epoch time
- Don't mix "magic" values with actual data
  - Good floating-point features don't contain peculiar out-of-range discontinuities or "magic" values. For example, suppose a feature holds a floating-point value between 0 and 1.
  - A missing value should not be marked with '-1'
  - To work around magic values, convert feature into two features
    - one feature holds only quality ratings, never magic values
    - one feature holds a boolean value indicating whether or not the value exists
- Account for upstream instability
  - The definition of a feature shouldn't change over time.
  - A value inferred by another model should not be used, ie. a number that may hold meaning for one situation, but changes meaning in another context.  Perhaps like an area code across a long timespan.

## Somes notes on Cleaning Data

- Scaling feature values
  - Scaling means converting floating-point feature values from their natural range (for example, 100 to 900) into a standard range (for example, 0 to 1 or -1 to +1). If a feature set consists of only a single feature, then scaling provides little to no practical benefit. If, however, a feature set consists of multiple features, then feature scaling provides the following benefits
    - Helps gradient descent converge more quickly.
    - Helps avoid the "NaN trap," in which one number in the model becomes a NaN (e.g., when a value exceeds the floating-point precision limit during training), and—due to math operations—every other number in the model also eventually becomes a NaN.
    - Helps the model learn appropriate weights for each feature. Without feature scaling, the model will pay too much attention to the features having a wider range.
  - You don't have to give every floating-point feature exactly the same scale. Nothing terrible will happen if Feature A is scaled from -1 to +1 while Feature B is scaled from -3 to +3. However, your model will react poorly if Feature B is scaled from 5000 to 100000.
- Handling extreme outliers
  - Take the log of every value? This could/would still leave a tail
  - Clipping
    - Clipping the feature value at a certain point doesn't mean ignoring all values greater than that point. Clipping at 4.0 means that all values greater than 4.0 become 4.0.  This scaled feature set is more useful than the original data.  
- Binning
  - Using latitude as a home value feature is bad because a house at lat 35 is not inherently more valuable than that at 34.  Instead, use binning and one-hot encode the bins.  By giving each latitude its own boolean, the model can learn different weights for each latitude.
  - Alternatively, bin by quantile, which ensures the number of examples in each bin is equal.  This also alleviates much of the problem of outliers.
- Scrubbing
  - Common problems of:
    - omitted values
    - duplicate examples
    - bad labels
    - bad feature values - accidental extra digit for example
  - omitted and duplicate values are easily found, bad features are harder to find
    - use max/min, mean/median, and std. deviation.
    - generate lists of the most common values for discrete features
      - should rows with `country:uk` have `language:jp`?
- General Rules:
  - Keep in mind what you think your data should look like.
  - Verify that the data meets these expectations (or that you can explain why it doesn’t).
  - Double-check that the training data agrees with other sources (for example, dashboards).  

## Feature Crosses

- Define a new synthetic feature, a feature cross
  - This new feature is a product of two other features, and can enable nonlinear learning in a linear learner.
    - Important because linear learners scale well with big datasets (vowpal-wabit, sofia-ml)
  - For example a feature cross between number of rooms, binned latitude and binned longitude, would make it clear to the model that there is a difference between three rooms in San Francisco vs Fresno, and other cities
  - For example in tic-tac-toe, would allow for the use of top-right, btm-left etc coordinates to the model
- Kinds of feature crosses
  - We can create many different kinds of feature crosses. For example:
  - [A X B]: a feature cross formed by multiplying the values of two features.
  - [A x B x C x D x E]: a feature cross formed by multiplying the values of five features.
  - [A x A]: a feature cross formed by squaring a single feature.  
  - Thanks to stochastic gradient descent, linear models can be trained efficiently. Consequently, supplementing scaled linear models with feature crosses has traditionally been an efficient way to train on massive-scale data sets.
- Crossing One-Hot Vectors
  - In practice, machine learning models seldom cross continuous features. However, machine learning models do frequently cross one-hot feature vectors. Think of feature crosses of one-hot feature vectors as logical conjunctions. For example, suppose we have two features: country and language. A one-hot encoding of each generates vectors with binary features that can be interpreted as `country=USA, country=France or language=English, language=Spanish`. Then, if you do a feature cross of these one-hot encodings, you get binary features that can be interpreted as logical conjunctions, such as: `country:usa AND language:spanish`.

```txt
As another example, suppose you bin latitude and longitude, producing separate one-hot five-element feature vectors. For instance, a given latitude and longitude could be represented as follows:

  binned_latitude = [0, 0, 0, 1, 0]
  binned_longitude = [0, 1, 0, 0, 0]

Suppose you create a feature cross of these two feature vectors:

  binned_latitude X binned_longitude
  
This feature cross is a 25-element one-hot vector (24 zeroes and 1 one). The single 1 in the cross identifies a particular conjunction of latitude and longitude. Your model can then learn particular associations about that conjunction.
```

- Now suppose our model needs to predict how satisfied dog owners will be with dogs based on two features:
  - Behavior type (barking, crying, snuggling, etc.)
  - Time of day
  - Build the feature [behavior x time of day]
    - This results with vastly more predictive ability than either feature on its own. For example, if a dog cries (happily) at 5:00 pm when the owner returns from work will likely be a great positive predictor of owner satisfaction. Crying (miserably, perhaps) at 3:00 am when the owner was sleeping soundly will likely be a strong negative predictor of owner satisfaction.