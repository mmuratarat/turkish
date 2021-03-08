---
layout: post
title: "Cross Entropy for Tensorflow"
author: "MMA"
comments: true
---

Cross entropy can be used to define a loss function (cost function) in machine learning and optimization. It is defined on probability distributions, not single values. It works for classification because classifier output is (often) a probability distribution over class labels. 

For discrete distributions $p$ and $q$, it's given as follows:
 
$$H(p, q) = -\sum_y p(y) \log q(y)$$
 
When the cross entropy loss is used with 'hard' class labels, what this really amounts to is treating $p$ as the conditional empirical distribution over class labels. This is a distribution where the probability is 1 for the observed class label and 0 for all others. $q$ is the conditional distribution (probability of class label, given input) learned by the classifier. For a single observed data point with input $x_0$ and class $y_0$, we can see that the expression above reduces to the standard log loss (which is averaged over all data points):

$$-\sum_y I\{y = y_0\} \log q(y \mid x_0) = -\log q(y_0 \mid x_0)$$

Here, $I\\{\cdot\\}$ is the indicator function, which is $1$ when its argument is true or $0$ otherwise. The sum is taken over the set of possible class labels.

The concept of cross entropy applies equally well to continuous distributions. But, it can't be used for regression models that output a point estimate (e.g. the conditional mean because the regression function is the conditional mean of $Y$ over $X$, $E [ Y \mid X]$) but it can be used for models that output a full probability distribution. 

If you have a model that gives the full conditional distribution (probability of output given input), you can use cross entropy as a loss function.

For continuous distributions $p$ and $q$, the cross entropy is defined as:

$$H(p, q) = -\int_{Y} p(y) \log q(y) dy$$

Just considering a single observed input/output pair $(x, y)$, $p$ would be the empirical conditional distribution (a delta function over the observed output value), and $q$ would be the modeled conditional distribution (probability of output given input). In this case, the cross entropy reduces to $-\log q(y \mid x)$. Summing over data points, this is just the negative log likelihood!

# ENTROPY

Entropy is a measure of the uncertainty associated with a given distribution $p(y)$ with $K$ distinct states. The higher the entropy, the less certain we are about the value we're going to get. Calculating the information for a random variable is called "information entropy", "Shannon entropy", or simply "entropy".

$$
H(p) = - \sum_{k=1}^{K} p(y_{k}) \log p(y_{k})
$$

In a scenario of binary classification, we will have two classes: positive class and negative class. If all the points are positive, the uncertainty of that distribution will be zero. After all, there would be no doubt about the class of a data point: it is always positive! So, entropy is zero!

\begin{equation}
H(p) = - 1 \log_2 (1) = 0
\end{equation}

On the other hand, what if we knew exactly half of the points were positive and the other half, negative? That’s the worst case scenario. It is totally random! For that case, entropy is given by (we have two classes with probability 0.5 for each, half/half, $p=q$):

$$
H(p) = - 0.5 \log_{2}(0.5) - 0.5 \log_{2}(0.5) = 1
$$

Where $log()$ is the base-2 logarithm and p(x) is the probability of the event x. The choice of the base-2 logarithm means that the units of the information measure is in bits (binary digits).

For every other case in between, we can compute the entropy of a distribution, using the formula below, where $K$ is the number of classes ($K$ discrete states):

$$
H(p) = - \sum_{k=1}^{K} p(y_{k}) \log p(y_{k})
$$

(This can also be thought as in the following. There are $K$ distinct events. Each event $k$ has probability $p(y_{k})$)

Note that the $log()$ function uses base-2 and the units are bits. A natural logarithm can be used instead.

**An Example**

{% highlight python %}
from numpy import log

p = {'rain': .14, 'snow': .37, 'sleet': .03, 'hail': .46}

def entropy(prob_dist):
    return -sum([ p*log(p) for p in prob_dist.values() ])

entropy(p)
#1.1055291211185652
{% endhighlight %}

If we know the true distribution of a random variable, we can compute its entropy. However, we cannot always know the true distribution. That is what Machine Learning algorithms do. We try to approximate the true distribution with an other distribution, say, $q(y)$.

Let’s assume data points follow this other distribution $q(y)$. But we know they are actually coming from the true (unknown) distribution $p(y)$.

If we compute entropy (uncertainty) between these two (discrete) distributions, we are actually computing the cross-entropy between them:

$$H(p, q) = -\sum_{k=1}^{K} p(y_{k}) \log q(y_{k})$$

If we can find a distribution $q(y)$ as close as possible to $p(y)$, values for both cross-entropy and entropy will match as well. However, this is not the always case. Therefore, cross-entropy will be greater than the entropy computed on the true distribution.

$$ H(p, q) - H(p) > 0 $$

This difference between cross-entropy and entropy is called *Kullback-Leibler Divergence*.

# KL DIVERGENCE

The Kullback-Leibler Divergence, or KL Divergence for short, is a measure of dissimilarity between two distributions. It can be interpreted as measuring the likelihood that samples represented by the empirical distribution $p$ were generated by a fixed distribution q. If $D_{KL} (p \mid \mid q)=0$, we can guarantee that $p$ is generated by $q$. As $D_{KL} (p \mid \mid q) \rightarrow \infty$, we can say that it is increasingly unlikely that $p$ was generated by $q$.

$$ 
\begin{split}
D_{KL} (p || q) = H(p, q) - H(p) &= \mathbb{E}_{p(y_{k})} \left [ \log \left ( \frac{p(y_{k})}{q(y_{k})} \right ) \right ] \\
&= \sum_{k=1}^{K} p(y_{k}) \log\left[\frac{p(y_{k})}{q(y_{k})}\right] \\
&=\sum_{k=1}^{K} p(y_{k}) \left[\log p(y_{k}) - \log q(y_{k})\right]
\end{split}
$$

This means that, the closer $q(y)$ gets to $p(y)$, the lower the divergence and consequently, the cross-entropy will be. In other words, KL divergence gives us "distance" between 2 distributions, and that minimizing it is equivalent to minimizing cross-entropy. Minimizing cross-entropy will make $q(y)$ converge to $p(y)$, and $H(p, q)$ itself will converge to $H(p)$. Therefore, we need to approximate to a good distribution by using the classifier. 

Now, for one particular data point, if $p \in \\{y,1-y\\}$ and $q \in \\{\hat{y}, 1-\hat{y}\\}$, we can re-write cross-entropy as:

\begin{equation}
H(p, q) = -\sum_{k=1}^{K=2} p(y_{k}) \log q(y_{k}) =-y\log \hat{y}-(1-y)\log (1-\hat{y})
\end{equation}
which is nothing but logistic loss.

The final step is to compute the average of all points in both classes, positive and negative, will give [binary cross-entropy formula](#binary-cross-entropy).

**NOTE**: Why don't we use KL-Divergence in machine learning models instead of the cross entropy? The KL-Divergence between distributions requires us to know both the true distribution and distribution of our predictions thereof. Unfortunately, we never have the former: that's why we build the model.

**NOTE**: It may be tempting to think of KL Divergence as a distance metric, however we cannot use KL Divergence to measure the distance between two distributions. The reason for this is that KL Divergence is not symmetric, meaning that $D_{KL} (p 	\|\| q)$ may not be equal to $D_{KL} (q 	\|\| p)$.

{% highlight python %}
# example of calculating the kl divergence between two mass functions
from math import log2
# define distributions
events = ['red', 'green', 'blue']
p = [0.10, 0.40, 0.50]
q = [0.80, 0.15, 0.05]

# calculate the kl divergence
def kl_divergence(p, q):
    return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))

# calculate (P || Q)
kl_pq = kl_divergence(p, q)
print('KL(P || Q): %.3f bits' % kl_pq)
# KL(P || Q): 1.927 bits

# calculate (Q || P)
kl_qp = kl_divergence(q, p)
print('KL(Q || P): %.3f bits' % kl_qp)
# KL(Q || P): 2.022 bits

# They are not equal!
{% endhighlight %}

**NOTE**: KL divergence is always a non-negative value that indicates how close two probability distributions are. Proofs can be found in [here](https://stats.stackexchange.com/a/335201/16534){:target="_blank"}

# WHAT IS A COST (LOSS) FUNCTION?

In simple terms, predictive modeling can be described as the mathematical problem of approximating a mapping function ($f$) from input variables ($X$) to output variables ($y$):  $f: X \rightarrow y$. This is called the problem of function approximation. Stated in other words, the model learns how to take $X$ (i.e., features, or independent variable(s)) in order to predict $y$ (the target, the response or the dependent variable).  

If $y$ is discrete/categorical variable, then this is classification problem. If $y$ is real number/continuous, then this is a regression problem.

The goal is to approximate mapping function as accurately as possible, which consists of some parameters, given the time and resources available.

Once the model learns these parameters, they can be used to compute estimated values of $y$ given new values of $X$. In other words, you can use these learned parameters to predict values of $y$ when you don’t know what $y$ is, i.e., one has a predictive model!

In predictive modeling, cost functions are used to estimate how badly models are performing. Put it simply, a cost function is a measure of how wrong the model is in terms of its ability to estimate the relationship between $X$ and $y$. This is typically expressed as a difference or distance between the predicted value and the actual value. The cost function (you may also see this referred to as loss or error) can be estimated by iteratively running the model to compare estimated predictions against "ground truth", i.e., the known values of y.

The objective here, therefore, is to find parameters, weights/biases or a structure that minimizes the cost function.

# WHAT IS A LOGIT?

A logit (also called a score) is a raw unscaled value associated with a class before computing the probability. In terms of a neural network architecture, this means that a logit is an output of a dense (fully-connected) layer. 

# BINARY CROSS-ENTROPY

Binary cross-entropy (a.k.a. log-loss/logistic loss) is a special case of categorical cross entropy. Withy binary cross entropy, you can classify only two classes, With categorical cross entropy, you are not limited to how many classes your model can classify.

Binary cross entropy formula is as follows:

$$ L(\theta) = - \frac{1}{n} \sum_{i=1}^{n}  \left[y_{i} \log (p_i) + (1-y_{i}) \log (1- p_{i}) \right]$$

where $i$ indexes samples/observations. where $y$ is the label (1 for positive class and 0 for negative class) and p(y) is the predicted probability of the point being positive for all $n$ points. In the simplest case, each $y$ and $p$ is a number, corresponding to a probability of one class.

Reading this formula, it tells you that, for each positive point ($y_{i}=1$), it adds $log(p_{i})$ to the loss, that is, the log probability of it being positive. Conversely, it adds $log(1-p_{i})$, that is, the log probability of it being negative, for each negative point ($y_{i}=0$). 

**How to obtain this formula?**

In a classification problem, we try to find posterior probabilities of classes. in binary classification, we have two outcomes, 0 or 1. Since $P(y^{(i)}=0 \mid \mathbf{x}^{(i)}, \theta) = 1- P(y^{(i)}=1 \mid \mathbf{x}^{(i)}, \theta) $, we can say that $y^{(i)}=1$ with probability $ h_{\theta} ( \mathbf{x}^{(i)} )$ and $y^{(i)}=0$ with probability $1− h_{\theta} ( \mathbf{x}^{(i)} )$.

This can be combined into a single equation as follows because, for binary classification, $y^{(i)}$ follows a Bernoulli distribution:

$$P(y^{(i)} \mid \mathbf{x}^{(i)}, \theta) =  \left[h_{\theta} ( \mathbf{x}^{(i)} )\right]^{y^{(i)}} \times \left(1− h_{\theta} ( \mathbf{x}^{(i)} ) \right)^{1-y^{(i)}}$$

Assuming that the $m$ training examples were generated independently, the likelihood of the training labels, which is the entire dataset $\mathbf{X}$, is the product of the individual data point likelihoods. Thus,

$$ L(\theta) = P(\mathbf{y} \mid \mathbf{X}, \theta) = \prod_{i=1}^{m} L(\theta; y^{(i)} \mid \mathbf{x}^{(i)}) =\prod_{i=1}^{m} P(\mathbf{y} = y^{(i)} \mid \mathbf{X} = \mathbf{x}^{(i)}, \theta) = \prod_{i=1}^{m} \left[h_{\theta} ( \mathbf{x}^{(i)} )\right]^{y^{(i)}} \times \left[1− h_{\theta} ( \mathbf{x}^{(i)} ) \right]^{1-y^{(i)}} $$

Now, Maximum Likelihood principle says that we need to find the parameters that maximise the likelihood $L(\theta)$.

Logarithms are used because they convert products into sums and do not alter the maximization search, as they are monotone increasing functions. Here too we have a product form in the likelihood. So, we take the natural logarithm as maximising the likelihood is same as maximising the log likelihood, so log likelihood $\mathcal{L}(\theta)$ is now:

$$ \mathcal{L}(\theta) = \log L(\theta) =  \sum_{i=1}^{m} y^{(i)} \log(h_{\theta} ( \mathbf{x}^{(i)} )) + (1 - y^{(i)} ) \log(1− h_{\theta} ( \mathbf{x}^{(i)} )) $$

Since in linear regression we found the $\theta$ that minimizes our cost function , here too for the sake of consistency, we would like to have a minimization problem. And we want the average cost over all the data points. Currently, we have a maximimization of $\mathcal{L}(\theta)$ . Maximization of $\mathcal{L}(\theta)$ is equivalent to minimization of $ - \mathcal{L}(\theta)$. And using the average cost over all data points, our cost function for logistic regresion comes out to be:

$$
\begin{align}
J(\theta) &=  - \dfrac{1}{n} \mathcal{L}(\theta)\\
&= - \dfrac{1}{m} \sum_{i=1}^{n} y^{(i)} \log(h_\theta(\mathbf{x}^{(i)})) + (1 - y^{(i)}) \log(1-h_\theta(\mathbf{x}^{(i)}))
\end{align}
$$

As you can see, maximizing the log-likelihood (minimizing the *negative* log-likelihood) is equivalent to minimizing the binary cross entropy. 

Let’s take a closer look at this relationship. The plot below shows the Log Loss contribution from a single positive instance where the predicted probability ranges from 0 (the completely wrong prediction) to 1 (the correct prediction). It’s apparent from the gentle downward slope towards the right that the Log Loss gradually declines as the predicted probability improves. Moving in the opposite direction though, the Log Loss ramps up very rapidly as the predicted probability approaches 0. 

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/log-loss-curve.png?raw=true)

Two properties in particular make it reasonable to interpret the cross-entropy as a cost function. First, it's non-negative, that is, $L(\theta) \geq 0$. To see this, notice that: (a) all the individual terms in the sum in the equation are negative, since both logarithms are of numbers in the range 0 to 1; and (b) there is a minus sign out the front of the sum.
Second, if the neuron's actual output is close to the desired output for all training inputs, $x$, then the cross-entropy will be close to zero. To see this, suppose for example that $y =0$ and $p \approx 0$ for some input $x$. This is a case when the neuron is doing a good job on that input. We see that the first term in the equation for the cost function vanishes, since $y=0$, while the second term is just $−ln(1 -p) \approx 0$. A similar analysis holds when $y=1$ and  $p \approx 1$. And so the contribution to the cost will be low provided the actual output is close to the desired output.

# CATEGORICAL CROSS-ENTROPY

Multi-class cross entropy formula is as follows:

$$ L(\theta) = - \frac{1}{n} \sum_{i=1}^{n}  \sum_{j=1}^{K} \left[y_{ij} \log (p_{ij}) \right]$$

where $i$ indexes samples/observations and $j$ indexes classes. Here, $y_{ij}$ and $p_{ij}$ are expected to be probability distributions over $K$ classes. In a neural network, $y_{ij}$ is one-hot encoded labels and $p_{ij}$ is scaled (softmax) logits. 

When $K=2$, one will get binary cross entropy formula. 

We can also find this formulation easily.

A common example of a Multinoulli distribution in machine learning might be a multi-class classification of a single example into one of $K$ classes.

Namely, suppose you have a model which predicts $K$ classes $\\{1,2, \ldots , K \\}$ and their hypothetical occurance probabilities $p_{1}, p_{2}, \ldots , p_{K}$. Suppose that you have a data point (observation) $i$, and you observe (in reality) $n_{1}$ instances of class 1, $n_{2}$ instances of class 2,..., $n_{K}$ instances of class K. According to your model, the likelihood of this happening is:

$$
P(data \mid model) = p_{1}^{n_{1}} \times p_{2}^{n_{2}} \ldots p_{K}^{n_{K}}
$$

Negative log-likelihood is written as 

$$
- log P(data \mid model) = -n_{1} \log(p_{1}) -n_{2} \log(p_{2})- \ldots -n_{K} \log(p_{K}) = - \sum_{j=1}^{K} n_{j} \log(p_{j})
$$

One can easily see that $n_{1} + n_{2} + \ldots + n_{K} = n$ which is the number of observations in the dataset. Basically, now, you have a multinomial distribution with parameters $n$ (independent trials) and $p_{1}, p_{2}, \ldots , p_{K}$. Empirical probabilities are then computed as $y_{j} = \frac{n_{j}}{p_{j}}$. Therefore, loss for one observation is then computed as:

$$
L(\theta \mid x_{i}) = - \sum_{j=1}^{K} n_{j} \log(p_{j})
$$

If we compute the cross-entropy over $n$ observations, we will have:

$$ L(\theta) = - \frac{1}{n} \sum_{i=1}^{n}  \sum_{j=1}^{K} \left[y_{ij} \log (p_{ij}) \right]$$

# TENSORFLOW IMPLEMENTATIONS

Tensorflow has many built-in Cross Entropy functions.

## Sigmoid functions family

* [tf.nn.sigmoid_cross_entropy_with_logits](https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits){:target="_blank"}
* [tf.nn.weighted_cross_entropy_with_logits](https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits){:target="_blank"}
* [tf.losses.sigmoid_cross_entropy](https://www.tensorflow.org/api_docs/python/tf/losses/sigmoid_cross_entropy){:target="_blank"}

Sigmoid loss function is for binary classification. But tensorflow functions are more general and allow to do multi-label classification, when the classes are independent. In other words, `tf.nn.sigmoid_cross_entropy_with_logits` solves $N$ binary classifications at once. 

The labels must be one-hot encoded or can contain soft class probabilities.

`tf.losses.sigmoid_cross_entropy` in addition allows to set the in-batch weights, i.e. make some examples more important than others. `tf.nn.weighted_cross_entropy_with_logits` allows to set class weights (remember, the classification is binary), i.e. make positive errors larger than negative errors. This is useful when the training data is unbalanced.

## Softmax functions family

* [tf.nn.softmax_cross_entropy_with_logits](https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits){:target="_blank"} (DEPRECATED IN 1.5)
* [tf.nn.softmax_cross_entropy_with_logits_v2](https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits_v2){:target="_blank"}
* [tf.losses.softmax_cross_entropy](https://www.tensorflow.org/api_docs/python/tf/losses/softmax_cross_entropy){:target="_blank"}

These loss functions should be used for multinomial mutually exclusive classification, i.e. pick one out of $N$ classes. Also applicable when $N = 2$.

The labels must be one-hot encoded or can contain soft class probabilities: a particular example can belong to class A with 70% probability and class B with 30% probability. 

Just like in sigmoid family, `tf.losses.softmax_cross_entropy` allows to set the in-batch weights, i.e. make some examples more important than others. 

## Sparse functions family

* [tf.nn.sparse_softmax_cross_entropy_with_logits](https://www.tensorflow.org/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits){:target="_blank"}
* [tf.losses.sparse_softmax_cross_entropy](https://www.tensorflow.org/api_docs/python/tf/losses/sparse_softmax_cross_entropy){:target="_blank"}

Like ordinary softmax above, these loss functions should be used for multinomial mutually exclusive classification, i.e. pick one out of $N$ classes. The difference is in labels encoding: the classes are specified as integers (class index), not one-hot vectors. Obviously, this doesn't allow soft classes, but it can save some memory when there are thousands or millions of classes. However, note that logits argument must still contain logits per each class, thus it consumes at least `[batch_size, classes]` memory.

Like above, `tf.losses` version has a `weights` argument which allows to set the in-batch weights.

Like above, labels are not one-hot encoded, but have the shape `[batch_size, num_true]`.

## Sampled softmax functions family

* [tf.nn.sampled_softmax_loss](https://www.tensorflow.org/api_docs/python/tf/nn/sampled_softmax_loss){:target="_blank"}
* [tf.contrib.nn.rank_sampled_softmax_loss](https://www.tensorflow.org/api_docs/python/tf/contrib/nn/rank_sampled_softmax_loss){:target="_blank"}
* [tf.nn.nce_loss](https://www.tensorflow.org/api_docs/python/tf/nn/nce_loss){:target="_blank"}

These functions provide another alternative for dealing with huge number of classes. Instead of computing and comparing an exact probability distribution, they compute a loss estimate from a random sample.

The arguments `weights` and `biases` specify a separate fully-connected layer that is used to compute the logits for a chosen sample.

Sampled functions are only suitable for training. In test time, it's recommended to use a standard softmax loss (either sparse or one-hot) to get an actual distribution.

Another alternative loss is `tf.nn.nce_loss`, which performs noise-contrastive estimation. NCE guarantees approximation to softmax in the limit.

**NOTE**: See [below](#difference-between-tfnnsoftmax_cross_entropy_with_logits-and-tfnnsparse_softmax_cross_entropy_with_logits) for the difference between `tf.nn` and `tf.loses`.

# DIFFERENCE BETWEEN OBJECTIVE FUNCTION, COST FUNCTION AND LOSS FUNCTION
From Section 4.3 in "Deep Learning" by Ian Goodfellow, Yoshua Bengio, Aaaron Courville:
![Placeholder image](https://raw.githubusercontent.com/mmuratarat/mmuratarat.github.io/master/assets/DL_CE.jpeg "Image with caption")

In this book, at least, loss and cost are the same.

In Andrew NG's words:
<blockquote>Finally, the loss function was defined with respect to a single training example. It measures how well you're doing on a single training example. I'm now going to define something called the cost function, which measures how well you're doing an entire training set. So the cost function J which is applied to your parameters W and B is going to be the average with one of the m of the sum of the loss function applied to each of the training examples and turn.</blockquote>

The terms *cost function* and *loss function* are synonymous, some people also call it *error function*. 

However, there are also some different definitions out there. The loss function computes the error for a single training example, while the cost function will be average over all data points.

# HOW TO COMPUTE CROSS ENTROPY FOR BINARY CLASSIFICATION?
<script src="https://gist.github.com/mmuratarat/3db39c59e0436ec4768f27a3ad524808.js"></script>

# HOW TO COMPUTE CROSS ENTROPY FOR MULTICLASS CLASSIFICATION?
<script src="https://gist.github.com/mmuratarat/b7469a36d88fa88056b8511d8b1aac26.js"></script>

# DIFFERENCE BETWEEN tf.nn.softmax_cross_entropy_with_logits AND tf.nn.sparse_softmax_cross_entropy_with_logits

The function arguments for `tf.losses.softmax_cross_entropy` and `tf.losses.sparse_softmax_cross_entropy` are different, however, they produce the same result. 

The difference is simple:

* For `sparse_softmax_cross_entropy_with_logits`, labels must have the shape `[batch_size]` and the dtype is int32 or int64. Each label is an integer in range `[0, num_classes-1]`.
* For `softmax_cross_entropy_with_logits`, labels must have the shape `[batch_size, num_classes]` and dtype is float32 or float64.

Labels used in `softmax_cross_entropy_with_logits` are the one hot version of labels used in `sparse_softmax_cross_entropy_with_logits`.

**NOTE:** `tf.losses.softmax_cross_entropy` creates a cross-entropy loss using `tf.nn.softmax_cross_entropy_with_logits_v2`. Similarly, `tf.losses.sparse_softmax_cross_entropy` creates cross-entropy loss using `tf.nn.sparse_softmax_cross_entropy_with_logits`. Convenience is that using `tf.nn.softmax_cross_entropy_with_logits_v2` or `tf.nn.sparse_softmax_cross_entropy_with_logits`, one can calculate individual entropy values and then using `tf.reduce_mean`, the average of the entire training set can be found.

<script src="https://gist.github.com/mmuratarat/f295d1017bcbb54c2f9ac5cd6d9f762d.js"></script>

# LINKS
1. [https://scikit-learn.org/stable/modules/multiclass.html](https://scikit-learn.org/stable/modules/multiclass.html){:target="_blank"}
2. [https://stackoverflow.com/a/47034889/1757224](https://scikit-learn.org/stable/modules/multiclass.html){:target="_blank"}
3. [https://stats.stackexchange.com/a/327396/16534](https://stats.stackexchange.com/a/327396/16534){:target="_blank"}
4. [https://stackoverflow.com/a/48317496/1757224](https://stackoverflow.com/a/48317496/1757224){:target="_blank"}
5. [https://scikit-learn.org/stable/modules/multiclass.html](https://scikit-learn.org/stable/modules/multiclass.html){:target="_blank"}
6. [https://chrisyeh96.github.io/2018/06/11/logistic-regression.html](https://chrisyeh96.github.io/2018/06/11/logistic-regression.html){:target="_blank"}
7. [https://stats.stackexchange.com/questions/327348/how-is-softmax-cross-entropy-with-logits-different-from-softmax-cross-entropy-wi](https://stats.stackexchange.com/questions/327348/how-is-softmax-cross-entropy-with-logits-different-from-softmax-cross-entropy-wi){:target="_blank"}
8. [https://stackoverflow.com/questions/47034888/how-to-choose-cross-entropy-loss-in-tensorflow](https://stackoverflow.com/questions/47034888/how-to-choose-cross-entropy-loss-in-tensorflow){:target="_blank"}
9. [https://stats.stackexchange.com/questions/260505/machine-learning-should-i-use-a-categorical-cross-entropy-or-binary-cross-entro](https://stats.stackexchange.com/questions/260505/machine-learning-should-i-use-a-categorical-cross-entropy-or-binary-cross-entro){:target="_blank"}
10. [https://stackoverflow.com/questions/49044398/is-there-any-difference-between-cross-entropy-loss-and-logistic-loss](https://stackoverflow.com/questions/49044398/is-there-any-difference-between-cross-entropy-loss-and-logistic-loss){:target="_blank"}
11. [https://stackoverflow.com/a/37317322/1757224](https://stackoverflow.com/a/37317322/1757224){:target="_blank"}
12. [https://datascience.stackexchange.com/a/9408/54046](https://datascience.stackexchange.com/a/9408/54046){:target="_blank"}
13. [https://stats.stackexchange.com/a/215484/16534](https://stats.stackexchange.com/a/215484/16534){:target="_blank"}
14. [https://stats.stackexchange.com/a/215495/16534](https://stats.stackexchange.com/a/215495/16534){:target="_blank"}
15. [http://neuralnetworksanddeeplearning.com/chap3.html](http://neuralnetworksanddeeplearning.com/chap3.html){:target="_blank"}
16. [https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a](https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a){:target="_blank"}
17. [https://math.stackexchange.com/a/1672834/45210](https://math.stackexchange.com/a/1672834/45210){:target="_blank"}
18. [http://www.awebb.info/blog/cross_entropy](http://www.awebb.info/blog/cross_entropy){:target="_blank"}
