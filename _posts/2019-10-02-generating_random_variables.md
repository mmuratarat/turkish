---
layout: post
title: "Generating random variables"
author: "MMA"
comments: true
---
# Transformation of Random Variables

Let's consider how to take the transformation of a random variable $X$ with cumulative distribution function $F_{X}(x)$. Let $Y=t(X)$, that is, $Y$ is the transformation of $X$ via function $t(\cdot)$.

In order to get the CDF of $Y$ we use the definition of CDFs:

$$
F_{Y}(y) = P(Y \leq y) = P(t(X) \leq y)
$$

We have $F_{X}(x)$ and want to know how to compute $F_{Y}(y)$ in terms of $F_{X}(x)$. To get there we can take the inverse of $t(x)$ on both sides of the inequality:

$$
F_{Y}(y) = P(Y \leq y) = P(t(X) \leq y) = P(X \leq t^{-1}(y))
$$

This is the CDF of $X$:

$$
P(X \leq t^{-1}(y)) = F_{X}(t^{-1}(y))
$$

and that's how we get $F_{Y}(y)$ in terms of $F_{X}(x)$. We can compute the density function $f_{Y}(y)$ by differentiating $F_{Y}(y)$, applying the chain rule:

$$
f_{Y}(y) = f_{y}(t^{-1}(y)) \times \frac{d}{dy} t^{-1}(y) dy
$$

Note that it is only this simple if $t(x)$ is one-to-one and strictly monotone increasing; it gets more complicated to reason about the regions where $Y$ is defined otherwise.

How do we use this result?

Let $U \sim U(0, 1)$. Then $F(X) = U$ means that the random variable $F^{−1}(U)$ has the same distribution as $X$.

# Inverse transform sampling

It is a basic method for pseudo-random number sampling, i.e. for generating sample numbers at random from any probability distribution given its cumulative distribution function. The basic principle is to find the inverse function of $F$, $F^{-1}$ such that $F~ F^{-1} = F^{-1} ~ F = I$.

The problem that the inverse transform sampling method solves is as follows:

* Let $X$ be a random variable whose distribution can be described by the cumulative distribution function $F_{X}$.
* We want to generate values of $X$ which are distributed according to this distribution.

The inverse transform sampling method works as follows:

* Generate a random number $u$ from the standard uniform distribution in the interval $[0,1]$, e.g. from $U\sim Unif [0,1]$.
* Find the inverse of the desired CDF, e.g. $F_{X}^{-1}(x)$. Inverse cumulative distribution function is also called quantile function.
* Compute $x = F_{X}^{-1}(u)$ (Solve the equation $F_{X}(x) = u$ for $X$). The computed random variable $X$ has distribution $F_{X}(x)$.

Expressed differently, given a continuous uniform variable $U$ in $[0,1]$ and an invertible cumulative distribution function $F_{X}$, the random variable $X=F_{X}^{-1}(U)$ has distribution $F_{X}$ (or, $X$ is distributed $F_{X}$).

$$
\begin{split}
F_{X}(x) = P(X \leq x) &= P(F_{X}^{-1}(U)\leq x)\\
&=P(U \leq F_{X}(x))\\
&= F_{U}(F_{X}(x))\\
&= F_{X}(x) 
\end{split}
$$

Remember that the cumulative distribution function of continuous uniform distribution on the interval $[0,1]$ is $F_{U}(u)=u
$.

Computationally, this method involves computing the quantile function of the distribution — in other words, computing the cumulative distribution function (CDF) of the distribution (which maps a number in the domain to a probability between 0 and 1) and then inverting that function many times. This is the source of the term "inverse" or "inversion" in most of the names for this method. Note that for a discrete distribution, computing the CDF is not in general too difficult: we simply add up the individual probabilities for the various points of the distribution. For a continuous distribution, however, we need to integrate the probability density function (PDF) of the distribution, which is impossible to do analytically for most distributions (including the normal distribution). As a result, this method may be computationally inefficient for many distributions and other methods are preferred; however, it is a useful method for building more generally applicable samplers such as those based on rejection sampling.

For the normal distribution, the lack of an analytical expression for the corresponding quantile function means that other methods (e.g. the Box–Muller transform) may be preferred computationally. It is often the case that, even for simple distributions, the inverse transform sampling method can be improved on.

(Note: technically this only works when the CDF has a closed form inverse function)

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Screen%20Shot%202019-10-06%20at%2019.45.52.png?raw=true)

# Continuous Example: Exponential Distribution

The exponential distribution has CDF:

$$
F_X(x) = 1 - e^{-\lambda x}
$$

for $x \geq 0$ (and $0$ otherwise). By solving $u=F(x)$ we obtain the inverse function

$$
\begin{split}
1 - e^{-\lambda x} &= u\\
x &= \frac{-1}{\lambda}ln(1 - y)
\end{split}
$$

so

$$
F^{-1}_X(x) = \frac{-1}{\lambda}ln(1 - u)
$$

It means that if we draw some $u$ from $U \sim Unif(0,1)$ and compute $x = F^{-1}_X(x) = \frac{-1}{\lambda}ln(1 - u)$, this $X$ has exponential distribution.

Note that in practice, since both $u$ AND $1-u$ are uniformly distributed random number, so the calculation can be simplified as:

$$
x = F^{-1}_X(x) = \frac{-1}{\lambda}ln(u)
$$

{% highlight python %} 
import math
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

def inverse_exp_dist(lmbda=1.0):
    return (-1 / lmbda)*math.log(1 - np.random.random())

plt.hist([inverse_exp_dist() for i in range(10000)], 50)
plt.title('Samples from an exponential function')
plt.savefig('inverse_pdf_exp_dist')
plt.show()
{% endhighlight %}

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/inverse_pdf_exp_dist.png?raw=true)

and just to make sure this looks right, let's use numpy's exponential function and compare:

{% highlight python %} 
plt.hist([np.random.exponential() for i in range(10000)], 50)   
plt.title('Samples from numpy.random.exponential')
plt.savefig('numpy_random_exponential_dist')
plt.show()
{% endhighlight %}

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/numpy_random_exponential_dist.png?raw=true)

# Functions with no inverses

In general, there are no inverses for functions that can return same value for different inputs, for example density functions (e.g., the standard normal density function is symmetric, so it returns the same values for −2 and 2 etc.). The normal distribution is an interesting example for one more reason—it is one of the examples of cumulative distribution functions that do not have a closed-form inverse. Not every cumulative distribution function has to have a closed-form inverse! Therefore,  the inverse transform method is not efficient. Hopefully in such cases the inverses can be found using numerical methods.

# Normal Distribution

There's no closed form expression for the inverse cdf of a normal distributio (a.k.a. the quantile function of a normal distribution). This is often a problem with the inverse transform method. There are various ways to express the function and numerous approximations.

Let's think of a standard normal distribution. The drawback of using inverse CDF method is that it relies on calculation of the probit function $\Phi^{-1}$, which cannot be done analytically (Note that In probability theory and statistics, the probit function is the quantile function associated with the standard normal distribution, which is commonly denoted as N(0,1)). Some approximate methods are described in the literature. One of the easiest way is to do a table lookup. E.g., If $U = 0.975$, then $Z = \Phi^{-1}(U) = 1.96$ because z-table gives $\Phi(Z)$.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/z_table.png?raw=true)

If we are willing to accept numeric solution, inverse functions can be found. One of the inverse c.d.f. of the standard normal distribution was proposed by Schmeiser:

$$
Z = \Phi^{-1}(U) \approx \frac{U^{0.135} - (1 - U)^{0.135}} {0.1975}
$$

for $0.0013499 \le U \le 0.9986501$ which matches the true normal distribution with one digit after decimal point. 

There is one another approximation. The following approximation has absolute error $\leq 0.45 \times 10^{−3}$:

$$
Z = sign(U − 1/2) \left(t - \frac{c_{0} + c_{1}t + c_{2} t^{2}}{1 + d_{1}t + d_{2} t^{2} + d_{3}t^{3}} \right)
$$

where sign(x) = 1, 0, −1 if $X$ is positive, zero, or negative, respectively,

$$
t = \left\{- \ln \left[min (U, 1-U) \right]^{2} \right\}^{1/2}
$$

and $c_{0} = 2.515517, c_{1} = 0.802853, c_{2} = 0.010328, d_{1} = 1.432788, d_{2} = 0.189269, d_{3} = 0.001308$.

In any case, rather than sampling x directly, we could instead sample $Z \sim N(0, 1)$ and transform samples of $Z$ into samples of $X$. If $Z \sim N(0, 1)$ and you want $X \sim N(\mu, \sigma^{2})$, just take $X \leftarrow \mu + \sigma Z$. Suppose you want to generate $X \sim N(3, 16)$, and you start with $U = 0.59$. Then,

$$
X = \mu + \sigma Z = 3 + 4 \Phi^{-1}(0.59) = 3 + 4(0.2275) = 3.91
$$

because $\Phi^{-1}(0.59) = Z \rightarrow \Phi(Z) = P(Z \leq Z) = 0.59$. What is this $Z$? Using a [online calculator](https://stattrek.com/online-calculator/normal.aspx){:target="_blank"}, it is $0.2275$.

Let's see an example in Python. 

{% highlight python %} 
n = 10000  # Samples to draw
mean = 3
variance = 16
Z = np.random.normal(loc=0, scale=1.0, size=(n,))
X = mean + (np.sqrt(variance) * Z)

print(np.mean(X))
#3.0017206638273097

print(np.std(X))
#4.022342597707669

count, bins, ignored = plt.hist(X, 30, normed=True)
plt.plot(bins, univariate_normal(bins, mean, variance),linewidth=2, color='r')
plt.savefig('generated_normal_dist.png')
plt.show()
{% endhighlight %}

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/generated_normal_dist.png?raw=true)

### The Box–Muller method 
Now let's consider a more direct and exact transformation. Let $Z_1, Z_2$ be two standard normal random variates. Plot the two as a point in the plane and represent them in a polar coordinate system as $Z_1 = B \cos \theta$ and $Z_2 = B \sin \theta$.

It is known that $B^2 = {Z_1}^2 + {Z_2}^2$ has the chi-square distribution with 2 degrees of freedom, which is equivalent to an exponential distribution with mean 2 (this comes from the fact that if one has $k$ i.i.d normal random variables where $X_i\sim N(0,\sigma^2)$, sum of squares of those random variables, $X_1^2+X_2^2+\dots+X_k^2\sim\sigma^2\chi^2_k$):

$$
Y = \lambda e^{-\lambda t},\,\,\,\,\, t \geq 0
$$

where $E[Y] = 2 = \lambda$. Thus, the raidus $B$ can be generated using $B = \sqrt{-2\ln U}$. 

Note that here, we use alternative formulation of Exponential distribution, where:

$$
f(x) = \frac{1} {\lambda} e^{-x/\lambda},\,\,\,\,\, x \geq 0; \lambda > 0
$$

with mean $E(X) = \lambda$ and variance $Var(X)=\lambda^{2}$

$$
F(x) = 1 - e^{-x/\lambda},\,\,\,\,\, x \ge 0; \lambda > 0
$$

So, the formula for inverse of CDF (quantile function or the percent point function) of the exponential distribution is

$$
F^{-1}_{X}(x) = -\lambda\ln(1 - u)
$$

Again, in practice, since both $u$ AND $1-u$ are uniformly distributed random number.

So a standard normal distribution can be generated by any one of the following.

$$
Z_1 = \sqrt{-2 \ln U_1} \cos (2\pi U_2)
$$

and

$$
Z_2 = \sqrt{-2 \ln U_1} \sin (2\pi U_2)
$$

where $U_1$ and $U_2$ are uniformly distributed over $(0,1)$ and they will be independent. In order to obtain normal variates $X_i$ with mean $\mu$ and variance $\sigma^2$, transform $X_i = \mu + \sigma Z_i$.

```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')

# uniformly distributed values between 0 and 1
u1 = np.random.rand(1000)
u2 = np.random.rand(1000)

# transformation function
def box_muller(u1,u2):
    z1 = np.sqrt(-2*np.log(u1))*np.cos(2*np.pi*u2)
    z2 = np.sqrt(-2*np.log(u1))*np.sin(2*np.pi*u2)
    return z1,z2

# Run the transformation
z1 = box_muller(u1, u2)
z2 = box_muller(u1, u2)

# plotting the values before and after the transformation
plt.figure(figsize = (20, 10))
plt.subplot(221) # the first row of graphs
plt.hist(u1)     # contains the histograms of u1 and u2 
plt.subplot(222)
plt.hist(u2)
plt.subplot(223) # the second contains
plt.hist(z1)     # the histograms of z1 and z2
plt.subplot(224)
plt.hist(z2)
plt.show()
```

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/box_muller.png?raw=true)

The Box-Muller method is not the fastest way to generate $N(0, 1)$ random variables, and numerical computing environments don't always use it. There is some cost in computing cos, sin, log and sqrt that, with clever programming can be avoided. Box-Muller remains very popular because it is simple to use.

There is also The Marsaglia polar method which is a modification of the Box–Muller method which does not require computation of the sine and cosine functions

However, in this computer age, most statistical software would provide you with quantile function for normal distribution already implemented. The inverse of the normal CDF is know and given by:

$$
F^{-1}(Z)\; =\; \sqrt2\;\operatorname{erf}^{-1}(2Z - 1), \quad Z\in(0,1).
$$

Hence:

$$
Z = F^{-1}(U)\; =\; \sqrt2\;\operatorname{erf}^{-1}(2U - 1), \quad U\in(0,1)
$$

where erf is error function.

# Characterization method (Convolution Method)

This method is another approach to sample from a distribution. In some cases, $X$ can be expressed as a sum of independent random variables $Y_{1}, Y_{2}, \ldots , Y_{n}$ where $Y_{j}$'s are iid and n is fixed and finite:

$$
X = Y_{1} + Y_{2} + \ldots + Y_{n}
$$

called n-fold convolution of distribution $Y_{j}$. Here, $Y_{j}$'s are generated more easilt.

**Algorithm**:
* Generate independent $Y_{1}, Y_{2}, \ldots , Y_{n}$ each with distribution function $F_{Y}(y)$ using the inverse transform method.
* Return $X = Y_{1} + Y_{2} + \ldots + Y_{n}$.

For example, an Erlang random variable $X$ with parameters $(n, \lambda)$ can be shown to be the sum of $n$ independent exponential random variables $Y_{i}, i=1,2, \ldots ,n$, each having a mean of $\frac{1}{\lambda}$.

$$
X = \sum_{i=1}^{n} Y_{i}
$$

Using inverse CDF method that can generate an exponential variable, an Erlang variate can be generated:

$$
X = \sum_{i=1}^{n}  \frac{-1}{\lambda}ln(u_{i}) = \frac{-1}{\lambda} ln \left(\prod_{i=1}^{n} u_{i} \right)
$$

Other examples:

* If $X_{1}, \ldots , X_{n}$ are i.i.d. Geometric(p), then $\sum_{i=1}^{n} X_{i} \sim NegBin(n, p)$
* If $X_{1}, \ldots , X_{n}$ are i.i.d. Normal(0,1), then $\sum_{i=1}^{n} X_{i} \sim \chi_{n}^{2}$
* If $X_{1}, \ldots , X_{n}$ are i.i.d. Bernoulli(p), then $\sum_{i=1}^{n} X_{i} \sim Binomial(n, p)$

# Composition Method

This method applies when the distribution function $F$ can be expressed as a mixture of other distribution functions $F_{1}, F_{2}, \ldots$:

$$
F(x) = \sum_{j=1}^{\infty} p_{j}F_{j}(x),
$$

where $p_{j} \geq 0$ and $\sum_{j=1}^{\infty} p_{j} =1$, meaning that the $p_{j}$ form a discrete probability distribution

Equivalently, we can decompose the density function $f(x)$ or mass function $p(x)$ into convex combination of other density or mass functions. This method is useful if it is easier to sample from $F_{j}$'s than from $F$.

**Algorithm**:
* Generate a discrete random variable $j$ such that $P(J = j) = p_{j}$.
* Return $X$ with CDF $F_{J}(x)$ (given $J=j$, $x$ is generated independent of $J$).

For fixed $x$:
$$
\begin{split}
P(X \leq x) &= \sum_{j} P(X \leq x \mid J = j)P(J = j) \text{ (condition on } J=j\text{)}\\
&= \sum_{j} P(X \leq x \mid J = j)p_{j}\text{ (distribution of J)}\\
&= \sum_{j} F_{j}(x)p_{j} \text{ (given } J = j, X \sim F_{j}\text{)}\\
&=F_{X}(x) \text{ (decomposition of F)}
\end{split}
$$

The trick is to find $F_{j}$’s from which generation is easy and fast.

# Acceptance-Rejection Method

The majority of CDFs cannot be inverted efficiently. In other words, finding an explicit formula for $F^{−1}(y)$ for the cdf of a random variable $X$ we wish to generate, $F(x) = P(X \leq x)$, is not always possible. Moreover, even if it is, there may be alternative methods for generating a random variable, distributed as $F$ that is more efficient than the inverse transform method or other methods we have come across.

Rejection Sampling is one of the simplest sampling algorithm.

We start by assuming that the $F$ we wish to simulate from has a probability density function $f(x)$ (we cannot easily sample from); that can be either continuous or discrete distribution. 

The basic idea is to find an alternative probability distribution $G$, with density function $g(x)$ (like a Normal distribution or perhaps a  t-distribution), from which we already have an efficient algorithm for generating from (because there’s a built in function or someone else wrote a nice function), but also such that the function $g(x)$ is "close" to $f(x)$. In other words, we assume there is another density $g(x)$ and a constant $c$ such that $f(x) \leq cg(x)$. Then we can sample from $g$ directly and then "reject" the samples in a strategic way to make the resulting "non-rejected" samples look like they came from $f$. The density $g$ will be referred to as the "candidate density" and $f$ will be the "target density".

In particular, we assume that the ratio $f(x)/g(x)$ is bounded by a constant $c > 0$; $sup_{x}\{f(x)/g(x)\} \leq c$. (And
in practice we would want $c$ as close to 1 as possible). The easiest way to satisfy this assumption is to make sure that  
$g$ has heavier tails than $f$. We cannot have that $g$ decreases at a faster rate than $f$ in the tails or else rejection sampling will not work.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/acception_rejection_algorithm.png?raw=true)

Here is the rejection sampling algorithm for drawing a sample from the target density $f$ is then:

1. Generate a random variable $Y$, distributed as $g$.
2. Generate $U \sim Uniform(0, 1)$ (independent from $Y$).
3. If
  $$
  U\leq\frac{f(Y)}{c\,g(Y)}
  $$
  then set $X = Y$ (*accept*) ; otherwise go back to 1 (*reject*).

The algorithm can be repeated until the desired number of samples from the target density $f$ has been accepted.

Some notes:
* $f(Y)$ and $g(Y)$ are random variables, hence, so is the ratio $\frac{f(Y)}{c\,g(Y)}$ and this ratio is independent of $U$ in Step (2).
* The ratio is bounded between 0 and 1; $0 < \frac{f(Y)}{c\,g(Y)} \leq 1$.
* The number of times $N$ that steps 1 and 2 need to be called (e.g., the number of iterations needed to successfully generate X ) is itself a random variable and has a geometric distribution with "success" probability $p = P(U\leq\frac{f(Y)}{c\,g(Y)})$. $P(N = n) = (1−p)^{n−1} p, \,\,\, n \geq 1$. Thus on average the number of iterations required is given by $E(N) = \frac{1}{p}$.
* In the end we obtain our $X$ as having the conditional distribution of a $Y$ given that the event $U \leq \frac{f(Y)}{cg(Y)}$ occurs.

A direct calculation yields that $p = \frac{1}{c}$, by first conditioning on Y, $P(U\leq\frac{f(Y)}{c\,g(Y)} \mid Y = y) = \frac{f(y)}{c\,g(y)}$, thus, unconditioning and recalling that $Y$ has density $g(y)$ yields

$$
\begin{split}
p &= \int_{- \infty}^{+ \infty} \frac{f(y)}{c\,g(y)} \times g(y) \times dy\\
&= \frac{1}{c} \int_{- \infty}^{+ \infty} f(y)d(y)\\
&= \frac{1}{c}
\end{split}
$$

where the last equality follows since f is a density function (hence by definition integrates to 1).  Thus $E(N) = c$, the bounding constant, and we can now indeed see that it is desirable to choose our alternative density g so as to minimize this constant $c = sup_{x}\{f(x)/g(x)\}$. Of course the optimal function would be $g(x) = f(x)$ which is not what we have in mind since the whole point is to choose a different (easy to simulate) alternative from $f$. In short, it is a bit of an art to find an appropriate $g$.

There are two main problems with this method. The first major problem is that if distributions are chosen poorly, like if $f(x)$ is not remotely related to $g(x)$, a lot of samples may be generated and tossed away, wasting computation cycles (as an example, if the enveloping function $cg(x)$ is considerably higher than $f(x)$ at all points, the algorithm will reject most attempted draws, which implies that an incredible number of draws may need to be made before finding a single value from $f(x)$). It may also be difficult to find an envelope with values that are greater at all points of support for the density of interest. Consider trying to use a uniform density as an envelope for sampling from a normal density. The domain of $x$ for the normal density runs from $-\infty$ to $+ \infty$, but there is no corresponding uniform density. In the limit, a $U(-\infty, +\infty)$ density would have an infinitely low height, which would make $g(x)$ fall below $f(x)$ in the center of the distribution, regardless of the constant multiple $c$ chosen. Another trouble is that a lot of samples may be taken in a specific area, getting us a lot of unwanted samples. The choices of $c$ and $g$ affect the computational efficiency of the algorithm. In the case of multidimensional random vectors, we have high chance of running straight into the curse of dimensionality, where chances are corners and edges of our multidimensional density simply don't get the coverage we were hoping for.

For example, let's try to simulate random normal variates using Gamma distribution. Let the target distribution, $f(x)$ be a normal distribution with a mean of 4.5 and a standard deviation of 1. Let's choose a candidate distribution as Gamma distribution with a mean of 4 and a standard deviation 2, which results in parameters shape = 4 and scale = 1 (There is no particular reason to use the gamma distribution here – it was chosen primarily to distinguish it from the normal target distribution). Though theoretically the normal distribution extends from $-\infty$ to $\infty$ and the gamma distribution
extends from $0$ to $\infty$, it is reasonable here to only consider values of $x$ between $0$ and $13$. We choose $c = 3$ here to blanket over the target distribution. The target distribution, candidate distribution and blanket distribution (also known as envelope function) were shown below:

```python
import numpy as np
from scipy.stats import gamma, norm
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')
import seaborn as sns

mu = 4.5; # Mean for normal distribution
sigma = 1; # standard deviation for normal distribution
shape = 4 # Shape parameter for gamma distributiom
scale = 1 # Rate parameter for gamma distribution

# Choose value for c
c = 3

# Define x axis
x = np.arange(start = 0, stop = 12, step = 0.01)
plt.figure(num=1, figsize = (20, 10))
# Plot target distribution
plt.plot(x, norm.pdf(x, loc=mu, scale=sigma), lw=2, label=' Target Distribution - Normal Distribution')
# Plot candidate distribution
plt.plot(x, gamma.pdf(x, a = shape, loc=0, scale=scale), lw=2, label = 'Candidate Distribution - Gamma Distribution')
# Plot the blanket function
plt.plot(x, c * gamma.pdf(x, a = shape, loc=0, scale=scale), lw=2, label = 'Blanket function - c * Gamma Distribution')
plt.xlabel("x")
plt.ylabel("PDF")
plt.legend(loc="upper right")
plt.savefig('target_candidate_blanket_dists.png')
plt.show()
```

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/target_candidate_blanket_dists.png?raw=true)

Having verified that the blanket function satisfies $f(x) \leq cg(x)$, the sampling process can begin.

```python
# Choose number of values desired (this many values will be accepted)
N = 20000;

accept = []
reject = []
i = 0
j = 0
while i < N:
    Y = gamma.rvs(a =shape, loc=0, scale=scale, random_state=None)
    U = np.random.rand(1)
    if U * c * gamma.pdf(Y, a = shape, loc=0, scale=scale) <= norm.pdf(Y, loc=mu, scale=sigma):
        accept.append(Y)
        i += 1
    else:
        reject.append(Y)
        j += 1
    
#PLOT RESULTS OF SAMPLING
plt.figure(num = 2, figsize = (20, 10))
plt.plot(x, norm.pdf(x, loc=mu, scale=sigma), lw=2, label=' Target Distribution - Normal Distribution')
plt.hist(accept, bins = 40, density=True)
plt.savefig('accept_reject_algo_example.png')
plt.show()
```

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/accept_reject_algo_example.png?raw=true)

This sampling method is performed until 20,000 values of $x$ were accepted (40295 generated values are rejected). By inspection, the histogram of values sampled from $f(x)$ reproduce $f(x)$ demonstrating that the rejection method successfully drew random samples from the target distribution $f$. The mean and standard deviation of sampled data points are given below:

```python
print(np.mean(accept))
#4.50210853818239
print(np.std(accept))
#0.9957279285265107
```

# Importance Sampling

With accept-reject sampling, we ultimately obtain a sample from a target density $f$. With that sample, we can create any number of summaries, statistics or visualizations. However, what if we are interested in more narrow problem of computing a mean, such as $E_{f} \left[h(X)  \right]$ for some function $h: R^{k} \to R$. 

Clearly this is a problem that can be solved with rejection sampling. First obtain a sample $x_{1}, x_{2}, \dots, x_{n} \sim f$ then compute:

$$
\hat{\mu_{n}}= \frac{1}{n} \sum_{i=1}^n h(x_i)
$$

with the obtained sample. As $n\rightarrow\infty$ we know by the Law of Large Numbers that $\hat{\mu_{n}} \rightarrow E_{f}[h(X)]$. Further, the Central Limit Theorem gives us $\sqrt{n}(\hat{\mu_{n}} - E_{f} [h(X)])\longrightarrow\mathcal{N}(0,\sigma^2)$.

However, with rejection sampling, in order to obtain a sample of size $n$, we must generate, on average, $c \times n$ candidates from $g$ (the candidate density) and then reject about $(c-1) \times n$ of them. If  $c \approx 1$ then this will not be too inefficient. But in general, if $c$ is much larger than 1, then we will be generating a lot of candidates from $g$ and ultimately throwing most of them away.

It’s worth noting that in most cases, the candidates generated from $g$ fall within the domain of $f$, so that they are in fact values that could plausibly come from $f$. They are simply over- or under-represented in the frequency with which they appear. For example, if $g$ has heavier tails than $f$, then there will be too many extreme values generated from $g$. Rejection sampling simply thins out those extreme values to obtain the right proportion. But what if we could take those rejected values and, instead of discarding them, simply downweight or upweight them in a specific way?

We can rewrite the target estimation as follows,

$$
E_{f}[h(X)] =  E_{g}\left[\frac{f(X)}{g(X)}h(X)\right]
$$

Hence, if $x_{1}, x_{2}, \dots, x_{n} \sim g$ drawn from the candidate density, we can say

$$
\tilde{\mu_{n}} = \frac{1}{n}\sum_{i=1}^{n} \frac{f(x_{i})}{g(x_{i})}h(x_{i}) = \frac{1}{n}\sum_{i=1}^{n} w_{i} h(x_{i}) \approx E_{f} [h(X)]
$$

where $w_{i} = \frac{f(x_{i})}{g(x_{i})}$ are referred as the importance weights (or importance correction) because they take each of the candidates $x_{i}$ generated from $g$ and reweight them when taking the average.  Note that if $f = g$, so that we are simply sampling from the target density, then this estimator is just the sample mean of the $h(x_{i})$'s. The estimator $\tilde{\mu_{n}}$ is known as the importance sampling estimator.

For estimating expectations, one might reasonably believe that the importance sampling approach is more efficient than the rejection sampling approach because it does not discard any data.

Just like accept-reject algorithm, for importance sampling, it does make clear now that the choice of the distribution from which to draw random variables will affect the quality of Monte-Carlo estimator.

In summary, a good importance sampling function, $g(x)$ should have the following properties:

* $g(x) > 0$ whenever $f(x) \neq 0$.
* $g(x)$ should be close to being proportional to $\mid f(x) \mid$.
* $g(x)$ should be easy to simulate samples from.
* It should be easy to compute the density $g(x)$ for any value $x$ you might realize.

Additionally, the importance sampling does not work well in high dimensions, either.

# Gibbs Sampling

Gibbs sampling (also called alternating conditional sampling), proposed in the early 1990s, is a special case (simplified case) of a family of Metropolis-Hasting (MH) algorithms and commonly used as a means of statistical inference, especially Bayesian inference. It is a the most basic randomized algorithm (i.e. an algorithm that makes use of random numbers), and is an alternative to deterministic algorithms for statistical inference such as the expectation-maximization algorithm (EM). It is a very useful way of sampling observations from multivariate distributions (generally full posterior distributions, i.e. $P(\theta)=P(\theta_{1}, \theta_{2}, \dots,\theta_{p})$) that are difficult to simulate from, directly but its conditional distributions, which are lower in dimension, are tractable to work with.

The primary reason why Gibbs sampling was introduced was to break the curse of dimensionality (which impacts both acception-rejection algorithm and importance sampling) by producing a sequence of low dimension simulations that still converge to the right target. Even though the dimension of the target impacts the speed of convergence.

Gibbs sampling is attractive because it can sample from high-dimensional posteriors. The main idea is to break the problem of sampling from the high-dimensional joint distribution into a series of samples from low-dimensional conditional distributions. Because the low-dimensional updates are done in a loop, samples are not independent as in rejection sampling. The dependence of the samples turns out to follow a Markov distribution, leading to the name Markov chain Monte Carlo (MCMC) because each sample is dependent on the previous sample.

The algorithm begins by setting initial values for all parameters, $\mathbf{\theta}^{(0)} = (\theta_{1}^{(0)}, \theta_{2}^{(0)}, \ldots, \theta_{p}^{(0)})$. The initial values of the variables can be determined randomly or by some other algorithm such as expectation-maximization. Variables are then sampled one at a time from their full conditional distribution

$$
P\left( \theta_{j}  \mid \theta_{1}, ..., \theta_{j-1}, \theta_{j+1}, ..., \theta_{p}, \mathbf{y} \right)
$$

Rather than 1 sample from $p$-dimensional joint, we make $p$ 1-dimensional samples. The process is repeated until the required number of samples have been generated. It is common to ignore some number of samples at the beginning (the so-called burn-in period). Thinning can also be used — i.e. keeping every kth sample of the Markov Chain. Formally, the algorithm is:

1. Initialize $\mathbf{\theta}^{(0)} = (\theta_{1}^{(0)}, \theta_{2}^{(0)}, \ldots, \theta_{p}^{(0)})$
2. for $j = 1, 2, \ldots$ do:<br/>
  $$
  \begin{split}
  \theta_{1}^{(j)} &\sim P(\theta_{1}^{(j)} \mid \theta_{2}^{(j - 1)}, \theta_{3}^{(j - 1)}, \ldots , \theta_{p}^{(j - 1)})\\
  \theta_{2}^{(j)} &\sim P(\theta_{2}^{(j)} \mid \theta_{1}^{(j - 1)}, \theta_{3}^{(j - 1)}, \ldots , \theta_{p}^{(j - 1)})\\
  & \ldots \ldots \ldots \\
  \theta_{p}^{(j)} &\sim P(\theta_{p}^{(j)} \mid \theta_{1}^{(j - 1)}, \theta_{2}^{(j - 1)}, \ldots , \theta_{p-1}^{(j - 1)})\\
  \end{split}
  $$
3. end for

This process continues until "convergence". 

In other words, Gibbs sampling involves ordering the parameters and sampling from the conditional distribution for each parameter given the current value of all the other parameters and repeatedly cycling through this updating process. Each "loop" through these steps is called an "iteration" of the Gibbs sampler, and when a new sampled value of a parameter is obtained, it is called an "updated" value.

The main advantage of Gibbs sampling is that we do not need to tune the proposal distribution (like we need with Accept-Reject algorithm and we will do with Metropolis-Hastings algorithm below). Most of the time, it is easy to evaluate the conditional distributions. Conditionals may be conjugate and we can sample from them exactly. 

Disadvantages of Gibbs Sampling:
1. We need to be able to derive conditional probability distributions for each of the variables.
2. We need to be able to draw random samples from contitional probability distributions. In other words, even if we can extract the conditional distributions they may not be known forms  (no conjugacy), so we could not draw from them. Where it is difficult to sample from a conditional distribution, we can sample using a Metropolis-Hastings algorithm instead - this is known as Metropolis within Gibbs.
3. As the correlation between variables increases, the performance of the Gibbs sampler decreases. This is because the sequence of draws from the Gibbs sampler becomes more correlated.
4. Drawing from multiple conditional distributions may be slow and inefficient.

For example, let's consider a bivariate normal posterior distribution such that:

$$
\begin{pmatrix} \theta_1\\ \theta_2 \end{pmatrix} \sim N\left[\left(\begin{array}{c} 0\\ 0 \end{array}\right),\left(\begin{array}{ccc} 1 & \rho\\ \rho & 1 \end{array}\right)\right]
$$

where $\theta_{1}$ and $\theta_{2}$ are unknown parameters of the model, while $\rho$ is the known posterior correlation between $\theta_{1}$ and $\theta_{2}$. Gibbs sampling requires conditional distributions for each variable. In the case of Gaussians, there’s a closed-form for the conditional. Using the properties of the multivariate normal distribution

$$
\theta_1 \mid \theta_2,\: y \sim N(\rho\theta_2,\: 1-\rho^2) \sim \rho\theta_2 + \sqrt{1-\rho^2}N(0,\:1)
$$

and

$$
\theta_2 \mid \theta_1,\: y \sim N(\rho\theta_1,\: 1-\rho^2) \sim \rho\theta_1 + \sqrt{1-\rho^2}N(0,\:1)
$$

Our first step is to set the specifications of the Gibbs sampler such that:

* We have 10,000 total draws.
* We will make 1000 burn-in draws.
* The known  is equal to 0.6.

```python
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')

# Known correlation parameter
rho = 0.6;      
 
# Burn-in for the Gibbs sampler
burn_in = 1000     
 
# Draws to keep from sampler
keep_draws = 10000 

# Initialize the variables from zero
theta_1 = np.zeros(shape = (burn_in + keep_draws,))        
theta_2 = np. zeros(shape = (burn_in + keep_draws,))

for i in range(1,burn_in + keep_draws):
    theta_2[i] = (np.sqrt(1 - rho**2) * norm.rvs(loc=0, scale=1, size=1, random_state=None)) + (rho * theta_1[i-1])
    theta_1[i] = (np.sqrt(1 - rho**2) * norm.rvs(loc=0, scale=1, size=1, random_state=None)) + (rho * theta_2[i])

print("The first 5 values of 'theta_1' and 'theta_2' are:")
print(theta_1[1:5])
# [1.7417311  1.33021091 0.25966946 1.77545821]
print(theta_2[1:5])
# [-0.05888882  1.32659969  0.79265393  1.1013163 ]

# plotting the values before and after the transformation
plt.figure(figsize = (20, 10))
plt.subplot(211) # the first row of graphs
plt.plot(theta_1)
plt.ylabel("$\\theta_{1}$")
plt.subplot(212)
plt.plot(theta_2)
plt.ylabel("$\\theta_{2}$")
plt.savefig('theta1_theta2_gibbs_sampler.png')
plt.show()
```

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/theta1_theta2_gibbs_sampler.png?raw=true)

Finally, we finish our Gibbs sampler by examining the mean and variance-covariance matrix of the Gibbs sampler distribution.

```python
params = np.append(arr= theta_1[burn_in:burn_in+keep_draws, ].reshape(-1,1), values = theta_2[burn_in:burn_in+keep_draws, ].reshape(-1,1), axis = 1)
print(np.mean(params, axis = 0))
# [-0.00467987 -0.01182868]
print(np.cov(params, rowvar=False, bias=False))
# [[1.00003663 0.5958002 ]
#  [0.5958002  0.99836987]]
```

Some drawbacks of Gibbs sampling are: (1) long convergence time especially with the dimensionality of the data growing because Convergence time also depends on the shape of the distribution and (2) difficulty in finding the posterior for each variable.

Let's give another example. Here we are going to study the properties of Gibbs sampling through simulation. We start by assuming the size of a claim $X$ is exponentially distributed with parameter $\lambda$. Further, we treat the parameter $\lambda$ as a random variable that follows a gamma distribution with parameters $\alpha$ and $\beta$.
$\alpha$ and $\beta$ are constants. Stemming from our Bayesian approach, we can write this information as follows: the conditional distribution of $X$ given $\lambda$ as

$$
f(x \mid \lambda) = \lambda e^{-\lambda x}, \,\,\,\, x > 0
$$

and the mixing distribution of $\lambda$, with parameters $\alpha$ and $\beta$ as

$$
f(\lambda \mid \alpha, \beta) \frac{\beta^{\alpha}}{\Gamma (\alpha)} \lambda^{\alpha - 1} e^{-\beta \lambda}, \,\,\,\, \lambda > 0
$$

Now, the joint distribution of $X$ and $\lambda$ can be obtained.

$$
\begin{split}
f(x, \lambda) &= f(x \mid \lambda) f(\lambda)\\
&= \frac{1}{\Gamma (\alpha)} \beta^{\alpha} \lambda^{\alpha} e^{-\lambda (x + \beta)}
\end{split}
$$

By integrating the above joint density with respect to $\lambda$, we can obtain the marginal distribution of $X$.

$$
\begin{split}
f(x) &= \int_{0}^{\infty} f(x, \lambda) d\lambda \\
&= \frac{\alpha \beta^{\alpha}}{(\beta + x)^{\alpha + 1}}, \,\,\,\, x>0\\
&\sim Pareto(\alpha, \beta)
\end{split}
$$

Note that $f(x)$ simplifies to a Pareto distribution with parameters $\alpha$ and $\beta$. This density is the closed form solution; in most problems, this is not possible to
obtain. However, we chose distributions such that there would be a closed form solution in order to compare our results to. Finally, we can solve for each conditional distribution. Having already been given $f(x \mid \lambda)$, $f(\lambda \mid x)$ is derived as follows:

$$
\begin{split}
f(\lambda \mid x) &= \frac{f(x, \lambda)}{f(x)}\\
&= \frac{(x + \beta)^{\alpha + 1} \lambda^{\alpha} e^{-\lambda (x + \beta)}}{\Gamma (\alpha + 1)}\\
&\sim gamma (\alpha + 1, x + \beta)
\end{split}
$$

Notice here that the conditional distribution of $f(\lambda \mid x)$ simplifies to a gamma distribution. This is because of the use of a conjugate prior distribution. This simply means that the mixing distribution is from the same family as the variable of interest.

The next step in Gibbs sampling is to run our two conditional distributions through the iterative algorithm defined below:

1. Select arbitrary initial values $x^{(0)}$ and $\lambda^{(0)}$
2. Set counter index i = 0
3. Sample $x^{(i + 1)}$ from $f(x \mid \lambda^{(i)}) \sim exponential(\lambda^{(i)})$
4. Sample $\lambda^{(i + 1)}$ from $f(\lambda \mid x^{(i + 1)}) \sim gamma(\alpha + 1, x^{(i + 1)} + \beta)$
5.  Set $i = i + 1$ and return to step 3

For illustrative purposes, assume $\alpha = 5$ and $\beta = 100$. This reduces the uncertainty to the random variables $X$ and $\lambda$. Using the principles of Gibbs sampling as shown above, $100,000$ random numbers are generated for $X$ and $\lambda$.

```python
import numpy as np
from scipy.stats import expon, gamma, pareto
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')

alpha = 5
beta = 100

# Burn-in for the Gibbs sampler
burn_in = 500     
 
# Draws to keep from sampler
N = 100000

# Initialize the variables from zero
x = np.zeros(shape = (N,))        
lambdas = np. zeros(shape = (N,))
lambdas[0] = 0.5 

i = 0
for i in range(1,N):
    x[i] = expon.rvs(loc=0, scale=(1/lambdas[i-1]), size=1, random_state=None)
    lambdas[i] = gamma.rvs(a = alpha + 1, loc=0, scale=(1/(x[i] + beta)), size=1, random_state=None)
```

Let's show the last 100 sampled values for $X$ and the last 100 sampled values for $\lambda$:

```python
# plotting the values before and after the transformation
plt.figure(num = 1, figsize = (20, 10))
iteration_num = np.arange(start = N-100, stop = N, step = 1)
plt.subplot(211) # the first row of graphs
plt.scatter(iteration_num, x[-100:])
plt.xlabel('Iterations')
plt.ylabel("x")
plt.subplot(212)
plt.scatter(iteration_num, lambdas[-100:])
plt.ylabel("$\\lambda$")
plt.savefig('x_lambda_gibbs_sampler.png')
plt.show()
```

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/x_lambda_gibbs_sampler.png?raw=true)

As one can see from the figures, there is no pattern among the generated random numbers. Therefore, they can be considered as independent random samples.

Next, figures show the histograms of the last $99,500$ sampled values of $X$ and $\lambda$, respectively. These values are generated from dependent sampling schemes, which were based on the full conditional distributions of $f(x \mid \lambda)$ and $f(\lambda \mid x)$. The first $500$ values of each sequence are discarded as these are considered to be the burn-in iterations. Additionally, the respective marginal density curves of $X$ and $\lambda$ are overlaid on their corresponding histogram.

```python
plt.figure(num = 2, figsize = (20, 10))
iteration_num = np.arange(start = N-100, stop = N, step = 1)
plt.subplot(211) # the first row of graphs
plt.hist(x[burn_in:N, ], bins = 1000, density = True)
plt.xlim(0, 150)
a=sorted(x[burn_in:N, ])
plt.plot(a, pareto.pdf(a, b=alpha, loc=-beta, scale=beta)) #True distribution, f(x) follow Pareto (alpha, beta)
plt.xlabel('x')
plt.ylabel("Density")
plt.title('Histogram of generated sample values of X')
plt.subplot(212)
z=sorted(lambdas[burn_in:N, ])
plt.plot(z, gamma.pdf(z, a=alpha, loc = 0, scale = 1/beta)) # True distribution, f(lambda) follows gamma(alpha, beta)
plt.hist(lambdas[burn_in:N, ], bins = 100, density = True)
plt.xlabel('$\\lambda$')
plt.ylabel("Density")
plt.title(' Histogram of generated sample values of $\\lambda$')
plt.savefig('generated_samples_x_lambda_gibbs_sampler.png')
plt.show()
```

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/generated_samples_x_lambda_gibbs_sampler.png?raw=true)

The marginal densities appear to line up very well with the sampled values, which indicates the implemented dependent sampling schemes have generated random samples from their respective marginal distributions.

This is a property of Gibbs sampling. In effect, by taking very large random samples from the conditional posterior distributions, it appears as if the samples were taken from their respective marginal distributions. Thus, the generated random variates can be used to study the properties of the distribution of interest. With more complicated models, sampling from the marginal distributions directly would have been impossible; but with Gibbs sampling, it can be simulated. With these results in mind, we can formally state Gibbs sampling as:

The realization that as the number of iterations approaches infinity, the samples from the conditional posterior distributions converge to what the actual target distribution is that could not be sampled from directly.

Next, consider the following integral:

$$
f(x) = \int f(x \mid \lambda) f(\lambda) d\lambda
$$

This states that the marginal distribution of $X$ can now be interpreted as the average of the conditional distribution of $X$ given $\lambda$ taken with respect to the marginal distribution of $\lambda$. This fact suggests that an estimate for the actual value of $f(x)$ at the point $x$ may be obtained by taking the simulated average of $f(x \mid \lambda)$ over the sampled values of $\lambda$ as shown below:

$$
f(\hat{x}) = \frac{1}{99,500} \sum_{i=500}^{100,000} f(x \mid \lambda_{i})
$$

We can do these calculations in Python and compare it theoretical mean of Pareto distribution with $\alpha = 5$ and $\beta = 100$:

```python
np.mean(x[burn_in:N, ])
#24.86028647864501

pareto.mean(b=alpha, loc=-beta, scale=beta)
#25.0
```

A similar calculation can be done for $\lambda$.

$$
f(\hat{\lambda}) = \frac{1}{99,500} \sum_{i=500}^{100,000} f(\lambda \mid x_{i})
$$

```python
np.mean(lambdas[burn_in:N, ])
#0.0499839746712420

gamma.mean(a=alpha, loc = 0, scale = 1/beta)
#0.05
```

# Metropolis-Hastings Algorithm

Metropolis-Hastings (MH) algorithm is a Markov chain method for obtaining a sequence of random samples from a probability distribution from which direct sampling is difficult. This sequence can be used to approximate the distribution (e.g., generate a histogram) or to compute an integral (e.g., an expected value). MH algorithm and other MCMC methods are generally used for sampling from multi-dimensional distributions, especially when the number of dimensions is high.

While the Gibbs sampler relies on conditional distributions, the Metropolis-Hastings sampler uses a full joint density distribution to generate a candidate draws. The candidate draws are not automatically added to the chain but rather an acceptance probability distribution is used to accept or reject candidate draws.

Let $q(Y \mid X)$ be a transition density (also called candidate generating density) for $p$-dimensional $X$ and $Y$ from which we can easily simulate and it is either expilicitly available (up to a multiplicative constant, independent of $X$) or symmetric. Let $\pi (X)$ be our target density (i.e., stationary distribution that Markov Chain will eventually converge to). MH procedure is an iterative algorithm where at each stage, there are three steps. Suppose we are currently in state $x$ and we want to know how to move to the next state in state space.

1. Simulate a candidate value $y \sim q(Y \mid x)$. Note that the candidate value depends on our current state $x$.
2. Let 
  $$
  \alpha (y \mid x) = \min \left\{\frac{\pi (y) q(x \mid y)}{\pi (x) q(y \mid x)} , 1 \right\}
  $$
  <br/>
  $\alpha (y \mid x)$ is referred as the acceptance ratio.
  
3. Simulate $u \sim \text{Uniform}(0, 1)$. If $u \leq \alpha (y \mid x) $ then next state is equal to $y$. Otherwise, the next state is still $x$ (we stay in the same place). 

This three step procedure represents the transition kernel for our Markov Chain which we are simulating. We hope that after many simulations, the Markov Chain will converge to the stationary distribution. Eventually, we can be reasonably sure that samples that we draw from this process are draws from the stationary distribution, i.e., $\pi (x)$. 
  
## Random Walk Metropolis-Hastings Algorithm

A second natural approach for the practical construction of a MH algorithm is to take into account the value previously simulated to generate the following value. The idea is used in algorithms such as the simulated annealing algorithm and the stochastic gradient descent. 

Let $q(y \mid x)$ be defined as $y = x + \varepsilon$ where $\varepsilon\sim g$ and and  $g$ is a probability density symmetric about $0$. The most common distributions in this setup are the uniform distributions on spheres centered of the origin or standard distributions like normal and the Student's t-distribution (Note that these distributions need to be scaled).

Given this definition, we have 

$$
q(y\mid x) = g(\varepsilon)
$$

and 

$$
q(x\mid y) = g(-\varepsilon) = g(\varepsilon)
$$

Because $q(y\mid x)$ is symmetric in $x$ and $y$, the Metropolis-Hastings acceptance ratio $\alpha (y \mid x)$ simplifies to

$$
\begin{split}
\alpha(y\mid x) & = \min\left\{ \frac{\pi(y)q(x\mid y)}{\pi(x)q(y\mid x)}, 1 \right\}\\
& = \min\left\{\frac{\pi(y)}{\pi(x)}, 1 \right\}
\end{split}
$$

Given our current state $x$, the random walk Metropolis-Hastings algorithm proceeds as follows:

1. Simulate $\varepsilon\sim g$ and let $y = x + \varepsilon$.
2. Compute $\alpha (y \mid x) =  \min\left\\{\frac{\pi(y)}{\pi(x)}, 1 \right\\}$.
3. Simulate $u \sim \text{Uniform}(0, 1)$. If $u \leq \alpha (y \mid x) $ then next state is equal to $y$. Otherwise, the next state is still $x$ (we stay in the same place). 

It should be noted that this form of the Metropolis-Hastings algorithm was the original form of the *Metropolis algorithm*.

Let's do a simple example. We can show how random walk Metropolis-Hastings can be used to sample from a standard Normal distribution. Let $g$ be a uniform distribution over the interval  $(- \delta, \delta)$, where $\delta$ is small and $>0$ (its exact value doesn’t matter). Then we can do:

1. $\varepsilon \sim \text{Uniform}(- \delta, \delta)$ and let $y = x + \varepsilon$.
2. Compute $\alpha (y \mid x) = \min \left\\{ \frac{\varphi (y)}{\varphi (x)}, 1 \right\\}$ where $\varphi$ is the standard Normal density.
3. Simulate $u \sim \text{Uniform}(0, 1)$. If $u \leq \alpha(y \mid x)$ then accept $y$ as the next state, otherwise stay at x!

```python
import numpy as np
from scipy.stats import uniform, norm
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')

delta = 0.5
N = 100000
x = np.zeros(shape = (N,))

for i in range(1, N):
    eps = uniform.rvs(loc= -delta, scale= 2*delta, size=1, random_state=None)
    y = x[i - 1] + eps
    alpha = min((norm.pdf(y, loc=0, scale=1)/norm.pdf(x[i-1], loc=0, scale=1)), 1)
    u = uniform.rvs(loc=0, scale=1, size=1, random_state=None)
    if u <= alpha:
        x[i] = y
    else:
        x[i] = x[i - 1]
```

We can take a look at a histogram of the samples to see if they look like a Normal distribution.

```python
plt.figure(num = 1, figsize = (20, 10))
z = np.arange(start = -4, stop = 4, step = 0.01)
plt.plot(z, norm.pdf(z, loc=0, scale=1), lw=2, label=' Target Distribution - Normal Distribution')
plt.hist(x, bins = 50, density = True)
plt.xlabel('x')
plt.ylabel("Frequency")
plt.title("Histogram of x")
plt.savefig('random_walk_MH_example.png')
plt.show()

np.mean(x)
#-0.023697071990286523

np.std(x)
#0.9966458012631055
```

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/random_walk_MH_example.png?raw=true)

## Independent Metropolis-Hastings Algorithm

This method appears a straightforward generalization of the Accept-Reject algorithm in a sense that instrumental distribution $q$ is independent of $x$, $q(y \mid x) = q(y)$. In other words, the candidate proposals do not depend on the current state of $x$. Otherwise, the algorithm works the same as MH algorithm, with a modified acceptance ratio:

$$
\alpha(y\mid x) = \min\left\{
\frac{\pi(y)q(x)}{\pi(x)q(y)}, 1
\right\}
$$

Since it is similar, this sampler works well in situations where accept-reject method might be reasonable, i.e., relatively low-dimensional problems.

Slice Sampler, Hit and Run Sampler and Single Component Metropolis-Hastings are other types of MH algorithm.

#### REFERENCES

1. [https://www.quora.com/What-is-an-intuitive-explanation-of-inverse-transform-sampling-method-in-statistics-and-how-does-it-relate-to-cumulative-distribution-function/answer/Amit-Sharma-2?srid=X8V](https://www.quora.com/What-is-an-intuitive-explanation-of-inverse-transform-sampling-method-in-statistics-and-how-does-it-relate-to-cumulative-distribution-function/answer/Amit-Sharma-2?srid=X8V){:target="_blank"}
2. [http://www.eg.bucknell.edu/~xmeng/Course/CS6337/Note/master/node48.html](http://www.eg.bucknell.edu/~xmeng/Course/CS6337/Note/master/node48.html){:target="_blank"}
3. [http://www.columbia.edu/~ks20/4404-Sigman/4404-Notes-ITM.pdf](http://www.columbia.edu/~ks20/4404-Sigman/4404-Notes-ITM.pdf){:target="_blank"}
4. [http://people.duke.edu/~ccc14/sta-663-2016/15A_RandomNumbers.html](http://people.duke.edu/~ccc14/sta-663-2016/15A_RandomNumbers.html){:target="_blank"}
5. [http://karlrosaen.com/ml/notebooks/simulating-random-variables/](http://karlrosaen.com/ml/notebooks/simulating-random-variables/){:target="_blank"}
6. [https://stephens999.github.io/fiveMinuteStats/inverse_transform_sampling.html](https://stephens999.github.io/fiveMinuteStats/inverse_transform_sampling.html){:target="_blank"}
7. [https://stats.stackexchange.com/a/236157/16534](https://stats.stackexchange.com/a/236157/16534){:target="_blank"}
8. [https://www2.isye.gatech.edu/~sman/courses/6644/Module07-RandomVariateGenerationSlides_171116.pdf](https://www2.isye.gatech.edu/~sman/courses/6644/Module07-RandomVariateGenerationSlides_171116.pdf){:target="_blank"}
9. [http://bjlkeng.github.io/posts/sampling-from-a-normal-distribution/](http://bjlkeng.github.io/posts/sampling-from-a-normal-distribution/){:target="_blank"}
10. [https://en.wikipedia.org/wiki/Normal_distribution#Generating_values_from_normal_distribution](https://en.wikipedia.org/wiki/Normal_distribution#Generating_values_from_normal_distribution){:target="_blank"}
11. [https://web.ics.purdue.edu/~hwan/IE680/Lectures/Chap08Slides.pdf](https://web.ics.purdue.edu/~hwan/IE680/Lectures/Chap08Slides.pdf){:target="_blank"}
12. [https://www.win.tue.nl/~marko/2WB05/lecture8.pdf](https://www.win.tue.nl/~marko/2WB05/lecture8.pdf){:target="_blank"}
13. [http://www.columbia.edu/~ks20/4703-Sigman/4703-07-Notes-ARM.pdf](http://www.columbia.edu/~ks20/4703-Sigman/4703-07-Notes-ARM.pdf){:target="_blank"}
14. [http://statweb.stanford.edu/~owen/mc/Ch-nonunifrng.pdf](http://statweb.stanford.edu/~owen/mc/Ch-nonunifrng.pdf){:target="_blank"}
15. [http://www.jarad.me/courses/stat544/slides/Ch11/Ch11a.pdf](http://www.jarad.me/courses/stat544/slides/Ch11/Ch11a.pdf){:target="_blank"}
16. [http://patricklam.org/teaching/mcmc_print.pdf](http://patricklam.org/teaching/mcmc_print.pdf){:target="_blank"}
