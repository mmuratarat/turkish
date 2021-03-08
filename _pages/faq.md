---
layout: post
title: "Frequently Asked Questions (and Answers)"
author: MMA
comments: false
permalink: /faq/
---

[Mathematics and Linear Algebra](#mathematics-and-linear-algebra)

1. [What are scalars, vectors, matrices, and tensors?](#{what-are-scalars-vectors-matrices-and-tensors)
2. [How do you normalize a vector?](#how-do-you-normalize-a-vector)
2. [What is a dot-product?](#what-is-a-dot-product)
2. [What is Hadamard product of two matrices?](#what-is-hadamard-product-of-two-matrices)
3. [What is a scalar valued function?](#what-is-a-scalar-valued-function)
4. [What is a vector valued function?](#what-is-a-vector-valued-function)
5. [What is the gradient?](#what-is-the-gradient)
6. [What is a Jacobian matrix?](#what-is-a-jacobian-matrix)
7. [What is a Hessian matrix?](#what-is-a-hessian-matrix)
8. [What is an identity matrix?](#what-is-an-identity-matrix)
9. [What is the transpose of a matrix?](#what-is-the-transpose-of-a-matrix)
10. [What is an inverse matrix?](#what-is-an-inverse-matrix)
11. [When does inverse of a matrix exist?](#when-does-inverse-of-a-matrix-exist)
12. [If inverse of a matrix exists, how to calculate it?](#if-inverse-of-a-matrix-exists-how-to-calculate-it)
13. [What is the determinant of a square matrix? How is it calculated? What is the connection of determinant to eigenvalues?](#what-is-the-determinant-of-a-square-matrix-how-is-it-calculated-what-is-the-connection-of-determinant-to-eigenvalues)
14. Discuss span and linear dependence.
15. What is Ax = b? When does Ax =b has a unique solution?
16. In Ax = b, what happens when A is fat or tall?
17. [What is a norm? What is $L^{1}$, $L^{2}$ and $L^{\infty}$ norm? What are the conditions a norm has to satisfy?](#what-is-a-norm-what-is-l_1-l_2-and-l_infty-norm-what-are-the-conditions-a-norm-has-to-satisfy)
18. Why is squared of L2 norm preferred in ML than just L2 norm?
19. When L1 norm is preferred over L2 norm?
20. Can the number of nonzero elements in a vector be defined as $L^{0}$ norm? If no, why?
21. [What is Frobenius norm?](#what-is-frobenius-norm)
22. [What is a diagonal matrix?](#what-is-a-diagonal-matrix)
23. Why is multiplication by diagonal matrix computationally cheap? How is the multiplication different for square vs. non-square diagonal matrix?
24. At what conditions does the inverse of a diagonal matrix exist?
25. [What is a symmetric matrix?](#what-is-a-symmetric-matrix)
26. [What is the Moore Penrose pseudo inverse and how to calculate it?](#what-is-a-symmetric-matrix)](#what-is-the-moore-penrose-pseudo-inverse-and-how-to-calculate-it)
27. When are two vectors x and y orthogonal?
28. At $\mathbb{R}^n$ what is the maximum possible number of orthogonal vectors with non-zero norm?
29. When are two vectors x and y orthonormal?
30. What is an orthogonal matrix? Why is computationally preferred?
31. [What is eigendecomposition, eigenvectors and eigenvalues? How to find eigenvalues of a matrix?](#what-is-eigendecomposition-eigenvectors-and-eigenvalues-how-to-find-eigenvalues-of-a-matrix)
32. [What is Spectral Decomposition (Eigendecompoisition)?](#what-is-spectral-decomposition-eigendecompoisition)
33. [What is Singular Value Decomposition?](#what-is-singular-value-decomposition)
34. [What is the Moore Penrose pseudo inverse and how to calculate it?](#what-is-the-moore-penrose-pseudo-inverse-and-how-to-calculate-it)
35. [What is the trace of a matrix?](#what-is-the-trace-of-a-matrix)
36. [How to write Frobenius norm of a matrix A in terms of trace?](#how-to-write-frobenius-norm-of-a-matrix-a-in-terms-of-trace)
37. Why is trace of a multiplication of matrices invariant to cyclic permutations?
38. [What is the trace of a scalar?](#what-is-the-trace-of-a-scalar)
39. [What do positive definite, positive semi-definite and negative definite/negative semi-definite mean?](#what-do-positive-definite-positive-semi-definite-and-negative-definitenegative-semi-definite-mean)
40. [How to make a positive definite matrix with a matrix that’s not symmetric?](#how-to-make-a-positive-definite-matrix-with-a-matrix-thats-not-symmetric)
41. [How is the set of positive semi-definite matrices a convex set?](#how-is-the-set-of-positive-semi-definite-matrices-a-convex-set)
42. [What are the one-dimensional and multi-dimensional Taylor expansions?](#what-are-the-one-dimensional-and-multi-dimensional-taylor-expansions)


[Numerical Optimization](#numerical-optimization)

1. [What is underflow and overflow?](#what-is-underflow-and-overflow)
2. [How to tackle the problem of underflow or overflow for softmax function or log softmax function?](#how-to-tackle-the-problem-of-underflow-or-overflow-for-softmax-function-or-log-softmax-function)
3. What is poor conditioning and the condition number?
4. [What is second derivative test?](#what-is-second-derivative-test)
5. [Describe convex function.](#describe-convex-function)
6. [What are the Karush-Kuhn-Tucker conditions?](#what-are-the-karush-kuhn-tucker-conditions)
7. [What is Lagrangian function?](#what-is-lagrangian-function)
8. [What is Jensen's inequality?](#what-is-jensens-inequality)


[Set Theory](#set-theory)

1. [What is a random experiment?](#what-is-a-random-experiment)
2. [What is a sample space?](#what-is-a-sample-space)
3. [What is an empty set?](#what-is-an-empty-set)
4. [What is an event?](#what-is-an-event)
5. [What are the operations on a set?](#what-are-the-operations-on-a-set)
6. [What is mutually exclusive (disjoint) events?](#what-is-mutually-exclusive-disjoint-events)
7. [What is a non-disjoint event?](#what-is-a-non-disjoint-event)
8. [What is an independent event?](#what-is-an-independent-event)
9. [What is exhaustive events?](#what-is-exhaustive-events)
10. [What is Inclusion-Exlusive Principle?](#what-is-inclusion-exlusive-principle)


[Probability and Statistics](#probability-and-statistics)

9. [Explain about statistics branches?](#explain-about-statistics-branches)
9. [What is the permutation and combination?](#what-is-the-permutation-and-combination)
9. [What is a probability?](#what-is-a-probability)
10. [What are the probability axioms?](#what-are-the-probability-axioms)
11. [What is a random variable?](#what-is-a-random-variable)
12. [Compare "Frequentist statistics" vs. "Bayesian statistics"](#compare-frequentist-statistics-vs-bayesian-statistics)
13. [What is a probability distribution?](#what-is-a-probability-distribution)
17. [What is a probability mass function? What are the conditions for a function to be a probability mass function?](#what-is-a-probability-mass-function-what-are-the-conditions-for-a-function-to-be-a-probability-mass-function)
18. [What is a probability density function? What are the conditions for a function to be a probability density function?](#what-is-a-probability-density-function-what-are-the-conditions-for-a-function-to-be-a-probability-density-function)
16. [What is a joint probability distribution? What is a marginal probability? Given the joint probability function, how will you calculate it?](#what-is-a-joint-probability-distribution-what-is-a-marginal-probability-given-the-joint-probability-function-how-will-you-calculate-it)
20. [What is conditional probability? Given the joint probability function, how will you calculate it?](#what-is-conditional-probability-given-the-joint-probability-function-how-will-you-calculate-it)
21. [State the Chain rule of conditional probabilities.](#state-the-chain-rule-of-conditional-probabilities)
22. [What are the conditions for independence and conditional independence of two random variables?](#what-are-the-conditions-for-independence-and-conditional-independence-of-two-random-variables)
23. [What are expectation, variance and covariance?](#what-are-expectation-variance-and-covariance)
24. [What is the covariance for a vector of random variables?](#what-is-the-covariance-for-a-vector-of-random-variables)
25. [What is cross-covariance?](#what-is-cross-covariance)
26. [What is the correlation for a vector of random variables? How is it related to covariance matrix?](#what-is-the-correlation-for-a-vector-of-random-variables-how-is-it-related-to-covariance-matrix)
26. [Explain some Discrete and Continuous Distributions.](#explain-some-discrete-and-continuous-distributions)
26. [What is a moment?](#what-is-a-moment)
26. [What is moment generating function?](#what-is-moment-generating-function)
26. [What is characteristic function?](#what-is-characteristic-function)
26. [What are the properties of Distributions?](#what-are-the-properties-of-distributions)
27. [What are the measures of Central Tendency: Mean, Median, and Mode?](#what-are-the-measures-of-central-tendency-mean-median-and-mode)
27. [How to compute the median of a probability distribution?](#how-to-compute-the-median-of-a-probability-distribution)
27. [How to find the distribution of Order Statistics?](#how-to-find-the-distribution-of-order-statistics)
28. [What are the properties of an estimator?](#what-are-the-properties-of-an-estimator)
29. [Explain Method Of Moments (MOM), Maximum A Posteriori (MAP), and Maximum Likelihood Estimation (MLE).](#explain-method-of-moments-mom-maximum-a-posteriori-map-and-maximum-likelihood-estimation-mle)
30. [What is score function and Fisher Information Matrix?](#what-is-score-function-and-fisher-information-matrix)
25. [What is a Bernoulli distribution? Calculate the expectation and variance of a random variable that follows Bernoulli distribution?](#what-is-a-bernoulli-distribution-calculate-the-expectation-and-variance-of-a-random-variable-that-follows-bernoulli-distribution)
26. [What is Binomial distribution?](#what-is-binomial-distribution)
26. [What is a multinoulli distribution?](#what-is-a-multinoulli-distribution)
26. [What is a multinomial distribution?](#what-is-a-multinomial-distribution)
27. [What is a normal distribution?](#what-is-a-normal-distribution)
28. [What is a continuous uniform distribution?](#what-is-a-uniform-distribution)
28. [What is exponential distribution?](#what-is-exponential-distribution)
28. [What is Poisson distribution?](#what-is-poisson-distribution)
28. [What is Chi-square distribution?](#what-is-chi-square-distribution)
28. [What is Student’s t-distribution?](#what-is-students-t-distribution)
29. [What is the central limit theorem?](#what-is-the-central-limit-theorem)
36. [Write the formulae for Bayes rule.](#write-the-formulae-for-bayes-rule)
37. [What is conjugate prior?](#what-is-conjugate-prior)
38. [How does the denominator in Bayes rule act as normalizing constant?](#how-does-the-denominator-in-bayes-rule-act-as-normalizing-constant)
39. [What is uninformative prior?](#what-is-uninformative-prior)
46. [What is population mean and sample mean?](#what-is-population-mean-and-sample-mean)
47. [What is population standard deviation and sample standard deviation?](#what-is-population-standard-deviation-and-sample-standard-deviation)
48. [Why population standard deviation has N degrees of freedom while sample standard deviation has N-1 degrees of freedom? In other words, why 1/N inside root for population and 1/(N-1) inside root for sample standard deviation?](#why-population-standard-deviation-has-n-degrees-of-freedom-while-sample-standard-deviation-has-n-1-degrees-of-freedom-in-other-words-why-1n-inside-root-for-population-and-1n-1-inside-root-for-sample-standard-deviation)
48. [What is the trading-off between bias and variance to minimize mean squared error of an estimator?](#what-is-the-trading-off-between-bias-and-variance-to-minimize-mean-squared-error-of-an-estimator)
48. [What is the unbiased estimator and its proof?](#what-is-the-unbiased-estimator-and-its-proof)
48. [What is the consistency of an estimator?](#what-is-the-consistency-of-an-estimator)
48. [What is the sufficiency of an estimator?](#what-is-the-sufficiency-of-an-estimator)
48. [What is the standard error of the estimate?](#what-is-the-standard-error-of-the-estimate)
49. [What is the sampling distribution of the sample mean?](#what-is-the-sampling-distribution-of-the-sample-mean)
50. [What is the sampling distribution of the sample variance?](#what-is-the-sampling-distribution-of-the-sample-variance)
50. [What is the sampling distribution of sample proportion, p-hat?](#what-is-the-sampling-distribution-of-sample-proportion-p-hat)
52. [What is Normal approximation to the Binomial and Continuity Correction?](#what-is-normal-approximation-to-the-binomial-and-continuity-correction)
52. [What does statistically significant mean?](#what-does-statistically-significant-mean)
52. [What is a p-value?](#what-is-a-p-value)
52. [What is confidence interval?](#what-is-confidence-interval)
53. [What do Type I and Type II errors mean?](#what-do-type-i-and-type-ii-errors-mean)
53. [What is the power of a statistical test?](#what-is-the-power-of-a-statistical-test)
71. [How to determine sample size?](#how-to-determine-sample-size)
72. [What are the sampling strategies?](#what-are-the-sampling-strategies)
54. [What is the difference between ordinal, interval and ratio variables?](#what-is-the-difference-between-ordinal-interval-and-ratio-variables)
55. [What is the general approach to hypothesis testing?](#what-is-the-general-approach-to-hypothesis-testing)
56. [What are the types of hypothesis tests?](#what-are-the-types-of-hypothesis-tests)
57. [When to use the z-test versus t-test?](#when-to-use-the-z-test-versus-t-test)
58. [How to do one sample test of means?](#how-to-do-one-sample-test-of-means)
59. [How to do two samples test of means?](#how-to-do-two-samples-test-of-means)
60. [How to do paired t-test?](#how-to-do-paired-t-test)
61. [How to do one sample test of proportions?](#how-to-do-one-sample-test-of-proportions)
62. [How to do two samples test of proportions?](#how-to-do-two-samples-test-of-proportions)
63. [How to do chi-square test for variance?](#how-to-do-chi-square-test-for-variance)
64. [How to do F-test for equality of two variances?](#how-to-do-f-test-for-equality-of-two-variances)
65. [What is Chi-square Test for Goodness-of-fit Test?](#what-is-chi-square-test-for-goodness-of-fit-test)
65. [What is Chi-square Test for Test of Independence?](#what-is-chi-square-test-for-test-of-independence)
65. [What is the post-hoc pairwise comparison of chi-squared test?](#what-is-the-post-hoc-pairwise-comparison-of-chi-squared-test)
65. [What is Fisher's Exact test?](#what-is-fishers-exact-test)
65. [What does statistical interaction mean?](#what-does-statistical-interaction-mean)
66. [Explain generalized linear model](#explain-generalized-linear-model).
66. [What does link function do?](#what-does-link-function-do)
67. [Given X and Y are independent variables with normal distributions, what is the mean and variance of the distribution of 2X - Y when the corresponding distributions are X follows N (3, 4) and Y follows N(1, 4)?](#given-x-and-y-are-independent-variables-with-normal-distributions-what-is-the-mean-and-variance-of-the-distribution-of-2x---y-when-the-corresponding-distributions-are-x-follows-n-3-4-and-y-follows-n1-4)
68. [A system is guaranteed to fail 10% of a time within any given hour, what's the failure rate after two hours ? after n-hours?](#a-system-is-guaranteed-to-fail-10-of-a-time-within-any-given-hour-whats-the-failure-rate-after-two-hours--after-n-hours)
68. [What is the relation between variance and sum of squares?](#what-is-the-relation-between-variance-and-sum-of-squares)
69. [What is analysis of variance (ANOVA)?](#what-is-analysis-of-variance-anova)
70. [What is analysis of covariance (ANCOVA)?](#what-is-analysis-of-covariance-ancova)
70. [What is Homogeneity of Variances? When and how should we check it?](#what-is-homogeneity-of-variances-when-and-how-should-we-check-it)
73. [When is a biased estimator preferable to unbiased one?](#when-is-a-biased-estimator-preferable-to-unbiased-one)
74. [What is the difference between t-test and linear regression?](#what-is-the-difference-between-t-test-and-linear-regression)
74. [What are qq-plots and pp-plots?](#what-are-qq-plots-and-pp-plots)
75. [What to do when normality assumption is violated?](#what-to-do-when-normality-assumption-is-violated)
76. [How to see non-Spherical disturbances?](#how-to-see-non-spherical-disturbances)
77. [What is the sum of the independent normal distributed random variables?](#what-is-the-sum-of-the-independent-normal-distributed-random-variables)
78. [What are the non-parametric equivalent of some parametric tests?](#what-are-the-non-parametric-equivalent-of-some-parametric-tests)
79. [Explain A/B test and its variants.](#explain-ab-test-and-its-variants)
80. [What is the difference between a mixture model and a multimodal distribution?](#what-is-the-difference-between-a-mixture-model-and-a-multimodal-distribution)
81. What is a confounding variable?

[General Machine Learning](#general-machine-learning)

1. [What is hypothesis in Machine Learning?](#what-is-hypothesis-in-machine-learning)
2. [What is the matrix used to evaluate the predictive model? How do you evaluate the performance of a regression prediction model vs a classification prediction model?](#what-is-the-matrix-used-to-evaluate-the-predictive-model-how-do-you-evaluate-the-performance-of-a-regression-prediction-model-vs-a-classification-prediction-model)
3. [What is confusion matrix and its elements?](#what-is-confusion-matrix-and-its-elements)
4. [What is a linear regression model?](#what-is-a-linear-regression-model)
4. [What are the assumptions required for linear regression?](#what-are-the-assumptions-required-for-linear-regression)
4. [What is the standard error of the coefficient?](#what-is-the-standard-error-of-the-coefficient)
4. [What is collinearity and what to do with it? How to remove multicollinearity?](#what-is-collinearity-and-what-to-do-with-it-how-to-remove-multicollinearity)
5. [What is Heteroskedasticity and weighted least squares?](#what-is-heteroskedasticity-and-weighted-least-squares)
5. [What are the assumptions required for logistic regression?](#what-are-the-assumptions-required-for-logistic-regression)
6. [Why is logistic regression considered to be linear model?](#why-is-logistic-regression-considered-to-be-linear-model)
7. [Why sigmoid function in Logistic Regression?](#why-sigmoid-function-in-logistic-regression)
7. [What is the loss function for Logistic Regression?](#what-is-the-loss-function-for-logistic-regression)
7. [How do you find the parameters in logistic regression?](#how-do-you-find-the-parameters-in-logistic-regression)
8. [What is Softmax regression and how is it related to Logistic regression?](#what-is-softmax-regression-and-how-is-it-related-to-logistic-regression)
8. What is odds ratio? How to interpret it? How to compute confidence interval for it?
10. [What is R squared?](#what-is-r-squared)
11. [You have built a multiple regression model. Your model $R^{2}$ isn't as good as you wanted. For improvement, your remove the intercept term, your model $R^{2}$ becomes 0.8 from 0.3. Is it possible? How?](#you-have-built-a-multiple-regression-model-your-model-r2-isnt-as-good-as-you-wanted-for-improvement-your-remove-the-intercept-term-your-model-r2-becomes-08-from-03-is-it-possible-how)
12. [How do you validate a machine learning model?](#how-do-you-validate-a-machine-learning-model)
13. [What is the Bias-Variance Tradeoff?](#what-is-the-bias-variance-tradeoff)
14. [What is the Bias-variance trade-off for Leave-one-out and k-fold cross validation?](#what-is-the-bias-variance-trade-off-for-leave-one-out-and-k-fold-cross-validation)
15. [Describe Machine Learning, Deep Learning, Supervised Learning, Unsupervised Learning, Semi-Supervised Learning, Reinforcement Learning with examples](#describe-machine-learning-deep-learning-supervised-learning-unsupervised-learning-semi-supervised-learning-reinforcement-learning-with-examples)
16. [What is batch learning and online learning?](#what-is-batch-learning-and-online-learning)
17. [What is instance-based and model-based learning?](#what-is-instance-based-and-model-based-learning)
18. [What are the main challenges of machine learning algorithms?](#what-are-the-main-challenges-of-machine-learning-algorithms)
19. [What are the most important unsupervised learning algorithms?](#what-are-the-most-important-unsupervised-learning-algorithms)
19. [What is a Machine Learning pipeline?](#what-is-a-machine-learning-pipeline)
20. [What is Tensorflow?](#what-is-tensorflow)
21. [Why Deep Learning is important?](#why-deep-learning-is-important)
22. [What are the three respects of an learning algorithm to be efficient?](#what-are-the-three-respects-of-an-learning-algorithm-to-be-efficient)
23. [What are the differences between a parameter and a hyperparameter?](#what-are-the-differences-between-a-parameter-and-a-hyperparameter)
24. [Why do we have three sets: training, validation and test?](#why-do-we-have-three-sets-training-validation-and-test)
25. [What are the goals to build a learning machine?](#what-are-the-goals-to-build-a-learning-machine)
26. [What are the solutions of overfitting?](#what-are-the-solutions-of-overfitting)
27. [Is it better to design robust or accurate algorithms?](#is-it-better-to-design-robust-or-accurate-algorithms)
29. [What are some feature scaling (a.k.a data normalization) techniques? When should you scale your data? Why?](#what-are-some-feature-scaling-aka-data-normalization-techniques-when-should-you-scale-your-data-why)
30. [What are the types of feature selection methods?](#what-are-the-types-of-feature-selection-methods)
31. [When should you reduce the number of features?](#when-should-you-reduce-the-number-of-features)
32. [When is feature selection is unnecessary?](#when-is-feature-selection-is-unnecessary)
31. [How can you prove that one improvement you've brought to an algorithm is really an improvement over not doing anything?](#how-can-you-prove-that-one-improvement-youve-brought-to-an-algorithm-is-really-an-improvement-over-not-doing-anything)
32. [What are the hyperparameter tuning methods?](#what-are-the-hyperparameter-tuning-methods)
33. [How do we use probability in Machine Learning/Deep Learning framework?](#how-do-we-use-probability-in-machine-learningdeep-learning-framework)
34. [What are the differences and similarities between Ordinary Least Squares Estimation and Maximum Likelihood Estimation methods?](#what-are-the-differences-and-similarities-between-ordinary-least-squares-estimation-and-maximum-likelihood-estimation-methods)
35. [Do you suggest that treating a categorical variable as continuous variable would result in a better predictive model?](#do-you-suggest-that-treating-a-categorical-variable-as-continuous-variable-would-result-in-a-better-predictive-model)
36. [Considering the long list of machine learning algorithm, given a data set, how do you decide which one to use?](#considering-the-long-list-of-machine-learning-algorithm-given-a-data-set-how-do-you-decide-which-one-to-use)
37. [What is selection bias?](#what-is-selection-bias)
38. [What’s the difference between a generative and discriminative model?](#whats-the-difference-between-a-generative-and-discriminative-model)
39. [What cross-validation technique would you use on a time series dataset?](#what-cross-validation-technique-would-you-use-on-a-time-series-dataset)
40. [What is the difference between "long" and "wide" format data?](#what-is-the-difference-between-long-and-wide-format-data)
41. [Can you cite some examples where a false positive is important than a false negative, and where a false negative important than a false positive, and where both false positive and false negatives are equally important?](#can-you-cite-some-examples-where-a-false-positive-is-important-than-a-false-negative-and-where-a-false-negative-important-than-a-false-positive-and-where-both-false-positive-and-false-negatives-are-equally-important)
42. [Describe the difference between univariate, bivariate and multivariate analysis?](#describe-the-difference-between-univariate-bivariate-and-multivariate-analysis)
43. [What is the difference between dummying and one-hot encoding?](#what-is-the-difference-between-dummying-and-one-hot-encoding)
44. [What is out-of-core learning?](#what-is-out-of-core-learning)
45. [How do you detect outliers in a dataset?](#how-do-you-detect-outliers-in-a-dataset)
46. [What is the difference between norm and distance?](#what-is-the-difference-between-norm-and-distance)
47. [What is Hamming Distance?](#what-is-hamming-distance)
48. [How to find distance between mixed categorical and numeric data points?](#how-to-find-distance-between-mixed-categorical-and-numeric-data-points)
49. [What is the difference between Mahalanobis distance and Euclidean distance?](#what-is-the-difference-between-mahalanobis-distance-and-euclidean-distance)
50. [What is the difference between Support Vector Machines and Logistic Regression?](#what-is-the-difference-between-support-vector-machines-and-logistic-regression)
50. [What is the best separating hyperplane?](#what-is-the-best-separating-hyperplane)
51. [What is the optimization problem for Support Vector Machines?](#what-is-the-optimization-problem-for-support-vector-machines)
51. [What does the parameter C do in SVM?](#what-does-the-parameter-c-do-in-svm)
52. [Why do we find the dual problem when fitting SVM?](#why-do-we-find-the-dual-problem-when-fitting-svm)
53. [What is the output of Support Vector Machines?](#what-is-the-output-of-support-vector-machines)
54. [What are the support vectors in Support Vector Machines?](#what-are-the-support-vectors-in-support-vector-machines)
55. [What is the Kernel Trick?](#what-is-the-kernel-trick)
55. [How does SVM work in multiclass classification?](#how-does-svm-work-in-multiclass-classification)
55. [How does Kernel change the data?](#how-does-kernel-change-the-data)
55. [What is the bias-variance tradeoff for sigma parameter in RBF kernel?](#what-is-the-bias-variance-tradeoff-for-sigma-parameter-in-rbf-kernel)
56. [What is the output of Logistic Regression?](#what-is-the-output-of-logistic-regression)
57. [Can you interpret probabilistically the output of a Support Vector Machine?](#can-you-interpret-probabilistically-the-output-of-a-support-vector-machine)
58. [What are the advantages and disadvantages of Support Vector Machines?](#what-are-the-advantages-and-disadvantages-of-support-vector-machines)
59. [What is a parsimonious model?](#what-is-a-parsimonious-model)
60. [How do you deal with imbalanced data?](#how-do-you-deal-with-imbalanced-data)
61. [What is the difference between L1/L2 regularization?](#what-is-the-difference-between-l1l2-regularization)
62. [What is curse of dimensionality?](#what-is-curse-of-dimensionality)
62. [Why is dimension reduction important?](#why-is-dimension-reduction-important)
63. [Why would you want to avoid dimensionality reduction techniques to transform your data before training?](#why-would-you-want-to-avoid-dimensionality-reduction-techniques-to-transform-your-data-before-training)
64. [If you have large number of predictors how would you handle them?](#if-you-have-large-number-of-predictors-how-would-you-handle-them)
65. [How can you compare a neural network that has one layer, one input and output to a logistic regression model?](#how-can-you-compare-a-neural-network-that-has-one-layer-one-input-and-output-to-a-logistic-regression-model)
66. [What are the assumptions of Principle Component Analysis?](#what-are-the-assumptions-of-principle-component-analysis)
68. [What is micro-averaging and macro-averaging?](#what-is-micro-averaging-and-macro-averaging)
68. [If the model isn't perfect, how would you like to select the threshold so that the model outputs 1 or 0 for label?](#if-the-model-isnt-perfect-how-would-you-like-to-select-the-threshold-so-that-the-model-outputs-1-or-0-for-label)
69. [What's the difference between convex and non-convex cost function? what does it mean when a cost function is non-convex?](#whats-the-difference-between-convex-and-non-convex-cost-function-what-does-it-mean-when-a-cost-function-is-non-convex)
69. [How do you deal with missing value in a data set?](#how-do-you-deal-with-missing-value-in-a-data-set)
70. [How to find a confidence interval for accuracy of a model?](#how-to-find-a-confidence-interval-for-accuracy-of-a-model)
71. [Does tree-based methods such as Decision Tree handle multicollinearity by itself?](#does-tree-based-methods-such-as-decision-tree-handle-multicollinearity-by-itself)
72. [How to model count data?](#how-to-model-count-data)
73. [Why should weights of Neural Networks be initialized to random numbers?](#why-should-weights-of-neural-networks-be-initialized-to-random-numbers)
74. [Why is loss function in Neural Networks not convex?](#why-is-loss-function-in-neural-networks-not-convex)
74. [Is it possible to train a neural network without backpropagation?](#is-it-possible-to-train-a-neural-network-without-backpropagation)
74. [In neural networks, why do we use gradient methods rather than other metaheuristics?](#in-neural-networks-why-do-we-use-gradient-methods-rather-than-other-metaheuristics)
75. [What is the difference between a loss function and decision function?](#what-is-the-difference-between-a-loss-function-and-decision-function)
76. [What is the difference between SVM and Random Forest?](#what-is-the-difference-between-svm-and-random-forest)
76. [What is the difference between fitting a model via closed-form equations vs. Gradient Descent and its variants?](#what-is-the-difference-between-fitting-a-model-via-closed-form-equations-vs-gradient-descent-and-its-variants)
76. [What are some of the issues with K-means?](#what-are-some-of-the-issues-with-k-means)
76. [Why multicollinearity does not affect the predictive performance?](#why-multicollinearity-does-not-affect-the-predictive-performance)
76. [How does multicollinearity affect feature importances in random forest classifier?](#how-does-multicollinearity-affect-feature-importances-in-random-forest-classifier)
76. [How come do the loss functions have 1/m and 2 from the square cancels out?](#how-come-do-the-loss-functions-have-1m-and-2-from-the-square-cancels-out)
76. [What is feature engineering?](#what-is-feature-engineering)
76. [When to choose Decision Tree over Random Forest?](#when-to-choose-decision-tree-over-random-forest)
77. [What is a machine learning project lifecycle?](#what-is-a-machine-learning-project-lifecycle)
78. [What are the unknowns of a machine learning project?](#what-are-the-unknowns-of-a-machine-learning-project)
79. [What are the properties of a successful model?](#what-are-the-properties-of-a-successful-model)
80. [How to convert an RGB image to grayscale?](#how-to-convert-an-rgb-image-to-grayscale)
81. [What are the Model Selection Criterion, i.e., AIC and BIC?](#what-are-the-model-selection-criterion-ie-aic-and-bic)
82. [How to encode cyclical continuous features?](#how-to-encode-cyclical-continuous-features)
83. [Why does bagging work so well for decision trees, but not for linear classifiers?](#why-does-bagging-work-so-well-for-decision-trees-but-not-for-linear-classifiers)
84. [Is decision tree a linear model?](#is-decision-tree-a-linear-model)
85. [In machine learning, how can we determine whether a problem is linear/nonlinear?](#in-machine-learning-how-can-we-determine-whether-a-problem-is-linearnonlinear)
86. [What are the data augmentation techniques for images?](#what-are-the-data-augmentation-techniques-for-images)
86. [What are the data augmentation techniques for text?](#what-are-the-data-augmentation-techniques-for-text)
86. Why is data augmentation classified as a type of regularization?
87. Why using probability estimates and non-thresholded decision values give different AUC values?
88. What is Minkowski distance? How is it related to Manhattan distance and Euclidean distance
88. What to do before clustering?
89. How to cluster only categorical data with K-means?
90. What is K-medoids algorithm?
91. How to choose number of clusters in clustering analysis?
92. How to select a clustering method? 

[SQL](#SQL)

1. What is SQL?
2. What is Database?
3. What are the different subsets of SQL?
4. What is a query?
5. What is subquery?
1. [What is a primary key and a foreign key?](#what-is-a-primary-key-and-a-foreign-key)
2. [In which order do SQL queries happen?](#in-which-order-do-sql-queries-happen)
4. [What is the difference between UNION and UNION ALL?](#what-is-the-difference-between-union-and-union-all)
5. [What's the difference between VARCHAR and CHAR?](#whats-the-difference-between-varchar-and-char)
6. [How to insert rows into a table?](#how-to-insert-rows-into-a-table)
7. [How to update rows in a table?](#how-to-insert-rows-into-a-table)
8. [How to delete rows in a table?](#how-to-update-rows-in-a-table)
9. [How to create a new database table?](#how-to-delete-rows-in-a-table)
10. [How to drop a table?](#how-to-create-a-new-database-table)
11. What are the data types in PostgreSQL?
12. What are the constraints?
13. How to drop a table?
14. How to add a new column to a table?
15. How to remove a column from a table? 
16. How to rename a table?
17. [What is the difference between BETWEEN and IN operators in SQL?](#what-is-the-difference-between-between-and-in-operators-in-sql)
18. [What is the difference between primary key and unique constraints?](#what-is-the-difference-between-primary-key-and-unique-constraints)
19. [What is the difference between a Fact Table and a Dimension Table?](#what-is-the-default-ordering-of-data-using-the-order-by-clause-how-could-it-be-changed)
20. [What are the aggregate functions?](#what-are-the-aggregate-functions)
21. What Is a Equi and Non-equi Join in SQL?
21. [What is a join in SQL? What are the types of joins?](#what-is-a-join-in-sql-what-are-the-types-of-joins)
22. [What is the difference between a Fact Table and a Dimension Table?](#what-is-the-difference-between-a-fact-table-and-a-dimension-table)
23. What is Data Normalization?
24. What is a View?
25. How to drop a view?
26. Which operator is used in query for pattern matching?
27. What is an Index
28. What are the different types of relationships in SQL?
29. What is an Alias in SQL?
30. How to get random records from a table?
32. How to concatenate two string columns?
33. How to split a string?
34. How to retrieve values from one table that do not exists in another table?
35. How does EXTRACT and DATE_PART work in PostgreSQL?
36. What is the difference between EXTRACT and DATE_PART in PostgreSQL?
37. How to convert from  12 hours timestamp format to 24 hours timestamp or other way around?
38. How does AGE function work in PostgreSQL?
39. How to get yesterday's date?
40. How to get current date, time, timestamp?
41. How to use ROW_NUMBER(), RANK(), DENSE_RANK() window functions?
42. How to find modulus?
43. How to use DATE_TRUNC Function?
44. How to use REPLACE and TRANSLATE functions? 
45. How to pad a string on left or right?
46. How to convert a value of one data type into another?
47. How to find the position of a substring in a string?
48. How to replace Null in PostgreSQL?
49. How to find date differences?
50. How to extract n number of characters specified in the argument from the left or right of a given string?
51. How to extract a part of string?
52. How to remove (trim) characters from the beginning, end or both sides of a string?
53. How to randomly select a row?
54. What is the Dual Table? How is Oracle to PostgreSQL conversion?

[Miscellaneous](#miscellaneous)

1. [What is a good code?](#what-is-a-good-code)
2. [What is a data structure?](#what-is-a-data-structure)
3. [What are the commonly used Data Structures?](#what-are-the-commonly-used-data-structures)
4. [What is a big-O notation?](#what-is-a-big-o-notation)
5. [What are the different big-O notation measures?](#what-are-the-different-big-o-notation-measures)
6. [What is space complexity and time complexity?](#what-is-space-complexity-and-time-complexity)
7. [What are the best, average, worst case scenarios in big-O notation?](#what-are-the-best-average-worst-case-scenarios-in-big-o-notation)
8. [What are the built-in data types in Python?](#what-are-the-built-in-data-types-in-python)
9. [What are the data structures in Python?](#what-are-the-data-structures-in-python)
10. [What is `*args` and `**kwargs` in Python?](#what-is-args-and-kwargs-in-python)
11. [What are the mutable and immutable objects in Python?](#what-are-the-mutable-and-immutable-objects-in-python)
12. [What is the difference between linked list and array?](#what-is-the-difference-between-linked-list-and-array)
13. [What is the difference between stack and queue?](#what-is-the-difference-between-stack-and-queue)
14. [Explain Class, Object (Instance), Instance Attribute, Class Attribute, Instance Method with an example.](#explain-class-object-instance-instance-attribute-class-attribute-instance-method-with-an-example)
15. [How to create a JSON file? How to load a JSON file?](#how-to-create-a-json-file-how-to-load-a-json-file)
16. What is the time complexity for Binary Search Tree?
17. Where does the log of O(log n) come from?
18. What is Big Data?

## Mathematics and Linear Algebra

#### What are scalars, vectors, matrices, and tensors?

Scalars are single numbers and are an example of a 0th-order tensor. The notation $x \in \mathbb{R}$ states that the scalar value $x$ is an element of (or member of) the set of real-valued numbers, $\mathbb{R}$.

There are various sets of numbers of interest within machine learning. $\mathbb{N}$ represents the set of positive integers $(1,2,3, ...)$. $\mathbb{Z}$ represents the integers, which include positive, negative and zero values. $\mathbb{Q}$ represents the set of rational numbers that may be expressed as a fraction of two integers.

Vectors are ordered arrays of single numbers and are an example of 1st-order tensor.  An $n$-dimensional vector itself can be explicitly written using the following notation:

\begin{equation}
\boldsymbol{x}=\begin{bmatrix}
  \kern4pt x_1 \kern4pt \\
  \kern4pt x_2 \kern4pt \\
  \kern4pt \vdots \kern4pt \\
  \kern4pt x_n \kern4pt
\end{bmatrix}
\end{equation}

We can think of vectors as identifying points in space, with each element giving the coordinate along a different axis

One of the primary use cases for vectors is to represent physical quantities that have both a magnitude and a direction. Scalars are only capable of representing magnitudes.

Matrices are rectangular arrays consisting of numbers and are an example of 2nd-order tensors. If $m$ and $n$ are positive integers, that is $m, n \in \mathbb{N}$ then the $m \times n$ matrix contains $mn$ numbers, with $m$ rows and $n$ columns.

If all of the scalars in a matrix are real-valued then a matrix is denoted with uppercase boldface letters, such as $A \in \mathbb{R}^{m \times n}$. That is the matrix lives in a $m \times n$-dimensional real-valued vector space. 

Its components are now identified by two indices $i$ and $j$. $i$ represents the index to the matrix row, while $j$ represents the index to the matrix column. Each component of $A$ is identified by $a_{ij}$.

The full $m \times n$ matrix can be written as:

$$
\boldsymbol{A}=\begin{bmatrix}
   a_{11} & a_{12} & a_{13} & \ldots & a_{1n} \\
   a_{21} & a_{22} & a_{23} & \ldots & a_{2n} \\
   a_{31} & a_{32} & a_{33} & \ldots & a_{3n} \\
   \vdots & \vdots & \vdots & \ddots & \vdots \\
   a_{m1} & a_{m2} & a_{m3} & \ldots & a_{mn} \\
\end{bmatrix}
$$

It is often useful to abbreviate the full matrix component display into the following expression:

\begin{equation}
\boldsymbol{A} = [a_{ij}]_{m \times n}
\end{equation}

Where $a_{ij}$ is referred to as the $(i,j)$-element of the matrix $A$. The subscript of $m \times n$ can be dropped if the dimension of the matrix is clear from the context.

Note that a column vector is a size $m \times 1$ matrix, since it has $m$ rows and $1$ column. Unless otherwise specified all vectors will be considered to be column vectors.

Tensor is n-dimensional array. It encapsulates the scalar, vector and the matrix. For a 3rd-order tensor elements are given by $a_{ijk}$, whereas for a 4th-order tensor elements are given by $a_{ijkl}$.

#### How do you normalize a vector?

A vector is normalized when its norm is equal to one. To normalize a vector, we divide each of its elements by its norm or length (also known as magnitude). The norm of a vector is the square root of the sum of squares of the elements.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/norm_a_Vector.png?raw=true)

$$
\bar{x} = \frac{x}{\lVert x \rVert}
$$

For example,

$$
x = \begin{bmatrix} 2 \\ 1\\ 2\end{bmatrix}
$$

For example, the length (norm) of this vector is:

$$
\lVert x \rVert =\sqrt{2^{2} + 1^{2} + 2^{2}} = 3
$$

Therefore, if we normalize the vector $x$, we will get:

$$
\bar{x} = \begin{bmatrix} 2/3 \\ 1/3 \\ 2/3\end{bmatrix}
$$

A Unit Vector has a magnitude of 1. The term normalized vector is sometimes used as a synonym for unit vector.

#### What is a dot-product?

The dot product is an algebraic operation which takes two equal-sized vectors and returns a single scalar (which is why it is sometimes referred to as the scalar product). In Euclidean geometry, the dot product between the Cartesian components of two vectors is often referred to as the inner product.

The dot product is represented by a dot operator:

$$
s = \mathbf{x} \cdot \mathbf{y}
$$

It is defined as:

$$
s = \mathbf{x} \cdot \mathbf{y} = \sum_{i=1}^{n}x_iy_i = x_1y_1 + x_2y_2 + \dots + x_ny_n
$$

```python
# Define x and y
x = np.array([1, 3, -5])
y = np.array([4, -2, -1])

x @ y
#3

np.dot(x,y) # Alternatively, we can use the np.dot() function
#3
```

Keeping to the convention of having $\mathbf{x}$ and $\mathbf{y}$ as column vectors, the dot product is equal to the matrix multiplication $\mathbf{x}^{T}\mathbf{y}$:

```python
np.matmul(x.T, y)
```

In Euclidean space, a Euclidean vector has both magnitude and direction. The magnitude of a vector $\mathbf{x}$ is denoted by $\mid \mathbf{x} \mid$. The dot product of two Euclidean vectors $\mathbf{x}$ and $\mathbf{y}$ is defined by:

$$
\mathbf{x} \cdot \mathbf{y} =|\mathbf {x} |\ |\mathbf {y} |\cos(\theta)
$$

where $\theta$ is the angle between the two vectors. In general $cos \theta$ tells you the similarity in terms of the direction of the vectors (it is $−1$ when they point in opposite directions). If the angel between vectors is 0, i.e., $\theta = 0$, $cos (\theta) = cos (0) = 1$, the vectors are colinear, the dot product is the product of the magnitudes of the vectors. $cos \theta$ also gives us an easy way to test for orthogonality between vectors. If $\mathbf{x}$ and $\mathbf{y}$ are orthogonal (the angle between vectors is ($90$ degrees) then since $cos(90) = 0$, it implies that the dot product of any two orthogonal vectors must be $0$.

```python
# Let's test this by defining two vectors we know are orthogonal
x = [1, 0, 0]
y = [0, 1, 0]
print("The dot product of x and y is", np.dot(x, y))
```

#### What is Hadamard product of two matrices?

Hadamard product is also known as Element-wise Multiplication. It is named after French Mathematician, Jacques Hadamard. Elements corresponding to same row and columns of given vectors/matrices are multiplied together to form a new vector/matrix.

$$
    \begin{bmatrix} 
3 & 5 & 7 \\
4 & 9 & 8 
\end{bmatrix} \times \begin{bmatrix} 
1 & 6 & 3 \\
0 & 2 & 9 
\end{bmatrix} = \begin{bmatrix} 
3 \times 1 & 5 \times 6 & 7 \times 3 \\
4 \times 0 & 9 \times 2 & 8 \times 9 
\end{bmatrix} = \begin{bmatrix} 
3 & 30 & 21 \\
0 & 18 & 72 
\end{bmatrix}
$$

#### What is a scalar valued function?

A scalar valued function is a function that take one or more values but returns a single value. For example:

$$
f(x, y, z) = x^{2} + 2y z^{5}
$$

A $n$-variable scalar valued function acts as a map from the space $\mathbb{R}^{n}$ to the real number line $\mathbb{R}$. That is: $f:\mathbb{R}^{n} \to \mathbb{R}$

#### What is a vector valued function?

A vector valued function (also known as vector function) is a function where the domain is a subset of real numbers and the range is a vector. For example:

$$
r(t) = <2x+1, x^{2}+3>
$$

presents a function whose input is a scalar $t$ and whose output is a vector in $\mathbb{R}^{2}$

#### What is the gradient?

Suppose we have a function $y = f(x)$ where both $x$ and $y$ are real numbers. The derivative of this function is denoted as $f^{\prime} (x)$ or as $\frac{dy}{dx}$. The derivative $f^{\prime} (x)$ gives the flope of $f(x)$ at the point $x$. In other words, it specifies how to scale a small change in the input to obtain the corresponding change in the output. The derivative is therefore useful for minimizing a function because it tells us how to change in $x$ in order to make a small improvement in $y$We can thus reduce $f(x)$ by moving $x$ in small steps with the opposite sign of the derivative. This technique is called gradient descent.

When $f^{\prime} (x) = 0$, the derivatives provides no information about which direction to move. Points where $f^{\prime} (x) = 0$ are known as **critical points**, or **stationary points**. A **local minimum** is a point where $f(x)$ is lower than at all neighboring points, so it is no longer possible to decrease $f(x)$ by making steps. A **local maximum** is a point where $f(x)$ is higher than at all neighboring points, so it is not possible to increase $f(x)$ by making steps. Some critical points are neither maxima nor minima. These points are known as saddle points. 

A point that obtains the absolute lowest value of $f(x)$ is a **global minimum**. There can be only one global minimum or multiple global minima of the function. It is also possible for there to be a local minima that are not globally optimum. In the context of deep learning, we optimize functions that may have many local minima that are not optimal and many saddle points surrounded by flat regions. All of this makes optimization difficult, especially when the input to the function is multi-dimensional. We therefore usually settle for finding the value of $f$ that is very low but not necessarily minimal in any formal sense.

Gradient generalizes the notion of derivative to the case where the derivative is with respect to a vector: the gradient of f is the vector containing all the partial derivatives.

The gradient of a function $f$, denoted as $\nabla f$ is the collection of all its first-order partial derivatives into a vector. Here, $f$ is a scalar-valued (real-valued) multi-variable function $f:\mathbb{R}^{n}\to \mathbb{R}$.

$$
\nabla f(x_{1}, x_{2}, x_{3}, ...) = \begin{bmatrix} \dfrac{\partial f}{\partial x_{1}} \\[6pt] \dfrac{\partial f}{\partial x_{2}}\\[6pt] \dfrac{\partial f}{\partial x_{3}}\\[6pt] .\\.\\. \end{bmatrix}
$$

In particular, $\nabla f(x_{1}, x_{2}, x_{3}, ...)$ is a vector-valued function, which means it is a vector and we cannot take the gradient of a vector. 

It is very important to remember that the gradient of a function is only defined if the function is real-valued, that is, if it returns a scalar value. 

The most important thing to remember about the gradient is that the gradient of $f$, if evaluated at an input $(x_{0},y_{0})$, points in the direction of the steepest ascent. So, if you walk in that direction of the gradient, you will be going straight up the hill. Similarly, the magnitude of the vector $\nabla f(x_{0},y_{0})$ tells you what the slope of the hill is in that direction, meaning that if you walk in that direction, you will increase the value of $f$ at most rapidly. Therefore, for numerical methods being used in Deep Learning, such as gradient descent algorithm, we go in the negative direction of the gradient because we want to minimize the loss function. In this case, $f$ is a loss function which we decrease by moving in the direction of the negative gradient.

Gradient descent proposes a new point:

$$
x^{\prime} = x - \epsilon \nabla_{x} f(x)
$$

where $\epsilon$ is learning rate, a positive scalar determining the size of the step. We can choose $\epsilon$ in several different ways. A popular approach is to set $\epsilon$ to a small constant. Another approach is to evaluate $f(x - \epsilon \nabla_{x} f(x))$ for several values of $\epsilon$ and choose the one that results in the smallest objective value. This last strategy is called a **line search**. Gradient descent converges when every element of the gradient is zero (or, in practice, close to zero). In some cases, we may be able to avoid running this iterative algorithm and just jump directly to the critical point by solving the equation $\nabla_{x} f(x) = 0$ for $x$ and get analytical (closed-form) results. 

Although the gradient descent is limited to optimization in continuous spaces, the general concept of repeatedly making a small move (that is approximately the best small move) toward better configurations can be generalized to discrete spaces. Ascending an objective function of discrete parameters is called **hill climbing**.

Note that the symbol $\nabla$ is referred to either as nabla or del. 

Note that the gradient of a vector-valued function is the same as obtaining the Jacobian of this function.

#### What is a Jacobian matrix?

Jacobian matrix is a matrix of all first-order partial derivatives of a vector-valued function. Suppose $f:\mathbb{R}^{n} \to \mathbb{R}^{m}$ is a function which takes as input the vector $x \in \mathbb{R}^{n}$ and produces as output the vector $f(x) \in \mathbb{R}^{m}$. Then, the Jacobian matrix J of $f$ is a $m \times n$ matrix:

$$
J = \begin{bmatrix}
\dfrac{\partial f}{\partial x_{1}} & \cdots &\dfrac{\partial f}{\partial x_{n}}
\end{bmatrix} =
\begin{bmatrix}
\dfrac{\partial f_{1}}{\partial x_{1}} &\cdots &\dfrac{\partial f_{1}}{\partial x_{n}} \\[6pt]
& \cdots & \\[6pt]
\dfrac{\partial f_{m}}{\partial x_{1}} &\cdots &\dfrac{\partial f_{m}}{\partial x_{n}} \\[6pt]
\end{bmatrix}
$$

Note that when $m=1$, the Jacobian is the same as gradient because it is a generalization of the gradient.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/jacobian_example1.png?raw=true)

#### What is a Hessian matrix?

When out function has multiple input dimensions, there are many second derivatives. These derivatives can be collected together into a matrix called **Hessian matrix**. 

The hessian matrix is a square matrix of the second-order partial derivatives of a scalar-valued (real-valued) multi-variable function $f:\mathbb{R}^{n}\to \mathbb{R}$.

If we have a scalar-valued multi-variable function $f(x_{1}, x_{2}, x_{3}, ...)$, its Hessian with respect to x, is the $n \times n$ matrix of partial derivatives:

$$
H_{f} \in \mathbb{R}^{n\times n}= \begin{bmatrix}
\dfrac{\partial^{2}f(x)}{\partial x_{1}^{2}} & \dfrac{\partial^{2}f(x)}{\partial x_{1} \partial x_{2}} & \cdots & \dfrac{\partial^{2}f(x)}{\partial x_{1} \partial x_{n}}\\[7pt]
\dfrac{\partial^{2}f(x)}{\partial x_{2} \partial x_{1}} & \dfrac{\partial^{2}f(x)}{\partial x_{2}^{2}} & \cdots & \dfrac{\partial^{2}f(x)}{\partial x_{2} \partial x_{n}} \\[7pt]
\vdots & \vdots & \cdots & \vdots \\[7pt]
\dfrac{\partial^{2}f(x)}{\partial x_{n} \partial x_{1}} & \dfrac{\partial^{2}f(x)}{\partial x_{n} x_{2}} & \cdots & \dfrac{\partial^{2}f(x)}{\partial x_{n}^{2}}
\end{bmatrix}
$$

Similar to the gradient, the Hessian is defined only when $f(x)$ is real-valued.

Hessian is NOT the gradient of the gradient!

Note that Hessian of a function $f:\mathbb{R}^{n}\to \mathbb{R}$ is the Jacobian of its gradient, i.e., $H(f(x)) = J(\nabla f(x))^{T}$.

The second derivatives tells. us how the first derivative will change as we vary the input. We can think of the second derivative as measuring the curvature. 

#### What is an identity matrix?

identity matrix, $I \in \mathbb{R}^{n \times n}$, is a square matrix with ones on the diagonal and zeros everywhere else.

$$
    I_{ij} = \left\{ \begin{array}{ll}
         1 & \mbox{if $i=j$};\\
        0 & \mbox{if $i \neq j$}.\end{array} \right.
$$

It has the property that for all $A \in \mathbb{R}^{m \times n}$

$$
    AI = IA = A
$$

Generally the dimensions of $I$ are inferred from context so as to make matrix multiplication possible. 

#### What is the transpose of a matrix?

The transpose of a matrix results from "flipping" the rows and columns. Given a matrix $A \in \mathbb{R}^{m \times n}$, its transpose, written as $A^{T} \in \mathbb{R}^{n \times m}$, is the matrix whose entries are given by $\left( A_{ij}^{T} \right) = A_{ji}$.

The following properties of transposes are easily verified:
1. $\left(A^{T}\right)^{T} =A$
2. $\left(AB\right)^{T} = B^{T}A^{T}$
3. $\left(A + B\right)^{T} = A^{T} + B^{T}$

#### What is an inverse matrix?

The inverse of a square matrix $A$, sometimes called a reciprocal matrix, is a matrix $A^{-1}$ such that

$$
AA^{-1} = I = A^{-1}A
$$

where $I$ is the identity matrix. 
Note that for square matrices, the left inverse and right inverse are equal.

Non-square matrices do not have inverses by definition. Note that not all square matrices have inverses. A square matrix which has an inverse is called invertible or non-singular, and a square matrix without an inverse is called non-invertible or singular.

#### When does inverse of a matrix exist?

**Determine its rank**. In order for a square matrix $A$ to have an inverse $A^{-1}$, then $A$ must be full rank. The rank of a matrix is a unique number associated with a square matrix. The rank of a matrix is the number of non-zero eigenvalues
of the matrix. The rank of a matrix gives the dimensionality of the Euclidean space which can be used to represent this matrix. Matrices whose rank is equal to their dimensions are full rank and they are invertible. When the rank of a matrix is smaller than its dimensions, the matrix is not invertible and is called rank-deficient, singular, or multicolinear. For example, if the rank of an $n \times n$ matrix is less than $n$, the matrix does not have an inverse. 

**Compute its determinant**. The determinant is another unique number associated with a square matrix. The determinant det(A) of a square matrix A is the product of its eigenvalues. When the determinant for a square matrix is equal to zero, the inverse for that matrix does not exist.

$A, B \in \mathbb{R}^{n\times n}$ are non-singular. 
1. $(A^{-1})^{-1} = A$
2. $(AB)^{-1} = B^{-1}A^{-1}$
3. $(A^{-1})^{T} = (A^{T})^{-1}$

#### If inverse of a matrix exists, how to calculate it?

$$
\begin{split}
 Ax &= b\\
A^{-1}Ax &= A^{-1}b\\
I_{n}x &= A^{-1}b\\
x &= A^{-1}b  
\end{split}
$$

where $I_{n} \in \mathbb{R}^{n\times n}$

For example, for $2 \times 2$ matrix, the inverse is:

$$
    \begin{bmatrix}
    a & b \\
    c & d
    \end{bmatrix}^{-1} = \dfrac{1}{ad - bc}  \begin{bmatrix}
    d & -b \\
    -c & a
    \end{bmatrix}
$$

where $ad-bc$ is the determinant of this matrix. In other words, swap the positions of $a$ and $d$, put the negatives in front of $b$ and $c$ and divide everything by determinant. $AA^{-1} = I = A^{-1}A$ should be satisfied.

Now let's find the inverse of a bigger matrix, which is $3 \times 3$:

$$
    \begin{bmatrix}
    1 & 3 & 3 \\
    1 & 4 & 3 \\
    1 & 3 & 4
    \end{bmatrix}
$$

First, we write down the entries the matrix, but we write them in a double-wide matrix:

$$
    \begin{bmatrix}
    1 & 3 & 3 & | &  &  &  \\
    1 & 4 & 3 & | &  &  & \\
    1 & 3 & 4 & | &  &  & 
    \end{bmatrix}
$$

In the other half of the double-wide, we write the identity matrix:

$$
    \begin{bmatrix}
    1 & 3 & 3 & | & 1 & 0 & 0 \\
    1 & 4 & 3 & | & 0 & 1 & 0\\
    1 & 3 & 4 & | & 0 & 0 & 1
    \end{bmatrix}
$$

Now we'll do matrix row operations to convert the left-hand side of the double-wide into the identity. (As always with row operations, there is no one "right" way to do this. What follows are just one way. Your calculations could easily look quite different.)

$$
\begin{split}
    \begin{bmatrix}
    1 & 3 & 3 & | & 1 & 0 & 0 \\
    1 & 4 & 3 & | & 0 & 1 & 0\\
    1 & 3 & 4 & | & 0 & 0 & 1
    \end{bmatrix}&\underset{\overset{-r_{1}+r_{2}}{\longrightarrow}}{\overset{-r_{1}+r_{3}}{\longrightarrow}}
    \begin{bmatrix}
    1 & 3 & 3 & | & 1 & 0 & 0 \\
    0 & 1 & 0 & | & -1 & 1 & 0\\
    0 & 0 & 1 & | & -1 & 0 & 1
    \end{bmatrix}\\ &\overset{-3r_{2}+r_{1}}{\longrightarrow}
    \begin{bmatrix}
    1 & 0 & 3 & | & 4 & -3 & 0 \\
    0 & 1 & 0 & | & -1 & 1 & 0\\
    0 & 0 & 1 & | & -1 & 0 & 1
    \end{bmatrix}\\ &\overset{-3r_{3}+r_{1}}{\longrightarrow}
    \begin{bmatrix}
    1 & 0 & 0 & | & 7 & -3 & -3 \\
    0 & 1 & 0 & | & -1 & 1 & 0\\
    0 & 0 & 1 & | & -1 & 0 & 1
    \end{bmatrix}
    \end{split}
$$

Now that the left-hand side of the double-wide contains the identity, the right-hand side contains the inverse. That is, the inverse matrix is the following:

$$
\begin{bmatrix}
    7 & -3 & -3 \\
    -1 & 1 & 0\\
    -1 & 0 & 1
    \end{bmatrix}
$$

#### What is the determinant of a square matrix? How is it calculated? What is the connection of determinant to eigenvalues?

The determinant of a square matrix, denoted $det(A)$, is a function that maps matrices to real scalars, i.e., $det: \mathbb{R}^{n \times n} \to \mathbb{R}$. The determinant is equal to the product of all the eigenvalues of the matrix. The absolute value of the determinant can be thought of as a measure of how much multiplication by the matrix expands or contracts space. If the determinant is 0, then space is contracted completely along at least one dimension, causing it to lose all its volume. If the determinant is 1, then the transformation preserves volume.

The determinant is a real number, it is not a matrix. The determinant can be a negative number. The determinant only exists for square matrices. The determinant of a $1 \times 1$ matrix is that single value in the determinant. The inverse of a matrix will exist only if the determinant is not zero.

For example let's find a determinant of a $3 \times 3$ matrix:

$$
A = \begin{bmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33} 
\end{bmatrix}
$$

$$
\begin{split}
det(A) = |A| &=
a_{11} \begin{bmatrix}
a_{22} & a_{23} \\
a_{32} & a_{33}
\end{bmatrix}
- a_{12}\begin{bmatrix}
a_{21} &  a_{23} \\
a_{31} & a_{33} 
\end{bmatrix}
+ a_{13}\begin{bmatrix}
a_{21} & a_{22} \\
a_{31} & a_{32}
\end{bmatrix}\\
&= a_{11}\times \left(a_{22}a_{33}-a_{23}a_{32} \right)
-a_{12}\left(a_{21}a_{33} -a_{23}a_{31}  \right)
+a_{13} \left(a_{21}a_{32} - a_{22}a_{31}\right)
\end{split}
$$

It is the similar idea for $4\times 4$ and higher matrices.

You do not have to use the first row to compute the determinant. You can use any row or any column, as long as you know where to put the plus and minus signs.

$$
A = \begin{bmatrix}
+ & - & + & \cdots \\
- & + & - & \cdots \\
+ & - & + & \cdots \\
\vdots & \vdots & \vdots &  \cdots
\end{bmatrix}
$$

For example, 

$$
A = \begin{bmatrix}
1 & 0 & 2 & -1 \\
3 & 0 & 0 & 5 \\
2 & 1 & 4 & -3 \\
1 & 0  & 5 & 0
\end{bmatrix}
$$

The second column of this matrix has a lot of zeros.

$$
\begin{split}
det(A) = |A| &=
-0 \begin{bmatrix}
3  & 0 & 5 \\
2  & 4 & -3 \\
1  & 5 & 0
\end{bmatrix}
+0\begin{bmatrix}
1  & 2 & -1 \\
2  & 4 & -3 \\
1  & 5 & 0
\end{bmatrix}
-1\begin{bmatrix}
1 & 2 & -1 \\
3 & 0 & 5 \\
1 & 5 & 0
\end{bmatrix}
+0 \begin{bmatrix}
1 & 2 & -1 \\
3 & 0 & 5 \\
2 & 4 & -3 \\
\end{bmatrix}\\
&= -1\begin{bmatrix}
1 & 2 & -1 \\
3 & 0 & 5 \\
1 & 5 & 0
\end{bmatrix}\\
&=-1\left( 1\begin{bmatrix}
0 & 5 \\
5 & 0
\end{bmatrix}-2\begin{bmatrix}
3  & 5 \\
1  & 0
\end{bmatrix}+(-1)\begin{bmatrix}
3 & 0 \\
1 & 5 
\end{bmatrix} \right)\\
&=-1(-25+10-15)\\
&=30
\end{split}
$$

#### What is a norm? What is $L^{1}$, $L^{2}$ and $L^{\infty}$ norm? What are the conditions a norm has to satisfy?

Sometimes, we need to measure the size of a vector (length of the vector). In machine learning, we usually measure the size of vectors using a function called a norm. Formally, assuming $x$ is a vector and $x_{i}$ is its $i$th-element, the $L^{p}$ norm is given by

$$
    \lVert x \rVert_{p} = \left(\sum_{i=1}\lvert x_{i} \rvert^{p}  \right)^{1/p}
$$

for $p \in \mathbb{R}$, and $p \geq 1$.

* $L^{1}$ is known as Manhattan Distance (norm).
* $L^{2}$ is known as Euclidean Distance (norm) which gives the magnitude of a vector. However, confusion is that the Frobenius norm (a matrix norm) is also sometimes called the Euclidean norm.
* $L^{\infty} = \underset{i}{\max} \lvert x_{i} \rvert$ also known as the max norm (sup norm). This norm simplifies to the absolute value of the element with the largest magnitude in the vector.

The higher the norm index, the more it focuses on large values and neglects small ones. This is why the Root Mean Squared Error (RMSE, which corresponds to Euclidean norm) is more sensitive to outliers than Mean Absolute Error (MAE which corresponds to Manhattan norm). But when outliers are exponentially rare (like in a bell-shaped curve), the RMSE performs very well and is generally preferred.

Norms, including the $L^{p}$ norm, are functions mapping vectors to non-negative values. On an intuitive level, the norm of a vector $x$ measures the distance from the origin to the point $x$. More rigorously, a norm is any function $f$ that satisﬁes the following properties

* $f(x) = 0 \Rightarrow x = 0$ (Definiteness)
* $f(x) \geq 0$ (non-negativity)
* $f(x + y) \leq f (x) + f(y)$ (the triangle inequality)
* $\forall \alpha \in \mathbb{R}, f( \alpha x) = \lvert \alpha \rvert f(x)$ (homogenity)

Note that any valid norm $\lVert \cdot \rVert$ is a convex function. We can prove it using the triangle inequality and homogeneity of the norm for any $x, y \in \mathbb{R}^{n}$ and any $\theta \in (0, 1)$:

$$
\lVert \theta x + (1 - \theta) y \rVert \leq \lVert \theta x \rVert + \lVert (1-\theta) y \rVert = \theta \lVert x \rVert + (1 - \theta) \lVert y \rVert
$$

#### What is Frobenius norm?

Sometimes we may also wish to measure the size of a matrix. In the context of deep learning, the most common way to do this is with the otherwise obscure Frobenius norm.

The Frobenius norm, sometimes also called the Euclidean norm (a term unfortunately also used for the vector $L^{2}$-norm), is matrix norm of an $m \times n$ matrix $A$ defined as the square root of the sum of the squares of its elements:

$$
\lVert A \rVert_{F} = \sqrt{\sum_{i=1}^{m}\sum_{j=1}^{n} a_{ij}^{2}}
$$

which is analogous to the $L^{2}$-norm of a vector

#### What is a diagonal matrix?

Diagonal matrices consist mostly of zeros and have nonzero entries only along the main diagonal. Formally, a matrix $D$ is diagonal if and only if $D_{i,j}= 0$ for all $i \neq j$.

$$
    D_{ij} = \left\{ \begin{array}{ll}
         d_{i} & \mbox{if $i=j$};\\
        0 & \mbox{if $i \neq j$}.\end{array} \right.
$$

Identity matrix, where all the diagonal entries are 1 is an example of a diagonal matrix.  Clearly, $I = \text{diag}(1,1,1,...,1)$.

Not all diagonal matrices need be square. It is possible to construct a rectangular diagonal matrix.

$$
\begin{bmatrix} 
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix} \begin{bmatrix} 
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0
\end{bmatrix} \begin{bmatrix} 
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1 \\
0 & 0 & 0
\end{bmatrix}
$$

Entries in $i=j$ can technically be called the "main diagonal" of the rectangular matrix, though the diagonal of such a matrix is not necessarily as "useful" as it is in a square matrix. 

#### What is a symmetric matrix?

A symmetric matrix is any matrix that is equal to its own transpose:

$$
A = A^{T}
$$

It is anti-symmmetric if $A = -A^{T}$

#### What is the Moore Penrose pseudo inverse and how to calculate it?

Not all matrices have an inverse. It is unfortunate because the inverse is used to solve system of equations. In some cases, a system of equation has no solution, and thus the inverse doesn’t exist. However it can be useful to find a value that is almost a solution (in term of minimizing the error). The Moore-Penrose pseudoinverse is a direct application of the SVD.

The inverse of a matrix $A$ can be used to solve the equation $Ax = b$:

$$
\begin{split}
A^{-1} Ax &= A^{-1}b\\
I_n x &= A^{-1}b\\
x &= A^{-1}b
\end{split}
$$

But in the case where the set of equations have 0 or many solutions the inverse cannot be found and the equation cannot be solved. The pseudoinverse is $A^{+}$ such as:

$$
A A^{+} \approx I_n
$$

minimizing

$$
\lVert A A^{+} - I_n \rVert_{2}
$$

The following formula can be used to find the pseudoinverse:

$$
A^+= V D^{+} U^{T}
$$

with $U$, $D$ and $V$ respectively the left singular vectors, the singular values and the right singular vectors of $A$.

$A^{+}$ is the pseudoinverse of $A$ and $D^{+}$ the pseudoinverse of $D$. We saw that $D$ is a diagonal matrix (Remember that $D$ are the singular values that need to be put into a diagonal matrix) and thus $D^{+}$ can be calculated by taking the reciprocal of the non zero values of $D$.

For example, let's find a non square matrix A, calculate its singular value decomposition and its pseudoinverse.

$$A = \begin{bmatrix} 7 & 2\\\\ 3 & 4\\\\ 5 & 3 \end{bmatrix}$$

{% highlight python %}
import numpy as np

A = np.array([[7, 2], [3, 4], [5, 3]])
U, D, V = np.linalg.svd(A)

# U 
# array([[-0.69366543,  0.59343205, -0.40824829],
#        [-0.4427092 , -0.79833696, -0.40824829],
#        [-0.56818732, -0.10245245,  0.81649658]])

# D
# array([10.25142677,  2.62835484])

# V
# array([[-0.88033817, -0.47434662],
#        [ 0.47434662, -0.88033817]])

D_plus = np.zeros((A.shape[0], A.shape[1])).T
D_plus[:D.shape[0], :D.shape[0]] = np.linalg.inv(np.diag(D))

A_plus = V.T.dot(D_plus).dot(U.T)
A_plus
# array([[ 0.16666667, -0.10606061,  0.03030303],
#        [-0.16666667,  0.28787879,  0.06060606]])
{% endhighlight %}

We can now check with the `pinv()` function from Numpy that the pseudoinverse is correct:

{% highlight python %}
np.linalg.pinv(A)
# array([[ 0.16666667, -0.10606061,  0.03030303],
#        [-0.16666667,  0.28787879,  0.06060606]])
{% endhighlight %}

It looks good! We can now check that it is really the near inverse of A. Since we know that

$$A^{-1} A = I_{n}$$

$$I_{2}=\begin{bmatrix} 1 & 0 \\\\ 0 & 1 \end{bmatrix}$$

{% highlight python %}
A_plus.dot(A)
# array([[1.00000000e+00, 2.63677968e-16],
#        [5.55111512e-17, 1.00000000e+00]])
{% endhighlight %}

This is not bad! This is almost the identity matrix!

A difference with the real inverse is that $A^{+} A \approx I$ but $A A^{+} \neq I$. 

Another way of computing the pseudoinverse is to use this formula:

$$(A^{T} A)^{−1}A^{T}$$

This formula comes from the fact that $A$ is a tall matrix.

The result is less acurate than the SVD method and Numpy `pinv()` uses the SVD. Here is an example from the same matrix $A$:

{% highlight python %}
A_plus_1 = np.linalg.inv(A.T.dot(A)).dot(A.T)
A_plus_1
# array([[ 0.16666667, -0.10606061,  0.03030303],
#        [-0.16666667,  0.28787879,  0.06060606]])
{% endhighlight %}

In this case the result is the same as with the SVD way.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/tall_fat_inverse.png?raw=true)

#### What is the trace of a matrix?

The trace operator gives the sum of all the diagonal entries of a matrix:

$$
Tr(A) = \sum_{i} A_{i,i}
$$

It is easy to show that the trace is a linear map, so that

$$
Tr(\lambda A) = \lambda Tr(A) = \lambda \sum_{i} A_{i,i}
$$

and 

$$
Tr(A + B) = Tr(A) + Tr(B)
$$

The trace operator is invariant to the transpose operator: $Tr(A) = Tr(A^{T})$. 

#### How to write Frobenius norm of a matrix A in terms of trace?

The trace operator provides an alternative way of writing the Frobenius norm of a matrix:

$$
\lvert A \rvert_{F} = \sqrt{tr(A A^{T})} = \sqrt{tr(A^{T} A)}
$$

{% highlight python %}
import numpy as np

A = np.array([[2, 9, 8], [4, 7, 1], [8, 2, 5]])

A_tr = np.trace(A)
A_tr
# 14

np.linalg.norm(A)
#The Frobenius norm of A is 17.549928774784245.

# With the trace the result is identical:
np.sqrt(np.trace(A.dot(A.T)))
# 17.549928774784245
{% endhighlight %}

Since the transposition of a matrix doesn’t change the diagonal, the trace of the matrix is equal to the trace of its transpose:

$$
Tr(A) = Tr( A^T)
$$

#### What is eigendecomposition, eigenvectors and eigenvalues? How to find eigenvalues of a matrix?

In linear algebra, eigendecomposition or sometimes spectral decomposition is the factorization of a matrix into a canonical form, whereby the matrix is represented in terms of its eigenvalues and eigenvectors. Only diagonalizable matrices can be factorized in this way. The eigen-decomposition can also be use to build back a matrix from it eigenvectors and eigenvalues. For details look [here](#what-is-spectral-decomposition-eigendecompoisition){:target="_blank"}.

Given a $p \times p$ matrix $A$, the real number $u$ and the vector $v$ are an eigenvalue and corresponding eigenvector of $A$ if

$$
Av = uv
$$

Here, $u$ is a scalar (which may be either real or complex).  Any value of $u$ for which this equation has a solution is known as an eigenvalue of the matrix $A$. It is sometimes also called the characteristic value.  The vector, v, which corresponds to this value is called an eigenvector.  The eigenvalue problem can be rewritten as 

The eigenvalue problem can be rewritten as:

$$
\begin{split}
Av &= uv \\
Av - uIv &= 0 \\
(A-uI)v &= 0 \\
\end{split}
$$

where $I$ is the $p \times p$ identity matrix. Now, in order for a non-zero vector $v$ to satisfy this equation, $A − uI$ must not be invertible, i.e.,

$$
\text{det} (A - uI) = 0
$$

This equation is known as the characteristic equation of A, and the left-hand side is known as the characteristic polynomial, and is an $p$-th order polynomial in $u$ with $p$ roots. These roots are called the eigenvalues of $A$.  We will only deal with the case of $p$ distinct roots, though they may be repeated.  For each eigenvalue there will be an eigenvector for which the eigenvalue equation is true.  

#### What is Spectral Decomposition (Eigendecompoisition)?

Spectral decomposition, sometimes called _eigendecomposition_, recasts a real symmetric $p \times p$ matrix $A$ with its eigenvalues $u_{1}, u_{2}, \ldots, u_{p}$ and corresponding orthonormal eigenvectors $v_{1}, v_{2}, \ldots, v_{p}$, then, 

$$
\begin{split}
A &= \underbrace{\begin{bmatrix} \uparrow & \uparrow & \ldots & \uparrow \\
v_{1} & v_{2} & \ldots &  v_{p} \\
\downarrow & \downarrow & \ldots & \downarrow \\
\end{bmatrix}}_{\text{Call this Q}}\underbrace{\begin{bmatrix}
u_{1} & 0 & \ldots & 0 \\
0 & u_{2} & \ldots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \ldots & u_{p}\\
\end{bmatrix}}_{\Lambda}\underbrace{\begin{bmatrix}
\leftarrow & v_{1} & \rightarrow \\
\leftarrow & v_{2} & \rightarrow \\
\ldots & \ldots & \ldots \\
\leftarrow & v_{p} & \rightarrow \\
\end{bmatrix}}_{Q^{T}} \\
&= Q \Lambda Q^{T}
\end{split}
$$

or

$$
A = \sum_{i=1}^{p} u_{i} \mathbf{v}_{i} \mathbf{v}_{i}^{T}
$$

Eigendecomposition can only exists for square matrices, and even among square matrices sometimes it doesn't exist.

#### What is Singular Value Decomposition?
The data matrix $X \in \mathbb{R}^{n \times p}$ can be decomposed in a singular-value decomposition (SVD) as

$$
X_{n \times p} = U_{n\times n} D_{n \times p} V_{p\times p}^{T}
$$

where

1. $U$ is orthogonal in $\mathbb{R}^{n}$: $U^{T}U = U U^{T} = 1_{n \times n}$.
2. $V$ is orthogonal in $\mathbb{R}^{p}$: $V^{T}V = V V^{T} = 1_{p \times p}$.
3. D diagonal in $\mathbb{R}^{n \times p}$.

If $n > p$, the decomposition looks like:

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/sv_decomposition.png?raw=true)

where

1. the columns of $U$ contain the left-singular vectors $u_{1}, \cdots, u_{n}$ that form an orthonormal basis of $\mathbb{R}^{n}$.
2. the columns of $V$ contain the right-singular vectors $v_{1}, \cdots, v_{n}$ that form an orthonormal basis of $\mathbb{R}^{p}$.
3. and the diagonal elements of $D$ (usually denoted by $d_{1} \geq d_{2} \geq \cdots d_{min(n,p)}$ where $d_{i} = D_{ii}\,\,\, for\,\,\, i = 1, \cdots , min(n, p)$) contain the non-negative singular values which are ordered in decreasing magnitude.

Calculating the SVD of $X$ consists of finding the eigenvalues and eigenvectors of $X^{T}X$ and $XX^{T}$. 

The second moment $X^{T}X \in \mathbb{R}^{p \times p}$ is sometimes called the Gram matrix and $X^{T}X / (n-1)$ is equal to the empirical covariance if the columns of $X$ are mean-centered. It can be decomposed, using the SVD of $X$, as

$$
\begin{split}
X^{T}X &= \left(U D V^{T}\right)^{T} \left(U D V^{T}\right)\\
&= V D^{T} \underset{1_{n \times n}}{\underbrace{U^{T} U}} D V^{T}\\
&= V D^{T} D V^{T}
\end{split}
$$

where 

$$
D^{T} D = \begin{bmatrix}
    D_{11}^{2} & & &\\
    & D_{22}^{2} & &\\
    & & \ddots &\\
    & & & D_{pp}^{2}
  \end{bmatrix}
$$

The eigenvalue decomposition of $X^{T}X \in \mathbb{R}^{p \times p}$ is hence given by

$$
X^{T}X = V \Lambda V^{T}
$$

where the orthogonal matrix $V \in  \mathbb{R}^{p \times p}$ is identical to the matrix in the SVD of $X$ and $\Lambda \in  \mathbb{R}^{p \times p}$ is the diagonal matrix containing the eigenvalues.

$$ \lambda_{k} := \Lambda_{kk} = D_{kk}^{2}\,\,\, for\,\,\, k = 1, 2, \cdots, p$$

The eigenvectors of $X^{T}X$ make up the columns of $V$, the eigenvectors of $XX^{T}$ make up the columns of $U$. Also, the singular values in $D$ are square roots of eigenvalues from $X^{T}X$ or $XX^{T}$. The singular values are the diagonal entries of the $D$ matrix and are arranged in descending order. The singular values are always real numbers. If the matrix $X$ is a real matrix, then $U$ and $V$ are also real.

{% highlight python %}
import numpy as np
X = np.random.rand(50,3)

eigenvalues_XTX, eigenvectors_XTX = np.linalg.eig(X.T @ X)
eigenvalues_XTX.shape, eigenvectors_XTX.shape
# ((3,), (3, 3))

U, s, Vt = np.linalg.svd(X, full_matrices=True)
U.shape, s.shape, Vt.shape
#((50, 50), (3,), (3, 3))

eigenvalues_XTX
#array([44.10686679,  3.50061913,  4.88378011])

s
#array([6.64130008, 2.20992762, 1.87099416])

np.sqrt(eigenvalues_XTX)
#array([6.64130008, 1.87099416, 2.20992762])

Vt.T
# array([[-0.57735925,  0.08795263, -0.81173927],
#        [-0.60991707,  0.61449879,  0.50039225],
#        [-0.54282361, -0.78399973,  0.30114275]])

eigenvectors_XTX
# array([[ 0.57735925,  0.81173927, -0.08795263],
#        [ 0.60991707, -0.50039225, -0.61449879],
#        [ 0.54282361, -0.30114275,  0.78399973]])
{% endhighlight %}

#### What is the trace of a scalar?

A scalar is its own trace $a=Tr(a)$

#### What do positive definite, positive semi-definite and negative definite/negative semi-definite mean?

A matrix $A$ is positive semi-definite if it is symmetric and all its eigenvalues are non-negative. If all eigenvalues are strictly positive then it is called a positive definite matrix.

A square symmetric matrix $A \in  \mathbb{R}^{n \times n}$ is positive semi-definite if 

$$
v^{T} A v \geq 0,\,\,\, \forall v \in \mathbb{R}^{n \times 1}
$$ 

and positive definite if the inequality holds with equality only for vectors $v=0$, i.e., $v^{T} A v > 0,\,\,\, \forall v \in \mathbb{R}^{n \times 1}$.

A square symmetric matrix $A \in  \mathbb{R}^{n \times n}$ is negative semi-definite if

$$
v^{T} A v \leq 0,\,\,\, \forall v \in \mathbb{R}^{n \times 1}
$$ 

and negative definite if the inequality holds with equality only for vectors $v=0$, i.e., $v^{T} A v < 0,\,\,\, \forall v \in \mathbb{R}^{n \times 1}$

Positive (semi)definite and negative (semi)definite matrices together are called definite matrices. A symmetric matrix that is not definite is said to be indefinite. 

A symmetric matrix is positive semi-definite if and only if all eigenvalues are non-negative. It is negative semi-definite if and only if all eigenvalues are non-positive. It is positive definite if and only if all eigenvalues are positive. It is negative definite if and only if all eigenvalues are negative.

The matrix $A$ is positive sem-definite if any only if $−A$ is negative semi-definite, and similarly a matrix $A$ is positive definite if and only if $−A$ is negative definite.

Now, let's see how we can use the quadratic form to check the positive definiteness:

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/pd_nd_psd_nsd.png?raw=true)

To check if the matrix is positive definite/positive semi-definite/negative definite/negative semi-definite or not, you just have to compute the above quadratic form and check if the value is strictly positive/positive/strictly negative or negative.


Let's prove some of the statements above for positive definite matrix. 

* If a real symmetric matrix is positive definite, then every eigenvalue of the matrix is positive.
  Let's say here $v \in \mathbb{R}^{n}$ is the eigenvector and $u \in \mathbb{R}$ is eigenvalue. Using eigenvalue equation,
  
  $$
\begin{split}
A v = u v &\Rightarrow v^{T} A v = u \left(v^{T} v \right)\\
&\Rightarrow u = \dfrac{v^{T} A v}{v^{T} v} = \dfrac{v^{T} A v}{\left\Vert v \right\Vert_{2}^{2}}
\end{split}
$$
  
  Since $A$ is positive definite, its quadratic form is positive, i.e., $v^{T} A v > 0$. $v$ is a nonzero vector as it is an eigenvector. Since $\left\Vert v \right\Vert_{2}^{2}$ is positive, we must have $u$ is positive, which is $u > 0$.

* If every eigenvalue of a real symmetric matrix is positive, then the matrix is positive definite.

  By the spectral theorem, a real symmetric matrix has an eigenvalue decomposiiton, so,
  
  $$
  A = Q \Lambda Q^{T}
  $$
  
  For the quadratic function defined by $A$:
  
  $$
  v^{T} A v = \underbrace{v^{T} Q}_{y^{T}} \Lambda \underbrace{Q^{T} v}_{y} = \sum_{i=1}^{n} u_{i}y_{i}^{2}
  $$
  
  Since eigenvalues $u_{i}$'s are positive and $y_{i}^{2} > 0$, this summation is always positive, therefore, $v^{T} A v > 0$.

#### How to make a positive definite matrix with a matrix that’s not symmetric?

The problem with definite matrices is that they are not always symmetric. However, we can simply multiply the matrix that’s not symmetric by its transpose and the product will become symmetric, square, and positive definite!

Let's say the matrix $B \in \mathbb{R}^{m\times n}$. Then, $B^{T}B \in  \mathbb{R}^{n\times n}$ which is a square matrix in real space. If $v^{T}B^{T}Bv = \left( Bv\right)^{T}\left( Bv\right) = \left\Vert Bv \right\Vert_{2}^{2} > 0$, then $B^{T}B$ is positive definite matrix.

#### How is the set of positive semi-definite matrices a convex set?

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Screen%20Shot%202019-11-23%20at%2010.18.51.png?raw=true)

#### What are the one-dimensional and multi-dimensional Taylor expansions?

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Taylor_expansion_single_variable.jpeg?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Taylor_Expansion_multiple_variable.jpeg?raw=true)


## Numerical Optimization

#### What is underflow and overflow?

Machine learning algorithms usually require a high amount of numerical computation. This typically refers to algorithms that solve mathematical problems by methods that update estimates of the solution via an iterative process, rather than analytically deriving a formula to provide a symbolic expression for the correct solution. Common operations include optimization (ﬁnding the value of an argument that minimizes or maximizes a function) and solving systems of linear equations. Even just evaluating a mathematical function on a digital computer can be diﬃcult when the function involves real numbers, which cannot be represented precisely using a ﬁnite amount of memory.

The fundamental difficulty in performing continuous math on a digital computeris that we need to represent infinitely many real numbers with a finite number of bit patterns. This means that for almost all real numbers, we incur some approximation error when we represent the number in the computer. In many cases, this is just rounding error. Rounding error is problematic, especially whenit compounds across many operations, and can cause algorithms that work in theory to fail in practice if they are not designed to minimize the accumulation of rounding error

One form of rounding error that is particularly devastating is underflow. Underflow occurs when numbers near zero are rounded to zero. Many functions behave qualitatively differently when their argument is zero rather than a small positive number. For example, we usually want to avoid division by zero (some software environments will raise exceptions when this occurs, others will return a result with a placeholder not-a-number value) or taking the logarithm of zero (thisis usually treated as $-\infty$, which then becomes not-a-number if it is used for many further arithmetic operations)

Another highly damaging form of numerical error is overﬂow. Overﬂow occurs when numbers with large magnitude are approximated as $\infty$ or $- \infty$. Further arithmetic will usually change these inﬁnite values into not-a-number values.

#### How to tackle the problem of underflow or overflow for softmax function or log softmax function?

The softmax function is often used to predict the probabilities associated with a multinoulli distribution. The softmax function is deﬁned to be:

$$
softmax(\mathbf{x})_{i} = \frac{exp(x_{i})}{\sum_{j=1}^{n} exp(x_{j})}
$$

However, Softmax function is prone to two issues: overflow and underflow.

* **Overflow**: It occurs when very large numbers are approximated as infinity
* **Underflow**: It occurs when very small numbers (near zero in the number line) are approximated (i.e. rounded to) as zero

To combat these issues when doing softmax computation, a common trick is to shift the input vector by subtracting the maximum element in it from all elements.

Subtracting $max_{i} x_{i}$ results in the largest argument to $exp$ being 0, which rules out the possibility of overﬂow. Likewise, at least one term in the denominator has a value of 1, which rules out the possibility of underﬂow in the denominator leading to a division by zero.

This is done for stability reasons: when you exponentiate even large-ish numbers, the result can be quite large. numpy will return `inf` when you exponentiate values over 710 or so which can be seen in the small code below:

```python
import numpy as np

def softmax(w):
    """Calculate the softmax of a list of numbers w.

    Parameters
    ----------
    w : list of numbers

    Return
    ------
    a list of the same length as w of non-negative numbers
    """
    e = np.exp(np.array(w))
    softmax_result = e / np.sum(e)
    return softmax_result

softmax([0.1, 0.2])
#array([0.47502081, 0.52497919])

softmax([710, 800, 900])
#array([nan, nan, nan])
```

So let's implement it in pure Python:

```python
import numpy as np

def softmax(x):
    z = np.array(x) - np.max(np.array(x), axis=-1, keepdims=True)
    numerator = np.exp(z)
    denominator = np.sum(numerator, axis=-1, keepdims=True)
    softmax_result = numerator / denominator
    return softmax_result

print(softmax([0.1, 0.2]))
#[0.47502081 0.52497919]

print(softmax([710, 800, 900]))
#[0. 0. 1.]
```

So that solves the numerical stability problem, but is it mathematically correct? To clear this up, let's write out the softmax equation with the subtraction terms in there.

$$
softmax(\mathbf{x})_{i} = \frac{exp(x_{i} - max(\mathbf{x}))}{\sum_{j=1}^{n} exp(x_{j} - max(\mathbf{x}))}
$$

Subtracting within an exponent is the same as dividing between exponents ($e^{a-b} = e^a / e^b$), so:

$$
\frac{exp(x_{i}) / max(\mathbf{x})}{\sum_{j=1}^{n} exp(x_{j}) / max(\mathbf{x})}
$$

Then you just cancel out the maximum terms, and you're left with the original equation:

$$
\frac{exp(x_{i})}{\sum_{j=1}^{n} exp(x_{j})}
$$

There is still one small problem. Underﬂow in the numerator can still cause the expression as a whole to evaluate to zero. This means that if we implement log softmax(x) by ﬁrst running the softmax subroutine then passing the result to the log function (`log( exp(x_i) / exp(x).sum() )`), we could erroneously obtain $- \infty$. Instead, we must implementa separate function that calculates log softmax in a numerically stable way. The log softmax function can be stabilized using the same trick as we used to stabilize the softmax function.

#### What is second derivative test?

The second derivative test can be used to determine whether a critical point is a local maximum, local minimum, or a saddle point. Suppose $f(x)$ is a function of $x$ that is twice differentiable at a stationary point $x_0$. Recall that on a critical point $f^{\prime} (x_0) = 0$.

1. If $f^(\prime \prime)(x_0) > 0$, then $f$ has a local minimum at $x_0$.
2. If $f^(\prime \prime)(x_0)<0$, then f has a local maximum at $x_0$.

Unfortunately, when $f^(\prime \prime)(x_0) = 0$, the test is inconclusive.

#### Describe convex function.

A function $f(x): M \rightarrow \mathbb{R}$, defined on a nonempty subset $M$ of $\mathbb{R}^{n}$ and taking real values, is convex on an interval $[a,b]$ if for any two points $x_1$ and $x_2$ in $[a,b]$ and any $\lambda$ where $0< \lambda < 1$,

* the domain $M$ of the function is convex, meaning it is a convex set if it contains all convex combinations of any two points within it.

and 

* $$
f[\lambda x_{1} + (1 - \lambda) x_{2}] \leq \lambda f(x_{1}) + (1 - \lambda) f(x_{2}) 
$$

If $f(x)$ has a second derivative in $[a,b]$, then a necessary and sufficient condition for it to be convex on on the interval $[a,b]$ is that the second derivative $f^{''}(x) \geq 0$ for all $x$ in $[a,b]$. However, the converse need not be true. (i.e., take the first derivative of the function and set it to zero. You will obtain the critical value. It could be either a minimum or a maximum point or a point where the derivative changes the sign. So to know whether the point is local minimum or local maximum, we will have to take the second derivative in order to be 100% sure. When the second derivative is positive we have a local minimum point. When the second derivative is negative we have a local maximum point).

The prototypical convex function is shaped something like the letter U.

If the inequality above is strict for all $x_{1}$ and $x_{2}$, then $f(x)$ is called strictly convex.

* $$
f[\lambda x_{1} + (1 - \lambda) x_{2}] < \lambda f(x_{1}) + (1 - \lambda) f(x_{2}) 
$$

An inequality is strict if replacing any "less than" and "greater than" signs with equal signs never gives a true expression. For example, $a \leq b$ is not strict, whereas $a < b$ is.

Some convex function examples are shown below:

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/convex_func_examples.png?raw=true)

By contrast, the following function is not convex. A non-convex function is wavy - has some 'valleys' (local minima) that aren't as deep as the overall deepest 'valley' (global minimum). Optimization algorithms can get stuck in the local minimum, and it can be hard to tell when this happens.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/nonconvex_func_example.png?raw=true)

Notice how the region above the graph is not a convex set. A set is convex if, given any two points in the set, the line segment connecting them lies entirely inside the set.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/2-Figure1-1.png?raw=true)

A strictly convex function has exactly one local minimum point, which is also the global minimum point. The classic U-shaped functions are strictly convex functions. However, some convex functions (for example, straight lines) are not U-shaped.

If the functions $f$ and $g$ are convex, then any linear combination $a f + b g$ where $a$, $b$ are positive real numbers is also convex.

The introduced concept of convexity has a simple geometric interpretation. Geometrically, the line segment connecting $(x_{1}, f(x_{1}))$ to $(x_{2}, f(x_{2}))$ must sit above the graph of $f$ and never cross the graph itself.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/graphic_convex.jpeg?raw=true)

If a function has the opposite property, namely that every chord lies on or below the function, it is called concave, with a corresponding definition for strictly concave. If a function $f(x)$ is convex, then $−f(x)$ will be concave.

#### What are the Karush-Kuhn-Tucker conditions?

The SVM problem can be expressed as a so-called "convex quadratic" optimization problem, meaning the objective is a quadratic function and the constraints form a convex set (are linear inequalities and equalities). There is a neat theorem that addresses such, and it’s the "convex quadratic" generalization of the Lagrangian method. The result is due to Karush, Kuhn, and Tucker conditions.

The "original" optimization problem is often called the "primal" problem. While a "primal problem" can be either a minimization or a maximization (and there is a corresponding KKT theorem for each) we'll use the minimization form here. Next we define a corresponding "dual" optimization problem, which is a maximization problem whose objective and constraints are related to the primal in a standard way.

Duality is an optimization problem that allows us to transform one problem into another equivalent problem. In SVM case, we have convex minimization problem. Its dual problem will be concave over a convex set and a maximization problem because minimizing a convex function and maximizing a concave function over a convex set are both convex problems. Therefore, minimizing a convex f is maximizing −f, which is concave. 

Slater's condition holds for the primal convex problem of SVM, therefore, the duality gap is 0, meaning that strong duality holds, so that the optimal solutions for the primal and dual problems have equal objective value.

The critical point of Lagrangian occurs at saddle points rather than local minima (or maxima), which is why the Karush–Kuhn–Tucker theorem is sometimes referred to as the saddle-point theorem. To utilize numerical optimization techniques, we must first transform the problem such that the critical points lie at local minima. This is done by calculating the magnitude of the gradient of Lagrangian. Next we turn to the conditions that must necessarily hold at the saddle point and thus the solution of the problem. These are called the KKT conditions (which stands for Karush-Kuhn-Tucker).

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/kkt_conditions.png?raw=true)

#### What is Lagrangian function?

The notation

$$
\begin{split}
\text{minimize } &f_{0}(x) \\
\text{subject to } &f_{i}(x) \leq 0, 1 \leq i \leq m \\
&h_{j} (x) = 0, 1 \leq j \leq k
\end{split}
$$

describes the problem of finding the $x$ that minimizes $f_{0}(x)$ among all $x$ that satisfy the constrains. It is called a convex optimization problem if $f_{0}, \cdots, f_{m}$ are convex functions and $h_{1}, \cdots, h_{k}$ are affine. For a constrained optimization problem like that, we define the Lagrangian:

$$
L(x, \lambda, \nu) = f_{0}(x) + \sum_{i=1}^{m} \lambda_{i}f_{i}(x) + \sum_{j=1}^{k} \nu_{j} h_{j}(x)
$$

The idea of the Lagrangian duality is to take the constrains into account by augmenting the objective function with a weighted sum of the constraint functions. So, we can now solve a constrained minimization problem using unconstrained optimization of generalized Lagrangian. 

#### What is Jensen's inequality?

Let f be a function whose domain is set of real numbers. f is a convex function if $f^{\prime \prime}(x) \geq 0$  (for all $x \in \mathbb{R}$). In the case of $f$ taking vector-valued inputs ($\mathbb{R}^{n}$), this is generalized to the condition that its Hessian $H$ is positive semi-definite ($H \geq 0$). If $f^{\prime \prime}(x) > 0$ for all $x$, then we say that $f$ is strictly convex (in the vector-valued case, the corresponding statement is that $H$ must be positive definite, i.e. $H > 0$). Jensen's inequality can be stated ass follows:

Let $f$ be a convex function and let $x$ be a random variable. Then,

$$
E[f(x)] \geq f[E(x)]
$$

Moreover if $f$ is strictly convex, then $E[f(x)] = f[E(x)]$ hold true if and only if $x = E(x)$ with probability 1, i.e., if $x$ is a constant.

Geometrically we can show the inequality above:

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/jensen.png?raw=true)

Similarly, $f$ is (strictly) concave if and only if $-f$ is (strictly) convex (i.e., $f^{\prime \prime}(x) \leq 0$ ($f^{\prime \prime}(x) < 0$) or $H \leq 0$ ($H < 0$)).

Jensen's inequality also holds for concave functions $f$ but with the direction of all inequalities reversed:

$$
E[f(x)] \leq f[E(x)]
$$


## Set Theory
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/set_theory.gif?raw=true)

#### What is a random experiment?
A random experiment is an experiment or a process for which the outcome cannot be predicted with certainty.

#### What is a sample space?
The sample space (mostly denoted by $S$) of a random experiment is the set of all possible outcomes. $S$ is called the certain event.

Let $A$ be a set. The notation $x \in A$ means that $x$ belongs to $A$. 

#### What is an empty set?
In mathematics, the empty set, mostly denoted by $\emptyset$, is a set with no elements; its size or cardinality (count of elements in a set) is zero. Empty set is called the impossible event. 

Let $A$ be a set. 
* The intersection of any set with the empty set is the empty set, i.e., $A \cap \emptyset = \emptyset$.
* The union of any set with the empty set is the set we started with, i.e., $A \cup \emptyset = A$.
* The complement of the empty set is the universal set (U) for the setting that we are working in, i.e., $\emptyset^C = U - \emptyset = U$. Also, the complement of $U$ is the empty set: $U^{c} =  U - U = \emptyset$.
* The empty set is a subset of any set.

#### What is an event?
An event (mostly denoted by $E$) is a subset of the sample space. We say that $E$ has occured if the observed outcome $x$ is an element of $E$, that is $x \in E$

**Examples**:
* Random experiment: toss a coin, sample sample $S = \{ \text{heads}, \text{tails}\}$
* Random experiment: roll a dice, sample sample $S = \{1,2,3,4,5,6\}$

#### What are the operations on a set?
When working with events, __intersection__ means "and", and __union__ means "or".

$$
P(A \cap B) = P(\text{A and B}) = P(A, B)
$$

$$
P(A \cup B) = P(\text{A or B})
$$

#### What is mutually exclusive (disjoint) events?
The events in the sequence $A_{1}, A_{2}, A_{3}, \ldots$ are said to be mutually exclusive events if $E_{i} \cap E_{j} = \emptyset\text{ for all }i \neq j$ where $\emptyset$ represents the empty set. 

In other words, the events are said to be mutually exclusive if they do not have any outcomes (elements) in common, i.e., they are pairwise disjoint. 

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/mutually_exclusive.png?raw=true)

For example, if events $A$ and $B$ are mutually exclusive:

$$
P(A \cup B) = P(A) + P(B)
$$

#### What is a non-disjoint event?
Disjoint events, by definition, can not happen at the same time. A synonym for this term is mutually exclusive. Non-disjoint events, on the other hand, can happen at the same time. For example, a student can get grade A in Statistics course and A in History course at the same time.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/nondisjoint_events.png?raw=true)

For example, if events $A$ and $B$ are non-disjoint events, the probability of A or B happening (union of these events) is given by:

$$
P(A \cup B) = P(A) + P(B) - P(A \cap B)
$$

This rule is also called Addition rule.

#### What is an independent event?

An independent event is an event that has no connection to another event's chances of happening (or not happening). In other words, the event has no effect on the probability of another event occurring.

Let's say that we have two events, $A$ and $B$ and they are independent. Intersection of these two events has:

$$
P(A\cap B) = P(A)P(B)
$$

This rule is also called multiplication rule.

Therefore, their union is:

$$
P(A\cup B) = P(A) + P(B) - P(A\cap B) =  P(A) + P(B) - P(A)P(B)
$$

#### What is exhaustive events?
When two or more events form the sample space ($S$) collectively, then it is known as collectively-exhaustive events. The $n$ events $A_{1}, A_{2}, A_{3}, \ldots, A_{n}$ are said to be exhaustive if $A_{1} \cup A_{2} \cup A_{3} \cup \ldots \cup A_{n} = S$. 

#### What is Inclusion-Exlusive Principle?
* For $n=2$ events:

$$
P(A \cup B) = P(A) + P(B) - P(A \cap B)
$$

Let's prove this principle:

$$
\begin{split}
P(A \cup B) &= P(A \cup (B-A))\\
&= P(A) + P(B-A)\\
&= P(A) + P(B) - P(A \cap B)
\end{split}
$$

* For events $A_{1}, A_{2}, A_{3}, \ldots, A_{n}$ in a probability space:

$$
\begin{split}
P\left(\cup_{i=1}^{n} A_{i}\right) =\sum_{i=1}^{n} P(A_{i}) &- \sum_{1 \leq i \leq j \leq n} P(A_{i} \cap A_{j})\\
&+ \sum_{1 \leq i \leq j \leq k \leq n} P(A_{i} \cap A_{j} \cap A_{k}) - \ldots \\
& + \left(-1\right)^{n-1} P\left(\cap_{i=1}^{n} A_{i}\right)
\end{split}
$$


## Probability and Statistics 

#### Explain about statistics branches?

The two main branches of statistics are descriptive statistics and inferential statistics.

* **Descriptive statistics**: It summarizes the data from a sample using indexes such as mean or standard deviation. Descriptive Statistics methods include displaying, organizing and describing the data.
* **Inferential Statistics**: It draws the conclusions from data that are subject to random variation such as observation errors and sample variation.

#### What is the permutation and combination?

**Permutation** : It is the different arrangements of a given number of elements taken one by one, or some, or all at a time. For example, if we have two elements A and B, then there are two possible arrangements, AB and BA.

Number of permutations when 'r' elements are arranged out of a total of 'n' elements is $^n P_r =\frac{n!}{(n-r)!}$. For example, let $n = 4$ (A, B, C and D) and $r = 2$ (All permutations of size 2). The answer is $4!/(4-2)! = 12$. The twelve permutations are AB, AC, AD, BA, BC, BD, CA, CB, CD, DA, DB and DC.

**Combination** : It is the different selections of a given number of elements taken one by one, or some, or all at a time. For example, if we have two elements A and B, then there is only one way select two items, we select both of them.

Number of combinations when 'r' elements are selected out of a total of 'n' elements is ${n \choose x} = ^n C_r =\frac{n!}{r! (n-r)!}$. For example, let $n = 4$ (A, B, C and D) and $r = 2$ (All combinations of size 2). The answer is $4!/((4-2)! 2!) = 6$. The six combinations are AB, AC, AD, BC, BD, CD.

Note that the order of the objects matters in permutation. 

Also note that ${n \choose r} = ^n C_r = \frac{^n P_r}{r!}$

##### Calculate the total number of combinations over n elements, where the number of elements in each subset is in {0,..,n}?

If order doesn't matter, you can use: ${n \choose 0} + {n \choose 1} + {n \choose 2} + \dots + {n \choose n} = 2^n$.

The intuition here is that we're simply summing up all the possible combinations of different length-ed sets. We start with the combinations of length zero (there's only one - the empty set), and then add the combinations of length one (there's 𝑛 of them), etc. until we get to adding the combinations of length 𝑛 (there's only one).

If order does matter you can use: ${n \choose 0}0! + {n \choose 1}1! + {n \choose 2}2! + \dots + {n \choose n}n!$.

All we do here is multiply each element in the last equation by the number of different arrangements that are possible for each length. If, for example, we're considering a set of 7 elements, there are 7! different ways of rearranging that set. Note that this assumes that your set elements are unique (i.e. it's a proper set). If you have duplicate elements, don't use this formula (because, for example, it will count the ordered set AAB twice, given that we can switch the positions of the As).

Perhaps an easier way to think about the ordered case it we're just summing permutations of each possible length: $^n P_0 + ^n P_1 + ^n P_2 + \dots + ^n P_n = \frac{n!}{n!} +  \frac{n!}{(n-1)!} + \frac{n!}{(n-2)!} + \cdots + \frac{n!}{0!}$.

#### What is a probability?

We assign a probability measure $P(A)$ to an event $A$. This is a value between $0$ and $1$ that shows how likely the event is. If $P(A)$ is close to $0$, it is very unlikely that the event $A$ occurs. On the other hand, if $P(A)$ is close to $1$, $A$ is very likely to occur. 

#### What are the probability axioms?

* **Axiom 1** For any event $A$, $P(A) \geq 0$
* **Axiom 2** Probability of the sample space is $P(S)=1$ 
* **Axiom 3** If $A_{1}, A_{2}, A_{3}, \ldots$ are disjoint (mutually exclusive) (even countably infinite) events, meaning that they have an empty intersection, the probability of the union of the events is the same as the sum of the probabilities: $P(A_{1} \cup A_{2} \cup A_{3} \cup \ldots) = P(A_{1}) + P(A_{2}) + P(A_{3}) + \dots$.

#### What is a random variable?

A random variable is a variable whose values depend on all the possible outcomes of a natural phenomenon. In short, a random variable is a quantity produced by a random process. Each numerical outcome of a random variable can be assigned a probability. There are two types of random variables, discrete and continuous. 

A discrete random variable is one which may take on either a finite number of values, or an infinite, but countable number of values, such as 0,1,2,3,4,.... Discrete random variables are usually (but not necessarily) counts. If a random variable can take only a finite number of distinct values, then it must be discrete. Examples of discrete random variables include the number of children in a family, the Friday night attendance at a cinema, the number of patients in a doctor's surgery, the number of defective light bulbs in a box of ten.

A continuous random variable is one which takes on an uncountably infinite number of possible values. Continuous random variables are usually measurements. Examples include height, weight, the amount of sugar in an orange, the time required to run a mile.

#### Compare "Frequentist statistics" vs. "Bayesian statistics"

In a short way, statistics is the science of changing your mind. There are two schools of thoughts. The more popular one - Frequentist statistics - is all about checking whether you should leave your default action. Bayesian statistics is all about having a prior opinion and updating that opinion with data. If your mind is truly blank before you begin, look at your data and just go with it.

#### What is a probability distribution?

A random variable can take multiple values. One very important thing is to know if some values will be more often encountered than others. The description of the probability of each possible value that a random variable can take is called its probability distribution.

In probability theory and statistics, a probability distribution is a mathematical function that provides the probabilities of occurrence of different possible outcomes in an experiment. In more technical terms, the probability distribution is a description of a random phenomenon in terms of the probabilities of events. For instance, if the random variable $X$ is used to denote the outcome of a coin toss ("the experiment"), then the probability distribution of $X$ would take the value $0.5$ for $X = \text{heads}$, and $0.5$ for $X = \text{tails}$ (assuming the coin is fair). Examples of random phenomena can include the results of an experiment or survey.

Discrete probability functions are referred to as probability mass functions and continuous probability functions are referred to as probability density functions. 

#### What is a probability mass function? What are the conditions for a function to be a probability mass function?

Discrete probability function is referred to as probability mass function (pms). It is a function that gives the probability that a discrete random variable is exactly equal to some value. The mathematical definition of a discrete probability function, $p(x)$, is a function that satisfies the following properties:

1. The probability that $x$ can take a specific value is $p(x)$. That is

  $$P[X=x]=p(x)=p_{x}$$

2. $p(x)$ is non-negative for all real $x$.

4. The sum of $p(x)$ over all possible values of $x$ is $1$, that is

  $$
  \sum_{j}p_{j} = 1
  $$ 
  
  where $j$ represents all possible values that $x$ can have and $p_{j}$ is the probability at $x_j$.

  One consequence of properties 2 and 3 is that $0 \leq p(x) \leq 1$.

A discrete probability function is a function that can take a discrete number of values (not necessarily finite). This is most often the non-negative integers or some subset of the non-negative integers. There is no mathematical restriction that discrete probability functions only be defined at integers, but in practice this is usually what makes sense. For example, if you toss a coin 6 times, you can get 2 heads or 3 heads but not 2 1/2 heads. Each of the discrete values has a certain probability of occurrence that is between zero and one. That is, a discrete function that allows negative values or values greater than one is not a probability function. The condition that the probabilities sum to one means that at least one of the values has to occur.

#### What is a probability density function? What are the conditions for a function to be a probability density function?

Continuous probability function is referred to as probability density function (pdf). It is a function of a continuous random variable, whose integral across an interval gives the probability that the value of the variable lies within the same interval. Unlike discrete random variable, the probability for a given continuous variable can not be specified directly; instead, it is calculated as an integral (area under curve) for a tiny interval around the specific outcome.

The mathematical definition of a continuous probability function, $f(x)$, is a function that satisfies the following properties:

* The probability that $x$ is between two points $a$ and $b$ is

$$
p[a \leq x \leq b]=\int_{a}^{b} f(x)dx
$$

* It is non-negative for all real $x$.

* The integral of the probability function is one, that is

$$
\int_{\infty}^{\infty} f(x)dx = 1
$$

#### What is a joint probability distribution? What is a marginal probability? Given the joint probability function, how will you calculate it?

In general, if $X$ and $Y$ are two random variables, the probability distribution that defines their simultaneous behavior is called a joint probability distribution, shown as $P(X =x, Y = y)$. If $X$ and $Y$ are discrete, this distribution can be
described with a _joint probability mass function_. If $X$ and $Y$ are continuous, this distribution can be described with a _joint probability density function_. If we are given a joint probability distribution for $X$ and $Y$ , we can obtain the individual probability distributions for $X$ or for $Y$ (and these are called the _Marginal Probability Distributions_).

Note that when there are two random variables of interest, we also use the term _bivariate probability distribution_ or _bivariate distribution_ to refer to the joint distribution.

The joint probability mass function of the discrete random variables $X$ and $Y$, denoted as $P_{XY} (x, y)$, satisfies:

* $P_{XY} (x, y) \geq 0$ for all x, y
* $\sum_{x} \sum_{y} P_{XY} (x, y) = 1$
* $P_{XY} (x, y) = P(X = x, Y = y)$

If $X$ and $Y$ are discrete random variables with joint probability mass function $P_{XY} (x, y)$, then the marginal probability mass functions oP $X$ and $Y$ are,

$$
P_{X} (x) = P(X=x) = \sum_{y_{j} \in \mathbb{R}_{y}} P_{XY}(X=x,Y=y_{j}) = \sum_{y} P_{XY} (x, y)
$$

and

$$
P_{Y} (y) = P(Y=y) = \sum_{x_{i} \in \mathbb{R}_{x}} P_{XY}(X=x_{i},Y=y) = \sum_{x} P_{XY} (x, y)
$$

where the sum for $P_{X} (x)$ is over all points in the range of $(X, Y)$ for which $X = x$ and the sum for $P_{Y} (y)$ is over all points in the range of $(X, Y)$ for which $Y = y$.

A joint probability density function for the continuous random variable $X$ and $Y$, denoted as $f_{XY} (x, y)$, satisfies the following properties:

* $f_{XY} (x, y) \geq 0$ for all x, y
* $\int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f_{XY} (x, y) dx dy = 1$
* For any region \$\mathbb{R}$ of 2-D space:

$$
P((X, Y) \in \mathbb{R}) = \int_{\mathbb{R}} f_{XY} (x, y) dx dy
$$

If $X$ and $Y$ are continuous random variables with joint probability density function $f_{XY} (x, y)$, then the marginal density functions for $X$ and $Y$ are:

$$
f_{X} (x) = \int_{y} f_{XY} (x, y) dy
$$

and

$$
f_{Y} (y) = \int_{x} f_{XY} (x, y) dx
$$

where the first integral is over all points in the range of $(X, Y)$ for which $X = x$, and the second integral is over all points in the range of $(X, Y)$ for which $Y = y$.

The joint cumulative distribution function (CDF) of two random variables $X$ and $Y$ is defined as:

$$
\begin{split}
F_{XY}(x,y)&=P(X \leq x, Y \leq y) \\
&= P\big((X \leq x)\text{ and }(Y\leq y)\big)=P\big((X \leq x)\cap(Y\leq y)\big).
\end{split}
$$

The joint CDF satisfies the following properties:

* $F_X(x)=F_{XY}(x, \infty)$, for any $x$ (marginal CDF of $X$)
* $F_Y(y)=F_{XY}(\infty,y)$, for any $y$ (marginal CDF of $Y$)
* $F_{XY}(\infty, \infty)=1$
* $F_{XY}(-\infty, y)=F_{XY}(x,-\infty)=0$
* $P(x_1<X \leq x_2, \hspace{5pt} y_1<Y \leq y_2)= F_{XY}(x_2,y_2)-F_{XY}(x_1,y_2)-F_{XY}(x_2,y_1)+F_{XY}(x_1,y_1)$
* If X and Y are independent, then $F_{XY}(x,y)=F_X(x)F_Y(y)$. 

#### What is conditional probability? Given the joint probability function, how will you calculate it?

Let's say we have two events, $A$ and $B$. The conditional probability of an event $B$ is the probability that the event will occur given the knowledge that an event $A$ has already occurred. This probability is written $P(B \mid A)$, notation for the probability of $B$ given $A$.  In the case where events $A$ and $B$ are independent (where event $A$ has no effect on the probability of event $B$), the conditional probability of event $B$ given event $A$ is simply the probability of event $B$, that is $P(B)$.

$$
P(B \mid A) = P(B)
$$

Because $P(A \cap B) = P(A) \times P(B)$ when $A$ and $B$ are independent events.

However, If events $A$ and $B$ are not independent, then the probability of the intersection of $A$ and $B$ (the probability that both events occur) is defined by $P(A\text{ and }B) = P(A \cap B) = P(A)P(B \mid A)$, which $P(A\text{ and }B)$ is the joint probability. Intuitively it states that the probability of observing events $A$ and $B$ is the probability of observing $A$, multiplied by the probability of observing $B$, given that you have observed $A$.

From this definition, the conditional probability $P(B \mid A)$ is easily obtained by dividing by $P(A)$:

$$
P(B \mid A) = \dfrac{P(A \cap B)}{P(A)} 
$$

Note that this expression is only valid when $P(A)$ is greater than 0. If this is not the case, we cannot compute the conditional probability conditioned on an event that never happened. 

Technically speaking, when you condition on an event happening, you are entering the universe where that event has taken place. Mathematically, if you condition on $A$, then $A$ becomes your new sample space. In the universe where $A$ has taken place, all axioms of probability still hold! In particular,

* __Axiom 1:__ For any event $B$, $P(B \mid A) \geq 0$.
* __Axiom 2:__ Conditional probability of $A$ given $A$ is 1, i.e., $P(A \mid A)=1$.
* __Axiom 3:__ If $B_{1}, B_{2}, B_{3}, \ldots $ are disjoint events, then $P(B_{1} \cup B_{2} \cup B_{3} \cup \ldots \mid A) = P(B_{1} \mid A) + P(B_{2} \mid A) + P(B_{3} \mid A) + \dots $

#### State the Chain rule of conditional probabilities.

To calculate the probability of the intersection of more than two events, the conditional probabilities of all of the preceding events must be considered. In the case of three events, $A$, $B$, and $C$, the probability of the intersection $P(A\text{ and }B\text{ and }C) = P(A)P(B \mid A)P(C \mid A\text{ and }B)$, which we call the _Chain Rule_. Here is the general form of the Chain Rule when $n$ events are given:

$$
P(A_{1} \cap A_{2} \cap \ldots \cap A_{n}) = P(A_{1})P(A_{2} \mid A_{1})P(A_{3} \mid A_{2}, A_{1}) \ldots P(A_{n} \mid A_{n-1} A_{n-2} \cdots A_{1})
$$

#### What are the conditions for independence and conditional independence of two random variables?

Two random variables x and y are independent if their probability distribution can be expressed as a product of two factors, one involving only x and one involving only y:

$$
P(X=x,Y=y)=P(X=x)P(Y=y), \text{ for all } x,y.
$$

Intuitively, two random variables $X$ and $Y$ are independent if knowing the value of one of them does not change the probabilities for the other one. In other words, if $X$ and $Y$ are independent, we can write

$$
P(Y=y \mid X=x) = P(Y=y), \text{ for all } x, y.
$$

We can extend the definition of independence to $n$ random variables.

They are conditionally independent if they are unrelated after taking account of a 3rd variable. 

$$
P(X  = x, Y = y \mid Z = z) = P(X = x | Z = z ) P(Y = y | Z = z)
$$

We can denote the independence and conditional independence with compact notation: $x \perp y$ means that x and y are independent, while $x \perp y \mid z$ means that x and y conditionally independent given z.

#### What are expectation, variance and covariance?

In probability, the average, or mean value of some random variable X is called the **expected value** or the expectation, denoted by $E(x)$.

Suppose $X$ is a discrete random variable that takes values $x_{1}, x_{2}, . . . , x_{n}$ with probabilities $p(x_{1}), p(x_{2}), . . . , p(x_{n})$. The expected value of $X$ is defined by:

$$
E(X) = \sum_{j=1}^{n}  x_{j}p(x_{j}) = x_{1}p(x_{1}) + x_{2}p(x_{2})  + . . . + x_{n}p(x_{n}).
$$

Let $X$ be a continuous random variable with range $[a, b]$ and probability density function $f(x)$. The expected value of $X$ is defined by

$$
E(X) = \int_{a}^{b} xf(x) dx.
$$

Expectations are linear, for example,

$$
E\left(\alpha f(x) + \beta g(x) \right) = \alpha E[f(x)] + \beta E[g(x)]
$$

In probability, the **variance** of some random variable $X$, denoted by $Var(X)$ is a measure of how much values in the distribution vary on average (i.e., with respect to the mean) as we sample different values of $X$ from its probability distribution. Variance is calculated as the average squared difference of each value in the distribution from the expected value. Or the expected squared difference from the expected value.

$$
\begin{split}
Var(X) &= E\left[\left(X - E[X]\right)^{2}\right] \\
&= E\left[X^{2}-2XE(X) + \left(E(X) \right)^{2}\right] \\
&= E(X^{2}) - 2E(X)E(X) + \left(E(X) \right)^{2}\\
&= E(X^{2}) - \left(E(X) \right)^{2}
\end{split}
$$

In probability, **covariance** is the measure of the joint probability for two random variables. It describes how the two variables change (vary) together. It’s similar to variance, but where variance tells you how a single variable varies, covariance tells you how two variables vary together. It is denoted as the function $cov(X, Y)$, where $X$ and $Y$ are the two random variables being considered.

$$
cov(X, Y) = E\left[\left(X - E[X]\right) \left(Y - E[Y]\right)\right] =  E(XY)- E(X)E(Y)
$$

Note that $Var(X) = cov(X, X) = E\left[(X - E[X])^{2}\right]$. 

High absolute values of the covariance mean that the values change very much and are both far from their respective means at the same time. The sign of the covariance can be interpreted as whether the two variables increase together (positive) or decrease together (negative). The magnitude of the covariance is not easily interpreted. A covariance value of zero indicates that both variables are completely independent.

The covariance can be normalized to a score between $-1$ and $1$ to make the magnitude interpretable by dividing it by the standard deviation of X and Y. The result is called the correlation of the variables, also called the _Pearson correlation coefficient_, named for the developer of the method, Karl Pearson.

$$
corr(X, Y) = \rho_{X, Y}= \frac{cov(X,Y)}{\sigma_{X}\sigma_{Y}}
$$

where $\sigma_{X} = \sqrt{Var(X)}$ and $\sigma_{Y} = \sqrt{Var(Y)}$.

As one can tell easily that correlation is just the covariance normalized.
 
The covariance is especially useful when looking at the variance of the sum of two random 'correlated' variates, since

$$
Var(X+Y) = Var(X)+ Var(Y) + 2cov(X,Y)
$$

If the variables are uncorrelated (that is, $cov(X, Y)=0$ -- However, just because they have zero covariance does not mean that they are independent. See the note below), then

$$
Var(X+Y) = Var(X) + Var(Y).
$$

This comes from the fact that 

$$
cov(X, Y) =  E(XY)- E(X)E(Y) = E(X)E(Y) - E(X)E(Y) = 0
$$

We generalize those results,

$$
Var\left( \sum_{i=1}^{n} X_i \right)= \sum_{i,j=1}^{N}  cov(X_{i},X_{j}) = \sum_{i=1}^{n} Var( X_i) + \sum_{i \neq j} cov(X_{i},X_{j}).
$$

If for each $i \neq j$, $X_i$ and $X_j$ are uncorrelated, in particular if the $X_i$ are pairwise independent (that is, $X_i$ and $X_j$ are independent whenever $i \neq j$), then,

$$
Var\left( \sum_{i=1}^{n} X_i \right)=  \sum_{i=1}^{n} Var( X_i) .
$$

The covariance is symmetric by definition since

$$
cov(X,Y)=cov(Y,X). 
$$

**NOTE**: The notations of covariance and dependence are related but distinct concepts. They are related because two variables that are independent have zero covariance and two variable that have non-zero covariance are dependent. Independent, however, is a distinct property from covariance. For two variables to have zero covariance, there must be no linear dependence between them. Independence is a stronger requirement than zero covariance because independence also excludes nonlinear relationships. It is possible for two variable to be dependent but have zero covariance. For example, suppose we first sample a real number $x$ from a uniform distribution over the interval $[-1, 1]$. We next sample a random variable $s$. With probability $1/2$, we choose the value of $s$ to be 1. Otherwise, we choose the value of $s$ to be $-1$. We can then generate a random variable $y$ by assigning $y=sx$. Clearly, x and y are not independent, because $x$ completely determines the magnitude of $y$. However, $Cov(x, y) = 0$.

#### What is the covariance for a vector of random variables?

A random vector is a vector of random variables:

$$
\mathbf{X} = \begin{bmatrix} X_{1} \\ X_{2}\\ \vdots \\ X_{n} \end{bmatrix}
$$

If $X$ is a random vector, the covariance matrix of $X$, denoted by $\Sigma$, is then given by:

$$
\begin{split}
\Sigma = cov(\mathbf{X}) &= E\left[ \left( \mathbf{X} - E(\mathbf{X}) \right) \left( \mathbf{X} - E(\mathbf{X}) \right)^{T} \right]\\
&= E\left[\mathbf{X}\mathbf{X}^{T} \right] - E[\mathbf{X}]\left(E[\mathbf{X}]\right)^{T}
\end{split}
$$

and defined as

$$
\Sigma = cov(\mathbf{X}) = 
\begin{bmatrix} Var(X_{1}) & cov(X_{1},X_{2}) & \ldots & cov(X_{1},X_{n}) \\
cov(X_{2}, X_{1}) & Var(X_{2}) & \ldots & cov(X_{2},X_{n}) \\
\vdots & \vdots & \ddots & \vdots \\
cov(X_{n}, X_{1}) & cov(X_{n}, X_{2}) & \ldots & Var(X_{n}) \\
\end{bmatrix}
$$

As we previously mentioned, covariance matrix is symmetric, meaning that $\Sigma_{i,j} = \Sigma_{j,i}$.

$$
\Sigma_{i,j} = cov(X_{i}, X_{j}) = E\left[\left(X_{i} - E(X_{i}) \right)\left(X_{j} - E(X_{j}) \right)\right] = E\left[\left(X_{i} - \mu_{i} \right)\left(X_{j} - \mu_{j} \right)\right]
$$

Note that If $X_{1}, X_{2}, \ldots , X_{n}$ are independent, then the covariances are $0$ and the covariance matrix is equal to $diag \left(Var(X_{1}), Var(X_{2}), \ldots , Var(X_{n})\right)$ if the $X_{i}$ have common variance $\sigma^{2}$.

__Properties:__

* **Addition to the constant vectors**: Let a be a constant $n \times 1$ vector and let $X$ be a $n \times 1$ random vector. Then, $cov(a + \mathbf{X}) = cov(\mathbf{X})$.

* **Multiplication by constant matrices**: Let $b$ be a constant $m \times n$ matrix and let $X$ be a $n \times 1$ random vector. Then, $cov(b \mathbf{X}) = b cov(\mathbf{X}) b^{T}$.

* **Linear transformations** Let a be a constant $n \times 1$ vector, and $b$ be a constant $m \times n$ matrix and $X$ be a $n \times 1$ random vector. Then, combining the two properties above, one obtains $cov(a + b \mathbf{X})= b cov(\mathbf{X}) b^{T}$.

* **Symmetry**: The covariance matrix $cov(\mathbf{X})$ is a symmetric matrix, that is, it is equal to its transpose: $cov(\mathbf{X}) = cov(\mathbf{X})^{T}$.

* **Positive semi-definiteness**: The covariance matrix $cov(\mathbf{X})$ is positive semi-definite, that is, for a constant $n \times 1$ vector, 
  
  
  $$
a^{T} cov(\mathbf{X}) a \geq 0
$$
  
  This is easily proved,
  
  $$
  \begin{split}
  a^{T} cov(\mathbf{X}) a &= a^{T} E\left[ \left( \mathbf{X} - E(\mathbf{X}) \right) \left( \mathbf{X} - E(\mathbf{X}) \right)^{T} \right] a \\
  &=E\left[a^{T} \left( \mathbf{X} - E(\mathbf{X}) \right) \left( \mathbf{X} - E(\mathbf{X}) \right)^{T} a\right]\\
  &=E\left[\left(\left( \mathbf{X} - E(\mathbf{X}) \right)^{T}a\right)^{T} \left(\left( \mathbf{X} - E(\mathbf{X}) \right)^{T}a\right)\right]\\
  &= E\left[\left(\left( \mathbf{X} - E(\mathbf{X}) \right)^{T}a\right)^{2}\right] \geq 0
  \end{split}
  $$

#### What is cross-covariance?

Let $\mathbf{X}$ be a $K \times 1$ random vector and $\mathbf{Y}$ be a $L	\times 1$ random vector. The covariance matrix between $\mathbf{X}$ and $ \mathbf{Y}$, or cross-covariance between $\mathbf{X}$ and $\mathbf{Y}$ is denoted by $cov(\mathbf{X}, \mathbf{Y})$. It is defined as follows:

$$
cov(\mathbf{X}, \mathbf{Y}) = E \left[\left(\mathbf{X}-E[\mathbf{X}]\right)\left(\mathbf{Y}-E[\mathbf{Y}]\right)^{T}\right]
$$

provided the above expected values exist and are well-defined.

It is a multivariate generalization of the definition of covariance between two scalar random variables.

Let $X_{1}, \ldots, X_{K}$ denote the $K$ components of the vector $\mathbf{X}$ and $Y_{1}, \ldots, Y_{L}$ denote the $L$ components of the vector $\mathbf{Y}$ .

$$
\begin{split}
cov(\mathbf{X}, \mathbf{Y}) &= 
\begin{bmatrix} E \left[\left(X_{1}-E[X_{1}]\right)\left(Y_{1}-E[Y_{1}]\right)\right] & \ldots & E \left[\left(X_{1}-E[X_{1}]\right)\left(Y_{L}-E[Y_{L}]\right)\right] \\
E \left[\left(X_{2}-E[X_{2}]\right)\left(Y_{1}-E[Y_{1}]\right)\right] & \ldots & E \left[\left(X_{2}-E[X_{2}]\right)\left(Y_{L}-E[Y_{L}]\right)\right] \\
\vdots & \ddots & \vdots \\
E \left[\left(X_{K}-E[X_{K}]\right)\left(Y_{1}-E[Y_{1}]\right)\right] & \ldots & E \left[\left(X_{K}-E[X_{K}]\right)\left(Y_{L}-E[Y_{L}]\right)\right] \\
\end{bmatrix} \\
&=
\begin{bmatrix} cov (X_{1}, Y_{1}) & \cdots & cov (X_{1}, Y_{L})\\
cov (X_{2}, Y_{1}) & \cdots & cov (X_{2}, Y_{L})\\
\vdots & \ddots & \vdots \\
cov (X_{K}, Y_{1}) & \cdots & cov (X_{K}, Y_{L})\\
\end{bmatrix} 
\end{split}
$$

Note that $cov(\mathbf{X}, \mathbf{Y})$ is not the same as $cov(\mathbf{Y}, \mathbf{X})$. In fact, $cov(\mathbf{Y}, \mathbf{X})$ is a $L \times K$ matrix equal to the transpose of $cov(\mathbf{X}, \mathbf{Y})$:

$$
\begin{split}
cov(\mathbf{Y}, \mathbf{X}) &= E \left[\left(\mathbf{Y}-E[\mathbf{Y}]\right)\left(\mathbf{X}-E[\mathbf{X}]\right)^{T}\right]\\
& = E \left[\left(\mathbf{Y}-E[\mathbf{Y}]\right)\left(\mathbf{X}-E[\mathbf{X}]\right)^{T}\right]^{T}\\
&= E \left[\left(\mathbf{X}-E[\mathbf{X}]\right)\left(\mathbf{Y}-E[\mathbf{Y}]\right)^{T}\right] \\
&= cov(\mathbf{X}, \mathbf{Y})
\end{split}
$$

by using the fact that $\left(A B \right)^{T} = B^{T} A^{T}$.

#### What is the correlation for a vector of random variables? How is it related to covariance matrix?

The correlation matrix of $\mathbf{X}$ is defined as

$$
corr(\mathbf{X}) = corr(X_{i}, X_{j}) = 
\begin{bmatrix} 1 & corr(X_{1},X_{2}) & \ldots & corr(X_{1},X_{n}) \\
corr(X_{2}, X_{1}) & 1 & \ldots & corr(X_{2},X_{n}) \\
\vdots & \vdots & \ddots & \vdots \\
corr(X_{n}, X_{1}) & corr(X_{n}, X_{2}) & \ldots & 1 \\
\end{bmatrix}
$$

Denote $cov(\mathbf{X})$ by $\Sigma = (\sigma_{ij})$. Then the correlation matrix and covariance matrix are related by

$$
cov(\mathbf{X}) = diag\left(\sqrt{\sigma_{11}},\sqrt{\sigma_{22}}, \ldots,\sqrt{\sigma_{nn}}\right) \times corr(\mathbf{X}) \times diag\left(\sqrt{\sigma_{11}},\sqrt{\sigma_{22}}, \ldots,\sqrt{\sigma_{nn}}\right)
$$

This is easily seen using $corr(X_{i}, X_{j}) = \dfrac{cov(X_{i}, X_{j})}{\sigma_{ii}\sigma_{jj}}$

Do not forget that covariance indicates the direction of the linear relationship between variables. Correlation on the other hand measures both the strength and direction of the linear relationship between two variables. Correlation is a function of the covariance. 

Note that covariance and correlation are the same if the features are standardized, i.e., they have mean 0 and variance 1.

#### Explain some Discrete and Continuous Distributions.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/discrete_distributions_table.png?raw=true)

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/continuous_distributions_table.png?raw=true)


#### What is a moment?

In mathematics, a moment is a specific quantitative measure of the shape of a function. The concept is used in both mechanics and statistics a lot. There are two different moments, (1) Uncentered (moment about the origin, also called as raw moments) and (2) Centered (moment about the mean). If the function is a probability distribution, then, $E(X^{k})$ is the $k$th (theoretical) moment of the distribution (about the origin), for $k = 1, 2, \dots $ and $E\left[(X-\mu)^{k}\right]$ is the $k$th (theoretical) moment of the distribution (about the mean), also called central moment, for $k = 1, 2, \dots $. For example, the zeroth moment is the total probability (i.e. one), the first moment is the expected value, the second central moment is the variance, the third standardized moment is the skewness, and the fourth standardized moment is the kurtosis.

* Expected Value: $E(X) = \mu$
* Variance: $Var\left[(X - \mu)^{2} \right] = \sigma^{2}$
* Skewness: $E\left[(X - \mu)^{3} \right]/\sigma^{3}$
* Kurtosis: $E\left[(X - \mu)^{4} \right]/\sigma^{4}$

#### What is moment generating function?

Moment generating function is literally the function that generates the moments. The moment generating function (MGF) of a random variable $X$ is a function $M_{X}(s)$ defined as:

$$
M_{X}(t)=E\left[e^{tX}\right]
$$

We say that MGF of $X$ exists, if there exists a positive constant a such that $M_{X}(s)$ is finite for all $s \in [−a,a]$.

For continuous distributions, it is given by:

$$
M_X(t) = E(e^{tX}) = \sum\limits_{\text{all }x} e^{tx} P(x)
$$

and for discrete distributions:

$$
M_X(t) = E(e^{tX}) = \int_x e^{tx} f(x) \, \mathrm{d}x
$$

Why is the MGF useful? There are basically two reasons for this. First, the MGF of X gives us all moments of X. That is why it is called the moment generating function. Second, the MGF (if it exists) uniquely determines the distribution. That is, if two random variables have the same MGF, then they must have the same distribution. That comes from the fact that moment generating functions are unique. Thus, if you find the MGF of a random variable, you have indeed determined its distribution. This method is very useful when we work on sums of several independent random variables. For example, suppose $X_{1}, X_{2}, \dots , X_{n}$ are $n$ independent random variables, and the random variable $Y$ is defined as:

$$
Y=X_1 + X_2 + \cdots + X_n
$$

Then,

$$
\begin{split}
M_Y(t)&=E[e^{tY}] \\
&=E[e^{t(X_1+X_2+ \cdots +X_n)}]\\
&=E[e^{tX_{1}} e^{tX_{2}} \cdots e^{tX_{n}}] \\
&=E[e^{tX_1}] E[e^{tX_2}] \cdots E[e^{tX_n}]  \hspace{10pt} \textrm{(since the $X_i$'s are independent)}\\
&=M_{X_1}(t) M_{X_2}(t) \cdots M_{X_n}(t)
\end{split}
$$

If $X_{1}, X_{2}, \dots , X_{n}$ are $n$ *independent* random variables, then, we can write:

$$
M_{X_1+X_2+\cdots +X_n}(t)=M_{X_1}(s)M_{X_2}(t) \cdots M_{X_n}(t)
$$

Note that if you take a derivative of MGF $k$ times with respect to $t$ and plug $t = 0$ in. Then, you will get the $k$th moment of the random variable $X$ about zero, i.e., $E(X^{k})$:

$$
E(X^{k}) = \frac{d^{k}}{d t^{k}} M_{X}(t) \Big|_{t = 0}
$$

#### What is characteristic function?

A characteristic function completely defines a probability distribution. Completely defining a probability distribution involves defining a complex function (a mix of real numbers and imaginary numbers). It is very similar to moment generating function. **The characteristic function has one big advantage, that it always exists — even when there is no moment generating function**.

There are random variables for which the moment generating function does not exist on any real interval with positive length. For example, consider the random variable $X$ that has a *Cauchy distribution*:

$$
f_X(x)=\frac{\frac{1}{\pi}}{1+x^2}, \hspace{10pt} \textrm{for all }x \in \mathbb{R}
$$

You can show that for any nonzero real number

$$
M_{X}(t)=\int_{-\infty}^{\infty} e^{tx}\frac{\frac{1}{\pi}}{1+x^2} dx=\infty
$$

Therefore, the moment generating function does not exist for this random variable on any real interval with positive length. If a random variable does not have a well-defined MGF, we can use the characteristic function defined as:

$$
\phi_{X}(\omega) = E[e^{j \omega X}]
$$

where $j = \sqrt{-1}$ and $\omega$ is a real number. It is worth noting that $e^{j \omega X}$ is a complex-valued random variable. You can imagine that a complex random variable can be written as $X = Y + j Z$, where $Y$ and $Z$ are ordinary real-valued random variables. Thus, working with a complex random variable is like working with two real-valued random variables. The advantage of the characteristic function is that it is defined for all real-valued random variables. Specifically, if $X$ is a real-valued random variable, we can write

$$
|e^{j \omega X}|=1
$$

Therefore, we conclude

$$
\begin{split}
|\phi_{X}(\omega)|&=|E[e^{j \omega X}]| \\
&\leq E[|e^{j \omega X}|]\\
&\leq 1
\end{split}
$$

The characteristic function has similar properties to the moment generating function. For example, if $X$ and $Y$ are independent

$$
\begin{split}
\phi_{X+Y}(\omega)&=E[e^{j \omega (X+Y)}]\\
&=E[e^{j \omega X} e^{j \omega Y}]\\
&=E[e^{j \omega X}]E[e^{j \omega Y}] \hspace{10pt} \textrm{(since $X$ and $Y$ are independent)}\\
&=\phi_{X}(\omega) \phi_{Y}(\omega)
\end{split}
$$

More generally, if $X_{1}, X_{2}, \dots, X_{n}$ are $n$ independent random variables, then

$$
\phi_{X_1+X_2+\cdots +X_n}(\omega)=\phi_{X_1}(\omega) \phi_{X_2}(\omega) \cdots \phi_{X_n}(\omega)
$$

For example, let's find the characteristic function of an exponential distributed random variable, i.e., $X \sim Exponential (\lambda)$. Recall that the PDF of $X$ is:

$$
f_X(x)=\lambda e^{-\lambda x},\,\,\,\, x \geq 0, \lambda > 0
$$

We conclude that:

$$
\begin{split}
\phi_{X}(\omega)&=E[e^{j \omega X}]  \\
&=\int_{0}^{\infty}\lambda e^{-\lambda x} e^{j \omega x}dx\\
&=\left[\frac{\lambda}{j \omega-\lambda} e^{(j \omega-\lambda) x}\right]_{0}^{\infty}\\
&=\frac{\lambda}{\lambda-j \omega}.
\end{split}
$$

Note that since $\lambda > 0$, the value of $e^{(j \omega-\lambda) x}$, when evaluated at $x = + \infty$, is zero.

* Let $X$ and $Y$ be two random variables. Denote by $F_{X}(x)$ and $F_{Y}(y)$ their distribution functions and by $\phi_{X}(\omega)$ and $\phi_{Y}(\omega)$ their characteristic functions. Then, $X$ and $Y$ have the same distribution, i.e., $F_{X}(x) = F_{Y}(y)$ for any x, if and only if they have the same characteristic function, i.e., $\phi_{X}(\omega) = \phi_{Y}(\omega)$ for any $\omega$. In applications, this proposition is often used to prove that two distributions are equal, especially when it is too difficult to directly prove the equality of the two distribution functions. 

* Like the moment generating function of a random variable, the characteristic function can be used to derive the moments of $X$ by differentiation of characteristic function.

* Many widely known distributions have known characteristic functions but they do not have known expressions for their respective density functions (e.g., Levy stable distributions).

#### What are the properties of Distributions?

* **Measures of Central Tendancy**

  - The mean is measured by taking the sum divided by the number of observations.
  - The median is the middle observation in a series of numbers. If the number of observations are even, then the two middle observations would be divided by two.
  - The mode refers to the most frequent observation.
  - The main question of interest is whether the sample mean, median, or mode provides the most accurate estimate of central tendancy within the population.

* **Measures of Dispersion**

  - The standard deviation of a set of observations is the square root of the average of the squared deviations from the mean. The squared deviations from the mean is called the variance.

* **The Shape of Distributions**

  - Unimodal distributions have only one peak while multimodal distributions have several peaks.
  - An observation that is skewed to the right contains a few large values which results in a long tail towards the right hand side of the chart.
  - An observation that is skewed to the left contains a few small values which results in a long tail towards the left hand side of the chart.
  - The kurtosis of a distribution refers to the degree of peakedness of a distribution.
  
  ![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/pearson-mode-skewness.jpg?raw=true)

#### What are the measures of Central Tendency: Mean, Median, and Mode?

The central tendency of a distribution represents one characteristic of a distribution. In statistics, the three most common measures of central tendency are the mean, median, and mode. Each of these measures calculates the location of the central point using a different method. The median and mean can only have one value for a given data set. The mode can have more than one value.

Choosing the best measure of central tendency depends on the type of data you have.

* **Mean**: The mean is the arithmetic average. Calculating the mean is very simple. You just add up all of the values and divide by the number of observations in your dataset.

  $$
\bar{x} = \frac{x_{1}+x_{2}+\cdots +x_{n}}{n}
$$

  The calculation of the mean incorporates all values in the data. If you change any value, the mean changes. However, the mean doesn't always locate the center of the data accurately. In a symmetric distribution, the mean locates the center accurately. However, in a skewed distribution, the mean can miss the mark. Outliers have a substantial impact on the mean. Extreme values in an extended tail pull the mean away from the center. As the distribution becomes more skewed, the mean is drawn further away from the center. Consequently, it’s best to use the mean as a measure of the central tendency when you have a symmetric distribution.

* **Median**: The median is the middle value. It is the value that splits the dataset in half. To find the median, order your data from smallest to largest, and then find the data point that has an equal amount of values above it and below it. The method for locating the median varies slightly depending on whether your dataset has an even or odd number of values.

  If the number of observations is odd, the number in the middle of the list is the median. This can be found by taking the value of the $(n+1)/2$-th term, where n is the number of observations. Else, if the number of observations is even, then the median is the simple average of the middle two numbers. 

  In a symmetric distribution, the mean and median both find the center accurately. They are approximately equal.

  Outliers and skewed data have a smaller effect on the median. Unlike the mean, the median value doesn’t depend on all the values in the dataset. When you have a skewed distribution, the median is a better measure of central tendency than the mean.

  You can also use median when you have ordinal data.
  
* **Mode**: The mode is the value that occurs the most frequently in your data set. Typically, you use the mode with nominal (categorical), ordinal, and discrete (count) data. In fact, the mode is the only measure of central tendency that you can use with norminal (categorical) data. However, with nominal (categorical) data, there is not a central value because you can not order the groups. With ordinal and discrete (count) data, the mode can be a value that is not in the center. Again, the mode represents the most common value.

  In the continuous data, no values repeat, which means there is no mode. With continuous data, it is unlikely that two or more values will be exactly equal because there are an infinite number of values between any two values. When you are working with the raw continuous data, don’t be surprised if there is no mode. However, you can find the mode for continuous data by locating the maximum value on a probability distribution plot. If you can identify a probability distribution that fits your data, find the peak value and use it as the mode.

* When you have a symmetrical distribution for continuous data, the mean, median, and mode are equal.
* When to use the mean: Symmetric distribution, Continuous data
* When to use the median: Skewed distribution, Continuous data, Ordinal data
* When to use the mode: Categorical data, Ordinal data, Count data, Probability Distributions

#### How to compute the median of a probability distribution?

A median by definition is a real number $m$ that satisfies:

$$
P(X\leq m)=\frac{1}{2}
$$

For example, let's find the median of exponential distribution whose distribution function is given below:

$$
f(x; \lambda) = \lambda e^{- \lambda x}, x \geq 0 
$$

So, we need to compute:

$$
\begin{split}
P(X \leq m) = \int_{0}^{m} \lambda e^{-\lambda x} dx &= \frac{1}{2}\\
1-e^{-\lambda m} &= \frac{1}{2}\\
m &= \frac{\ln 2}{\lambda}
\end{split}
$$

#### How to find the distribution of Order Statistics?

Let $X_{1}, X_{2}, \ldots, X_{n}$ be a random sample of size $n$ for some distribution. We denote the order statistics by:

$$
\begin{split}
X_{(1)} &\Rightarrow min(X_{1}, X_{2}, \ldots, X_{n})\\
X_{(2)} &\Rightarrow \text{ the 2nd smallest of } X_{1}, X_{2}, \ldots, X_{n}\\
& \ldots \\
X_{(n)} &\Rightarrow max(X_{1}, X_{2}, \ldots, X_{n})
\end{split}
$$

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Screen%20Shot%202019-10-07%20at%2021.25.52.png?raw=true)

* **Distribution of minimum**

  $$
\begin{split}
F_{X_{(1)}} (x) &= P(X_{(1)} \leq x) \\
&= 1 - P(X_{(1)} > x)\\
& = 1 - P(X_{1} > x, X_{2} > x, \ldots , X_{n} > x)\\
&= 1- \left[P(X_{1} > x)P(X_{2} > x)\ldots P(X_{n} > x) \right]\,\,\, \text{(By independence)}\\
&= 1- \prod_{i=1}^{n} P(X_{i} > x)\\
&= 1- \prod_{i=1}^{n} 1 - P(X_{i} \leq x)\\
&= 1 - \prod_{i=1}^{n} 1 - F_{X}(x)\,\,\, \text{(because $X_{i}$'s are identically distributed)}\\
&= 1 - \left[1 - F_{X}(x) \right]^{n}
\end{split}
$$

$$
\begin{split}
f_{X_{(1)}} (x)  = \frac{d}{dx} F_{X_{(1)}} (x) &=  \frac{d}{dx} \left\{1 - \left[1 - F_{X}(x) \right]^{n} \right\}\\
&= n (1-F_{X}(x))^{n-1}f_{X}(x)
\end{split}
$$

  So, for example, if $X \sim U(0,1)$, $f(x) = I_{(0,1)}(x)$. The pdf of minimum in this case is:

$$
f_{X_{(1)}} (x) = n(1-x)^{n-1}I_{(0,1)}(x)
$$

  This is the pdf of Beta distribution with the parameters 1 and $n$ denoted by $X_{(1)} \sim Beta(1,n)$.

* **Distribution of maximum**

  Similarly,

$$
\begin{split}
F_{X_{(n)}} (x) &= P(X_{(n)} \leq x) \\
& = P(X_{1} \leq x, X_{2} \leq x, \ldots , X_{n} \leq x)\\
&= P(X_{1} \leq x)P(X_{2} \leq x)\ldots P(X_{n} \leq x)\,\,\, \text{(By independence)}\\
&= \prod_{i=1}^{n} P(X_{i} \leq x)\\
&= \prod_{i=1}^{n} F_{X}(x)\,\,\, \text{(because $X_{i}$'s are identically distributed)}\\
&= \left[F_{X}(x) \right]^{n}
\end{split}
$$

$$
\begin{split}
f_{X_{(n)}} (x)  = \frac{d}{dx} F_{X_{(n)}} (x) &=  \frac{d}{dx} \left\{\left[F_{X}(x) \right]^{n} \right\}\\
&= n (F_{X}(x))^{n-1}f_{X}(x)
\end{split}
$$

  In case of a random sample of size $n$ from Uniform distribution on the interval $[0,1]$:

$$
f_{X_{(n)}} (x) = n(x)^{n-1}I_{(0,1)}(x)
$$

  This is the pdf of $Beta(n,1)$ distribution.

* **General Formula for Uniform Distribution**

  Let $X_{1}, X_{2}, \ldots, X_{n}$ be i.i.d. $U(0,1)$. So $f_{X}(x) =1$ and $F_{X}(x) = x$ for all $x \in [0,1]$. Therefore, the pdf of $j$-th order statistics is:


$$
f_{X_{(j)}} (x) = \frac{n!}{(j-1)! (n-j)!}x^{j-1}(1-x)^{n-j}, \,\,\, 0 < x_{j} < 1
$$

  Hence, $X_{(j)} \sim Beta(j, n − j + 1)$. From this we can deduce

$$
E(X_{(j)}) = \frac{j}{n+1}
$$

  and

$$
Var(X_{(j)}) = \frac{j(n-j+1)}{(n+1)^{2}(n+2)}
$$

* **General Formula for Distribution of $j$-th Order Statistics**

  Let $X_{(1)}, X_{(2)}, \ldots, X_{(n)}$ denote the order statistics of a random sample $X_{1}, X_{2}, \ldots, X_{n}$ from a continuous population with pdf $f_{X}(x)$ and cdf $F_{X}(x)$. General formula for the pdf of $j$-th order statistics is given by

$$
f_{X_{(j)}} (x) = \frac{n!}{(j-1)! (n-j)!} f_{X}(x) [F_{X}(x)]^{j-1} [1-F_{X}(x)]^{n-j}
$$

#### What are the properties of an estimator?

Let $\theta$ be a population parameter. Let $\hat{\theta}$ a sample estimate of that parameter. Desirable properties of $\hat{\theta}$ are: 

* **Unbiased**: A statistic (estimator) is said to be an unbiased estimate of a given parameter when the mean of the sampling distribution of that statistic can be shown to be equal to the parameter being estimated, that is, $E(\hat{\theta}) = \theta$. For example, $E(\bar{X}) = \mu$ and $E(s^{2}) = \sigma^{2}$.

* **Efficiency**: The most efficient estimator among a group of unbiased estimators is the one with the smallest variance. For example, both the sample mean and the sample median are unbiased estimators of the mean of a normally distributed variable. However, $\bar{X}$ has the smallest variance.

* **Sufficiency**: An estimator is said to be sufficient if it uses all the information about the population parameter that the sample can provide. The sample median is not sufficient, because it only uses information about the ranking of observations. The sample mean is sufficient. 

* **Consistency**: An estimator is said to be consistent if it yields estimates that converge in probability to the population parameter being estimated as $N$ becomes larger. That is, as $N$ tends to infinity, $E(\hat{\theta}) = \theta$ , $V(\hat{\theta}) = 0$. For example, as $N$ goes to infinity, $V(\bar{X}) = \frac{\sigma^{2}}{N} = 0$. 

#### Explain Method Of Moments (MoM), Maximum A Posteriori (MAP), and Maximum Likelihood Estimation (MLE).

MAP (maximum a priori estimate), MLE (maximum likelihood estimate) and MoM (method of moments) in this context refer to point estimation problems and are among many other estimation methods in statistics.

Let $x_{1}, x_{2}, \dots , x_{n}$ be a set of $n$ independent and identically distributed data points. A point estimator or statistic is any function of data $\hat{\theta} = g(x_{1}, x_{2}, \dots , x_{n})$. The definition does not require that $g$ return a value that is close to the true $\theta$ or even that the range of $g$ be the same as the set of allowable values of $theta$. This definition of point estimator is very general and would enable the designer of an estimator great flexibility. 

##### Method of Moments Estimation (MoM)

In short, the method of moments involves equating sample moments with theoretical moments. So, let's start by making sure we recall the definitions of theoretical moments, as well as learn the definitions of sample moments.

1. $E(X^k)$ is the $k$th (theoretical) moment of the distribution (about the origin), for $k = 1, 2, \dots $

2. $E\left[(X-\mu)^k\right]$ is the $k$th (theoretical) moment of the distribution (about the mean), for $k = 1, 2, \dots $

3. $M_k=\dfrac{1}{n}\sum\limits_{i=1}^n X_i^k$ is the $k$th sample moment, for $k = 1, 2, \dots $

4. $M_k^\ast =\dfrac{1}{n}\sum\limits_{i=1}^n (X_i-\bar{X})^k$ is the $k$th sample moment about the mean, for k = 1, 2, ...

The basic idea behind this form of the method is to:

1. Equate the first sample moment about the origin $M_1=\dfrac{1}{n}\sum\limits_{i=1}^n X_i=\bar{X}$ to the first theoretical moment $E(X)$.

2. Equate the second sample moment about the origin $M_2=\dfrac{1}{n}\sum\limits_{i=1}^n X_i^2$ to the second theoretical moment $E(X^{2})$.

3. Continue equating sample moments about the origin, $M_{k}$, with the corresponding theoretical moments $E(X^{k}), k = 3, 4, \dots $ until you have as many equations as you have parameters.

4. Solve for the parameters.

The resulting values are called method of moments estimators. It seems reasonable that this method would provide good estimates, since the empirical distribution converges in some sense to the probability distribution.  Therefore, the corresponding moments should be about equal.

In some cases, rather than using the sample moments about the origin, it is easier to use the sample moments about the mean. Doing so, provides us with an alternative form of the method of moments.

The basic idea behind this form of the method is to:

1. Equate the first sample moment about the origin $M_1=\dfrac{1}{n}\sum\limits_{i=1}^n X_i=\bar{X}$ to the first theoretical moment $E(X)$.

2. Equate the second sample moment about the mean $M_2^\ast=\dfrac{1}{n}\sum\limits_{i=1}^n (X_i-\bar{X})^2$ to the second theoretical moment about the mean $E[(X-\mu)^2]$.

3. Continue equating sample moments about the mean M∗k with the corresponding theoretical moments about the mean $E[(X-\mu)^k], k = 3, 4, \dots $. until you have as many equations as you have parameters.

4. Solve for the parameters.

Again, the resulting values are also called method of moments estimators. 

Method of moments is simple (compared to other methods like the maximum likelihood method) and can be performed by hand. They might provide starting values in search for maximum likelihood estimates.

However, the parameter estimates may be inaccurate. This is more frequent with smaller samples and less common with large samples. The method may not result in sufficient statistics. In other words, it may not take into account all of the relevant information in the sample. They may not be unique in a given set of data (Multiple solutions to set of equations). Also, they need not exist.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/MoM_ex1.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/MoM_ex2.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/MoM_ex3.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/MoM_ex4.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/MoM_ex5.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/MoM_ex6.png?raw=true)

##### Maximum Likelihood Estimation (MLE)

Maximum likelihood estimation is a method that determines values for the parameters of a model. The parameter values are found such that they maximise the likelihood that the process described by the model produced the data that were actually observed.

Let $X_{1}, X_{2}, \dots , X_{n}$ be a random sample from a distribution that depends on one or more unknown parameters $\theta_{1}, \theta_{2}, \dots , \theta_{m}$ with probability density (or mass) function $f(x_{i}; \theta_{1}, \theta_{2}, \dots , \theta_{m})$. Suppose that $(\theta_{1}, \theta_{2}, \dots , \theta_{m})$ is restricted to a given parameter space $\omega$. Then:

When regarded as a function of $\theta_{1}, \theta_{2}, \dots , \theta_{m}$, the joint probability density (or mass) function of $X_{1}, X_{2}, \dots , X_{n}$:

$$
L(\theta_1, \theta_2, \ldots ,\theta_m) = \prod\limits_{i=1}^n f(x_i;\theta_1,\theta_2,\ldots,\theta_m)
$$

is called the likelihood function. 

Note that the likelihood function is a function of $\theta_{1}, \theta_{2}, \dots , \theta_{m}$ and It is not a probability density function. Then we need to find the values of $\theta_1,\theta_2,\ldots,\theta_m$ that maximizes this likelihood. Rather than maximising this product which can be quite tedious  (for computational issues), we often use the fact that the logarithm is an increasing function so it will be equivalent to maximise the log likelihood:

$$
l(\theta_1,\theta_2,\ldots,\theta_m) = log\left( L(\theta_1,\theta_2,\ldots,\theta_m) \right) =  \sum\limits_{i=1}^n log(f(x_i;\theta_1,\theta_2,\ldots,\theta_m))
$$

In order to use MLE, we have to make two important assumptions, which are typically referred to together as the i.i.d. assumption. These assumptions state that: Data must be independently distributed. Data must be identically distributed.

We can find the MLEs graphically, analytically (We use calculus to find it by taking the derivative of the likelihood function and setting it to 0) or numerically (using Grid Search or gradient descent algorithm or Newton-Raphson algorithm). 

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/MLE_ex1.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/MLE_ex2.png?raw=true)

Note that maximum likelihood estimation involves treating the problem as an optimization or search problem, where we seek a set of parameters that results in for the best fit for the joint probability of the data sample. A limitation of maximum likelihood estimation is that it assumes that the dataset is complete or fully observed. This does not mean that the model has access to all the data; instead, it assumes that all the variables that are relevant to the problem are present. However, this is not always the case. There may be datasets where only some of the relevant variables can be observed and some cannot and although they influence other random variables in the dataset, they remain hidden. More generally, these unobserved or hidden variables are referred to as latent variables. In such cases, a technique called EM algorithm can be used. It stands for expectation-maximization algorithm. EM Algorithm is an iterative approach that cycles between these two modes. The first mode attempts to estimate the missing or latent variables, called the estimation-step or E-step. The second mode attempts to optimize the parameters of the model to best explain the data, called the maximization-step or M-step. The EM algorithm can be applied quite widely, although is perhaps most well-known in machine learning for use in unsupervised learning problems, such as density estimation and clustering.

#### Maximum A Posteriori (MAP) Estimation

Rather than simply returning maximum likelihood estimate, we can still gain some of the benefit of Bayesian approach by allowing the prior influence the choice of the point estimate. Because, as the name suggests, MAP estimation works on a posterior distribution, not only the likelihood.

Recall, with Bayes' rule, we could get the posterior as a product of likelihood and prior:

$$
\begin{align}
P(\theta \vert X) &= \frac{P(X \vert \theta) P(\theta)}{P(X)} \\
                  &\propto P(X \vert \theta) P(\theta)
\end{align}
$$

We are ignoring the normalizing constant as we are strictly speaking about optimization here, so proportionality is sufficient.

We know that MLE can be written as:

$$
\begin{align}
\theta_{MLE} &= \mathop{\rm arg\,max}\limits_{\theta} \log P(X \vert \theta) \\
             &= \mathop{\rm arg\,max}\limits_{\theta} \log \prod_i P(x_i \vert \theta) \\
             &= \mathop{\rm arg\,max}\limits_{\theta} \sum_i \log P(x_i \vert \theta)

\end{align}
$$

If we replace the likelihood in the MLE formula above with the posterior, we get:

$$
\begin{align}
\theta_{MAP} &= \mathop{\rm arg\,max}\limits_{\theta} P(\theta \vert X) \\
             &= \mathop{\rm arg\,max}\limits_{\theta} P(X \vert \theta) P(\theta) \\
             &= \mathop{\rm arg\,max}\limits_{\theta} \log P(X \vert \theta) + \log P(\theta) \\
             &= \mathop{\rm arg\,max}\limits_{\theta} \log \prod_i P(x_i \vert \theta) + \log P(\theta) \\
             &= \mathop{\rm arg\,max}\limits_{\theta} \sum_i \log P(x_i \vert \theta) + \log P(\theta)
\end{align}
$$

Here, $P(\theta)$ is the prior. The prior is the probability of the parameter and represents what was thought before seeing the data. $P(X \vert \theta)$ is the likelihood. The likelihood is the probability of the data given the parameter and
represents the data now available. In the end, we will have the posterior. The posterior represents what is thought given both prior information and the data just seen.

Comparing both MLE and MAP equation, the only thing differs is the inclusion of prior $P(\theta)$ in MAP, otherwise they are identical. What it means is that, the likelihood is now weighted with some weight coming from the prior.

Let’s consider what if we use the simplest prior in our MAP estimation, i.e. uniform prior. This means, we assign equal weights everywhere, on all possible values of the $\theta$. The implication is that the likelihood equivalently weighted by some constants. Being constant, we could be ignored from our MAP equation, as it will not contribute to the maximization. So, we will be back at MLE equation again!

If we use different prior, say, a Gaussian, then our prior is not constant anymore, as depending on the region of the distribution, the probability is high or low, never always the same.

What we could conclude then, is that MLE is a special case of MAP, where the prior is uniform!

Use Bayesian estimations when you have a domain expert; otherwise, use MLE. Use MoM only for computational issues when (1) the posterior (or likelihood function) is not convex and (2) big data.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/MAP_ex1.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/MAP_ex2.png?raw=true)

#### What is score function and Fisher Information Matrix?

Suppose we have a model parameterized by parameter vector $p(x \mid \theta)$ that models a distribution. For
simplicity, we assume that $\theta$ is a scalar. The MLE of $\theta$ is obtained by maximising the log-likelihood function:

$$
l(\theta) = log\left( L(\theta) \right) =  \sum\limits_{i=1}^n log(p(x_i;\theta))
$$

The first and second derivatives of log-likelihood with respect to $\theta$ are important and have their own names.

The first derivative of the log-likelihood function at $\theta$:

$$
S(\theta) = \frac{\partial l(\theta)}{\partial \theta} 
$$

is called the score function (sometimes also called the informant or the score). Computation of the MLE is typically done by solving the score equation $S(\theta) = 0$. Expected value of score function is zero, i.e., $E(S) = 0$.  

$$
\begin{split}
    \mathop{\mathbb{E}}_{p(x \vert \theta)} \left[ S(\theta) \right] &= \mathop{\mathbb{E}}_{p(x \vert \theta)} \left[ \nabla \log p(x \vert \theta) \right] \\[5pt]
    &= \int \nabla \log p(x \vert \theta) \, p(x \vert \theta) \, \text{d}x \\[5pt]
    &= \int \frac{\nabla p(x \vert \theta)}{p(x \vert \theta)} p(x \vert \theta) \, \text{d}x \\[5pt]
    &= \int \nabla p(x \vert \theta) \, \text{d}x \\[5pt]
    &= \nabla \int p(x \vert \theta) \, \text{d}x \\[5pt]
    &= \nabla 1 \\[5pt]
    &= 0
\end{split}
$$

The second derivative, the curvature, of the log-likelihood function is also of central importance and has its own name. The negative of the expected value of second derivative of the log-likelihood function:

$$
I(\theta) = - E\left[ \frac{\partial^{2} l(\theta)}{\partial \theta^{2}} \right]
$$

is called the Fisher information. Strictly, this definition corresponds to the expected Fisher information. For one parameter, Fisher Information is just the variance of score function, $Var(S) = I(\theta)$. Fisher Information Matrix is defined as the covariance of score function for multiple parameters.

However, in practice, the true value of $\theta$ is not known and has to be inferred from the observed data. The value of the Fisher information at the MLE $\hat{\theta_{MLE}}$, i.e. $I\left(\hat{\theta_{MLE}} \right)$, is the *observed* Fisher information. Note that the MLE $\hat{\theta_{MLE}}$ is a function of the observed data, which explains the terminology “observed” Fisher information for $I\left(\hat{\theta_{MLE}} \right)$.

Fisher information is a key concept in the theory of statistical inference and essentially describes the amount of information data provide about an unknown parameter. More formally, it measures the expected amount of information given by a random variable ($X$) for a parameter($\theta$) of interest. The concept is related to the law of entropy, as both are ways to measure disorder in a system

It has applications in finding the variance of an estimator (The Cramer–Rao bound states that the inverse of the Fisher information is a lower bound on the variance of any unbiased estimator of $\theta$), as well as in the asymptotic behavior of maximum likelihood estimates, and in Bayesian inference (for example, a default prior by Jeffreys’s rule). 

Note that the negative expected Hessian of log-likelihood is equal to the Fisher Information Matrix. Hessian is the Jacobian of the gradient of log likelihood, which is defined as:

$$
H(\theta) = \frac{\partial^{2} l(\theta)}{\partial \theta^{2}}
$$

Explicit formulas for the MLE and the observed Fisher information can typically only be derived in simple models. In more complex models, numerical techniques have to be applied to compute maximum and curvature of the log-likelihood function. 

#### What is a Bernoulli distribution? Calculate the expectation and variance of a random variable that follows Bernoulli distribution?

Suppose you perform an experiment with two possible outcomes: either success or failure. Success happens with probability $p$ while failure happens with probability $1-p$. A random variable that takes value $1$ in case of success and $0$ in case of failure is called a Bernoulli random variable.

$X$ has Bernoulli distribution with parameter $p$, the shorthand $X \sim Bernoulli(p), 0 \leq p \leq 1$, its probability mass function is given by:

$$
P_{X}(x) = \left\{ \begin{array}{ll}
         p & \mbox{if $x = 1 $};\\
        1-p & \mbox{if $x  = 0 $}.\end{array} \right.
$$

This can also be expressed as:

$$
P_{X}(x) = p^{x} (1-p)^{1-x},\,\,\, x \in \{0, 1 \}\,\,\,\text{for}\,\,\, 0 \leq p \leq 1
$$

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/bernoulli(p)%20color.png?raw=true)

Bernoulli distribution is a special case of Binomial distribution. If $X_{1}, X_{2}, \dots ,X_{n}$ are independent, identically distributed (i.i.d.) random variables, all Bernoulli trials with success probability $p$, then their sum ($X=X_1+X_2+...+X_n$) is distributed according to a Binomial distribution with parameters $n$ and $p$:

$$
\sum_{k=1}^{n} X_{k} \sim Binomial(n,p)
$$

The Bernoulli distribution is simply $Binomial(1,p)$, also written as $Bernoulli(p)$.

Its expected value is:

$$
E(X) = \sum x p(x) = 1 \times p + 0 \times (1-p) = p
$$

Its variance is:

$$
Var(X) = E(X^{2}) - \left(E(X) \right)^{2} = \sum x^{2} p(x) - p^{2} = 1^{2} \times p + 0^{2} (1-p) - p^{2} = p - p^{2} = p (1-p)
$$

Distribution function of a Bernoulli random variable is:

$$
F_{X}(x) = P(X \leq x) = \left\{ \begin{array}{ll}
         0 & \mbox{if $x < 0 $};\\
        1-p & \mbox{if $0 \leq x < 1 $};\\
        1 & \mbox{if $0 \geq 1 $}.\end{array} \right.
$$

and the fact that $X$ can take either value $0$ or value $1$. If $x<0$, then $P(X \leq x) = 0$ because $X$ can not take values strictly smaller than $0$. If $0 \leq x < 1$, then $P(X \leq x) = 1-p$ because $0$ is the only value strictly smaller than 1 that $X$ can take. Finally, if $x \geq 1$, then $P(X \leq x) = 1$ because all values $X$ can take are smaller than or equal to $1$.

Finding maximum likelihood estimation of the parameter $p$ of Bernoulli distribution is trivial. However, if our experiment is a single Bernoulli trial and we observe $X = 1$ (success) then the likelihood function is $L(p; x) = p$. This function reaches its maximum at $\hat{p} = 1$. If we observe $X = 0$ (failure) then the likelihood is $L(p; x) = 1 − p$ , which reaches its maximum at $\hat{p} = 0$. Of course, it is somewhat silly for us to try to make formal inferences about an estimator on the basis of a single Bernoulli trial; usually multiple trials are available.

If the experiment consists of $n$ Bernoulli trial with success probability $p$, then

$$
L(p;x) = \prod\limits_{i=1}^n p(x_i;p) = \prod\limits_{i=1}^n p^{x_{i}}(1-p)^{1-x_{i}}
$$

Differentiating the log of $L(p;x)$ with respect to $p$ and setting the derivative to zero shows that this function achieves a maximum at $\hat{p} = \frac{\sum_{i=1}^{n} x_{i}}{n}$.

#### What is Binomial distribution?

A Binomial distribution can be thought of as simply the probability of success or failure outcome in an experiment or a survey that is repeated multiple times. The Binomial distribution is a type of distribution that has two possible outcomes (the prefix 'bi' means two or twice). For example, a coin toss has only two possible outcomes: heads or tails. It deals with the number of successes in a fixed number of independent trials. In other words, none of the trials (experiments) have an effect on the probability of the next trials.

Binomial distribution is probably the most commonly used discrete distribution. 

Suppose that $x = (x_{1}, x_{2}, \ldots, x_{n})$ represents the outcomes of $n$ independent Bernoulli trials, each with success probability $p$, then, its probabiliy mass function is given by:

$$
Binomial(x; n,p) = ^nC_{x} p^{x} (1-p)^{n-x} =  {n \choose x} p^{x} (1-p)^{n-x} = \frac{n!}{x!(n-x)!} p^{x} (1-p)^{n-x}
$$

where $n$ is the number of trials ($n \in \{0,1,2,\ldots \}$, $x$ is the number of successes ($k \in \{0,1,\ldots ,n\}$) and $p$ is the success probability for each trial ($p\in [0,1]$).

__Binomial coefficient__, $^nC_{x}$, stated as "n choose k", is also known as "the number of possible ways to choose $k$ successes from $n$ observations.

The formula for the Binomial cumulative probability function is:

$$
F(x; n, p) = P(X \leq x) = \sum_{i=1}^{x} p^{i} (1-p)^{n-i}
$$

Its mean is $E(x) = np$ and its variance is $Var(x) = np(1-p)$. 

Finding maximum likelihood estimation of the parameter $p$ of Binomial distribution is straightforward. Since we know the probability mass function of Binomial distribution, we can write its likelihood function (joint distribution, since Since $x_{1}, x_{2}, \ldots, x_{n}$ are iid random variables):

$$
\begin{split}
L(p) &= \prod_{i=1}^{n} p(x_{i}; n, p)\\
&=\prod_{i=1}^{n} \frac{n!}{x_{i}!(n-x_{i})!} p^{x_{i}} (1-p)^{n-x_{i}}\\
&=p^{\sum_{i=1}^{n} x_{i}} (1-p)^{\left(n-\sum_{i=1}^{n} x_{i}\right)} \left(\prod_{i=1}^{n} \frac{n!}{x_{i}!(n-x_{i})!}  \right)
\end{split}
$$

Since the last term does not depend on the parameter $p$, we consider it as a constant. We can omit the constant, because it is statistically irrelevant.

In practice, it is more convenient to maximize the log of the likelihood function. Because the logarithm is monotonically increasing function of its argument, maximization of the log of a function is equivalent to maximization of the function itself. Taking the log not only simplifies the subsequent mathematical analysis, but it also helps numerically because the product of a large number of small probabilities can easily underflow the numerical precision of the computer, and this is resolved by computing instead the sum of the log probabilities. Therefore, let's take the log of this likelihood function:

$$
ln L(p) = ln(p)\sum_{i=1}^{n} x_{i} + ln(1-p) \left(n - \sum_{i=1}^{n} x_{i} \right)
$$

In order to find maximum likelihood estimator, we need to take first-order derivative of this function with respect to $p$ and set it to zero:

$$
\frac{\partial}{\partial p} ln L(p) = \frac{1}{p}\sum_{i=1}^{n} x_{i} + \frac{1}{1-p} \left(n - \sum_{i=1}^{n} x_{i} \right) = 0
$$

Solving this equation will yield:

$$
\hat{p} = \frac{\sum_{i=1}^{n} x_{i}}{n}
$$

The numerator $\sum_{i=1}^{n} x_{i}$ is the total number of successes observed in $n$ independent trials. $\hat{p}$ is the observed proportion of successes in the $n$ trials. We often call $\hat{p}$ the sample proportion to distinguish it from $p$, the "true" or "population" proportion.

The fact that the MLE based on $n$ independent Bernoulli random variables and the MLE based on a single binomial random variable are the same is not surprising, since the binomial is the result of $n$ independent Bernoulli trials anyway. In general, whenever we have repeated, independent Bernoulli trials with the same probability of success $p$ for each trial, the MLE will always be the sample proportion of successes. This is true regardless of whether we know the outcomes of the individual trials or just the total number of successes for all trials.

#### What is a multinoulli distribution?

In probability theory and statistics, a categorical distribution (also called a generalized Bernoulli distribution, multinoulli distribution) is a discrete probability distribution that describes the possible results of a random variable that can take on one of $k$ possible categories, with the probability of each category separately specified. The categorical distribution is the generalization of the Bernoulli distribution for a categorical random variable, i.e. for a discrete variable with more than two possible outcomes, such as the roll of a die. A single roll of a die that will have an outcome in $\\{1, 2, 3, 4, 5, 6\\}$, e.g. $k=6$. In the case of a single roll of a die, the probabilities for each value would be $1/6$, or about $0.166$ or about $16.6\%$. On the other hand, the categorical distribution is a special case of the multinomial distribution, in that it gives the probabilities of potential outcomes of a single drawing rather than multiple drawings, therefore, $n=1$. 

A common example of a Multinoulli distribution in machine learning might be a multi-class classification of a single example into one of $K$ classes, e.g. one of three different species of the iris flower.

There is only one trial which produces $k \geq 2$ possible outcomes, with the probabilities, $\pi_{1}, \pi_{2}, \ldots ,\pi_{k}$, respectively. If $X$ is a multinoulli random variables, it can take $x_{1}, x_{2}, \ldots , x_{k}$ different values/outcomes, each with different probabilities (In Bernoulli case, a random variables $X$ can only take two values: either success (1) with probability $p$ or failure(0) with probability $1-p$).

$$
\pi_1+\pi_2+\ldots+\pi_k = 1,\,\,\,\,\, 0 \leq \pi_{j} \leq 1\text{ for } j=1,2, \ldots k
$$

Therefore, probability mass function is given by:

$$
P_{X} (x_{1}, x_{2}, \ldots , x_{k})  = \prod_{i=1}^{k} \pi_{i}^{x_{i}}
$$

where $\sum_{i=1}^{k} x_{i} = 1$.

If you are puzzled by the above definition of the joint pmf, note that when $x_{i}=1$ because the i-th outcome has been obtained, then all other entries are equal to 0 and

$$
\begin{split}
\prod_{i=1}^{k} \pi_{i}^{x_{i}} &= \pi_{1}^{x_{1}} \pi_{2}^{x_{2}} \ldots \pi_{i-1}^{x_{i-1}} \pi_{i}^{x_{i}} \pi_{i+1}^{x_{i+1}} \ldots \pi_{k}^{x_{k}}\\
&=\pi_{1}^{0} \pi_{2}^{0} \ldots \pi_{i-1}^{0} \pi_{i}^{1} \pi_{i+1}^{0} \ldots \pi_{k}^{0}\\
&= \pi_{i}
\end{split}
$$

When $\pi_{i} = \frac{1}{k}$ we get discrete uniform distribution, which is a special case.

Note that a sum of independent Multinoulli random variables is a multinomial random variable. 

#### What is a multinomial distribution?

Multinomial distribution is a generalization of Binomial distribution, where each trial has $k \geq 2$ possible outcomes.

Suppose that we have an experiment with

* $n$ independent trials, where
* each trial produces exactly one of the events $E_{1}, E_{2}, \ldots, E_{k}$ (i.e. these events are mutually exclusive and collectively exhaustive), and
* on each trial, $E_{j}$ occurs with probability $\pi_{j}$, $j = 1, 2, \ldots, k$.

Notice that $\pi_{1} + \pi_{2} + \ldots + \pi_{k} = 1$. The probabilities, regardless of how many possible outcomes, will always sum to 1.

Here, random variables are:

$$
\begin{split}
X_{1} &= \text{ number of trials in which }E_{1}\text{ occurs}\\
X_{2} &= \text{ number of trials in which }E_{2}\text{ occurs}\\
&\cdots \\
X_{k} &= \text{ number of trials in which }E_{k}\text{ occurs}
\end{split}
$$

Then $X = (X_{1}, X_{2}, \ldots, X_{k})$ is said to have a multinomial distribution with index $n$ and parameter $\pi = (\pi_{1}, \pi_{2}, \ldots , \pi_{k})$. In most problems, $n$ is regarded as fixed and known.

The individual components of a multinomial random vector are binomial and have a binomial distribution,

$$
\begin{split}
X_{1} &\sim Bin(n, \pi_{1})\\
X_{2} &\sim Bin(n, \pi_{2})\\
&\ldots\\
X_{k} &\sim Bin(n, \pi_{k})
\end{split}
$$

The trials or each person's responses are independent, however, the components or the groups of these responses are not independent from each other. The sample sizes are different now and known. The number of responses for one can be determined from the others. In other words, even though the individual $X_{j}$'s are random, their sum:

$$
X_{1} + X_{2} + \ldots + X_{k} = n
$$

is fixed. Therefore, the $X_{j}$'s are negatively correlated.

If $X = (X_{1}, X_{2}, \ldots, X_{k})$ is multinomially distributed with index $n$ and parameter $\pi = (\pi_{1}, \pi_{2}, \ldots , \pi_{k})$, then we will write $X \sim Mult($n$, \pi)$.

The probability that $X = (X_{1}, X_{2}, \ldots, X_{k})$ takes a particular value $x = (x_{1}, x_{2}, \ldots, x_{k})$ is

$$
P(X_{1} = x_{1}, X_{2} = x_{2}, \ldots, X_{k} =x_{k}  \mid n, \pi_{1}, \pi_{2}, \ldots , \pi_{k}) =\dfrac{n!}{x_1!x_2!\cdots x_k!}\pi_1^{x_1} \pi_2^{x_2} \cdots \pi_k^{x_k}
$$

The possible values of $X$ are the set of $x$-vectors such that each $x_{j} \in \\{0, 1, \ldots , n\\}$ and $x_{1} + \ldots + x_{k} = n$.

If $X \sim Mult(n, \pi)$ and we observe $X = x$, then the loglikelihood function for $\pi$ is:

$$
l(\pi;x)=x_1 \text{log}\pi_1+x_2 \text{log}\pi_2+\cdots+x_k \text{log}\pi_k
$$

Using multivariate calculus, it's easy to maximize this function subject to the constraint

$$
\pi_1+\pi_2+\ldots+\pi_k = 1
$$

the maximum is achieved at

$$
\begin{split}
p &= n^{-1}x\\ &= (x_1/n,x_2/n,\ldots,x_k/n)
\end{split}
$$

the vector of sample proportions. The ML estimate for any individual $\pi_{j}$ is $p_{j} = \dfrac{x_{j}}{n}$, and an approximate $95\%$ confidence interval for $\pi_{j}$ is

$$
p_j \pm 1.96 \sqrt{\dfrac{p_j(1-p_j)}{n}}
$$

because $X_{j} \sim Bin(n, \pi_{j})$. Therefore, the expected number of times the outcome $i$ was observed over $n$ trials is

$$
E(X_{i}) = n \pi_{i}
$$

The covariance matrix is as follows. Each diagonal entry is the variance of a binomially distributed random variable, and is therefore

$$
Var(X_{i})=n \pi_{i}(1-\pi_{i})
$$

The off-diagonal entries are the covariances:

$$
Cov(X_{i}, X_{j})= -n \pi_{i} \pi_{j}
$$

for $i, j$ distinct.

Note that when $k$ is 2 and $n$ is 1, the multinomial distribution is the _Bernoulli distribution_. When $k$ is 2 and $n$ is bigger than 1, it is the _binomial distribution_. When $k$ is bigger than 2 and $n$ is 1, it is the _categorical distribution (multinoulli distribution)_. The Bernoulli distribution models the outcome of a single Bernoulli trial. In other words, it models whether flipping a (possibly biased) coin one time will result in either a success (obtaining a head) or failure (obtaining a tail). The binomial distribution generalizes this to the number of heads from performing n independent flips (Bernoulli trials) of the same coin. The multinomial distribution models the outcome of $n$ experiments, where the outcome of each trial has a categorical distribution, such as rolling a $k$-sided die $n$ times.

**Example 1**: Roll a fair die five times. Here $n = 5$, $k = 6$, and $\pi_{i} = \dfrac{1}{6}$. Our vector $x$ might look like this:

$$
\begin{bmatrix}
1\\2\\0\\2\\0 \\0
\end{bmatrix}
$$

Then $p = \dfrac{5!}{1!2!2!} \dfrac{1}{6}^{1} \dfrac{1}{6}^{2} \dfrac{1}{6}^{2} = 0.0038580246913580236$.

{% highlight python %}
from scipy.stats import multinomial
rv = multinomial(5, [1/6.]*6)
rv.pmf([1,2,0,2,0,0])
#0.003858024691358019
{% endhighlight %}

**Example 2**: Suppose that two chess players had played numerous games and it was determined that the probability that Player A would win is 0.40, the probability that Player B would win is 0.35, and the probability that the game would end in a draw is 0.25. If these two chess players played 12 games, what is the probability that Player A would win 7 games, Player B would win 2 games, and the remaining 3 games would be drawn?

Multinomial for 3 outcomes:

$$
p = \frac{n!}{n_{1}! n_{2}! n_{3}!} p_{1}^{n_{1}} p_{2}^{n_{2}} p_{3}^{n_{3}} = \frac{12!}{7!2!3!} 0.40^{7} 0.35^{2} 0.25^{3} = 0.02483
$$

where

$n = 12$ (12 games are played),

$n_{1} = 7$ (number won by Player A),

$n_{2} = 2$ (number won by Player B),

$n_{3} = 3$ (the number drawn),

$p_{1} = 0.40$ (probability Player A wins)

$p_{2} = 0.35$ (probability Player B wins)

$p_{3} = 0.25$ (probability of a draw)

{% highlight python %}
from scipy.stats import multinomial
rv = multinomial(12, [0.40, 0.35, 0.25])
rv.pmf([7, 2, 3])
#0.02483711999999996
{% endhighlight %}

#### What is a normal distribution?

The normal distribution, also known as Gaussian distribution, is defined by two parameters, mean $\mu$, which is expected value of the distribution and standard deviation $\sigma$ which corresponds to the expected squared deviation from the mean. Mean, $\mu$ controls the Gaussian's center position and the standard deviation controls the shape of the distribution. The square of standard deviation is typically referred to as the variance $\sigma^{2}$. We denote this distribution as $N(\mu, \sigma^{2})$. Properties of Normal Distribution are as follows:

1. Unimodal (Only one mode)
2. Symmetrical (left and right halves are mirror images)
3. Bell-shaped (maximum height (mode) at the mean)
4. Mean, Mode, and Median are all located in the center
5. Asymptotic

Given the mean  and variance, one can calculate probability distribution function of normal distribution with a normalised Gaussian function for a value $x$, the density is:

$$
P(x \mid \mu, \sigma^{2}) = \frac{1}{\sqrt{2\pi \sigma^{2}}} exp \left(- \frac{(x - \mu)^{2}}{2\sigma^{2}} \right)
$$

We call this distribution univariate because it consists of one random variable.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/univariate_normal_distribution.png?raw=true)

Given the assumption that the observations from the sample are i.i.d., the likelihood function can be written as:

$$
\begin{split}
L(\mu, \sigma^{2}; x_{1}, x_{2}, \ldots , x_{n}) &= \prod_{i=1}^{n} f_{X}(x_{i} ; \mu , \sigma^{2})\\
&= \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi \sigma^{2}}} exp \left(- \frac{(x_{i} - \mu)^{2}}{2\sigma^{2}} \right)\\
&= \left(2\pi \sigma^{2} \right)^{-n/2} exp \left(-\frac{1}{2\sigma^{2}} \sum_{i=1}^{n}(x_{i} - \mu)^{2} \right)\\
\end{split}
$$

The log-likelihood function is

$$
l(\mu, \sigma^{2}; x_{1}, x_{2}, \ldots , x_{n}) = \frac{-n}{2}ln(2\pi) -\frac{n}{2} ln(\sigma^{2})-\frac{1}{2\sigma^{2}}\sum_{i=1}^{n}(x_{i} - \mu)^{2} 
$$

If we take first-order partial derivatives of the log-likelihood function with respect to the mean $\mu$ and variance $\sigma^{2}$ and set the equations zero and solve them, we will have the maximum likelihood estimators of the mean and the variance, which are:

$$
\mu_{MLE} = \frac{1}{n} \sum_{i=1}^{n} x_{i}
$$

and 

$$
\sigma_{MLE}^{2} = \frac{1}{n} \sum_{i=1}^{n} \left(x_{i} - \mu_{MLE}\right)^{2}
$$

respectively. Thus, the estimator $\mu_{MLE}$ is equal to the sample mean and the estimator $\sigma_{MLE}^{2}$ is equal to the unadjusted sample variance.

#### What is a uniform distribution?

Uniform Distribution is a family of symmetric probability distribution

$$
f(x) = \frac{1}{b-a}\,\,\, a\leq x \leq b
$$

where $a$ is location parameter and $b-a$ is the scale parameter.

The case where $a=0$ and $b=1$ is called a standard uniform distribution.

The cumulative distribution function of the distribution is given by

$$
\begin{split}
F(x) = P(X \leq x) &= \int_{a}^{x} \frac{1}{b-a} du \\
&= \frac{u}{b-a} \Big|_{a}^{x} \\
&= \frac{x-a}{b-a}\,\,\,  a\leq x \leq b
\end{split}
$$

Inverse cumulative distribution function (percent-point function, quantile function) is

$$
F^{-1}(x) = a + x(b-a)
$$

* **Sampling from Uniform Distribution**

  If $u$ is a value sampled from standard uniform distribution, then the value $a + u(b-a)$ follows uniform distribution according to inverse transform method.
  
  Uniform distribution is also useful for sampling from arbitrary distributions. A general method is the inverse transform sampling method which uses the CDF of the target random variable. However, when the CDF is not known in a closed form, alternative methods such as rejection sampling can be used. 
  
The mean and the variance are given by

$$
E(x) = \frac{a+b}{2}
$$

and

$$
Var(x) = \frac{(b-a)^{2}}{12}
$$

* How to find an estimator for $a$ and $b$?

  Consider having received samples $x_{1}, x_{2}, \ldots , x_{n}$, likelihood function can be written as

  $$
\begin{split}
L(a,b \mid x) &= \prod_{i=1}^{n} f(x_{i}; a,b)\\
&= \frac{1}{(b-a)^{n}} I_{x_{i} \in [a,b]} \\
&=\frac{1}{(b-a)^{n}} I_{a \leq x_{i} \leq b}
\end{split}
$$

  Here, the indicator function $I(A)$ equals to 1 if event $A$ happens and 0 otherwise.

  If $b$ is less than the maximum of observations, then the likelihood is 0. Similarly, if $a$ is greater than the minimum of observations, then the likelihood is also 0, since you have observations lying outside the range of the distribution $[a, b]$, which has probability 0. Then, you make $b$ bigger than maximum or $a$ smaller than the minimum, the denominator of the likelihood gets bigger,  since the difference of $a$ and $b$ clearly gets bigger, so the likelihood is necessarily lower than $b=max(x_{i})$ and $a=min(x_{i})$. Because you want to maximize the likelihood. In order to do so, we need to minimize the value $(b-a)$ in the denominator subject to having all data contained in $[a, b]$.

* **Another Example**

  Uniform distribution $U(0, \theta)$ on the interval $[0, \theta]$. The distribution has pdf

  $$
f(x \mid \theta) = \frac{1}{\theta},\,\,\, 0 \leq x \leq \theta
$$

  The likelihood function is

  $$
L(\theta \mid x_{1}, x_{2}, \ldots , x_{n}) = \frac{1}{\theta^{n}}I_{x_{1}, \ldots , x_{n} \in [0, \theta]}
$$

  The indicator above means is that the likelihood will be $0$ if at least one of the factors is 0 and this will happen if at least one of the observation $x_{i}$ will fall outside of allowed interval $[0, \theta]$. Therefore if we choose $\hat{\theta} = max(x_{1}, x_{2}, \ldots , x_{n})$ as MLE, this likelihood will be maximized.

* **Another Example**

  Suppose again $x_{1}, x_{2}, \ldots , x_{n}$ random sample from Uniform distribution on the interval $(0, \theta)$ where the parameter $\theta > 0$ and is unknown. However, suppose now, we write the density function as

  $$
f(x \mid \theta) = \frac{1}{\theta},\,\,\, 0 < x < \theta
$$

  Be careful, this time, we have strict inequality ($<$ instead of $\leq$). This equation could still be used as the pdf of the uniform distribution, then, an MLE of $\theta$ will be a value of $\theta$ for which $\theta > x_{i}$ for $i=1,2,\ldots ,n$ and which maximizes $\frac{1}{\theta^{n}}$ among all such values. It should be noted that the possible values of $\theta$ no longer include the value $\hat{\theta} = max(x_{1}, x_{2}, \ldots , x_{n})$ since $\theta$ must be strictly greater than each observed value $x_{i}, i=1,2,\ldots , n$. Since $\theta$ can be chosen arbitrarily close to the value $max(x_{1}, x_{2}, \ldots , x_{n})$ but cannot be chosen equal to this value, it follows that the MLE of $\theta$ does not exist in this case.

* **Another Example**

  For $Uniform(-\theta, \theta)$, the likelihood $L(\theta \mid x) = \frac{1}{(2\theta)^{n}}$ for any sample. To maximize this, we need to minimize the value of $\theta$, yet we must keep all samples withing the range $\forall x_{i}, -\theta \leq x_{i} \leq \theta$. An MLE for $\theta$ would be $\hat{\theta} = max(\mid x_{i} \mid )$. This is the smallest value that promises that all sampled points are in the required range.

* **Unbiased Estimator**

  However, $\hat{\theta} = max(x_{1}, x_{2}, \ldots , x_{n}) = max(x_{i}) = X_{max}$ is not unbiased estimator of $\theta$. Since $X_{max}$ is an order statistics, we can find its density function.

  $$
\begin{split}
F_{X_{max}} (x) &= P(X_{max} \leq x) \\
& = P(X_{1} \leq x, X_{2} \leq x, \ldots , X_{n} \leq x)\\
&= P(X_{1} \leq x)P(X_{2} \leq x)\ldots P(X_{n} \leq x)\,\,\, \text{(By independence)}\\
&= \prod_{i=1}^{n} P(X_{i} \leq x)\\
&= \prod_{i=1}^{n} F_{X}(x)\,\,\, \text{(because $X_{i}$'s are identically distributed)}\\
&= \left[F_{X}(x) \right]^{n}\\
&= \left(\frac{x}{\theta} \right)^{n}
\end{split} 
$$

  This is the CDF of $X_{max}$. If we take first-order derivative of it with respect to $x$, we will have the density function of $X_{max}$:

  $$
f_{X_{max}} (x) = \frac{d}{d x} F_{X_{max}} (x) = \frac{n}{\theta} \left( \frac{x}{\theta} \right)^{n-1}
$$

  for $0 \leq x \leq \theta$. If we find the mean (expected value) of this distribution, 
  
  $$
E(X_{max}) = \int_{0}^{\theta} x  \frac{n}{\theta} \left(\frac{x}{\theta} \right)^{n-1} dx = \frac{n}{n+1} \theta
$$

  Therefore $X_{max}$ is biased because $E(\hat{\theta}) = E(X_{max}) \neq \theta$. However, it can be readily patched as

  $$
\hat{X_{max}} =\frac{n+1}{n} X_{max}
$$

  which is the unbiased estimator of $\theta$.
  
#### What is exponential distribution?

In the context of deep learning, we often want to have a probability distribution with a sharp point $x=0$. To accomplish this, we can use exponential distribution. Exponential distribution describes time between events an is mostly used in Poisson events. For example, let's say Poisson distribution models the number of births in a given time period. The time in between each birth can be modeled with an exponential distribution. It is a continuous probability distribution used to model the time we need to wait before a next event occurs (i.e., success, failure, arrival). In other words, it deals with the time between occurrences of successive events as time flows by continuously. Moreover, it is memoryless. It's a special case of the Gamma distribution. For example, "How long will the transmission in my car last before it breaks?". Therefore, it also works for reliability (failure) modeling and service time modeling (queuing theory). Exponential distribution is also special case of Gamma distribution. 

A continuous random variable X is said to be exponential distribution with parameter $\lambda > 0$, shown as $X \sim Exponential(\lambda)$, if its PDF is given by 

$$
f_{X}(x) = \lambda e^{-\lambda x},\,\,\, x\in [0, \infty ); \lambda > 0
$$

Here, $\lambda$ is the event rate (it is not a time duration). Suppose, a passenger is waiting for a bus at a stop and a bus usually arrives at the stop in every 10 mins. Now I define $\lambda$ to be the rate of arrival of a bus **per minute**. So, $\lambda = 1/10$.

Note that this form is parmeterization by rate. If parameterization is done by scale parameter, we will have PDF of exponential distribution as $X\sim Exp(\lambda) \rightarrow f(x) = \frac{1}{\lambda} e^{\frac{-x}{\lambda}}$. One benefit of the second form is that the $\lambda$ there is exactly the mean of the distribution ($E(x) = \lambda$ and $Var(x) = \lambda^{2}$). 

Let's find its CDF:

$$
\begin{split}
F_{X} (x) = P(X \leq x) &= \int_{0}^{x} \lambda e^{-\lambda t} dt\\
&= \lambda \frac{-1}{\lambda} e^{-\lambda t} \Big|_{0}^{x}\\
&= - e^{-\lambda t} \Big|_{0}^{x}\\
&= 1 - e^{-\lambda x}
\end{split}
$$

Let's plot the PDF and CDF of exponential distribution for various $\lambda$'s:

```python
import numpy as np
from scipy.stats import expon
import matplotlib.pyplot as plt

def plot_exp(x_range, loc=0, lamb=1, cdf=False, **kwargs):
    '''
    scale = 1 / lamb
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.expon.html
    '''
    if cdf:
        y = expon.cdf(x, loc=0, scale=1/lamb)
    else:
        y = expon.pdf(x, loc=0, scale=1/lamb)
    plt.plot(x, y, **kwargs)
    
x = np.linspace(0, 5, 5000)

plt.figure(1)
plot_exp(x, 0, 0.5, color='red', lw=2, ls='-', alpha=0.5, label='$\lambda = 0.5$')
plot_exp(x, 0, 1, color='blue', lw=2, ls='-', alpha=0.5, label='$\lambda = 1$')
plot_exp(x, 0, 1.5, color='green', lw=2, ls='-', alpha=0.5, label='$\lambda = 1.5$')
plt.title('PDF of Exponential Distribution')
plt.xlabel('x')
plt.ylabel('Probability density')
plt.axis([0, 5, -0.05, 1.5])
plt.legend()

plt.figure(2)
plot_exp(x, 0, 0.5, cdf=True, color='red', lw=2, ls='-', alpha=0.5, label='$\lambda = 0.5$')
plot_exp(x, 0, 1, cdf=True, color='blue', lw=2, ls='-', alpha=0.5, label='$\lambda = 1$')
plot_exp(x, 0, 1.5, cdf=True, color='green', lw=2, ls='-', alpha=0.5, label='$\lambda = 1.5$')
plt.title('CDF of Exponential Distribution')
plt.xlabel('x')
plt.ylabel('$P(X \leq x)$')
plt.axis([0, 5, -0.05, 1.02])
plt.legend()
```

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/exponential_dist_PDF_CDF.png?raw=true)

Let's find the expected value and variance of the distribution. In order to do so, we need to compute $E(X)$ and $E(X^{2})$.

$$
\begin{split}
E (x) = P(X \leq x) &= \int_{0}^{\infty} x \lambda e^{-\lambda x} dx\\
&= \frac{1}{\lambda} \int_{0}^{\infty} y e^{-y}dy \,\,\,\,\,\,\,\, (\text{choosing } y=\lambda x )\\
&= \frac{1}{\lambda}
\end{split}
$$

Now, let's find $Var(x)$. We have:

$$
\begin{split}
E (x^{2}) = P(X \leq x) &= \int_{0}^{\infty} x^{2} \lambda e^{-\lambda x} dx\\
&= \frac{1}{\lambda^{2}} \int_{0}^{\infty} y^{2} e^{-y}dy \\
&= \frac{1}{\lambda^{2}}\left[-2 e^{-y}-2y e^{-y}-y^{2} e^{-y}  \Big|_{0}^{\infty}\right]
&= \frac{2}{\lambda^{2}}
\end{split}
$$

Thus, we obtain:

$$
Var(x) = E (x^{2}) - \left[E(x) \right]^{2} = \frac{2}{\lambda^{2}}-\left(\frac{1}{\lambda} \right)^{2} = \frac{1}{\lambda^{2}}
$$

The most important propert of exponential distribution is memoryless property (also called the forgetfulness property). Let $X$ be exponentially distributed with parameter $\lambda$. Suppose we know $X > a$. We can state the memoryless property formally as follows:

$$
P(X > x + a \mid X > a) = P(X>x), \,\,\, \text{for } a, x \geq 0
$$

Let's first prove it:

$$
\begin{split}
P(X > x + a \mid X > a) &= \frac{P(X > x + a , X > a)}{P(X > a)} \\
&=\frac{P(X > x + a)}{P(X > a)} \\
&=\frac{1- P(X \leq x + a)}{1 - P(X \leq a)} \\
&=\frac{1- F_{X} (x+a)}{1 - F_{X} (a)} \\
&=\frac{1- (1 - e^{-\lambda (x+a)})}{1 - (1 - e^{-\lambda a})} \\
&=\frac{e^{-\lambda (x+a)}}{e^{-\lambda a}} \\
&=\frac{e^{-\lambda x} e^{-\lambda a} }{e^{-\lambda a}} \\
&= e^{-\lambda x}\\
&=P(X > x)
\end{split}
$$

It turns out that the conditional probability does not depend on $a$! The probability of an exponential random variable exceeding the value $x+a$ given $a$ is the same as the variable originally exceeding that value $x$, regardless of $a$. For an example, the probability that a job runs for one additional hour is the same as the probability that it ran for one hour originally, regardless of how long it’s been running.

The exponential distribution is memoryless because the past has no bearing on its future behavior. Every instant is like the beginning of a new random period, which has the same distribution regardless of how much time has already elapsed ($P(X > 14 \mid X > 8) = P(X > 6)$).

The exponential distribution is the only continuous distribution that is memoryless (or with a constant rate). Geometric distribution, its counterpart, is the only distribution that is memoryless.

There is an interesting relationship between the exponential distribution and the Poisson distribution. The exponential distribution models the time between events, while the Poisson distribution is used to represent the number of events within a unit of time. 

Suppose that the time that elapses between two successive events follows the exponential distribution with a mean of $\lambda$ units of time. Also assume that these times are independent, meaning that the time between events is not affected by the times between previous events. If these assumptions hold, then the number of events per unit time follows a Poisson distribution with mean $\mu = \frac{1}{\lambda}$. Conversely, if the number of events per unit time follows a Poisson distribution, then the amount of time between events follows the exponential distribution

#### What is Poisson distribution?

A Poisson process is a model for a series of discrete event where the average time between events is known but exact timing of events is random. The arrival of an event if independent of the event before (waiting time between events is memoryless). A Poisson process meets the following criteria (however, in reality, many phenomena modeled as Poisson processes do not meet these exactly):

1. Events are independent of each other. The occurance of one event does not affect the probability of another event will occur.
2. The average rate (events per time period) is constant.
3. Two events cannot occur at the same time. 

Poisson distribution represents the distribution of Poisson processes and it is discrete distribution that measures the probability of a given number of events that occur randomly in a specified interval time (space). Poisson distribution is in fact a limited case of binomial distribution. 

Let the discrete random variable $X$ denote the number of times an event occurs in an interval of time (or space). An event can occur $0, 1, 2, \dots$ times in an interval. The average number of events in an interval is designated as $\mu$. $\mu$ is the event rate, also called the *rate parameter*.

The Poisson distribution has a single parameter, $\mu$. The probability mass function of a Poisson distribution is defined as:

$$
f(X = x; \mu) = \frac{\mu^{x} e^{-\mu}}{x!},\,\,\,\, x = 0,1,2,...\text{ and } \mu > 0
$$

which gives the probability of $X$ events occur in an interval. Expected value of this distribution, $E(X) = \mu$ and the variance, $Var(X) = \mu$. 

We can find an estimator of $\mu$ using Maximum Likelihood Estimation approach. Let's write down the log-likelihood function of Poisson distribution:

$$
\begin{split}
l(\mu) = log \prod_{i=1}^{n} P(X= x_{i}) &= \sum_{i=1}^{n} log \left( \frac{\mu^{x_{i}} e^{-\mu}}{x_{i}!} \right)\\
&= \sum_{i=1}^{n} - \mu + x_{i} log(\mu) - log(x_{i}!)\\
&= -n \mu + log(\mu) \sum_{i=1}^{n} x_{i} - \sum_{i=1}^{n} log(x_{i}!)
\end{split}
$$

Then, we take the derivative of $l(\mu)$ with respect to $\mu$ and equate it to zero. Solving for $\mu$ will yield the MLE of $\mu$:

$$
\hat{\mu_{MLE}} = \frac{1}{n} \sum_{i=1}^{n} x_{i}
$$

The Poisson distribution can be understood as a special case of the binominal distribution when studying large numbers with a rare (not zero) but constant occurrence of "successes". This is called Poisson approximation to Binomial distribution. The Poisson distribution approximates the binomial distribution closely when $n$ is very large ($n \geq 20$) and $p$ is very small ($p \leq 0.05$). It is the limiting form of the binomial distribution when $n \to \infty$, $p \to 0$, and $np = \mu$ is a positive constant. So, we can say that the probability mass function of $X$, when $X \sim Binomial(n, p)$ can be approximated by the probability mass function of a $Poisson(\mu)$ random variable. The importance of this is that Poisson PMF is much easier to compute than the binomial.

Its proof can be seen below:

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/poisson_approx_to_binom.png?raw=true)

#### What is Chi-square distribution?

A chi-square distribution is a continuous distribution with $k$ degrees of freedom. This distribution is used for multiple purposes. A chi-square goodness of fit test determines if a sample data matches a population. It is also used to test the independence of two categorical variables, to compare two variables in a contingency table in order to see if they are related. In a more general sense, it tests to see whether distributions of categorical variables differ from each another. It is also used to determine if the standard deviation of a population is equal to a pre-specified value (one-sample hypothesis testing for variance). 

Chi-square distribution is a special case of Gamma distribution when $k = 2 \beta$ and $\alpha = 2$. The gamma distribution has probability density function with $\alpha$ shape parameter and $\beta$ scale parameter:

$$
f(x; \alpha, \beta) = \frac{1}{\alpha^{\beta} \Gamma (\beta)} x^{\beta - 1} e^{-x/\alpha}, \,\,\,\, x \in (0, \infty), \alpha >0, \beta > 0
$$

When $k = 2\beta$ and $\alpha = 2$, this reduces to:

$$
\frac{1}{2^{k/2}\Gamma (k/2)} x^{k/2-1} e^{-x/2},\,\,\,\, \text{$x\in (0,+\infty )$ if $k = 1$, otherwise $x\in [0,+\infty )$}
$$

which is the probability density function of a chi-square random variable with $k$ degrees of freedom, shown as $\chi_{k}^{2}$. The mean of this distribution is $k$, i.e., $E(x) = k$ and its variance is $2k$.

There are couple of transformations to and from Chi-square distribution. Let's look at them now. 

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/chi_square_transformations_1.jpeg?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/chi_square_transformations_2.jpeg?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/chi_square_transformations_3.jpeg?raw=true)

We can use Chi-square distribution to compute the confidence interval for population variance and hypothesis testing:

$$
\begin{split}
P\left( \chi_{1-\alpha/2}^2 \le \frac{(n-1)s^{2}}{\sigma^{2}} \le \chi_{\alpha/2}^2 \right) &= 1 - \alpha\\
P\left( \dfrac{(n-1)s^2}{\chi_{\alpha/2}^2} \le \sigma^2 \le \dfrac{(n-1)s^2}{\chi_{1-\alpha/2}^2} \right) &= 1 - \alpha
\end{split}
$$

F distribution can be approximated by using Chi-square distribution. The ratio of two independent chi-square variables $\chi_{\nu_{1}}^{2}$ and $\chi_{\nu_{2}}^{2}$, each divided by their respective degrees of freedom, is said to have an F distribution with $\nu_{1}$ and $\nu_{2}$ degrees of freedom, denoted by $F_{\nu_{1}, \nu_{2}}$ , and refer to $\nu_{1}$ and $\nu_{2}$ as the "numerator and denominator degrees of freedom", respectively. The expected value and variance of an $F_{\nu_{1}, \nu_{2}}$ random variable, $X$, are $E(X) = \frac{\nu_{2}}{\nu_{2} - 2}$ (assuming $\nu_{2} > 2$ and $Var(X) = \frac{2 \nu_{2}^{2} (\nu_{1}+ \nu_{2} - 2)}{\nu_{1}(\nu_{2} - 2)^{2}(\nu_{2} - 4)}$ (assuming $\nu_{2} > 4$), respectively.

F distribution is used to compare the variances of two populations on the basis of two independent samples of sizes $n_{1}$ and $n_{2}$ taken at random from these populations. Under the assumption of a normal distribution, we can see that the ratio of the two sample variances will have an F distribution multiplied by the ratio of the two population variances. So, if the two population variances are equal, the ratio of the two sample variances will have an F distribution.

Remember that if we take a sample of size $n$ from $X \sim N(0, \sigma^{2}$, we will have $\frac{(n-1)s^{2}}{\sigma^{2}} \sim \chi_{n-1}^{2}$. We can apply the same idea here, We have two samples: one with standard deviation $s_{1}^{2}$ of size $n_{1}$ and one with standard deviation $s_{2}^{2}$ of size $n_{2}$. Therefore:

$$
\begin{split}
\frac{s_{1}^{2}}{s_{2}^{2}} &\sim \frac{\frac{\sigma_{1}^{2} \chi_{n_{1}-1}^{2}}{n_{1}-1}}{\frac{\sigma_{2}^{2} \chi_{n_{2}-1}^{2}}{n_{2}-1}}\\
&\sim \frac{\sigma_{1}^{2}}{\sigma_{2}^{2}} \frac{\chi_{n_{1}-1}^{2}/n_{1}-1}{\chi_{n_{2}-1}^{2}/n_{2}-1}\\
&\sim \frac{\sigma_{1}^{2}}{\sigma_{2}^{2}} F_{n_{1}-1, n_{2}-1}
\end{split}
$$

which can also be written as:

$$
\frac{s_{1}^{2}/\sigma_{1}^{2}}{s_{2}^{2}/\sigma_{2}^{2}} \sim F_{n_{1}-1, n_{2}-1}
$$

The preceding result gives rise to an extremely simple test for comparing two variances. The null hypothesis is $H_{0}: \sigma_{1}^{2} = \sigma_{2}^{2}$ (In this case, $\sigma_{1}^{2}/\sigma_{2}^{2}$ will be $1$), and so the test as traditionally performed is two-sided, comparing the test statistics with the $\alpha/2$ and $1 − \alpha/2$ quantiles of the F distribution with $n_{1} − 1$ and $n_{2} − 1$ degrees of freedom. We can even build a confidence interval for the ratio of two population variances.

#### What is Student’s t-distribution?

The t distribution (also called Student’s t Distribution) is a family of distributions that look almost identical to the normal distribution curve, only a bit shorter and fatter. The t distribution is used instead of the normal distribution when you have small samples. The larger the sample size, the more the t distribution looks like the normal distribution. In fact, for sample sizes larger than 20 (e.g. more degrees of freedom), the distribution is almost exactly like the normal distribution.

When you look at the t-distribution tables, you will see that you need to know the "degrees of freedom" and is just the sample size minus one. 

If $X \sim N(\mu, \sigma^{2})$, then we know $\frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \sim N(0,1)$ (from the sampling distribution of sample mean, which allows for both hypothesis testing and constructing of confidence intervals, when $\sigma^{2}$ is known. When it is unknown, the above test statistics replaces the true (but unknown) variance $\sigma^{2}$ with (Bessel-corrected) sample variance $s^{2}$:

$$
t = \frac{\bar{X} - \mu}{s/\sqrt{n}}, \,\,\,\,\text{ where } s = \frac{\sum_{i=1}^{n} (X_{i} - \bar{X})^{2}}{n-1}
$$

Notice that 

$$
t = \left(\frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \right) \left(\frac{1}{\sqrt{s^{2}/\sigma^{2}}} \right) = \frac{u}{\sqrt{\frac{\chi_{n-1}^{2}}{n-1}}}
$$

where $u$ is the standard normal variable, $u \sim N(0,1)$, and $\chi^{2}$ is a chi-square random variable with $n-1$ degrees of freedom. Therefore, Student's t is the distribution with $n-1$ degrees of freedom

This distribution has mean zero and variance:

$$
\sigma^{2} = \frac{\nu}{\nu -2} (\nu > 2)
$$

The shape of the t-distribution is influenced by its degrees of freedom. Student's t-distribution has the probability density function given by

$$
f(t) = \frac{\Gamma(\frac{\nu+1}{2})} {\sqrt{\nu\pi}\,\Gamma(\frac{\nu}{2})} \left(1+\frac{t^2}{\nu} \right)^{\!-\frac{\nu+1}{2}}
$$

where $\nu$  is the number of degrees of freedom and $\Gamma$  is the gamma function. 

This may also be written as

$$
f(t) = \frac{1}{\sqrt{\nu}\,\mathrm{B} (\frac{1}{2}, \frac{\nu}{2})} \left(1+\frac{t^2}{\nu} \right)^{\!-\frac{\nu+1}{2}}
$$

where $B$ is the Beta function.

#### What is the central limit theorem?

Suppose that we are interested in estimating the average height among all people. Collecting data for every person in the world is impractical, bordering on impossible. While we can’t obtain a height measurement from everyone in the population, we can still sample some people. The question now becomes, what can we say about the average height of the entire population given a single sample

The gist of Central Limit Theorem is that if we take a sufficient number of random samples from any type of distribution with some variance, the distribution of the sample means will be a normal distribution. This new distribution is called a sampling distribution. The mean of the sampling distribution should be approximately equal to the population mean.

Suppose we are sampling from a population with mean $\mu$ and standard deviation $\sigma$. Let $\bar{X}$ be a random variable representing the sample mean of $n$ independently drawn observations.

Assuming that $X_{i}$'s are independent and identically distributed, we know that:

* The mean of sampling distribution of the sample mean $\bar{X}$ is equal to the population mean.

  $$
  \mu_{\bar{X}} = E(\bar{X}) = E\left(\frac{X_{1} + X_{2} + \ldots + X_{n}}{n} \right) = \frac{\mu + \mu + \ldots + \mu}{n} = \frac{n\mu}{n} = \mu
  $$

* Standard deviation of the sampling distribution of the sample mean $\bar{X}$ is equal to $\frac{\sigma}{\sqrt{n}}$.

  $$
  Var(\bar{X}) = Var \left(\frac{X_{1} + X_{2} + \ldots + X_{n}}{n} \right) = \frac{1}{n^{2}} \left(\sigma^{2} + \sigma^{2}+ \ldots + \sigma^{2} \right) = \frac{\sigma^{2}}{n}
  $$
  
  which is called the "standard error of the mean".

Given any random variable $X$, discrete or continuous, with finite mean $\mu$ and finite $\sigma^{2}$. Then, regardless of the shape of the population distribution of $X$, as the sample size $n$ gets larger (The approximation given by the CLT is valid in general for sample sizes larger than $n=30$), the sampling distribution of $\bar{X}$ becomes increasingly closer to normal with mean $\mu$ and variance $\frac{\sigma^{2}}{n}$, that is $\bar{X} \sim N\left(\mu ,  \frac{\sigma^{2}}{n}\right)$ approximately. 

More formally,

$$
Z = \frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \sim N(0,1) \text{ as } n \to \infty
$$
  
What is the population standard deviation $\sigma$ is unknown? Then, it can be replaced by the sample standard deviation $s$, provided $n$ is large that is $\bar{X} \sim N\left(\mu ,  \frac{s^{2}}{n}\right)$. The standard deviation of $\bar{X}$ is referred to as the true standard error of the mean. Since the value $s/\sqrt{n}$ is a sample-based estimate of the true standard error (s.e.), it is commonly denoted as $\hat{s.e.}$. 
  
Sample variance $s^{2}$ is an unbiased estimator of the population variance $\sigma^{2}$ that is $E(s^{2})=\sigma^{2}$.

$$
s^{2} = \frac{1}{n-1} \sum_{i=1}^{n} \left(X_{i} - \bar{X} \right)^{2}
$$

The denominator $n-1$ in the sample variance is necessary to ensure unbiasedness of the population variance.

{% highlight python %}
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

np.random.seed(42)
population = np.random.normal(loc=181, scale=24, size=6000)

sns.distplot(population, color="darkslategrey")
plt.title("Population Distribution", y=1.015, fontsize=20);

pop_mean = population.mean()
#180.92508820056628

pop_std= population.std()
#23.959903286074418

sample_means = []
n = 35
for sample in range(0, 300):
    sample_values = np.random.choice(population, size=n)    
    sample_mean = np.mean(sample_values)
    sample_means.append(sample_mean)
    
sns.distplot(sample_means)
plt.title("Distribution of Sample Means ($n=35$)", y=1.015, fontsize=12)
plt.xlabel("sample mean", labelpad=14)
plt.ylabel("frequency of occurence", labelpad=14);

# Calculate Mean of Sample Means
mean_of_sample_means = np.mean(sample_means)
# 180.88531324062302

std_dev_of_sample_means = np.std(sample_means)
#4.1080239177902005
#which is equal to 23.959903286074418/np.sqrt(34), unbiased estimator of the population variance
{% endhighlight %}

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Screen%20Shot%202019-11-25%20at%2016.11.43.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Screen%20Shot%202019-11-25%20at%2016.11.49.png?raw=true)

#### Write the formulae for Bayes rule.

Bayes' theorem is a formula that describes how to update the probabilities of hypotheses when given evidence. It follows simply from the axioms of conditional probability, but can be used to powerfully reason about a wide range of problems involving belief updates.

Bayes rule provides us with a way to update our beliefs based on the arrival of new, relevant pieces of evidence. For example, if we were trying to provide the probability that a given person has cancer, we would initially just say it is whatever percent of the population has cancer. However, given additional evidence such as the fact that the person is a smoker, we can update our probability, since the probability of having cancer is higher given that the person is a smoker. This allows us to utilize prior knowledge to improve our probability estimations.

Given a hypothesis $A$ and evidence $B$, Bayes' theorem states that the relationship between the probability of the hypothesis before getting the evidence $P(A)$ and the probability of the hypothesis after getting the evidence $P(A \mid B)$ is

$$
\underbrace{P(A \mid B)}_\text{posterior} = \frac{P(A \cap B)}{P(B)} = \frac{P(B \mid A)} {P(B)} \overbrace{P(A)}^\text{prior}
$$

So in the equation we have two random variables $A$ and $B$ and their conditional and marginal probabilities, that's all. Prior $P(A)$ is the probability of $A$ "before" learning about $B$, while posterior $P(A \mid B)$ is the probability of $A$ "after" learning about $B$, where the "before" and "after" refer to your procedure of calculating the probabilities, not any chronological order. The naming convention is that the left hand side is the posterior, while the prior appears in the right hand side part. Using Bayes theorem you can easily switch the sides back and forth (that's the point of the theorem). The usual use case is when you know only $P(B \mid A)$ and $P(A)$, but you don't know $P(A \mid B)$ and want to learn about it

The specific case is Bayesian inference, where we use Bayes theorem to learn about the distribution of the parameter of interest $\theta$ given the data $X$, i.e. obtain the posterior distribution $f(\theta \mid X)$. This is achieved by looking at the likelihood function (that looks at the "evidence" you gathered) and the prior (the distribution of $\theta$ that is assumed before looking at the data).

$$
\underbrace{f(\theta|X)}_\text{posterior}=\frac{\overbrace{f(X|\theta)}^\text{likelihood}\,\overbrace{f(\theta)}^\text{prior}}{\underbrace{f(X)}_\text{normalizing constant}}
$$

#### What is conjugate prior?

Prior probability is the probability of an event before we see the data. In Bayesian Inference, the prior is our guess about the probability based on what we know now, before new data becomes available. Conjugate prior just can not be understood without knowing Bayesian inference. 

Within the Bayesian framework the parameter θ is treated as a random quantity. This requires us to specify a prior distribution $p(\theta)$, from which we can obtain the posterior distribution $p(\theta \mid x)$ using the likelihood from data $P(x \mid \theta)$ via Bayes theorem:

$$
\begin{split}
\underbrace{p(\theta|x)}_{\text{posterior}} =& \frac{p(x|\theta)p(\theta)}{p(x)}\\
&\propto \underbrace{p(x|\theta)}_{\text{likelihood}} \cdot \underbrace{p(\theta)}_{\text{prior}}
\end{split}
$$
 
Basically, the denominator, $p(x)$ is nothing but a normalising constant, i.e., a constant that makes the posterior density integrate to one. Note that this integration will have one well known result if the prior and the likelihood are conjugate.

For some likelihood functions, if you choose a certain prior, the posterior ends up being in the same distribution as the prior. Such a prior then is called a Conjugate Prior.

One problem in the implementation of Bayesian approaches is analytical tractability. In other words, the computations in Bayesian Inference can be heavy or sometimes even intractable. However, when you know that your prior is a conjugate prior, you can skip the `posterior = likelihood * prior` computation. Furthermore, if your prior distribution has a closed-form form expression, you already know what the maximum posterior is going to be. 

All members of the exponential family have conjugate priors.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/exp_families_priors.png?raw=true)

For example, the Beta distribution is conjugate to the Binomial distribution. We know that posterior distribution is computed as:

$$
\underbrace{p(\theta|x)}_{\text{posterior}} \propto \underbrace{p(x|\theta)}_{\text{likelihood}} \cdot \underbrace{p(\theta)}_{\text{prior}}
$$

Likelihood function is coming from a Binomial distribution:

$$
P(x \mid \theta) = {n \choose k} \theta^{x} (1 - \theta)^{n-x}
$$

Let's put Beta prior on $\theta$, where $0 \leq \theta \leq 1$ and domain of Beta distribution is between 0 and 1:

$$
Beta(\theta; a, b) = \frac{\Gamma (a + b)}{\Gamma (a) \Gamma (b)} \theta^{a-1} (1 - \theta)^{b-1}
$$

So posterior is:

$$
\begin{split}
p(\theta|x) &\propto p(x|\theta) \cdot p(\theta)\\
&\propto {n \choose k} \theta^{x} (1 - \theta)^{n-x} \frac{\Gamma (a + b)}{\Gamma (a) \Gamma (b)} \theta^{a-1} (1 - \theta)^{b-1}\\
&\propto \theta^{x} (1 - \theta)^{n-x}\theta^{a-1} (1 - \theta)^{b-1}\\
&\propto \theta^{x + a -1}(1 - \theta)^{n-x+b -1}
\end{split}
$$

The posterior distribution is simply a $Beta(x+a, n-x+b)$ distribution. Effectively, our prior is just adding $a-1$ successes and $b-1$ failures to the dataset.

Note that a mixture of conjugate priors is also conjugate. We can use a mixture of conjugate priors as a prior. For example, suppose we are modelling coin tosses, and we think the coin is either fair, or is biased towards heads. This cannot be represented by a Beta distribution. However, we can model it using a mixture of two Beta distributions. For example, we might use:

$$
p(\theta) = 0.5 Beta(\theta \mid 20, 20) + 0.5 Beta(\theta \mid 30, 10) 
$$

If $\theta$ comes from the first distribution, the coin is fair, but if it comes from the second, it is biased towards heads.

#### How does the denominator in Bayes rule act as normalizing constant?

From a technical point of view, here is the argument:

For densities (but the argument is analogous in the discrete case), we write:

$$
\pi \left( \theta |y\right) =\frac{f\left( y|\theta \right) \pi \left(\theta \right) }{f(y)}
$$

The normalizinging constant can be obtained as, by writing a marginal density as a joint density and then writing the joint as conditional times marginal, with the other parameter integrated out,

$$
\begin{split}
f(y)&=\int f\left( y,\theta \right) d\theta\\
&=\int f\left( y|\theta \right) \pi \left(\theta \right)d\theta
\end{split}
$$

It ensures integration to 1 because

$$
\begin{split}
\int \pi \left( \theta |y\right) d\theta&=\int\frac{f\left( y|\theta \right) \pi \left(\theta \right) }{\int f\left( y|\theta \right) \pi \left(\theta \right)d\theta}d\theta\\ &=\frac{\int f\left( y|\theta \right) \pi \left(\theta \right) d\theta}{\int f\left( y|\theta \right) \pi \left(\theta \right)d\theta}\\
&=1,
\end{split}
$$

where we can "take out" the integral in the denominator out of integral over entire data space, because $\theta$ had already been integrated out there (i.e., denominator does not depend on $\theta$).

#### What is uninformative prior?

If we do not have strong beliefs about what $\theta$ should be, it is common to use an uninformative or non-informative prior, and to let the data speak for itself.

An uninformative prior gives you vague information about probabilities. It’s usually used when you don’t have a suitable prior distribution available. However, you could choose to use an uninformative prior if you don’t want it to affect your results too much.

The uninformative prior isn't really "uninformative", because any probability distribution will have some information. However, it will have little impact on the posterior distribution because it makes minimal assumptions about the model.

The main difficulty in putting noninformative priors is the function used, as a prior probability density has typically an infinite integral and is thus not, strictly speaking, a probability density at all. When formally combined with the data likelihood, sometimes it yields an improper posterior distribution. 

the Jeffreys prior, named after Sir Harold Jeffreys, is a non-informative (objective) prior distribution for a parameter space; it is proportional to the square root of the determinant of the Fisher information matrix:

$$
p\left(\theta \right)\propto {\sqrt  {\det {\mathcal  {I}}\left( \theta\right)}}
$$

which is most often improper, i.e. does not integrate to a finite value. The label "non-informative" associated with Jeffreys' priors is rather unfortunate, as they represent an input from the statistician, hence are informative about something! 

Here, the Fisher information $\mathcal{I}\left(\theta \right)$ is defined when $\theta$ is unidimensional by the second derivative of the log-likelihood:

$$
\mathcal{I}\left(\theta \right) = - E_{\theta} \left(\frac{d^{2} log\left(p(x \mid \theta \right)}{d \theta^{2}} \right)
$$

Jeffreys prior provides a method for constructing a prior distribution over parameters for a given model (likelihood function) such that the prior distribution is "invariant under reparameterization."
 
Jeffrey's prior is not conjugate prior. Jeffreys priors work well for single parameter models, but not for models with multidimensional parameters. They are based on a principle of invariance: one should be able to apply these priors to certain situations, apply a change of variable, and still get the same answer. Suppose we are provided with some model and some data, i.e. with a likelihood function $p(x \mid \theta)$. One should be able to manipulate the likelihood and get a prior on $\theta$, from the likelihood only. Note how this approach goes contrary to the subjective Bayesian frame of mind, in which one first chooses a prior on then $\theta$ and then applies it to the likelihood to derive the posterior:

For example, suppose $x$ is binomially distributed, $x \sim Bin(n, \theta),\,\,\, 0 \leq \theta \leq 1$, whose pdf is given by:

$$
P(x \mid \theta) = {n \choose k} \theta^{x} (1 - \theta)^{n-x}
$$

We want to choose a prior $\pi (\theta)$ that is invariant under reparameterizations. Let's derive a Jeffreys prior for $\theta$. Ignoring the terms that do not depend on $\theta$, we have:

$$
\begin{split}
log\left(p(x \mid \theta \right) &= x log(\theta) + (n-1) log(1-\theta)\\
\frac{d}{d \theta} log\left(p(x \mid \theta \right) &= \frac{x}{\theta} - \frac{n-x}{1-\theta}\\
\frac{d^{2}}{d \theta^{2}} log\left(p(x \mid \theta \right) &= - \frac{x}{\theta^{2}} -  \frac{n-x}{(1-\theta)^{2}}
\end{split}
$$

Since $x \sim Bin(n, \theta)$, $E_{\theta}(x) = n\theta$. Then, we can easily compute the Fisher Information:

$$
\begin{split}
\mathcal{I}\left(\theta \right) &= - E_{\theta} \left(\frac{d^{2} log\left(p(x \mid \theta \right)}{d \theta^{2}} \right)\\
&= - E_{\theta} \left(- \frac{x}{\theta^{2}} -  \frac{n-x}{(1-\theta)^{2}} \right)\\
&= - E_{\theta} \left(- \frac{x}{\theta^{2}} \right) + E_{\theta} \left(\frac{n-x}{(1-\theta)^{2}} \right)\\
&= \frac{1}{\theta^{2}} E_{\theta}(x) + \frac{1}{(1-\theta)^{2}} \left(n - E_{\theta}(x) \right)\\
&= \frac{1}{\theta^{2}} n\theta +  \frac{1}{(1-\theta)^{2}}\left(n - n\theta \right)\\
&= \frac{n}{\theta} + \frac{n}{(1-\theta)}\\
&=\frac{n}{\theta(1-\theta)}
\end{split}
$$

Therefore, Jeffreys prior is

$$
\mathcal{I}\left( \theta\right) \propto \left(\frac{n}{\theta(1-\theta)}\right)^{\frac{1}{2}}
$$

which can be rewritten as

$$
\mathcal{I}\left( \theta\right) \propto \theta^{-\frac{1}{2}} (1 - \theta)^{-\frac{1}{2}}
$$

which is nothing but a $Beta(\frac{1}{2}, \frac{1}{2})$ distribution.

```python
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

def plot_beta(x_range, a, b, loc=0, scale=1, cdf=False, **kwargs):
    '''
    Plots the f distribution function for a given x range, a and b
    If mu and sigma are not provided, standard beta is plotted
    If cdf=True cumulative distribution is plotted
    Passes any keyword arguments to matplotlib plot function
    '''
    x = x_range
    if cdf:
        y = ss.beta.cdf(x, a, b, loc, scale)
    else:
        y = ss.beta.pdf(x, a, b, loc, scale)
    plt.plot(x, y, **kwargs)
    
x = np.linspace(0, 1, 5000)

plot_beta(x, 0.5, 0.5, 0, 1, color='red', lw=2, ls='-', alpha=0.5, label='pdf- Beta(1/2, 1/2)')
plot_beta(x, 1, 1, 0, 1, color='blue', lw=2, ls='-', alpha=0.5, label='pdf - Beta(1, 1)')
plt.xlabel('$\Theta$')
plt.ylabel('$\pi (\Theta)$')
plt.axis([0, 1, 0, 2])
plt.legend()
```

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/jeffreys_and_flat_prior_beta.png?raw=true)

Figure compares the prior density Jeffreys prior with that for a flat prior (which is equivalent to a $Beta(1, 1)$ distribution).

Note that in this case the prior is inversely proportional to the standard deviation. Why does this make sense?
We see that the data has the least effect on the posterior when the true $\theta = \frac{1}{2}$ and has the greatest effect near the extremes, $\theta = 0$ or $\theta = 1$. The Jeffreys prior compensates for this by placing more mass near the extremes of the range, where the data has the strongest effect.

We then find:

$$
\begin{split}
p(\theta|x) &\propto p(x|\theta) \cdot p(\theta)\\
&\propto \theta^{x} (1 - \theta)^{n-x} \theta^{1/2 - 1} (1 - \theta)^{1/2 - 1}\\
&\propto \theta^{x - 1/2} (1 - \theta)^{n - x - 1/2}
\end{split}
$$

Thus, $\theta \mid x \sim Beta(x + 1/2, n - x + 1/2)$.

#### What is population mean and sample mean?

A population is a collection of persons, objects or items of interest. Population mean is the average of the all the elements in the population. Suppose that whole population consists of $N$ observations:

$$
\mu = \frac{1}{N} \sum_{i=1}^{N} X_{i}
$$

A sample is a portion of the whole and, if properly taken, is representative of the population. Sample mean is the arithmetic mean of random sample values drawn from the population. Let's assume that we take $n$ samples from this population:

$$
\bar{X} = \frac{1}{n} \sum_{i=1}^{n} X_{i}
$$

#### What is population standard deviation and sample standard deviation?

Standard deviation measures the spread of a data distribution around the mean of this distribution. It measures the typical distance between each data point and the mean. Standard deviation is the square root of the variance. 

Population standard deviation ($\sigma$):

$$
\sigma^{2} = \frac{\sum_{i=1}^{N} (X_{i} - \mu)^{2}}{N}
$$

Sample standard deviation ($s$):

$$
s^{2} = \frac{\sum_{i=1}^{n} (X_{i} - \bar{X})^{2}}{n-1}
$$

#### Why population standard deviation has N degrees of freedom while sample standard deviation has n-1 degrees of freedom? In other words, why 1/N inside root for population and 1/(n-1) inside root for sample standard deviation?

When we calculate the sample standard deviation from a sample of $n$ values, we are using the sample mean already calculated from that same sample of $n$ values.  The calculated sample mean has already "used up" one of the "degrees of freedom of variability" that is available in the sample.  Only $n-1$ degrees of freedom of variability are left for the calculation of the sample standard deviation.

Another way to look at degrees of freedom is that they are the number of values that are free to vary in a data set. What does “free to vary” mean? Here’s an example using the mean (average): Suppose someone else draws a random sample of, say, $10$ values from a population. They tell you what $9$ of the $10$ sample values are, and they also tell you the sample mean of the $10$ values. From this information, even though they haven't told you the tenth value, you can now calculate it for yourself. Given the nine sample values and the sample mean, the tenth sample value cannot vary:  it is totally predetermined. The tenth value is not free to vary. Essentially, only nine of the ten values are useful for determining the variability of the sample.  In other words, we would need to use $n-1$ as the degrees of freedom for the variability in the sample.

Statistically, it also comes from the fact that $s^{2}$ is the unbiased estimator of $\sigma^{2}$. In statistics, using an unbiased estimator is preferred. 

#### What is the trading-off between bias and variance to minimize mean squared error of an estimator?

Bias and variance measure two different sources of error in an estimator. Bias measures the expected deviation from the true value of the function or parameter. Variance on the other hand provides a measure of the deviation from the expected estimator value that any particular sampling of the data is likely to cause.

What happens when we are given a choice between two estimators, one with more bias and one with more variance? How do we choose between them? 

The most common way to negotiate this trade-off is to compare the mean squared error of the estimates. Let's say we try to estimate $\theta$, named $\hat{\theta}$.

$$
\begin{split}
MSE(\hat{\theta}) &= E \left[(\hat{\theta} - \theta)^{2} \right]\\
&= E \left[(\hat{\theta} - E(\hat{\theta}) + E(\hat{\theta}) - \theta)^{2} \right]\\
&= E \left[ (\hat{\theta} - E(\hat{\theta}))^{2} + 2((\hat{\theta} - E(\hat{\theta}))(E(\hat{\theta}) - \theta)) + (E(\hat{\theta}) - \theta)^{2}\right]\\
&= E\left[(\hat{\theta} - E(\hat{\theta}))^{2} \right] + 2E\left[2((\hat{\theta} - E(\hat{\theta}))(E(\hat{\theta}) - \theta)) \right] + E\left[(E(\hat{\theta}) - \theta)^{2} \right]\\
&= E\left[(\hat{\theta} - E(\hat{\theta}))^{2} \right] +  2 \left(E(\hat{\theta}) - \theta \right) \underbrace{E\left[\hat{\theta} - E(\hat{\theta}) \right]}_{E(\hat{\theta}) - E(\hat{\theta}) = 0} + E\left[(E(\hat{\theta}) - \theta)^{2} \right]\\
&= E\left[(\hat{\theta} - E(\hat{\theta}))^{2} \right] + E\left[(E(\hat{\theta}) - \theta)^{2} \right]\\
&= Var(\hat{\theta}) + \left[Bias(\hat{\theta}, \theta) \right]^{2}
\end{split}
$$

The MSE measures the overall expected deviation -in a squared error sense - between the estimator and the true value of the parameter $\theta$. Evaluation the MSE incorporates both the bias and the variance. Desirable estimators are those with small MSE and these are estimators that manage to keep both their bias and variance somewhat in check. 

#### What is the unbiased estimator and its proof?

In daily life, we use the word “bias” to mean that there is "...a tendency to believe that some people, ideas, etc., are better than others that usually results in treating some people unfairly" (Merriam Webster). In statistics, the word bias — and its opposite, unbiased — means the same thing, but the definition is a little more precise: If your statistic is not an underestimate or overestimate of a population parameter, then that statistic is said to be unbiased.

In everyday life, people who are working with the same information arrive at different ideas/decisions based on the same information. Given the same sample measurements/data, people may derive different estimators for the population parameter (mean, variance, etc.). For this reason, we need to evaluate the estimators on some criteria (bias, etc.) to determine which is best.

If you use an estimator once, and it works well, is that enough proof for you that you should always use that estimator for that parameter? Visualize calculating an estimator over and over with different samples from the same population, i.e. take a sample, calculate an estimate using that rule, then repeat. This process yields sampling distribution for the estimator. We look at the mean of this sampling distribution to see what value our estimates are centered around. We look at the spread of this sampling distribution to see how much our estimates vary. 

Unbiasness is one of the properties of an estimator in Statistics. An estimator is unbiased if, on average, it hits the true parameter value. That is, the mean of the sampling distribution of the estimator is equal to the true parameter value.

If the following holds, where $\hat{\theta}$ is the estimate of the true population parameter $\theta$:

$$
E(\hat{\theta}) = \theta
$$

then the statistic $\hat{\theta}$ is unbiased estimator of the parameter $\theta$. Otherwise, $\hat{\theta}$ is the biased estimator.

In essence, we take the expected value of $\hat{\theta}$, we take multiple samples from the true population and compute the average of all possible sample statistics.

For exampke, if $X_{i}$ is a Bernoulli random variable with a parameter $p$, then, finding maximum likelihood estimation of the parameter $p$ of Bernoulli distribution is trivial. 

$$
L(p;X) = \prod\limits_{i=1}^n p(X_i;p) = \prod\limits_{i=1}^n p^{X_{i}}(1-p)^{1-X_{i}}
$$

Differentiating the log of $L(p;X)$ with respect to $p$ and setting the derivative to zero shows that this function achieves a maximum at $\hat{p} = \frac{\sum_{i=1}^{n} X_{i}}{n}$.

Let's find out the maximum likelihood estimator of $p$ is an unbiased estimator of $p$ or not. 

Since $X_{i} \sim Bernoulli(p)$, we know that $E(X_{i}) = p,\,\, i=1,2, \ldots , n$. Therefore,

$$
E(\hat{p}) =  E \left(\frac{\sum_{i=1}^{n} X_{i}}{n} \right) = \frac{1}{n} \sum_{i=1}^{n} E(X_{i}) = \frac{1}{n}\sum_{i=1}^{n}p = \frac{1}{n} np = p
$$

Therefore, we can safely say that the maximum likelihood estimator is an unbiased estimator of $p$.

However, this is not always the true for some other estimates of population parameters. In statistics, Bessel's correction is the use of $n-1$ instead of $n$ in the formula for the sample variance where $n$ is the number of observations in a sample. 

This method corrects the bias in the estimation of the population variance. It also partially corrects the bias in the estimation of the population standard deviation. However, the correction often increases the mean squared error in these estimations.

In the estimating population variance from a sample where population mean is unknown, the uncorrected sample variance is the mean of the squares of the deviations of sample values from the sample mean (i.e., using a multiplicative factor $\frac{1}{n}$). In this case, the sample variance is a biased estimator of the population variance.

Multiplying the uncorrected sample variance by the factor $\frac{n}{n-1}$ gives the unbiased estimator of the population variance. In some literature, the above factor is called Bessel's correction.

Let $X_{1}, X_{2}, \ldots, X_{n}$ be an i.i.d. random variables, each with the expected value $\mu$ and variance $\sigma^{2}$. For the entire population, $\sigma^{2} = E\left[\left(X_{i} -\mu \right)^{2}\right]$.

When we sample from this population, we want a statistic such that $E(s^{2}) = \sigma^{2}$. Intuitively, we would guess $s^{2} = \frac{\sum_{i=1}^{n} (X_{i} - \bar{X})^{2}}{n}$ where $\bar{X}$ is the mean of the sample, $\bar{X} = \frac{\sum_{i=1}^{n} X_{i}}{n}$.

$$
\begin{split}
E(s^{2}) = E\left(\frac{\sum_{i=1}^{n} (X_{i} - \bar{X})^{2}}{n} \right) &= \frac{1}{n}  E\left(\sum_{i=1}^{n} (X_{i} - \bar{X})^{2} \right)\\
&= \frac{1}{n}  E\left[ \sum_{i=1}^{n} \left((X_{i} - \mu)^{2} - (\bar{X} - \mu) \right)^{2} \right]\\
&=\frac{1}{n}  E\left[ \sum_{i=1}^{n} (X_{i} - \mu)^{2} - 2 \sum_{i=1}^{n} (X_{i} - \mu)(\bar{X} - \mu) + \sum_{i=1}^{n}(\bar{X} - \mu)^{2}  \right]\\
&= \frac{1}{n} \left[ \sum_{i=1}^{n} E (X_{i} - \mu)^{2} - n E (\bar{X} - \mu)^{2}  \right]\\
\end{split}
$$

Substituting $\sigma^{2} = E(X_{i} - \mu)^{2}$ and $Var(\bar{X}) = E (\bar{X} - \mu)^{2} = \frac{\sigma^{2}}{n}$ (from central limit theorem) results in the following:

$$
\begin{split}
E(s^{2}) &= \frac{1}{n} \left(\sum_{i=1}^{n} \sigma^{2} - n \frac{\sigma^{2}}{n} \right)\\
&= \frac{1}{n} \left(n \sigma^{2} - \sigma^{2} \right) \\
&=\frac{n-1}{n} \sigma^{2}
\end{split}
$$

Thus, sample variance $s^{2}$ is a biased estimate of $\sigma^{2}$ because $E(\hat{\theta}) \neq \theta$. Therefore, if we multiple both sides of the equation with $\frac{n}{n-1}$ will do the job.

$$
\frac{n}{n-1} E\left(s^{2}\right) = E\left(\frac{n}{n-1} s^{2}\right) = E\left(\frac{n}{n-1} \frac{\sum_{i=1}^{n} (X_{i} - \bar{X})^{2}}{n}\right) = E\left(\frac{\sum_{i=1}^{n} (X_{i} - \bar{X})^{2}}{n-1}\right) = \sigma^{2}
$$

$s^{2} = \frac{\sum_{i=1}^{n} (X_{i} - \bar{X})^{2}}{n-1}$ is the statistic that is always an unbiased estimator of the desired population parameter $\sigma^{2}$. However note that $s$ is not an unbiased estimator of $\sigma$.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/unbiased_estimator1.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/unbiased_estimator2.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/unbiased_estimator3.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/unbiased_estimator4.png?raw=true)

#### What is the consistency of an estimator?

Let $x_{1}, x_{2}, \dots , x_{n}$ be a set of $n$ independent and identically distributed data points. An estimator $\hat{\theta}$, which is any function of data $\hat{\theta} = g(x_{1}, x_{2}, \dots , x_{n})$, of a parameters $\theta$, can also be shown as $\hat{\theta_{n}}$. So far, we considered the properties of estimators for a training set of fixed size. However, we are also concerned with the behavior of an estimator as the amount of training data grows. In particular, we usually wish that, as the number of data points $n$ in our dataset increases, out point estimates converge to the true value of the corresponding paramters. More formally, we would like to have:

$$
\lim\limits_{n \to \infty} \hat{\theta_{n}} = \theta
$$

Be careful, this limit indicates convergence in probability, meaning that for any $\epsilon > 0$, $P\left(\mid \hat{\theta_{n}} - \theta \mid > \epsilon \right) \to 0$. This condition is known as consistency. It is sometimes referred to as weak consistency, with strong consistency referring to the almost sure convergence of $\hat{\theta_{n}}$ to $\theta$.

The term consistent estimator is short for "consistent sequence of estimators", an idea found in convergence in probability. The basic idea is that you repeat the estimator’s results over and over again, with steadily increasing sample sizes. Eventually — assuming that your estimator is consistent — the sequence will converge on the true population parameter. This convergence is called a limit, which is a fundamental building block of calculus.

Consistency ensures that the bias induced by the estimator diminishes as th enumber of data examples grows. However, the reverse is not true - asymptotic unbiasedness does not imply consistency. 

#### What is the sufficiency of an estimator?

Suppose we have a random sample $X_{1}, X_{2}, \dots , X_{n}$ be a set of $n$ independent and identically distributed data points, taken from a distribution $f(X \mid \theta)$ which relies on an unknown parameter $\theta$ in a parameter space $\Theta$. The purpose of parameter estimation is to estimate the parameter $\theta$ from the random sample. We have already studied three parameter estimation methods: method of moment, maximum likelihood, and Bayes estimation. We can see from the previous examples that the estimators can be expressed as a function of the random sample $X_{1}, X_{2}, \dots , X_{n}$. Such a function is called a statistic (estimator).

Formally, any real-valued function $T = g\left(X_{1}, X_{2}, \dots , X_{n}\right)$ of the observations in the sample is called a statistic. For example, 

$$
\begin{split}
\bar{x} &= \frac{1}{n} \sum_{i=1}^{n} X_{i},\,\,\,\,\, (\text{the sample mean})\\
s^{2} &= \frac{1}{n-1} \sum_{i=1}^{n} \left( X_{i} - \bar{x}\right)^{2} ,\,\,\,\,\, (\text{the sample variance})\\
T_{1} &= max \{X_{1}, X_{2}, \dots , X_{n}\}\\
T_{2} &= 5
\end{split}
$$

The last statistic is a bit strange (it completely igonores the random sample), but it is still a statistic. We say a statistic $T$ is an estimator of a population parameter if $T$ is usually close to $\theta$. The sample mean is an estimator for the population mean; the sample variance is an estimator for the population variation. 

Obviously, there are lots of functions of $X_{1}, X_{2}, \dots , X_{n}$ and so lots of statistics. When we look for a good estimator, do we really need to consider all of them, or is there a much smaller set of statistics we could consider? Another way to ask the question is if there are a few key functions of the random sample which will by themselves contain all the information the sample does.

The concept of sufficiency arises as an attempt to answer the following question: Is there a statistic, i.e. a function $T\left(X_{1}, X_{2}, \dots , X_{n}\right)$, that contains all the information in the sample about $\theta$?

We know that the estimators we obtained are always functions of the observations, i.e., the estimators are statistics, e.g. sample mean, sample standard deviations, etc. In some sense, this process can be thought of as "compress" the original observation data: initially we have $n$ numbers, but after this "compression", we only have 1 numbers. This "compression" always makes us lose information about the parameter, can never makes us obtain more information. The best case is that this "compression" result contains the same amount of information as the information contained in the $n$ observations. We call such a statistic as sufficient statistic.

The mathematical definition is as follows. A statistic $T = g\left(X_{1}, X_{2}, \dots , X_{n}\right)$ is a sufficient statistic for $\theta$ if for each $t$, the conditional distribution of $X_{1}, X_{2}, \dots , X_{n}$ given $T = t$ and $\theta$ does not depend on $\theta$.

$$
\begin{split}
p(X \mid \theta, T = t) &= P(X_{1} = x_{1}, X_{2} = x_{2}, \dots, X_{n} = x_{n} \mid \theta, T = t)\\
&= P(X_{1} = x_{1}, X_{2} = x_{2}, \dots, X_{n} = x_{n} \mid T = t) \\
&= \frac{P(X_{1} = x_{1}, X_{2} = x_{2}, \dots, X_{n} = x_{n})}{P(T = t)}
\end{split}
$$

To motivate the mathematical definition, we consider the following "experiment". Let $T = g\left(X_{1}, X_{2}, \dots , X_{n}\right)$ be a sufficient statistic. There are two statisticians; we will call them A and B. Statistician A knows the entire random sample $X_{1}, X_{2}, \dots , X_{n}$, but statistician B only knows the value of $T$, call it $t$. Since the conditional distribution of $X_{1}, X_{2}, \dots , X_{n}$ given $\theta$ and $T$ does not depend on $\theta$, statistician B knows this conditional distribution. So he can
use his computer to generate a random sample $X_{1}^{\prime}, X_{2}^{\prime}, \dots , X_{n}^{\prime}$ which has this conditional distribution. But then his random sample has the same distribution as a random sample drawn from the population (with its unknown value of $\theta$). So statistician B can use his random sample $X_{1}^{\prime}, X_{2}^{\prime}, \dots , X_{n}^{\prime}$ to compute whatever statistician A computes using his random sample $X_{1}, X_{2}, \dots , X_{n}$, and he will (on average) do as well as statistician A. Thus the mathematical definition of sufficient statistic implies the heuristic definition.

Based on this information, we can give some examples. The median, because it considers only rank, is not sufficient. The sample mean considers each member of the sample as well as its size, so is a sufficient statistic. Or, given the sample mean, the distribution of no other statistic can contribute more information about the population mean.

It is difficult to use the definition because you need to evaluate a conditional distribution to check if a statistic is sufficient or to find a sufficient statistic. Luckily, there is a theorem that makes it easy to find sufficient statistics.

**Factorization Theorem**: Let $X_{1}, X_{2}, \dots , X_{n}$ form a random sample from either a continuous distribution or a discrete distribution for which the pdf or the pmf is $f(x \mid \theta)$, where the value of $\theta$ is unknown and belongs to a given parameter space $\Theta$. A statistic $T\left(X_{1}, X_{2}, \dots , X_{n}\right)$ is a sufficient statistic for $\theta$ if and only if the joint pdf or the joint pmf $f_{n}( X \mid \theta)$ of $X_{1}, X_{2}, \dots , X_{n}$ (the likelihood function of a random variable) can be factorized as follows for all values of $X = \left(X_{1}, X_{2}, \dots , X_{n} \right) \in  R^{n}$ and all values of $\theta \in \Theta$:

$$
f_{n}( X \mid \theta) = u(X) v[T(X), \theta]
$$

Here, the function $u$ and $v$ are nonnegative, the function $u$ may depend on $X$ but does not depend on $\theta$, and the function $v$ depends on $\theta$ but will depend on the observed value $X$ only through the value of the statistic $T(X)$.

#### What is the standard error of the estimate? 

The variance, or the standard error, of an estimator provides a measure of how we would expect the esimate we compute from the data to vary as we independently resample the dataset from the underlying data-generating process. For a given sample size, the standard error equals the standard deviation divided by the square root of the sample size. 

standard deviation: $s = \sqrt{\frac{\sum_{i=1}^{n} (X_{i} - \bar{X})^{2}}{n-1}}$

Variance = $s^{2}$

standard error: $s_{\bar{X}} = \sqrt{\frac{\sigma^{2}}{n}}$

where $n$ is the size of the sample and $\bar{X}$ is the sample mean.

#### What is the sampling distribution of the sample mean?

The sampling distribution of a population mean is generated by repeated sampling and recording of the means obtained. This forms a distribution of different means, and this distribution has its own mean and variance. 

When you are conducting research, you often only collect data of a small sample of the whole population. Because of this, you are likely to end up with slightly different sets of values with slightly different means each time. If you take enough samples from a population, the means will be arranged into a distribution around the true population mean. The standard deviation of this distribution, i.e. the standard deviation of sample means, is called the standard error. The standard error tells you how accurate the mean of any given sample from that population is likely to be compared to the true population mean. When the standard error increases, i.e. the means are more spread out, it becomes more likely that any given mean is an inaccurate representation of the true population mean.

The sample mean follows a normal distribution with mean $\mu$ and variance $\frac{\sigma^{2}}{n}$. This comes from the fact that sum of independent normal random variables. For details, look [here](https://newonlinecourses.science.psu.edu/stat414/node/172/){:target="_blank"} and [here](https://newonlinecourses.science.psu.edu/stat414/node/173/){:target="_blank"}.

Assuming that $X_{i}$'s are independent and identically distributed, we know that:

* The mean of sampling distribution of the sample mean $\bar{X}$ is equal to the population mean.

  $$
  \mu_{\bar{X}} = E(\bar{X}) = E\left(\frac{X_{1} + X_{2} + \ldots + X_{n}}{n} \right) = \frac{\mu + \mu + \ldots + \mu}{n} = \frac{n\mu}{n} = \mu
  $$

* Standard deviation of the sampling distribution of the sample mean $\bar{X}$ is equal to $\frac{\sigma}{\sqrt{n}}$.

  $$
  Var(\bar{X}) = Var \left(\frac{X_{1} + X_{2} + \ldots + X_{n}}{n} \right) = \frac{1}{n^{2}} \left(\sigma^{2} + \sigma^{2}+ \ldots + \sigma^{2} \right) = \frac{\sigma^{2}}{n}
  $$
  
  which is called the "standard error", that is the Standard Deviation of the population mean.

#### What is the sampling distribution of the sample variance?

If $X_{1}, X_{2}, \ldots , X_{n}$ are iid $N(\mu, \sigma^{2})$ random variables, then,

$$
\frac{n-1}{\sigma^{2}}s^{2} \sim \chi_{n-1}^{2}
$$

The proof is given below from this [Stackexchange.com link](https://math.stackexchange.com/a/47013/45210){:target="_blank"}.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/proof_sampling_dist_sample_variance.png?raw=true)

For the chi-square distribution, mean and the variance are

$$
E(\chi_{\nu}^{2}) = \nu
$$

and 

$$
Var(\chi_{\nu}^{2}) = 2\nu
$$

we can use this to get the mean and variance of $s^{2}$.

$$
E(s^{2}) = E\left(\frac{\sigma^{2} \chi_{n-1}^{2}}{n-1} \right) = \frac{\sigma^{2}}{n-1}n-1 = \sigma^{2}
$$

and similarly,

$$
Var(s^{2}) = Var \left(\frac{\sigma^{2} \chi_{n-1}^{2}}{n-1} \right) =  \frac{\sigma^{4}}{(n-1)^{2}} 2(n-1) = \frac{2\sigma^{4}}{n-1}
$$

#### What is the sampling distribution of sample proportion, p-hat?

The Central Limit Theorem has an analogue for the population proportion $\hat{p}$. it is also called Central Limit Theorem with a dichotomous outcome. For large samples, the sample proportion, $\hat{p}$, is approximately normally distributed, with mean $\mu_{\hat{p}} = p$ and standard deviation $\sigma_{\hat{p}} = \sqrt{\frac{p(1-p)}{n}}$, which is also called standard error of $p$. Let's see how this is possible...

Often sampling is done in order to estimate the proportion of a population that has a specific characteristic, such as the proportion of all items coming off an assembly line that are defective or the proportion of all people entering a retail store who make a purchase before leaving. The population proportion is denoted $p$ and the sample proportion is denoted $\hat{p}$. Thus if in reality $43\%$ of people entering a store make a purchase before leaving, $p = 0.43$; if in a sample of $200$ people entering the store, $78$ make a purchase, $\hat{p} = 78/200=0.39$.The sample proportion is a random variable: it varies from sample to sample in a way that cannot be predicted with certainty. Viewed as a random variable it will be written $\hat{p}$. 

Let $X$ count the number of observations in a sample of a specified type. For a random sample, we often model $X \sim Binomial(n, p)$ where $n$ is the sample size; and $p$ is the population proportion. The sample proportion is:

$$
\hat{p} = \frac{X}{n}
$$

Adding a hat to a population parameter is a common statistical notation to indicate an estimate of the parameter calculated from sampled data. So, What is the sampling distribution of $\hat{p}$?

For a binomial distributed random variable, the expected value if $E(X) = np$ and variance is $Var(X) = np(1-p)$. The number $1/n$ is a constant, so

$$
E(\hat{p}) = E\left(\frac{X}{n} \right) = \frac{1}{n} E(X) = \frac{np}{n} = p
$$

Similarly,

$$
Var(\hat{p}) = Var\left(\frac{X}{n} \right) = \frac{1}{n^{2}} Var(X) = \frac{np(1-p)}{n^{2}} = \frac{p(1-p)}{n}
$$

In practice, we don’t know the true population proportion $p$, so we cannot compute the variance of $\hat{p}$. Replacing $p$ with $\hat{p}$ in the standard deviation expression gives us an estimate that is called the standard error of $\hat{p}$:

$$
s.e.(\hat{p}) = \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}
$$

The standard error is an excellent approximation for the standard deviation. We will use it to find confidence intervals, but will not need it for sampling distribution or hypothesis tests because we assume a specific value for $p$ in those cases.

In general, if $np \geq 10$ and $n(1- p) \geq 10$ (SPECIAL NOTE: Some textbooks use 15 instead of 10 believing that 10 is to liberal), the sampling distribution of $\hat{p}$ is about normal with mean of $p$ and standard error $s.e.(p) = \sqrt{\frac{p(1-p)}{n}}$.

#### What is Normal approximation to the Binomial and Continuity Correction?

If $X \sim B(n, p)$ and if $n$ is large and/or $p$ is close to $\frac{1}{2}$, then $X$ is approximately $N(np, np(1-p))$. The binomial distribution is for discrete random variables, whereas the normal distribution is continuous distribution.  The basic difference here is that with discrete values, we are talking about heights but no widths, and with the continuous distribution we are talking about both heights and widths. We need to take this into account when we are using the normal distribution to approximate a binomial using a continuity correction. The correction is to either add or subtract 0.5 of a unit from each discrete x-value. This fills in the gaps to make it continuous.

Once we've made the continuity correction, the calculation reduces to a normal probability calculation. 

Examples:
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/continuity_correction.png?raw=true)

**An example**:

Let $Y\sim Binom(25,0.4)$. Using the normal distribution, compute the probability that $Y \leq 8$ and that $Y=8$. Here, binomial distribution can be approximated by a Normal distribution with parameters $\mu = np = 25 \times 0.4 = 10$ and $\sigma^{2} = np(1-p) = 25 \times 0.4 \times 0.6 = 6@.

$P(Y\leq 8)$ can be approximated as

$$
P(Y\leq 8)\cong P(Y \leq 8.5)= F(8.5)
$$

and its approximated value is

{% highlight python %}
from scipy.stats import norm
import numpy as np
norm.cdf(8.5, loc = 10, scale = np.sqrt(6))
#0.2701456873037099
{% endhighlight %}

and its actual value is

{% highlight python %}
from scipy.stats import binom
import numpy as np
binom.cdf(k=8, n=25, p=0.4, loc=0)
#0.2735314501445727
{% endhighlight %}

The probability of $Y=8$ is computed as

$$
P(Y=8) \cong p(7.5\leq \tilde{Y}\leq 8.5) = F(8.5) - F(7.5)
$$

and its approximated value is

{% highlight python %}
from scipy.stats import norm
import numpy as np
norm.cdf(8.5, loc = 10, scale = np.sqrt(6)) - norm.cdf(7.5, loc = 10, scale = np.sqrt(6))
#0.11642860434001223
{% endhighlight %}

its actual value is

{% highlight python %}
from scipy.stats import binom
import numpy as np
binom.pmf(k=8, n=25, p=0.4, loc=0)
#0.1199797153886391
{% endhighlight %}

#### What does statistically significant mean?

In principle, statistical significance helps quantify whether a result is likely due to chance (or error) or to some factor of interest. More technically, it means that if the Null Hypothesis is true (which means there really is no difference), there’s a low probability of getting a result that is different (In other words, it is very unlikely to observe such a result assuming the null hypothesis is true). To determine whether the observed difference is statistically significant, we look at two outputs of a statistical test: (1) p-value (a conventional threshold for declaring statistical significance is a p-value of less than 0.05), and (2) Confidence interval (a confidence interval around a difference that does not cross zero also indicates statistical significance).

Statistical significance does not mean practical significance. Practical significance refers to the magnitude of the difference, which is known as the effect size. Cohen's D is one of the most common ways to measure effect size. However, when computing the observed effect size, the sample size is not included in the calculation.  For example, Cohen’s d is calculated as the difference between the 2 sample means divided by the pooled standard deviation (SD) for all the data in the 2 independent samples. It is thus the difference in means in standard deviation units, which allows for comparisons of results across studies. The following guidelines or cut-points are conventionally applied in interpreting Cohen’s d:

* A small effect size with a Cohen’s d < 0.2
* A medium effect size with a Cohen’s d of 0.2–0.8
* A large effect size with a Cohen’s d > 0.8

Results are practically significant when the difference is large enough to be meaningful in real life. What is meaningful may be subjective and may depend on the context. Practical significance refers to the importance or usefulness of the result in some real-world context. Many sex differences are statistically significant—and may even be interesting for purely scientific reasons—but they are not practically significant. In clinical practice, this same concept is often referred to as “clinical significance.”

A statistically significant result is not attributed to chance and depends on two key variables: sample size and effect size. Sample size refers to how large the sample for your experiment is. The larger your sample size, the more confident you can be in the result of the experiment (assuming that it is a randomized sample). You will run into sampling errors if your sample size is too low. There is a small effect size (say a 0.1% increase in conversion rate) you will need a very large sample size to determine whether that difference is significant or just due to chance (or error). 

#### What is a p-value?

Before we talk about what p-value means, let’s begin by understanding hypothesis testing where p-value is used to determine the statistical significance of our results, which is our ultimate goal. Hypothesis testing is a technique for evaluating a theory using data. It is the art of testing if variation between two sample distributions can just be explained through random chance or not, based on samples from them (but if we have to conclude that two distributions vary in a meaningful way, we must take enough precaution to see that the differences are not just through random chance. Please, see Type I and Type II error for more explanation). The "hypothesis" refers to researcher's initial belief about the situation before the study. This initial theory is known as *alternative hypothesis* and the opposite is known as *null hypothesis*.

When you perform a hypothesis test in statistics, a p-value helps you determine the significance of your results. Hypothesis tests are used to test the validity of a claim that is made about a population using sample data. This claim that’s on trial is called the null hypothesis. The alternative hypothesis is the one you would believe if the null hypothesis is concluded to be untrue. It is the opposite of the null hypothesis; in plain language terms this is usually the hypothesis you set out to investigate. 

In other words, we’ll make a claim (null hypothesis) and use a sample data to check if the claim is valid. If the claim isn’t valid, then we’ll choose our alternative hypothesis instead. Simple as that. These two hypotheses specify two statistical models for the process that produced the data. 

To know if a claim is valid or not, we use a p-value to weigh the strength of the evidence to see if it’s statistically significant. If the evidence supports the alternative hypothesis, then we’ll reject the null hypothesis and accept the alternative hypothesis. 

p-value is a measure of the strength of the evidence provided by our sample against the null hypothesis. In other words, p-value is the probability of getting the observed value of the test statistics, or a value with even greater evidence against null hypothesis, if the null hypothesis is true. Smaller the p-value, the greater the evidence against the null hypothesis. $0 \leq p \leq 1$. Note that the p-value is not the probability that the null hypothesis is true, or the probability that the alternative hypothesis is false. The p-value is computed under the assumption that a certain model, usually the null hypothesis, is true. This means that the p-value is a statement about the relation of the data (sample we took from the population and compute a statistic from it) to that hypothesis, i.e., `Pr(data | hypothesis)`.

The term significance level, denoted by $\alpha$ or alpha, is used to refer to a pre-chosen probability and the term "p-value" is used to indicate a probability that you calculate after a given study. Traditionally we try to set significance level as .05 or .01 - as in there is only a 5 or 1 in 100 chance that the variation that we are seeing is due to chance. The significance level is the probability of rejecting the null hypothesis when it is true. For example, a significance level of 0.05 indicates a 5% risk of concluding that a difference exists when there is no actual difference. This means that 5% of the time, you are willing to accept a false-positive result.

For example, if you do 100 statistical tests, and for all of them, the null hypothesis is actually true, you'd expect about 5 of the tests to be significant at the P<0.05 level, just due to chance. In that case, you'd have about 5 statistically significant results, all of which were false positives. The cost, in time, effort and perhaps money, could be quite high if you based important conclusions on these false positives, and it would at least be embarrassing for you once other people did further research and found that you'd been mistaken.

If we are given a significance level, i.e., alpha, then we reject null hypothesis if p-value is less than equal the chosen significance level, i.e., accept that your sample gives reasonable evidence to support the alternative hypothesis ($p < \alpha$). The term significance level (alpha) is used to refer to a pre-chosen probability and the term "p-value" is used to indicate a probability that you calculate after a given study. The choice of significance level at which you reject null hypothesis is arbitrary. Conventionally the $5\%$ (less than $1$ in $20$ chance of being wrong), $1\%$ and $0.1\%$ ($p < 0.05, 0.01\text{ and }0.001$) levels have been used. Most authors refer to statistically significant as $p < 0.05$ and statistically highly significant as $p < 0.001$ (less than one in a thousand chance of being wrong).

Some tests do not return a p-value. A fixed level alpha test can be calculated without first calculating a p-value. This is done by comparing the test statistic with a critical value of the null distribution corresponding to the level alpha. This is usually the easiest approach when doing hand calculations and using statistical tables, which provide percentiles for a relatively small set of probabilities. Most statistical software produces p-values which can be compared directly with alpha. There is no need to repeat the calculation by hand.

* If test statistic < critical value: Fail to reject the null hypothesis.
* If test statistic >= critical value: Reject the null hypothesis.

You can use either p-values or confidence intervals to determine whether your results are statistically significant. If a hypothesis test produces both, these results will agree.

The confidence level, which is the probability that the value of a parameter falls within a specified range of values, is equivalent to 1 – the alpha level. So, if your significance level is $0.05$, the corresponding confidence level is $95\%$.

* If the p-value is less than your significance (alpha) level, the hypothesis test is statistically significant.
* If the confidence interval does not contain the null hypothesis value, the results are statistically significant.
* If the p-value is less than alpha, the confidence interval will not contain the null hypothesis value.

```python
# Gaussian Critical Values
# The example below calculates the percent point function 
#for 95% on the standard Gaussian distribution.

# Gaussian Percent Point Function
from scipy.stats import norm
# define probability
p = 0.95
# retrieve value <= probability
value = norm.ppf(p)
print(value)
#1.6448536269514722

# confirm with cdf
p = norm.cdf(value)
print(p)
#0.95

#Running the example first prints the value that marks 95% or less of the observations 
#from the distribution of about 1.65. This value is then confirmed by retrieving the 
#probability of the observation from the CDF, which returns 95%, as expected.

# Student’s t Critical Values
#The example below calculates the percentage point function 
#for 95% on the standard Student’s t-distribution with 10 degrees of freedom.

# Student t-distribution Percent Point Function
from scipy.stats import t
# define probability
p = 0.95
df = 10
# retrieve value <= probability
value = t.ppf(p, df)
print(value)
#1.8124611228107335

# confirm with cdf
p = t.cdf(value, df)
print(p)
#0.949999999999923

#Running the example returns the value of about 1.812 or less that covers 95% 
#of the observations from the chosen distribution. The probability of the value 
#is then confirmed (with minor rounding error) via the CDF.

#Chi-squared Critical Values
#The example below calculates the percentage point function for 95% 
#on the standard Chi-Squared distribution with 10 degrees of freedom.

# Chi-Squared Percent Point Function
from scipy.stats import chi2
# define probability
p = 0.95
df = 10
# retrieve value <= probability
value = chi2.ppf(p, df)
print(value)
#18.307038053275146

# confirm with cdf
p = chi2.cdf(value, df)
print(p)
#0.95

Running the example first calculates the value of 18.3 or less that covers 95% 
#of the observations from the distribution. The probability of this observation 
#is confirmed by using it as input to the CDF.
```

#### What is confidence interval?

The purpose of taking a random sample from a population and computing a statistic, such as the mean from the data, is to approximate the mean of the population. How well the sample statistic estimates the underlying population value is always an issue. In statistical inference, one wishes to estimate population parameters using observed sample data. A confidence interval gives an estimated range of values which is likely to include an unknown population parameter, the estimated range being calculated from a given set of sample data

Confidence intervals are constructed at a confidence level, such as $95\%$, selected by the user. The confidence level refers to the long-term success rate of the method, that is, how often this type of interval will capture the parameter of interest. In other words, if the same population is sampled on numerous occasions and interval estimates are made on each occasion, the resulting intervals would bracket the true population parameter in approximately $95\%$ of the cases.

For example, when we try to construct confidence interval for the true mean of heights of men, the "$95\%$" says that $95$ of $100$ experiments will include the true mean, but $5$ won't. So there is a 1-in-20 chance ($5\%$) that our confidence interval does NOT include the true mean. 

In the same way that statistical tests can be one or two-sided, confidence intervals can be one or two-sided. A two-sided confidence interval brackets the population parameter from above and below. A one-sided confidence interval brackets the population parameter either from above or below and furnishes an upper or lower bound to its magnitude.

Confidence intervals only assess sampling error in relation to the parameter of interest. (Sampling error is simply the error inherent when trying to estimate the characteristic of an entire population from a sample.) Consequently, you should be aware of these important considerations:

* As you increase the sample size, the sampling error decreases and the intervals become narrower (the only limitations are time and financial constraints). If you could increase the sample size to equal the population, there would be no sampling error. In this case, the confidence interval would have a width of zero and be equal to the true population parameter. In general, the narrower the confidence interval, the more information we have about the value of the unknown population parameter. Therefore, we want all of our confidence intervals to be as narrow as possible.
* Confidence intervals only tell you about the parameter of interest and nothing about the distribution of individual values.

General form of (most) confidence intervals is:

$$
\text{Sample estimate} \pm \text{margin of error}
$$

The lower limit is obtained by:

$$
\text{the lower limit L of the interval} = \text{estimate} - \text{margin of error}
$$

The upper limit is obtained by:

$$
\text{the upper limit U of the interval} = \text{estimate} + \text{margin of error}
$$

Once we've obtained the interval, we can claim that we are really confident that the value of the population parameter is somewhere between the value of L and the value of U.

#### What do Type I and Type II errors mean?

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/type_I_type_II_errors.png?raw=true)

__Type I error__ is the rejection of a true null hypothesis (also known as a "false positive" finding or conclusion - null hypothesis is true, but there is evidence against it), while a __type II error__ is the non-rejection of a false null hypothesis (also known as a "false negative" finding or conclusion - null hypothesis is not true, but no evidence against it in our sample).

When the null hypothesis is true and you reject it, you make a type I error. That means that after your hypothesis testing, you conclude that there is a difference while there is actually no difference and the difference we observed is totally due to pure chance. In the literature, the probability of making a type I error is alpha ($\alpha$), which is the level of significance you set prior for your hypothesis test.

The power of a test is one minus the probability of type II error (beta, $\beta$), which is the probability of rejecting the null hypothesis when it is false. In other words, it is the ability to detect a fault when there is actually a fault to be detected. Therefore, power should be maximized when selecting statistical methods. 

The chances of committing these two types of errors are inversely proportional -that is, decreasing Type I error rate increases Type II error rate, and vice versa.Finding the right ballance is both art and science. To decrease your chance of committing a Type I error, simply make your alpha value more stringent. To reduce your chance of committing a Type II error, increase your analyses' power by either increasing your sample size or relaxing your alpha level!

In a drug effectiveness study, a false positive could cause the patient to use an ineffective drug. Conversely, a false negative could mean not using a drug that is effective at curing the disease. Both cases could have a very high cost to the patient’s health.

In a machine learning A/B test, a false positive might mean switching to a model that should increase revenue when it doesn’t. A false negative means missing out on a more beneficial model and losing out on potential revenue increase.

A statistical hypothesis test allows you to control the probability of false positives by setting the significance level, and false negatives via the power of the test. If you pick a false positive rate of 0.05, then out of every 20 new models that don’t improve the baseline, on average 1 of them will be falsely identified by the test as an improvement.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/confusion_matrix.png?raw=true)

In the context of confusion matrix, we can say Type I error occurs when we classify a value as positive (1) when it is actually negative (0). For example, false-positive test result indicates that a person has a specific disease or condition when the person actually does not have it. 

Type II error occurs when we classify a value as negative (0) when it is actually positive (1). Similarly, as an example, a false negative is a test result that indicates a person does not have a disease or condition when the person actually does have it

#### What is the power of a statistical test?

When you make a decision in a hypothesis test, there’s never a 100 percent guarantee you’re right. You must be cautious of Type I errors (rejecting a true claim) and Type II errors (failing to reject a false claim). Instead, you hope that your procedures and data are good enough to properly reject a false claim. Mathematically, power is defined to be 1 - Type II error (it is the probability of avoiding a Type II error). Just as the significance level (alpha) of a test gives the probability that the null hypothesis will be rejected when it is actually true (a wrong decision), power quantifies the chance that the null hypothesis will be rejected when it is actually false (a correct decision). In other words, Power is the probability of making a correct decision (to reject the null hypothesis) when the null hypothesis is false. 

$$
\text{power} = Pr \left( \text{reject } H_{0} \mid H_{1} \text{ is true} \right)
$$

Although you can conduct a hypothesis test without it, power must be calculated prior to starting the study so it will help you ensure that the sample size is large enough for the purpose of the test. Otherwise, the test may be inconclusive, leading to wasted resources. On rare occasions the power may be calculated after the test is performed, but this is not recommended except to determine an adequate sample size for a follow-up study (if a test failed to detect an effect, it was obviously underpowered – nothing new can be learned by calculating the power at this stage). 

In reality, a researcher wants both Type I and Type II errors to be small. In terms of significance level and power, this means we want a small significance level (close to 0) and a large power (close to 1).

The power of a hypothesis test is between 0 and 1; if the power is close to 1, the hypothesis test is very good at detecting a false null hypothesis Type II error ($\beta$) of 0.2 was chosen by Cohen, who postulated that an $\alpha$ error was more serious than a $\beta$ error. Therefore, the estimated the $\beta$ error at 4 times the $\alpha: 4 \times 0.05  =  0.20$. In other words, Type II error is commonly set at 0.2, but may be set by the researchers to be smaller. Consequently, power may be as low as 0.8, but may be higher. Powers lower than 0.8, while not impossible, would typically be considered too low for most areas of research.

For example, if experiment E has a statistical power of 0.7, and experiment F has a statistical power of 0.95, then there is a stronger probability that experiment E had a type II error than experiment F. This reduces experiment E's sensitivity to detect significant effects. However, experiment E is consequently more reliable than experiment F due to its lower probability of a type I error.

Before moving on, let's explain some terms that you can see below. Hypothesized distribution of the test statistic is the one under alternative hypothesis and the true distribution of the test statistic is the one under the null hypothesis.

To increase the power of your test, you may do any of the following:

1. Increase the effect size (Magnitude of the effect of the variable - the difference between the null and alternative values) to be detected (The larger the effect, the more powerful the test is, because smaller differences are more difficult to detect which requires larger sample size. When the effect is large, the true distribution of the test statistic is far from its hypothesized distribution, so the two distributions are distinct, and it’s easy to tell which one an observation came from.)
2. Increase the sample size(s) (As n increases, so does the power of the significance test. This is because a larger sample size narrows the distribution of the test statistic. The hypothesized distribution of the test statistic and the true distribution of the test statistic (should the null hypothesis in fact be false) become more distinct from one another as they become narrower, so it becomes easier to tell whether the observed statistic comes from one distribution or the other.)
3. Decrease the variability in the sample(s) (As the variability increases, the power of the test of significance decreases)
4. Increase the significance level (alpha) of the test (If all other things are held constant, then as alpha increases, so does the power of the test. This is because a larger alpha means a larger rejection region for the test and thus a greater probability of rejecting the null hypothesis. That translates to a more powerful test.)

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/power_of_test_graphical.jpg?raw=true)

Using a directional test (i.e., left- or right-tailed) as opposed to a two-tailed test would also increase power, because going from a two-tailed to a one-tailed test cuts the p-value in half. In all of these cases, we say that statistically power is increased. 

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/StatsPower_ex1.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/StatsPower_ex2.png?raw=true)
```python
from scipy.stats import norm
1 - norm.cdf(-0.355, loc=0, scale=1)
#0.6387052043836872
```
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/power_calculation_example1.jpeg?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/StatsPower_ex3.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/StatsPower_ex4.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/StatsPower_ex5.png?raw=true)

#### How to determine sample size?

When you are designing an experiment, it is a good idea to estimate the sample size you'll need. This is especially true if you're proposing to do something painful to humans or other vertebrates, where it is particularly important to minimize the number of individuals (without making the sample size so small that the whole experiment is a waste of time and suffering), or if you're planning a very time-consuming or expensive experiment. Methods have been developed for many statistical tests to estimate the sample size needed to detect a particular effect, or to estimate the size of the effect that can be detected with a particular sample size.

Given magnitude of desired margin of error, confidence level and variability, it is computed for mean:

$$
\frac{\sigma^{2}z^{2}}{ME^{2}}
$$

and for proportion:

$$
\frac{p(1-p)z^{2}}{ME^{2}}
$$

Note that the point estimate becomes more precise as sample size increases. Sample size required to ensure that a test has high power. The sample size computations depend on the level of significance, $\alpha$, the desired power of the test (equivalent to $1 - \beta$), the variability of the outcome (standard deviation), and the effect size. 

The effect size is the difference in the parameter of interest that represents a meaningful difference. This is the size of the difference between your null hypothesis and the alternative hypothesis that you hope to detect. Similar to the margin of error in confidence interval applications, the effect size is determined based on practical criteria and not statistical criteria. Generally, effect size is calculated by taking the difference between the two groups (e.g., the mean of treatment group minus the mean of the control group) and dividing it by the standard deviation of one of the groups. 

To interpret the resulting number, most social scientists use this general guide developed by Cohen:

* `< 0.1` = trivial effect
* `0.1 - 0.3` = small effect
* `0.3 - 0.5` = moderate effect
* `> 0.5` = large difference effect

Your estimate of the standard deviation can come from pilot experiments or from similar experiments in the published literature. 

The standard deviation and effect size can be either determined from previous studies from published literature or from pilot studies. Your standard deviation once you do the experiment is unlikely to be exactly the same, so your experiment will actually be somewhat more or less powerful than you had predicted. Larger the effect size, less sample size we need because the overlap between true distribution and hypothesized distribution will be less. As standard deviation gets bigger, it gets harder to detect a significant difference, so you'll need a bigger sample size. 

The significance level (type 1 error) and the power of the study are fixed before the study. The significance level is normally set at 0.05 or 0.01. For more accuracy, the significance level should be set at lower levels which increase the sample sizes (because larger sample size will narrow the distribution, increasing the power of the test, so overlapping between true distribution and hypothesized distribution will be less and so will alpha value). Anything more than these two levels can affect the study impact and should be done with caution unless it is essential for the study design. For appreciable inference, the power is normally set at 20% chance of missing difference or 80% chance of detecting a difference as statistically significant. This shall provide appreciable study impact.   

Before starting to calculate the sample size, you also need to know what type of test you plan to use (e.g., independent t-test, paired t-test, ANOVA, regression, etc.)

An example:

The standard deviation of systolic blood pressure in US is about 25 mmHg. How large a sample is necessary to estimate the average systolic blood pressure with a margin of error of 4 mmHg at 95% confidence level. 

We know that margin of error at 95% is $ME_{95\%} = 1.96 \times S.E. = 1.96 \times \frac{sigma}{\sqrt{n}}$. We are given that $\sigma = 5$ and $ME_{95\%} = 4$. Therefore,

$$
4 = 1.96 \times \frac{25}{\sqrt{n}} \Rightarrow n =150.06
$$

We should choose a sample size of at least 151 people. 

#### What are the sampling strategies?

A sample is a subset of your population by which you select to be participants in your study. 

Sampling is simply stated as selecting a portion of the population, in your research area, which will be a representation of the whole population. The idea is that when we have huge population, we do not need to investigate every individual in the population. However, we need to use proper techniques while picking a sample, in order to find the best representation of the whole population. Sampling is done to draw conclusions about populations from samples, and it enables us to determine a population’s characteristics by directly observing only a portion (or sample) of the population. Therefore, it is a cost-efficient method

The sampling strategy is the plan you set forth to be sure that the sample you use in your research study represents the population from which you drew your sample. Reducing sampling error is the major goal of any selection techniques. The major groups of sample designs are probability sampling and non-probability sampling. 

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/sampling_strategies.png?raw=true)

Some sampling strategies can also be multi-stage. Multi-stage sampling divides large population into stages to make the sampling process more practical. Multi-stage sampling is also a probability sampling, each stage must involve a probability sampling method. For example, Australian Bureau of Statistics divides cities into "collection districts", then blocks, then households. Each stage uses random sampling, creating a need to list specific households only after the final stage of sampling. 

**Probability sampling**

In probability sampling, all examples have a chance to be selected to be included in a sample. These techniques involve randomness. There are four main methods include: 1) simple random, 2) stratified random, 3) cluster, and 4) systematic. 

1. **Random Sampling**: 
  It is the simplest strategy. Samples are drawn with a uniform probability from the domain (each example has an equal chance, or probability, of being selected - $1/n$ with $n$ being the number of examples in the population). Simplicity is the great advantage of this sampling method, and it can be easily implemented as any programming language can serve as a random number generator. A disadvantage of simple random sampling is that you may not select enough examples that would have a certain property of interest. Consider the situation where you extract a sample from a large imbalanced dataset, and in doing so, you accidentally fail to capture a sufficient number of examples from the minority class - or you may not select any from minority class, at all!
  Also note that random sampling can be done with or without replacement. When we sample with replacement, the two sample values are independent. Practically, this means that what we get on the first one doesn’t affect what we get on the second. Mathematically, this means that the covariance between the two is zero. In sampling without replacement, the two sample values aren’t independent. Practically, this means that what we got on the first one affects what we can get for the second one. Mathematically, this means that the covariance between the two isn’t zero. That complicates the computations.

2. **Systematic Sampling**: 
  In this scheme, samples are drawn using a pre-specified pattern, such as at intervals. The first individual is selected randomly and others are selected using a fixed 'sampling interval'. This is similar to lining everyone up and numbering off "1,2,3,4; 1,2,3,4; etc". When done numbering, all people numbered 4 would be used. First, we create a list containing all example. From that list, we randomly select the first example from the first $k$ elements on the list. Then, we select every $k$th (i.e., 5th) element on the list. We choose such a value of $k$ that will give you a sample of the desired size. For example, if the population size is N=1,000 and a sample size of $n = 100$ is desired, then the sampling interval is $1000/100 = 10$, so every tenth person is selected into the sample. The selection process begins by selecting the first person at random from the first ten subjects in the sampling frame using a random number table; then $10$th subject is selected.
  One advantage of the systematic sampling over the simple random sampling is that random sampling can be inefficient and time-consuming. It also draws examples from the whole range of values, while the simple random sampling may result in examples with some specific properties under-represented in the sample. However, systematic sampling is inappropriate if the list of examples has periodicity or some kind of repetitive pattern. In that latter case, the obtained sample can exhibit a bias. However, if the list of examples is randomized, then systematic sampling often results in a better sample compared to simple random sampling.
  ![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/systematic_sampling.jpg?raw=true)

3. **Stratified Sampling**: 
  An important objective in any estimation problem is to obtain an estimator of a population parameter which can take care of the salient features of the population. If the population is homogeneous with respect to the characteristic under study, then the method of simple random sampling will yield a homogeneous sample, and in turn, the sample mean will serve as a good estimator of the population mean. Thus, if the population is homogeneous with respect to the characteristic under study, then the sample drawn through simple random sampling is expected to provide a representative sample. Moreover, the variance of the sample mean not only depends on the sample size and sampling fraction but also on the population variance. In order to increase the precision of an estimator, we need to use a sampling scheme which can reduce the heterogeneity in the population. If the population is heterogeneous or dissimilar with respect to the characteristic under study, stratified sampling techniques are generally used where certain homogeneous, or similar, sub-populations can be isolated (strata).
  Stratified random sampling is a better method than simple random sampling. Stratified random sampling divides a population into non-overlapping groups, i.e., subgroups or strata (depending characteristics of your population, such as gender, age range, race, country of nationality, and career background), and random samples are taken (using a random sampling method like simple random sampling or systematic sampling), in proportion to the population, from each of the strata created. The members in each of the stratum formed have similar attributes and characteristics. This method of sampling is widely used and very useful when the target population is heterogeneous. A simple random sample should be taken from each stratum. ("Stratum" is singular and "strata" is plural)
  If our objective is to use an allocation that gives us a specified amount of information at minimum cost, then the best allocation scheme is affected by the following three factors: (1) the total number of elements in each stratum, (2) the variability of the measurements within each stratum, and (3) the cost associated with obtaining an observation from each stratum. For example, if a population contains 70% men and 30% women, and we want to ensure the same representation in the sample, we can stratify and sample the numbers of men and women to ensure the same representation. For example, if the desired sample size is $n = 200$, then $n = 140$ men and $n = 60$ women could be sampled either by simple random sampling or by systematic sampling.
  Stratified sampling is the slowest of the sampling methods due to the additional overhead of working with several independent strata.  Sometimes, we can not confidently classify every member of the population into a subgroup because the strata must be mutually exclusive and collectively exclusive. Besides, a stratified random sample can only be carried out if a complete list of the population is available. However, its potential benefit of producing a less biased sample typically outweighs its drawbacks.

4. **Cluster Sampling**: 
  Cluster sampling is typically used in market research. In this sampling method, the whole dataset is first partitioned into distinct clusters. Then a number of clusters are randomly selected and all examples from the selected clusters are then added to the sample. This is different from the stratified sampling where examples are selected from each stratum; in cluster sampling, if a cluster was not selected, none of the examples from this cluster will get to the sample. One drawback of this method, which is similar to that of the stratified sampling, is that the analyst has to have a good understanding of the properties of the dataset.
  At first glance, cluster sampling and stratified sampling seem very similar. For a stratified random sample, a population is divided into stratum, or sub-populations. However, in cluster sampling the actual cluster is the sampling unit; in stratified sampling, analysis is done on elements within each strata. In cluster sampling, a researcher will only study selected clusters; with stratified sampling, a random sample is drawn from each strata. Therefore, cluster sampling is often more economical or more practical than stratified sampling or simple random sampling. However, from all the different type of probability sampling, this technique is the least representative of the population because researcher needs to maintain homogenity between clusters (unlike stratified sampling where we need to maintain homogenity within strata). The tendency of individuals within a cluster is to have similar characteristics and with a cluster sample, there is a chance that the researcher can have an overrepresented or underrepresented cluster which can skew the results of the study. This is also a probability sampling technique with a possibility of high sampling error. This is brought by the limited clusters included in the sample leaving off a significant proportion of the population unsampled.
  ![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/cluster-sampling.png?raw=true)
  There are two types of cluster sampling. In single-stage cluster sampling, a simple random sample of clusters is selected, and data are collected from every unit in the sampled clusters. For instance, let examples in your dataset come from a variety of time periods. You can put in one cluster examples from the same time period. In this case, by applying cluster sampling you include all examples from selected time periods into the sample. In two-stage cluster sampling, a simple random sample of clusters is selected and then a simple random sample is selected from the units in each sampled cluster. For example, an airline company can decide to first randomly choose planes, and then add all passengers from the selected planes to the sample.

**Non-probability sampling**

Non-probability sampling is not random. To build a sample, it follows a fixed deterministic sequence of heuristic actions. This means that some examples don’t have a chance of being selected, no matter how many samples you build. Nonprobability methods are easier for a human to execute manually, though. However, this advantage is not significant for a data analyst working on a computer and using a software or programming code that greatly simplifies sampling of examples even from a very large data asset. The main drawback of nonprobability sampling techniques is that they provide non-representative samples and might systematically exclude important examples from consideration. This drawback outweighs the possible advantages of nonprobability sampling methods.

1. **Convenience Sampling**: 
  This sampling strategy is a type of non-probability sampling, which doesn’t include random selection of participants. Convenience sampling (also called accidental sampling or grab sampling) is where you include people who are easy to reach, rather than selecting subjects at random from the entire population. Although convenience sampling is, like the name suggests -convenient— it runs a high risk that your sample will not represent the population (possibility of under- or over-representation of the population). However, sometimes a convenience sample is the only way you can drum up participants. 
  Convenience sampling does have its uses, especially when you need to conduct a study quickly or you are on a shoestring budget. It is also one of the only methods you can use when you can’t get a list of all the members of a population. For example, let’s say you were conducting a survey for a company who wanted to know what Walmart employees think of their wages. It’s unlikely you’ll be able to get a list of employees, so you may have to resort to standing outside of Walmart and grabbing whichever employees come out of the door (hence the name "grab sampling").
  Results from this sampling method are easy to analyze but hard to replicate. The results are prone to bias due to the reasons why some people choose to take part and some do not. Perhaps the biggest problem with convenience sampling is dependence. Dependent means that the sample items are all connected to each other in some way. This dependency interferes with statistical analysis. Most hypothesis tests (e.g. the t-test or chi-square test) and statistics (e.g. the standard error of measurement), have an underlying assumption of random selection, which you do not have. Perhaps most problematic is the fact that p-values produced for convenience samples can be very misleading.

2. **Quota Sampling**: 
  In quota sampling, we determine a specific number of individuals to select into our sample in each of several specific groups. This is similar to stratified sampling in that we develop non-overlapping groups and sample a predetermined number of individuals within each. For example, suppose our desired sample size is n=300, and we wish to ensure that the distribution of subjects' ages in the sample is similar to that in the population. We know from census data that approximately 30% of the population are under age 20; 40% are between 20 and 49; and 30% are 50 years of age and older. We would then sample n=90 persons under age 20, n=120 between the ages of 20 and 49 and n=90 who are 50 years of age and older.
  
  | Age Group 	| Distribution in Population 	| Quota to Achieve n=300 	|
  |:---------:	|:--------------------------:	|:----------------------:	|
  |    <20    	|             30%            	|          n=90          	|
  |   20-49   	|             40%            	|          n=120         	|
  |    50+    	|             30%            	|          n=90          	|
  
  Sampling proceeds until these totals, or quotas, are reached. Quota sampling is different from stratified sampling, because in a stratified sample individuals within each stratum are selected at random. Quota sampling achieves a representative age distribution, but it isn't a random sample, because the sampling frame is unknown. Therefore, the sample may not be representative of the population.

3. **Snowball Sampling**: 
  This sampling method is also called chain sampling, chain-referral sampling, referral sampling.  This sampling technique is often used in hidden populations, for studies about casual illegal downloading, cheating on exams, shoplifting, drug use, prostitution, or any other "unacceptable" societal behavior, which researchers might have difficulties to access to samples.  It’s called snowball sampling because (in theory) once you have the ball rolling, it picks up more "snow" along the way and becomes larger and larger.
  Snowball sampling consists of two steps: (1) Identify potential subjects in the population. Often, only one or two subjects can be found initially and (2) Ask those subjects to recruit other people (and then ask those people to recruit). Participants should be made aware that they do not have to provide any other names. These steps are repeated until the needed sample size is found. Ethically, the study participants should not be asked to identify other potential participants. Rather, they should be asked to encourage others to come forward. When individuals are named, it’s sometimes called “cold-calling”, as you are calling out of the blue. Cold-calling is usually reserved for snowball sampling where there’s no risk of potential embarrassment or other ethical dilemmas. Snowball sampling can be a tricky ethical path to navigate. Therefore, you’ll probably be in contact with an institutional review board or another department similarly involved in ethics.

4. **Judgment Sampling**: 
  Judgment sampling, also referred to as judgmental sampling or authoritative sampling, is a non-probability sampling technique where the researcher selects units to be sampled based on his own existing knowledge, or his professional judgment. It can also be referred to as purposive sampling. Results obtained from a judgment sample are subject to some degree of bias, due to the frame and population not being identical. The frame is a list of all the units, items, people, etc., that define the population to be studied.

#### What is the difference between ordinal, interval and ratio variables?

In the 1940s, Stanley Smith Stevens introduced four scales of measurement: nominal, ordinal, interval, and ratio. These are still widely used today as a way to describe the characteristics of a variable. Knowing the scale of measurement for a variable is an important aspect in choosing the right statistical analysis

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Ratio%20Interval%20Ordinal%20Nominal.png?raw=true)

* **Nominal**: A nominal scale describes a variable with categories that do not have a natural order or ranking. You can code nominal variables with numbers if you want, but the order is arbitrary and any calculations, such as computing a mean, median, or standard deviation, would be meaningless. Examples of nominal variables include: genotype, blood type, zip code, gender, race, eye color, political party.
* **Ordinal**: An ordinal scale is one where the order matters but not the difference between values. Examples of ordinal variables include: socio economic status (“low income”,”middle income”,”high income”), education level (“high school”,”BS”,”MS”,”PhD”), income level (“less than 50K”, “50K-100K”, “over 100K”), satisfaction rating (“extremely dislike”, “dislike”, “neutral”, “like”, “extremely like”). Note the differences between adjacent categories do not necessarily have the same meaning. For example, the difference between the two income levels “less than 50K” and “50K-100K” does not have the same meaning as the difference between the two income levels “50K-100K” and “over 100K”.
* **Interval**: An interval scale is one where there is order and the difference between two values is meaningful. Examples of interval variables include: temperature (Farenheit), temperature (Celcius), pH, SAT score (200-800), credit score (300-850).
* **Ratio**: A ratio variable, has all the properties of an interval variable, and also has a clear definition of 0.0. When the variable equals 0.0, there is none of that variable. Examples of ratio variables include: enzyme activity, dose amount, reaction rate, flow rate, concentration, pulse, weight, length, temperature in Kelvin (0.0 Kelvin really does mean “no heat”), survival time. When working with ratio variables, but not interval variables, the ratio of two measurements has a meaningful interpretation. For example, because weight is a ratio variable, a weight of 4 grams is twice as heavy as a weight of 2 grams. However, a temperature of 10 degrees C should not be considered twice as hot as 5 degrees C. If it were, a conflict would be created because 10 degrees C is 50 degrees F and 5 degrees C is 41 degrees F. Clearly, 50 degrees is not twice 41 degrees.  Another example, a pH of 3 is not twice as acidic as a pH of 6, because pH is not a ratio variable.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/types_of_measurements.png?raw=true)

#### What is the general approach to hypothesis testing?

The purpose of hypothesis testing is to determine which of the two hypotheses is correct. The hypothesis testing framework involves comparing a "null hypothesis" to an “alternate hypothesis”. Typically, the null is the status quo, and the alternate is your hypothesis — the one you want to test.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/general_approach_to_hypothesis_testing.png?raw=true)

#### What are the types of hypothesis tests?

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Screen%20Shot%202019-10-07%20at%2009.42.22.png?raw=true)

#### When to use the z-test versus t-test?

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/z-vs-t-distribution-flowchart.jpg?raw=true)

#### How to do one sample test of means?

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC110619-11062019161955-01.png?raw=true)

#### How to do two samples test of means?

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC110619-11062019161955-02.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC110619-11062019161955-03.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC110619-11062019161955-04.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC110619-11062019161955-05.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC110619-11062019161955-06.png?raw=true)

#### How to do paired t-test?

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC110619-11062019161955-07.png?raw=true)

#### How to do one sample test of proportions?

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC110619-11062019161955-08.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC110619-11062019161955-09.png?raw=true)

#### How to do two samples test of proportions?

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC110619-11062019161955-10.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC110619-11062019161955-11.png?raw=true)

NOTE that the reason why we use z-test depends on the assumption of Normal approximation to the Binomial distribution, which requires therefore sufficiently large sample sizes ($n \geq 30$). Given that $n_1 p_1 \ge 10$, $n_1(1-p_1) \ge 10$, $n_2 p_2 \ge 10$, and $n_2(1-p_2) \ge 10$, where the subscript 1 represents the first group and the subscript 2 represents the second group. When this assumption is not met, Pearson chi-square test for large samples (comparing it with $\chi_{1}^{2}$ distribution), Yates's corrected version of Pearson's chi-squared statistics (or Yates's chi-squared test) for intermediate sample sizes, and the Fisher Exact test (using $2 \times 2$ contingency tables) for small samples. 

In order to compare multiple proportions, the Marascuillo Procedure can be used. It is a procedure to simultaneously test the differences of all pairs of proportions when there are several populations under investigation:

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Marascuillo_Procedure.png?raw=true)

#### How to do chi-square test for variance?

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC110619-11062019161955-12.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC110619-11062019161955-13.png?raw=true)

#### How to do F-test for equality of two variances?

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC110619-11062019161955-14.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC110619-11062019161955-15.png?raw=true)

#### What is Chi-square Test for Goodness-of-fit Test?

Chi-Square goodness of fit test is a non-parametric test that is used to find out how the observed value of a given phenomena is significantly different from the expected value. 

To apply the Chi-Square Test for any distribution to any data set, let your null hypothesis be that your data is sampled from a distribution and apply the Chi-Square Goodness of Fit Test.

Let $x_{1}, x_{2}, \dots, x_{n}$ be the observed values of a random variable $x$. Assume, that the observed values $x_{1}, x_{2}, \dots, x_{n}$ are i.i.d. 

* Let's first categorize the observations ($n$) into $k$ categories.
* Calculate the frequencies $O_{i}, i  = 1, 2, ..., k$, where $O_{i}$ is the observed frequency of the category $i$. Note that $\sum_{i=1}^{k} O_{i} = n$.
* you will need to calculate the expected values under the given distribution for every data point. Let $p_{i}$ be the probability, that under null hypothesis, the random variable $x$ belongs to the category $i$. Calculate the expected frequencies $E_{i} = n \times p_{i}$ of the observations in category $i$. Note that $\sum_{i=1}^{k} p_{i} = 1$.

The null hypothesis is a particular claim concerning how the data is distributed. The null and alternative hypotheses for each chi square test can be stated as

$$
\begin{split}
H_{0} : O_{i} = E_{i}\\
H_{1} : O_{i} \neq E_{i}
\end{split}
$$

If the claim made in the null hypothesis is true, the observed and the expected values are close to each other and $O_{i} − E_{i}$ is small for each category. When the observed data does not conform to what has been expected on the basis of the null hypothesis, the difference between the observed and expected values, $O_{i} − E_{i}$, is large. 

After computing expected values and setting the hypotheses and we then use the formula:

$$
\chi^{2} = \sum_{i=1}^{k} \frac{(O_{i} - E_{i})^{2}}{E_{i}}
$$

to find the chi-square statistic where $O_{i}$ is the observed value and $E_{i}$ is the expected value. 

Compare this test statistic to the critical chi-square value from a chi-square table, given your degrees of freedom and desired alpha level. 

In Chi-Square goodness of fit test, the degree of freedom depends on the distribution of the sample. It can be shown that, if the population follows the hypothesized distribution, test statistics $\chi^{2}$ has approximately a chi-square distribution with $k-p-1$ degrees of freedom, where $p$ represents the number of parameters of the hypothesized distribution estimated by sample statistics. This approximation improves as $n$ increases. For instance, when checking a binomial distribution, $p=1$, and when checking a three-co-variate Weibull distribution, $p=3$, and when checking a normal distribution (where the parameters are mean and standard deviation), $p=2$, and when checking a Poisson distribution (where the parameter is the expected value), $p=1$. Thus, there will be $k-p-1$ degrees of freedom, where $k$ is the number of categories.

If your chi-square statistic is larger than the table value, you may conclude your data is not following the given distribution.

Note that if your variable is continuous, you will need to bin the data before using the chi-square test for normality.

Note: If the value of the test statistic is large, the sample frequencies differ greatly from the expected value, and it is clear that the null hypothesis should be rejected. However, if the value is very small, then the sample frequencies differ less than expected. This is called overfitting.

Alternatives to Chi-Square Test for Normality include:
* The Kolmogorov-Smirnov (K-S) test
* The Lilliefors corrected K-S test
* The Shapiro-Wilk test
* The Anderson-Darling test
* The Cramer-von Mises test
* The D’Agostino-Pearson omnibus test
* The Jarque-Bera test

All of these tests have different strength and weaknesses, but the Shapiro Wilk test may have the best power for any given significance.

```python
# Assumptions: Observations in each sample are independent and identically distributed (iid).

# Interpretation
# H0: the sample has a Gaussian distribution.
# H1: the sample does not have a Gaussian distribution
    
# Example of the Kolmogorov-Smirnov Normality Test
from scipy.stats import kstest 
data = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]

stat, p = kstest(data, 'norm')
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably Gaussian')
else:
    print('Probably not Gaussian')
    
# stat=0.328, p=0.186
# Probably Gaussian

# Example of the Shapiro-Wilk Normality Test
from scipy.stats import shapiro
data = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]

stat, p = shapiro(data)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably Gaussian')
else:
    print('Probably not Gaussian')
#stat=0.895, p=0.193
#Probably Gaussian

# Example of the D'Agostino's K^2 Normality Test
from scipy.stats import normaltest
data = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
stat, p = normaltest(data)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably Gaussian')
else:
    print('Probably not Gaussian')
    
# stat=3.392, p=0.183
# Probably Gaussian

# Example of the Anderson-Darling Normality Test
from scipy.stats import anderson
data = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
result = anderson(data)
print('stat=%.3f' % (result.statistic))
for i in range(len(result.critical_values)):
    sl, cv = result.significance_level[i], result.critical_values[i]
    if result.statistic < cv:
        print('Probably Gaussian at the %.1f%% level' % (sl))
    else:
        print('Probably not Gaussian at the %.1f%% level' % (sl))
        
# stat=0.424
# Probably Gaussian at the 15.0% level
# Probably Gaussian at the 10.0% level
# Probably Gaussian at the 5.0% level
# Probably Gaussian at the 2.5% level
# Probably Gaussian at the 1.0% level
```

#### What is Chi-square Test for Test of Independence?

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/chi_square_gof1.jpeg?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/chi_square_gof2.jpeg?raw=true)

```python
import scipy
from scipy.stats import binom

n=3
p=1/6
b = scipy.stats.binom(n,p)

print('P(roll 0 sixes) = P(X = 0) = {:.2f}'.format(b.pmf(0)))
print('P(roll 1 sixes) = P(X = 1) = {:.2f}'.format(b.pmf(1)))
print('P(roll 2 sixes) = P(X = 2) = {:.2f}'.format(b.pmf(2)))
print('P(roll 3 sixes) = P(X = 3) = {:.2f}'.format(b.pmf(3)))

# P(roll 0 sixes) = P(X = 0) = 0.58
# P(roll 1 sixes) = P(X = 1) = 0.35
# P(roll 2 sixes) = P(X = 2) = 0.07
# P(roll 3 sixes) = P(X = 3) = 0.00
```

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/chi_square_gof3.jpeg?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/chi_square_gof4.jpeg?raw=true)

#### What is Chi-square Test for Test of Association?

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/chi_square_test_of_independence_1.jpeg?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/chi_square_test_of_independence_2.jpeg?raw=true)

```python
import scipy
from scipy.stats import chi2

#Percent point function (inverse of cdf — percentiles)
value = chi2.ppf(q = 0.95, df= 3)
print(value)
#7.814727903251179

# confirm with cdf
p = chi2.cdf(value, df = 3)
print(p)
#0.95

#OR

p_value = chi2.sf(8.006, df= 3)
#0.04588786639310048
#p < alpha so we reject the null hypothesis
```

#### What is the post-hoc pairwise comparison of chi-squared test?

The chi-squared test assesses a global question whether relation between two variables is independent or associated. If there are three or more levels in either variable, a post-hoc pairwise comparison is required to compare the levels of each other. 

If this omnibus null hypothesis is rejected about whether relation between two variables is independent or associated, it may be desirable to perform post-hoc analyses to determine which groups differ. This can be accomplished by testing for differences in $2 \times 2$ subtables created by considering only two columns at a time. The only form of multiple-comparisons adjustment available for this analysis is the Bonferroni method, in which a total of $\frac{C(C-1)}{2}$ possible groupwise comparisons can be made. Most statistical software packages will not do this analysis automatically; the user must manually construct $2 \times 2$ tables for each of the various pairwise comparisons and adjust the $\alpha$ level of each test to control the overall type I error rate.

#### What is Fisher's Exact test?

z- and t-tests concern quantitative data (or proportions in the case of z), chi-squared tests are appropriate for qualitative data. Again, the assumption is that observations are independent of one another. In this case, you are not seeking a particular relationship. Your null hypothesis is that no relationship (or the association) exists between variable one and variable two. Your alternative hypothesis is that a relationship does exist. This doesn't give you specifics as to how this relationship exists (i.e. In which direction does the relationship go) but it will provide evidence that a relationship does (or does not) exist between your independent variable and your groups.

However, one drawback to the chi-squared test is that it is asymptotic. This means that the p-value is accurate for very large sample sizes. However, if your sample sizes are small (some people argue total observations in a contingency table is less than 1,000), then the p-value may not be quite accurate. As such, Fisher's exact test allows you to exactly calculate the p-value of your data and not rely on approximations that will be poor if your sample sizes are small because the usual rule of thumb for deciding whether the chi-squared approximation is good enough is that the chi-squared test is not suitable when the expected values in any of the cells of a contingency table are below 5, or below 10 when there is only one degree of freedom (this rule is now known to be overly conservative). In fact, for small, sparse, or unbalanced data, the exact and asymptotic p-values can be quite different and may lead to opposite conclusions concerning the hypothesis of interest.

Fisher's exact test is a statistical significance test used in the analysis of contingency tables. Although in practice it is employed when sample sizes are small, it is valid for all sample sizes. 

For hand calculations, the test is only feasible in the case of a $2 \times 2$ contingency table. However the principle of the test can be extended to the general case of an $m \times n$ table. However, the only problem with applying Fisher's exact test to tables larger than $2 \times 2$ is that the calculations become much more difficult to do. Therefore, some statistical packages provide a calculation (sometimes using a Monte Carlo method to obtain an approximation) for the more general case.

Fisher's exact test is based on the hypergeometric distribution. For example, given a $2 \times 2$ cross-table:

|                  	|  Men  	| Women 	|      Row Total     	|
|:----------------:	|:-----:	|:-----:	|:------------------:	|
|     Studying     	|   a   	|   b   	|        a + b       	|
|   Non-studying   	|   c   	|   d   	|        c + d       	|
|   Column Total   	| a + c 	| b + d 	| a + b + c + d (=n) 	|

We compute the p-value as:

$$
p = \frac{(a+b)!(c+d)!(a+c)!(b+d)!}{a!b!c!d!n!}
$$

#### What does statistical interaction mean?

Statistical interaction means the effect of one independent variable(s) on the dependent variable depends on the level of another independent variable(s). Conversely, additivity (i.e., no interaction) means that the effect of one independent variable(s) on the dependent variable does NOT depend on the value of another independent variable(s).

In order to find an interaction, you must have a factorial design, in which the two (or more) independent variables are "crossed" with one another so that there are observations at every combination of levels of the two independent variables. EX: stress level and practice to memorize words: together they may have a lower performance.

#### Explain generalized linear model.
The generalized linear model (GLM) is a flexible generalization of ordinary linear regression that allows for response variables that have error distribution models other than a normal distribution. The GLM generalizes linear regression by allowing the linear model to be related to the response variable via a link function and by allowing the magnitude of the variance of each measurement to be a function of its predicted value.

Ordinary linear regression predicts the expected value of a given unknown quantity (the response variable, a random variable) as a linear combination of a set of observed values (predictors).

In a generalized linear model (GLM), each outcome Y of the dependent variables is assumed to be generated from a particular distribution in an exponential family, a large class of probability distributions that includes the normal, binomial, Poisson and gamma distributions.

The GLM consists of three elements:

1. An exponential family of probability distributions.
2. A linear predictor $\eta = \mathbf{X}\beta$. 
3. A link function $g$ such that $E(Y \mid X) = \mu = g^{-1}(\eta)$, i.e., $g(\mu) = \eta = \mathbf{X}\beta$. The link function provides the relationship between the linear predictor and the mean of the distribution function. For the most common distributions, the mean $\mu$  is one of the parameters in the standard form of the distribution's density function.

The unknown parameters, $\beta$, are typically estimated with maximum likelihood, maximum quasi-likelihood, or Bayesian techniques.

Example:

1. Normal distribution, we have identity, link function:

  $$
g(\mu) = \mu = \mathbf{X}\beta
$$

2. Poisson distribution, we have log link function:

  $$
g(\mu) = ln(\mu) = \mathbf{X}\beta \Rightarrow \mu = exp\left(\mathbf{X}\beta \right)
$$

3. For Bernoulli, Binomial, Multinoulli (categorical) and Multinominal distributions, we have logit link function:

  $$
g(\mu) = logit(\mu) = ln\left(\frac{\mu}{1-\mu}\right) = \mathbf{X}\beta \Rightarrow \mu =  \frac{exp(\mathbf{X}\beta)}{1+exp(\mathbf{X}\beta)} = \frac{1}{1+exp(-\mathbf{X}\beta)}
$$

#### What does link function do?

Linear regression assumes that the response variable is normally distributed. Generalized linear models can have response variables with distributions other than the Normal distribution– they may even be categorical rather than continuous. Thus they may not range from  $- \infty$ to $+ \infty$. Besides, relationship between the response and explanatory variables need not be of the simple linear form.

Generalized linear models include a link function that relates the expected value of the response to the linear predictors in the model. A link function transforms the probabilities of the levels of a categorical response variable to a continuous scale that is unbounded. Once the transformation is complete, the relationship between the predictors and the response can be modeled with linear regression. For example, a binary response variable can have two unique values. Conversion of these values to probabilities makes the response variable range from $0$ to $1$. When you apply an appropriate link function to the probabilities, the numbers that result range from $- \infty$ to $+ \infty$. The general form of the link function follows:

It links the mean of the dependent variable $Y_{i}$, which is $E(Y_{i}) = \mu_{i}$ to the linear term $X_{i}^{T} \beta$ in such a way that the range of the non-linearly transformed mean $g(\mu_{i})$ ranges from $- \infty$ to $+ \infty$. Thus you can actually form a linear equation $g(\mu_{i}) = X_{i}^{T} \beta$ where $g(\cdot)$ is the link function, $\mu_{i}$ is the mean response of the $i$th row, $X_{i}$ is the vector of predictor variables for the $i$th row and use an iteratively reweighted least squares method for maximum likelihood estimation of the model parameters.

#### Given X and Y are independent variables with normal distributions, what is the mean and variance of the distribution of 2X - Y when the corresponding distributions are X follows N (3, 4) and Y follows N(1, 4)?

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC111819-11182019114711-1.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC111819-11182019114711-2.png?raw=true)

#### A system is guaranteed to fail 10% of a time within any given hour, what's the failure rate after two hours ? after n-hours?

Let $P_{fail} = 0.1$ be the probability that the system fails in any given hour. Let $P_{no fail} = 1 - P_{fail} = 0.9$, be the probability that the system does not fail in any given hour. The joint probability of two independent events is the product of the probability of each event (that is, $P(A \cap B) = P(A) \times P(B)$). Therefore, The chance of it failing in 2 hours is $P_{fail}^{2} = 0.1^{2}$. The chance of it NOT failing in $n$ hours is $P_{no fail}^{n} = 0.9^{n}$. The chance of it failing in $n$ hours is `1 - (chance of not failing in n hours)`, which is $ 1 - 0.9^{n}$.

#### What is the relation between variance and sum of squares? 

Before we talk above Analysis of Variance, let's see the relationship between variance and sum of squares.

The unbiased estimator of variance of an observed data set can be estimated using the following relationship:

$$
s^{2} = \frac{\sum_{i=1}^{n} (y_{i} - \hat{y})^{2}}{n-1}
$$

where $s$ is the standard deviation, $y_{i}$ is the $i$th observation, $n$ is the number of observations and $\hat{y}$ is the mean of the $n$ observations. 

The quantity in the numerator of the previous equation is called the sum of squares. It is the sum of the squares of the deviations of all the observations, $y_{i}$, from their mean, $\hat{y}$. In the context of ANOVA, this quantity is called the total sum of squares (abbreviated SST) because it relates to the total variance of the observations. Thus:

$$
\text{S.S.}_{T} = \sum_{i=1}^{n} (y_{i} - \hat{y})^{2}
$$

The denominator in the relationship of the sample variance is the number of degrees of freedom associated with the sample variance. Therefore, the number of degrees of freedom associated with $\text{S.S.}_{T}$, dof(SST), is $(n-1)$. The sample variance is also referred to as a mean square because it is obtained by dividing the sum of squares by the respective degrees of freedom. Therefore, the total mean square (abbreviated MST) is:

$$
MST = \frac{\text{S.S.}_{T}}{\text{dof}(\text{S.S.}_{T})} = \frac{\text{S.S.}_{T}}{n-1}
$$

#### What is analysis of variance (ANOVA)?

Commonly, ANOVAs are used in three ways: one-way ANOVA, two-way ANOVA, and N-way ANOVA (MANOVA).

**One-Way ANOVA**
  
One-Way ANOVA has only one independent variable (a factor). The main purpose of a one-way ANOVA is to test if two or more groups differ from each other significantly. For example, One-way ANOVA has one continuous response variable (e.g. Test Score) compared by three or more levels of a factor variable (e.g. Level of Education).

 A one way ANOVA will tell you that at least two groups were different from each other. But it won’t tell you which groups were different. If your test returns a significant f-statistic, you may need to run an ad hoc test to tell you exactly which groups had a difference in means.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/one-way.ANOVA.png?raw=true)
Let's give an example how to do one-way ANOVA by hand:
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/ANOVA_example1.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/ANOVA_example2.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/ANOVA_example3.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/ANOVA_example4.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/ANOVA_example5.png?raw=true)
  
**Two-Way ANOVA**

A Two Way ANOVA is an extension of the One Way ANOVA. With a One Way, you have one independent variable affecting a dependent variable. It refers to an ANOVA using two independent variables.  Expanding the example above, Two-way ANOVA has one continuous response variable (e.g. Test Score) compared by more than one factor variable (e.g. Level of Education and Zodiac Sign). Two-way ANOVA can also be used to examine the interaction between the two independent variables. Interactions indicate that differences are not uniform across all categories of the independent variables. Two-way ANOVAs are also called factorial ANOVAs.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/two-way+anova.png?raw=true)

In two-way ANOVA, the factors can be crossed and nested.

With Two-Way Crossed ANOVA, we can estimate the effect of each factor (Main Effects) as well as any interaction between the factors. The main effect is similar to a One Way ANOVA: each factor’s effect is considered separately. With the interaction effect, all factors are considered at the same time. 

$$
y_{ijk} = m + a_{i} + b_{j} + (ab)_{ij} + \epsilon_{ijk}
$$

This equation just says that the $k$th data value for the $j$th level of Factor B and the $i$th level of Factor A is the sum of five components: the common value (grand mean), the level effect for Factor A, the level effect for Factor B, the interaction effect, and the residual. Note that (ab) does not mean multiplication; rather that there is interaction between the two factors.

Thus, there are three different hypotheses to be tested in two-way ANOVA:

$$
\begin{split}
H_{01} &: \text{The mean of the test scores is the same for all the educational levels.}\\
H_{02} &: \text{The mean of the test scores is the same for all the zodiac signs.}
\end{split}
$$

For multiple observations in cells, you would also be testing a third hypothesis:

$$
\begin{split}
H_{03}&: \text{The factors, education levels and zodiac signs, are independent}\\ 
& (\text{the interaction effect between education levels and zodiac signs does not exist})
\end{split}
$$

An F-statistic is computed for each hypothesis you are testing.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/two_way_ANOVA_table.png?raw=true)

Sometimes, constraints prevent us from crossing every level of one factor with every level of the other factor. In these cases we are forced into what is known as a nested layout. We say we have a nested layout when fewer than all levels of one factor occur within each level of the other factor. An example of this might be if we want to study the effects of different machines and different operators on some output characteristic, but we can't have the operators change the machines they run. In this case, each operator is not crossed with each machine but rather only runs one machine.

If Factor B is nested within Factor A, then a level of Factor B can only occur within one level of Factor A and there can be no interaction. This gives the following model:

$$
y_{ijk} = m + a_{i} + b_{j(i)} + \epsilon_{ijk}
$$

This equation indicates that each data value is the sum of a common value (grand mean), the level effect for Factor A, the level effect of Factor B nested within Factor A, and the residual.

**N-Way ANOVA**
  
A researcher can also use more than two independent variables, and this is an n-way ANOVA (with n being the number of independent variables you have).  For example, potential differences in IQ scores can be examined by Country, Gender, Age group, Ethnicity, etc, simultaneously.

**Multivariate ANOVA (MANOVA)**

Multivariate analysis of variance (MANOVA) is simply an ANOVA with several dependent variables. If there is one independent variable and multiple dependent variables, it is called one-way MANOVA. For example, One-way MANOVA compares two or more continuous response variables (e.g. Test Score and Annual Income) by a single factor variable (e.g. Level of Education).

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/ONE-WAY+MANOVA+figure.001.png?raw=true)

If there is two or more dependent variables as well as two or more independent variables, it is called factorial MANOVA. For example, Two-way MANOVA compares two or more continuous response variables (e.g. Test Score and Annual Income) by two or more factor variables (e.g. Level of Education and Zodiac Sign).

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/two-way+manova.png?raw=true)

**NOTE** The Kruskal-Wallis H test is a non-parametric test that is used in place of a one-way ANOVA. Besides, ANOVA can also be used for repeated measures. The Friedman test is a non-parametric alternative to ANOVA with repeated measures. No normality assumption is required. 
  
**Repeated measures ANOVA**:
  
It is the equivalent of the one-way ANOVA, but for related, not independent groups, and is the extension of the dependent t-test. A repeated measures ANOVA is also referred to as a within-subjects ANOVA or ANOVA for correlated samples.

**Assumptions for ANOVA** 

* The population from which samples are drawn should be normally distributed (can be tested using histograms, the values of skewness and kurtosis, or using tests such as Shapiro-Wilk or Kolmogorov-Smirnov). If it is violated: you can (1) transform your data using various algorithms so that the shape of your distributions become normally distributed or (2) choose the nonparametric Kruskal-Wallis H Test which does not require the assumption of normality.
* Observation must be independent (can be determined from the design of the study). A lack of independence of cases has been stated as the most serious assumption to fail. Often, there is little you can do that offers a good solution to this problem.
* Population variances must be equal. (The assumption of homogeneity of variance must be tested before ANOVA, such as Hartley's Fmax Test, Cochran's Test, Levene's Test, Fligner Killeen Test and Barlett's test). There are two tests that you can run that are applicable when the assumption of homogeneity of variances has been violated: (1) Welch's Test (especially with unequal sample sizes) or (2) Brown and Forsythe test. Alternatively, you could run a Kruskal-Wallis H Test, non-parametric version of ANOVA. For most situations it has been shown that the Welch test is best. 
* Groups must have equal sample sizes.
* Factor effects are additive

**So what if you find statistical significance?  Multiple comparison tests**

When you conduct an ANOVA, you are attempting to determine if there is a statistically significant difference among the groups.  If you find that there is a difference, you will then need to examine where the group differences lay. At this point you could run post-hoc tests for double comparisons, which are t-tests examining mean differences between the groups. There are several multiple comparison tests that can be conducted that will control for Type I error rate (adjusting $\alpha$'s). The Tukey post hoc test is generally the preferred test for conducting post hoc tests on a one-way ANOVA, but there are many others including the LSD, Bonferroni, Sidak's, Student-Newman-Keuls test (or short S-N-K), Scheffé's method (uses F-test), and Dunnet test.

#### What is analysis of covariance (ANCOVA)?

ANCOVA is an extension of the ANOVA, the researcher can still can assess main effects and interactions to answer their research hypotheses.  The difference between an ANCOVA and an ANOVA is that an ANCOVA model includes a “covariate” that is correlated with the dependent variable and means on the dependent variable are adjusted due to the effects the covariate has on it. This technique answers the question: Are mean differences or interactive effects likely to have occurred by chance after scores have been adjusted on the dependent variable because of the effect of the covariate?

The task  is to remove the extraneous variation from the dependent variable. For example, ANCOVA compares a continuous response variable (e.g. Test Score) by levels of a factor variable (e.g. Level of Education), controlling for a continuous covariate (e.g. Number of Hours Spent Studying). 

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/ancova.png?raw=true)

MANCOVA is a statistical technique that is the extension of ANCOVA. MANCOVA feature two or more dependent variables and one or more more covariates are added to the mix. MANCOVA removes the effects of one or more covariates from your model. For example, MANCOVA compares two or more continuous response variables (e.g. Test Scores and Annual Income) by levels of a factor variable (e.g. Level of Education), controlling for a covariate (e.g. Number of Hours Spent Studying).

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/mancova.png?raw=true)


MANCOVA, MANOVA, ANOVA, ANCOVA: it can all get a little confusing to remember which is which. However, all of the tests can be thought of as variants of the MANCOVA, if you remember that the “M” in MANCOVA stands for Multiple and the “C” stands for Covariates.

* ANOVA: a MANCOVA without multiple dependent variables and covariates (hence the missing M and C).
* ANCOVA: a MANCOVA without multiple dependent variables (hence the missing M).
* MANOVA: A MANCOVA without covariates (hence the missing C).

#### What is Homogeneity of Variances? When and how should we check it?

One-way ANOVA assumes that the data come from populations that are Gaussian and have equal variances. Similarly, the unpaired t test assumes that the data are sampled from Gaussian populations with equal variances.

So, it is an assumption underlying both t tests and F tests (analyses of variance, ANOVAs). In correlations and regressions, the term “homogeneity of variance in arrays,” also called “homoskedasticity,” refers to the assumption that, within the population, the variance of Y for each value of X is constant.

The F test presented in Two Sample Hypothesis Testing of Variances can be used to determine whether the variances of two populations are equal. For three or more variables the following statistical tests for homogeneity of variances are commonly used: Hartley's Fmax Test, Cochran's Test, Levene's Test, Fligner Killeen Test and Barlett's test. Several of these assessments have been found to be too sensitive to non-normality and are not frequently used.  Of these tests, the most common assessment for homogeneity of variance is Levene’s test.  The Levene’s test uses an F-test to test the null hypothesis that the variance is equal across groups. 

The following null and alternative hypotheses are used for all of these tests:

$$
\begin{split}
H_{0} &: \sigma_1^2 = \sigma_2^2 = \dots = \sigma_k^2\\
H1 &:\text{ Not all variances are equal (i.e. $\sigma_i^2 \neq \sigma_j^2$ for some $i, j$)}
\end{split}
$$

If these tests result in a small p-value, you have evidence that the variance (and thus standard deviations) of the groups differ significantly.

This gives you strong evidence that the groups are not selected from identical populations. You haven't yet tested whether the means are distinct, but you already know that the variances are different. This may be a good stopping point. You have strong evidence that the populations the data are sampled from are not identical. Often the best approach is to transform the data. Often transforming to logarithms or reciprocals does the trick, restoring equal variance. If you want one-way ANOVA, the standard methods for dealing with heterogeneity of variance are the Welch or Brown-Forsythe F-tests. Since nonparametric tests do not assume Gaussian distributions, you can also switch to using the nonparametric Kruskal-Wallis ANOVA  when comparing multiple groups or the Mann-Whitney test when comparing two groups. 

Similar to the assumption of normality, when we test for violations of constant variance we should not rely on only one approach to assess our data. Rather, we should understand the variance visually and by using multiple testing procedures to come to our conclusion of whether or not homogeneity of variance holds for our data.
  
#### When is a biased estimator preferable to unbiased one?

Although a biased estimator does not have a good alignment of its expected value with its true value, there are many practical instances when a biased estimator can be useful.

Often it is the case that we are interested in minimizing the mean squared error, which can be decomposed into variance + bias squared. This is an extremely fundamental idea in machine learning, and statistics in general. Frequently we see that a small increase in bias can come with a large enough reduction in variance that the overall MSE decreases.

A standard example is ridge regression. We have $\hat{\theta_{ridge}} = \left(\mathbf{X}^{T} \cdot \mathbf{X} +\lambda I\right)^{-1} \cdot \mathbf{X}^{T} y$ which is biased; but if $\mathbf{X}$ is ill conditioned, we will have singular $\mathbf{X}^{T} \cdot \mathbf{X}$, then $Var(\hat{\theta_{OLS}}) = \sigma^{2} \left(\mathbf{X}^{T} \cdot \mathbf{X} \right)^{-1}$ may be huge whereas $Var (\hat{\theta_{ridge}})$ can be much more modest. In these cases, variance of biased estimator will be smaller than the unbiased one.

Another example is the kNN classifier. Think about $k = 1$: we assign a new point to its nearest neighbor. If we have a ton of data and only a few variables we can probably recover the true decision boundary and our classifier is unbiased; but for any realistic case, it is likely that $k = 1$ will be far too flexible (i.e. have too much variance) and so the small bias is not worth it (i.e. the MSE is larger than more biased but less variable classifiers).

Finally, here's a picture. Suppose that these are the sampling distributions of two estimators and we are trying to estimate 0. The flatter one is unbiased, but also much more variable. Overall I think I'd prefer to use the biased one, because even though on average we won't be correct, for any single instance of that estimator we'll be closer.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/BigyK.png?raw=true)

#### What is the difference between t-test and linear regression?

The t-test and the test of the regression coefficient (the slope) are exactly the same. The t-test does not allow to include other variables, but the regression does.

Although these methods have, historically, developed along separate tracks, most statisticians would nowadays consider them as special cases of the General Linear Model (GLM). The GLM-framework incorporates regression analyses, ANOVAs, and t-tests, but also many other techniques, such as ANCOVA, MANOVA, and MANCOVA. The key part to understand is that the aforementioned models can all be written as special cases of GLM, as a regression equation (perhaps with slightly differing interpretations than their traditional forms):

* **Regression**:

  $$
  \begin{split}
  Y=\beta_0 + \beta_1X_{\text{(continuous)}} + \varepsilon  \\
  \text{where }\varepsilon\sim\mathcal N(0, \sigma^2)
  \end{split}
  $$
 
* **t-test**:

  $$
  Y=\beta_0 + \beta_1X_{\text{(dummy code)}} + \varepsilon  \\
  \text{where }\varepsilon\sim\mathcal N(0, \sigma^2)
  $$
  
* **ANOVA**:

  $$
  Y=\beta_0 + \beta_1X_{\text{(dummy code)}} + \varepsilon  \\
  \text{where }\varepsilon\sim\mathcal N(0, \sigma^2)
  $$

The prototypical regression is conceptualized with $x$ as a continuous variable. However, the only assumption that is actually made about $x$ is that it is a vector of known constants. However, the only assumption that is actually made about $X$ is that it is a vector of known constants. It could be a continuous variable, but it could also be a dummy code (i.e., a vector of $0$'s and $1$'s that indicates whether an observation is a member of an indicated group--e.g., a treatment group). Thus, in the second equation, $X$ could be such a dummy code (or contrast coding where one group is coded as -1 and other one is coded), and the p-value would be the same as that from a t-test in its more traditional form. For example:

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/1_21VcolLbpikvFuMvZyjSOQ.png?raw=true)

The meaning of the betas would differ here, though. In this case, $\beta_{0}$ would be the mean of the control group (for which the entries in the dummy variable would be $0$'s), and $\beta_{1}$ would be the difference between the mean of the treatment group and the mean of the control group.

There is again one special case where adding a control variable to a regression model has an equivalent (direct) t-test:
say you have $n$ subjects, the response is is measured from each before some treatment, and then it is measured again after the treatment. Here you have two factors: time (before/after) and subject ($1, 2, 3, \dots, n$). The test of the time-slope in a two-factorial regression model (including dummy-coded time and dummy-coded subject ID) is identical to the test of the (within-subject) pairwise differences (a paired t-test).

Let's see an example in Python:

```python
## Import the packages
import numpy as np
from scipy import stats
import statsmodels.api as sm

## Define 2 random distributions
#Sample Size
N = 10
#Gaussian distributed data with mean = 2 and var = 1
a = np.random.randn(N) + 2
# array([1.54521498, 0.07929609, 1.50214032, 0.93720253, 1.21987251,
#        3.39567805, 2.3685942 , 3.62647803, 2.03051962, 1.61038559])

#Gaussian distributed data with with mean = 0 and var = 1
b = np.random.randn(N)
# array([ 0.33036931,  0.66242071, -1.63292861,  0.43102331, -1.05377199,
#         0.24495993,  0.21372323,  0.96371525,  0.67941075,  1.72399428])

## Calculate the Standard Deviation
#Calculate the variance to get the standard deviation

#For unbiased max likelihood estimate we have to divide the var by N-1, 
#and therefore the parameter ddof = 1
var_a = a.var(ddof=1)
var_b = b.var(ddof=1)

#std deviation
s = np.sqrt((var_a + var_b)/2)
#1.022035824071384

## Calculate the t-statistics
t = (a.mean() - b.mean())/(s*np.sqrt(2/N))

## Compare with the critical t-value
#Degrees of freedom
df = 2*N - 2
#18

#p-value after comparison with the t 
p = 1 - stats.t.cdf(t,df=df)

print("t = " + str(t)) #t = 3.446413857090997
print("p = " + str(2*p)) #p = 0.0028795519402942116
#You can see that after comparing the t statistic with the critical t value (computed internally) 
#we get a good p value of 0.0028795519402942116 and thus we reject the null hypothesis
#and thus it proves that the mean of the two distributions are different and statistically significant.

## Cross Checking with the internal scipy function
t2, p2 = stats.ttest_ind(a,b)
print("t = " + str(t2))
print("p = " + str(p2))
# t = 3.446413857090997
# p = 0.002879551940294324

# Linear Regression
#So we used dummy encoding, 0s represents sample a and 1s represent sample b.

X = np.concatenate((np.zeros(shape = a.shape), np.ones(shape = b.shape)), axis = 0).reshape(-1,1)
Y = np.concatenate((a, b), axis = 0).reshape(-1,1)

X = sm.add_constant(X) # adding a constant B0

X
# array([[1., 0.],
#        [1., 0.],
#        [1., 0.],
#        [1., 0.],
#        [1., 0.],
#        [1., 0.],
#        [1., 0.],
#        [1., 0.],
#        [1., 0.],
#        [1., 0.],
#        [1., 1.],
#        [1., 1.],
#        [1., 1.],
#        [1., 1.],
#        [1., 1.],
#        [1., 1.],
#        [1., 1.],
#        [1., 1.],
#        [1., 1.],
#        [1., 1.]])

Y
# array([[ 1.54521498],
#        [ 0.07929609],
#        [ 1.50214032],
#        [ 0.93720253],
#        [ 1.21987251],
#        [ 3.39567805],
#        [ 2.3685942 ],
#        [ 3.62647803],
#        [ 2.03051962],
#        [ 1.61038559],
#        [ 0.33036931],
#        [ 0.66242071],
#        [-1.63292861],
#        [ 0.43102331],
#        [-1.05377199],
#        [ 0.24495993],
#        [ 0.21372323],
#        [ 0.96371525],
#        [ 0.67941075],
#        [ 1.72399428]])

model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 

print_model = model.summary()
print(print_model)
```

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Screen%20Shot%202020-05-19%20at%2008.31.32.png?raw=true)

```python
#so const (B0) in regression represents mean of the sample a (or rather the mean of the group being compared to), 
#while B1 represents the difference between the means of sample a and sample b
#In the t-test above we do (a.mean() - b.mean()), comparing sample a's mean with sample b's mean,
# in other words, we are comparing, sample a to sample b.
#but in regression, we compare sample b to sample a.
#This is the reason we have negative coef and t-value for coef.

print(a.mean())
#1.8315381920347527

print(a.mean()- b.mean())
#1.5752465765649406

#Confidence interval for x1 is [ -2.536, -0.615]. So, it does not contain 0.
#So, we can say that there is a statistically significant difference between means of sample a and sample b.
```

#### What are qq-plots and pp-plots?

We use these plots to visually compare data coming from different datasets (distributions). The possible scenarios involve comparing:

* two empirical sets
* one empirical and one theoretical set
* two theoretical sets (exponential, normal, etc.)

These plots can be used to determine how well a theoretical distribution models a data distribution. But they are mostly used to figure out whether normality assumptions for some statistical tests are satisfied. Therefore, we compare one empirical and one theoretical sets (for the case of test of normality, it is Gaussian distribution).

A pp-plot compares the empirical cumulative distribution function of a data set with a specified theoretical cumulative distribution function $F(\cdot)$. pp-plots require fully specified distributions, which requires the location and scale parameters of $F(\cdot)$ (CDF) to evaluate the cdf at the ordered data values. For example, if we are using Gaussian as the theoretical distribution we should specify the location and scale parameters.

To construct a pp-plot, the $n$ non-missing values are first sorted in increasing order:

$$
x_{1} \leq x_{2} \leq x_{3} \leq \cdots \leq x_{n}
$$

Then, the empirical cumulative distribution function, denoted by $F_{n}(x)$, is defined as the proportion of non-missing observations less than or equal to $x$, so that $F_{n}(x_{i}) = \frac{i}{n}$. Then the $i$th ordered value $x_{i}$ is represented on the plot by the point whose x-coordinate is $F(x_{i})$ and whose y-coordinate is $\frac{i}{n}$.

A qq-plot compares the quantiles of a data distribution with the quantiles of a standardized theoretical distribution from a specified family of distributions. The construction of a qq-plot does not require that the location or scale parameters of $F(\cdot)$ (CDF) be specified. The theoretical quantiles are computed from a standard distribution within the specified family. A linear point pattern indicates that the specified family reasonably describes the data distribution, and the location and scale parameters can be estimated visually as the intercept and slope of the linear pattern. 

Plotting the first data set's quantiles along the x-axis and plotting the second data set's quantiles along the y-axis is how the plot is constructed. In practice, many data sets are compared to the normal distribution. The normal distribution is the base distribution and its quantiles are plotted along the x-axis as the "Theoretical Quantiles" while the sample quantiles are plotted along the y-axis as the "Sample Quantiles". Here's what QQ-plots look like (for particular choices of distribution) on average:

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/qq_plot_examples.png?raw=true)

#### What to do when normality assumption is violated?

There are three other major ways to approach violations of normality.

* You can still try to transform your data using Box-cox transformation.
* Have enough data and invoke the central limit theorem. This is the simplest. If you have sufficient data and you expect that the variance of your errors (you can use residuals as a proxy) is finite, then you invoke the central limit theorem and do nothing. Your $\hat{\beta}$ will be approximately normally distributed, which is all you need to construct confidence intervals and do hypothesis tests.
* Bootstrapping. This is a non-parametric technique involving resampling in order to obtain statistics about one’s data and construct confidence intervals.
* Use a generalized linear model. Generalized linear models (GLMs) generalize linear regression to the setting of non-Gaussian errors. Thus if you think that your responses still come from some exponential family distribution, you can look into GLMs.

#### How to see non-Spherical disturbances?

A Gaussian distribution is completely determined by its covariance matrix and its mean (a location in space). The covariance matrix of a Gaussian distribution determines the directions and lengths of the axes of its density contours, all of which are ellipsoids.

Our basic linear regression model is:

$$
\mathbf{Y} = \mathbf{X}\beta + \mathbf{\varepsilon},\,\,\,\,\, \mathbf{\varepsilon} \sim N\left(\mathbf{0}, \sigma^{2}I_{n}\right)
$$

Let's generalize the specification of the error term in the model:

$$
E(\mathbf{\varepsilon} ) = 0, \,\,\,\, E\left(\mathbf{\varepsilon} \mathbf{\varepsilon} ^{\prime} \right) = \Sigma = \sigma^{2}\Omega, \,\,\,\, (\text{and Normal})
$$

This allows for the possibility of one or both of

* Heteroskedasticity
* Autocorrelation (Cross-sectional data, Time-series, Panel data)

##### Spherical Disturbances – Homoskedasticity and Non-Autocorrelation

```python
import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(20,20))
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline 
#Parameters to set

mu_x = 0
variance_x = 1
mu_y = 0
variance_y = 1
ro = 0 #correlation between X and Y

#Create grid and multivariate normal
x = np.linspace(-3,3,500)
y = np.linspace(-3,3,500)
X, Y = np.meshgrid(x,y)
position = np.empty(X.shape + (2,))
position[:, :, 0] = X; 
position[:, :, 1] = Y
rv = multivariate_normal(mean = [mu_x, mu_y] , cov = [[variance_x, ro], [ro, variance_y]])
props = rv.pdf(position)

#Make a 3D plot
fig1 = plt.figure(num = 1, figsize=(20,10))
ax = fig1.gca(projection='3d')
ax.plot_surface(X, Y, props, cmap='viridis',linewidth=0)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

fig2 = plt.figure(num = 2, figsize=(10,10))
plt.contour(props.reshape(500,500))
plt.title('$\mu_{1} = 0, \sigma_{1} = 1, \mu_{2} = 0, \sigma_{2} = 1, \\rho = 0$')

fig1.savefig(fname = 'Homoskedasticity_Nonautocorrelation1.png')
fig2.savefig(fname = 'Homoskedasticity_Nonautocorrelation2.png')

plt.show()
```

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Homoskedasticity_Nonautocorrelation2.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Homoskedasticity_Nonautocorrelation1.png?raw=true)

##### Non-Spherical Disturbances – Heteroskedasticity and Non-Autocorrelation

```python
import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline 
#Parameters to set

mu_x = 0
variance_x = 0.5
mu_y = 0
variance_y = 1
ro = 0 #correlation between X and Y

#Create grid and multivariate normal
x = np.linspace(-3,3,500)
y = np.linspace(-3,3,500)
X, Y = np.meshgrid(x,y)
position = np.empty(X.shape + (2,))
position[:, :, 0] = X; 
position[:, :, 1] = Y
rv = multivariate_normal(mean = [mu_x, mu_y] , cov = [[variance_x, ro], [ro, variance_y]])
props = rv.pdf(position)

#Make a 3D plot
fig = plt.figure(num = 1, figsize=(20,10))
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, props,cmap='viridis',linewidth=0)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

fig2 = plt.figure(num = 2, figsize=(10,10))
plt.contour(props.reshape(500,500))
plt.title('$\mu_{1} = 0, \sigma_{1} = 0.5, \mu_{2} = 0, \sigma_{2} = 1, \\rho = 0$')
fig.savefig(fname = 'Heteroskedasticity_Nonautocorrelation1.png')
fig2.savefig(fname = 'Heteroskedasticity_Nonautocorrelation2.png')

plt.show()
```

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Heteroskedasticity_Nonautocorrelation2.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Heteroskedasticity_Nonautocorrelation1.png?raw=true)

##### Non-Spherical Disturbances – Homoskedasticity and Autocorrelation

```python
import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline 
#Parameters to set

mu_x = 0
variance_x = 1
mu_y = 0
variance_y = 1
ro = 0.5 #correlation between X and Y

#Create grid and multivariate normal
x = np.linspace(-3,3,500)
y = np.linspace(-3,3,500)
X, Y = np.meshgrid(x,y)
position = np.empty(X.shape + (2,))
position[:, :, 0] = X; 
position[:, :, 1] = Y
rv = multivariate_normal(mean = [mu_x, mu_y] , cov = [[variance_x, ro], [ro, variance_y]])
props = rv.pdf(position)

#Make a 3D plot
fig = plt.figure(num = 1, figsize=(20,10))
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, props,cmap='viridis',linewidth=0)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

fig2 = plt.figure(num = 2, figsize=(10,10))
plt.contour(props.reshape(500,500))
plt.title('$\mu_{1} = 0, \sigma_{1} = 1, \mu_{2} = 0, \sigma_{2} = 1, \\rho = 0.5$')
fig.savefig(fname = 'Homoskedasticity_autocorrelation1.png')
fig2.savefig(fname = 'Homoskedasticity_autocorrelation2.png')

plt.show()
```

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Homoskedasticity_autocorrelation2.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Homoskedasticity_autocorrelation1.png?raw=true)

##### Non-Spherical Disturbances – Heteroskedasticity and Autocorrelation

```python
import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline 
#Parameters to set

mu_x = 0
variance_x = 0.5
mu_y = 0
variance_y = 1
ro = 0.5 #correlation between X and Y

#Create grid and multivariate normal
x = np.linspace(-3,3,500)
y = np.linspace(-3,3,500)
X, Y = np.meshgrid(x,y)
position = np.empty(X.shape + (2,))
position[:, :, 0] = X; 
position[:, :, 1] = Y
rv = multivariate_normal(mean = [mu_x, mu_y] , cov = [[variance_x, ro], [ro, variance_y]])
props = rv.pdf(position)

#Make a 3D plot
fig = plt.figure(num = 1, figsize=(20,10))
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, props,cmap='viridis',linewidth=0)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

fig2 = plt.figure(num = 2, figsize=(10,10))
plt.contour(props.reshape(500,500))
plt.title('$\mu_{1} = 0, \sigma_{1} = 0.5, \mu_{2} = 0, \sigma_{2} = 1, \\rho = 0.5$')
fig.savefig(fname = 'Heteroskedasticity_autocorrelation1.png')
fig2.savefig(fname = 'Heteroskedasticity_autocorrelation2.png')

plt.show()
```

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Heteroskedasticity_autocorrelation2.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Heteroskedasticity_autocorrelation1.png?raw=true)

#### What is the sum of the independent normal distributed random variables?

If $x_{1}, x_{2}, \dots , x_{n}$ are mutually independent Gaussian distributed random variables with means $\mu_{1}, \mu_{2}, \dots , \mu_{n}$ and variances $\sigma_{1}^{2}, \sigma_{2}^{2}, \dots , \sigma_{n}^{2}$ then linear combination $y = \sum_{i=1}^{n} c_{i} x_{i}$ follows normal distribution with mean $\sum_{i=1}^{n} c_{i} \mu_{i}$ and variance $\sum_{i=1}^{n} c_{i}^{2} \sigma_{i}^{2}$. 

We can prove this using moment generating function (MGF). If $x \sim N(\mu, \sigma^{2})$, its moment generating functon is given by 

$$
M_{x}(t) = E\left(e^{tx} \right) = \exp \left(\mu t + \frac{\sigma^{2} t^{2}}{2} \right)
$$

We know that $y = \sum_{i=1}^{n} c_{i} x_{i}$, so we can write $y$'s MGF as follows:

$$
\begin{split}
M_{y} (t) &= E\left(e^{ty} \right)\\
&= E\left(e^{t \left(c_{1}x_{1} + c_{2}x_{2} + \dots + c_{n}x_{n} \right)} \right)\\
&= E\left(e^{t c_{1}x_{1} + t c_{2}x_{2} + \dots + t c_{n}x_{n} } \right)\\
&= E\left(e^{t c_{1}x_{1}} e^{t c_{2}x_{2}} \dots e^{t c_{n}x_{n}}\right)\\
&= E\left(e^{t c_{1}x_{1}}\right) E\left(e^{t c_{2}x_{2}}\right) \dots E\left(e^{t c_{n}x_{n}}\right) \,\,\,\,\text{since $x_{i}$s are independent}\\
&= \prod_{i=1}^{n} E\left(e^{t c_{i}x_{i}}\right) \\
&= \prod_{i=1}^{n} M_{x}(c_{i} t) \\
&= \prod_{i=1}^{n} \exp \left(\mu_{i} c_{i} t_{i} + \frac{\sigma_{i}^{2} (c_{i}t)^{2}}{2} \right)\\
&= \exp \left[t \left(\sum_{i=1}^{n} c_{i} \mu_{i} \right) + \frac{t^{2}}{2} \left(c_{i}^{2} \sigma_{i}^{2} \right) \right]
\end{split}
$$

We have just shown that moment generating function of $y$ is the same as the moment generating function of a normal random variable with mean $\sum_{i=1}^{n} c_{i} \mu_{i}$ and variance $\sum_{i=1}^{n} c_{i}^{2} \sigma_{i}^{2}$. Therefore, by uniquess property of moment generating functions, we can say that $y$ also follows a normal distribution with said mean and said variance.

#### What are the non-parametric equivalent of some parametric tests?

Most of the hypothesis testing and confidence interval procedures are based on the assumption that we are working with random samples from normal population. Traditionally, we have called these procedures **parametric methods** because they are based on a particular family of distributions - in this case, the normal. Altenately, sometimes we say that these procedures are not *distribution-free* because they depend of the assumption of normality. Fortunately, most of these procedures are relatively insensitive to slight departures from normality. Still... there are nonparametric and distribution-free methods exist in the literature. 

Mann–Whitney U test (also called the Mann–Whitney–Wilcoxon (MWW), Wilcoxon rank-sum test, or Wilcoxon–Mann–Whitney test) is the nonparametric equivalent of the two sample t-test and is used to test whether two samples are likely to derive from the same population (i.e., that the two populations have the same shape). Some investigators interpret this test as comparing the medians between the two populations. Recall that the parametric test compares the means ($H_{0}: \mu_{1} = \mu_{2}$) between independent groups.

The Wilcoxon signed-rank test is a non-parametric statistical hypothesis test used to compare two related samples, matched samples, or repeated measurements on a single sample to assess whether their population mean ranks differ (i.e. it is a paired difference test). It can be used as an alternative to the paired Student's t-test (also known as "t-test for matched pairs" or "t-test for dependent samples")

The Kruskal–Wallis test by ranks, Kruskal–Wallis H test, or one-way ANOVA on ranks is a non-parametric method for testing whether samples originate from the same distribution. It is used for comparing two or more independent samples of equal or different sample sizes. It extends the Mann–Whitney U test, which is used for comparing only two groups. The parametric equivalent of the Kruskal–Wallis test is the one-way analysis of variance (ANOVA).

#### Explain A/B test and its variants.

When marketers (including product developers and designers) create a landing pages, write email copy or design call-to-action buttons, it can be tempting to use the intuition to predict what will make people click and convert, but basing marketing decisions off of a feeling can be pretty detrimental to results. Rather than relying on guesses or assumptions to make decisions, we are better off running an A/B test because marketing is always changing and the best way to learn what works and what does not work is through trial and error.

A/B testing, also known as split testing, refers to an experiment technique to determine whether a new design brings improvement, according to a chosen metric (such as average time spent on the landing page per session, conversion rate, defined as proportions of sessions ending up with a transaction, cart abandonment rate, bounce rate, click-through rate, the number of applications or registrations, the number of purchases or the average check, e-mail open rate, and so on) using live traffic. A/B testing is a way to compare two versions of a single variable, typically by testing a subject's response to variant A (sometimes it is called control or baseline) against variant B and determining which of the two variants is more effective. An A/B test will enable us to understand that a trial is effective and cost-efficient. It includes application of data-backed statistical hypothesis testing or "two-sample hypothesis testing" as used in the field of statistics.

 ![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/AB_test_example1.png?raw=true)

In web analytics, the idea is to challenge an existing version of a website (A) with a new one (B) by randomly splitting the traffic (sometimes during A/B tests you may notice that traffic numbers of each variation are not identical. This does not mean that anything is wrong with test, just that random variations work, well, randomly. As more traffic hits your site, however, the numbers should become closer to 50-50.) and comparing metrics on each of the splits (however, A/B tests can be used for all other types of stuff such as emails, paid search ad campaigns etc.), in order to increase web traffic or increase conversion rate or decrease bounce rate or decrease cart abandonment.  The knowledge of a User Experience (UX) designer is crucial in singling out feature suggestions that are likely to work. It would often follow the best practices in UX or design examples, that proved to be successful in other similar contexts. However, no prior assumption can beat the real live test that is A/B test. The A/B test measures performance live, with real clients. Provided it is well executed, with no bias when sampling populations A and B, it gives you the best estimate of what would happen if you were to deploy version B.

There is also A/B/n testing which is a type of website testing where multiple versions of a web page are compared against each other to determine which has the highest conversion rate. In this type of test, traffic is split randomly and evenly distributed (equally) between the different versions of the page to determine which variation performs the best. A/B/n testing is an extension of A/B testing, in which two versions of a page (a version A and version B) are tested against each other. However, with an A/B/n test, more than two versions of a page are compared against each other at once. “n” refers to the number of versions being tested, anywhere from two versions to the “nth” version.  

Though they’re often confused, A/B/n testing is not the same as multivariate testing. While A/B/n testing is testing multiple variations of a page, it isn’t testing multiple variables. That’s an important concept to understand. A multivariate test compares multiple versions of a page at once, by testing all possible combinations of variations at once. Multivariate testing is more comprehensive than A/B/n testing and is used to test changes to specific elements on a page whereas A/B/n testing can be used to test completely different versions of a page against each other.

Potential downsides of an A/B/n test are that testing too many variations can further divide traffic to the website among many variations. This can increase the amount of time and traffic required to reach statistically significant result and create what some might call “statistical noise” in the process. Another consideration to be mindful of when running multiple A/B/n tests is to not lose sight of the bigger picture. Just because different variables performed the best in their own experiments, it doesn’t always mean those variables would work well combined. Consider running multivariate tests to test all variations and make sure that improvements to top level metrics carry all the way through the conversion funnel.

A/B testing is best used to measure the impact of multiple variations of an element of the website till you find the best possible version. Tests with more variables take longer to run, and in and of itself, A/B testing will not reveal any information about interaction between variables on a single page. If you need information about how many different elements interact with one another, multivariate testing is the optimal approach! Multivariate testing uses the same core mechanism as A/B testing, but compares a higher number of variables, and reveals more information about how these variables interact with one another. As in an A/B test, traffic to a page is split between different versions of the design. The purpose of a multivariate test, then, is to measure the effectiveness each design combination has on the ultimate goal. Testing all possible combinations of a multivariate test is also known as full factorial testing, and is one of the reasons why multivariate testing is often recommended only for sites that have a substantial amount of daily traffic — the more variations that need to be tested, the longer it takes to obtain meaningful data from the test. It is, however, the most accurate way to run a multivariate test

The total number of testing variations (also called challengers) depends on the number of elements you will test on a page (headline, image, buttons, etc.) and the number of variations you will be testing for each of these elements. You can calculate the total number of challengers in a multivariate test multiplying the number of different variations of each of the elements. For example, for a webpage in which we will be testing (N) number of elements, we calculate:

Total number of page variations = Number of variations of 1st element  x Number of variations of the 2nd element x Number of variations of the 3rd element x …x Number of variations of the Nth element

Since all experiments are fully factorial, too many changing elements at once can quickly add up to a very large number of possible combinations that must be tested. Even a site with fairly high traffic might have trouble completing a test with more than 25 combinations in a feasible amount of time. Therefore, sometimes it may not be worth the extra time necessary to run a full multivariate test when several well-designed A/B tests will do the job well because it might be more difficult to determine the impact of each individual change you make to each page.

Some people confuse Split URL with A/B testing but two are fundamentally different. Split URL testing is testing multiple versions of the webpage hosted on different URLs. The website traffic is split between control, and variations and each of their conversion rates is measured to decide the winning session. The main difference between a Split URL test and A/B test is that in case of a Split Test, the variations are hosted on different URLs. A/B testing is preferred when only front-end changes are required but split URL testing is preferred when significant design changes are necessary, and you do not want to touch existing website design. 

There are two different statistical approaches to testing. (1) Frequentist approach of probability defines the probability of an event with relation to how frequently (hence the name) a particular event occurs in a large number of trials/data points. (2) On the other hand, Bayesian Statistics is a theory based on the Bayesian interpretation of probability, where probability is expressed as a degree of belief in an event. Once you have locked down on either one of these types and approaches based on your website’s needs and business goals, kick of the test and wait for the stipulated time for achieving statistically significant results. However, no matter what method you choose, your testing method and statistical accuracy will determine the end results. 

A/B testing most commonly fails because the test itself has unclear goals, so you need to decide what you are testing. For example, testing the hypothesis “are people more likely to click a red button or a blue button?” is easier because we can easily quantify this change. You'll have more conclusive results for your test if your website A and website B are identical except for the variable that you're testing. However, people run into trouble with A/B testing when their theories are too vague, like, testing two entirely different designs with multiple variants. If you have multiple different treatments running at once, you might have troubles. A number of factors from each different design can get in there and muddy the test result waters, so to speak. Therefore, making minor, incremental changes to your web page with A/B testing instead of getting the entire page redesigned is essential. You need to isolate one “independent variable” and measure its performance. Otherwise, you cannot be sure which one was responsible for changes in performance. After choosing the variable to test, you also carefully pick a primary metric to focus on – before you run the test. This is your “dependent variable”.

If you want your A/B test to be successful you need to follow additional rules. In A/B testing, hypothesis is formulated before conducting a test. If you start with wrong hypothesis, the probability of test succeeding decreases. For any experiment, there is a null hypothesis (status quo), which states there’s no relationship between the two things you’re comparing, and an alternative hypothesis. An alternative hypothesis typically tries to prove that a relationship exists and is the statement you’re trying to back up. When running statistical significance tests, it’s useful to decide whether your test will be one sided or two sided (sometimes called one tailed or two tailed). A one-sided test assumes that your alternative hypothesis will have a directional effect, while a two-sided test accounts for if your hypothesis could have a negative effect on your results, as well. Generally, a two-sided test is the more conservative choice. 

Additionally, in order to obtain meaningful results, we want our test to have sufficient statistical power. And, sample size influence statistical power. As sample size increases, the statistical power increases. Therefore, for our test to have desirable statistical power (usually 0.80), we want to estimate the minimum sample size required which depends on the desired power of the test, significance level, variability of the outcome, and the effect size (The effect size is the difference in the parameter of interest that represents a meaningful difference (practical significance)). The standard deviation and effect size can be either determined from previous studies from literature or from pilot studies. You need to use balanced traffic in order to get significant results. Using lower or higher traffic than required for testing increases the chances of your campaign failing or generating inconclusive results. Testing too many elements of a website together makes it difficult to pinpoint which element influenced the success or failure of the test most. 

Apart from this, more the elements tested, more needs to be the traffic on that page to justify statistically significant testing. Based on your traffic and goals, you need to run A/B test for a certain length of time for it to achieve statistical significance by obtaining a substantial sample size. Running test for too long or too short might result in test failing or producing insignificant results. Because one version of your website appears to be winning within the first few days of starting the test does not mean that you should call it off before time and declare a winner! Stopping a test before the predefined sample size is reached does not only causes a statistical bias toward detecting a difference that is not there, but that bias is not something we can quantify in advance and correct. In addition to that, letting a campaign run for too long is also common blunder that businesses commit. Duration for which you need to run your test depends on your company and various factors like existing traffic, existing conversion rate, expected improvement and so on. In order to get a representative sample and for your data to be accurate, experts recommend that you run your test for a minimum of one to two week. It is also  recommended running the test for a maximum period of four weeks but no more than four to avoid problems due to sample pollution and cookie deletion. 

A/B testing is an iterative process, with each test building upon the results of the previous tests, Businesses give up on A/B testing after their first test fails. But to improve the chances of your next test succeeding, you should draw insights from your last tests while planning and deploying your tests. Additionally, you should not stop testing after a successful test. Test each element repetitively to produce the most optimized version of it even if they are a product of a successful campaign. 

Tests should be run in comparable periods and simultaneously to produce meaningful results. It is wrong to compare website traffic on the days when it gets the highest traffic to the days when it witnesses the lowest traffic because of the external factors such as sale, holidays and so on. The only exception here is if you are testing timing itself, like finding optimal times for sending out emails. The right experiment should cover all the weekdays, all traffic sources and so on (essentially a full business cycle). 

Before you can test the results of A/B test, you have to be sure that the test has reached the statistical significance – the point at which you can have a 95% confidence or more in results. The good news is that many A/B testing tools have statistical significance built right in so you can get an indication as to when your test is ready for interpretation.

**SOME NOTES TO CONSIDER**:

* While hypothesis testing looks promising, it is, in reality, often far from bulletproof because it relies on certain hidden assumptions that are often not satisfied in real-life scenarios. The first assumption is usually pretty solid: we assume that the “samples” (namely the visitors we expose to the variations) are independent of each other, and their behavior is not inter-dependent. This assumption is usually valid unless we expose the same visitor repeatedly and count these occurrences as different exposures. The second assumption is that the samples are identically distributed. Simply stated, this means that the probability of converting is the same for all visitors. This, of course, is not the case. The probability of converting may depend on time, location, user preferences, referrer, and many other potential factors. For example, if during the experiment, some marketing campaign is running, it may cause a surge of traffic from Facebook. This may cause a drastic and sudden change in CTRs (click-through rates), which is based on the fact that people coming from that particular campaign have different characteristics than the general visitor population. The last assumption is that the measures we sample, e.g., the CTR or conversion rate, are normally distributed. It might sound like some obscure mathematical term to some, but the “magic” confidence level formulas depend on this assumption, which is much shakier and often does not hold. In general, the bigger the sample size and the higher the number of conversions we have, the stronger this assumption holds – thanks to a mathematical theorem called the central-limit theorem.
* You also need to be careful of the data noise in you’re A/B test. Statistical criteria cannot catch it. For example, many people can be interested in a new feature or a new design when it is initially released. This leads to abnormal behavior and should be cleaned up in a final comparison.
* Product thinking is critical in A/B test. Sometimes a chance is obviously better UX but the test would take months to be statistically significant. If you are confident that the change aligns with your product strategy and creates better experience for users, you may forgo an A/B test. In these cases, you can take qualitative approaches to validate ideas such as running usability tests or user interviews to get feedback from users. It is a judgement call. If A/B tests are not practical for a given situation, you need to use another tool in the toolbox to make progress in order to continue improvement of the product. In many cases, A/B testing is just one approach to validate a change. 
* Sometimes, the results of an A/B tests can be inconclusive, with no measurable differences between the baseline and the new version, either positive or negative. In these situations, you can either stay with the original version or you can decide to make a change to a new version, depending on other product considerations. 

#### What is the difference between a mixture model and a multimodal distribution?

Mixture models simply assume that the dataset is a mixture in unknown proportions, which results from two or more different unknown populations. We don’t have to assume that the populations are normal distributions, but that assumption is usually plausible and useful. The computing task is then finding estimates of the means and standard deviations of the component distributions and the proportion in which they are present. A very efficient way to do that is to use the **EM algorithm**, which employs maximum likelihood estimation.

Multimodal distribution is a continuous probability distribution with two or more modes and generally can arise when results from two or more different processes are combined. If there are two modes, the distribution is called bimodal.

Imagine a scenario where two distributions (e.g., two univariate normals) have the same mean, but different variances. In this example, they together form a unimodal distribution from a mixture of two different populations.

You can also consider the beta distribution with $\alpha =0.5$ and $\beta = 0.5$. It is illustrated by the red line in the figure below. As you can see, it is multimodal (viz., bimodal), but it isn't a mixture distribution:

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/bimodal_beta.png?raw=true)


#### What is a confounding variable?

In statistics, a confounder is a variable that influences both the dependent variable and independent variable. For an example, if you are researching whether a lack of exercise leads to weight gain: lack of exercise is here independent variable and  weight gain is the dependent variable. A confounding variable here would be any other variable that affects both of these variables, such as the age of the subject.


## General Machine Learning

#### What is hypothesis in Machine Learning?

Machine Learning, specifically supervised learning, can be described as the desire to use available data to learn a function that best maps inputs to outputs. Technically, this is a problem called function approximation, where we approximating an unknown target function (that we assume it exists) that can best map inputs to outputs on all possible observations from the problem domain. An example of a model that approximates the target function and performs mappings of inputs to outputs is called a hypothesis in Machine Learning. The choice of algorithm (e.g. neural network) and the configuration of the algorithm (e.g. network architecture and hyperparameters) define the space of the possible hypothesis that the model may represent.

#### What is the matrix used to evaluate the predictive model? How do you evaluate the performance of a regression prediction model vs a classification prediction model?

Confusion Matrix, also known as an error matrix, is a specific table layout that allows visualization of the complete performance of an algorithm. Each row of the matrix represents the instances in a predicted class while each column represents the instances in an actual class (or vice versa). The name stems from the fact that it makes it easy to see if the system (algorithm) is confusing two classes (i.e. commonly mislabeling one as another). It is a special kind of contingency table, with two dimensions ("actual" and "predicted"), and identical sets of "classes" in both dimensions (each combination of dimension and class is a variable in the contingency table).

* **Regression problems**: Mean Absolute Error, Mean Squared Error, Root Mean Squared Error, R-squared
* **Classification problems**: Accuracy, Precision, Recall, Sensitivity, Specificity, False Positive Rate, F1 Score, AUC, Lift and gain charts

There are different metrics to classify a dataset, depending on the use case, for example, if we have an imbalanced dataset we would concentrate more on F1 score and less on accuracy.

#### What is confusion matrix and its elements?

A much better way to evaluate the performance of a classifier is look at the confusion matrix. In order to compute the confusion matrix, you first need to have a set of predictions, so they can be compared to the actual targets.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/confusion_matrix_elements.png?raw=true)

Let’s first understand the concepts of True Positive (TP), True Negative (TN), False Positive (FP), and False Negative (FN). In all of those, the first word refers to wether the classifier got it right or not, and the second to the predicted value. For example a True Positive is when the classifier predicted a Positive and it was True. A False Negative is when an algorithm predicted a Negative but that prediction was False (it was in fact a Positive).

* **Positive (P)**: Observation is positive

* **Negative (N)**: Observation is not positive

* **True Positive (TP)**: Observation is positive and is predicted to be positive

* **True Negative (TN)**: Observation is negative and is predicted to be negative

* **False Positive (FP)**: Observation is negative and is predicted to be positive. It is also called Type I error which is rejection of true null hypothesis.

* **False Negative (FN)**: Observation is positive and is predicted to be negative. It is also called Type II error which is non-rejection of false null hypothesis. 1 - FNR will give the power of the hypothesis test. 

* **Accuracy (ACC)**: Accuracy is calculated as the number of correct predictions divided by total number of observations in the dataset. The best accuracy is 1.0 where as the worst is 0.0.

  $$
  ACC = \frac{TP + TN}{TP + TN + FP + FN}
  $$
  
* **Error Rate (ERR)**: Error rate if calculated as the number of all incorrect predictions, divided by the total number of observations in the dataset. The best error rate is 0.0 where as the worst is 1.0.

  $$
  ERR = \frac{FP + FN}{TP + TN + FP + FN}
  $$

* **True Positive Rate (TPR/Recall/Sensitivity)**: When it is actually positive, how often does the classifier predict positive? It is also called _Recall_ or _Sensitivity_. The best sensitivity is 1.0 whereas the worst is 0.0. It is a measure of a classifier’s completeness. Low sensitivity (recall) indicates a high number of false negatives. 

  $$
  TPR = \frac{TP}{TP+FN}
  $$
  
  Recall actually calculates how many of the Actual Positives our model capture through labeling it as Positive (True Positive). Recall shall be the model metric we use to select our best model when there is a high cost associated with False Negative. For instance, in fraud detection or sick patient detection. If a fraudulent transaction (Actual Positive) is predicted as non-fraudulent (Predicted Negative), the consequence can be very bad for the bank. Similarly, in sick patient detection. If a sick patient (Actual Positive) goes through the test and predicted as not sick (Predicted Negative). The cost associated with False Negative will be extremely high if the sickness is contagious.
  
* **True Negative Rate (TNR / Specificity)**: When it is actually negative, how often does the classifier predict negative? It is also known as _Specificity_. It is equivalent of 1 - FPR.

  $$
  TNR = \frac{TN}{TN + FP}
  $$
  
* **False Positive Rate (FPR)**: When it is actually negative, how often does the classifier predict positive?  FPR is equal to the significance level, which is Type I error.  It is also known as false alarm rate, fall-out or 1 - specificity.

  $$
  FPR = \frac{FP}{FP+TN}
  $$

* **False Negative Rate (FNR)**: When it is actually positive, how often does the classifier predict negative? FNR is Type II error. 1-FNR equals to the power (sensitivity) of the test in statistical hypothesis testing. It is also known as Miss-rate.

  $$
  FNR = \frac{FN}{FN + TP}
  $$

* **Precision**: Out of all positive classes that we have predicted, how many are actually positive? Precision is also called Positive Predictive Value. It is a measure of a classifier’s exactness. Low precision indicates a high number of false positives.

  $$
  Precision = \frac{TP}{TP + FP}
  $$
  
  The denominator is actually the Total Predicted Positive. Precision talks about how precise/accurate your model is out of those predicted positive, how many of them are actual positive. Precision is a good measure to determine, when the costs of False Positive is high. For instance, email spam detection. In email spam detection, a false positive means that an email that is non-spam (actual negative) has been identified as spam (predicted spam). The email user might lose important emails if the precision is not high for the spam detection model.

* **F1 Score**: This is harmonic mean of TPR (Sensitivity / Recall) and Precision. F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0. Therefore, the F1 score can not be greater than precision. It gives a better measure of the incorrectly classified cases than the Accuracy Metric. Accuracy is used when the True Positives and True negatives are more important while F1-score is used when the False Negatives and False Positives are crucial. It is a better metric when there exists uneven class distributions.

  $$
  \text{F1 Score} = \left(\frac{2}{Recall^{-1} + Precision^{-1}} \right)= \frac{ 2 \times Recall \times Precision}{Recall + Precision}
  $$
  
  It is difficult to compare two models with low precision and high recall or vice versa. So, in order to make them comparable, we use F1 Score. F1 Score helps to measure Recall and Precision at the same time. It is needed when you want to seek a balance between Precision and Recall. 
  
  Note that Precision, Recall and F1 scores are mostly used when the costs of having a misclassified actual positive (or false negative) is very high. For example, positive is actually someone who is sick and carrying a virus that can spread very quickly? Or the positive case represents a fraud case? Or the positive represents a terrorist that the model says its a non-terrorist?

* **ROC curves and AUC**: ROC curves are two-dimensional graphs in which true positive rate (TPR) is plotted on the Y axis and false positive rate (FPR) is plotted on the X axis. An ROC graph depicts relative tradeoffs between benefits (true positives, sensitivity) and costs (false positives, 1-specificity) (any increase in sensitivity will be accompanied by a decrease in specificity). AUC is computed as area under this curve. It is a performance measurement (evaluation metric) for classification problems that consider all possible classification threshold settings. The probabilistic interpretation of ROC-AUC score is that if you randomly choose a positive case and a negative case, the probability that the positive case outranks the negative case according to the classifier is given by the AUC. Here, rank is determined according to order by predicted values.

* **Matthews correlation coefficient**: It is used in machine learning as a measure of the quality of binary (two-class) classifications and is equivalent to Karl Pearson's phi coefficient, which is a measure of association for two binary variables. can be calculated directly from the confusion matrix using the formula:

$$
\text{MCC}=\frac{ TP \times TN - FP \times FN}{\sqrt { (TP + FP)(TP + FN)(TN + FP)(TN + FN)}}
$$

**Which one to choose?**

Here is a set of questions you need to ask yourself, to decide the model selection metrics.

First Question: Does both True Positive and True Negatives matters to the business or just True Positives? If both is important, Accuracy is what you go for.

Second Question: After establishing that True Positive is what you are concerned with more, ask yourself, which one has a higher costs to business, False Positives or False Negatives?

If having large number of False Negatives has a higher cost to business, choose Recall.

If having large number of False Positives has a higher cost to business, choose Precision.

If you cannot decide, using the best of both worlds is way to go, then choose F1.

#### What is a linear regression model?

Linear regression is perhaps one of the most well known and well understood algorithms in statistics and machine learning.  it studies the linear, additive relationships between variables. The overall idea of regression is to examine two things: (1) does a set of predictor variables do a good job in predicting an outcome (dependent) variable?  (2) Which variables in particular are significant predictors of the outcome variable, and in what way do they–indicated by the magnitude and sign of the beta estimates–impact the outcome variable? The values of the coefficients of the regression equation are estimated using Ordinary Least Square method or a numerical approach, such as Gradient Descent.

#### What are the assumptions required for linear regression?

* Linear Relationship between the features and target. Violations of linearity or additivity are extremely serious. The assumptions of linearity and additivity are both implicit in a simple linear regression equation. When linearity is violated, we can deal with it by applying a nonlinear transformation to the dependent and/or independent variables if you can think of a transformation that seems appropriate. you can use $log(y)$ or $y^{2}$ or $\sqrt{y}$ or $1/y$ instead of y for the outcome. (It never matters whether you choose natural vs. common log). In general a non-linear regression model should be considered. You can also include polynomial terms ($X$, $X^{2}$, $X^{3}$) in your model to capture the non-linear effect.

* The number of observations must be greater than number of features.

* No Multicollinearity between the features: Multicollinearity is a state of very high inter-correlations or inter-associations among the independent variables. It is therefore a type of disturbance in the data if present weakens the statistical power of the regression model. Pair plots and heatmaps(correlation matrix) can be used for identifying highly correlated features. If this assumption are violated, estimates of coefficients will be unrealistically large, untrustable / unstable, meaning that if you construct estimators from different data samples you will potentially get wildly different estimates of your coefficient values. Similarly, the variance of the estimates will also blow up. Large estimator variance also undermines the trustworthiness of hypothesis testing of the significance of coefficients, which will thus yield a lower t-value, which could lead to the false rejection of a significant predictor (ie. a type II error). Let's see this mathematically. The variance of the estimates is given by

$$
Var(\hat{\theta}_{OLS}) = \sigma^{2} \left(\mathbf{X}^{T} \cdot \mathbf{X} \right)^{-1}
$$

If you dive into the matrix algebra, you will find that the term $\mathbf{X}^{T} \cdot \mathbf{X}$ is equal to a matrix with ones on the diagonals and the pairwise Pearson’s correlation coefficients ($\rho$) on the off-diagonals (this is the true when columns are standardized.):

$$
(\mathbf{X}^{T} \cdot \mathbf{X}) =\begin{bmatrix} 1 & \rho \\ \rho & 1 \end{bmatrix}
$$

As the correlation values increase, the values within $(\mathbf{X}^{T} \cdot \mathbf{X})^{-1}$ also increase. Even with a low residual variance, multicollinearity can cause large increases in estimator variance. Here are a few examples of the effect of multicollinearity using a hypothetical regression with two predictors:

$$
\begin{split}
 \rho = .3 &\rightarrow (\mathbf{X}^{T} \cdot \mathbf{X})^{-1} =\begin{bmatrix} 1 & \rho \\ \rho & 1 \end{bmatrix}^{-1} = \begin{bmatrix} 1.09 & -0.33 \\ -0.33 & 1.09 \end{bmatrix}\\
 \rho = .9 &\rightarrow (\mathbf{X}^{T} \cdot \mathbf{X})^{-1} =\begin{bmatrix} 1 & \rho \\ \rho & 1 \end{bmatrix}^{-1} = \begin{bmatrix} 5.26 & -4.73 \\ -5.26 & -4.73 \end{bmatrix}\\
 \rho = .999 &\rightarrow (\mathbf{X}^{T} \cdot \mathbf{X})^{-1} =\begin{bmatrix} 1 & \rho \\ \rho & 1 \end{bmatrix}^{-1} = \begin{bmatrix} 500.25 & -499.75 \\ -499.75 & 500.25\end{bmatrix}
\end{split}
$$

Large estimator variance also undermines the trustworthiness of hypothesis testing of the significance of coefficients. Because, consequently, corresponding t-statistics are typically lower:

$$
t_{n-2} = \frac{\hat{\beta_{j}} - 0}{s_{\beta_{j}}}
$$

An estimator with an inflated standard deviation, $s_{\beta_{j}}$, will thus yield a lower t-value,  which could lead to the false rejection of a significant predictor (ie. a type II error).

* Homoscedasticity of residuals or equal variance $Var \left(\varepsilon \mid X_{1} = x_{1}, \cdots, X_{p}=x_{p} \right) = \sigma^{2}$: Homoscedasticity describes a situation in which the error term (that is, the "noise" or random disturbance in the relationship between the features and the target) is the same across all values of the independent variables. More specifically, it is assumed that the error (a.k.a residual) of a regression model is homoscedastic across all values of the predicted value of the dependent variable. A scatter plot of residual values vs predicted values is a good way to check for homoscedasticity. There are a couple of tests that comes handy to establish the presence or absence of heteroscedasticity – The Breush-Pagan test and the NCV test. There should be no clear pattern in the distribution and if there is a specific pattern, the data is heteroscedastic. It means that the model has not perfectly captured the information in the data. Typically, transformed data will satisfy the assumption of homoscedasticity (for example, Box-cox transformation of dependent variable). Homoscedasticity is one of the Gauss Markov assumptions that are required for OLS to be the best linear unbiased estimator (BLUE). The Gauss-Markov Theorem is telling us that the least squares estimator for the coefficients is unbiased and has minimum variance among all unbiased linear estimators, given that we fulfill all Gauss-Markov assumptions. Absence of homoscedasticity may give unreliable standard error estimates of the parameters. Parameter estimates are still unbiased. But the estimates may not efficient(not BLUE). Given heteroscedasticity, you are not able to properly estimate the variance-covariance matrix. Hence, the standard errors of the coefficients are wrong. This means that one cannot compute any t-statistics and p-values and confidence intervals and consequently hypothesis testing is not possible. Also note that in case of heteroscedasticity, Weighted Least Squares Regression can be used.
  ![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/heteroscedastic-relationships.png?raw=true)
     
* Normal distribution of error terms $\varepsilon \sim N(0, \sigma^{2})$: The fourth assumption is that the error(residuals) follow a normal distribution. However, a less widely known fact is that, as sample sizes increase, the normality assumption for the residuals is not needed. More precisely, if we consider repeated sampling from our population, for large sample sizes, the distribution (across repeated samples) of the ordinary least squares estimates of the regression coefficients follow a normal distribution. As a consequence, for moderate to large sample sizes, non-normality of residuals should not adversely affect the usual inferential procedures. This result is a consequence of an extremely important result in statistics, known as the central limit theorem. Since a number of the most common statistical tests rely on the normality of a sample or population, it is often useful to test whether the underlying distribution is normal, or at least symmetric. This can be done via the following approaches: (1) we can review the distribution graphically (via histograms, boxplots, QQ plots of residuals), (2) we can analyze the skewness and kurtosis and/or (3) we can employ statistical tests (esp. Chi-square, Kolmogorov-Smironov, Shapiro-Wilk, Jarque-Barre, D’Agostino-Pearson)
      
* No autocorrelation of residuals (Independence of errors $Cov \left( \varepsilon_{i} \varepsilon_{j} \mid X_{1} = x_{1}, \cdots, X_{p}=x_{p} \right) = 0, \,\,\, i \neq j$): Autocorrelation occurs when the residual errors are dependent on each other. The presence of correlation in error terms drastically reduces model's accuracy, meaning that there is still room to improve the model.  We need to find a way to incorporate that information into the regression model itself. This assumption violation usually occurs in time series models where the next instant is dependent on previous instant. Autocorrelation can be tested with the help of Durbin-Watson test. The null hypothesis of the test is that there is no serial correlation. Durbin-Watson statistic must lie between 0 and 4. If DW = 2, implies no autocorrelation, 0 < DW < 2 implies positive autocorrelation while 2 < DW < 4 indicates negative autocorrelation. If this assumption is violated, residual variance is often (not always) under-estimated, thus, variances of the estimates of regression coefficients is (often) too low. That will make t-values (often) large, making us reject the too often. Hence, t and F tests are wrong, inference is misleading, we will get wrong conclusions.

If the assumptions are not violated, then the Gauss-Markov theorem indicates that the usual OLS estimates are optimal in the sense of being unbiased and having minimum variance (Best Linear Unbiased Estimator (BLUE)). If one or more of the assumptions are violated, then estimated regression coefficients may be biased (i.e. systematically wrong), and not minimum variance (i.e. their uncertainty increases).

#### What is the standard error of the coefficient?

The standard deviation of an estimate is called the standard error. The standard error of the coefficient measures how precisely the model estimates the coefficient's unknown value. The standard error of the coefficient is always positive.

Use the standard error of the coefficient to measure the precision of the estimate of the coefficient. The smaller the standard error, the more precise the estimate. Dividing the coefficient by its standard error calculates a t-value. If the p-value associated with this t-statistic is less than your alpha level, you conclude that the coefficient is significantly different from zero.


The linear model is written as

$$
\left|
\begin{array}{l}
\mathbf{y} = \mathbf{X} \mathbf{\beta} + \mathbf{\epsilon} \\
 \mathbf{\epsilon} \sim N(0, \sigma^2 \mathbf{I}),
\end{array}
\right.
$$

where $\mathbf{y}$ denotes the vector of responses, $\mathbf{\beta}$ is the vector of fixed effects parameters, $\mathbf{X}$ is the corresponding design matrix whose columns are the values of the explanatory variables, and $\mathbf{\epsilon}$ is the vector of random errors.

It is well known that an estimate of $\mathbf{\epsilon}$ is given by 

$$
\hat{\mathbf{\beta}} = (\mathbf{X}^{\prime} \mathbf{X})^{-1} \mathbf{X}^{\prime} \mathbf{y}.
$$

Hence

$$
E(\hat{\beta}) = E((\mathbf{X}^{T} \mathbf{X})^{-1} \mathbf{X}^{T} \mathbf{y}) =  E[(\mathbf{X}^{T} \mathbf{X})^{-1}\mathbf{X}^{T}(\mathbf{X} \beta + \epsilon)] = \beta
$$

and

$$
\textrm{Var}(\hat{\mathbf{\beta}}) =
 (\mathbf{X}^{\prime} \mathbf{X})^{-1} \mathbf{X}^{\prime}
 \;\sigma^2 \mathbf{I} \; \mathbf{X}  (\mathbf{X}^{\prime} \mathbf{X})^{-1}
= \sigma^2 (\mathbf{X}^{\prime} \mathbf{X})^{-1} (\mathbf{X}^{\prime}
 \mathbf{X})  (\mathbf{X}^{\prime} \mathbf{X})^{-1}
= \sigma^2  (\mathbf{X}^{\prime} \mathbf{X})^{-1},
$$

so that we have $\hat{\beta} \sim \mathcal N(\beta, \sigma^2 (\mathbf{X}^T \mathbf{X})^{-1})$. In other words, OLS estimator is also distributed multivariate normal.

$$
\widehat{\textrm{Var}}(\hat{\mathbf{\beta}}) = \hat{\sigma}^2  (\mathbf{X}^{\prime} \mathbf{X})^{-1},
$$

What does the variance-covariance matrix of the OLS estimator, i.e., $Var(\hat{\beta})$ look like?

$$
Var(\hat{\beta}) = E[(\hat{\beta} - \beta)(\hat{\beta} - \beta)^{\prime}] = \begin{bmatrix}
Var(\hat{\beta_{1}}) & Cov(\hat{\beta_{1}}, \hat{\beta_{2}}) & \cdots & Cov(\hat{\beta_{1}}, \hat{\beta_{p}}) \\
Cov(\hat{\beta_{2}}, \hat{\beta_{1}}) & Var(\hat{\beta_{2}}) & \cdots & Cov(\hat{\beta_{2}}, \hat{\beta_{p}}) \\
\vdots & \vdots & \ddots & \vdots \\
Cov(\hat{\beta_{p}}, \hat{\beta_{1}}) & Cov(\hat{\beta_{p}}, \hat{\beta_{2}}) & \cdots & Var(\hat{\beta_{p}})
\end{bmatrix}
$$

As you can see, the standard errors of the $\hat{\beta}$ (the coefficients of estimators), i.e., $se(\hat{\beta_{j}})$., are given by the square root of the elements along the main diagonal of this variance-covariance matrix.

$$
se(\hat{\beta_{j}}) = \sqrt{\widehat{\textrm{Var}}(\hat{\beta_{j}})} = \sqrt{[\hat{\sigma}^2  (\mathbf{X}^{\prime} \mathbf{X})^{-1}]_{jj}}
$$

Let's say we want to test whether the coefficient of $j$-th variable is significant or not. The hypotheses we have:

$$
\begin{split}
H_0 & : & \beta_{j} = \beta_{j,0} \\
H_1 & : & \beta_{j} \neq \beta_{j,0}
\end{split}
$$

The test statistic used for this test is:

$$
T_{j}= \frac{\hat{\beta_{j}}- \beta_{j,0}}{se(\hat{\beta_{j}})}
$$

Because we test whether the parameter of interest is 0 or not (i.e. $\beta_{j} = 0$), the test statistic simplifies to

$$
T_{j}= \frac{\hat{\beta_{j}}}{se(\hat{\beta_{j}})}
$$

where $\hat{\beta_{j}}$ is the least square estimate of $\beta_{j}$, and $se(\hat{\beta_{j}})$ is its standard error. 

t-test for an estimator has $n-p-1$ degrees of freedom where $p$ is number of explanatory parameters in the model and $n$ is number of training samples.

#### What is collinearity and what to do with it? How to remove multicollinearity?

**Collinearity/Multicollinearity:**
* In multiple regression: when two or more variables are highly correlated or improper use of dummy variables (e.g. failure to exclude one category).
* They provide redundant information
* In case of perfect multicollinearity: $\hat{\beta_{OLS}} =\left(\mathbf{X}^{T} \cdot \mathbf{X} \right)^{-1} \cdot \mathbf{X}^{T}y$ does not exist. When two (or multiple) features are fully linearly dependent, we have singular (noninvertible) $\mathbf{X}^{T} \cdot \mathbf{X}$ since Gramian matrix $\mathbf{X}^{T} \cdot \mathbf{X}$ is not full rank (_rank deficiency_). This is obviously going to lead to problems because since $\mathbf{X}^{T} \cdot \mathbf{X}$ is not invertible.
* It doesn't affect the model as a whole, doesn't bias results
* The standard errors of the regression coefficients of the affected variables tend to be large because $Var(\hat{\beta_{OLS}}) = \sigma^{2} \left(\mathbf{X}^{T} \cdot \mathbf{X} \right)^{-1}$
* The test of hypothesis that the coefficient is equal to zero, may lead to a failure to reject a false null hypothesis of no effect of the explanatory (Type II error)
* Leads to overfitting
* The marginal contribution of any one predictor variable in reducing the error sum of squares depends on which other predictors are already in the model.

There are two types of multicollinearity:

1. **Structural multicollinearity** is a mathematical artifact caused by creating new predictors from other predictors — such as, creating the predictor $x_{2}$ from the predictor $x$.
2. **Data-based multicollinearity**, on the other hand, is a result of a poorly designed experiment, reliance on purely observational data, or the inability to manipulate the system on which the data are collected. In this situation, we could opt to remove one of the two correlated predictors from the model. Alternatively, if we have a good scientific reason for needing both of the predictors to remain in the model, we could go out and collect more data. In order to reduce the multicollinearity that exists, it is not sufficient to go out and just collect any ol' data. The data have to be collected in such a way to ensure that the correlations among the violating predictors is actually reduced. That is, collecting more of the same kind of data won't help to reduce the multicollinearity. The data have to be collected to ensure that the "base" is sufficiently enlarged. Doing so, of course, changes the characteristics of the studied population, and therefore should be reported accordingly.

In the case of structural multicollinearity, the multicollinearity is induced by what you have done. Data-based multicollinearity is the more troublesome of the two types of multicollinearity. Unfortunately it is the type we encounter most often!

**Remove multicollinearity:**
* Make sure you have not fallen into the dummy variable trap
* Obtain more data, if possible. 
* Drop some of affected variables
* Combine the affected variables
* Standardize your independent variables. This may help reduce a false flagging of a condition index above 30.
* Removing correlated variables might lead to loss of information. In order to retain those variables, we can use penalized regression models like ridge or lasso regression. 
* Principal component regression or partial least squares regression can be used.

**Detection of multicollinearity:**
* Large changes in the individual coefficients when a predictor variable is added or deleted
* Insignificant regression coefficients for the affected predictors (due to the t-test computation which uses $Var(\hat{\beta_{OLS}})$), but a rejection of the joint hypothesis that those coefficients are all zero (F-test)
* The extent to which a predictor is correlated with the other predictor variables in a linear regression can be quantified as the R-squared statistic of the regression where the predictor of interest is predicted by all the other predictor variables. The variance inflation factor (VIF) for variable $i$ is then computed as:
    
    \begin{equation}
        VIF = \frac{1}{1-R_{i}^{2}}
    \end{equation}
    
    A rule of thumb for interpreting the variance inflation factor: 
    * 1 = not correlated.
    * Between 1 and 10 = moderately correlated.
    * Greater than 10 = highly correlated.
    
     The rule of thumb cut-off value for VIF is 10. Solving backwards, this translates into an R-squared value of 0.90. Hence, whenever the R-squared value between one independent variable and the rest is greater than or equal to 0.90, you will have to face multicollinearity.
     
     Tolerance (1/VIF) is another measure to detect multicollinearity.  A tolerance close to 1 means there is little multicollinearity, whereas a value close to 0 suggests that multicollinearity may be a threat. 

* Correlation matrix. However, unfortunately, multicollinearity does not always show up when considering the variables two at a time. Because correlation is a bivariate relationship whereas multicollinearity is multivariate.
    
* Eigenvalues of the correlation matrix of the independent variables near zero indicate multicollinearity. Instead of looking at the numerical size of the eigenvalue, we can use the condition number defined as the square root of the ratio of the largest and smallest eigenvalues in the predictor matrix. Large condition numbers indicate multicollinearity.

  $$
  CN = \sqrt{\frac{\lambda_{max}}{\lambda_{min}}}
  $$
  
  $CN > 15$ indicates the possible presence of multicollinearity, while a $CN > 30$ indicates serious multicollinearity problems. One advantage of this method is that it also shows which variables are causing the problem
  
* Investigate the signs of the regression coefficients. Variables whose regression coefficients are opposite in sign from what you would expect may indicate multicollinearity

#### What is Heteroskedasticity and weighted least squares?

The method of ordinary least squares assumes that there is constant variance in the errors (which is called homoscedasticity). If this assumption is violated, we have heteroskedasticity problem. Coming from the ancient Greek, "hetero" means "different", and "skedasis", meaning "dispersion". 

The model under consideration is

$$
\textbf{Y}=\textbf{X}\beta+\epsilon
$$

where now $\epsilon$ is assumed to be (multivariate) normally distributed, with mean vector 0, i.e., $E[\epsilon_i \mid x_i] = 0$ and constant variance-covariance matrix $V[\epsilon_i \mid x_i] = \sigma^2$ for all $i = 1,2, \dots n$, i.e., $\epsilon_i \sim N(0, \sigma^2)$. Under heteroskedasticity, the last assumption no longer holds; we have $V[\epsilon_i \mid x_i] \neq V[\epsilon_j \mid x_j]$ for some $i, j$. If we continue to assume that there is no autocorrelation—that the covariance of each pair of distrinct $\epsilon_{i}$ and $\epsilon_{j}$ is 0 -then we can write the variance matrix of the vector $\epsilon$ as

$$
V[\epsilon \mid \mathbf{X}] = \begin{bmatrix}
\sigma_1^2 & 0 & \cdots & 0 \\
0 & \sigma_2^2 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \sigma_n^2 \end{bmatrix} =
\sigma^2 \begin{bmatrix}
w_1 & 0 & \cdots & 0 \\
0 & w_2 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & w_n \end{bmatrix} = \sigma^2 W
$$

Under heteroskedasticity, the OLS estimator is unbiased, consistent, and asymptotically normal despite heteroskedasticity. However, due to the Gauss-Markov theorem, they are not efficient anymore. If the errors are heteroskedastic, then there is an unbiased linear estimator with a lower variance than OLS. The problem is, to use that estimator, we must know each individual error variance up to a multiplicative constant. In other words, we must know $W$. We usually don't. So there's a more efficient estimator out there, but we’re unlikely to know what it is.

Under homoskedasticity, the variance matrix of the OLS estimator is:

$$
\Sigma = \sigma^2 (\mathbf{X}^\top \mathbf{X})^{-1}
$$

The typical estimate of this variance matrix is

$$
\hat{\Sigma}_{\text{OLS}} = \frac{\sum_{i=1}^{n} \hat{\epsilon}_{i}^{2}}{n - p} (\mathbf{X}^\top \mathbf{X})^{-1},
$$

where $n$ is the number of observations, $p$ is the number of predictors and $\hat{e_{i}}$ is the residual of the $i$-th observation under the OLS estimate. Under homoskedasticity, this is an unbiased and consistent estimator of the true variance matrix. With heteroskedasticity, however, $\hat{\Sigma_{\text{OLS}}}$ is biased and inconsistent. If we go ahead and use it to calculate inferential statistics, our measures of uncertainty will be misleading. Typically, we will too readily reject the null hypothesis—our reported p-values will be understated. 

To sum up, although heteroskedasticity doesn't cause much of a problem for the OLS estimate of $\beta$ itself, it does throw a wrench into our efforts to draw inferences about $\beta$ from the OLS estimate. We are left with two options:

1. Use an estimator other than OLS.
2. Make a correction to the estimated standard errors that accounts for the possibility of heteroskedasticity.

These correspond, respectively, to the cases when the heteroskedasticity is of *known* and *unknown* form.

 A scatter plot of residual values vs predicted values is a good way to check for homoscedasticity. There are also a few tests, such as the Breusch-Pagan test and the NCV test to detect whether there’s heteroskedasticity at all.

The method of weighted least squares can be used when the ordinary least squares assumption of constant variance in the errors is violated (which is called heteroscedasticity). It is is an estimation technique which weights the observations proportional to the reciprocal of the error variance for that observation and so overcomes the issue of non-constant variance.

Non-constant variance-covariance matrix is given by 

$$
\left(\begin{array}{cccc} \sigma^{2}_{1} & 0 &   \ldots & 0 \\ 0 & \sigma^{2}_{2} & \ldots & 0 \\  \vdots  & \vdots & \ddots & \vdots \\ 0 & 0 & \ldots  &  \sigma^{2}_{n} \\ \end{array} \right)
$$

If we define the reciprocal of each estimated variance, $\sigma_{i}^{2}$, as the weight, $w_{i} = \frac{1}{\sigma_{i}^{2}}$ which are known positive constants, then let matrix $W$ be a diagonal matrix containing these weights:

$$
\textbf{W} = \left(\begin{array}{cccc} w_{1} & 0 & \ldots & 0 \\ 0& w_{2} & \ldots & 0 \\ \vdots & \vdots & \ddots &   \vdots \\ 0& 0 & \ldots & w_{n} \\ \end{array}   \right)
$$

Weighted OLS regression assumes that the errors have the distribution $\epsilon_i \sim N(0, \sigma^2/w_{i})$, where the $w_{i}$ are known weights and $\sigma^{2}$ is an unknown parameter that is estimated in the regression.
This is the difference from variance-weighted least squares: in weighted OLS, the magnitude of the error variance is estimated in the regression using all the data.

The weighted least squares estimate is then

$$
\begin{split}
\hat{\beta}_{WLS} (\textbf{Y}, \textbf{X}, w) &= \arg\min_{\beta}\sum_{i=1}^{n} \hat{\epsilon}_{i}^{2}\\
&=(\textbf{X}^{T}\textbf{W}\textbf{X})^{-1}\textbf{X}^{T}\textbf{W}\textbf{Y}
\end{split}
$$

Notice that OLS is a special case of WLS, with $w = (1, 1, \dots , 1)$. Just like OLS, WLS is unbiased and (under reasonable conditions) consistent, even if $\textbf{W}$ is misspecified. But if we have $\textbf{W}$ right—and only if we have $\textbf{W}$ right—then WLS is efficient in the class of linear unbiased estimators. In addition, our estimated variance matrix,

$$
\hat{\Sigma}_{\text{WLS}} = \frac{\sum_{i=1}^{n} \hat{\epsilon}_{i}^{2} / w_i}{n - p} (\mathbf{X}^\top W^{-1} \mathbf{X})^{-1},
$$

is unbiased and consistent. Note that here $n=p$ is the degrees of freedom for sum of squares of errors. 

Since each weight is inversely proportional to the error variance, it reflects the information in that observation. So, an observation with small error variance has a large weight since it contains relatively more information than an observation with large error variance (small weight). 

Apart from the violation of the assumption of homoscedasticity, Weighted Least Squares can also be used when:

1. you have any other situation where data points should not be treated equally. This method gives us an easy way to remove one observation from a model by setting its weight equal to 0.
2. You want to concentrate on certain areas

To apply weighted least squares, the weights, $w_{i}$'s, have to be known up to a proportionality constant (in other words, we know the form of $\textbf{W}$). However, in many real-life situations, the weights are not known apriori (i.e., the structure of $\textbf{W}$ is usually unknown). In such cases we need to estimate the weights in order to use weighted least squares. In some cases, the values of the weights may be based on theory or prior research. There are other circumstances where the weights are known:

1. If the $i$-th response is an average of $n_{i}$ equally variable observations, then $Var(y_{i}) = \sigma^{2}/n_{i}$ and $w_{i} = n_{i}$.
2. If the $i$-th response is a total of ni observations, then $Var(y_{i}) = n_{i}\sigma^{2}$ and $w_{i}  = \frac{1}{n_{i}}$.
3. If variance is proportional to some predictor $x_{i}$, then $Var(y_{i}) = x_{i}\sigma^{2}$ and $w_{i} = \frac{1}{x_{i}}$.

Again... the trick with weighted least squares is the estimation of $W$. If the variances of the observations or their functional form are somehow known, you can use that. More likely, you need to estimate $W$ from residual plots. So, we have to perform an ordinary least squares (OLS) regression first. Provided the regression function is appropriate, the $i$-th squared residual from the OLS fit is an estimate of $\sigma_{i}^{2}$ and the $i$-th absolute residual is an estimate of $\sigma_{i}$ (which tends to be a more useful estimator in the presence of outliers). The residuals are much too variable to be used directly in estimating the weights, $w_{i}$, so instead we use either the squared residuals to estimate a variance function or the absolute residuals to estimate a standard deviation function. We then use this variance or standard deviation function to estimate the weights.

Some possible variance and standard deviation function estimates include:
1. If a residual plot against a predictor exhibits a megaphone shape, then regress the absolute values of the residuals against that predictor. The resulting fitted values of this regression are estimates of $\sigma_{i}$. (And remember $w_{i}= 1/\sigma_{i}^{2}$).
2. If a residual plot against the fitted values exhibits a megaphone shape, then regress the absolute values of the residuals against the fitted values. The resulting fitted values of this regression are estimates of $\sigma_{i}$.
3. If a residual plot of the squared residuals against a predictor exhibits an upward trend, then regress the squared residuals against that predictor. The resulting fitted values of this regression are estimates of $\sigma_{i}^{2}$.
4. If a residual plot of the squared residuals against the fitted values exhibits an upward trend, then regress the squared residuals against the fitted values. The resulting fitted values of this regression are estimates of $\sigma_{i}^{2}$.

This method is also sensitive to outliers. A rogue outlier given an inappropriate weight could dramatically skew the results. However, if we choose proper weights we can downweight outlier or influential points to reduce their impact on the overall model.

WLS can only be used in the rare cases where you know what the weight estimates are for each data point. When heteroscedasticity is a problem, it’s far more common to run OLS instead, using a difference variance estimator, such as White’s heteroskedasticity-consistent estimator. While White’s consistent estimator doesn’t require heteroscedasticity, it isn't a very efficient strategy. However, if you don’t know the weights for your data, it may be your best choice. 

#### What are the assumptions required for logistic regression?

First, logistic regression does not require a linear relationship between the dependent and independent variables. Actually, it is a linear model for the log odds.  Second, the error terms (residuals) do not need to be normally distributed.  Third, homoscedasticity is not required.  Finally, the dependent variable in logistic regression is not measured on an interval or ratio scale.
 
However, some other assumptions still apply.

* __ASSUMPTION OF APPROPRIATE OUTCOME STRUCTURE:__ Binary logistic regression requires the dependent variable to be binary and ordinal logistic regression requires the dependent variable to be ordinal and Multinomial logistic regression requires the dependent variable to be multinomial.

* __ASSUMPTION OF OBSERVATION INDEPENDENCE:__ Logistic regression requires the observations to be independent of each other.  In other words, the observations should not come from repeated measurements or matched data.

* __ASSUMPTION OF THE ABSENCE OF MULTICOLLINEARITY:__ Logistic regression requires there to be little or no multicollinearity among the independent variables. This means that the independent variables should not be too highly correlated with each other.

* __ASSUMPTION OF LINEARITY OF INDEPENDENT VARIABLES AND LOG ODDS:__ Logistic regression assumes linearity of independent variables and log odds. Although this analysis does not require the dependent and independent variables to be related linearly, it requires that the independent variables are linearly related to the log odds.

* __ASSUMPTION OF A LARGE SAMPLE SIZE:__ Logistic regression typically requires a large sample size.

#### Why is logistic regression considered to be linear model?

Logistic regression is not a linear model. It is a generalized linear model.  In order to call a particular method to be GLM, that method should have following three components.

1. Random Component: It refers a response variable (y), which need to satisfy some PDF assumption. For example: Linear regression of y (dependent variable) follows normal distribution. Logistic regression response variable follows binomial distribution.

2. Systematic Component: It is nothing but explanatory variables in the model. Systematic components helps to explain the random component.

3. Link Function: It is link between systematic and random component. Link function tells how the expected value of response variable relates to explanatory variable. Link function of linear regression is $E[y]$ and link function of logistic regression is $logit(p)$.

Logistic Regression is of the form:

$$
\text{logit}(P(y=1)) = log\left(\frac{P(y=1)}{1-P(y=1)}\right)=log\left(\frac{P(y=1)}{P(y=0)}\right)=\theta_{0}+\theta_{1}x_{1}+\ldots+\theta_{p}x_{p}
$$

More generally, in a Generalized Linear Model, the mean, $\mu$, of the distribution depends on the independent variables, $x$, through:

$$
g(\mu) = g(E(y_{i} \mid x_{i}))= \theta_{0}+\theta_{1}x_{1}+\ldots+\theta_{p}x_{p}
$$

where $\mu$ is the expected value of the response given the covariates and $g$ is the link function.

Consequently, its decision boundary is linear. The decision boundary is the set of $x$ such that

$$
\frac{1}{1 + e^{-{X \cdot \theta}}} = 0.5
$$

This is equivalent to

$$
1 = e^{-{X \cdot \theta}}
$$

and, taking the natural log of both sides,

$$
0 = -X \cdot \theta = -\sum\limits_{i=0}^{p} \theta_i x_i = \theta_{0}+\theta_{1}x_{1}+\ldots+\theta_{p}x_{p}
$$

which defines a straight line. So the decision boundary is linear.

#### Why sigmoid function in Logistic Regression?

One of the nice properties of logistic regression is that the sigmoid function outputs the conditional probabilities of the prediction, the class probabilities because the output range of a sigmoid function is between 0 and 1. This transform ensures that probability lies between 0 and 1.

$$
sigmoid (x)=\dfrac{1}{1+e^{-x}}
$$

![](https://raw.githubusercontent.com/mmuratarat/mmuratarat.github.io/master/_posts/images/sigmoid.png)

#### What is the loss function for Logistic Regression?

A very common scenario in Machine Learning is supervised learning, where we have data points $\mathbf{x}^{(i)}$ and their labels $y^{(i)}$, for $i=1, 2, \cdots, m$, building up our dataset where we’re interested in estimating the conditional probability of $y^{(i)}$ given $\mathbf{x}^{(i)}$, or more precisely $P(\mathbf{y} \mid \mathbf{X}, \theta)$.

Now logistic regression says that the probability that class variable value $y^{(i)} = 1$, for $i=1, 2, \cdots, m$ can be modelled as follows

$$P(y^{(i)}=1 \mid \mathbf{x}^{(i)}, \theta) = h_{\theta} ( \mathbf{x}^{(i)} = \dfrac{1}{1+exp(-\theta^{T} \cdot \mathbf{x}^{(i)})} $$

Since $P(y^{(i)}=0 \mid \mathbf{x}^{(i)}, \theta) = 1- P(y^{(i)}=1 \mid \mathbf{x}^{(i)}, \theta) $, we can say that so $y^{(i)}=1$ with probability $ h_{\theta} ( \mathbf{x}^{(i)} )$ and $y^{(i)}=0$ with probability $1− h_{\theta} ( \mathbf{x}^{(i)} )$.

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
J(\theta) &=  - \dfrac{1}{m} \mathcal{L}(\theta)\\
&= - \dfrac{1}{m} \sum_{i=1}^{m} y^{(i)} \log(h_\theta(\mathbf{x}^{(i)})) + (1 - y^{(i)}) \log(1-h_\theta(\mathbf{x}^{(i)}))
\end{align}
$$

As you can see, maximizing the log-likelihood (minimizing the *negative* log-likelihood) is equivalent to minimizing the binary cross entropy. 

Now we can also understand why the cost for single data point comes as follows... The cost for a single data point is $- \log ( P( \mathbf{x}^{(i)} \mid y^{(i)} )) $, which can be written as:

$$ -\left( y^{(i)} \log(h_\theta(\mathbf{x}^{(i)})) + (1 - y^{(i)}) \log(1-h_\theta(\mathbf{x}^{(i)})) \right)$$

We can now split the above into two depending upon the value of $y^{(i)}$. Thus we get:

$$\mathrm{Cost}(h_{\theta}(\mathbf{x}^{(i)}), y^{(i)}) =
\begin{cases}
-\log(h_\theta(\mathbf{x}^{(i)})) & \mbox{if $y^{(i)} = 1$} \\
-\log(1-h_\theta(\mathbf{x}^{(i)})) & \mbox{if $y^{(i)} = 0$}
\end{cases}$$

#### How do you find the parameters in logistic regression?

There is no closed form solution for estimating the parameters of a logistic regression. Instead, an iterative search algorithm is used. The most common choices are the Newton-Raphson algorithm and Gradient-descent algorithm, but there are [many possibilities](https://en.wikipedia.org/wiki/Search_algorithm).

#### What is Softmax regression and how is it related to Logistic regression?

Softmax Regression (a.k.a. Multinomial Logistic, Maximum Entropy Classifier, or just Multi-class Logistic Regression) is a generalization of logistic regression that we can use for multi-class classification (under the assumption that the classes are mutually exclusive). In contrast, we use the (standard) Logistic Regression model in binary classification tasks.

#### What is odds ratio? How to interpret it? How to compute confidence interval for it?

An odds ratio (OR) is a measure of association between an exposure and an outcome. The OR represents the odds that an outcome will occur given a particular exposure, compared to the odds of the outcome occurring in the absence of that exposure. Odds ratios are most commonly used in case-control studies, however they can also be used in cross-sectional and cohort study designs as well (with some modifications and/or assumptions).

When a logistic regression is calculated, the regression coefficient ($\beta_{1}$) is the estimated increase in the log odds of the outcome per unit increase in the value of the exposure. In other words, the exponential function of the regression coefficient ($e^{\beta_{1}}$) is the odds ratio associated with a one-unit increase in the exposure.

The 95% confidence interval (CI) is used to estimate the precision of the OR. A large CI indicates a low level of precision of the OR, whereas a small CI indicates a higher precision of the OR. It is important to note however, that unlike the p value, the 95% CI does not report a measure’s statistical significance. In practice, the 95% CI is often used as a proxy for the presence of statistical significance if it does not overlap the null value (e.g. $OR=1$ because we are taking the exponential of log odds, that means log odds equals to 0). Nevertheless, it would be inappropriate to interpret an OR with 95% CI that spans the null value as indicating evidence for lack of association between the exposure and outcome.

There are three ways to determine if an odds ratio is statistically significant:

1. Fisher's Exact Test
2. Chi-square Test (It compares the observed values to expected values to determine whether there is no relationship between two variabkes)
3. Wald Test (This is used for logistic regression)

#### What is R squared?

R-squared ($R^{2}$) is a statistical measure that represents the proportion of the variance for a dependent variable that's explained by an independent variable or variables in a regression model. Whereas correlation explains the strength of the relationship between an independent and dependent variable, R-squared explains to what extent the variance of one variable explains the variance of the second variable. So, if the $R^{2}$ of a model is $0.50$, then approximately half of the observed variation can be explained by the model's inputs. It may also be known as the coefficient of determination. It is nothing but is a ratio of the explained sum of squares to the total sum of squares:

$$
\begin{split}
R^{2} \left(y_{true}, y_{pred} \right) =&  1- \frac{\text{Sum of Squared}_{residuals}}{\text{Sum of Squared}_{total}} 
&= 1 - \frac{\sum \left(y_{true} - y_{pred}\right)^{2}}{\sum \left(y_{true} - \bar{y} \right)^{2}}
\end{split}
$$

where 
$$
\bar{y} = \frac{1}{n_{samples}}\sum {y_{true}}$$

The denominator is the variance in $y$ values. It is the total sum of squares which tells you how much variation there is in the dependent variable.

Higher the MSE (the nominator), smaller the $R^{2}$ and poorer is the model.

On the web, one can see that the range of $R^{2}$ lies between 0 and 1 which is not actually true. The maximum value of $R^{2}$ is 1 but minimum can be negative infinity

R squared alone cannot be used as a meaningful comparison of models with very different numbers of independent variables. It only works as intended in a simple linear regression model with one explanatory variable. R-squared is monotone increasing with the number of variables included—i.e., it will never decrease because when we add a new variable, regression model will try to minimize the sum of squared of residuals but total sum of squared will be the same. Thus, a model with more terms may seem to have a better fit just for the fact that it has more terms. This leads to the alternative approach of looking at the adjusted R squared. The adjusted R-squared compares the descriptive power of regression models that include diverse numbers of predictors. The adjusted R-squared compensates for the addition of variables and only increases if the new term enhances the model above what would be obtained by probability and decreases when a predictor enhances the model less than what is predicted by chance. In an overfitting condition, an incorrectly high value of R-squared, which leads to a decreased ability to predict, is obtained. This is not the case with the adjusted R-squared.

While standard R-squared can be used to compare the goodness of two or model different models, adjusted R-squared is not a good metric for comparing nonlinear models or multiple linear regressions.

$$
\bar{R}^2 = 1- (1- {R^2})\frac{n-1}{n-p-1}
$$

where $p$ is the total number of explanatory variables in the model (not including the constant term), and n is the sample size.

#### You have built a multiple regression model. Your model $R^{2}$ isn't as good as you wanted. For improvement, your remove the intercept term, your model $R^{2}$ becomes 0.8 from 0.3. Is it possible? How?

Yes, it is possible. We need to understand the significance of intercept term in a regression model. The intercept (often labeled the constant) is the expected mean value of $y$ when all $X = 0$. In other words, it gives model predictions without any independent variable i.e. mean prediction ($\hat{\bar{y}}$). The denominator of the formula of $R^{2}$ contains $\bar{y}$.

When intercept term is present, $R^{2}$ value evaluates your model with respect to to the mean model. In absence of intercept term, the model can make no such evaluation, with large denominator,

$$
R^2 =1 - \frac{\sum \left(y_{true} - y_{pred} \right)^2}{\sum \left(y_{true} \right)^2}
$$

The value of the second component becomes smaller than actual because the denominator of the second component is bigger now and therefore it yields a higher $R^{2}$.

#### How do you validate a machine learning model?

The most important thing you can do to properly evaluate your model is to not train the model on the entire dataset.

* **The train/test/validation split**: A typical train/test split would be to use $70\%$ of the data for training and $30\%$ of the data for testing. It's important to use new data when evaluating our model to prevent the likelihood of overfitting to the training set. However, sometimes it's useful to evaluate our model as we're building it to find that best parameters of a model - but we can't use the test set for this evaluation or else we'll end up selecting the parameters that perform best on the test data but maybe not the parameters that generalize best. To evaluate the model while still building and tuning the model, we create a third subset of the data known as the validation set. A typical train/test/validation split would be to use $60\%$ of the data for training, $20\%$ of the data for validation, and $20\%$ of the data for testing.

* **Random Subsampling (Hold-out) Validation**: This is a simple kind of cross validation technique. You reserve around half of the original dataset for testing (or validation), and the other half for training. Once you get an estimate of the model’s error, you may also use the portion previously used for training for testing now, and vice versa. Effectively, this gives you two estimates of how well your model works. One disadvantage of using only holdout set for model validation is that we have lost a portion of our data to the model training. Even, sometimes, half the dataset does not contribute to the training of the model! This is not optimal, and can cause problems – especially if the initial set of training data is small. One way to address this is to use cross-validation and its variants.

* **Leave-One-Out Cross-Validation**: This is the most extreme way to do cross-validation. Assuming that we have $n$ labeled observations, LOOCV trains a model on each possible set of $n-1$ observations, and evaluate the model on the left out one; the error reported is averaged over the $n$ models trained. This technique is computationally very, very intensive- you have to train and test your model as many times as there are number of data points. This can spell trouble if your dataset contains millions of them. 

* **Cross-Validation**: When you do not have a decent validation set to tune the hyperparameters of the model on, the common approach that can help is called cross-validation. In the case of having a few training instances, it could be prohibitive to have both validation and test set separately. You would prefer to use more data to train the model. In such a situation, you only split your data into a training and a test set. Then you can use cross-validation on the training set to simulate a validation set. Cross-validation works as follows. First, you fix the values of hyperparameters you want to evaluate. Then you split your training set into several subsets of the same space. Each subset is called a _fold_. Typically, five-fold or ten-fold provides a good compromise for the bias-variance trade-off. With five-fold CV, you randomly split your training data into five folds: $\{F_{1}, F_{2}, ..., F_{5}\}$. Each $F_{k}$ contains $20\%$ of the training data. Then you train five models as follows. To train the first model, $f_{1}$, you use all examples from folds  $\{F_{2}, F_{3}, F_{4}, F_{5}\}$ as the training set and the examples from $F_{1}$ as the validation set. To train the second model, $f_{2}$, you use the examples from fold $\{F_{1}, F_{3}, F_{4}, F_{5}\}$ to train and the examples from $F_{2}$ as the validation set. You continue building models iteratively like this and compute the value of the metric of interest on each validation sets, from $F_{1}$ to $F_{5}$. Then you average the five values of the metric to get the final value. You can use grid search with cross-validation to find the best values of hyperparameters for your model. Once you have found those values, you use the entire training set to build the model with these best values of parameters you have found via cross-validation. Finally, you assess the model using the test set. 

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Cross-Validation-Diagram.jpg?raw=true)

{% highlight python %}
all_folds = split_into_k_parts(all_training_data)
 
for set_p in hyperparameter_sets:
    model = InstanceFromModelFamily()
 
    for fold_k in all_folds:
        training_folds = all_folds besides fold_k
        fit model on training_folds using set_p
        fold_k_performance = evaluate model on fold_k
 
    set_p_performance = average all k fold_k_performances for set_p
 
select set from hyperparameter_sets with best set_p_performance
{% endhighlight %}

* **Stratified Cross Validation**:  When we split our data into folds, we want to make sure that each fold is a good representative of the whole data. The most basic example is that we want the same proportion of different classes in each fold. Most of the times it happens by just doing it randomly, but sometimes, in complex datasets, we have to enforce a correct distribution for each fold.

* **Bootstrapping Method**: Under this technique training dataset is randomly selected with replacement and the remaining data sets that were not selected for training are used for testing. The error rate of the model is average of the error rate of each iteration  as estimation of our model performance, the value is likely to change from fold-to-fold during the validation process.

#### What is the Bias-Variance Tradeoff?

If we denote the variable we are trying to predict as $Y$ and our covariates as $X$, we may assume that there is a relationship relating one to the other such as $Y = f(X) + \varepsilon$ where the error term $\varepsilon$ is normally distributed with a mean of zero like so $\varepsilon \sim N(0, \sigma_{\varepsilon})$.

We may estimate a model $\hat{f}(X)$ of $f(X)$ using linear regressions or another modeling technique. In this case, the expected squared prediction error at a point $x$ is:

$$
Err(x)=E[( Y - \hat{f}(x))^{2}]
$$

This error may then be decomposed into bias and variance components:

$$
Err(x)= \left( E \left[ \hat{f}(x) \right] − f(x) \right)^{2} + E \left[ \left( \hat{f}(x) − E \left[ \hat{f}(x) \right] \right)^{2} \right] + \sigma_{\varepsilon}^{2}
 $$

Hence, in a machine learning algorithm, the bias–variance decomposition is a way of analyzing a learning algorithm's expected generalization error with respect to a particular problem as a sum of three terms, the bias, variance, and a quantity called the irreducible error, resulting from noise in the problem itself:

$$
\text{Total Error} = \text{Bias}^{2} + \text{Variance} + \text{Irreducible Error}
$$

Our goal is to minimize the total error, which has been decomposed into the sum of a (squared) bias, a variance, and a constant noise term. As we shall see later, there is a trade-off between bias and variance and this trade-off leads to two concepts called Underfitting and Overfitting. In summary, Underfitting occurs when the model is not able to obtain a sufficiently low error value on the training set. Overfitting occurs when the gap between the training error and test error is too large. We can control whether a model is more likely to overfit or underfit by altering its capacity. Informally, a model's capacity is its ability to fit a wide variety of functions (there are many ways to change a model's capacity). It is very close (if not a synonym) for model complexity. It's a way to talk about how complicated a pattern or relationship a model can express. The most common way to estimate the capacity of a model is to count the number of parameters. The more parameters, the higher the capacity in general. Of course, often a smaller network learns to model more complex data better than a larger network, so this measure is also far from perfect.

Models with high capacity generally are very flexible models having low bias and high variance and they can overfit easily by memorizing the properties of the training set that do not serve them well on the test set. Models with low capacity relatively are rigid models having high bias and low variance. Models with insufficient capacity are unable to solve complex tasks. The model with the optimal predictive capability (sufficiently complex) is appropriate for the true complexity of the task and is the one that leads to the best balance between bias and variance.

We can create a graphical visualization of bias and variance using a bulls-eye diagram, representing combinations of both high and low bias and variance.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/bias_variance_tradeoff_illustration.png?raw=true)

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Screen%20Shot%202019-10-30%20at%2021.10.33.png?raw=true)

The error due to bias taken as the difference between the expected (or average) prediction of our model and the correct value which we are trying to predict. Of course you only have one model so talking about expected or average prediction values might seem a little strange. However, imagine you could repeat the whole model building process more than once: each time you gather new data and run a new analysis creating a new model. Due to randomness in the underlying data sets, the resulting models will have a range of predictions. Bias measures how far off in general these models' predictions are from the correct value.

![](https://elitedatascience.com/wp-content/uploads/2017/05/noisy-sine-linear.png)

It refers to an error from an estimator that is general and does not learn signal in the data that would allow it to make better predictions. This part of the generalization error is due to wrong assumptions, such as assuming that the data is linear when it is actually quadratic. A high-bias model is most likely to underfit the training data. There could be several reasons for underfitting, the most important of which are: (1) your model is too simple for the data (for example a linear model can often underfit), (2) the features you engineered are not informative enough. The solution to the problem of underfitting is to try a more complex model (either through more layers/ tree or different architecture) or to engineer features with higher predictive power. An inflexible model is said to have a high bias because it makes assumptions about the training data (it is biased toward the pre-conceived ideas of the data, we have imposed more rules on the target function). For example, a linear classifier makes the assumption that data is linear, and does not have enough flexibility to fit non-linear relationships. An inflexible model may not have enough capacity to fit even the training data (cannot learn the signal from the data) and the model is not able to generalize well to a new data. 

Examples of low bias ML algorithms: Decision Trees, k-Nearest Neighbors, SVM etc...

Examples of high bias ML algorithms: Generally, parametric algorithms have high bias, making them fast to learn and easier to understand but generally less flexible. In turn, they have lower predictive performance on complex problems that fail to meet the simplifying assumptions of the algorithms bias. Linear Regression, Naive Bayes algorithm, Linear Discriminant Analysis, Logistic Regression etc...

The error due to variance is taken as the variability of a model prediction for a given data point. Again, imagine you can repeat the entire model building process multiple times. The variance is how much the predictions for a given point vary between different realizations of the model.

![](https://elitedatascience.com/wp-content/uploads/2017/02/noisy-sine-decision-tree.png)

It refers to an error from an estimator being too spefic and learning relationships that are specific to the training set but will not generalize well to new observations, as well. This part is due to the model's excessive sensitivity to small variations in the training data. A model with many degrees of freedom (such as a high-degree polynomial model) is like to have high variance and thus to overfit the training data. Overfitting happens when a model learns not only the actual relationships (signals) in the training data but also any noise that is present, to the extent that it negatively impacts the performance of the model on a new data. This means that the noise or random fluctuations in the training data is picked up and learned as concepts by the model. The problem is that these concepts do not apply to new data and negatively impacts the model ability to generalize. Overfitting occurs when we have a very flexible model (a model which has a high capacity, i.e., it has more power to capture the distribution of the data) which essentially memorizes the training data by fitting it too closely. A flexible model is said to have a high variance because the learned parameters (such as the structure of the decision tree) will vary considerably with the training data. Models with small variance error will not change much if you replace couple of samples in the training set. Models with high variance might be affected even with small changes in the training set. 

Examples of low variance ML algorithms:  Linear Regression, Linear Discriminant Analysis, Logistic Regression etc...

Examples of high variance ML algorithms: Generally, non-parametric ML algorithms that have a lot of flexibility have a high variance, such as, Decision Trees, k-Nearest Neighbors, SVM etc...

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/overfit_underfit_example.png?raw=true)

Irreducible error is the noise term in the true relationship that cannot be fundamentally be reduced by any model and its parameter selection. The only was to reduce this part of the error is to clean up the data (e.g., fix the data sources, such as broken sensors, or detect and remove outliers). In predictive modeling, signal can be thought of as the true underlying pattern one wished to learn from the data. Noise, on the other hand, refers to the irrelevant information or randomness in the dataset. 

The balance between creating a model that is so flexible (high capacity / complex model), it memorizes the training data, versus, an inflexible model (low capacity) that cannot learn the training data is known as bias-variance trade-off and is a fundamental concept in Machine Learning. In other words, as you decrease the variance, you tend to increase the bias. As you decrease the bias, you tend to increase the variance. Generally speaking, the goal is to create models that minimize the overall error by careful model selection and tuning to ensure that there is a balance between bias and variance; general enough to make a good predictions on a new data, but specific enough to pick up as much signal as possible. Given the true model and the infinite data to calibrate it, we should be able to reduce both bias and variance terms to 0. However, in a world with imperfect models and finite data, this is not the case.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/capacity_vs_error.png?raw=true)

#### What is the Bias-variance trade-off for Leave-one-out and k-fold cross validation?

When we perform Leave-One-Out Cross Validation (LOOCV), we are in effect averaging the outputs of $n$ fitted models (assuming we have $n$ observations), each of which is trained on an almost identical set of observations; therefore, these outputs are highly (positively) correlated with each other. In contrast, when we perform $k$-fold CV with $k < n$ are averaging the outputs of $k$ fitted models that are somewhat less correlated with each other, since the overlap between the training sets in each model is smaller. Since the mean of many highly correlated quantities has higher variance than does the mean of many quantities that are not as highly correlated, the test error estimate resulting from LOOCV tends to have higher variance than does the test error estimate resulting from $k$-fold CV (the variance of the sum of correlated variables increases with the amount of covariance, e.g., $Var[X+Y] = Var[X] + Var[Y] + 2Cov[X,Y]$). However, LOOCV estimator is approximately unbiased for the true (expected) prediction error. The	bias	of	the	LOOCV estimator will be small because we will have many training points while estimating.

To summarize, there is a bias-variance trade-off associated with the choice of $k$ in $k$-fold cross-validation.

Note that while two-fold cross validation doesn't have the problem of overlapping training sets, it often also has large variance because the training sets are only half the size of the original sample. A good compromise is ten-fold cross-validation.

Typically, given these considerations, one performs $k$-fold cross-validation with $k=5$ or $k=10,$ as these values have been shown empirically to yield test error rate estimates that suffer neither from excessively high bias nor from very high variance.

#### Describe Machine Learning, Deep Learning, Supervised Learning, Unsupervised Learning, Semi-Supervised Learning, Reinforcement Learning with examples

* **Machine Learning** is the science of getting computers to learn and act like humans do and improve their learning over time in an autonomous fashion, by feeding them data and the information in the form of observations and real-world interactions. It seeks to provide knowledge to computers through data, observations and interacting with the world. That acquired knowledge allows computers to correctly generalize to new settings (can adapt to new data). It is a subset of Artificial Intelligence that uses statistical techniques to give machine the ability to "learn" from data without explicitly given the instructions for how to do so. This process is knows as "training" a "model" using a learning "algorithm" that progressively improves the model performance on a specific task. 

* **Deep Learning** is an area of Machine Learning that attempts to mimic the activity in the layers of neurons in the brain to learn how to recognize the complex patterns in the data. The "deep" in deep learning refers to the large number of layers of neurons. Therefore, we can summarize that Deep Learning is a class of machine learning algorithm that uses multiple stacked layers of processing unitsnto learn high-level representations from structured/unstructured data.

* **Supervised Learning**: Supervised learning is where you have input variables (x) and an output variable (Y) and you use an algorithm to learn the mapping function from the input to the output. The goal is to approximate the mapping function given a set of features called predictors so well that when you have new input data (x) that you can predict the output variables (Y) for that data. It is called supervised learning because the process of an algorithm learning from the training dataset can be thought of as a teacher supervising the learning process. We know the correct answers, the algorithm iteratively makes predictions on the training data and is corrected by the teacher. Learning stops when the algorithm achieves an acceptable level of performance. Supervised learning problems can be further grouped into regression and classification problems. Classification (automatically assigning a label to an unlabelled example) is a type problem where the response variable is qualitative (like a category, such as "red" or "blue" or "disease" and "no disease"). Regression (predicting a real-valued label given an unlabelled example) is another type of problem when the output variable (response variable) is a quantitative (real value, integer or floating point number), such as "dollars" or "weight". Some supervised learning algorithms are Support vector machines, neural networks, linear regression, logistic regression, extreme gradient boosting.

* **Unsupervised Learning**: Unsupervised learning is where you only have input data (X) and no corresponding output variables. The goal for unsupervised learning is to model the underlying structure or distribution in the data in order to learn more about the data. These are called unsupervised learning because unlike supervised learning above there is no correct answers and there is no teacher. Algorithms are left to their own devises to discover and present the interesting structure in the data. Unsupervised learning problems can be further grouped into clustering and association problems. Clustering is used to discover the inherent groupings in the data based on some notation of similarity, such as grouping customers by purchasing behavior, where as an association rule learning problem is to discover rules (interesting relations) that describe large portions of your data, such as people that buy X also tend to buy Y. Some unsupervised learning algorithms are principal component analysis, singular value decomposition; identify group of customers

* **Semi-supervised Learning**: In this type of learning, the dataset contains both labeled and unlabeled examples. Usually, the quantity of unlabeled examples is much higher than the number of labeled examples. Many real world machine learning problems fall into this area. This is because it can be expensive or time-consuming to label data as it may require access to domain experts. Whereas unlabeled data is cheap and easy to collect and store. It could look counter-intuitive that learning could benefit from adding more unlabeled examples. It seems like we add more uncertainty to the problem. However, when you add cheap and abundant unlabeled examples, you add more information about the problem. A larger sample reflects better the probability distribution of the data we labeled came from. Most semi-supervised learning algorithms are combinations of unsupervised and supervised algorithms. For example, deep belief networks are based on unsupervised components called restricted Boltzmann machines (RBMs), stacked on the top of another. RBMs are trained sequentially in an unsupervised manner, and then the whole system is fine-tuned using supervised learning techniques, 

* **Reinforcement Learning**: It is a sub-field of machine learning where the machine "lives" in an environment and is capable of perceiving the state of that environment as a vector of features. The machine can also execute actions in every state. Different actions bring different rewards and could also move the machine to another state of the environment. The goal of a reinforcement learning algorithm is to learn a policy (which is the best strategy). In Reinforcement Learning, we want to develop a learning system (called an _agent_) that can learn how to take actions in the real world. The most common approach is to learn those actions by trying to maximize some kind of reward (or minimize penalties in the form of negative rewards) encouraging the desired state of the environment. For example, many robots implement Reinforcement Learning algorithms to learn how to walk. DeepMind's AlphaGo program is also a good example of Reinforcement Learning.

#### What is batch learning and online learning?

Another criterion used to classify Machine Learning systems is whether or not system can learn incrementally from a stream of incoming data. 

In __Batch Learning__, the system is incapable of learning incrementally, it must be trained using all the available data. This will generally take a lot of time and computing resources, so it is typically done offline. First the system is trained, and then it is launched into production and runs without learning anymore; it just applied what it has learned. This is called _offline learning_.

If you want a batch learning system to know about the new data, you need to train a new version of the system from scratch on the full dataset (not just the new data but also the old data), then stop the old system and replace it with the new one. 

Fortunately, the whole process of training, evaluating and launching a Machine Learning system can be automated fairly easily, so even, a batch learning system can adapt to change. Simply, update the data and train a new version of the system from scratch as often as needed. 

This solution is simple and often works fine but training the full set of data can take many hours so you would typically train a new system only ever 24 hours or even just weekly. If your system needs to adapt to rapidly changing data, then you need a more reactive solution. 

Also training on the full set of data required a lot of computing resources (CPU, memory space, disk space, disk I/0, network I/O etc.). If you have a lot of data and you automate your system to train from scratch every day, it will end up costing you a lot of money. If the amount of data is huge, it may even be impossible to use a batch learning algorithm. 

Finally, if your system needs to be able to learn autonomously, and it has limited resources, then carrying around large amounts of training data and taking up a lot of resources to train for hours everyday is a show-stopped.

Fortunately, a better option in all these cases is to use algorithms that are capable of learning incrementally.

In __online learning__, you train the system incrementally by feeding it data instances sequentially, either individually or by small groups called _mini-batches_. Each learning step is fast and cheap, so the system can learn about new data on the fly, as it arrived. 

Online learning is great for systems that receive the data as continuous flow and need to adapt to change rapidly or autonomously. It is also a good option if you have limited computing resources: once an online learning system has learned about new data instances, it does not need them anymore so you can discard them (unless you want to be able to roll back to a previous state and 'replay' the data). This can save a huge amount of space.

Online learning algorithms can also be used to train systems on huge datasets that cannot fit in one machine's main memory (this is called _out-of-core_ learning). An out-of-core learning algorithm chops the data into mini-batches, runs a training step on that data, then repeats the process until it has run on all of the data.

One important parameter of online learning systems is how fast they should adapt to changing data: this is called _learning rate_. If you set a high learning rate, then your system will rapidly adapt to new data but it will also tend to quickly forget the old data. Conversely, if you set a low learning rate, the system will have more inertia; that is, it will learn more slowly, but it will also be less sensitive to noise in the new data or to to sequences of non-representative data points.

A big challenge with online learning if that if bad data is fed to the system, the system's performance will gradually decline. In order to reduce the risk, you need to monitor the system closely and promptly switch the learning off (and possibly revert to a previously working state) if you detect a drop in performance. You may also want to monitor the input data and react to abnormal data (e.g., using an anomaly detection algorithm).

#### What is instance-based and model-based learning?

Another way to categorize Machine Learning systems is by how they generalize.  Most Machine Learning tasks are about making predictions. This means that given a number of training examples, the system needs to be able to generalize to examples it has never seen before. Having a good performance measure on the training data is good but insufficient. True goal here is to perform well on new instances. There are two main approaches to generalization: instance-based learning and model-based learning.

Instance-based learning simply compares new data points to known data points. It is possibly the most trivial form of learning, it is simply to learn by heart, then generalizes to new cases using a similarity measure. K-nearest neighbor algorithm is a well known instance-based learning algorithm. 

Model-based learning detects patterns in the training data and build a predictive model, much like scientists do. Then we use that model to make predictions on unseen data.

#### What are the main challenges of machine learning algorithms?

* Insufficient quantity of data
* Non-representative training data
* Poor quality data (polluted data full of errors, outliers and noise)
* Irrelevant features (feature engineering, feature selection, feature extraction)
* Overfitting the training data
* Underfitting the training data

#### What are the most important unsupervised learning algorithms?

In supervised learning, the training data is unlabeled. The system tries to learn without a teacher. Here are some of the most important unsupervised learning algorithms:

* Clustering
    - k-means
    - Hierarchial Cluster Analysis
    - Expectation-Maximization
* Visualization and dimensionality reduction
    - Principal Component Analysis
    - Kernel PCA
    - Locally Linear Embedding
    - t-distributed Stochastic Neighbor Embedding (t-SNE)
* Association rule learning
    - Apriori
    - Eclat

#### What is a Machine Learning pipeline?

A machine learning pipeline is a sequence of operations on the dataset. A pipeline can include, among others, such stages as data partitioning, missing data imputations, feature extraction, data augmentation, class imbalance reduction, dimensionality reduction and model building. In practice, when we deploy a model in production, we usually deploy an entire pipeline, Furthermore, an entire pipeline is usually optimized when hyperparameters are tuned. 

#### What is Tensorflow?

Created by the Google Brain team, TensorFlow is  a Python-friendly open source library for numerical computation and large-scale machine learning. TensorFlow bundles together a slew of machine learning and deep learning models and algorithms. It uses Python to provide a convenient front-end API for building applications within the framework, while executing those applications in high-performance C++.

#### Why Deep Learning is important?

In deep learning we want to find a mapping function from inputs to outputs like any other machine learning algorithms. The function to be learned should be expressible as multiple levels of composition of simpler functions where different levels of functions can be viewed as different levels of abstraction. Functions at lower levels of abstraction should be found useful for capturing some simpler aspects of data distribution, so that it is possible to first learn the simpler functions and then compose them to learn more abstract concepts. It is all about learning hierarchical representations: low-level features, mid-level representations, high level concepts. Animals and humans do learn this way with simpler concepts earlier in life, and higher-level abstractions later, expressed in terms of previously learned concepts. 

Deep Learning requires high-end machines contrary to traditional Machine Learning algorithms. GPU/TPU has become a integral part now to execute any Deep Learning algorithm.

In traditional Machine learning techniques, most of the applied features need to be identified by an domain expert in order to reduce the complexity of the data and make patterns more visible to learning algorithms to work. The biggest advantage Deep Learning algorithms are that they try to learn high-level features from data in an incremental manner. This eliminates the need of domain expertise and hard core feature extraction, therefore, less human effort. Most of the Machine Learning algorithms require structured data which are typically organized in a tabular format. However, you can work on unstructured data such as images, audios, videos using Deep Learning algorithms.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/difference_ML_and_DL.png?raw=true)

Another major difference between Deep Learning and Machine Learning technique is the problem solving approach. Deep Learning techniques tend to solve the problem end to end, where as Machine learning techniques need the problem statements to break down to different parts to be solved first and then their results to be combine at final stage.

Usually, a Deep Learning algorithm takes a long time to train due to large number of parameters. 

Interpretability is the main issue why many sectors are still using other Machine Learning techniques over Deep Learning. 

#### What are the three respects of an learning algorithm to be efficient?

We want learning algorithm to be efficient in three main dimensions:

* __computational__: the amount of computing resources required to reach a given level of performance.
* __statistical__: the amount of training data required (especially labeled data) for good generalizations.
* __human involvement__: the amount of human effort (labor) required to tailor the algorithm to a task, i.e., specify the prior knowledge built into the model before training (explicitly or implicitly through engineering designs with a human-in-the-loop). 

#### What are the differences between a parameter and a hyperparameter?

A hyperparameter, also called a tuning parameter,
* is external to the model.
* cannot be estimated from data.
* is often specified by the researcher.
* is often set using heuristics.
* is often tuned for a given predictive modeling problem.
* is often used in a process to help estimate the model parameters.

An example of an hyper parameter is the learning rate for training a neural network.

A parameter 

* is a configuration variable that is internal to the model and whose value is estimated directly from data.
* is often not set manually by the researcher.
* is often saved as a part of the learned model.

An example of a parameter is the weights in a neural network.

Note that if you have to specify a model parameter manually, then it is probably a hyperparameter. 

#### Why do we have three sets: training, validation and test?

In practice, we have three distinct sets of labeled examples:

* training set
* validation set, and
* test set

Once you have your annotated dataset, the first thing you do is to shuffle the examples and split the data set into three subsets. Training set is usually the biggest one, you use it to build model. Validation and test sets are roughly the same sizes, much smaller than the size of the training set. The learning algorithm cannot use these examples from these two subsets to build the model. That is why those two subsets are often called _holdout sets_. 

There is no optimal proportion to split the dataset into three subsets. The reason why we have three sets and not one is because we do not want the model to do well at predicting labels of examples the learning algorithm has already seen. A trivial algorithm that simply memorizes all the training examples and then uses the memory to "predict" their labels will make no mistakes when asked to predict the labels of the training examples but such an algorithm would be useless in practice. What we really want is a model that is good at predicting examples that the learning algorithm did not see: we want good performance on a holdout set.

Why do we need two holdout sets and not one? We use the validation set to (1) to choose the learning algorithm and (2) find the best values of hyperparameters. We use then the test set to assess the model before delivering it to the client or putting it in production. 

#### What are the goals to build a learning machine?

If our goal is to build a learning machine, our research should concentrate on devising learning models with following features:

* A highly flexible way to specify prior knowledge, hence a learning algorithm that can function with a large repertoire of architectures.
* A learning algorithm that can deal with deep architectures, in which a decision involves the manipulation of many intermediate concepts, and multiple levels of non-linear steps.
* A learning algorithm that can handle large families of functions, parameterized with million of individual parameters.
* A learning algorithm that can be trained efficiently even, when the number of training examples becomes very large. This excludes learning algorithms requiring to store and iterate multiple times over the whole training set, or for which the amount of computations per example increases as more examples are seen. This strongly suggest the use of on-line learning.
* A learning algorithm that can discover concepts that can be shared easily among multiple tasks and multiple modalities (multi-task learning), and that can take advantage of large amounts of unlabeled data (semi-supervised learning).

#### What are the solutions of overfitting?

There are several solutions to the problem of overfitting:

* We can try a simpler model because in the case of overfitting, the model might be complex for the dataset, e.g., linear instead of polynomial regression, or SVM with a linear kernel instead of radial basis function, a neural network with fever layers/units.
* We can reduce the dimensionality of the dataset (removing some  irrelevant features, or using one of the dimensionality reduction techniques. Even some algorithms have built-in feature selection.) Adding more input features, or columns (to a fixed number of examples) may increase overfitting because more features may be either irrelevant or redundant and there's more opportunity to complicate the model in order to fit the examples at hand.
* We can add more training data. This should reduce variance. More training data size leads to increase Signal to Noise Ratio (SNR). Increasing SNR means that noise is decreased. When the noise has decreased, the variance of the model will be decreased. Variance has appeared from noise and clean data do not cause variance in model. Adding more data decreases the generalization error (test error) because your model becomes more general by virtue of being trained on more examples. After all, your model has seen a larger part of the population. In one condition!... the new and old data should of course come from the same underlying distribution. If the new data does not come from the same data-generating process, it can make overfitting worse. However, increasing the size of training data will have no effect on bias. More data can even make bias worse - it gives your model the chance to give highly precise, wrong answers.
* We can use data augmentation. Data augmentation is a strategy that enables practitioners to significantly increase the diversity of data available for training models, without actually collecting new data.
* We can try to use early stopping in order to prevent over-training by monitoring model performance. It is probably the most commonly used form of regularization in deep learning. Its popularity is due both to its effectiveness and its simplicity. In the case of neural networks, while the network seems to get better and better, i.e., the error on the training set decreases, at some point during training it actually begins to get worse again, i.e., the error on unseen examples increases. Early stopping may underfit by stopping too early.
* We can use regularization methods. Regularization is any modification we make to a learning algorithm that is intended to reduce its generalization error but not its training error. For example, you could prune a decision tree, use batch normalization/dropout on a neural network, or add a penalty parameter (L1/L2 Regularization) to the cost function in regression.
* We can use Ensembling methods (Bagging and Boosting). Ensembles are machine learning methods for combining predictions from multiple separate models. Bagging uses complex base models and tries to "smooth out" their predictions, while boosting uses simple base models and tries to "boost" their aggregate complexity.
* Cross-validation is a powerful preventative measure against overfitting. Cross-validation simply repeats the experiment multiple times, using all the different parts of the training set as unseen data which we use to validate the model. This gives a more accurate indication of how well the model generalizes to unseen data.  Cross-validation does not prevent overfitting in itself, but it may help in identifying a case of overfitting.
* Transfer learning is a technique in which we take pre-trained weights of a neural net trained on some similar or more comprehensive data and fine tune certain parameters to best solve a more specific problem.

#### Is it better to design robust or accurate algorithms?

* The ultimate goal is to design systems with good generalization capacity, that is, systems that correctly identify patterns in data instances not seen before
* The generalization performance of a learning system strongly depends on the complexity of the model assumed
* If the model is too simple, the system can only capture the actual data regularities in a rough manner. In this case, the system has poor generalization properties and is said to suffer from underfitting
* By contrast, when the model is too complex, the system can identify accidental patterns in the training data that need not be present in the test set. These spurious patterns can be the result of random fluctuations or of measurement errors during the data collection process. In this case, the generalization capacity of the learning system is also poor. The learning system is said to be affected by overfitting
* Spurious patterns, which are only present by accident in the data, tend to have complex forms. This is the idea behind the principle of Occam’s razor for avoiding overfitting: simpler models are preferred if more complex models do not significantly improve the quality of the description for the observations
* Quick response: Occam’s Razor. It depends on the learning task. Choose the right balance
* Ensemble learning can help balancing bias/variance (several weak learners together = strong learner)

#### What are some feature scaling (a.k.a data normalization) techniques? When should you scale your data? Why?

Feature scaling is the method used to standardize the range of features of data. Since the range of values of data may vary widely, it becomes a necessary step in data processing while using ML algorithms. 

* **Min-Max Scaling**: You transform the data such that the features are within a specific range, e.g. [0,1]
     \begin{equation}
         X^{'} = \frac{X- X_{min}}{X_{max} - X_{min}}
     \end{equation}
     where $X^{'}$ is the normalized value. Min-max normalization has one fairly significant downside: it does not handle outliers very well. For example, if you have 99 values between 0 and 40, and one value is 100, then the 99 values will all be transformed to a value between 0 and 0.4. But 100 will be squished into 1, meaning that that data is just as squished as before, still an outlier!
* **Normalization**: The point of normalization is to change your observations so they can be described as a normal distribution.
     \begin{equation}
         X^{'} = \frac{X- X_{mean}}{X_{max} - X_{min}}
     \end{equation}
     All the values will be between 0 and 1. 
* **Standardization**: Standardization (also called z-score normalization) transforms your data such that the resulting distribution has a mean 0 and a standard deviation 1. 
     \begin{equation}
         X^{'} = \frac{X- X_{mean}}{\sigma}
     \end{equation}
     where $X$ is the original feature vector, $X_{mean}$ is the mean of the feature vector, and $\sigma$ is its standard deviation. Z-score normalization is a strategy of normalizing data that avoids the outlier issue of Min-Max Scaling. The only potential downside is that the features aren’t on the exact same scale.
     
 You should scale your data,
 
* when your algorithm will weight each input, e.g. gradient descent used by many neural networks, or use distance metrics, e.g., kNN, model performance can often be improved by normalizing, standardizing, otherwise scaling your data so that each feature is given relatively equal weight.
* It is also important when features are measured in different units, e.g. feature A is measured in inches, feature B is measured in feet, and feature C is measured in dollars, that they are scaled in a way that they are weighted and/or represented equally.
     In some cases, efficacy will not change but perceived feature importance might change, e.g., coefficients in a linear regression.
* Scaling your data typically does not change the performance or feature importance for tree-based models which are not distance based models, since the split points will simply shift to compensate for the scaled data. 

Note that when you standardize all your variables, the intercept will be zero.

#### What are the types of feature selection methods?

 There are three types of feature selection methods
 
* **Filter Methods**: Feature Selection is done independent of the learning algorithm before any modeling is done. One example is finding the correlation between every feature and the target and throwing out those that do not meet a threshold. Easy, fast but naive and not as performant as other methods. Chi-squared test can be used to find the relationship between the categorical response and categorical predictor. We can also use one-way ANOVA with having the response as continuous and predictor as categorical. The Pearson correlation method can be used for numerical variables. 
* **Wrapper Methods**: Train models on subsets of features and use the subset that results in the best performance. Examples are Forward, Backward, Stepwise or Recursive Feature selection. Advantages are that it considers each feature in the context of other features but can be computationally expensive.
* **Embedded Methods**: Learning algorithms have built-in feature selection, e.g., Random forest's feature importance and L1-Regularization.

#### When should you reduce the number of features?

1. When there is strong collinearity between features
2. There are an overwhelming number of features
3. There is not enough computational power to process all features
4. The algorithm forces the model to use all features, even when they are not useful (parametric or linear models)
5. When you wish to make the model simpler for any reasons, e.g., easier to explain, less computational power needed etc.

#### When is feature selection is unnecessary?

1. There are relatively few features
2. All features contain useful and important signal
3. There is no collinearity between features
4. The model will automatically select the most useful features
5. The computing resources can handle processing all of the features
6. Throroughly explaining the model to a non-technical audience is not critical

#### How can you prove that one improvement you've brought to an algorithm is really an improvement over not doing anything?

You can always check the model performance after adding or removing a features, if the performance of model is dropping or improving you can see if the inclusion of that variable makes sense or not. Apart from that, you tweak different built-in model parameters (hyperparameters) like you increase number of trees to grow or number of iterations to do in random forest, you add a regularisation term in linear regression, you change threshold parameters in logistic regression, you assign weights to several algorithms, if you compare the accuracies and other statistics before and after making such change to model, you can understand if these result into any improvement or not.

#### What are the hyperparameter tuning methods?

Hyperparameters are not optimized by the learning algorithm itself. The researcher has to tune the hyperparameters by experimentally finding the best combination o fvalues, one per hyper parameter.

One typical way to do that, when you have enough data to have a decent validation set and the number of hyperparameters and their range is not too large is to use __grid search__.

Grid search is the most simple hyperparameter tuning technique. It builds a model for every combination of hyperparameters specified and evaluates each model. Finally, you keep the model that performs the best according to the metric. Once the best pair of hyperparameters is found, you can try to explore the values close to the best ones in some region around them. Sometimes, this can result in an even better model. Finally you assess the selected model using the test set.

However, trying all combinations of hyperparameters, especially if there are more than a couple of them, could be time-consuming, especially for large datasets. There are more efficient techniques seuch as __random search__ and __Bayesian hyperparameter optimization__. 

Random search differs from grid search in that you no longer provide a discrete set of values to explore for each hyperparameter. Instead, you provide a statistical distribution for each hyperparameter from which values are randomly sampled and set the total number of combinations (number of iterations) you want to try.

Bayesian techniques differ from random or grid search in that they use past evaluation results to choose the next values to evaluate. The idea is to limit the number of expensive optimizations of the objective function by choosing the next hyperparameter values based on those that have done well in the past. 

There are also __gradient-based techniques__, __evolutionary optimization techniques__, and other algorithmic hyperparameter tuning techniques. 

#### How do we use probability in Machine Learning/Deep Learning framework?

1. **Class Membership Requires Predicting a Probability**: Classification predictive modeling problems are those where an example is assigned a given label. Therefore, we model the problem as directly assigning a class label to each observation (hard class classification). A more common approach is to frame the problem as a probabilistic class membership, where the probability of an observation belonging to each known class is predicted (soft class classification. Therefore,  this probability is more explicit for the network. Framing the problem as a prediction of class membership simplifies the modeling problem and makes it easier for a model to learn. It allows the model to capture ambiguity in the data, which allows a process downstream, such as the user to interpret the probabilities in the context of the domain. The probabilities can be transformed into a hard class label by choosing the class with the largest probability. The probabilities can also be scaled or transformed using a probability calibration process.

2. **Some Algorithms Are Designed Using Probability**: There are algorithms that are specifically designed to harness the tools and methods from probability. Naive Bayes, Probabilistic Graphical Models, Bayesian Belief Networks are three of those algorithms. 

3. **Models Can Be Tuned With a Probabilistic Framework**: It is common to tune the hyperparameters of a machine learning model, such as k for kNN or the learning rate in a neural network. Typical approaches include grid searching ranges of hyperparameters or randomly sampling hyperparameter combinations. Bayesian optimization is a more efficient to hyperparameter optimization that involves a directed search of the space of possible configurations based on those configurations that are most likely to result in better performance. As its name suggests, the approach was devised from and harnesses Bayes Theorem when sampling the space of possible configurations.

4. **Models Are Trained Using a Probabilistic Framework**: Many machine learning models are trained using an iterative algorithm designed under a probabilistic framework. Perhaps the most common is the framework of maximum likelihood estimation. This is the framework that underlies the ordinary least squares estimate of a linear regression model. For models that predict class membership, maximum likelihood estimation provides the framework for minimizing the difference or divergence between an observed and predicted probability distribution. This is used in classification algorithms like logistic regression as well as deep learning neural networks. t is common to measure this difference in probability distribution during training using entropy, e.g. via cross-entropy. Entropy, and differences between distributions measured via KL divergence, and cross-entropy are from the field of information theory that directly build upon probability theory. For example, entropy is calculated directly as the negative log of the probability.

5. **Probabilistic Measures Are Used to Evaluate Model Skill**: For those algorithms where a prediction of probabilities is made, evaluation measures are required to summarize the performance of the model, such as AUC-ROC curve along with confusion matrix. Choice and interpretation of these scoring methods require a foundational understanding of probability theory.

#### What are the differences and similarities between Ordinary Least Squares Estimation and Maximum Likelihood Estimation methods?

Ordinary Least Squares (OLS) tries to answer the question "What estimates minimize the squared error of the predicted values from observed?", whereas Maximum Likelihood answers the question "What estimates maximize the likelihood function?". 

The ordinary least squares, or OLS, can also be called the linear least squares. This is a method for approximately determining the unknown parameters located in a linear regression model.

Maximum likelihood estimation, or MLE, is a method used in estimating the parameters, that are most likely to produce observed data, of a statistical model and for fitting a statistical model to data.

From Wikipedia, OLS chooses the parameters of a linear function of a set of explanatory variables by the principle of least squares: minimizing the sum of the squares of the differences between the observed dependent variable (values of the variable being predicted) in the given dataset and those predicted by the linear function. Maximum Likelihood Estimation (MLE) is a method of estimating the parameters of a distribution by maximizing a likelihood function, so that under the assumed statistical model the observed data is most probable. 

The OLS estimator is identical to the maximum likelihood estimator (MLE) under the normality assumption for the error terms

Let’s recall the simple linear regression model: $y_{i} = \alpha + \beta x_{i} + \varepsilon_{i}$ where the noise variables $\varepsilon_{i}$ all have the same expectation (0) and the same constant variance ($\sigma^{2}$), and $Cov[\varepsilon_{i}, \varepsilon_{j}] = 0$ (unless $i = j$, of course). This is a statistical model with two variables $X$ and $Y$, where we try to predict $Y$ from $X$. We also assume that errors follow normal distribution:

$$
f(x; \mu, \sigma^{2}) = \dfrac{1}{\sqrt{2\pi \sigma^{2}}} exp\left\{-\dfrac{(x-\mu)^{2}}{2 \sigma^{2}}  \right\}
$$

We know that $E(y_{i} \mid x_{i}) = \mu_{y_{i}} = \alpha +\beta x_{i}$. The mean of the conditional distribution of $Y$ depends on the value of $X$. Indeed, that's kind of the point of a regression model. We also know that $Var(y_{i} \mid x_{i}) = \sigma_{y_{i}}^{2} = \sigma^{2}$, since $x_{i}$ is a single fixed value.

Let's write the likelihood function for this linear model:

$$
\begin{split}
L(\alpha, \beta) &= \prod_{i=1}^{n} p(y_{i} \mid x_{i};\alpha, \beta) \\
&= \prod_{i=1}^{n}  \dfrac{1}{\sqrt{2\pi \sigma_{y_{i}}^{2}}} exp\left\{-\dfrac{(y_{i}-\mu_{y_{i}})^{2}}{2 \sigma_{y_{i}}^{2}}  \right\}\\
&= \dfrac{1}{\left(2\pi \sigma^{2} \right)^{n/2}} \prod_{i=1}^{n} exp\left\{-\dfrac{(y_{i} - \alpha - \beta x_{i})^{2}}{2 \sigma^{2}}  \right\}\\
& = \dfrac{1}{\left(2\pi \sigma^{2} \right)^{n/2}} exp\left\{- \dfrac{1}{2 \sigma^{2}} \sum_{i=1}^{n} \left(y_{i} - \alpha - \beta x_{i}\right)^{2}\right\}\\
\end{split}
$$

Obviously, maximizing this likelihood is equivalently minimizing,

$$
 \sum_{i=1}^{n} \left(y_{i} - \alpha - \beta x_{i}\right)^{2}
$$

which is nothing but the sum of squares of differences between observed and predicted values. 

#### Do you suggest that treating a categorical variable as continuous variable would result in a better predictive model?

For better predictions, categorical variable can be considered as a continuous variable only when the variable is ordinal in nature.

#### Considering the long list of machine learning algorithm, given a data set, how do you decide which one to use?

In machine learning, there's something called the "No Free Lunch" theorem. In a nutshell, it states that no one algorithm works best for every problem, and it’s especially relevant for supervised learning (i.e. predictive modeling).

Of course, the algorithms you try must be appropriate for your problem, which is where picking the right machine learning task comes in. 

Choosing a machine learning algorithm can be a difficult task. If you have much time, you can try all of them. However, usually the time you have to solve a problem is limited. You can ask yourself several questions before starting to work on the problem. Depending on your answers, you can shortlist some algorithms and try them on your data.

1. **Explainability**: Most very accurate learning algorithms are so-called "black boxes." They learn models that make very few errors, but why a model made a specific prediction could be very hard to understand and even harder to explain. Examples of such models are neural networks or ensemble models. On the other hand, kNN, linear regression, or decision tree learning algorithms produce models that are not always the most accurate, however, the way they make their prediction is very straightforward.

2. **In-memory vs. out-of-memory**: Can your dataset be fully loaded into the RAM of your server or personal computer? If
yes, then you can choose from a wide variety of algorithms. Otherwise, you would prefer incremental learning algorithms that can improve the model by adding more data gradually.

3. **Number of features and examples**: Number of training examples in the dataset and number of features to be handled by the algorithm can be troublesome for some. Some algorithms, including neural networks and gradient boosting, can handle a huge number of examples and millions of features. Others, like SVM, can be very modest in their capacity.

4. **Categorical vs. numerical features**: Depending on the data composed of categorical only, or numerical only features, or a mix of both, some algorithms cannot handle your dataset directly, and you would need to convert your categorical features into numerical ones.

5. **Nonlinearity of the data**: If the data is linearly separable or if it can be be modeled using a linear model, SVM with the linear kernel, logistic or linear regression can be good choices. Otherwise, deep neural networks or ensemble algorithms, might work better. Additionally, if you given to work on unstructured data such as images, audios, videos, then neural network would help you to build a robust model.

6. **Training speed**: Neural networks are known to be slow to train, even with GPU. Simple algorithms like logistic and linear regression or decision trees are much faster. Specialized libraries contain very efficient implementations of some algorithms; you may prefer to do research online to find such libraries. Some algorithms, such as random forests, benefit from the availability of multiple CPU cores because of its parallelizable nature, so their model building time can be significantly reduced on a machine with dozens of cores.

7. **Prediction speed**: The time spent for generating predictions is also considerably important for choosing the algorithm. Algorithms like SVMs, linear and logistic regression, and (some types of) neural networks, are extremely fast at the prediction time. Others, like kNN, ensemble algorithms, and very deep or recurrent neural networks, are slower. 

![](https://scikit-learn.org/stable/_static/ml_map.png)

#### What is selection bias?

Selection bias is the bias introduced by the selection of individuals, groups or data for analysis in such a way that proper randomization is not achieved, thereby ensuring that the sample obtained is not representative of the true population intended to be analyzed. For instance, you select only Asians to perform a study on the world population height. It is sometimes referred to as the selection effect. The phrase "selection bias" most often refers to the distortion of a statistical analysis, resulting from the method of collecting samples. If the selection bias is not taken into account, then some conclusions of the study may be false.

The types of selection bias include:

1. **Sampling bias**: It is a systematic error due to a non-random sample of a population causing some members of the population to be less likely to be included than others resulting in a biased sample.

2. **Time interval**: A trial may be terminated early at an extreme value (often for ethical reasons), but the extreme value is likely to be reached by the variable with the largest variance, even if all variables have a similar mean.

3. **Data**: When specific subsets of data are chosen to support a conclusion or rejection of bad data on arbitrary grounds, instead of according to previously stated or generally agreed criteria.

4. **Attrition**: Attrition bias is a kind of selection bias caused by attrition (loss of participants) discounting trial subjects/tests that did not run to completion.

#### What’s the difference between a generative and discriminative model?

Disriminative models learn the explicit (hard or soft) boundaries between classes (and not necessarily in a probabilistic manner). Generative models learn the distribution of individual classes, therefore, providing a model of how the data is actually generated, in terms of a probabilistic model. (e.g., logistic regression, support vector machines or the perceptron algorithm simply give you a separating decision boundary, but no model of generating synthetic data points). For more details, you can read [this blog post](https://mmuratarat.github.io/2019-08-23/generative-discriminative-models){:target="_blank"}.

#### What cross-validation technique would you use on a time series dataset?

Instead of using standard k-folds cross-validation, you have to pay attention to the fact that a time series is not randomly distributed data — it is inherently ordered by chronological order. When the data are not independent, cross-validation becomes more difficult as leaving out an observation does not remove all the associated information due to the correlations with other observations.

The "canonical" way to do time-series cross-validation is cross-validation on a rolling basis, i.e., "roll" through the dataset. Start with a small subset of data for training purpose, forecast for the later data points and then check the accuracy for the forecasted data points. The same forecasted data points are then included as part of the next training dataset and subsequent data points are forecasted.

To make things intuitive, here is an image for 5-fold CV:

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/cv_for_time_series.png?raw=true)

The forecast accuracy is computed by averaging over the test sets. This procedure is sometimes known as "evaluation on a rolling forecasting origin" because the "origin" at which the forecast is based rolls forward in time. For more details:

1. [https://robjhyndman.com/hyndsight/tscv/](https://robjhyndman.com/hyndsight/tscv/){:target="_blank"}
2. [https://robjhyndman.com/hyndsight/crossvalidation/](https://robjhyndman.com/hyndsight/crossvalidation/){:target="_blank"}

#### What is the difference between "long" and "wide" format data?

In the wide format, a subject’s repeated responses will be in a single row, and each response is in a separate column. In the long format, each row is a one-time point per subject. You can recognize data in wide format by the fact that columns generally represent groups.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/wide_long_format_data.png?raw=true)

#### Can you cite some examples where a false positive is important than a false negative, and where a false negative important than a false positive, and where both false positive and false negatives are equally important?

False Positives are the cases where you wrongly classified a non-event as an event (a.k.a Type I error). False Negatives are the cases where you wrongly classify events as non-events (a.k.a Type II error).

An example of where a false positive is important than a false negative is that in the medical field, assume you have to give chemotherapy to patients. Assume a patient comes to that hospital and he is tested positive for cancer, based on the lab prediction but he actually doesn’t have cancer. This is a case of false positive. Here it is of utmost danger to start chemotherapy on this patient when he actually does not have cancer. In the absence of cancerous cell, chemotherapy will do certain damage to his normal healthy cells and might lead to severe diseases, even cancer

An example of where a false negative important than a false positive is that what if Jury or judge decides to make a criminal go free?

An example of where both false positive and false negatives are equally important is that, in the Banking industry giving loans is the primary source of making money but at the same time if your repayment rate is not good you will not make any profit, rather you will risk huge losses. Banks don't want to lose good customers and at the same point in time, they don’t want to acquire bad customers. In this scenario, both the false positives and false negatives become very important to measure

#### Describe the difference between univariate, bivariate and multivariate analysis?

Univariate analysis is the simplest form of data analysis where the data being analyzed contains only one variable. Since it's a single variable it doesn’t deal with causes or relationships.  The main purpose of univariate analysis is to describe the data and find patterns that exist within it

You can think of the variable as a category that your data falls into. One example of a variable in univariate analysis might be "age". Another might be "height". Univariate analysis would not look at these two variables at the same time, nor would it look at the relationship between them.  

Some ways you can describe patterns found in univariate data include looking at mean, mode, median, range, variance, maximum, minimum, quartiles, and standard deviation. Additionally, some ways you may display univariate data include frequency distribution tables, bar charts, histograms, frequency polygons, and pie charts.

Bivariate analysis is used to find out if there is a relationship between two different variables. Something as simple as creating a scatterplot by plotting one variable against another on a Cartesian plane (think X and Y axis) can sometimes give you a picture of what the data is trying to tell you. If the data seems to fit a line or curve then there is a relationship or correlation between the two variables.  For example, one might choose to plot caloric intake versus weight.

Multivariate analysis is the analysis of three or more variables.  There are many ways to perform multivariate analysis depending on your goals.  Some of these methods include Additive Tree, Canonical Correlation Analysis, Cluster Analysis, Correspondence Analysis / Multiple Correspondence Analysis, Factor Analysis, Generalized Procrustean Analysis, MANOVA, Multidimensional Scaling, Multiple Regression Analysis, Partial Least Square Regression, Principal Component Analysis / Regression / PARAFAC,  and Redundancy Analysis.

#### What is the difference between dummying and one-hot encoding?

Most algorithms (linear regression, logistic regression, neural network, support vector machine, etc.) require some sort of the encoding on categorical variables. This is because most algorithms only take numerical values as inputs. There are two different ways to encoding categorical variables. Say, one categorical variable has $k$ levels (categories). One-hot encoding converts it into $k$ variables (columns), while dummy encoding converts it into $k-1$ variables (columns). 

For unregularized generalized linear models it's usually not a good idea to one-hot encode and not remove one of the variables because of colinearity. By including dummy variable in a regression model however, one should be careful of the _Dummy Variable Trap_.

The Dummy Variable trap is a scenario in which the independent variables are multicollinear - a scenario in which two or more variables are highly correlated; in simple terms one variable can be predicted from the others.

In that case your one-hot encoded design matrix doesn't have full rank, and you cannot invert $\mathbf{X}^{T}\mathbf{X}$ (the Gramian matrix of the design matrix $\mathbf{X}$). This is obviously going to lead to problems because since $\mathbf{X}^{T}\mathbf{X}$ is not invertible, we cannot compute $\hat{\theta}_{OLS} = \left(\mathbf{X}^{T} \cdot \mathbf{X} \right)^{-1} \cdot \mathbf{X}^{T}y$. 

The solution to the dummy variable trap is to drop one of the categorical variables or alternatively, drop the intercept constant, meaning that you have two options: (1) Using $k−1$ indicators plus an intercept term, (2) Using $k$ indicators and no intercept term.

In most applications, and if you regularize your model, this is not an issue. 

Algorithms that do not require an encoding are algorithms that can directly deal with joint discrete distributions such as Markov chain / Naive Bayes / Bayesian network, Tree-based methods, etc.

Even without encoding, distance between data points with discrete variables can be defined, such as Hamming Distance (for categorical variables) or Levenshtein Distance (for strings).

#### What is out-of-core learning?

Consider the problem of learning a linear model: an out-of-core algorithm learns the model without loading the whole data set in memory. It reads and processes the data row by row, updating feature coefficients on the fly. This makes the algorithm very scalable since its memory footprint is independent of the number of rows, which is a very attractive property when dealing with data sets that don't fit in memory.

#### How do you detect outliers in a dataset?

When starting an outlier detection quest you have to answer two important questions about your dataset:

1. Which and how many features am I taking into account to detect outliers ? (univariate / multivariate)
2. Can I assume a distribution(s) of values for my selected features? (parametric / non-parametric)

**1** - **Interquartile Range Method**

Not all data is normal or normal enough to treat it as being drawn from a Gaussian distribution.

A good statistic for summarizing a non-Gaussian distribution sample of data is the Interquartile Range, or IQR for short. This is the simplest, nonparametric outlier detection method in a one dimensional feature space. Here outliers are calculated by means of the IQR (InterQuartile Range), also known as Tukey's fences.

The first and the third quartile (Q1, Q3) are calculated. The first quartile is the median of the data points to the left of the median. The third quartile is the median of the data points to the right of the median. Median is considered to be Q2/50th Percentile.

For example, let's say we have series: 25, 28, 29, 29, 30, 34, 35, 35, 37, 38. Our data is already in order. The median is the average of 5th and 6th observations which is (30+34)/2 = 32. The first quantile is 29 and third quantile is 35

An outlier is then a data point $x_{i}$ that lies outside the interquartile range.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/1_2c21SkzJMf3frPXPAR_gZA.png?raw=true)

{% highlight python %}
# generate gaussian data
from numpy.random import seed
from numpy.random import randn
from numpy import mean
from numpy import std
from numpy import percentile
# seed the random number generator
seed(1)
# generate univariate observations
data = 5 * randn(10000) + 50
# summarize
print('mean=%.3f stdv=%.3f' % (mean(data), std(data)))
#mean=50.049 stdv=4.994
{% endhighlight %}

{% highlight python %}
# calculate interquartile range
q25, q75 = percentile(data, 25), percentile(data, 75)
iqr = q75 - q25

# calculate the outlier cutoff
cut_off = iqr * 1.5
lower, upper = q25 - cut_off, q75 + cut_off

# identify outliers
outliers = [x for x in data if x < lower or x > upper]
{% endhighlight %}

{% highlight python %}
import matplotlib.pyplot as plt 
plt.boxplot(data, vert=0)
plt.savefig('Box_plot_data.png')
{% endhighlight %}

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Box_plot_data.png?raw=true)

##### Why 1.5?

We certainly CAN use whatever outlier bound we wish to use, but we will have to justify it eventually.

The image below is a comparison of a boxplot of a nearly normal distribution and the probability density function (pdf) for a normal distribution

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/1_NRlqiZGQdsIyAu0KzP7LaQ.png?raw=true)

The 3rd quartile (Q3) is positioned at 0.675 SD (standard deviation, sigma) for a normal distribution. The IQR (Q3 - Q1) represents 2 x 0.675 SD = 1.35 SD. The outlier fence is determined by adding Q3 to 1.5 x IQR, i.e., 0.675 SD + 1.5 x 1.35 SD = 2.7 SD. This level would declare 0.7% of the measurements to be outliers.

{% highlight python %}
x = randn(10000)

q25, q50, q75 = percentile(x, 25), percentile(x, 50), percentile(x, 75)

IQR = q75 - q25

w1 = q25 - 1.5* IQR
w2 = q75 + 1.5* IQR

from scipy.stats import norm
p = norm.cdf(w2, loc=0, scale=1) - norm.cdf(w1, loc=0, scale=1)
#0.9931445258100557
{% endhighlight %}

You will see that $p$ is 0.993, so that 99.3% of N(0,1) data are within the whiskers, 0.7% of the measurements are out.

**2** - **Standard Deviation Method**

If we know that the distribution of values in the sample is Gaussian or Gaussian-like, we can use the standard deviation of the sample as a cut-off for identifying outliers. Using the same plot above, three standard deviations from the mean is a common cut-off in practice for identifying outliers in a Gaussian or Gaussian-like distribution. For smaller samples of data, perhaps a value of 2 standard deviations (95%) can be used, and for larger samples, perhaps a value of 4 standard deviations (99.9%) can be used. A value that falls outside of these ranges is part of the distribution, but it is an unlikely or rare event.

{% highlight python %}
data_mean, data_std = mean(data), std(data)
# identify outliers
cut_off = data_std * 3
lower, upper = data_mean - cut_off, data_mean + cut_off

# identify outliers
outliers = [x for x in data if x < lower or x > upper]
{% endhighlight %}

Several methods are used to identify outliers in multivariate datasets: (1) Mahalanobis Distance, (2) Difference in Fits / Cook’s Distance, (3) Isolation Forest, (4)  DBScan (Density Based Spatial Clustering of Applications with Noise) Clustering. It is a clustering algorithm that is used cluster data into groups. It is also used as a density-based anomaly detection method with either single or multi-dimensional data. What this algorithm does is look for areas of high density and assign clusters to them, whereas points in less dense regions are not even included in the clusters (they are labeled as anomalies). Note that you cannot use K-means for outlier detection because all points are fitted into the clusters, so if you have anomalies in the training data these point will belong to the clusters and probably affect their centroids and, specially, the radius of the clusters. Another posibility is that you even form a cluster of anomalies, since there is no lower limit for the number of points in a cluster. If you have no labels (and you probably don’t, otherwise there are better methods than clustering), when new data comes in you could think it belongs to a normal-behavior cluster, when it’s actually a perfectly defined anomaly. However, you may use it in testing phase. Since we have the centroids of the clusters and the shape is expected to be quite regular we just need to compute the boundary distance for each cluster (usually it’s better not to choose the maximum distance to the centroid, in case we have outliers, something like the 95th or 99th percentile should work, depending on your data). Then, for the test data the distance to the centroids is computed. This distance is then compared with the boundary of each cluster, if the point doesn’t belong to any cluster (distance > boundary) it gets classified as an anomaly.

Gaussian mixture models is another approach. It is a probabilistic model that assumes all the data points are generated from a mixture of a finite number of gaussian distributions. The algorithms try to recover the original gaussian that generated this distribution. Once the algorithm it's trained and we get new data we can just pass it to the model and it would give us the probability for that point to belong to the different clusters. Here, a threshold can be set, to say that if the probability is below that value the point should be consider an anomaly.

You can even treat outliers as anomaly and use an Anomaly Detection algorithm. Modified Thompson Tau test is another method used to determine if an outlier exists in a data set. 

One-class classification is another field of machine learning that provides techniques for outlier and anomaly detection.

**Calculation of Cook’s distance**

Cook’s distance is used to estimate the influence of a data point when performing least squares regression analysis. It is one of the standard plots for linear regression in R and provides another example of the applicationof leave-one-out resampling.

$$
D_i = \frac{\sum_{j=1}^n (\hat Y_j - \hat Y_{j(i)})^2}{p\  \text{MSE}}
$$

```python
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

# create data set with outliers
nobs = 100
X = np.random.random((nobs, 2))
X = sm.add_constant(X)
beta = [1, .1, .5]
e = np.random.random(nobs)
y = np.dot(X, beta) + e
y[[7, 29, 78]] *= 3

n = len(X)
model = LinearRegression(fit_intercept=False, normalize = True)
fitted = model.fit(X, y)
yhat = fitted.predict(X)
p = len(fitted.coef_) #number of parameters
mse = np.sum((yhat - y)**2.0)/n
denom = p*mse
idx = np.arange(n)
d = np.array([np.sum((yhat - model.fit(X[idx!=i], y[idx!=i]).predict(X))**2.0) for i in range(n)])/denom
```


#### What is the difference between norm and distance?

A norm is a distance from origin. A distance function (alsp known as a metric) is a distance between two points. 

The distance is a two vectors function $d(x,y)$ ($d: X \times X \longrightarrow \mathbb{R_{+}}$) while the norm is a one vector function $\lVert x \rVert$ ($\lVert \cdot \rVert : X \longrightarrow \mathbb{R_{+}}$), meaning that you can take the norm of _one element_. However, frequently one can use the norm to calculate the distance by means of the difference of two vectors $\lVert x - y \rVert$. So a norm always induces a distance by:

$$
d(x,y) = \lVert x - y \rVert
$$

meaning that all norms can be used to create a distance. 

However, the other way around is not always true, i.e., not all distance functions have a corresponding norm. For a distance to come from a norm, it needs to verify:

$$
d(x,y)=d(x+a, y+a)\quad \text{(translation invariance)}
$$

and 

$$
d(\alpha x, \alpha y) = \lVert \alpha x - \alpha y \rVert =  \lvert \alpha \rvert  \cdot \lVert x - y \rVert = \lvert \alpha \rvert d(x,y)\quad \text{(homogenity)}
$$

These come from the properties of a norm.

For example the Euclidean distance is defined to be:

$$
dist(x,y)=\bigg(\sum_{i=1}^{n}{(x_{i}-y_{i})^2}\bigg)^{\frac{1}{2}}
$$

and the Euclidean Norm is defined to be

$$
\lVert x \rVert = \bigg(\sum_{i=1}^{n}{x_{i}^2}\bigg)^{\frac{1}{2}}
$$

If we take the discrete distance on any space:

$$
d(x,y) = \begin{cases}
1, \text{ if x $\neq$ y}\\
0, \text{ if x $=$ y}
\end{cases}
$$

which can be shown that is translation invariant but not homogenous, e.g. for $\alpha =2$.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/metrics_1.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/metrics_2.png?raw=true)

#### What is Hamming Distance?

It is a distance measure and is used for categorical variables. If the value (x) and the value (y) are the same, the distance will be equal to 0, otherwise it is 1.

$$
D_{H} = \sum_{i=1}^{n} \lvert x_{i} - y_{i} \rvert
$$

$$
\begin{split}
x = y  &\Rightarrow D = 0\\
x \neq y  &\Rightarrow D = 1
\end{split}
$$

{% highlight python %}
import numpy as np
s1 = np.random.randint(0,4,20)
# array([1, 2, 3, 3, 1, 1, 0, 0, 2, 1, 1, 2, 0, 1, 3, 3, 2, 0, 0, 2])

s2 = np.random.randint(0,4,20)
# array([1, 3, 1, 0, 3, 3, 3, 3, 2, 1, 2, 3, 3, 0, 2, 0, 1, 1, 2, 3])

def hamming_distance(s1, s2):
    """Return the Hamming distance between equal-length variables"""
    if len(s1) != len(s2):
        raise ValueError("Undefined for variables of unequal length")
    return sum(el1 != el2 for el1, el2 in zip(s1, s2))

hamming_distance(s1, s2)
#17
{% endhighlight %}

Hamming distance can be seen as Manhattan distance between categorical variables.

#### How to find distance between mixed categorical and numeric data points?

When the data point contains a mixture of numeric and categorical attributes, we can calculate the distance of each group and then treat each measure of distance as a separate dimension (numeric value).

$$
\text{distance final} = \alpha distance_{numeric} + (1- \alpha) distance_{categorical}
$$

#### What is the difference between Mahalanobis distance and Euclidean distance?

Some supervised and unsupervised algorithms, such as k-nearest neighbors and k-means clustering, depend on distance calculations.

**Euclidean distance**

The Euclidean distance is the "ordinary" straight-line distance between two points in Euclidean space. With this distance, Euclidean space becomes a metric space. The associated norm is called the Euclidean norm. Older literature refers to the metric as the Pythagorean metric. A generalized term for the Euclidean norm is the $L_{2}$ norm or $L_{2}$ distance.

$$
d(\vec{x},\vec{y}) = \sqrt{(\vec{x}-\vec{y})^T (\vec{x} - \vec{y})}
$$

where $\vec{x},\vec{y}$ are vectors, representing the features in the data.

In general, for an $n$-dimensional space, the distance is

$$
d(\mathbf{p} ,\mathbf{q}) = \sqrt { (p_{1}-q_{1})^{2} + (p_{2} - q_{2})^{2} + \cdots + (p_{i} - q_{i})^{2} + \cdots +(p_{n}-q_{n})^{2}} = \sqrt{\sum_{i=1}^{n} (p_{i}-q_{i})^{2}}
$$

When using Euclidean distance to compare multiple variables, we need to standardize the data which eliminates units and weights, both measures equally. To do so, we can calculate the z-score for each observation:

$$
z_{i} = \frac{x_{i} - \mu}{\sigma}
$$

where $x_{i}$ is an observation, $\mu$ and $\sigma$ are the mean and standard deviation of the variable, respectively.

**Mahalanobis distance**

Euclidean distance treats each variable as equally important in calculating. An alternative approach is to scale the contribution of the individual variables to the distance value according to the variability of each variable.

Mahalonobis distance is the distance between a point and a distribution. And not between two distinct points. It is effectively a multivariate equivalent of the Euclidean distance. Mahalanobis distance has excellent applications in multivariate anomaly detection, classification on highly imbalanced datasets and one-class classification

The Mahalanobis distance of an observation $\vec {x} = (x_{1}, x_{2}, x_{3}, \dots , x_{n})^{T}$ (row in a dataset) from a set of observations with mean $\vec {\mu } = (\mu_{1}, \mu_{2}, \mu_{3}, \dots ,\mu_{N})^{T}$ (mean of each column) and covariance matrix $S$ (covariance matrix of independent variables) is defined as:

$$
D_{M}(\vec{x}) = \sqrt {(\vec {x} - \vec {\mu})^{T} S^{-1}(\vec {x}- \vec {\mu})}
$$

Let’s take the $(\vec {x} - \vec {\mu})^{T} S^{-1}$ term.

$(\vec {x} - \vec {\mu})$ is essentially the distance of the vector from the mean. We then divide this by the covariance matrix (or multiply by the inverse of the covariance matrix).

If you think about it, this is essentially a multivariate equivalent of the regular standardization ($z_{i} = \frac{x_{i} - \mu}{\sigma}$. That is, z = (x vector) – (mean vector) / (covariance matrix).

So, what is the effect of dividing by the covariance?

If the variables in your dataset are strongly correlated, then, the covariance will be high. Dividing by a large covariance will effectively reduce the distance.

Likewise, if the x’s are not correlated, then the covariance is not high and the distance is not reduced much.

So effectively, it addresses both the problems of scale as well as the correlation of the variables that we talked about in the introduction.

Mahalanobis distance can also be defined as a dissimilarity measure between two random vectors $\vec {x}$ and $\vec {y}$ of the same distribution with the covariance matrix $S$:

$$
d(\vec{x}, \vec{y}) = \sqrt{(\vec{x} - \vec{y})^T S^{-1} (\vec{x} - \vec{y})}
$$

If the covariance matrix is the identity matrix, the Mahalanobis distance reduces to the Euclidean distance. If the covariance matrix is diagonal, then the resulting distance measure is called a standardized Euclidean distance.

Unlike the Euclidean Distance though, the Mahalanobis distance accounts for how correlated the variables are to one another. When two variables are correlated, there is a lot of redundant information in Euclidean distance calculation. By considering the covariance between points in the distance calculation, we remove the redundancy.

The question is which one to use and when. When in doubt, use Mahalanobis distance, because we do not have to standardize the data like we do for Euclidean distance. The covariance matrix calculation takes care of this. Also, it removed the redundant information from correlated variables. Even if your variables are not very correlated, it cannot hurt to use Mahalanobis distance, it will be just quite similar to the results you will get from Euclidean. 

One issue with Mahalanobis distance is that it depends on taking the inverse of the covariance matrix. If this matrix is not invertible, you can calculate Moore-Penrose pseudo inverse to calculate Mahalanobis distance.

{% highlight python %}
import numpy as np
from scipy.spatial import distance

x1 = np.array([2, 0, 0])
#(3,)

x2 = np.array([0, 1, 0])
#(3,)

iv = np.array([[1, 0.5, 0.5], 
      [0.5, 1, 0.5], 
      [0.5, 0.5, 1]])
#(3, 3)

distance.mahalanobis(x1, x2, iv)
# 1.7320508075688772

distance.euclidean(x1, x2)
#2.23606797749979
{% endhighlight %}

#### What is the difference between Support Vector Machines and Logistic Regression?

A support vector machine constructs a hyperplane or set of hyperplanes in a high- or infinite-dimensional space, which can be used for classification, regression, or other tasks. Intuitively, a good separation is achieved by the hyperplane that has the largest distance to the nearest training-data points  (the support vectors. If a point is not a support vector, it doesn’t really matter) of any class (so-called functional margin), since in general the larger the margin the lower the generalization error of the classifier. 

Logistic regression focuses on maximizing the probability of the data. The farther the data lies from the separating hyperplane (on the correct side), the happier LR is. 

Support Vector Machine (SVM) is an algorithm used for classification problems similar to Logistic Regression (LR). LR and SVM with linear Kernel function generally perform comparably in practice.

SVM minimizes hinge loss while logistic regression minimizes logistic loss (also called log loss). 

LR is more sensitive to outliers than SVM because the cost function of LR diverges faster than those of SVM.

Logistic Regression produces probabilistic values,that can be interpreted as confidence in a decision, while SVM produces deterministic values, i.e., -1 or 1 (or 0 or 1) (but we can use Platts model for probability score).

LR gives us an unconstrained, smooth objective, whereas SVM is constrained optimization problem.

LR can be (straightforwardly) used within Bayesian models. 

SVMs have a nice dual form, giving sparse solutions when using the kernel trick (better scalability). 

#### What is the best separating hyperplane?

Some methods find a separating hyperplane, but not the optimal one (e.g., neural net). Besides, we can find multiple separating hyperplane between two classes. However, the best separating hyperplane is  the one that maximizes the distance to the closest data points from both classes. We say it is the hyperplane with maximum margin.

#### What is the optimization problem for Support Vector Machines?

__Hard-Margin Classifier__

In an a training set where the data is linearly separable, and we use a hard margin (no slack allowed) classifier. The support vectors are the points which lie along the supporting hyperplanes (the hyperplanes parallel to the dividing hyperplane at the edges of the margin). This type of SVM is also called maximal margin classifier.

**Primal Problem**:

$$
\begin{array}{l} {\min \limits_{\vec{w},b} \quad \quad \quad \quad \quad \dfrac{1}{2} \left\| \vec{w}\right\| ^{2}} \\[10pt]  {\text{s.t.}\quad \quad y_{i} [(\vec{w}\cdot \vec{x}_{i} )+b]-1\ge 0,\quad i=1,...,n} \end{array}
$$

**Dual Problem**:

$$
\begin{array}{l} {\mathop{max}\limits_{\alpha } \quad \quad \; L_{D} (\vec{w},b,\alpha )=\sum \limits_{i=1}^{n}\alpha _{i}  -\frac{1}{2} \sum \limits_{i,j=1}^{n}\alpha _{i} \alpha _{j}  y_{i} y_{j} \vec{x}_{i}^{T} \vec{x}_{j} } \\[10pt]  {\text{s.t.} \quad \; \quad \quad \; \quad \quad \begin{array}{c} {\alpha _{i} \ge 0,\quad i=1,...,n} \\ {\sum \limits_{i=1}^{n}\alpha _{i} y_{i}  =0} \end{array}} \\ {\quad \; \quad \quad \; \quad \quad \quad \quad \quad \quad } \end{array}
$$

__Soft-Margin Classifier__

The maximal margin classifier is a very natural way to perform classification, is a separating hyperplane exists. However the existence of such a hyperplane may not be guaranteed, or even if it exists, the data is noisy so that maximal margin classifier provides a poor solution. In such cases, the concept can be extended where a hyperplane exists which almost separates the classes, using what is known as a soft margin. The generalization of the maximal margin classifier to the non-separable case is known as the support vector classifier, where a small proportion of the training sample is allowed to cross the margins or even the separating hyperplane. Rather than looking for the largest possible margin so that every observation is on the correct side of the margin, thereby making the margins very narrow or non-existent, some observations are allowed to be on the incorrect side of the margins.

In soft-margin SVM, We use the slack parameter C to control this. This gives us a wider margin and greater error on the training dataset, but improves generalization and/or allows us to find a linear separation of data that is not linearly separable.

**Primal Problem**:

$$
\begin{array}{l} {\min \limits_{\vec{w},b,\xi } \quad \quad \quad \quad \quad \frac{1}{2} \left\| \vec{w}\right\| ^{2} +C\sum \limits_{i=1}^{n}\xi _{i}  } \\[10pt]  {\text{s.t.}\quad \quad \begin{array}{c} {y_{i} \left[(\vec{w}\cdot \vec{x}_{i} )+b\right]-1+\xi _{i} \ge 0,\quad i=1,2,...,n} \\ {\; \xi _{i} \ge 0,\quad \forall i} \end{array}} \\ {\quad \quad \quad \quad \; \; } \end{array}
$$

**Dual Problem**:

$$
\begin{array}{l} {\mathop{max} \limits_{\alpha } \quad \quad \; L_{D} (\vec{w},b,\xi ,\alpha ,\mu )=\sum \limits_{i=1}^{n}\alpha _{i}  -\frac{1}{2} \sum \limits_{i,j=1}^{n}\alpha _{i} \alpha _{j}  y_{i} y_{j} \vec{x}_{i}^{T} \vec{x}_{j} } \\[10pt]  {\text{s.t.}\quad \; \quad \quad \; \quad \quad \begin{array}{c} {0\le \alpha _{i} \le C,\quad i=1,...,n} \\ {\sum \limits_{i=1}^{n}\alpha _{i} y_{i}  =0} \end{array}} \\ {\quad \; \quad \quad \; \quad \quad \quad \quad \quad \quad } \end{array}
$$

__Non-linear Classifiers__

**Hard-Margin Classifier**

$$
\begin{array}{l} {\mathop{max}\limits_{\alpha } \quad \quad \; L_{D} (\vec{w},b,\alpha )=\sum \limits_{i=1}^{n}\alpha _{i}  -\frac{1}{2} \sum \limits_{i,j=1}^{n}\alpha _{i} \alpha _{j}  y_{i} y_{j} \varphi (\vec{x}_{i} )^{T} \varphi (\vec{x}_{j} )} \\ {\quad \quad \quad \quad \quad \quad \quad \quad \quad =\sum \limits_{i=1}^{n}\alpha _{i}  -\frac{1}{2} \sum \limits_{i,j=1}^{n}\alpha _{i} \alpha _{j}  y_{i} y_{j} K(\vec{x}_{i} ,\vec{x}_{j} )} \\[7pt] {\text{s.t.}\quad \; \quad \quad \; \quad \quad \begin{array}{c} {\alpha _{i} \ge 0,\quad i=1,...,n} \\ {\sum \limits_{i=1}^{n}\alpha _{i} y_{i}  =0} \end{array}} \\ {\quad \; \quad \quad \; \quad \quad \quad \quad \quad \quad } \end{array}
$$

**Soft-Margin Classifier**

$$
\begin{array}{l} {\mathop{max}\limits_{\alpha } \quad \quad \; L_{D} (\vec{w},b,\xi ,\alpha ,\mu )=\sum \limits_{i=1}^{n}\alpha _{i}  -\frac{1}{2} \sum \limits_{i,j=1}^{n}\alpha _{i} \alpha _{j}  y_{i} y_{j} \varphi (\vec{x}_{i} )^{T} \varphi (\vec{x}_{j} )} \\ {\quad \quad \quad \quad \quad \quad \quad \quad \quad \; \; \; \quad =\sum \limits_{i=1}^{n}\alpha _{i}  -\frac{1}{2} \sum \limits_{i,j=1}^{n}\alpha _{i} \alpha _{j}  y_{i} y_{j} K(\vec{x}_{i} ,\vec{x}_{j} )} \\[7pt] {\text{s.t.}\quad \; \quad \quad \; \quad \quad \begin{array}{c} {0\le \alpha _{i} \le C,\quad i=1,...,n} \\ {\sum \limits_{i=1}^{n}\alpha _{i} y_{i}  =0} \end{array}} \\ {\quad \; \quad \quad \; \quad \quad \quad \quad \quad \quad } \end{array}
$$

#### What does the parameter C do in SVM?

$C$ parameter in soft-margin SVM is essentially NOT a regularisation parameter, but it controls the trade-off between classifying the training data well (minimizing the training error) and classifying the future examples (minimizing the testing error, in other words, to generalize your classifier to unseen data). If we have a large $C$, we prefer small number of misclassified examples because the term in the objective function of SVM, $C\sum_{i=1}^{n} \varepsilon_{i}$, will dominate, so, we will have a smaller margin. This is basically trying to fit the model exactly to training data, which can have tendency to overfit. Conversely, if we have a small $C$, the regularization term in the objective function of SVM, $\frac{\lVert w \rVert^{2}}{2}$, will dominate. So, we will have a larger margin but will allow potentially larger number of misclassified example (ignore those data points/underfit the data/larger training error). 

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/0_z00-0ici9ikQLBug.jpg?raw=true)

Therefore tuning C correctly is a vital step in best practice in the use of SVMs. $C$ parameter can be related to the "regular" regularization tradeoff in the following way. SVMs are usually formulated like

$$
\min_{w} \mathrm{regularization}(w) + C \, \mathrm{loss}(w; X, y)
$$

whereas ridge regression / LASSO / etc are formulated like:

$$
\min_{w} \mathrm{loss}(w; X, y) + \lambda \, \mathrm{regularization}(w)
$$

The two are of course equivalent with $C= \frac{1}{\lambda}$. As we increase $\lambda \to \infty$ we introduce more bias into regression model, therefore, a small $C$ will have a high bias, low variance, underfitting the data. As we decrease $\lambda$, we increase the variance in the model and lowering the bias, therefore, $C$ will go larger, so we will have a high variance, low bias SVM, which can overfit the data. 

Nonetheless, if you set $C=0$, then SVM will ignore the errors, and just try to minimise the sum of squares of the weights ($w$), perhaps you may get completely different results on the test set.

The theory for determining how to set $C$ is not very well developed at the moment, so it is recommended that a "grid-search" on $C$ and also $\gamma$ (if an RBF kernel is used) using cross-validation. Various pairs of $(C,\gamma)$ values are tried and the one with the best cross-validation accuracy is picked. We found that trying exponentially growing sequences of $C$ and $\gamma$ is a practical method to identify good parameters (for example, $C = 2^{-5},2^{-3},\ldots,2^{15};\gamma = 2^{-15},2^{-13},\ldots,2^{3}$).

#### Why do we find the dual problem when fitting SVM?

Solving the primal problem, we obtain the optimal $w$. In order to classify a query point $x$ we need to explicitly compute the scalar product $w^{T}x$, which may be expensive if the number of features in the data is large.

Solving the dual problem, we obtain the $\alpha_{i}$ (where $\alpha_{i}=0$ for all but a few points - the support vectors). In order to classify a query point $x$, we calculate

$$
w^{T}x + b = \left(\sum_{i=1}^{n}{\alpha_i y_i x_i} \right)^{T} x + b = \sum_{i=1}^{n}{\alpha_i y_i \langle x_i, x \rangle} + b
$$

This term is very efficiently calculated if there are only few support vectors (which there is often only a small number due to the sparse solution of SVM). Further, since we now have a scalar product only involving data vectors, we may apply the kernel trick which is the most important reasoning.

There are some algorithms like SMO(Sequential Minimal Optimization) solves the dual problem efficiently.

#### What is the output of Support Vector Machines?

A trained Support Vector Machine has a scoring function which computes a score for a new input, $sign[(w \cdot x_{i})+ b]$ This score tells us on which side of the hyperplane generated by the classifier we are (and how far we are away from it). A Support Vector Machine is a binary (two class) classifier; if the output of the scoring function is negative then the input is classified as belonging to class y = $-1$. If the score is positive, the input is classified as belonging to class $y = 1$.

#### What are the support vectors in Support Vector Machines?

Support vector machines are maximum-margin classifiers, which means they find the hyperplane that has the largest perpendicular distance between the hyperplane and the closest samples on either side.  The closest samples on either side are the support vectors. They influence the position and orientation of the hyperplane. Using these support vectors, we maximize the margin of the classifier. Deleting the support vectors will change the position of the hyperplane. These are the points that help us find the solution which introduces sparsity into network

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/support_vectors.png?raw=true)

#### What is the Kernel Trick?

The Kernel Trick is a mathematical technique that implicitly maps instances into a very high dimensional space (feature space), enabling nonlinear classification and regression with Support Vector Machines because a linear decision boundary in the high-dimensional feature space corresponds to a complex nonlinear decision boundary in the original space. In essence, what the kernel trick does for us is to offer a more efficient and less expensive way to transform data into higher dimensions.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Screen%20Shot%202020-02-09%20at%2021.20.29.png?raw=true)

Let's see a visual example. Suppose data is one-dimensional and they are all located on this axis

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Screen%20Shot%202020-05-18%20at%2009.41.43.png?raw=true)

Let's create a mapping function on-the-fly. Let's first choose a mapping function $f = x - 5$. It will shift the points to the left on the axis:

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Screen%20Shot%202020-05-18%20at%2009.44.01.png?raw=true)

Next step, let's square the points:

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Screen%20Shot%202020-05-18%20at%2009.44.46.png?raw=true)

Now, what we want to do is to see that these points are indeed linearly separable!

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Screen%20Shot%202020-05-18%20at%2009.44.53.png?raw=true)

There, you go!

For example, let's say we have two data points: $\mathbf x = (x_1, x_2)$ and $\mathbf y = (y_1, y_2)$. Let's move from 2-dimensional space to 6-dimensional space. So we have to compute $(1, x_1^2, x_2^2, \sqrt{2} x_1, \sqrt{2} x_2, \sqrt{2} x_1 x_2)$ and $(1, y_1^2, y_2^2, \sqrt{2} y_1, \sqrt{2} y_2, \sqrt{2} y_1 y_2)$. The dot product between these two vectors will be $1 + x_1^2 y_1^2 + x_2^2 y_2^2 + 2 x_1 y_1 + 2 x_2 y_2 + 2 x_1 x_2 y_1 y_2$. This is nothing but $(1 + x_1 \, y_1  + x_2 \, y_2)^2$. So, $k(\mathbf x, \mathbf y) = (1 + \mathbf x^T \mathbf y)^2 = \varphi(\mathbf x)^T \varphi(\mathbf y)$ computes a dot product in 6-dimensional space without explicitly visiting this space.

The kernel is effectively a distance and if different features vary on different scales then it is often recommended to do feature scaling (e.g. by normalization) when using a Support Vector Machines. 

#### How does SVM work in multiclass classification?

The SVM as defined so far works for binary classification. What happens if the number of classes is more than two?

* **One-versus-All**: If the number of classes is $K > 2$ then $K$ different 2-class SVM classifiers are fitted where one class is compared with the rest of the classes combined. A new observation is classified according to where the classifier value is the largest.

* **One-versus-One**: All ${K\choose 2}$ pairwise classifiers are fitted and a test observation is classified in the class which wins in the majority of the cases.

The latter method is preferable but if $K$ is too large, the former is to be used.

#### How does Kernel change the data?

Kernel is a way of computing the dot product of two vectors $\mathbf{x}$ and $\mathbf{y}$ in some (possibly very high dimensional) feature space, which is why kernel functions are sometimes called *generalized dot product*. Suppose we have a mapping $\varphi \, : \, \mathbb R^n \to \mathbb R^m$ that brings our vectors in $R^n$ to some feature space $R^m$. Then the dot product of $\mathbf{x}$ and $\mathbf{y}$ in this space is $\varphi(\mathbf x)^T \varphi(\mathbf y)$. A kernel is a function $k$ that corresponds to this dot product, i.e. $k(\mathbf x, \mathbf y) = \varphi(\mathbf x)^T \varphi(\mathbf y)$. A very simple and intuitive way of thinking about kernels (at least for SVMs) is a similarity function. The arguably simplest example is the linear kernel ($K(\mathbf{x}, \mathbf{y}) = < \mathbf{x}, \mathbf{y} >$, also called dot-product. Given two vectors, the similarity is the length of the projection of one vector on another. Another interesting kernel example is Gaussian kernel. Given two vectors, the similarity will diminish with the radius of $\sigma$. The distance between two objects is "reweighted" by this radius parameter. Note that in general, kernels are not interpretable in the original input space. 

Choosing the proper kernel is still an open research area. You have to examine different kernel functions and find the best one which produce less error.

#### What is the bias-variance tradeoff for sigma parameter in RBF kernel?

The RBF kernel on two samples $x$ and $x^{\prime}$, represented as feature vectors in some input space, is defined as:

$$
K(\mathbf {x} ,\mathbf{x^{\prime}} )=\exp \left(- \frac{\left\Vert \mathbf{x} -\mathbf{x^{\prime}} \right\Vert^{2}}{2\sigma ^{2}} \right)
$$

$\left\Vert \mathbf{x} - \mathbf{x^{\prime}} \right\Vert^{2}$ may be recognized as the squared Euclidean distance between the two feature vectors. $\sigma$  is a free parameter. An equivalent definition involves a parameter $\gamma = \dfrac {1}{2\sigma ^{2}}$:

$$
K(\mathbf {x} ,\mathbf {x^{\prime}} )=\exp(- \gamma ||\mathbf {x} - \mathbf {x^{\prime}} ||^{2})
$$

Since the value of the RBF kernel decreases with distance and ranges between zero (in the limit) and one (when $x = x^{\prime}$), it has a ready interpretation as a similarity measure.

We can see the effect of $\sigma$ with a small example. Here, though, we will be using reciprocal of $\sigma$, which is $\gamma$ because Scikit-learn SVM function uses scale parameter. Let's simulate some example and see the original data:

```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.datasets import make_circles
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions

X_train, y_train = make_circles(n_samples=300,
    shuffle=True,
    noise=0.3,
    random_state=42,
    factor=0.2)

plt.figure(figsize = (20, 10))
plt.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], marker='s', color='red', edgecolor='k', alpha=0.6, s=25)
plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], marker='^', color='blue', edgecolor='k', alpha=0.6, s=25)
plt.xlabel('X1', fontsize=14.5)
plt.ylabel('X2', fontsize=14.5)
plt.tight_layout()
plt.savefig('original_dataset')
plt.show()
```

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/original_dataset.png?raw=true)

```python
param_C = 1.0
gammas = [0.001, 0.1, 10]

for i in gammas:
    classifier = SVC(C=param_C,
    kernel='rbf',
    gamma=i,
    coef0=0.0,
    shrinking=True,
    probability=False,
    tol=0.001,
    cache_size=200,
    class_weight=None,
    verbose=False,
    max_iter=-1,
    decision_function_shape=None,
    break_ties=False,
    random_state=None)

    classifier.fit(X_train, y_train)
    
    plt.figure(figsize = (20, 10))
    plot_decision_regions(X= X_train, 
                          y = y_train,
                          clf=classifier,
                          res=0.02,
                          legend=None)
    plt.title('$\gamma = $' + str(i))
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.savefig('gamma_' + str(i) + '.png')
    plt.show()
```

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/gamma_0.001.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/gamma_0.1.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/gamma_10.png?raw=true)

We can see that for a small $\gamma$ (large $\sigma$), the decision tends to be flexible and smooth, that is too simple to separate the two classes. In other words, it tends to make wrong classification while predicting, but avoids the hazard of overfitting. 

For a larger $\gamma$ (smaller $\sigma$), the decision boundary tends to be strict and sharp (fairly complex decision boundary), in contrast to the former situation, it tends to overfit (memorizing the training set, cannot be generalizable).

If the distance between $x$ and $x^{\prime}$ is much larger than sigma, the kernel function tends to be zero. Thus, if the sigma is very small, only the $x$ within the certain distance can affect the predicting point. In other words, smaller sigma tends to make a local classifier, larger sigma tends to make a much more general classifier. 

As for the variance and bias explanation, smaller sigma (larger gamma) tends to be less bias and more variance while larger sigma (smaller gamma) tends to be less variance and more bias.

#### What is the output of Logistic Regression?

It gives posterior probability $P(y = 1 \mid x)$. Because of the sigmoid function, outputs are bounded between 0 and 1 which can be interpreted as probability.

#### Can you interpret probabilistically the output of a Support Vector Machine?

SVMs don’t output probabilities natively, but probability calibration methods can be used to convert the output to class probabilities. Various methods exist, including Platt scaling (particularly suitable for SVMs) and isotonic regression. For more details, look [here](https://mmuratarat.github.io/2019-10-12/probabilistic-output-of-svm){:target="_blank"}.

#### What are the advantages and disadvantages of Support Vector Machines?

**Advantages**

1. SVM works relatively well when there is clear margin of separation between classes.

3. SVM can efficiently handle non-linear data using Kernel trick. The feature mapping is implicitly carried out via simple dot products. It is also possible to specify custom kernels, instead of using common kernels in the literature.

4. SVM is effective in cases where number of dimensions is greater than the number of samples (high dimensional data).

5. SVM can be used to solve both classification and regression problems. SVM is used for classification problems while SVR (Support Vector Regression) is used for regression problems.

6. SVM works well with even unstructured and semi-structured data like text and images.

7. SVM results in sparsity. We are using only a subset of data points (support vectors) for the decision function (the sparseness of the solution).

8. SVM can control overfitting and underfitting (Using $C$ hyperparameter - the capacity control obtained by optimising the margin).

9. Due to the nature of convex optimization, the solution is guaranteed to be the global minimum not a local minimum (the absence of local minima).

10. Training a support vector machine involves solving a convex optimization problem defined with the hinge loss function. Due to convexity of the problem, the choice of optimization algorithm has no influence on the classifier obtained at the end of training.

11. It is useful for both linearly seperable (hard-margin classifier) and non-linearly seperable (soft margin classifier) data. The only thing to do is to come up with the optimal penalty variable C (the hyper-parameter that multiplies slack variables).

**Disadvantages**

1. Perhaps the biggest limitation of the support vector approach lies in choice of the kernel. Choosing an appropriate kernel function (to handle the non-linear data) is not an easy task. It could be tricky and complex. In case of using a high dimension kernel, you might generate too many support vectors which reduce the training speed drastically. 

2. Extensive memory requirement: Algorithmic complexity and memory requirements of SVM are very high. You need a lot of memory since you have to store all the support vectors in the memory and this number grows abruptly with the training dataset size.

3. The optimal design for multiclass SVM classifiers is a further area for research.

4. SVM does not perform very well, when the data set has more noise, i.e. target classes are overlapping.

5. SVMs do not directly provide probability estimates.

6. It is not that easy to fine-tune the hyper-parameters of SVM, such as the selection of the kernel function parameters - for Gaussian kernels the width parameter [sigma] - and the value of [epsilon] in the [epsilon]-insensitive loss function, that will allow for sufficient generalization performance. It is hard to visualize their impact.

7. SVMs are difficult to interpret: SVM model is difficult to understand and interpret by human beings unlike Decision Trees.

8. Support Vector Machine algorithms are not scale invariant, so it is highly recommended to scale your data. One must do feature scaling of variables before applying SVM. Because SVM constructs a hyperplane such that it has the largest distance to the nearest data points (called support vectors). If the dimensions have different ranges, the dimension with much bigger range of values influences the distance more than other dimensions. So its necessary to scale the features such that all the features have similar influence when calculating the distance to construct a hyperplane. Another advantage is to avoid the numerical difficulties during the calculation. Because kernel values usually depend on inner products of features vectors, large attribute values might cause numerical problems. Note that the same scaling must be applied to the test vector to obtain meaningful results.

9. SVM algorithm is not suitable for large data sets.

10. Discrete data presents another problem in SVMs.

11. Another limitation is speed and size, both in training and testing. SVM takes a long training time on large datasets. They can also be abysmally slow in test phase, although SVMs have good generalization performance.

#### What is a parsimonious model?

Parsimonious models are simple models with the least assumptions but with the greatest explanatory predictive power. They explain data with a minimum number of parameters, or predictor variables.

The idea behind parsimonious models stems from Occam’s razor (law of parsimony - Entities should not be multiplied unnecessarily). The most useful statement of the principle for scientists is "when you have two competing theories that make exactly the same predictions, the simpler one is the better". The law states that you should use no more "things" than necessary; In the case of parsimonious models, those "things" are parameters. Parsimonious models have optimal parsimony, or just the right amount of predictors needed to explain the model well.

There is generally a tradeoff between goodness of fit and parsimony: low parsimony models (i.e. models with many parameters) tend to have a better fit than high parsimony models. This is not usually a good thing; adding more parameters usually results in a good model fit for the data at hand, but that same model will likely be useless for predicting other data sets (generalization). Finding the right balance between parsimony and goodness of fit can be challenging. Popular methods include Akaike’s Information Criterion (AIC), Bayesian Information Criterion (BIC),  Mallow's Cp criteria, Bozdogan’s index of informational complexity (ICOMP), Bayes Factors and Minimum Description Length. The best model minimizes those critera.

#### How do you deal with imbalanced data?

Imbalanced classes are a common problem in machine learning classification where there are a disproportionate ratio of observations in each class. Class imbalance can be found in many different areas including medical diagnosis, spam filtering, and fraud detection. 

For example, you may have a 2-class (binary) classification problem with 100 instances (rows). A total of 80 instances are labeled with Class-1 and the remaining 20 instances are labeled with Class-2. This is an imbalanced dataset and the ratio of Class-1 to Class-2 instances is 80:20 or more concisely 4:1. In this case, examples of some class will be underrepresented in the training data. You can have a class imbalance problem on two-class classification problems as well as multi-class classification problems. Most techniques can be used on either.

**Buy or collect more data**

The first solution that comes to the mind is to buy or collect data. If collecting data is expensive and time-consuming, one can try other methods. 

**Use the right evaluation metrics**

Applying inappropriate evaluation metrics for model generated using imbalanced data can be dangerous. For example, if accuracy is used to measure the goodness of a model, a model which classifies all testing samples into "0" will have an excellent accuracy (99.8%), but obviously, this model won't provide any valuable information for us. This situation is called _accuracy paradox_. It is the case where your accuracy measures tell the story that you have excellent accuracy (such as 90%), but the accuracy is only reflecting the underlying class distribution. The Area Under the ROC curve (AUC), Precision/Recall, F1 Score, Matthews Correlation Coefficient (is equivalent to Karl Pearson's phi coefficient) and/or Cohen’s Kappa are other metrics to be used. 

**Cost sensitive learning:**

At the algorithm level, or after it, you can adjust the class weight (misclassification costs), you can adjust the decision threshold (Look at performance curves and decide for yourself what threshold to use), you can modify an existing algorithm to be more sensitive to rare classes (cost-sensitive training models require a custom loss function) or you can construct an entirely new algorithm to perform well on imbalanced data.

One of the solutions for imbalanced data is to weight the classes. Normally, each example and class in our loss function will carry equal weight. For example, if you use SVM with soft margin, you can define a cost for misclassified example because noise is always present in the training data, there are high chances, that many examples would end up in the wrong side of the decision boundary contributing to the cost. The SVM will try to move the hyperplane to avoid as much as possible misclassified examples. The minority class examples risk being misclassified in order to classify more numerous examples of majority class correctly. However, if you set the cost of misclassification of examples of the minority class higher, then the model will try to harder to avoid misclassifying those examples, obviously for the cost of misclassification of some examples of the majority class.  Some SVM implementations allow you to provide weights for every class. The learning algorithm takes this information into account when looking for the best hyperplane. Scikit-learn, for example, has many classifiers that take an optional `class_weight` parameter that can be set higher than one. Specifically, the `class_weight = balanced` argument will automatically weigh classes inversely proportional to their frequency as `n_samples / (n_classes * np.bincount(y)`

**Resample the training set**

Selecting the proper class weights can sometimes be complicated (doing a simple inverse-frequency might not always work very well) or a learning algorithm might not allow weighting classes. In these case, you can try resampling techniques. Two approaches to make a balanced dataset out of an imbalanced one are under-sampling and over-sampling randomly.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/resampling_unbalaned_classes.png?raw=true)

Undersampling means we will select only some of the data from the majority class, only using as many examples as the minority class has. This selection should be done to maintain the probability distribution of the class. Random under-sampling balances the dataset by reducing the size of the abundant class. This method is used when quantity of data is sufficient (tens or hundreds of thousands of instances or more). By keeping all samples in the rare class and randomly selecting an equal number of samples in the abundant class, a balanced new dataset can be retrieved for further modelling. This selection should be done to maintain the probability distribution of the majority class

On the contrary, random oversampling is used when the quantity of data is insufficient (tens of thousands of records or less). It tries to balance dataset by increasing the size of rare samples. The copies will be made such that the distribution of the minority class is maintained.

You can consider the following factors while thinking of applying under-sampling and over-sampling: (1) You can consider testing random and non-random (e.g. stratified) sampling schemes. (2) You can consider testing different resampled ratios (e.g. you don't have to target a 1:1 ratio in a binary classification problem, try other ratios)

Note that there is no absolute advantage of one resampling method over another. Application of these two methods depends on the use case it applies to and the dataset itself. A combination of over- and under-sampling is often successful as well.

However, in practice, these simple sampling approaches have flaws. Unlike undersampling, oversampling method leads to no information loss. Oversampling the minority can lead to model overfitting, since it will introduce duplicate instances (it makes variables appear to have lower variance than they do), drawing from a pool of instances that is already small. Similarly, undersampling the majority can end up leaving out important instances that provide important differences between the two classes. Hence, it might discard useful information. Undersampling can make the independent variables look like they have a higher variance than they do. The sample chosen by undersampling may be a biased sample. And it will not be an accurate representation of the population in that case. Therefore, it can cause the classifier to perform poorly on real unseen data.

Note that you must always split into test and train sets BEFORE trying oversampling  (or undersampling)techniques! Oversampling before splitting the data can allow the exact same observations to be present in both the test and train sets. This can allow our model to simply memorize specific data points and cause overfitting and poor generalization to the test data.

**K-fold Cross-Validation**

Do not also forget to use `StratifiedKFold` in an imbalanced data situation. `StratifiedKFold` is a variation of k-fold cross-validation which returns stratified folds. It is a proper cross validate approach. It will keep the distribution of classes in each of the folds.

**Data Generation**

There also exist more powerful sampling methods that go beyond simple oversampling or undersampling. Rather than getting rid of abundant samples, new rare samples can be generated (Data Augmentation) by using e.g. repetition, bootstrapping, SMOTE (Synthetic Minority Over-Sampling Technique) which is an oversampling method. Many modifications and extensions have been made to the SMOTE method ever since its proposal. SMOTE first considers the K nearest neighbors of the minority instances. It then constructs feature space vectors between these K neighbors, generating new synthetic data points on the lines.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/ImbalancedClasses_fig11.png?raw=true)

SMOTE is implemented in Python using the `imblearn` library. Details can be found [here](https://imbalanced-learn.readthedocs.io/en/stable/over_sampling.html#mathematical-formulation){:target="_blank"}.

Very similar to SMOTE, ADYSYN (Adaptive Synthetic Sampling Method) also creates synthetic data points with feature space vectors. However, for the new data points to be realistic, ADYSYN adds a small error to the data points to allow for some variance. This is because observations are not perfectly correlated in real life.
 
Instead of relying on random samples to cover the variety of the training samples, we can also cluster abundant classes. Let n be number of samples in the rare class. Cluster the abundant class into n clusters, and use the resulting cluster mediods/means as the training data for the abundant class. To be clear, you throw out the original training data from the abundant class, and use the mediods instead. Now your classes are balanced! But your dataset is much smaller, so that might be an issue.

You could even use a method like Naive Bayes that can sample each attribute independently when run in reverse.

A recent study shows that the combination of undersampling / oversampling with ensemble learning can achieve better results. Although you are undersampling the majority class for each individual model, as long as you build enough models, you'll be able to fully sample the training data.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/blagging.png?raw=true)

This technique has not been implemented in Scikit-learn, though a file called `blagging.py` (balanced bagging) is available that implements a BlaggingClassifier, which balances bootstrapped samples prior to aggregation. https://github.com/yanshanjing/learning-from-imbalanced-classes/blob/master/blagging.py

Another similar approach might be applied as can be seen [here](http://francescopochetti.com/extreme-label-imbalance-when-you-measure-the-minority-class-in-basis-points/){:target="_blank"}.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Screen%20Shot%202019-11-01%20at%2011.26.02.png?raw=true)

```python
def train_ensemble(est_name, n_split, oversample=True):
    """
    train_ensemble splits the train set corresponding to the
    majority class (0 in this case) in n_split random chunks.
    It then builds n_split new train sets concatenating each chunk
    with train set corresponding to the minority class (1 in this case).
    It then applies a random sampling strategy, among 7 possible,
    to each of the n_split new train sets and trains an est_name 
    classifier on top of it.
    It reurns a list of n_split classifiers.
    """
    pos = y_train[y_train.values == 0].index.values
    neg = y_train[y_train.values == 1].index.values
    np.random.shuffle(pos)
    pos_splits = np.array_split(pos, n_split)

    ensemble = []

    for i, chunk in enumerate(pos_splits):
        idx = np.hstack((neg, chunk))
        X_t, y_t = X_train.loc[idx], y_train.loc[idx]
    
        if oversample:
            ros = random.choice([SMOTE(), ADASYN(), RandomOverSampler(),
                                SMOTEENN(), SMOTETomek(), RandomUnderSampler(), TomekLinks()])
        
            X_resampled, y_resampled = ros.fit_resample(X_t, y_t)
            if i % 20 == 0: print(sorted(Counter(y_t).items()), sorted(Counter(y_resampled).items()))
    
        else:      
            X_resampled, y_resampled = X_t, y_t
        
        if est_name == 'rf':
            est = RandomForestClassifier(n_jobs=-1, class_weight='balanced',
                                         n_estimators=100, min_samples_leaf=9)
        elif isinstance(est_name, xgb.sklearn.XGBClassifier):
            est = xgb.XGBClassifier(objective = 'binary:logistic', scale_pos_weight=1200)
            est.set_params(n_estimators= est_name.get_params()['n_estimators'],
                      learning_rate= est_name.get_params()['learning_rate'],
                      subsample= est_name.get_params()['subsample'],
                      max_depth= est_name.get_params()['max_depth'],
                      colsample_bytree= est_name.get_params()['colsample_bytree'],
                      min_child_weight= est_name.get_params()['min_child_weight']);
    
        est.fit(X_resampled, y_resampled)
        ensemble.append(est)
        
    return ensemble

ensemble_rf = train_ensemble('rf', 100)

>>> [(0, 871), (1, 77)] [(0, 832), (1, 77)]
[(0, 871), (1, 77)] [(0, 837), (1, 77)]
[(0, 871), (1, 77)] [(0, 871), (1, 854)]
[(0, 871), (1, 77)] [(0, 871), (1, 858)]
[(0, 870), (1, 77)] [(0, 870), (1, 870)]
```

**Different algorithms**
You can try different algorithms. Some algorithms are less sensitive to the problem of imbalanced dataset. Tree-based algorithms such as decision trees often perform well on imbalanced datasets because their hierarchical structure allows them to learn signals from both classes. In modern applied machine learning, tree ensembles (Random Forests, Gradient Boosted Trees, etc.) almost always outperform singular decision trees.

**Anomaly detection**
In more extreme cases, it may be better to think of classification under the context of anomaly detection, a.k.a. outlier detection. In anomaly detection, we assume that there is a "normal" distribution(s) of data-points, and anything that sufficiently deviates from that distribution(s) is an anomaly. When we reframe our classification problem into an anomaly detection problem, we treat the majority class as the "normal" distribution of points, and the minority as anomalies. Thinking of the minority class as the outliers class which might help you think of new ways to separate and classify samples. There are many algorithms for anomaly detection such as clustering methods, One-class SVMs, and Isolation Forests.

#### What is the difference between L1/L2 regularization?

$L_{1}$ penalizes sum of absolute value of weights. $L_{1}$ has a sparse solution. $L_{1}$ has multiple solutions. $L_{1}$ has built in feature selection. $L_{1}$ is robust to outliers. $L_{1}$ generates model that are simple and interpretable but cannot learn complex patterns.

$L_{2}$ regularization penalizes sum of square weights. $L_{2}$ has a non sparse solution. $L_{2}$ has one solution. $L_{2}$ has no feature selection. $L_{2}$ is not robust to outliers. $L_{2}$ gives better prediction when output variable is a function of all input features. $L_{2}$ regularization is able to learn complex data patterns.

A regression model that uses L1 regularization technique is called Lasso Regression and model which uses L2 is called Ridge Regression. Ridge regression adds “squared magnitude” of coefficient as penalty term to the loss function. Lasso Regression (Least Absolute Shrinkage and Selection Operator) adds “absolute value of magnitude” of coefficient as penalty term to the loss function. The key difference between these techniques is that Lasso shrinks the less important feature’s coefficient to zero thus, removing some feature altogether. So, this works well for feature selection in case we have a huge number of features.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/geometrical_representation_of_l1_l2.png?raw=true)

Depending on on the situation, we can choose which regularization to use. One important remark is that since nowadays we are concerned with a number of features, hence $L_{1}$ would be preferred.

#### What is curse of dimensionality?

As the number of features (dimensionality) increases, the data becomes relatively more sparse and often exponentially, more samples are needed to make statistically significant predictions. 

Curse of Dimensionality refers to the fact that many problems that do not exist in low-dimensional space arise in high-dimensional space. It makes it very difficult to identify the patterns in the data without having plenty of training data because of sparsity of training data in the high dimensional space.

Imagine going from a $10 \times 10$ grid to a $10 \times 10 \times 10$ grid. We want ONE sample in each '$1 \times 1$ square', then the addition of the third parameter requires us to have 10 times as many samples (1000) as we needed when we had 2 parameters (100).

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/curse_of_dimensionality_example.png?raw=true)

High-dimensional datasets are at risk of being very sparse: most training instances are likely to be far away from each other. Of course, this also means that a new instance will likely be far away from any training instances, making predictions much less reliable than in lower dimensions , since they will be based on much larger extrapolations. In short, the more dimensions the training set has, the greater the risk of overfitting it.

In theory, one solution to the curse of dimensionality could be to increase the size of the training set to reach a sufficient density of training instances. Unfortunately, in practice, the number of training instances required to reach a given density grows exponentially with the number of dimensions. 

Linear models with no feature selection or regularization, kNN, Bayesian models are models that are most affected by curse of dimensionality. Models that are less affected by the curse of dimensionaliy are regularized models, random forest, some neural networks, stochastic models (e.g. monte carlo simulations).

#### Why is dimension reduction important?

Dimension reduction can allow you to:

1. Remove collinearity from the feature space
2. Speed up training by reducing the number of features 
3. May filter out some noise and unnecessary details in the training set (reconstruction)
4. Simply save space (compression)
4. Reduce memory usage by reducing the number of features
5. Identifying underlying, latent, features that impact multiple features in the original space
6. Make it much easier to find a good solution (extreme number of features make it much harder to find a good solution due to the curse of dimensionality)
7. It is also extremely useful for data visualization. It can make it possible to plot a high-dimensional training set on a graph and often gain some important insights by visually detecting patterns, such as clusters.

#### Why would you want to avoid dimensionality reduction techniques to transform your data before training?

Dimension reduction can:

1. Add extra unnecessary computation
2. Make the model difficult to interpret if the latent features are not easy to understand
3. Add complexity to the model pipeline
4. Reduce predictive power of the model if too much signal is lost

#### If you have large number of predictors how would you handle them?

This is a very open-ended question and in many cases domain knowledge will play a crucial role. Besides, there are no such hard and fast rules and this is wholly dependent upon the nature of the data. With such a large number of independent variables, there is a strong presence of multicollinearity. The variables will be highly correlated to each other and this will provide incorrect results. One should certainly investigate collinearity before embarking on any analysis.

You can apply regularization tools and select from Ridge, LASSO and Elastic Net regression. The best will be LASSO regression as it removes all the non-significant variables that cause multicollinearity. You can use appropriate dimension reduction (eg. PCA) to see if there are still multicolinearity among the independent factors. If some eigen-value is near zero, you may drop one of them, or define a new factor (transformation). You can also use Principal Component Regression which is similar PCA but does regression. Severe multicollinearity will be detected as very small eigenvalues. To rid the data of the multicollinearity, principal component omit the components associated with small eigen values. Partial Least Squares Regression can laos cut the number of predictors to a smaller set of uncorrelated components.

You may also use some intelligent information criterion such as Akaike’s information criterion or Bayesian information criterion or Mallows’ CP to decide how many predictors should be in. 

#### How can you compare a neural network that has one layer, one input and output to a logistic regression model?

Logistic regression is a Machine Learning technique used to predict in situations where there are exactly two possibilities. For example, you might want to predict the sex of a person (0 = male, 1 = female) based on three predictor variables such as age, height, and annual income.

A good way to compare logistic regression to a neural network is to understand that you can simulate logistic regression with a neural network that has one hidden layer with a single hidden node and the identity activation function, and a single output node with the logistic sigmoid activation function.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/logistic_ann.jpeg?raw=true)

In each case the final computed output is $p = 0.5474$, which corresponds to a prediction of class = 1 because the p (probability) is greater than 0.50. Notice that the neural network hidden node has a bias value that corresponds to the bias in LR. The neural network output node has a bias of 0. The single hidden-to-output weight has constant value of 1.

#### What are the assumptions of Principle Component Analysis?

1. **Sample size**: ideally, there should be 150+ cases and there should be ratio of at least five cases for each variable (Pallant, 2010)
2. **Correlations**: there should be some (moderate at least) correlation among the factors to be considered for PCA. Otherwise, the number of principal components will be almost the same as the number of features in the dataset, which means carrying out PCA would be pointless.
3. **Linearity**: it is assumed that the relationship between the variables are linearly related
4. **Outliers**: PCA is sensitive to outliers; they should be removed.

#### What is micro-averaging and macro-averaging?

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC112219-11222019121938-1.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC112219-11222019121938-2.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC112219-11222019121938-3.png?raw=true)

{% highlight python %}
from sklearn import metrics
# Constants
C="Cat"
F="Fish"
H="Hen"

# True values
y_true = [C,C,C,C,C,C, F,F,F,F,F,F,F,F,F,F, H,H,H,H,H,H,H,H,H]
# Predicted values
y_pred = [C,C,C,C,H,F, C,C,C,C,C,C,H,H,F,F, C,C,C,H,H,H,H,H,H]

# Print the confusion matrix
print(metrics.confusion_matrix(y_true, y_pred))
# [[4 1 1]
#  [6 2 2]
#  [3 0 6]]

# Print the precision and recall, among other metrics
print(metrics.classification_report(y_true, y_pred, digits=3))
#               precision    recall  f1-score   support

#          Cat      0.308     0.667     0.421         6
#         Fish      0.667     0.200     0.308        10
#          Hen      0.667     0.667     0.667         9

#     accuracy                          0.480        25
#    macro avg      0.547     0.511     0.465        25
# weighted avg      0.581     0.480     0.464        25
{% endhighlight %}

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC112219-11222019121938-4.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC112219-11222019121938-5.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC112219-11222019121938-6.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC112219-11222019121938-7.png?raw=true)

Micro-average is preferable if there is a class imbalance problem.

#### If the model isn't perfect, how would you like to select the threshold so that the model outputs 1 or 0 for label?

The statistical component of your exercise ends when you output a probability for each class of your new sample. Choosing a threshold beyond which you classify a new observation as 1 vs. 0 is not part of the statistics any more. It is part of the decision component. It should be evaluated and tuned in regard to the objective function of the whole process.

However, one basic idea is to plot ROC curve. It is a plot of the false positive rate (x-axis) versus the true positive rate (y-axis) for a number of different candidate threshold values between 0.0 and 1.0. Put another way, it plots the false alarm rate versus the hit rate. From that plot you can choose thresh hold, based on where you get much better results, which mean high True Positive Rate (TPR) but low False Positive Rate (FPR). Larger values on the y-axis of the plot indicate higher true positives and lower false negatives.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/roc_curve_faq.png?raw=true)

Another idea is to use Precision-Recall curves, that focus on the positive class. Precision-Recall curves summarize the trade-off between the true positive rate and the positive predictive value for a predictive model using different probability thresholds.

Note that using a ROC curve with an imbalanced dataset might be deceptive and lead to incorrect interpretations of the model skill. 

Therefore generally, the use of ROC curves and precision-recall curves are as follows:

1. ROC curves should be used when there are roughly equal numbers of observations for each class.
2. Precision-Recall curves should be used when there is a moderate to large class imbalance.

#### What's the difference between convex and non-convex cost function? what does it mean when a cost function is non-convex?

A convex function is a function which the line segment betwen any two points on the graph lies above the graph and never cross graph itself. For non-convex functions, there will be at least one intersection. 

One of the most prominent features of a convex optimization problem is that it can be reduced to the problem of finding a local minimum. Any local minimum is guaranteed to be a global minimum. Some convex functions may have a flat region at the bottom rather than a single global minimum point, but any point within such a flat region is an acceptable solution. When optimizing a convex function, we know that we have reached a good solution if we find a critical point of any kind. We can also prove that there is no feasible solution to the problem (strong theoretical guarantees). However, a non-convex optimization problem may have multiple locally optimal points (or only a local minima) and it can take a lot of time to identify whether the problem has no solution or if the solution is global (weak theoretical guarantees). Hence, the efficiency in time of the convex optimization problem is much better. A convex problem usually is much more easier to deal with in comparison to a non convex problem which takes a lot of time and it might lead you to a dead end.

Linear regression/ Ridge regression, with Tikhonov regularisation, Sparse linear regression with L1 regularisation, such as Lasso, Support vector machines are examples of algorithms with convex optimization problem. Neural networks algorithm is an example for non-convex problems.

#### How do you deal with missing value in a data set?

In some cases, the data comes in the form of a dataset with features already defined. In some cases, values of some features can be missing. That often happens when the dataset was handcrafted and the person working on it forgot to fill some values or did not get them measured at all.

The typical approaches of dealing with missing values depend on the kind of the problem. Thus, we can safely say that there is no good way to deal with missing data. Trying several techniques, building several models and selecting the one that works might be the best.

At the prediction time, if your example is not complete, you should use the same data imputation technique to fill the missing feature as the technique you used to complete the training data.

To sum up, data imputation is tricky and should be done with care. It is important to understand the nature of the data that is missing when deciding which algorithm to use for imputations. 

Based on the origin of the missing data, the following terminology is used to describe it:

* **Missing at Random (MaR)**

  This category of missing data refers to the attributes that could not be answered due to the way the survey was designed. For example, consider the following question in a survey: 
  
  (a) Do you smoke? Yes / No
  (b) If yes, how frequently? Once a week, once a day, twice a day, mote than 2 times in a day. 
  
  You can see that the answer to the question (b) can only be given if the answer to the question (a) is a 'Yes'. This kind of missing values in the dataset arises due to the dependency of one attribute on another attribute.
  
* **Missing Completely at Random (MCaR)**

  This category of missing data is trully missed data or data that was not captured due to the oversight or for other reasons. It means that the missingness is nothing to do with the person being studied. For example, a questionnaire might be lost or a blood sample might be damaged in the lab.
  
* **Missing Not at Random (MNaR)**

  This category of missing data is dependent on the value of the data itself. For example, a survey need people to reveal their 10th grade marks in the chemistry. It may happen that people with lower marks may choose not to reveal them so you would see only high marks in the data sample. 

## Methods

1. **Ignore the data row or a whole column:**
  If you dataset is big enough, you can sacrifice some training examples. This is a quick solution and typically is preferred in cases where the percentage of missing values is relatively low (< 5%). It is a dirty approach as you loose the data.
  However dropping one whole observation (row) just because one of the features had a missing value, even if the rest of features are perfectly filled and informative can be a no-no in some cases. Therefore, you can also select to drop the rows only if all the values in the row are missing.
  Sometimes you may want to keep only the rows with at least 4 non-na values (4 is an arbitrary number here).
  Sometimes you may just want to drop a column (variable) that has some missing values. But this approach is valid only if data is missing far more than 60% observations and that variable is insignificant. 

2. **Replace with some constant outside the fixed value range, -999, -1 etc...:**
  Another technique is to replace the missing value with a value outside the normal range of a feature. For example, if the normal range is $[0, 1]$, then you can set the missing value to 2 or -1. The idea is that the learning algorithm will learn what is the best to do when the feature has a value significantly different from the regular values. Alternatively, you can replace the missing value by a value in the middle of the range. For example, if the range for a feature is $[-1, 1]$, you can set the missing value to be equal to 0. Here, the idea is that the value in the middle of the range will not significantly affect the predictions. 
  
3. **Replace with mean and median value:**
  This simple imputation method is based on treating every variable (column) individually, ignoring any inter-relationships with other variables.
  Mean is suitable for continuous data without outliers. Median is suitable for continuous data with outliers.
  For categorical feature, you can select to fill in the missing values with the most common value (mode).
  Note that the mean, median and mode imputation diminishes any correlations involving the variable (s) that are imputed. This is because we assume that there is no relationship between imputed variable and any other measured variables. Thus, those imputations have some attractive properties for univariate analysis but become problematic for multivariate analysis.
  
4. **Do nothing:**
  That is an easy solution. You just let the algorithm handle the missing data. Some algorithms can factor in the missing values and learn the best imputations values for the missing data based on the training loss reduction (ie. XGBoost). Some others have the option to just ignore them (ie. LightGBM — use_missing=false). However, other algorithms will panic and throw an error complaining about the missing values (ie. Scikit learn — LinearRegression). In that case, you will need to handle the missing data and clean it before feeding it to the algorithm.
  
5. **Isnull feature:**
  Adding a new feature `isnull` indicating which rows have missing values for this feature. By doing so, the tree based methods can now understand that there was a missing value. The downside is that we double the number of features.
  
6. **Extrapolation and interpolation:**
  They try to estimate the values from other observations within the range of a discrete set of known data points.
  
7. **Linear Regression:**
  To begin, several predictors of the variable with missing values are identified using a correlation matrix. In other words, we use all other independent variables in the dataset against this variable with missing values. The best predictors are selected and used as independent variables in a regression equation. The variable with missing data is used as dependent variable. Cases with complete data for the predictor variables are used to generate the regression equation. This equation is then used to predict the missing values for incomplete cases. Here, one must assume that there is a linear relationship between the variables used in the regression equation where there may be not be one. Similar idea can be applied when we try to impute the categorical variable. Here, we create a predictive model to estimate values that will substitute the missing data. In this case, we divide the dataset into two sets. One set with no missing values (training set) and another one with missing values (testing set). We can then use the classification methods like logistic regression and/or ANOVA for predictions.
  
8. **K-nearest Neighbor:**
  The K nearest neighbors algorithm can be used for imputing missing data by finding the K closest neighbors to the observation with missing data and then imputing them based on the the non-missing values in the neighbors using the mode (for categorical variable) and/or average (for continuous variable). There are several possible approaches to this. You can use 1NN schema, where you find the most similar neighbor and then use its value as a missing data replacement. Alternatively you can use kNN, with 𝑘 neighbors and take mean of the neighbors, or weighted mean, where the distances to neighbors are used as weights, so the closer neighbor is, the more weight it has when taking the mean. Using weighted mean seems to be used most commonly.  
  It can be used both discrete attributes and continuous attributes. For continuous data, we can use Euclidean Distance, Manhattan Distance and Cosine Similarity metrics. For categorical data, Hamming distance is generally used. For mixed types of data, Gower distance can be selected to calculate the distance. Gower distance uses Manhattan for calculating the distance between continuous data point and Dice distance for calculating the distance between categorical data points. 
  One of the obvious drawbacks of KNN algorithm is that it becomes time consuming when analyzing the larger datasets because it searches for similar instances through the entire set. It is computationally expensive because it stores the whole training dataset in memory. It is also sensitive to outliers. 
  Random Forest algorithm can also be used here. It produces a robust results because it works well with non-linear and categorical data. 
  Stochastic Regression imputation, Hot-deck imputation, Soft-deck imputation, Multivariate Imputation by Chained Equation (MICE) are other approaches to be used. 
  
9. **Assigning a unique category:**
  This approach is valid only for a categorical feature. We assign another class for the missing value. This strategy will add more information into dataset which will result in the change of variance. Since they are categorical, we need to use one-hot encoding to convert it to a numeric form for the algorithm to understand it.
  
Also note that you should first split your data and then apply the imputation technique in order to prevent data leakage.

#### How to find a confidence interval for accuracy of a model?

In order to determine the confidence interval, we need to establish the probability distribution that governs the accuracy measure. A Binomial experiment has
* n independent trials, where each trial has two possible outcomes
* the probability of success, p, in each trial is constant.

The expected value of binomial distribution is $n \times p$ and its variance is $n p(1-p)$. Its probability mass function is given by:

$$
Binomial(x; n,p) = {n \choose x} p^{x} (1-p)^{n-x}
$$

where $x$ is the number of successes.

The task of predicting the class labels of test records can also be considered as a binomial experiment. The accuracy can be calculated based on a hold-out dataset not seen by the model during training, such as a validation or test dataset.

Given a test set that contains $n$ records, let $X$ be the number of records correctly predicted by a model and $p$ be the true accuracy of the model. Classification accuracy or classification error is a proportion or a ratio.

By modeling the prediction task as a binomial experiment, $X$ has Binomial distribution with mean $n p$ and variance $n p(1-p)$. It can be shown that the empirical accuracy $acc = X/n$ also has binomial distribution with mean $p$ and variance $p(1-p)/n$. Although the binomial distribution can be used to estimate the confidence interval for $acc$, it is often approximated by a normal distribution when $n$ is sufficiently large (e.g. more than 30). Based on the normal distribution, the following confidence interval for $acc$ can be derived:

$$
P \left(-Z_{\alpha/2} \leq  \frac{acc - p}{\sqrt{p(1-p)/n}} \leq Z_{1-\alpha/2} \right) = 1 - \alpha
$$

where $-Z_{\alpha/2}$ and $Z_{\alpha/2}$ are the upper and lower bounds obtained from a standard normal distribution at confidence level $(1-\alpha)$. 

Note that when you increase the sample size $n$, the confidence interval will be tighter. 

Consider a model with an error of 20\%, or 0.2 (error = 0.2), on a validation dataset with 50 examples ($n = 50$). We can calculate the 95% confidence interval (z = 1.96) as follows:

```python
# binomial confidence interval
from math import sqrt
interval = 1.96 * sqrt( (0.2 * (1 - 0.2)) / 50)
print('%.3f' % interval)
#0.111
```

Running the example, we see the calculated radius of the confidence interval calculated and printed 0.111.

We can then make claims such as:

* The classification error of the model is 20% +/- 11%
* The true classification error of the model is likely between 9% and 31%

#### Does tree-based methods such as Decision Tree handle multicollinearity by itself?

There appears to be an underlying assumption that not checking for collinearity is a reasonable or even best practice. This seems flawed. Therefore, it is a good practice to remove any redundant features from any dataset used for training, irrespective of the model's algorithm. 

Desicion trees make no assumptions on relationships between features. It just constructs splits on single features that improves classification, based on an impurity measure like Gini or entropy. If features A, B are heavily correlated, no /little information can be gained from splitting on B after having split on A. So it would typically get ignored in favor of C.

#### How to model count data?

When the response variable is the counted number of occurrences of an event, we apply different modeling techniques. 

The distribution of counts is discrete, not continuous, and is limited to non-negative values. Besides count data are positively skewed with many observations in the data set having a value of 0. 

1. Poisson Regression
2. Negative Binomial Regression
3. Zero-Inflated Count Models
4. Zero-inflated Poisson
5. Zero-inflated Negative Binomial
6. Zero-Truncated Count Models
7. Zero-truncated Poisson
8. Zero-truncated Negative Binomial
9. Hurdle Models

#### Why should weights of Neural Networks be initialized to random numbers?

Let's start with all zero initialization. It is something that we we should not do. Note that we do not know what the final value of every weight should be in the trained network, but with proper data normalization it is reasonable to assume that approximately half of the weights will be positive and half of them will be negative. A reasonable-sounding idea then might be to set all the initial weights to zero, which we expect to be the “best guess” in expectation. This turns out to be a mistake, because if every neuron in the network computes the same output, then they will also all compute the same gradients during backpropagation and undergo the exact same parameter updates. In other words, no learning will occur. This phenomenon is perhaps the most important example of a saddle point in neural net training. There is no source of asymmetry between neurons if their weights are initialized to be the same. We need to break the symmetry. The initialization is asymmetric (which is different) so you can find different solutions to the same problem since loss contours for a neural network can be messy and have a number of local minima since they require minimizing a high-dimensional non-convex loss function (the loss function of a neural network is non-convex in terms of the model parameters even though MSE and Cross-entropy are convex functions).

For large-size networks, most local minima are equivalent and yield similar performance on a test set. The probability of finding a "bad" (high value) local minimum is non-zero for small-size networks and decreases quickly with network size. Struggling to find the global minimum on the training set (as opposed to one of the many good local ones) is not useful in practice and may lead to overfitting.

There are several weight initialization strategies; each one is best suited for a type of activation function. For instance, Glorot's initialization aims at not saturating sigmoid activations, while He's initialization is meant for Rectified Linear Units (ReLUs).

#### Why is loss function in Neural Networks not convex?

I’ll be using mean squared error (MSE) loss for the discussion below, instead of cross-entropy loss, because the analysis is simpler with MSE, and the conclusions directly apply to cross-entropy loss.

MSE is always convex. But your confusion arises because the answer depends on what you’re taking the convexity with respect to.

Any loss function takes two parameters, the predicted label $\hat{y}$  and the true label $y$. Now MSE is defined as $L(\hat{y}, y) = \frac{1}{n} \sum_{i=1}^{n} \left(y_{i} - \hat{y_{i}} \right)^{2}$. So this is clearly convex with respect to $\hat{y}$ . This is irrespective of any algorithm — logistic regression or neural networks.

However, when people say that the loss function of neural networks is not convex, it is not the function above that they are referring to. Note that you do not control $\hat{y}$  directly. You control some weight parameters (called model parameters), which in turn change $\hat{y}$ . So you’re interested in convexity of the loss function with respect to these weight parameters. Let’s say $\hat{y} = f(w, x)$, where $w$ is the vector of all weights, $x$ is the input example and $f$ is a function mapping those to a label $\hat{y}$ . You are interested in the function $g(x, y, w) = L(\hat{y}, y) = L(f(w, x),y)$.

Even though MSE is indeed convex in $\hat{y_{i}}$. But if $\hat{y_{i}} = f(w, x_{i})$, it might not be convex in model parameters $w$, which is the situation with most non-linear models, and we actually care about convexity in $w$ because that's what we're optimizing the cost function over.

It turns out that, in general, $f(\cdot)$ is not convex, and so $g(\cdot)$ is also not convex. Thus, you can say that MSE loss is non-convex for neural networks, which is implicitly referring to $g(\cdot)$ and not $L(\cdot)$.

If you permute the neurons in the hidden layer and do the same permutation on the weights of the adjacent layers then the loss doesn't change. Hence if there is a non-zero global minimum as a function of weights, then it can't be unique since the permutation of weights gives another minimum (loss function might have also a number of local maxima and minima). Hence the function is not convex.

#### Is it possible to train a neural network without backpropagation?

The original neural networks, before the backpropagation revolution in the 70s, were "trained" by hand. There is a "school" of machine learning called "extreme learning machine" that does not use backpropagation. However, one can use pretty much any numerical optimization algorithm (such as Nelder-Mead, Simulated Annealing or a Genetic Algorithm) to optimize weights of a neural network. You can also use mixed continous-discrete optimization algorithms to optimize not only weights, but layout itself (number of layers, number of neurons in each layer, even type of the neuron). But, there's no optimization algorithm that do not suffer from "curse of dimensionality" and local optimas in some manner. Computational complexity is another trouble. They also may take a very long time to do so.

There are all sorts of local search algorithms you could use, backpropagation has just proved to be the most efficient for more complex tasks in general; there are circumstances where other local searches are better. However, in the broader class of derivative-free optimization (DFO) algorithms, there are many which are significantly better than these "classics", as this has been an active area of research in recent decades. 

Simulated Annealing, Particle Swarm Optimisation and Genetic Algorithms are good global optimisation algorithms that navigate well through huge search spaces and unlike Gradient Descent, they do not need any information about the gradient and could be successfully used with black-box objective functions and problems that require running simulations. 

#### In neural networks, why do we use gradient methods rather than other metaheuristics?

This is more a problem to do with the function being minimized than the method used, if finding the true global minimum is important, then use a method such a simulated annealing. This will be able to find the global minimum, but may take a very long time to do so.

For large-size networks, most local minima are equivalent and yield similar performance on a test set. The probability of finding a "bad" (high value) local minimum is non-zero for small-size networks and decreases quickly with networks size. Struggling to find the global minimum on the training set (as opposed to one of the many good local ones) is not useful in practice and may lead to overfitting. In this view, there's not a great reason to deploy heavy-weight approaches for finding the global minimum. That time would be better spent trying out new network topologies, features, data sets, etc.

In simpler words, for the case of neural nets, local minima are not necessarily that much of a problem. Some of the local minima are due to the fact that you can get a functionally identical model by permuting the hidden layer units, or negating the inputs and output weights of the network etc. Also if the local minima is only slightly non-optimal, then the difference in performance will be minimal and so it will not really matter. Lastly, and this is an important point, the key problem in fitting a neural network is over-fitting, so aggressively searching for the global minima of the cost function is likely to result in overfitting and a model that performs poorly.

#### What is the difference between a loss function and decision function?

The loss function is what is minimized to obtain a model which is optimal in some sense. The model itself has a decision function which is used to predict.

For example, in SVM classifiers:

* loss function: minimizes error and squared norm of the separating hyperplane 

  $$
  \mathcal{L}(\mathbf{w}, \xi) =\frac{1}{2}\|\mathbf{w}\|^2 + C\sum_i \xi_i
  $$

* decision function: signed distance to the separating hyperplane: $f(\mathbf{x}) = sign(\mathbf{w}^T\mathbf{x} + b)$.

#### What is the difference between SVM and Random Forest?

The no free lunch theorem basically says that there will always be data sets where one classifier is better than another. The choice depends very much on what data you have and what is your purpose.

Random Forest is intrinsically suited for multiclass classification problems, while SVM can only do binary classification. It cannot be naturally extended to multi-class problems. For multiclass problem you will need to reduce the problem using some schemes existing in the literature such as One-vs-All (usually referred to as One-vs-Rest or OVA classification) and One-vs-One (OVO, also known as All-versus-All or AVA).

Random Forest provides feature importance plot. This plot can be used as feature selection method. However, SVM does not have that option.

Since Random Forest is a tree-based model. Scaling the data typically does not change the performance. SVM maximizes the "margin" and thus relies on the concept of "distance" between different points. It is up to you to decide if "distance" is meaningful. Therefore, data scaling or normalization procedure is highly recommended at preprocessing step.

Random forest can handle categorical variables naturally. However, SVM cannot. As a consequence, one-hot encoding (or  another encoding strategy) for categorical features is a must-do. 

Each tree in Random Forest predicts class probabilities and these probabilities are averaged for the forest prediction. SVM gives you distance to the boundary, it does not directly provide probabilities, which are desirable in most classification problems. Various methods applied to the output of SVM, including Platt scaling and isotonic regression.

Random Forest has very little need for tuning of hyperparameters. This is not the case for SVM. With SVM, there are more things to worry the regularization penalty, choosing an appropriate kernel, kernel parameters etc. Random Forests are much more automated and thus "easier" to train compared to SVM.

Both random forests and SVMs are non-parametric models.

The more trees we have, the more expensive it is to build a random forest. Also, we can end up with a lot of support vectors in SVMs; in the worst-case scenario, we have as many support vectors as we have samples in the training set.

As a rule of thumb, I’d say that SVMs are great for relatively small data sets with fewer outliers. Random forests may require more data but they almost always come up with a pretty robust model. 

#### What is the difference between fitting a model via closed-form equations vs. Gradient Descent and its variants?

We can either solve the model parameters analytically (closed-form equations) or use an optimization algorithm (Gradient Descent, Stochastic Gradient Descent, Newton’s Method, Simplex Method, etc.). The closed-form solution may (should) be preferred for "smaller" datasets – if computing (a "costly") matrix inverse is not a concern. For very large datasets, or datasets where the inverse of $X^{T}X$ may not exist (the matrix is non-invertible or singular, e.g., in case of perfect multicollinearity), the GD or SGD approaches are to be preferred. 

#### What are some of the issues with K-means?

Only numerical data can be used. Generally K-means works best for 2 dimensional numerical data. Visualization is possible in 2D or 3D data. But in reality there are always multiple features to be considered at a time. However, we must be careful about curse of dimensionality. any more than few tens of dimensions mean that distance interpretation isn’t obvious and must be guarded against. Appropriate dimensionality reduction techniques and distance measure must be used.

K-Means clustering is prone to initial seeding i.e. random initialization of centroids which is required to kick-off iterative clustering process. Bad initialization may end up getting bad clusters.
 
The standard K-means algorithm isn't directly applicable to categorical data, for various reasons. The sample space for categorical data is discrete, and doesn't have a natural origin. A Euclidean distance function on such a space is not really meaningful. However, The clustering algorithm is free to choose any distance metric / similarity score. Euclidean is the most popular. But any other metric can be used that scales according to the data distribution in each dimension/attribute, for example the Mahalanobis metric.

The use of Euclidean distance as the measure of dissimilarity can also make the determination of the cluster means non-robust to outliers and noise in the data.

Inertia is not a normalized metric: we just know that lower values are better and zero is optimal. But in very high-dimensional spaces, Euclidean distances tend to become inflated (this is an instance of the so-called “curse of dimensionality”). Running a dimensionality reduction algorithm such as Principal component analysis (PCA) prior to k-means clustering can alleviate this problem and speed up the computations.

Categorical data (i.e., category labels such as gender, country, browser type) needs to be encoded (e.g., one-hot encoding for nominal categorical variable or label encoding for ordinal categorical variable) or separated in a way that can still work with the algorithm, which is still not perfectly right. There's a variation of K-means known as K-modes, introduced in [this paper](http://www.cs.ust.hk/~qyang/Teaching/537/Papers/huang98extensions.pdf) by Zhexue Huang, which is suitable for categorical data. 

K-Means does not behave very well when the clusters have varying sizes, different densities, or non-spherical shapes. In that case, one can use Mixture models using EM algorithm or Fuzzy K-means (every object belongs to every cluster with a membershio weight that is between 0 (absolutely does not belong) and 1 (absolutely belongs)). which both allow soft assignments. As a matter of fact, K-means is special variant of the EM algorithm with the assumption that the clusters are spherical. EM algorithm also starts with random initializations, it is an iterative algorithm, it has strong assumptions that the data points must fulfill, it is sensitive to outliers, it requires prior knowledge of the number of desired clusters. The results produced by EM are also non-reproducible.

The above paragraph shows the drawbacks of this algorithm. K-means assumes the variance of the distribution of each attribute (variable) is spherical; all variables have the same variance; the prior probability for all K clusters is the same, i.e., each cluster has roughly equal number of observations. If any one of these 3 assumptions are violated, then K-means will fail. [This Stackoverflow answer](https://stats.stackexchange.com/a/249288/16534) explains perfectly!

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Screen%20Shot%202020-05-20%20at%2019.08.40.png?raw=true)

It is important to scale the input features before you run K-Means, or else the clusters may be very stretched, and K-Means will perform poorly. Scaling the features does not guarantee that all the clusters will be nice and spherical, but it generally improves things.

K-Means clustering just cannot deal with missing values. Any observation even with one missing dimension must be specially handled. If there are only few observations with missing values then these observations can be excluded from clustering. However, this must have equivalent rule during scoring about how to deal with missing values. Since in practice one cannot just refuse to exclude missing observations from segmentation, often better practice is to impute missing observations. There are various methods available for missing value imputation but care must be taken to ensure that missing imputation doesn’t distort distance calculation implicit in k-Means algorithm. For example, replacing missing age with -1 or missing income with 999999 can be misleading!

Clustering analysis is not negatively affected by heteroscedasticity but the results are negatively impacted by multicollinearity of features/ variables used in clustering as the correlated feature/ variable will carry extra weight on the distance calculation than desired.

K-Means clustering algorithm might converse on local minima which might also correspond to the global minima in some cases but not always. Therefore, it’s advised to run the K-Means algorithm multiple times before drawing inferences about the clusters. However, note that it’s possible to receive same clustering results from K-means by setting the same seed value for each run. But that is done by simply making the algorithm choose the set of same random number for each run.

#### What are the possible termination conditions in K-Means?

1. For a fixed number of iterations.
2. Assignment of observations to clusters does not change between iterations. Except for cases with a bad local minimum.
3. Centroids do not change between successive iterations.
4. Terminate when inertia falls below a threshold.

#### Why multicollinearity does not affect the predictive performance?

In the literature, you can read statements like "Multicollinearity does not affect the predictive power but individual predictor variable’s impact on the response variable could be calculated wrongly". This may seem contradictory. besides, as parameters of independent variables are estimated wrongly, you would think that it would affect the predictive performance. 

There is a simple explanation for it. Let's assume that you have trained a model on a training dataset, and want to predict some values in a test/holdout dataset. Multicollinearity in your training dataset should only reduce predictive performance in the test dataset if the covariance between variables in your training and test datasets is different. If the covariance structure (and consequently the multicollinearity) is similar in both training and test datasets, then it does not pose a problem for prediction. Since a test dataset is typically a random subset of the full dataset, it's generally reasonable to assume that the covariance structure is the same. Therefore, multicollinearity is typically not an issue for this purpose.

#### How does multicollinearity affect feature importances in random forest classifier?

Random Forest is robust to multicollinearity issues in the aspect of prediction accuracy since it has the nature of selecting samples with replacement as well as selecting subsets of features on those samples randomly (Random Forest models handle quite well correlated/redundant variables, yes. But that does not mean your model necessarily benefits from unrelated or completely redundant variables(e.g. linear recombinations), it does not crash either). However, it definitely can affect variable importances in random forest models. Intuitively, it can be difficult to rank the relative importance of different variables if they have the same or similar underlying effect, which is implied by multicollinearity. That is- if we can access the underlying effect by measuring more than one variable, it's not easy to say which is causing the effect, or if they are mutual symptoms of a third effect.

When the dataset has two (or more) correlated features, then from the point of view of the model, any of these correlated features can be used as the predictor, with no concrete preference of one over the others. However once one of them is used, the importance of others is significantly reduced since effectively the impurity they can remove is already removed by the first feature.

As a consequence, they will have a lower reported importance. This is not an issue when we want to use feature selection to reduce overfitting, since it makes sense to remove features that are mostly duplicated by other features, But when interpreting the data, it can lead to the incorrect conclusion that one of the variables is a strong predictor while the others in the same group are unimportant, while actually they are very close in terms of their relationship with the response variable.

The effect of this phenomenon is somewhat reduced thanks to random selection of features at each node creation, but in general the effect is not removed completely.

#### How come do the loss functions have $1/m$ and $2$ from the square cancels out? 

In some books/tutorials/articles, you can see that the cost function for linear regression is defined as follows:

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left(\theta^{T} \mathbf{x}^{(i)} - y^{(i)}\right)^{2}$$

The $\frac{1}{m}$ is to "average" the squared error over the number of observations so that the number of observations doesn't affect the function.

So now the question is why there is an extra $\frac{1}{2}$. In short, it is merely for convenience, and actually, so is the $m$ - they're both constants. The expression for the gradient becomes prettier with the $\frac{1}{2}$, because the $2$ from the square term cancels out. However, the $m$ is useful if you solve this problem with gradient descent because it will take out the dependency on the number of observations. Then your gradient becomes the average of $m$ terms instead of a sum, so it's scale does not change when you add more data points.

#### What is feature engineering?

Feature engineering is the process of taking a dataset and constructing explanatory variables (features) that can be used to train a machine learning model for a prediction problem. Often, data is spread across multiple tables and must be gathered into a single table with rows containing the observations and features in the columns.

Traditional approach to feature engineering is to build features one at a time using domain knowledge, a tedious, time-consuming and error-prone process known as manual engineering. The code for manual feature engineering is a problem-dependent and must be written for each new dataset.

In a nutshell, feature engineering is to create new features from your existing ones to improve model performance.

* **Creating Indicator Variables**: Indicator variable from thresholds (Let’s say you’re studying alcohol preferences by U.S. consumers and your dataset has an `age` feature. You can create an indicator variable for `age >= 21` to distinguish subjects who were over the legal drinking age), indicator variable from multiple features (You’re predicting real-estate prices and you have the features n_bedrooms and n_bathrooms. If houses with 2 beds and 2 baths command a premium as rental properties, you can create an indicator variable to flag them), indicator variable for special events (You’re modeling weekly sales for an e-commerce site. You can create two indicator variables for the weeks of Black Friday and Christmas), and indicator variable for groups of classes (You’re analyzing website conversions and your dataset has the categorical feature traffic_source. You could create an indicator variable for paid_traffic by flagging observations with traffic source values of  "Facebook Ads" or "Google Adwords"). 

* Creating interaction Features such as Sum of two features, Difference between two features, Product of two features and Quotient of two features

* **Feature Representation**: the data won’t always come in the ideal format. You should consider if you’d gain information by representing the same feature in a different way, such as Date and time features (extracting year from purchase date or demcomposing month/hour/minutes/second variables into two by creating a sine and a cosine facet of each of these three variable to preserve the cyclical nature ), Numeric to categorical mappings, Grouping sparse classes, and Creating dummy variables

* **External Data**: External API’s (There are plenty of API’s that can help you create features. For example, the Microsoft Computer Vision API can return the number of faces from an image), Geocoding (Let’s say have you street_address, city, and state. Well, you can geocode them into latitude and longitude. This will allow you to calculate features such as local demographics (e.g. median_income_within_2_miles)), and using other sources to collect features (such as Fac

#### When to choose Decision Tree over Random Forest?

Even though it completely depends on the data we have and the output we are looking for, there might be some reasons why we can choose Decision Trees over Random Forest:

1. When entire dataset and features can be used
2. When we have limited computational power
3. When we want the model to be simple and explainable even to non-technical users, with a relatively lower accuracy that can be tolerated, since Decision Tree is easy to understand and interpret

If we need to analyse a bigger dataset, with a lot of features with a higher demand for accuracy a Random Forest would be a better choice.

#### What is a machine learning project lifecycle?

Overall, a machine learning project lifecycle consists of the following stages:

1. Goal definition
2. Data collection and preparation
3. Feature engineering
4. Model building
5. Model evaluation
6. Model deployment
7. Model serving
8. Model monitoring
9. Model maintenance

#### What are the unknowns of a machine learning project?

There are several major unknowns that are almost impossible to guess with confidence unless you worked on a similar project in the past or read about such a project. The unknowns are:

1. whether the required accuracy level (or the value of any other metric important to you) is attainable in practice,
2. how much data you will need to reach the required accuracy level,
3. what features and how many features are needed so that the model can learn and generalize sufficiently well,
4. how large the model should be (especially relevant for neural networks and ensemble
architectures),
5. how long will it take to train one model (in other words, how much time is needed to
run one experiment) and how many experiments will be needed to reach the desired
level of performance.

#### What are the properties of a successful model?

A successful model has the following four properties:

1. it respects the input and output specifications and the minimum performance requirement,
2. it benefits the organization (measured via cost reduction, increased sales or profit),
3. it benefits the user (measured via productivity, engagement, and sentiment),
4. it is scientifically rigorous

A scientifically rigorous model is characterized by a predictable behavior (for the input examples that are similar to the examples that were used for training) and is reproducible. The former property (predictability) means that if input feature vectors come from the same distribution of values as the training data, then the model, on average, has to make the same amount of errors as was observed on the holdout data when the model was trained. The latter property (reproducibility) means that a model with similar properties can be easily built once again from the same training data using the same algorithm and values of hyperparameters. The word “easily” means that no additional analysis, labeling, or coding is necessary to rebuild the model, only the compute power.

#### How to convert an RGB image to grayscale?

Consider a color image, given by its red, green, blue components R, G, B. The range of pixel values is often 0 to 255. Color images are represented as multi-dimensional arrays - a collection of three two-dimensional arrays, one each for red, green, and blue channels. Each one has one value per pixel and their ranges are identical. For grayscale images, the result is a two-dimensional array with the number of rows and columns equal to the number of pixel rows and columns in the image.

We present some methods for converting the color image to grayscale:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

img_path = 'img_grayscale_algorithms.jpg'

img = cv2.imread(img_path)
print(img.shape)
#(1300, 1950, 3)

#Matplotlib EXPECTS RGB (Red Greed Blue)
#but...
#OPENCV reads as Blue Green Red

#we need to transform this in order that Matplotlib reads it correctly
fix_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(fix_img)

#Let's extract the three channels
R, G, B = fix_img[:,:,0], fix_img[:,:,1],fix_img[:,:,2]
```

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/original_image.png?raw=true)

##### Weighted average

This is the grayscale conversion algorithm that OpenCV `cvtColor()` use (see the [documentation](https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#void%20cvtColor%28InputArray%20src,%20OutputArray%20dst,%20int%20code,%20int%20dstCn%29))

The formula used is:

$$
Y = 0.299\times R + 0.587 \times G + 0.114 \times B
$$

```python
Y = 0.299 * R + 0.587 * G + 0.114 * B
print(Y)
# [[ 85.967  88.967  92.967 ...  27.85   27.85   27.85 ]
#  [ 83.967  86.967  89.967 ...  27.85   27.85   27.85 ]
#  [ 81.956  83.956  86.967 ...  26.85   26.85   26.85 ]
#  ...
#  [121.015 112.243 108.998 ... 108.02  108.792 108.792]
#  [120.086 111.086 108.727 ... 107.02  106.792 106.792]
#  [118.086 111.086 109.727 ... 105.02  104.792 105.792]]

plt.imshow(Y, cmap='gray')
plt.savefig('image_weighted_average_byhand.png')
```

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/image_weighted_average_byhand.png?raw=true)

Let's compare it with the original function in `OpenCV`:

```python
img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
print(img_gray)
# [[ 86  89  93 ...  28  28  28]
#  [ 84  87  90 ...  28  28  28]
#  [ 82  84  87 ...  27  27  27]
#  ...
#  [121 112 109 ... 108 109 109]
#  [120 111 109 ... 107 107 107]
#  [118 111 110 ... 105 105 106]]

plt.imshow(img_gray, cmap='gray')
plt.savefig('image_weighted_average_OPENCV.png')
```

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/image_weighted_average_OPENCV.png?raw=true)


##### Average method

Average method is the most simple one. You just have to take the average of three colors. Since its an RGB image, so it means that you have add R with G with B and then divide it by 3 to get your desired grayscale image.

```python
grayscale_average_img = np.mean(fix_img, axis=2)
# (axis=0 would average across pixel rows and axis=1 would average across pixel columns.)
print(grayscale_average_img)
# [[82.         85.         89.         ... 26.         26.
#   26.        ]
#  [80.         83.         86.         ... 26.         26.
#   26.        ]
#  [77.66666667 79.66666667 83.         ... 25.         25.
#   25.        ]
#  ...
#  [92.         83.66666667 81.33333333 ... 81.         81.33333333
#   81.33333333]
#  [90.66666667 81.66666667 80.         ... 80.         79.33333333
#   79.33333333]
#  [88.66666667 81.66666667 81.         ... 78.         77.33333333
#   78.33333333]]

plt.imshow(grayscale_average_img, cmap='gray')
plt.savefig('image_average_method.png')
```

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/image_average_method.png?raw=true)

Since the three different colors have three different wavelength and have their own contribution in the formation of image, so we have to take average according to their contribution, not done it averagely using average method. Right now what we are doing is 33% of Red, 33% of Green, 33% of Blue. We are taking 33% of each, that means, each of the portion has same contribution in the image. But in reality that is not the case. The solution to this has been given by luminosity method.

##### The luminosity method 

This method is a more sophisticated version of the average method. It also averages the values, but it forms a weighted average to account for human perception. Through many repetitions of carefully designed experiments, psychologists have figured out how different we perceive the luminance or red, green, and blue to be. They have provided us a different set of weights for our channel averaging to get total luminance. The formula for luminosity is:

$$
Z = 0.2126\times R + 0.7152 G + 0.0722 B
$$

According to this equation, Red has contribute 21%, Green has contributed 72% which is greater in all three colors and Blue has contributed 7%.

```python
Z = 0.2126 * R + 0.7152 * G + 0.0722 * B
print(Z)
# [[ 84.7266  87.7266  91.7266 ...  27.404   27.404   27.404 ]
#  [ 82.7266  85.7266  88.7266 ...  27.404   27.404   27.404 ]
#  [ 81.0166  83.0166  85.7266 ...  26.404   26.404   26.404 ]
#  ...
#  [128.2216 119.366  115.8674 ... 114.2874 115.143  115.143 ]
#  [127.2898 118.2898 115.719  ... 113.2874 113.143  113.143 ]
#  [125.2898 118.2898 116.719  ... 111.2874 111.143  112.143 ]]

plt.imshow(Z, cmap='gray')
plt.savefig('image_luminosity_method.png')
```

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/image_luminosity_method.png?raw=true)

As you can see here, that the image has now been properly converted to grayscale using weighted method. As compare to the result of average method, this image is more brighter thank average method.

#### What are the Model Selection Criterion, i.e., AIC and BIC?

Goodness-of-fit tests are used to assess the performance of a model with respect to how well it explains the data. However, suppose we want to select from among several candidate models. What criterion can be used to select the best model?

Model selection is the process of fitting multiple models on a given dataset and choosing one over all others. Given a set of data, the objective is to determine which of the candidate models best approximates the data. The information-theoretic approach is used to derive the two most commonly used criteria in model selection — the Akaike information criterion and the Bayesian information criterion. These criterion are probabilistic measures which involve analytically scoring a candidate model using both model performance on the training dataset (goodness of fit) and the complexity of the model. Model performance may be evaluated using a probabilistic framework, such as log-likelihood under the framework of maximum likelihood estimation. Model complexity may be evaluated as the number of degrees of freedom or parameters in the model.

Very simple models are high-bias, low-variance while with increasing model complexity they become low-bias, high-variance. The concept of model complexity can be used to create measures aiding in model selection. The Akaike information criterion and the Bayesian information criterion are a few measures which explicitly deal with this trade-off between goodness of fit and model simplicity. Both penalize the number of model parameters but reward goodness of fit on the training set, hence the best model is the one with lowest AIC/BIC. However, these probabilistic measures are appropriate when using simpler linear models like linear regression or logistic regression where the calculating of model complexity penalty (e.g. in sample bias) is known and tractable.

**AKAIKE INFORMATION CRITERION**

Kullback and Leibler developed a measure to capture the information that is lost when approximating reality; that is, the Kullback and Leibler measure is a criterion for a good model that minimizes the loss of information. Akaike established a relationship between the Kullback-Leibler measure and maximum likelihood estimation method —an estimation method used in many statistical analyses- to derive a criterion (i.e., formula) for model selection. This criterion, referred to as the Akaike information criterion (AIC), is generally considered the first model selection criterion that should be used in practice.

The AIC is formulated as:

$$
AIC = - 2 L(\hat{\theta}) + 2k
$$

where $\theta$ is the set (vector) of model parameters, $L(\hat{\theta})$ is the likelihood of the candidate model given the data when evaluated at the maximum likelihood estimate of $\theta$, and k is the number of estimated parameters in the candidate model. The AIC in isolation is meaningless. Rather, this value is calculated for every candidate model and the "best" model is the candidate model with the smallest AIC.  Let’s look at the two components of the AIC. The first component, $- 2 L(\hat{\theta})$, is the value of the likelihood function, which is the probability of obtaining the data given the candidate model. Since the likelihood function’s value is multiplied by $–2$, ignoring the second component, the model with the minimum AIC is the one with the highest value for the likelihood function. However, to this first component we add an adjustment based on the number of estimated parameters. The more parameters, the greater the amount added to the first component, increasing the value for the AIC and penalizing the model. Hence, there is a trade-off: the better fit, created by making a model more complex by requiring more parameters, must be considered in light of the penalty imposed by adding more parameters. This is why the second component of the AIC is thought of in terms of a penalty.

**BAYESIAN INFORMATION CRITERION**

The Bayesian information criterion (BIC), proposed by Schwarz and hence also referred to as the Schwarz information criterion and Schwarz Bayesian information criterion, is another model selection criterion based on information theory but set within a Bayesian context. The difference between the BIC and the AIC is the greater penalty imposed for the number of parameters by the former than the latter (because second component of BIC is bigger than the second component of AIC). BIC penalizes model complexity stronger and hence favors models which are "more wrong" but simpler. AIC statistic penalizes complex models less, meaning that it may put more emphasis on model performance on the training dataset, and, in turn, select more complex models.

The BIC is computed as follows:

$$
BIC = - 2 L(\hat{\theta}) + k log(n)
$$

The best model is the one that provides the minimum BIC.

For an example of linear regression:

```python
# generate a test dataset and fit a linear regression model
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import log


# calculate aic for regression
def calculate_aic(n, mse, num_params):
    aic = n * log(mse) + 2 * num_params
    return aic

# calculate bic for regression
def calculate_bic(n, mse, num_params):
    bic = n * log(mse) + num_params * log(n)
    return bic


# generate dataset
X, y = make_regression(n_samples=100, n_features=5, noise=0.1)

# define and fit the model on all data
model = LinearRegression()
model.fit(X, y)

# number of parameters
num_params = len(model.coef_) + 1
print('Number of parameters: %d' % (num_params))
# Number of parameters: 6
    
# predict the training set
yhat = model.predict(X)

# calculate the error
mse = mean_squared_error(y, yhat)
print('MSE: %.3f' % mse)
# MSE: 0.014

# calculate the aic
aic = calculate_aic(len(y), mse, num_params)
print('AIC: %.3f' % aic)
# AIC: -418.317

# calculate the bic
bic = calculate_bic(len(y), mse, num_params)
print('BIC: %.3f' % bic)
#BIC: -402.686
```

Some caution is warranted when using AIC and BIC to compare models. The same data should be fit by models that are being compared. This becomes relevant when some cases are excluded from a model due to missing values on some of the variables. Attention should also be paid to ensure that the correct or full. ln likelihood is used to compute AIC or BIC when comparing models with different distributions. For some distributions, the full logarithm of the likelihood has an additive constant that only depends on the data. Regardless of the link or what is included in the linear predictors, this additive constant is the same; therefore, some programs only use the *kernel* of the likelihood (i.e., the logarithm of the likelihood without the additive constant). As a example, consider the Poisson distribution. The full logarithm of the likelihood is

$$
\begin{split}
ln\left(L(\mu; \mathbf{y})\right) &= ln \left(\prod_{i=1}^{n} \frac{e^{-\mu} \mu^{y_{i}}}{y_{i}!} \right)\\
&= \underbrace{\sum_{i=1}^{n} y_{i} ln(\mu) - n\mu}_{kernel} - \underbrace{\sum_{i=1}^{n} ln(y_{i}!)}_{\text{constant}}
\end{split}
$$

#### How to encode cyclical continuous features?

Some data is inherently cyclical. They have temporal structure. Time is a rich example of this: minutes, hours, seconds, day of week, week of month, month, season, and so on all follow cycles. Ecological features like tide, astrological features like position in orbit, spatial features like rotation or longitude, visual features like color wheels are all naturally cyclical.

Our problem is: how can we let our machine learning model know that a feature is cyclical?

A common method for encoding cyclical data is to transform the data into two dimensions using a sine and consine transformation. We can do that using the following transformations:

$$
x_{sin} = \sin(\frac{2 * \pi * x}{\max(x)})
$$

and

$$
x_{cos} = \cos(\frac{2 * \pi * x}{\max(x)})
$$

for hourly data,

```python
data['hour_sin'] = np.sin(2 * np.pi * data['hour']/23.0)
data['hour_cos'] = np.cos(2 * np.pi * data['hour']/23.0)
```

For example, if we do this for seconds in a day data:

```python
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
import numpy as np

def rand_times(n):
    """Generate n rows of random 24-hour times (seconds past midnight)"""
    rand_seconds = np.random.randint(0, 24*60*60, n)
    return pd.DataFrame(data=dict(seconds=rand_seconds))

n_rows = 1000

df = rand_times(n_rows)
# sort for the sake of graphing
df = df.sort_values('seconds').reset_index(drop=True)
df.head()

#    seconds
# 0   49
# 1   70
# 2  114
# 3  152
# 4  182

plt.figure(0)
df.seconds.plot();

seconds_in_day = 24*60*60

df['sin_time'] = np.sin(2*np.pi*df.seconds/seconds_in_day)
df['cos_time'] = np.cos(2*np.pi*df.seconds/seconds_in_day)

df.head()

plt.figure(1)
df.sin_time.plot()
plt.title('Sine')

#Notice that now, 5 minutes before midnight and 5 minutes after is 10 minutes apart!

#However, with just this sine transformation, you get a weird side-effect. 
#Notice that every horizontal line you draw across the graph touches two points. 
#So from this feature alone, it appears that midnight==noon, 1:15am==10:45am, and so on. 
#There is nothing to break the symmetry across the period. We really need two dimensions for a cyclical feature. 
# Cosine to the rescue!

plt.figure(2)
df.cos_time.plot()
plt.title('Cosine')
#With an additional out-of-phase feature (cos), the symmetry is broken. 
#Using the two features together, all times can be distinguished from each other.

#An intuitive way to show what we just did is to plot the two-feature transformation in 2D as a 24-hour clock. 
# The distance between two points corresponds to the difference in time as we expect from a 24-hour cycle. 
plt.figure(3)
df.plot.scatter('sin_time','cos_time').set_aspect('equal');

#We can feed the sin_time and cos_time features into our machine learning model, 
#and the cyclical nature of 24-hour time will carry over.
```

#### Why does bagging work so well for decision trees, but not for linear classifiers?

Bootstrap aggregating, also called bagging, is one of the first ensemble algorithms machine learning practitioners learn and is designed to improve the stability and accuracy of regression and classification algorithms. By model averaging, bagging helps to reduce variance and minimize overfitting. Although it is usually applied to decision tree methods, it can be used with any type of method. Some models have larger variance than others. Bagging works especially well for unstable, high variance base learners—algorithms whose predicted output undergoes major changes in response to small changes in the training data. This includes algorithms such as decision trees and KNN (when k is sufficiently small). However, for algorithms that are more stable or have high bias, bagging offers less improvement on predicted outputs since there is less variability (e.g., bagging a linear regression model will effectively just return the original predictions for large enough ensemble). 

Normal equations produce the estimates for the coefficient of a linear regression using ordinary least squares (OLS) method. Gauss-Markov Theorem states that these estimates are already BLUE which  is an acronym for best linear unbiased estimator.
In other words, the sampling distributions of regression coefficients are centered on the actual population value and are the tightest possible distributions. Therefore, the estimates, if the assumptions of the linear model are satisfied (homoskedasticity of the error term, no serial correlation of the errors, no exact multicollinearity, etc.) are already minimum variance and unbiased (in fact they are the maximum likelihood estimates.) Bagging is a procedure used to "balance out" the bias-variance tradeoff, but if OLS is BLUE, there is no need to employ a method like bagging.

#### Is decision tree a linear model?

Decision trees is a non-linear classifier like the neural networks, etc. It is generally used for classifying non-linearly separable data. Note however that it is a piecewise linear model: in each neighborhood (defined in a non-linear way), it is linear. In fact, the model is just a local constant. To see this in the simplest case, with one variable, $x$ and with one node $\theta$, the tree can be written as a linear regression:

$$
y_i = \alpha_1 1(x_i < \theta) + \alpha_2 1(x_i \geq \theta) + \epsilon_i
$$

Where $1(A)$ is the indicator function, taking value of $1$ if the event $A$ is true, and 0 otherwise.

#### In machine learning, how can we determine whether a problem is linear/nonlinear?

Most of the data in real world is not 2D and it is difficult to visualize more than 3D data easily, unless you use some methods to reduce the high dimensional data. Principal Component Analysis can do that for you but bringing it down to 2 dimensions will not be helpful.

So, one must start using some classification techniques. The thumb-rule is use the simple methods first (in accordance to Occam's razor) for e.g. a linear regression, perceptron (perceptron is mathematically proven method which is able to divide data correctly unless it is nonlinear) and SVM with linear kernel or you can choose some simple non-linear classifiers such as logistic regression, decision trees and naive Bayes. If your results are not good (high error, low accuracy), then, either your problem cannot be solved by linear classification methods or you may have to move to more complex non-linear classifiers, such as SVM with Kernels, Random Forest, Neural Network and so on.

#### What are the data augmentation techniques for images?

There are operations that can be easily applied to a given image to obtain one or more new images: flip, rotation, crop, color shift, noise addition, perspective change, contrast change, and information loss (for example, by randomly removing parts of image we can simulate situations when an object is recognizable but not entirely visible because of some visual obstacle.)

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/data_augmentation_for_images.png?raw=true)

In addition to these techniques, if you expect that the input images in your production system can come overcompressed, you can simulate the effect of overcompression by using some frequently used lossy compression methods and file formats, such as JPEG or GIF.

Only training data undergoes augmentation. Of course, it’s impractical to generate all these additional examples in advance and store them. In practice, the data augmentation techniques are applied to the original data on-the-fly during training

#### What are the data augmentation techniques for text?

One technique consists of replacing random words in a sentence with their exact or very close synonyms. For example, we can obtain several equivalent sentences from the following one:

The car stopped near a shopping mall. Some examples are:

* The automobile stopped near a shopping mall.
* The car stopped near a shopping center.
* The auto stopped near a mall.

A similar technique consists of using hypernyms instead of synonyms. A hypernym is a word that has more general meaning. For example, "mammal" is a hypernym for "whale" and "cat"; "vehicle" is a hypernym for "car" and "bus". From our example sentence above, by using hypernyms, we could obtain the following sentences:

* The vehicle stopped near a shopping mall.
* The car stopped near a building.

Another text data augmentation technique that works well is back translation. To obtain a new example from sentence text *t*
in English (it can be a sentence or a document), you translate it into language *l* using a machine translation system and then translate it back from *l* into English. If the text obtained by back translation is different from the original
text, you add it to the dataset by assigning the same label as the label of the original text.

A simple baseline for data augmentation is shuffling the text elements to create new text. For example, if we have labeled sentences and we want to get more, we can shuffle each sentence words to create a new sentence. This option is only valid for classification algorithms that don’t take into account the words order within a sentence. So in practice, we should tokenize each sentence into words. Then we shuffle those words and rejoin them to create new sentences.

If you represent words or documents in your dataset using word or document embeddings, you can apply slight Gaussian noise to randomly chosen features of an embedding to hopefully obtain a variation of the same word or document. You can tune the number of features to modify and the intensity of noise as hyperparameters by using the validation data.

Alternatively, to replace a given word *w* in the sentence, you can find *k* nearest neighbors to the word *w* in the word embedding space and generate *k* new sentences by replacing the word *w* by its respective neighbor. The nearest neighbors can be found using a metric such as **cosine similarity** or **Euclidean distance**. The choice of the metric, as well as the value of *k*, can be tuned as hyperparameters.

Similarly, if your problem is document classification and you have a large corpus of unlabeled documents and only a small corpus of labeled documents, you can do as follows. First, build document embeddings for all documents in your corpus. To do that you can use **doc2vec** or any other technique of document embedding. Then, for each labeled document d in your dataset, find k closest unlabeled documents in the document embedding space and label them with the same label as d. Again, tune k on the validation data.

The paper "EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks" by Jason Wei and Kai Zou (https://www.aclweb.org/anthology/D19-1670.pdf) also mentions some basic methods:

For a given sentence in the training set, we randomly choose and perform one of the following operations:

1. **Synonym Replacement (SR)**: Randomly choose n words from the sentence that are not stop words. Replace each of these words with one of its synonyms chosen at random.
2. **Random Insertion (RI)**: Find a random synonym of a random word in the sentence that is not a stop word. Insert that synonym into a random position in the sentence. Do this n times.
3. **Random Swap (RS)**: Randomly choose two words in the sentence and swap their positions. Do this n times.
4. **Random Deletion (RD)**: Randomly remove each word in the sentence with probability p

Since long sentences have more words than short ones, they can absorb more noise while maintaining their original class label. To compensate, the authors vary the number of words changed, n, for SR, RI, and RS based on the sentence length l with the formula $n = \alpha l$, where $\alpha$ is a parameter that indicates the percent of the words in a sentence are changed
(they use $p = \alpha$ for RD). Furthermore, for each original sentence, we generate naug augmented sentences. Examples of augmented sentences are shown in

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Screen%20Shot%202020-04-28%20at%2009.45.53.png?raw=true)

#### Why is data augmentation classified as a type of regularization?

Regularization (traditionally in the context of shrinkage) adds prior knowledge to a model; a prior, literally, is specified for the parameters. Augmentation is also a form of adding prior knowledge to a model; e.g. images are rotated, which you know does not change the class label. Increasing training data (as with augmentation) decreases a model's variance. Regularization also decreases a model's variance. They do so in different ways, but ultimately both decrease rgeneralization error (also known as the out-of-sample error) but not the training error.

#### Why using probability estimates and non-thresholded decision values give different AUC values?

The Receiver Operating Characteristic (ROC) Curve is computed by plotting the True Positive Rate (TPR) against the False Positive Rate (FPR) at uniformly distributed threshold values from 0 to 1. The Area Under the Curve (AUC) is then calculated to turn this into a numerical score. AUC is a ranking metric, meaning that it cares only about the order of predictions. Having probabilities instead of two-class predictions (0/1 for binary classification which are scores) gives it more granularity to rank the predictions. Different thresholds are calculated inside `roc_auc_score()` on the basis of this prediction probabilities. 

Scikit-Learn's `predict` returns only one class or the other. Then you compute a ROC with the results of predict on a classifier, there are only three thresholds (0, 1 and 2, see below to find where 2 comes from!). Your ROC curve looks like this:

```
      ..............................
      |
      |
      |
......|
|
|
|
|
|
|
|
|
|
|
|
```

Meanwhile, `predict_proba()` method returns an entire range of probabilities, so now you can put more than three thresholds on your data.

```
             .......................
             |
             |
             |
          ...|
          |
          |
     .....|
     |
     |
 ....|
.|
|
|
|
|
```
Hence different area. See a basic example below:

```python
X = df.iloc[:,2:4].values
y = df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
params = {'n_estimators': 100, 'criterion': 'entropy', 'oob_score': True, 'random_state': 42}
classifier = RandomForestClassifier(**params)
classifier.fit(X_train, y_train)

print(classifier.oob_score_)
# 0.8833333333333333

y_pred = classifier.predict(X_test)
y_pred_prob = classifier.predict_proba(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)
print(cm)
# [[57  6]
#  [ 4 33]]

test_acc = accuracy_score(y_test, y_pred)
print(test_acc)
#0.9

from sklearn.metrics import roc_curve, roc_auc_score
_, _, threshold1 = roc_curve(y_test, y_pred)
print(threshold1)
#array([2, 1, 0])

_, _, threshold2 = roc_curve(y_test, y_pred_prob[:,1])
print(threshold2)
# array([2.  , 1.  , 0.99, 0.98, 0.97, 0.94, 0.93, 0.89, 0.88, 0.86, 0.73,
#        0.71, 0.62, 0.56, 0.53, 0.48, 0.36, 0.11, 0.1 , 0.05, 0.04, 0.02,
#        0.01, 0.  ])

print(roc_auc_score(y_test, y_pred))
#0.8983268983268984

print(roc_auc_score(y_test, y_pred_prob[:,1]))
#0.9586014586014586

# Different AUC values!
```

The question is now, sometimes you can get a scalar "2" as a threshold. The documentation explains this as `thresholds : array, shape = [n_thresholds] Decreasing thresholds on the decision function used to compute fpr and tpr. thresholds[0] represents no instances being predicted and is arbitrarily set to max(y_score) + 1.`

Ok, so, how does number of thresholds get chosen in `roc_curve` function in scikit-learn?

By definition, a ROC curve represent all possible thresholds in the interval $(− \infty, + \infty)$.

This number is infinite and of course cannot be represented with a computer. Fortunately when you have some data you can simplify this and only visit a limited number of thresholds. This number corresponds to the number of unique values in the data + 1, or something like:

```
n_thresholds = len(np.unique(x)) + 1
```

where `x` is the array holding your target scores (`y_score`).

#### What is Minkowski distance? How is it related to Manhattan distance and Euclidean distance

The Minkowski distance is a metric in a normed vector space which can be considered as a generalization of both the Euclidean distance and the Manhattan distance. 

The Minkowski distance of order $p$ (where $p$ is an integer) between two points, $X=(x_{1},x_{2},\ldots ,x_{n})$ and $Y=(y_{1},y_{2},\ldots ,y_{n})\in {\mathbb  {R}}^{n}$ is defined as:

$$
D\left(X,Y\right) = \left(\sum_{i=1}^{n} \mid x_{i} - y_{i}\mid^{p}\right)^{\frac {1}{p}}
$$

Minkowski distance is typically used with {\displaystyle p}p being 1 or 2, which correspond to the Manhattan distance and the Euclidean distance, respectively. 

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/2880px-2D_unit_balls.svg.png?raw=true)

#### What to do before clustering?

You can't just run clustering on data and expect to get anything meaningful. Clustering quality depends heavily on the quality of your distance function, so this needs to be very carefully chosen first, taking into account what kind of data you have available. Many clustering algorithms (like DBSCAN or K-Means) use a distance measurement to calculate the similarity between observations. Because of this, certain clustering algorithms will perform better with continuous attributes. However, if you have categorical data, you can one-hot encode the attributes or use a clustering algorithm built for categorical data, such as K-Modes.

If you have outliers in the data, either you could exclude them or use another method that is less senitive because for example, K-means clustering algorithm is based on centroids for each cluster, so the mean could be sensitive to outliers.

If you have a lot of features, curse of dimensionality might be an issue. Therefore, an appropriate dimensionality reduction techniques must be used. The real problem with doing this is that it becomes more difficult to interpret since you are looking at clusters in terms of transformed and factorized, as opposed to the original variables themselves.

You could decide the number of the clusters or for which values of $K$ you are going to perform the clustering algorithm.

Most of the clustering methods is prone to initial seeding i.e. random initialization of centroids which is required to kick-off iterative clustering process. Bad initialization may end up getting bad clusters.

Distance computation in many methods weights each dimension equally and hence care must be taken to ensure that unit of dimension should not distort relative near-ness of observations. Normalization for Gaussian distribution, log-transform for power-law distribution that clumps data at the low end, or transform the data into quantiles if data does not conform to a Gaussian or power-law distribution.

If your dataset has examples with missing values for a certain feature but such examples occur rarely, then you can remove these examples. If such examples occur frequently, we have the option to either remove this feature altogether, or to predict the missing values from other examples by using a machine learning model. For example, you can infer missing numerical data by using a regression model trained on existing feature data.

In order to trust the clustering algorithm results, you must have a method for measuring the algorithm's performance. Clustering algorithm performance can be validated with either internal or external validation metrics. There are multiple metrics in the literature. You need to find which one is suitable for you. 

#### How to cluster only categorical data with K-means?

The basic concept of K-means stands on mathematical calculations (means, Euclidean distances). K-means explicitly minimizes within-cluster variance (squared distances from the mean) as defined in Euclidean space. But what if our data is non-numerical or, in other words, categorical? 

Euclidean distance is not defined for categorical data; therefore, K-means cannot be used directly.

We could think of transforming our categorical values in numerical values and eventually apply K-means using cosine similarity. However, you may run into curse of dimensionality issues due to the fact that you are increasing dimensionality. 

Another solution lies in the K-modes algorithm. K-modes is an extension of K-means. The distance metric used for K-modes is the Hamming distance (there are some other distance metrics proposed in the literature). The K-modes algorithm tries to minimize the sum of within-cluster Hamming distance from the mode of that cluster (The mode of a set of data values is the value that appears most often), summed over all clusters. The procedure is similar to K-means. First, we choose a number of clusters (K), and K cluster-mode vectors are chosen at random (or according to accepted heuristics). Step 1: Observations are assigned to the closest cluster mode by Hamming distance. Step 2: New cluster modes are calculated, each from the observations associated with an previous cluster mode. Steps 1 and 2 are repeated until the cluster modes stabilize. As with K-means, this stable condition could be due to a local minimum in the cost function.

Various clustering algorithms have been developed to group data into clusters in diverse domains. However, these clustering algorithms work effectively either on pure numeric data or on pure categorical data, most of them perform poorly on mixed categorical and numeric data types. For numerical and categorical data, another extension of these algorithms exists, basically combining K-means and K-modes. It is called K-prototypes, which mixes the Hamming distance for categorical features and the Euclidean distance for numeric features.

If you have a data which consists of both categorical and continuous variables, you can use Gower distance, which is a composite measure and apply hierarchical clustering (because it allows to select from a great many distance functions). Gower distance takes quantitative, ordinal, binary and nominal variables. However, be careful! Some methods of agglomeration (Ward's method) will call for (squared) Euclidean distance only.

If the computational costs of hierarchical clustering are too large, you can consider an alternative clustering method such as K-prototypes.

#### What is K-medoids algorithm?

K-medoids is another partitioning algorithm. Both K-means and K-medoids algorithms are breaking the dataset up into $K$ groups. Also, they are both trying to minimize the distance between points of the same cluster and a particular point which is the center of that cluster.

K-medoid is based on medoids (which is a point that belongs to the dataset) calculating by minimizing the absolute distance between the points and the selected centroid, rather than minimizing the square distance. As a result, it is more robust to noise and outliers than k-means since an object with an extremely large value may substantially distort the distribution of the data.

Note that a medoid is not equivalent to a median, a geometric median, or centroid. 

Consider this 1-dimensional example:

```
[1, 2, 3, 4, 100000]
```

Both the median and medoid of this set are 3. The mean is 20002.

In contrast to the K-means algorithm, K-medoids algorithm chooses data points as centers that belong to the dataset because a medoid has to be a member of the set, a centroid does not. This is true because most of the time, a centroid does not correspond to a data point. A medoid, by definition, must be a data point.

The real problem with doing this is that it becomes more difficult to interpret since you are looking at clusters in terms of transformed and factorized, as opposed to the original variables themselves

The most common implementation of K-medoids clustering algorithm is the Partitioning Around Medoids (PAM) algorithm. PAM algorithm uses a greedy search which may not find the global optimum solution. The main drawback of K-medoids is that it is much more expensive because PAM usually takes much longer to run than K-means. As it involves computing all pairwise distances to find the medoids. Thus, it's time consuming and computer intensive.

#### How to choose number of clusters in clustering analysis?

Determining the right number of clusters in a data set is important, not only because some clustering algorithms like k-means requires such a parameter, but also because the appropriate number of clusters controls the proper granularity of cluster analysis. determining the number of clusters is far from easy, often because the right number is ambiguous. The interpretations of the number of clusters often depend on the shape and scale of the distribution in a data set, as well as the clustering resolution required by a user. There are many possible ways to estimate the number of clusters. Here, we briefly introduce some simple yet popularly used and effective methods.

We often know the value of K. In that case we use the value of K. In general, there is no method for determining exact value of K.

A simple experienced method is to set the number of clusters to about $\sqrt{n/2}$ for a data set of $n$ points. In expectation, each cluster has $\sqrt{2n}$ points. Another approach is the Elbow Method. We run the algorithm for different values of K (say K = 1 to 10) and plot the K values against WCSSE (Within Cluster Sum of Squared Errors). WCSS is also called "inertia". Then, select the value of K that causes sudden drop in the sum of squared distances, i.e., for the elbow point as shown in the figure.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/elbow_method_kmeans.png?raw=true)

HOWEVER, it is important to note that inertia heavily relies on the assumption that the clusters are convex (of spherical shape).

A number of other techniques exist for validating K, including cross-validation, information criteria, the information theoretic jump method, the silhouette method (we want to have high silhouette coefficient for the number of clusters we want to use), and the G-means algorithm. In addition, monitoring the distribution of data points across groups provides insight into how the algorithm is splitting the data for each K. Some researchers also use Hierarchical clustering first to create dendrograms and identify the distinct groups from there.

#### How to select a clustering method? 

One of the biggest issue with cluster analysis is that we may happen to have to derive different conclusion when base on different clustering methods used (including different linkage methods in hierarchical clustering).You cannot know in advance which clustering algorithm would be better. There are many clustering algorithms in the literature because the notion of the notion "cluster" cannot be defined precisely. Clustering is in the eye of beholder. It totally depends on domain specific knowledge. 

#### What is the exact difference between error and residual?

For all practical purposes, and in a machine learning context, these two terms are treated as synonyms. The term "residual" is due to the origins of linear regression from statistics; since the term "error" in statistics had (has) a different meaning that in today's ML, a different term was needed to declare the difference between the estimated (predicted) values of a dependent variable and its observed ones, hence the "residual".

You can find more details in the Wikipedia entry for [Errors and residuals](Errors and residuals) (notice the plural); quoting:

> In statistics and optimization, errors and residuals are two closely related and easily confused measures of the deviation of an observed value of an element of a statistical sample from its "theoretical value". The **error** (or disturbance) of an observed value is the deviation of the observed value from the (unobservable) true value of a quantity of interest (for example, a population mean), and the **residual** of an observed value is the difference between the observed value and the estimated value of the quantity of interest (for example, a sample mean). The distinction is most important in regression analysis, where the concepts are sometimes called the regression errors and regression residuals and where they lead to the concept of studentized residuals.

Keep in mind that the above come from the statistics realm; in a ML context, we use the term "error" (singular) to mean the difference between predicted and observed values, and the term "residual(s)" is practically almost never used ...


## Deep Learning

#### What is structured and unstructured data?

Many types of machine learning algorithm require structured, tabular data as input, arranged into columns of features that describe each feature, For example, a person's age, income and number of website visits in the last month are all features that could help to predict if the person will subscribe to a particular online service in the coming month. We could use a structured table of these features to train a logistic regression, random forest or XGBoost model to predict the binary response variable - did the person subscribe (1) or not (0)?

Unstructured data refers to any data that is not naturally arranged into columns of features such as images, audio and text. There is of course spatial structure to an image, temporal structure to a recording, and both spatial and temporal structure to video data, but since the data does not arrive in columns of features, it is considered unstructured. When our data is unstructured, individual pixels, frequencies, or characters are almost entirely uninformative. For example, knowing that pixel 234 of an image is a muddy shade of brown does not really help to identify if the image is of a house or a cat. A deep learning model can learn how to build high-level informative features by itself, directly from the unstructured data.

#### What is an epoch, a batch and an iteration?

A batch is the complete dataset. Its size is the total number of training examples in the available dataset.

Mini-batch size is the number of training examples in one forward/backward pass. The higher the batch size, the more memory space you'll need.

A Mini-batch is a small part of the dataset of given mini-batch size.
One epoch = one forward pass and one backward pass of all the training examples

Number of iterations = An iteration describes the number of times a batch of data passed through the algorithm, number of passes, each pass using [batch size] number of examples. To be clear, one pass = one forward pass + one backward pass (we do not count the forward pass and backward pass as two different passes).

_Example_: if you have 1000 training examples, and your batch size is 500, then it will take 2 iterations to complete 1 epoch.

#### What is deep learning?

Deep Learning is an area of Machine Learning that attempts to mimic the activity in the layers of neurons in the brain to learn how to recognize the complex patterns in the data. The "deep" in deep learning refers to the large number of layers of neurons in contemporary ML models that help to learn rich representations of data to achieve better performance gains.

#### Why are deep networks better than shallow ones?

Both the Networks, be it shallow or deep are capable of approximating any function. However, a shallow network works with only a few features, as it can’t extract more. 

Deep learning's main distinguishing feature from "Shallow Learning" is that deep learning methods derive their own features directly from data (feature engineering), while shallow learning relies on the handcrafted features based upon heuristics of the target problem. Deep learning architecture can learn representations and features directly from the input with little to no prior knowledge. 

Deep networks have several hidden layers often of different types so they are able to build or create or extract better features/concepts than shallow models with fewer parameters. It makes your network more eager to recognize certain aspects of input data. 

Deep learning can learn multiple levels of representations that correspond to different levels of abstraction; the levels form  a hierarchy of concepts. It is all about learning hierarchical representations: low-level features, mid-level representations, high level concepts. Higher level concepts are derived from lower level features. Animals and humans do learn this way with simpler concepts earlier in life, and higher-level abstractions later, expressed in terms of previously learned concepts.

#### What is a perceptron?

Given a training data $(x_{i}, y_{i}), i=1,2, \ldots , n$ and $x_{i} \in \mathbb{R}^{m}$ and the target variable $y_{i} \in \\{-1, 1\\}$, we try to lean a perceptron classifier.

Next we define an activation function $g(z)$ which takes the input values $\mathbf{x}$ and weights $\mathbf{w}$ as input ($\mathbf{z} = w_1x_{1} + \dots + w_mx_{m}$) and if $g(\mathbf{z})$ is greater than a defined threshold $\theta$ we predict $+1$ and $-1$ otherwise; in this case, this activation function $g$ is an alternative form of a simple "unit step function", which is sometimes also called "Heaviside step function". 

$$
g(\mathbf{z}) =\begin{cases}
    1 & \text{if }\mathbf{z} \ge \theta\\
    -1 & \text{otherwise}.
  \end{cases}
$$

where

$$\mathbf{z} =  w_1x_{1} + \dots + w_mx_{m} = \sum_{j=1}^{m} w_{j}x_{j} \\ = \mathbf{w}^T\mathbf{x}$$

$\mathbf{w}$ is the weight vector, and $\mathbf{x}$ is an $m$-dimensional sample from the training dataset:

$$
\mathbf{w} = \begin{bmatrix}
    w_{1}  \\
    \vdots \\
    w_{m}
\end{bmatrix}
\quad  \mathbf{x} = \begin{bmatrix}
    x_{1}  \\
    \vdots \\
    x_{m}
\end{bmatrix}
$$

![](https://sebastianraschka.com/images/blog/2015/singlelayer_neural_networks_files/perceptron_unit_step.png)

In order to simplify the notation, we bring $\theta$ to the left side of the equation and define $w_{0} = - \theta$ and $x_{0} = 1$.

$$
\begin{equation}
 g({\mathbf{z}}) =\begin{cases}
    1 & \text{if } \mathbf{z} \ge 0\\
    -1 & \text{otherwise}.
  \end{cases}
\end{equation} 
$$

where

$$\mathbf{z} = w_0x_{0} + w_1x_{1} + \dots + w_mx_{m} = \sum_{j=0}^{m} w_{j}x_{j} \\ = \mathbf{w}^T\mathbf{x}$$

and 

$$
\mathbf{w} = \begin{bmatrix}
    w_{0} \\
    w_{1}  \\
    \vdots \\
    w_{m}
\end{bmatrix}
\quad  \mathbf{x} = \begin{bmatrix}
    1 \\
    x_{1}  \\
    \vdots \\
    x_{m}
\end{bmatrix}
$$

Perceptron rule is fairly simple and can be summarized by the following steps:

1. Initialize the weights to 0 or small random numbers.
2. For each training sample $\mathbf{x^{(i)}}$:
  1. Calculate the output value.
  2. Update the weights.
  
The output value is the class label predicted by the unit step function that we defined earlier ($output = g(z)$) and the weight update is as $w_{j} := w_{j} + \Delta w_{j}$.

The value for updating the weights at each increment is calculated by the learning rule

$$
\Delta w_j = \alpha \; (\text{target}^{(i)} - \text{output}^{(i)})\;x_{j}^{(i)}
$$

where $\alpha$ is the learning rate , "target" is the true class label (either $-1$ or $+1$), and the "output" is the predicted class label.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/perceptron_update_rule.jpeg?raw=true)

It is important to note that the convergence of the perceptron is only guaranteed if the two classes are linearly separable.

#### How to solve OR problem with perceptron?

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Image.jpeg?raw=true)

#### What are the shortcomings of a single layer perceptron?

There are two major problems:

* Single-layer Perceptrons cannot classify non-linearly separable data points.
* Complex problems, that involve a lot of parameters cannot be solved by Single-Layer Perceptrons

#### What does a neuron compute?
An artificial neuron calculates a "weighted sum" of its input, adds a bias ($z = Wx+b$), followed by an activation function.

#### What is the role of activation functions in a Neural Network?

The goal of an activation function is to introduce nonlinearity into the neural network so that it can learn more complex function i.e. converts the processed input into an output called the activation value. Without it, the neural network would be only able to learn function which is a linear combination of its input data.

If we do not apply an activation function then the output signal would simply be a simple linear function. A linear function is just a polynomial of one degree. 

$$
z=\beta_{0}1 + \beta_{1}x_{1} + \beta_{2}x_{2} + \ldots +\beta_{p}x_{p}\,\,\,\,\, \mathbf{(a)}
$$

Each input variable $x_{j}$ is represented with a node and each parameter $\beta_{j}$ with a link. Furthermore, the output $z$ is described as the sum of all terms $\beta_{j}x_{j}$. Note that we use 1 as the input variable corresponding to the bias term (a.k.a. offset term) $\beta_{0}$. 

To describe _nonlinear_ relationship between $x = \left[1\,\,x_{1}\,\,x_{2}\,\, \ldots \,\,x_{p}\right]^{T}$ and $z$, we introduce a nonlinear scalar-valued function called _activation function_ $\sigma: \mathbb{R} \to \mathbb{R}$. The linear regression model is now modified into a _generalized_ linear regression model where the linear combination of the inputs is squashed through the (scalar) activation function. 

$$
z = \sigma \left( \beta_{0}1 + \beta_{1}x_{1} + \beta_{2}x_{2} + \ldots +\beta_{p}x_{p} \right)\,\,\,\,\, \mathbf{(b)}
$$

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/glm_activation.png?raw=true)

Now, a linear equation is easy to solve but they are limited in their complexity and have less power to learn complex functional mappings from data. A neural network without activation function would simply be a linear regression model, which has limited power and does not performs good most of the times. Therefore, we make further extensions to increase the generality of the model by sequentially stacking these layers. 

Another important feature of an activation function is also that it should be differentiable. We need it to be this way so as to perform backpropogation optimization strategy while propogating backwards in the network to compute gradients of error (loss) with respect to parameters (weights/biases) and then accordingly optimize weights using Gradient Descent algorithm or any other optimization technique to reduce error.

####  How many types of activation functions are there ?

#### What does the term saturating nonlinearities mean?

A saturating activation function squeezes the input. 

The Rectified Linear Unit (ReLU) activation function, which is defined as $f(x)=max(0,x)$ is non-saturating because $\lim_{z\to+\infty} f(z) = +\infty$

![](https://raw.githubusercontent.com/mmuratarat/mmuratarat.github.io/master/_posts/images/relu.png)

The sigmoid activation function, which is defined as $f(x) = \frac{1}{1+e^{-x}}$ is saturating, because it squashes real numbers to range between $[0,1]$:

![](https://raw.githubusercontent.com/mmuratarat/mmuratarat.github.io/master/_posts/images/sigmoid.png)

The tanh (hyperbolic tangent) activation function, which is defined as $tanh(x)= \dfrac{sinh(x)}{cosh(x)} =\dfrac{e^{x}-e^{-x}}{e^{x}+e^{-x}} = 2 \times sigmoid(2x)-1$, is saturating, because it squashes real numbers to range between $[-1,1]$:

![](https://raw.githubusercontent.com/mmuratarat/mmuratarat.github.io/master/_posts/images/tanh.png)

#### Derivative of ReLU function at zero does not exist. Is not it a problem for backpropagation?

Yes, correct. Derivative of ReLU function at $x=0$ does not exist.

A function is only differentiable if the derivative exists for each value in the function's domain (for instance, at each point). One criterion for the derivative to exist at a given point is continuity at that point. However, the continuity is not sufficient for the derivative to exist. For the derivative to exist, we require the left-hand and the right-hand limit to exist and be equal.

General definition of the derivative of a continuous function $f(x)$ is given by:

$$
f^{\prime}(x) = \frac{d f}{dx} = \lim_{h \rightarrow 0} \frac{f(x+h) - f(x)}{h}
$$

where $\lim_{h \rightarrow 0}$ means "as the change in h becomes infinitely small (for instance h approaches to zero)".

Let's get back to ReLU function. If we substitute the ReLU equation into the limit definition of derivative above:

$$
f^{\prime} (x) = \frac{d f}{dx} = \lim_{x \rightarrow 0} \frac{max(0, x + \Delta x) - max(0, x)}{\Delta x}
$$

Next, let us compute the left- and right-side limits. Starting from the left side, where $\Delta x$ is an infinitely small, negative number, we get,

$$
f^{\prime} (0) = \lim_{x \rightarrow 0^{-}} \frac{0 - 0}{\Delta x} = 0.
$$

And for the right-hand limit, where $\Delta x$ is an infinitely small, positive number, we get:

$$
f^{\prime} (0) = \lim_{x \rightarrow 0^{+}} \frac{0+\Delta x - 0}{\Delta x} = 1.
$$

The left- and right-hand limits are not equal at $x=0$; hence, the derivative of ReLU function at $x=0$ is not defined. 

So we can say that ReLU is a convex function that has subdifferential at $x > 0$ and $x < 0$. The subdifferential at any point $x < 0$ is the singleton set $\{0\}$, while the subdifferential at any point $x > 0$ is the singleton set $\{1\}$. ReLU is actually not differentiable at $x = 0$, but it has subdifferential $[0,1]$. Any value in that interval can be taken as a subderivative, and can be used in gradient descent if we generalize from gradient descent to "subgradient descent". In the implementation, we choose the subgradient at $x = 0$ to be $0$ for simplicity. 

However, some people explain with "theory of limits" that the derivative at $x = 0$ is nearly zero and so we can take it while doing SGD but that’s mathematically wrong. Anything between the interval $[0,1]$ can be taken as subderivative to compute SGD. In built-in library functions (for example: `tf.nn.relu()`) derivative at $x = 0$ is taken zero to ensure a sparse matrix, otherwise if you write your own function for ReLU, you can code it anything random between the interval $[0,1]$ for $x = 0$. That would be mathematically fine!

#### What is a Multi-Layer-Perceptron

#### What is Deep Neural Network?

Deep neural networks can model complicated relationships and is one of the state of art methods in machine learning as of today.

We enumerate layers with index $l$. Each _layer_ is parameterized with a weight matrix $\mathbf{W}^{(l)}$ and a bias vector $\mathbf{b}^{(l)}$. For example, $\mathbf{W}^{(1)}$ and $\mathbf{b}^{(1)}$ belong to layer $l=1$; $\mathbf{W}^{(2)}$ and $\mathbf{b}^{(2)}$ belong to layer $l=2$ and so forth. We also have multiple layers of hidden units denoted by $\mathbf{h}^{(l-1)}$. Each such layer consists of $M_{l}$ hidden units, 

$$
\mathbf{h}^{(l)} = \left[\mathbf{h}^{(l)}_{1}, \ldots, \mathbf{h}^{(l)}_{M_{l}} \right]^{T}
$$

where the dimensions $M_{1}, M_{2}, \ldots$ can be different for different layers.

Each layer maps a hidden layer $\mathbf{h}^{(l-1)}$ to the next hidden layer $\mathbf{h}^{(l)}$ as:

$$
\mathbf{h}^{(l)} = \sigma \left(\mathbf{W}^{(l)T}\mathbf{h}^{(l-1)} + \mathbf{b}^{(l)T} \right)
$$

This means that the layers are stacked such that the output of the first layer $\mathbf{h}^{(1)}$ (the first layer of hidden units) is the input to the second layer, the output of the second layer $\mathbf{h}^{(2)}$ (the second layer of hidden units) is the input to the third layer, etc. By stacking multiple layers we have constructed a deep neural network. A deep neural network of $L$ layers can mathematically be described as:

$$
\begin{split}
\mathbf{h}^{(1)} &= \sigma \left(\mathbf{W}^{(1)T}\mathbf{x} + \mathbf{b}^{(1)T} \right)\\
\mathbf{h}^{(2)} &= \sigma \left(\mathbf{W}^{(2)T}\mathbf{h}^{(1)} + \mathbf{b}^{(2)T} \right)\\
&\cdots \\
\mathbf{h}^{(L-1)} &= \sigma \left(\mathbf{W}^{(L-1)T}\mathbf{h}^{(L-2)} + \mathbf{b}^{(L-1)T} \right)\\
z &= \mathbf{W}^{(L)T}\mathbf{h}^{(L-1)} + \mathbf{b}^{(L)T}\\
\end{split}
$$

The weight matrix $\mathbf{W}^{(1)}$ for the first layer $l = 1$ has the dimension $p \times M_{1}$ and the corresponding bias vector $\mathbf{b}^{(1)}$ the dimension $1 \times M_{1}$. In deep learning it is common to consider applications where also the output is multi-dimensional $\mathbf{z} = \left[z_{1}, \ldots, z_{K}\right]^{T}$. This means that for the last layer the weight matrix $\mathbf{W}^{(L)}$ has the dimension $M_{L−1} \times K$ and the bias vector $\mathbf{b}^{(L)}$ the dimension $1 \times K$. For all intermediate layers $l = 2, \ldots, L − 1$, $\mathbf{W}^{(l)}$ has the dimension $M_{l−1} \times M_{l}$ and the corresponding bias vector $1 \times M_{l}$.

The number of inputs $p$ and the number of outputs $K$ (number of classes) are given by the problem, but the number of layers
$L$ and the dimensions $M_{1}, M_{2},\ldots$ are user design choices that will determine the flexibility of the model.

#### What is softmax function and when we use it?

The softmax function is used in various multiclass classification methods. It takes an un-normalized vector, and normalizes it into a probability distribution. It is often used in neural networks, to map the non-normalized output to a probability distribution over predicted output classes. It is a function which gets applied to a vector in $z \in R^{K}$ and returns a vector in $[0,1] ^{K}$ with the property that the sum of all elements is 1, in other words, the softmax function is useful for converting an arbitrary vector of real numbers into a discrete probability distribution:

$$
softmax(z_j) = \frac{e^{z_{j}}}{\sum_{k=1}^K e^{z_{k}}} \;\;\;\text{ for } j = 1, \dots, K
$$

where $\mathbf{z} = \left[z_{1}, \ldots, z_{K}\right]^{T}$. The inputs to the softmax function, i.e., the variables $z_{1}, z_{2}, \ldots, z_{K}$ are referred to as _logits_.

Intiutively, the softmax function is a "soft" version of the maximum function. A "hardmax" function (i.e. argmax) is not differentiable. The softmax gives at least a minimal amount of probability to all elements in the output vector, and so is nicely differentiable. Instead of selecting one maximal element in the vector, the softmax function breaks the vector up into parts of a whole (1.0) with the maximal input element getting a proportionally larger chunk, but the other elements get some of it as well. Another nice property of it, the output of the softmax function can be interpreted as a probability distribution, which is very useful in Machine Learning because all the output values are in the range of (0,1) and sum up to $1.0$. This is especially useful in multi-class classification because we often want to assign probabilities that our instance belong to one of a set of output classes.

For example, let's consider we have 4 classes, i.e. $K=4$, and unscaled scores (logits) are given by $[2,4,2,1]$. The simple argmax function outputs $[0,1,0,0]$. The argmax is the goal, but it's not differentiable and we can't train our model with it. A simple normalization, which is differentiable, outputs the following probabilities $[0.2222,0.4444,0.2222,0.1111]$. That's really far from the argmax! Whereas the softmax outputs $[0.1025,0.7573,0.1025,0.0377]$. That's much closer to the argmax! Because we use the natural exponential, we hugely increase the probability of the biggest score and decrease the probability of the lower scores when compared with standard normalization. Hence the "max" in softmax.

Softmax is fundamentally a vector function. It takes a vector as input and produces a vector as output. In other words, it has multiple inputs and outputs.

####  What is the cost function? 

In predictive modeling, cost functions are used to estimate how badly models are performing. Put it simply, a cost function is a measure of how wrong the model is in terms of its ability to estimate the relationship between X and y. This is typically expressed as a difference or distance between the predicted value and the actual value. The cost function (you may also see this referred to as loss or error) can be estimated by iteratively running the model to compare estimated predictions against "ground truth", i.e., the known values of $y$.

The objective here, therefore, is to find parameters, weights/biases or a structure that minimizes the cost function.

The terms cost function and loss function are synonymous, some people also call it error function.

However, there are also some different definitions out there. The loss function computes the error for a single training example, while the cost function will be average over all data points.

**Regression Problem**
A problem where you predict a real-value quantity.

1. Output Layer Configuration: One node with a linear activation unit.
2. Loss Function: Mean Squared Error (MSE).

**Binary Classification Problem**
A problem where you classify an example as belonging to one of two classes. The problem is framed as predicting the likelihood of an example belonging to class one, e.g. the class that you assign the integer value 1, whereas the other class is assigned the value 0.

1. Output Layer Configuration: One node with a sigmoid activation unit.
2. Loss Function: Binary Cross-Entropy, also referred to as Logarithmic loss.

**Multi-Class Classification Problem**
A problem where you classify an example as belonging to one of more than two classes. The problem is framed as predicting the likelihood of an example belonging to each class.

1. Output Layer Configuration: One node for each class using the softmax activation function.
2. Loss Function: Categorical Cross-Entropy.

#### What is cross-entropy? How we define the cross-entropy cost function?

Entropy is a measure of the uncertainty associated with a given distribution $p(y)$ with $K$ distinct states. Calculating the information for a random variable is called "information entropy", "Shannon entropy", or simply "entropy". When we have $K$ classes, we compute the entropy of a distribution, using the formula below

$$
H(p) = - \sum_{k=1}^{K} p(y_{k}) \log p(y_{k})
$$

(This can also be thought as in the following. There are K distinct events. Each event $K$ has probability $p(y_{k})$)

If we know the true distribution of a random variable, we can compute its entropy. However, we cannot always know the true distribution. That is what Machine Learning algorithms do. We try to approximate the true distribution with an other distribution, say, $q(y)$.

If we compute entropy (uncertainty) between these two (discrete) distributions, we are actually computing the cross-entropy between them:

$$
H(p, q) = -\sum_{k=1}^{K} p(y_{k}) \log q(y_{k})
$$

If we can find a distribution $q(y)$ as close as possible to $p(y)$, values for both cross-entropy and entropy will match as well. However, this is not the always case. Therefore, cross-entropy will be greater than the entropy computed on the true distribution.

$$
H(p,q)−H(p) > 0
$$

This difference between cross-entropy and entropy is called _Kullback-Leibler Divergence_.

The Kullback-Leibler Divergence,or KL Divergence for short, is a measure of dissimilarity between two distributions.

$$
\begin{split}
D_{KL} (p || q) = H(p, q) - H(p) &= \mathbb{E}_{p(y_{k})} \left [ \log \left ( \frac{p(y_{k})}{q(y_{k})} \right ) \right ] \\
&= \sum_{k=1}^{K} p(y_{k}) \log\left[\frac{p(y_{k})}{q(y_{k})}\right] \\
&=\sum_{k=1}^{K} p(y_{k}) \left[\log p(y_{k}) - \log q(y_{k})\right]
\end{split}
$$

This means that, the closer $q(y)$ gets to $p(y)$, the lower the divergence and consequently, the cross-entropy will be. In other words, KL divergence gives us "distance" between 2 distributions, and that minimizing it is equivalent to minimizing cross-entropy. Minimizing cross-entropy will make $q(y)$ converge to $p(y)$, and $H(p,q)$ itself will converge to $H(p)$. Therefore, we need to approximate to a good distribution by using the classifier.

Now, for one particular data point, if $p \in \\{y, 1−y\\}$ and $q \in \\{\hat{y} ,1−\hat{y}\\}$, we can re-write cross-entropy as:

$$
H(p, q) = -\sum_{k=1}^{K=2} p(y_{k}) \log q(y_{k}) =-y\log \hat{y}-(1-y)\log (1-\hat{y})
$$

which is nothing but logistic loss.

The final step is to compute the average of all points in both classes, positive and negative, will give binary cross-entropy formula.

$$
L(\theta) = - \frac{1}{n} \sum_{i=1}^{n}  \left[y_{i} \log (p_i) + (1-y_{i}) \log (1- p_{i}) \right]
$$

where $i$ indexes samples/observations, n is the number of observations, where $y$ is the label ($1$ for positive class and $0$ for negative class) and $p(y)$ is the predicted probability of the point being positive for all $n$ points. In the simplest case, each $y$ and $p$ is a number, corresponding to a probability of one class.

Multi-class cross entropy formula is as follows:

$$
L(\theta) = - \frac{1}{n} \sum_{i=1}^{n}  \sum_{j=1}^{K} \left[y_{ij} \log (p_{ij}) \right]
$$

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

One can easily see that $n_{1} + n_{2} + \ldots + n_{K} = n$ which is the number of observations in the dataset. Basically, now, you have a multinomial distribution with parameters $n$ (independent trials) and $p_{1}, p_{2}, \ldots , p_{K}$. Empirical probabilities are then computed as $y_{j} = \frac{n_{j}}{n}$. Therefore, loss for one observation is then computed as:

$$
L(\theta \mid x_{i}) = - \sum_{j=1}^{K} n_{j} \log(p_{j})
$$

if you now divide the right-hand sum by the number of observations, we will have:

$$
L(\theta \mid x_{i}) = -\frac{1}{n} log P(data \mid model) = - \frac{1}{n} \sum_{j=1}^{K} n_{j} \log(p_{j}) = - \sum_{j=1}^{K} y_{j} \log(p_{j}) 
$$

If we compute the cross-entropy over $n$ observations, we will have:

$$ L(\theta) = - \frac{1}{n} \sum_{i=1}^{n}  \sum_{j=1}^{K} \left[y_{ij} \log (p_{ij}) \right]$$

#### Why don’t we use KL-Divergence in machine learning models instead of the cross entropy?

The KL-Divergence between distributions requires us to know both the true distribution and distribution of our predictions thereof. Unfortunately, we never have the former: that’s why we build a predictive model using a Machine Learning algorithm.

#### Can KL divergence be used as a distance measure?

It may be tempting to think of KL Divergence as a distance metric, however we cannot use KL Divergence to measure the distance between two distributions. The reason for this is that KL Divergence is not symmetric, meaning that $D_{KL}(p\mid \mid q)$ may not be equal to $D_{KL}(q\mid \mid p)$.

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

#### What is gradient descent?

Gradient is a derivative of a function at a certain point. Basically, gives the slope of the line at that point. In order to find a gradient geometrically (just consider a 2 D graph and any continuous function), we draw a tangent at that point crossing x-axis and a perpendicular to the x-axis from that point. It will form a triangle and now calculating slope is easy. Also if this tangent is parallel to x-axis the gradient is 0 and if it is parallel to y-axis the gradient is infinity. If we have a function which is convex then at the bottom the gradient or derivative is 0. Similarly, if we have a concave function at the top gradient or derivative is 0. Why are we interested in 0? This is because it helps us find either the lowest(convex) or highest(concave) value of the function. Now our machine learning has a cost function and they can either be concave or convex. If it is convex, we look for minimum point, therefore, we use Gradient Descent and if it is concave, we look for maximum point and we use Gradient Ascent.

To find a local minimum of a function using gradient descent, one takes steps proportional to the negative of the gradient (or of the approximate gradient) of the function at the current point.

If instead one takes steps proportional to the positive of the gradient, one approaches a local maximum of that function; the procedure is then known as gradient ascent.

In other words:

* gradient descent aims at minimizing some objective function: $\theta_j \leftarrow \theta_j-\alpha \frac{\partial}{\partial \theta_{j}} J(\theta)$
* gradient ascent aims at maximizing some objective function: $\theta_j \leftarrow \theta_j+\alpha \frac{\partial}{\partial \theta_{j}} J(\theta)$.

Most of the Machine Learning algorithms use convex cost (loss) function. We want to minimize the loss. Hence we use Gradient Descent algorithm. In order to find the minimum value for a function, essentially, there are two things that you should know to reach the minima, i.e. which way to go and how big a step to take.

Gradient descent is a first-order iterative optimization algorithm used to minimize some function by iteratively moving in the direction of steepest descent as defined by the negative of the gradient because the gradient points in the direction of the greatest increase of the function, that is, the direction of steepest ascent. In machine learning, we use gradient descent to update the parameters of our model. Parameters refer to coefficients in Linear Regression and weights in neural networks.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/gradient_cost.gif?raw=true)

{% highlight python %}
while True:
  weights_grad = evaluate_gradient(loss_fun, data, weights)
  weights += - step_size * weights_grad # perform parameter update
{% endhighlight %}

This simple loop is at the core of all Neural Network libraries. There are other ways of performing the optimization (e.g. LBFGS), but Gradient Descent is currently by far the most common and established way of optimizing Neural Network loss functions.

When we have a cost function $J(\theta)$ with parameters $\theta$, Gradient Descent will be implemented for each $\theta_{j}$ as follows:

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/0_8yzvd7QZLn5T1XWg.jpg?raw=true)

For example, the following code example applies the gradient descent algorithm to find the minimum of the function $f(x)=x^{4}-3x^{3}+2$ with derivative $f'(x)=4x^{3}-9x^{2}$.

Solving for $4x^{3}-9x^{2}=0$ and evaluation of the second derivative at the solutions shows the function has a plateau point at 0 and a global minimum at $x={\tfrac {9}{4}}$.

{% highlight python %}
next_x = 6  # We start the search at x=6
learning_rate = 0.01  # Step size multiplier
precision = 0.00001  # Desired precision of result
max_iters = 10000  # Maximum number of iterations

# Derivative function
def df(x):
    return 4 * x**3 - 9 * x**2

for _i in range(max_iters):
    current_x = next_x
    next_x = current_x - learning_rate * df(current_x)

    step = next_x - current_x
    if abs(step) <= precision:
        break

print("Minimum at", next_x)

# The output for the above will be something like
# "Minimum at 2.2499646074278457"
{% endhighlight %}

#### Explain the following three variants of gradient descent: batch, stochastic and mini-batch?

**1**- **Batch Gradient Descent**: 
Batch gradient descent is a variation of the gradient descent algorithm that calculates the error for each example in the training dataset, but only updates the model after all training examples have been evaluated. One cycle through the entire training dataset is called a training epoch. Therefore, it is often said that batch gradient descent performs model updates at the end of each training epoch.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/1_PwSPYLEdrKegThcH41IBYw.png?raw=true)

If the number of training examples is large, then batch gradient descent is computationally very expensive. Commonly, batch gradient descent is implemented in such a way that it requires the entire training dataset in memory and available to the algorithm. Hence if the number of training examples is large, then batch gradient descent is not preferred. Instead, we prefer to use stochastic gradient descent or mini-batch gradient descent.

{% highlight python %}
for i in range(num_epochs):
    grad = compute_gradient(data, params)
    params = params - learning_rate * grad
{% endhighlight %}

**2**- **Mini Batch gradient descent**:
Mini-batch gradient descent seeks to find a balance between the robustness of stochastic gradient descent and the efficiency of batch gradient descent (best of the both worlds). Instead of going over all examples, Mini-batch Gradient Descent sums up over lower number of examples based on the batch size. Let's say that here, batch size equals $b$, therefore, randomly selected $b < m$ data points are processed per iteration (if the training set size is not divisible by batch size, the remaining will be its own batch) where $m$ is the total number of observations. 

So even if the number of training examples is large, it is processed in batches of $b$ training examples in one go. Thus, it works for larger training examples and that lesser number of iterations, comparing to stochastic gradient descent. It adds noise to the learning process that helps improving generalization error. Therefore, it wanders around the minimum region but never converges. Due to the noise, the learning steps have more oscillations and requires adding learning-decay to decrease the learning rate as we become closer to the minimum.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/1_5mHkZw3FpuR2hBNFlRxZ-A.png?raw=true)

{% highlight python %}
for i in range(num_epochs):
    np.random.shuffle(data)
    for batch in random_minibatches(data, batch_size=32):
        grad = compute_gradient(batch, params)
        params = params - learning_rate * grad
{% endhighlight %}

Note: with batch size $b = m$ (number of training examples), we get the Batch Gradient Descent.

Mini-batch requires the configuration of an additional "mini-batch size" hyperparameter for the learning algorithm. It is usually chosen as power of 2 such as 32, 64, 128, 256, 512, etc. The reason behind it is because some hardware such as GPUs achieve better run time with common batch sizes such as power of 2.

**3**- **Stochastic Gradient Descent**: 

Not all cost functions are convex (i.e., bowl shaped). There may be local minimas, plateaus, and other irregular terrain of the loss function that makes finding the global minimum difficult. Stochastic gradient descent can help us address this problem by processing 1 training example per iteration. Hence, the parameters are being updated even after one iteration in which only a single example has been processed. It is "stochastic" because it involves randomly shuffling the training dataset before each iteration that causes different orders of updates to the model parameters.

Hence this is quite faster than batch gradient descent. But again, when the number of training examples is large, even then it processes only one example which can be additional overhead for the system as the number of iterations will be quite large. Because it’s using only one example at a time, it adds even more noise to the learning process than mini-batch, however, that helps improving generalization error. However, this would increase the run time.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/1_tUqDJ5IYOhegTourdKqL0w.png?raw=true)

{% highlight python %}
for i in range(num_epochs):
  np.random.shuffle(data)
  for example in data:
      grad = compute_gradient(example, params)
      params = params - learning_rate * grad
{% endhighlight %}

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/1_PV-fcUsNlD9EgTIc61h-Ig.png?raw=true)

As the figure above shows, SGD direction is very noisy compared to mini-batch.

For example:

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/sgd_example_lr.png?raw=true)

#### What is the relationship between training loss and validation loss in terms of overfitting/underfitting?

Generally speaking though, training error will almost always underestimate your validation error. However it is possible for the validation error to be less than the training. You can think of it two ways:

1. Your training set had many 'hard' cases to learn
2. Your validation set had mostly 'easy' cases to predict

That is why it is important that you really evaluate your model training methodology. If you don't split your data for training properly your results will lead to confusing, if not simply incorrect, conclusions. One can think of model evaluation in four different categories:

1. **Underfitting** – Validation and training error high
2. **Overfitting** – Validation error is high, training error low
3. **Good fit** – Validation error low, slightly higher than the training error
4. **Unknown fit** - Validation error low, training error 'high'

It is said 'unknown' fit because the result is counter intuitive to how machine learning works. The essence of ML is to predict the unknown. If you are better at predicting the unknown than what you have 'learned', the data between training and validation must be different in some way. Either your training set had many 'hard' cases to learn, or your validation set had mostly 'easy' cases to predict. This could mean you either need to re-evaluate your data splitting method, adding more data, or possibly changing your performance metric (i.e., are you actually measuring the performance you want?).

#### What will happen if the learning rate is set too low or too high?

The size of these steps is called the learning rate which controls the rate at which we’re updating the weights of each parameter in the opposite direction of the gradient. The greater the learning rate, the larger the change in the weights at each training step. With a high learning rate we can cover more ground each step, but we risk overshooting the lowest point (minima of the loss function) and diverging since the slope of the hill is constantly changing. With a very low learning rate, we can confidently move in the direction of the negative gradient since we are recalculating it so frequently. A low learning rate is more precise, but calculating the gradient is time-consuming, so it will take us a very long time to get to the bottom. This is a parameter you may want to tune or adjust during training.

#### What is backpropagation?

The magic of deep neural networks lies in finding the set of weights for each layer that results in the most accurate predictions. The process of finding these weights is what we mean by _training_ the network.

During the training process, batches of data are passed through the network and the output is compared to the ground truth. The error in the prediction is then propagated backward through the network, adjusting each set of weights a small amount in the direction that improves the prediction most significantly. This process is appropriately called backpropagation. Gradually each unit becomes skilled at identifying a particular feature that ultimately helpt the network to make better predictions.

#### What isthe difference between stochastic gradient descent and backpropagation?

[Source](https://machinelearningmastery.com/difference-between-backpropagation-and-stochastic-gradient-descent/){:target="_blank"}

Stochastic Gradient Descent is an optimization algorithm that can be used to train neural network models.

The algorithm is referred to as “stochastic” because the gradients of the target function with respect to the input variables are noisy (e.g. a probabilistic approximation). This means that the evaluation of the gradient may have statistical noise that may obscure the true underlying gradient signal, caused because of the sparseness and noise in the training dataset.

The Stochastic Gradient Descent algorithm requires gradients to be calculated for each variable in the model so that new values for the variables can be calculated.

Stochastic gradient descent can be used to train (optimize) many different model types, like linear regression and logistic regression, although often more efficient optimization algorithms have been discovered and should probably be used instead.

Back-propagation is an automatic differentiation algorithm that can be used to calculate the gradients for the parameters in neural networks.

Together, the back-propagation algorithm and Stochastic Gradient Descent algorithm can be used to train a neural network. We might call this “Stochastic Gradient Descent with Back-propagation.”

It is common for practitioners to say they train their model using back-propagation. Technically, this is incorrect. Even as a short-hand, this would be incorrect. Back-propagation is not an optimization algorithm and cannot be used to train a model.

As Page 204 of the book "Deep Learning" tells: "The term back-propagation is often misunderstood as meaning the whole learning algorithm for multi-layer neural networks. Actually, back-propagation refers only to the method for computing the gradient, while another algorithm, such as stochastic gradient descent, is used to perform learning using this gradient."

It would be fair to say that a neural network is trained or learns using Stochastic Gradient Descent as a shorthand, as it is assumed that the back-propagation algorithm is used to calculate gradients as part of the optimization procedure.

That being said, a different algorithm can be used to optimize the parameter of a neural network, such as a genetic algorithm that does not require gradients. If the Stochastic Gradient Descent optimization algorithm is used, a different algorithm can be used to calculate the gradients for the loss function with respect to the model parameters, such as alternate algorithms that implement the chain rule.

#### What is gradient checking? Why it is important?

When implementing a neural network from scratch, Backpropagation is more prone to mistakes. Therefore, a method to debug this step could potentially save a lot of time and headaches when debugging a neural network. This is what we call gradient checking.

This method consists in approximating the gradient using a numerical approach. If this numerical gradient is close to analytical gradient (gradient from backpropagation), then Backpropagation is implemented correctly.

While checking the gradients, we do not use all examples in the training dataset and also, we do not run gradient checking in every iteration at the training because gradient check is slow. You can pick random number of examples from the training data to compute both numerical and analytical gradients. After we are sure that the implementation is bug-free/correct, we turn it off and use backprop for actual learning. Another note is that gradient checking does not work with dropout. One would usually run the gradient checking without dropout to make sure that backpropagation implementation is correct. 

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Screen%20Shot%202019-10-30%20at%2009.36.24.png?raw=true)

Derivatives tell us the slope or how steep the function is. For example, for the function $y = x^{2}$, the first-order derivative is $y^{\prime} = 2x$. 

Definition of derivative in calculus is:

$$
f'\left( x \right) = \lim_{\Delta x \to 0} \frac{ f \left( x + \Delta x \right) - f \left( x \right) }{\Delta x}
$$

where the numerator is to give change in $y$ ($y_{2} - y_{1}$) and the denominator gives change in x ($x_{2} - x_{1}$).

For this example $f(x) = x^{2}$, we can write:

$$
\begin{split}
f'\left( x \right) &= \lim_{\Delta x \to 0} \frac{(x +  \Delta x)^{2} - x^{2}}{\Delta x}\\
&= \lim_{\Delta x \to 0} \frac{x^{2} + 2 x \Delta x + {\Delta x}^{2} - x^{2}}{\Delta x}\\
&= \lim_{\Delta x \to 0} \frac{2 x \Delta x + {\Delta x}^{2}}{\Delta x}\\
& = \lim_{\Delta x \to 0}  \Delta x + 2x \\
& = 2x
\end{split}
$$

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Screen%20Shot%202019-10-30%20at%2009.44.03.png?raw=true)

We will call the distance we move in each direction $\varepsilon$. 

There are two formulations to find numerical gradients. One-sided difference formula is given by:

$$
f'\left( x \right) = \lim_{\varepsilon \to 0} \frac{f\left(x + \varepsilon\right) - f\left( x \right)}{\varepsilon}
$$

The two-sided difference is then:

$$
f'\left( x \right) = \lim_{\varepsilon \to 0}  \frac{f\left(x + \varepsilon\right) - f\left( x - \varepsilon \right)}{2 \varepsilon}
$$

For the example above, if we compute analytical and numerical gradient,

{% highlight python %}
def f(x):
    return x**2

epsilon = 1e-4
x=1.5

numericGradient_onesided = (f(x+epsilon) - f(x))/(epsilon)
numericGradient_twosided = (f(x+epsilon) - f(x-epsilon))/(2*epsilon)
numericGradient_onesided, numericGradient_twosided, 2*x
#(3.0001000000012823, 2.9999999999996696, 3.0)
{% endhighlight %}

We see that the difference between analytical derivative and two-sided numerical gradient is almost zero, however, the difference between analytical derivative and one-sided derivative is 0.0001. Therefore, we’ll use two-sided epsilon method to compute the numerical gradients.

Two-sided difference formula (also known as centered difference) requires you to evaluate loss function twice to check every single dimension of the gradient, so it is about 2 times as expensive, but the gradient approximation turns out to be much more precise. In order to see this, you can use Taylor expansion of $f(x + \varepsilon)$ and $f(x − \varepsilon)$ and verify that the one-sided formula has an error on order of $O(\varepsilon)$, while the two-sided formula only has error terms on order of $O(\varepsilon^{2})$ (i.e. it is a second order approximation).

$\varepsilon = 10e-7$ is a common value used for the difference between analytical gradient and numerical gradient. If the difference is less than $10e-7$ then the implementation of backpropagation is correct.

For the case of a neural network, we generally use vectorized implementation. We take the weight and bias matrices and reshape them into a big vector $\theta$. Similarly, all their respective derivatives will be placed on a vector $d_{\theta}$. Therefore, the approximate gradient can be expressed as

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Screen%20Shot%202019-10-30%20at%2010.04.47.png?raw=true)

and

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Screen%20Shot%202019-10-30%20at%2010.08.43.png?raw=true)

Then, we apply following formulat for gradient check:

$$
\frac{\left\lVert d_{\theta_{approx}} - d_{\theta} \right\rVert_{2}}{\left\lVert d_{\theta_{approx}} \right\rVert_{2} + \left\lVert d_{\theta} \right\rVert_{2}}
$$

{% highlight python %}
import numpy as np

# X = (hours sleeping, hours studying), y = Score on test
X = np.array(([3,5], [5,1], [10,2]), dtype=float)
y = np.array(([75], [82], [93]), dtype=float)

# Normalize
X = X/np.amax(X, axis=0)
y = y/100 #Max test score is 100

class Neural_Network(object):
    def __init__(self):        
        #Define Hyperparameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3
        
        #Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)
        
    def forward(self, X):
        #Propogate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3) 
        return yHat
        
    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))
    
    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)
        return J
        
    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)
        
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)  
        
        return dJdW1, dJdW2
    
    #Helper Functions for interacting with other classes:
    def getParams(self):
        #Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
    
    def setParams(self, params):
        #Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))
        
    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

def computeNumericalGradient(N, X, y):
        paramsInitial = N.getParams()
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        #As a value for epsilon, we usually opt for 1e-7.
        e = 1e-7 #epsilon

        for p in range(len(paramsInitial)):
            #Set perturbation vector
            perturb[p] = e
            N.setParams(paramsInitial + perturb)
            loss2 = N.costFunction(X, y)
            
            N.setParams(paramsInitial - perturb)
            loss1 = N.costFunction(X, y)

            #Compute Numerical Gradient
            numgrad[p] = (loss2 - loss1) / (2*e)

            #Return the value we changed to zero:
            perturb[p] = 0
            
        #Return Params to original value:
        N.setParams(paramsInitial)

        return numgrad 
    
NN = Neural_Network()

numgrad = computeNumericalGradient(NN, X,y)

grad = NN.computeGradients(X,y)

print(numgrad)
#[ 0.03645479 -0.03909739  0.04243039  0.0234212  -0.02695703  0.02910736 -0.19200016 -0.08135978 -0.09284831]
print(grad)
#[ 0.03645479 -0.03909739  0.04243039  0.0234212  -0.02695703  0.02910736 -0.19200016 -0.08135978 -0.09284831]

#We have to quantify how similar they are
numerator = np.linalg.norm(grad - numgrad)                                 
denominator = np.linalg.norm(grad) + np.linalg.norm(numgrad) #OR np.linalg.norm(grad+numgrad)         
difference = numerator / denominator
#If gradient check return a value less than 1e-7, then it means that backpropagation was implemented correctly. 
#Otherwise, there is potentially a mistake in your implementation. If the value exceeds 1e-3, then you are sure that the code is not correct.

difference
#1.2870177769557495e-09
{% endhighlight %}

#### What is Early Stopping?
It is a regularization technique that stops the training process as soon as the validation loss reaches a plateau or starts to increase.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/early-stopping-graphic.jpg?raw=true)

#### Why is Weight Initialization important in Neural Networks?

#### What are the Hyperparameteres? Name a few used in any Neural Network.

Hyperparameters are the variables which determine the network structure, e.g., number of hidden units and the variables which determine how the network is trained, e.g., learning rate. Hyperparameters are set before training.

**Network Hyperparameters**:

* Number of Hidden Layers
* Network Weight Initialization
* Activation Function

**Training Hyperparameters**:

* Learning Rate
* Momentum
* Number of Epochs
* Batch Size

#### What is model capacity?

It is the ability to approximate any given function. The higher model capacity is the larger amount of information that can be stored in the network.

#### What is softmax function? What is the difference between softmax function and sigmoid function? In which layer softmax action function will be used ?
 
Softmax function and Sigmoid function can be used in the different layers of neural networks.

The softmax function is simply a generalization of the logistic function (sigmoid function) that allows us to compute meaningful class-probabilities in multi-class settings (a.k.a. MaxEnt, multinomial logistic regression, softmax Regression, Maximum Entropy Classifier). Sigmoid function is used for binary classification.
 
 In a neural network, it is mostly used in the output layer in order to have probabilistic output of the network. When performing classification you often want not only to predict the class label, but also obtain a probability of the respective label. This probability gives you some kind of confidence on the prediction. .

In binary logistic regression, the predicted probabilities are as follows, using the sigmoid function:

$$
\begin{split}
P(y^{(i)}=1 \mid \mathbf{x}^{(i)}, \theta) &= \dfrac{1}{1+exp(-\theta^{T} \cdot \mathbf{x}^{(i)})}\\
P(y^{(i)}=0 \mid \mathbf{x}^{(i)}, \theta) &= 1 - P(y^{(i)}=1 \mid \mathbf{x}^{(i)}, \theta) = \dfrac{exp(-\theta^{T} \cdot \mathbf{x}^{(i)})}{1+exp(-\theta^{T} \cdot \mathbf{x}^{(i)})}
\end{split}
$$

In the multiclass logistic regression, with $K$ classes, the predicted probabilities are as follows, using the softmax function:

$$
P(Y_{i}=k) = \dfrac{exp(\theta_{k}^{T} \cdot \mathbf{x}^{(i)})}{\sum_{0 \leq c \leq K} exp(\theta_{c}^{T} \cdot \mathbf{x}^{(i)})}
$$

#### What’s the difference between a feed-forward and a backpropagation neural network?

A Feed-Forward Neural Network is a type of Neural Network architecture where the connections are "fed forward", i.e. do not form cycles (there is no  feedback connections like Recurrent Neural Network).  The term "Feed-Forward" is also used when information ﬂows through from input layer to output layer. It travels from input to hidden layer and from hidden layer to the output layer.

Backpropagation is a training algorithm consisting of 2 steps:

* Feed-Forward the values.
* Calculate the error and propagate it back to the earlier layers.

So to be precise, forward-propagation is part of the backpropagation algorithm but comes before backpropagating.

#### What is Dropout and Batch Normalization?

#### What is the relationship between the dropout rate and regularization?

Higher dropout rate says that more neurons are active. So there would be less regularization.

#### What is Variational dropout?

#### Where to Insert Batch Normalization and Dropout?

Batch normalization may be used on the inputs to the layer before or after the activation function in the previous layer. It may be more appropriate after the activation function if for s-shaped functions like the hyperbolic tangent and logistic function. It may be appropriate before the activation function for activations that may result in non-Gaussian distributions like the rectified linear activation function, the modern default for most network types, as the authors of the original paper puts: "The goal of Batch Normalization is to achieve a stable distribution of activation values throughout training, and in our experiments we apply it before the nonlinearity since that is where matching the first and second moments is more likely to result in a stable distribution".

Typically, dropout is placed on the fully connected layers, after the non-linear activation function, only because they are the one with the greater number of parameters and thus they're likely to excessively co-adapting themselves causing overfitting. However, since it's a stochastic regularization technique, you can really place it everywhere. Usually, it's placed on the layers with a great number of parameters.

You can remember the order using acronym BAD (Batch Normalization > Activation > Dropout). However, Batch Normalization eliminates the need for Dropout in some cases because Batch Normalization provides similar regularization benefits as Dropout intuitively.

#### Name a few deep learning frameworks

* TensorFlow
* Caffe
* The Microsoft Cognitive Toolkit/CNTK
* Torch/PyTorch
* MXNet
* Chainer
* Keras

#### Explain a Computational Graph.

Everything in a tensorflow is based on creating a computational graph. It has a network of nodes where each node performs an operation, Nodes represent mathematical operations and edges represent tensors. Since data flows in the form of a graph, it is also called a “DataFlow Graph.”

#### What is an image data and why do we divide by 255?

Input images are also composed of multiple sublayers: one per color channel. They are typically three: red, green and blue (RGB). Grayscale images have just one channel but some images might have much more - for example, satellite images that capture extra light frequencies (such as infrared).

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/color_channels.png?raw=true)

By default an image data consists of integers between 0 and 255 for each pixel channel. neural networks work best when each input is inside the range $-1$ to $1$, so we do divide by 255.

#### What is a CNN?

Convolutional networks are simply neural networks that use convolution in place of general matrix multiplication in at least one of their layers.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/putting_all_together.png?raw=true)

#### Why CNN?

Why not simply use a regular deep neural network with fully connected layers for image recognition tasks?

This works for small images (e.g., MNIST data), it breaks down for larger images because of huge number of parameters it requires. 

Imagine you have and image with size $1000 \times 1000 \times 3$. Input dimension is 3 million. If the first layer has just $1,000$ neurons, total of 3 billion connections and that's just the first layer!!!

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/cnn_works.png?raw=true)

#### Explain the different Layers of CNN.

There are 4 different layers in a convolutional neural network:
1. **Convolution Layer**: The Conv layer is the core building block of a Convolutional Network that does most of the computational heavy lifting. It is the first layer to extract features from an input image.
2. **Activation Layer**: After each convolutional layer, it is convention to apply a nonlinear layer (or activation layer) immediately afterward. The purpose of this layer is to introduce nonlinearity, without affecting the receptive fields of the conv layer, to a system that basically has just been computing linear operations during the convolutional layers (just element-wise multiplications and summations). This stage is also called detector stage.
3. **Pooling Layer**: Spatial Pooling (also called subsampling or downsampling, shrink) reduces the dimensionality of each feature map but retains the most important information.  Spatial Pooling can be of different types: Max, Average, Sum etc. It is common to periodically insert a Pooling layer in-between successive layers in a architecture. Pooling is applied separately on each feature maps. Pooling neuron has no weights. All it does is to aggregate inputs using an aggregation fixed function, such as the max and mean.
4. **Fully-connected Layer (Dense Layer)**: The CNN process begins with convolution and pooling, breaking down the image into features, and analyzing them independently. The result of this process feeds into a fully connected neural network structure that drives the final classification decision. This layer is mostly used with sigmoid function (for two classes) or softmax function (for multiple classes) in order to provide the final probabilities for each label.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/components_CNN.png?raw=true)

#### What is convolution?

The convolution is a specialized kind of linear operation. It is the sum of the element-wise multiplication, spanning over channels resulting in one number. It is performed by multiplying the filter or kernel (feature detectors - these terms are used interchangeably) pixelwise with the portion of the image (by sliding the filter over the input) and summing the result, to then produce a feature map. You can think of the feature detector as a window consisting of $F^{2}$ ($F \times F$) cells.

The output is more positive when the portion of the image closely matches the filter and more negative when the portion of the image is the inverse of the filter. 

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/2019-10-07%2019.01.14.jpg?raw=true)

If we move the filter across the entire image, from left to right, from top to bottom, recording the convolutional output (called _feature map_) as we go, we obtain a new array that picks out a particular feature of the input, depending on the values in the filter. 

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Screen%20Shot%202019-10-07%20at%2017.47.48.png?raw=true)

That is exactly what a convolutional layer is designed to do, but with multiple filters rather than just one. A Convolutional layer simultaneously applies multiple filters to the input, making it capable of detecting multiple features anywhere in its inputs. 

For example:

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/conv_layers.png?raw=true)

6?  so 6 filters spit out 6 different feature maps. Feature maps allow you to learn the explanatory factors!

Note that the depth of the filters in a layer is always the same as the number of channels in the preceding layer.

When the input has 3 channels, we will have 3 filters (one for each channel) instead of one channel. Then, we calculate the convolution of each filter with its corresponding input channel (First filter with first channel, second filter with second channel and so on). The stride of all channels are the same, so they output matrices with the same size. Now, we can sum up all matrices and output a single matrix which is the only channel at the output of the convolution layer.

![](https://raw.githubusercontent.com/mmuratarat/mmuratarat.github.io/master/_posts/images/rgb.gif)

Within one feature map, we apply the same filter which means all neurons share the same parameters, which reduces the number of parameters to be learned dramatically. Different feature maps may have different parameters. Although it depends on the problem domain, the significance the number of feature detectors intuitively is the number of features (like edges, lines, object parts etc...) that the network can potentially learn.

If you are working with color images, then each filter would have three channels rather than one (i.e. each having shape $3 \times 3 \times 3$) to match the three channels (red, green, blue) of the image.

#### What is the core idea of CNN?

The core idea about convolutional neural networks is that, contrary to fully-connected layers, instead to assigning different weights per each pixel of the image, you have some kernel that is smaller then the input picture and slides through it. What follows, we apply same set of weights to different parts of the picture (so called weight sharing). By this we hope to detect same patterns in different parts of the image.

#### What is stride?

* Stride means the step of the convolution operation; the number of steps that you move the filter over the input image.
* When the stride is 1, we move the filter one pixel at a time. When we set the stride to 2 or 3 (uncommon), we move the filter 2 or 3 pixels at a time depending on the stride. 
* The value of the stride also controls the size of the output volume generated by the convolutional layer. 
* Bigger the stride, smaller the output volume size. 
* For example if the input image is $7 \times 7$ and stride is $1$, the output volume will be $5 \times 5$. On the other hand if we increase the stride to be 2, the output volume reduces to $3\times 3$.
* Stride is normally set in a way so that the output volume is an integer and not a fraction.

**Stride 1**
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Stride1.png?raw=true)

**Stride 2**
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Stride2.png?raw=true)

#### What is Zero Padding?
 
* Zero padding refers to padding the input volume with zeros around the border.
* The zero padding also allows us to control the spatial size of the output volume
* If we do not add any zero padding and we use a stride of 1, the spatial size of the output volume is reduced.
* However, with the first few convolutional layers in the network we would want to preserve the spatial size and make sure that the input volume is equal to the output volume. This is where the zero padding is helpful. In the $7 \times 7$ input image example, if we use a stride of $1$ and a zero padding of $1$, then the output volume is also equal to $7 \times 7$.
* If we just keep applying the convolution, we might lose the data too fast. Padding is the trick we can use here to fix this problem

For example, if we apply a $5 \times 5$ filter on a $28 \times 28$ image, the output will have $24\times 24$ image. The spatial dimension is decreasing but in initial layers we want to preserve as much data as we can, as it contains low level features. In order to ensure or maintain the same dimension we can add two rows and columns at respective edges as shown. Depending on the dimension of filter, the number of the rows or columns added may change.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/padding.png?raw=true)

"VALID" only ever drops the right-most columns (or bottom-most rows).

"SAME" tries to pad evenly left and right, but if the amount of columns to be added is odd, it will add the extra column to the right (the same logic applies vertically: there may be an extra row of zeros at the bottom).

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/padding_style.png?raw=true)

When the input dimension was 5, if we use zero padding of 1 with stride $S=1$, the output dimension will be equal. If there is no zero-padding used, then the output volume will have spatial dimension of only 3, because that it is how many neurons would have “fit” across the original input. In general, setting zero padding to be $P=(F−1)/2$ when the stride is $S=1$ ensures that the input volume and output volume will have the same size spatially.

#### Why Padding?

Padding is used when you do not want to decrease the spatial resolution of the image when you use convolution. Besides avoiding shrinkage, with padding you benefit more from the information contained in pixels on the edges of the picture. Otherwise, pixels on the edges processed by fewer filters than the pixels on the inner side.

#### What is Pooling in CNN and how does it work? Why does pooling work?

After convolutional layers and activation functions, we introduce a pooling function to modify the output of the layer further by downsampling. Pooling extracts low level features from neighbourhood and helps to make the representation approximately invariant to small translations of the input. Invariance to translation means that if we translate the input by small amount, the values of the most of the pooled outputs do not change.

Suppose that you have draw a figure with a very thick brush. Then draw the same figure with a thin brush. These two figures has same information but one of them has a lot of unneccesary painting. Pooling simplifies the image with hard coding. Not with the AI. If you built a pooling layer with $2 \times 2$  filter, you squeeze the information of four pixels into one pixel. Whether it is max pooling or avarage pooling it is squeezing and reduction. Reduction of the feature map does not only reduces computational cost (as $2\times 2$ max pooling/average pooling reduces $75\%$ data) but also controls the overfitting by controling the number of features the CNN model is learning 

Pooling mainly helps in extracting sharp and smooth features. Max-pooling helps in extracting low-level extreme features like edges, points, etc. while average pooling goes for smooth features.

It also makes the network robust and invarient to the small changes and disturbances. When you use average pooling on $2 \times 2$ pooling layer, small changes on four nodes makes a tiny effect on average of that four nodes. If you use max pooling instead of average, small changes on that small valued four nodes won’t affect the output. Because we get the biggest value of that four nodes.

#### What is the benefit of using average pooling rather than max pooling?

While the max and average pooling are rather similar, the use of average pooling encourages the network to identify the complete extent of the object. The basic intuition behind this is that the loss for average pooling benefits when the network identifies all discriminative regions of an object as compared to max pooling. In simpler words, max pooling rejects a big chunk of data and retains at max 1/4th. Average pooling on the other hand, do not reject all of it and retains more information, in comparison to max pooling. 

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/main-qimg-c9d498b278f6db5040d861eac821e7d9.png?raw=true)

#### What are the weights in a CNN?

The values stored in the filters are the weights that are learned by the neural network through training. Initially these are random but gradually, the filters adapt their weights to start picking out interesting features such as edges or particular color combinations.

#### How to compute the spatial size of output image after a Convolutional layer?

We don’t have to manually calculate the dimension (the spatial size) of the output, but it’s a good idea to do so to keep a mental account of how our inputs are being transformed at each step. We can compute the spatial size on each dimension (width/height/depth).

* The input volume size ($W_{1}$ and $H_{1}$, generally they are equal and $D_{1}$)
* Number of filters ($K)$
* the receptive field size of filter ($F$)
* the stride with which they are applied ($S$)
* the amount of zero padding used ($P$) on the border. 

produces a volume of size $W_{2} \times H_{2} \times D_{2}$ where:

$W_{2} = \dfrac{W_{1} - F + 2P}{S} + 1$

$H_{2} = \dfrac{H_{1} - F + 2P}{S} + 1$ (i.e., width and height are computed equally by symmetry)

$D_{2}= K$ 

* 2P comes from the fact that there should be a padding on each side.

#### What is the number of parameters in one CNN layer?

$F$ is the receptive field size of filter (kernel) and $K$ is the number of filters. $D_{1}$ is the depth (the number of channels) of the image.

In a Conv Layer, the depth of every kernel (filter) is always equal to the number of channels in the input image. So every kernel has $F^{2} \times D_{1}$ parameters, and there are $K$ such kernels.

Parameter sharing scheme is used in Convolutional Layers to control the number of parameters.

With parameter sharing, which means no matter the size of your input image, the number of parameters will remain fixed. $F \cdot F \cdot D_{1}$ weights per feature map are introduced and for a total of $(F \cdot F \cdot D_{1}) \cdot K$ weights and $K$ biases. Number of parameters of the Conv Layer is $(F \cdot F \cdot D_{1}) \cdot K + K$

#### How can I decide the kernel size?

Unfortunately there is absolutely no general answer to this question. No prinicipal method to determine these hyperparameters is known.

A conventional approach is to look for similar problems and  some of the most popular architectures which have already been shown to work. Than a suitable architecture can be developed by experimentation.

However, common kernel sizes are $3 \times 3$, $5 \times 5$ and $7 \times 7$. A well known architecture for classification is to use convolution, pooling and some fully connected layers on top. Just start of with a modest number of layers and increase the number while measuring you performance on the test set.

#### How convolution works in one dimension?

Convolution is a good way to identify patterns in data that is directly tied to space or time. Adjacent pixels in an image are adjacent for a reason. In the physical world, they collect light from neighboring locations. Time series data has a similar structure. Neighboring data points were produced close together in time and are much more likely to be related then points far apart. This inherent structure is what convolution exploits. It finds local patterns that reoccur in the data. It would not be effective, for instance, if we first scrambled the pixels in an image. That would hide the patterns, the spatial relationships, that convolution tries to learn.

In CNNs, there is nothing special about number of dimensions for convolution. They work the same way whether they have 1, 2, or 3 dimensions. The difference is the structure of the input data and how the filter, also known as a convolution kernel or feature detector, moves across the data. One way to think of this operation is that we are sliding the kernel over the input data. For each position of the kernel, we multiply the overlapping values of the kernel and data together, and add up the results. The number of dimensions is just a property of the problem being solved. For example, 1D for audio signals, 2D for images, 3D for movies.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/1D-convolutional-example.png?raw=true)

In this natural language processing (NLP) example, a sentence is made up of 9 words. Each word is a vector that represents a word. The filter covers at least one word; a height parameter specifies how many words the filter should consider at once. In this example the height is 2, meaning the filter moves 8 times to fully scan the data.

Let’s call our input vector $f$ and our kernel $g$, and say that $f$ has length $n$, and $g$ has length $m$. The convolution $f * g$ of $f$ and $g$ is defined as:

$$
(f * g)(i) = \sum\limits_{j=1}^{m} g(j) f(i - j +m /2)
$$

Let’s look at a simple example. Suppose our input is 1D data:

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/1D_convolution_example1.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/1D_convolution_example2.png?raw=true)

#### What is Depthwise Separable Convolutions?

The purpose of doing convolution is to extract useful features from the input. In image processing, there is a wide range of different filters one could choose for convolution. Each type of filters helps to extract different aspects or features from the input image, e.g. horizontal / vertical / diagonal edges. Similarly, in Convolutional Neural Network, different features are extracted through convolution using filters whose weights are automatically learned during training. All these extracted features then are ‘combined’ to make decisions.

There are a few advantages of doing convolution, such as weights sharing and translation invariant. Convolution also takes spatial relationship of pixels into considerations. These could be very helpful especially in many computer vision tasks, since those tasks often involve identifying objects where certain components have certain spatially relationship with other components (e.g. a dog’s body usually links to a head, four legs, and a tail).

In the regular 2D convolution performed over multiple input channels, the filter is as deep as the input and lets us freely mix channels to generate each element in the output. Depthwise convolutions don't do that - each channel is kept separate - hence the name depthwise. Here's a diagram to help explain how that works:

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/deptwise_conv.png?raw=true)

There are three conceptual stages here:

* Split the input into channels, and split the filter into channels (the number of channels between input and filter must match).
* For each of the channels, convolve the input with the corresponding filter, producing an output tensor (2D).
* Stack the output tensors back together.

In TensorFlow, the corresponding op is `tf.nn.depthwise_conv2d`; this operation has the notion of channel multiplier which lets us compute multiple outputs for each input channel (somewhat like the number of output channels argument `out_channels` in conv2d, which decides the number of feature maps after convolution).

For example, let's say you have a coloured image with length $100$, width $100$. So the dimensions are $100 \times 100 \times 3$. Let's use the same filter of width and height $5$. Lets say we want the next layer to have a depth of $8$.

In `tf.nn.conv2d` you define the kernel shape as `[width, height, in_channels, out_channels]`. In our case this means the kernel has shape `[5,5,3,out_channels]`. The weight-kernel that is strided over the image has a shape of $5 \times 5 \times 3$, and it is strided over the whole image 8 times to produce 8 different feature maps.

In `tf.nn.depthwise_conv2d` you define the kernel shape as `[width, height, in_channels, channel_multiplier]`. Now the output is produced differently. Separate filters of $5 \times 5 \times 1$ are strided over each dimension of the input image, one filter per dimension, each producing one feature map per dimension. So here, a kernel size `[5,5,3,1]` would produce an output with depth 3. The `channel_multiplier` argument tells you how many different filters you want to apply per dimension. So the original desired output of depth 8 is not possible with 3 input dimensions. Only multiples of 3 are possible.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/conv_and_depthconv.png?raw=true)

Output of Depthwise Convolution is fed to Pointwise Convolution, to get single pixel output similar to Normal Convolution, i.e. a $1 \times 1$ convolution, projecting the channels output by the depthwise convolution onto a new channel space.

In depthwise operation, convolution is applied to a single channel at a time unlike standard CNN’s in which it is done for all the $M$ channels. So here the filters/kernels will be of size $D_{k} \times D_{k} \times 1$. Given there are $M$ channels in the input data, then $M$ such filters are required. Output will be of size $D_{p} \times D_{p} \times M$.

In point-wise operation, a $1 \times 1$ convolution operation is applied on the $M$ channels together. So the filter size for this operation will be $1 \times 1 \times M$. Say we use $N$ such filters, the output size becomes $D_{p} \times D_{p} \times N$.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/2-229.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/3-167.png?raw=true)

After deptwise convolution, we move on pointwise convolution and this process is called depthwise seperable convolution. 

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/conv2d-depthwise-separable.png?raw=true)

Depthwise separable convolutions have become popular in DNN models recently, for two reasons:

* They have fewer parameters than "regular" convolutional layers, and thus are less prone to overfitting.
* With fewer parameters, they also require less operations to compute, and thus are cheaper and faster

In standard convolution 2D, when we have $N$ different $D_{k} \times D_{k} \times M$ filters where $M$ is the depth (it is 3 for the first layer of CNN), the number of parameters is $D_{k}^{2} \times M \times N$.

In Depthwise Convolution, since we are dealing with one channel at a time, we have $M$ different $D_{k} \times D_{k} \times 1$ filter, the number of parameters is $M \times D_{k}^{2}$.

Since output of depthwise convolution is fed to pointwise convolution, let's say that input to Pointwise Convolution has shape of $D_{p} \times D_{p} \times M$. So, we have N different $1 \times 1 \times M$ filters, the number of parameters is $N \times M$.

So, depthwise separable convolution has $M \times (N + D_{k}^{2})$
Ratio of number of parameters of standard convolution and depthwise separable convolution is then:

$$
\frac{\text{The number of parameters depthwise separable convolution}}{\text{The number of parameters in standard convolution}} = \frac{M \times (N + D_{k}^{2})}{D_{k}^{2} \times M \times N} = \frac{1}{N} + \frac{1}{D_{k}^{2}}
$$

#### What is an RNN?

#### What is the number of parameters in an RNN?

#### What are some issues faced while training an RNN?

#### What is Vanishing Gradient Problem?

#### What is Exploding Gradient Problem?

Exploding gradients can be dealt with by gradient clipping (truncating the gradient if it exceeds some magnitude)

ReLU in conjunction with batch normalization (or ELU or SELU) has effectively obviated both vanishing/ exploding gradients and the internal covariate shift problem.

The problem still remains for recurrent nets though (to some extent at least).

#### What are the different types of RNN?

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/0_1PKOwfxLIg_64TAO.jpeg?raw=true)

Each rectangle is a vector and arrows represent functions (e.g. matrix multiply). Input vectors are in red, output vectors are in blue and green vectors hold the RNN's state (more on this soon). From left to right: (1) Vanilla mode of processing without RNN, from fixed-sized input to fixed-sized output (e.g. image classification). (2) Sequence output (e.g. image captioning takes an image and outputs a sentence of words). (3) Sequence input (e.g. sentiment analysis where a given sentence is classified as expressing positive or negative sentiment). (4) Sequence input and sequence output (e.g. Machine Translation: an RNN reads a sentence in English and then outputs a sentence in French). (5) Synced sequence input and output (e.g. video classification where we wish to label each frame of the video). Notice that in every case are no pre-specified constraints on the lengths sequences because the recurrent transformation (green) is fixed and can be applied as many times as we like.

#### What is an LSTM cell? How does an LSTM network work? Explain the gates.

Theoretically recurrent neural network can work. But in practice, it suffers from two problems: vanishing gradient and exploding gradient, which make it unusable. 

Recurrent Neural Networks suffer from short-term memory. If a sequence is long enough, they will have a hard time carrying information from earlier time steps to later ones due to the vanishing gradient. So in recurrent neural networks, layers that get a small gradient update stops learning. Those are usually the earlier layers. Think about a recurrent neural network unrolled through time. So because these layers do not learn, RNNs can forget what it seen in longer sequences, thus having a short-term memory.

Then later, LSTM (long short term memory) was invented as the solution to short-term memory. In order to solve this issue, a memory unit, called the cell has been explicitly introduced into the network. They have internal mechanisms called gates that can regulate the flow of information.

Note that LSTM does not always protect you from exploding gradients! Therefore, successful LSTM applications typically use gradient clipping.

LSTMs are recurrent network where you replace each neuron by a memory unit. This unit contains an actual neuron with a recurrent self-connection. The activations of those neurons within memory units are the state of the LSTM network. This is the diagram of a LSTM building block

![](https://raw.githubusercontent.com/mmuratarat/mmuratarat.github.io/master/_posts/images/lstm.png)

The network takes three inputs. $X_t$ is the input of the current time step. $h_{t-1}$ is the output from the previous LSTM unit and $C_{t-1}$ is the "memory" of the previous unit. As for outputs, $h_{t}$ is the output of the current network. $C_{t}$ is the memory of the current unit.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/1_S0Y1A3KXYO7_eSug_KsK-Q.png?raw=true)
 
Equations below summarizes how to compute the cell’s long-term state, its short-term state, and its output at each time step for a single instance (the equations for a whole mini-batch are very similar).

1. Input gate:
$$ i_{t} = \sigma (W_{xi}^{T} \cdot X_{t} +  W_{hi}^{T} \cdot h_{t-1}  + b_{i})$$

2. Forget gate:
$$ f_{t} = \sigma (W_{xf}^{T} \cdot X_{t} + W_{hf}^{T} \cdot h_{t-1} + b_{f})$$

3. New Candidate:
$$ \widetilde{C}_{t} = tanh (W_{xc} \cdot X_{t} + W_{hc} \cdot h_{t-1} + b_{c})$$

4. Cell State:
$$ C_{t} = f_{t}\circ C_{t-1} + i_{t}  \circ \widetilde{C}_{t}$$

5. Output gate:
$$ o_{t} = \sigma (W_{xo} \cdot X_{t} + W_{ho} \cdot h_{t-1} + b_{o})$$

6. Hidden State:
$$ h_{t} = o_{t}\circ tanh(C_{t})$$

*  $W_{xi}$, $W_{xf}$, $W_{xc}$, $W_{xo}$ are the weight matrices of each of the three gates and block input for their connection to the input vector $X_{t}$.
*  $W_{hi}$, $W_{hf}$, $W_{hc}$, $W_{ho}$ are the weight matrices of each of the three gates and block input  for their connection to the previous short-term state $h_{t-1}$.
*  $b_{i}$, $b_{f}$, $b_{c}$ and $b_{o}$ are the bias terms for each of the three gates and block input . 
*  $\sigma$ is an element-wise sigmoid activation function of the neurons, and $tanh$ is an element-wise hyperbolic tangent activation function of the neurons
*  $\circ$ represents the Hadamard product (elementwise product).

**NOTE**: Sometimes, $h_t$ is called as the outgoing state and $c_t$ is called as the internal cell state.

Just like for feedforward neural networks, we can compute all these in one shot for a whole mini-batch by placing all the inputs at time step $t$ in an input matrix $X_{t}$. If we write down the equations for **all instances in a mini-batch**, we will have:

1. Input gate:
$$ i_{t} = \sigma (X_{t}\cdot W_{xi} + h_{t-1} \cdot W_{hi} + b_{i})$$

2. Forget gate:
$$ f_{t} = \sigma (X_{t} \cdot W_{xf} + h_{t-1} \cdot W_{hf} + b_{f})$$

3. New Candidate:
$$ \widetilde{C}_{t} = tanh (X_{t} \cdot W_{xc} + h_{t-1} \cdot W_{hc} + b_{c})$$

4. Cell State:
$$ C_{t} = f_{t}\circ C_{t-1} + i_{t}  \circ \widetilde{C}_{t}$$

5. Output gate:
$$ o_{t} = \sigma (X_{t} \cdot W_{xo} + h_{t-1} \cdot W_{ho} + b_{o})$$

6. Hidden State:
$$ h_{t} = o_{t}\circ tanh(C_{t})$$

We can concatenate the weight matrices for $X_{t}$ and $h_{t-1}$ horizontally, we can rewrite the equations above as the following:

1. Input gate:
$$ i_{t} = \sigma ( [X_{t} h_{t-1}] \cdot W_{i}  + b_{i})$$

2. Forget gate:
$$ f_{t} = \sigma ([X_{t} h_{t-1}] \cdot W_{f} + b_{f})$$

3. New Candidate:
$$ \widetilde{C}_{t} = tanh ( [X_{t} h_{t-1}] \cdot W_{c} + b_{c})$$

4. Cell State:
$$ C_{t} = f_{t}\circ C_{t-1} + i_{t}  \circ \widetilde{C}_{t}$$

5. Output gate:
$$ o_{t} = \sigma ([X_{t} h_{t-1}] \cdot W_{o}+ b_{o})$$

6. Hidden State:
$$ h_{t} = o_{t}\circ tanh(C_{t})$$

Let's denote $B$ as batch size, $F$ as number of features and $U$ as number of units in an LSTM cell, therefore, the dimensions will be computed as follows:

$X_{t} \in \mathbb{R}^{B \times F}$

$h_{t-1} \in \mathbb{R}^{B \times U}$

$h_{t} \in \mathbb{R}^{B \times U}$

$C_{t-1} \in \mathbb{R}^{B \times U}$

$W_{xi} \in \mathbb{R}^{F \times U}$

$W_{xf} \in \mathbb{R}^{F \times U}$

$W_{xc} \in \mathbb{R}^{F \times U}$

$W_{xo} \in \mathbb{R}^{F \times U}$

$W_{hi} \in \mathbb{R}^{U \times U}$

$W_{hf} \in \mathbb{R}^{U \times U}$

$W_{hc} \in \mathbb{R}^{U \times U}$

$W_{ho} \in \mathbb{R}^{U \times U}$

$W_{i} \in \mathbb{R}^{F+U \times U}$

$W_{c} \in \mathbb{R}^{F+U \times U}$

$W_{f} \in \mathbb{R}^{F+U \times U}$ 

$W_{o} \in \mathbb{R}^{F+U \times U}$ 

$b_{i} \in \mathbb{R}^{U}$

$b_{c} \in \mathbb{R}^{U}$

$b_{f} \in \mathbb{R}^{U}$

$b_{o} \in \mathbb{R}^{U}$

$i_{t} \in \mathbb{R}^{B \times U}$

$f_{t} \in \mathbb{R}^{B \times U}$

$C_{t} \in \mathbb{R}^{B \times U}$

$h_{t} \in \mathbb{R}^{B \times U}$

$o_{t} \in \mathbb{R}^{B \times U}$

**NOTE**: Batch size can be $1$. In that case, $B=1$.

1. **New temporary memory**: Use $X_{t}$ and $h_{t-1}$ to generate new memory that includes aspects of $X_{t}$.
2. **Input gate**: Use $X_{t}$ and $h_{t-1}$ to determine whether the temporary memory $\widetilde{C_{t}}$ is worth preserving.
3. **Forget gate**: Assess whether the past memory cell $C_{t-1}$ should be included in $C_{t}$.
4. **Update memory state**: Use forget and input gates to combine new temporary memory and the current memory cell state to get $C_{t}$.
5. **Output gate**: Decides which part of $C_{t}$ should be exposed to $h_{t}$. 

#### Why the 3 gates have sigmoid function as activation function?

Gates contains sigmoid activations. A sigmoid activation is similar to the tanh activation. Instead of squishing values between $-1$ and $1$, it squishes values between $0$ and $1$. That is helpful to update or forget data because any number getting multiplied by $0$ is $0$, causing values to disappears or be "forgotten". Any number multiplied by $1$ is the same value therefore that value stay’s the same or is "kept". The network can learn which data is not important therefore can be forgotten or which data is important to keep.

#### What is the number of parameters in an LSTM cell?

The LSTM has a set of 2 matrices: $W_{h}$ and $W_{x}$ for each of the (3) gates (forget gate/input gate/output gate). Each $W_{h}$ has $U \times U$ elements and each $W_{x}$ has $F \times U$ elements. There is another set of these matrices for updating the cell state (new candidate). Similarly, $W_{xc}$ has $F \times U$ and $W_{hc}$ has $U \times U$ elements. On top of the mentioned matrices, you need to count the biases. Each bias for 3 gates and new candidate has $U$ elements. Hence total number parameters is $4(UF +  U^{2} + U)$.

#### Why stacking LSTM layers?

The main reason for stacking LSTM cells is to allow for greater model complexity. The addition of more layers increases the capacity of the network, making it capable of learning a large training dataset and efficiently representing more complex mapping functions from inputs to outputs. In case of a simple feed forward network, we stack layers to create a hierarchial feature representation of the input data to then use for some machine learning task. The same applies for stacked LSTMs. 

#### Why do we stack a dense layer after an LSTM?

The output of a LSTM is not a softmax. Many frameworks just give you the internal state $h$ as output, so the dimensionality of this output is equals to the number of unit, which is propably not the dimensionality of your desired target. Output dimension of a dense layer would be the number of labels you want result.

#### What is an autoencoder?

An autoencoder neural network are not a true unsupervised learning technique (which would imply a different learning process altogether), they are a self-supervised technique, a specific instance of supervised learning where the targets are generated from the input data (we're discarding the labels).

They work by compressing the input into a latent-space representation which this process is to reduce the size of the inputs into a smaller representation, and then reconstructing the output from this representation. They apply backpropagation, setting the target values to be equal to the inputs. This network can be trained by minimizing the reconstruction error, $L(x, \hat{x})$, which measures the differences between our original input and the consequent reconstruction.

If anyone needs the original data, they can reconstruct it from the compressed data.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/1_8ixTe1VHLsmKB3AquWdxpQ-768x257.png?raw=true)

Autoencoders are generally used for

1. Feature Extraction
2. Dimensionality Reduction
3. Unsupervised pretraining of DNN
4. Generative models
5. Anomaly Detection (Autoencoders are generally bad at reconstructing outliers).

An Autoencoder consist of three layers:

* Encoder
* Code
* Decoder

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/1_44eDEuZBEsmG_TCAKRI3Kw@2x-1.png?raw=true)

Encoder: This part of the network compresses the input into a latent space representation. The encoder layer encodes the input image as a compressed representation in a reduced dimension. The compressed image is the distorted version of the original image.

Code: This part of the network represents the compressed input which is fed to the decoder.

Decoder: This layer decodes the encoded image back to the original dimension. The decoded image is a lossy reconstruction of the original image and it is reconstructed from the latent space representation.

The layer between the encoder and decoder, ie. the code, is also known as Bottleneck. This is a well-designed approach to decide which aspects of observed data are relevant information and what aspects can be discarded.

#### What is the difference between undercomplete and overcomplete Autoencoders?

One way to obtain useful features from the autoencoder is to constrain laten space $h$ to have smaller dimensions than $X$, in this case the autoencoder is called undercomplete. By training an undercomplete representation, we force the autoencoder to learn the most salient features of the training data by limiting the amount of information that can flow through the network. If the autoencoder is given too much capacity, it can learn to perform the copying task (memorizing the data) without extracting any useful information about the distribution of the data. This can also occur if the dimension of the latent representation is the same as the input, and in the overcomplete case, where the dimension of the latent representation is greater than the input. In these cases, even a linear encoder and linear decoder can learn to copy the input to the output without learning anything useful about the data distribution. Ideally, one could train any architecture of autoencoder successfully, choosing the code dimension and the capacity of the encoder and decoder based on the complexity of distribution to be modeled.

#### What are the practical applications of Autoencoders?

They are rarely used in practical applications. In 2012 they briefly found an application in greedy layer-wise pretraining for deep convolutional neural networks, but this quickly fell out of fashion as we started realizing that better random weight initialization schemes were sufficient for training deep networks from scratch. In 2014, batch normalization started allowing for even deeper networks, and from late 2015 we could train arbitrarily deep networks from scratch using residual learning

Today two interesting practical applications of autoencoders are data denoising, and dimensionality reduction for data visualization. With appropriate dimensionality and sparsity constraints, autoencoders can learn data projections that are more interesting than PCA or other basic techniques.

For 2D visualization specifically, t-SNE (pronounced "tee-snee") is probably the best algorithm around, but it typically requires relatively low-dimensional data. So a good strategy for visualizing similarity relationships in high-dimensional data is to start by using an autoencoder to compress your data into a low-dimensional space (e.g. 32 dimensional), then use t-SNE for mapping the compressed data to a 2D plane. 

#### What is denoising?

In this type of task, the machine learning algorithm is given a *corrupted example* $\widetilde{\boldsymbol{x}} \in R^{n}$ obtained by unknown corruption process from a *clean example* $\boldsymbol{x} \in R^{n}$. The learner must predict the clean example $\boldsymbol{x}$ from its corrupted version $\widetilde{\boldsymbol{x}}$, or more generally predict the conditional probability distribution $p(\boldsymbol{x} \mid \widetilde{\boldsymbol{x}})$.

#### Why to denoise autoencoder procedure?

The denoising autoencoder procedure was invented to help:

* The hidden layers of the autoencoder learn more robust filters
* Reduce the risk of overfitting in the autoencoder
* Prevent the autoencoder from learning a simple identify function

#### What are some limitations of deep learning?

1. Deep learning usually requires large amounts of training data and a computational power.
2. Deep neural networks are easily fooled. For example, researchers have now created an adversarial patch that makes an object invisible to YOLO v2.
3. Successes of deep learning are purely empirical, deep learning algorithms have been criticized as uninterpretable “black-boxes”.
4. Deep learning thus far has not been well integrated with prior knowledge.

#### What is transfer learning ?

A deep learning model trained on a specific task (a pre-trained model) can be reused for different problem in the same domain even if the amount of data is not that huge.

#### How to choose embedding size in an Embedding layer?

The rule of thumb for determining the embedding size is the cardinality size divided by 2, but no bigger than 50. Let's say that you have one categorical variable with $k$ categories (levels). Then embedding size is decided based on `min(50, (k+1)//2)`.

#### What is the difference between the version of Keras that’s built-in to TensorFlow, and the version I can find at keras.io?

Keras is an API standard for defining and training machine learning models. A reference implementation of Keras is maintained as an independent open source project, which you can find at www.keras.io. This project is independent of TensorFlow, and has an active community of contributors and users. TensorFlow includes a full implementation of the Keras API (in the tf.keras module) with TensorFlow-specific enhancements. TensorFlow includes an implementation of the Keras API (in the tf.keras module) with TensorFlow-specific enhancements. These include support for eager execution for intuitive debugging and fast iteration, support for the TensorFlow SavedModel model exchange format, and integrated support for distributed training, including training on TPUs. More can be found on [here](https://medium.com/tensorflow/standardizing-on-keras-guidance-on-high-level-apis-in-tensorflow-2-0-bad2b04c819a){:target="_blank"}

#### Why is the validation loss lower than the training loss?

At the most basic level, a loss function quantifies how “good” or “bad” a given predictor is at classifying the input data points in a dataset. 

$$
Loss = \frac{1}{n}\sum_{i=1}^{n} \left[y_{i} \neq f(x_{i}) \right]
$$

We, therefore, seek to drive our loss down therefore improving the model accuracy. We want to do so as fast as possible and with as little hyperparameter updates/experiments and all without overfitting our network and modeling the training data too closely.  However, choosing the values for the parameters that minimize the loss function on the training data is not necessarily the best policy. We want the learning machine to model true regularities in the data and ignore the noise in the data and to do well on test data that is not known during the learning. 

The training loss is calculated over the entire training dataset. Likewise, the validation loss is calculated over the entire validation dataset. The training set is typically at least 4 times as large as the validation (80-20). Given that the error is calculated over all samples, you could expect up to approximately 4X the loss measure of the validation set. You will notice, however, that the training loss and validation loss are approaching one another as training continues. This is intentional as if your training error begins to get lower than your validation error you would be beginning to overfit your model!

Though, we have 4 different cases while training a model: (1) _Underfitting_ - Validation and training error high, (2) _Overfitting_ - Validation error is high, training error low, (3) _Good fit_ - Validation error low, slightly higher than the training error and (4) _Unknown fit_ - Validation error low, training error 'high'. 

**Reason 1**: Regularization mechanisms, such as Dropout and $L_{1}$/$L_{2}$ weight regularization, are turned off at validation/testing time. When training a deep neural network we often apply regularization to help our model in order to obtain higher validation/testing accuracy and ideally, to generalize better to the data outside the validation and testing sets. Regularization methods often sacrifice training accuracy to improve validation/testing accuracy. 

Batch norm can also be considered of the regularization methods. During training batch-norm uses mean and variance of the given input batch, which might be different from batch to batch. But during evaluation batch-norm uses running mean and variance, both of which reflect properties of the whole training set much better than mean and variance of a single batch during training. Therefore, if it is turned off, training loss and validation loss can get closer.

**Reason 2**: The training loss is the average of the losses over each batch of training data. Because your model is changing over time, the loss over the first batches of an epoch is generally higher than over the last batches.
 
**Reason 3**: Training loss is measured during each epoch while validation loss is measured after each epoch. The training loss is continually reported over the course of an entire epoch; however, validation metrics are computed over the validation set only once the current training epoch is completed. This implies, that on average, training losses are measured half an epoch earlier. If you shift the training losses half an epoch to the left you’ll see that the gaps between the training and losses values are much smaller.
 
**Reason 4**: The validation set may be easier than the training set. This can happen by chance that if the validation set is smaller, but very much like it with less noise, or it was not properly sampled (e.g., too many easy classes) or there may be data leakage, i.e., training samples getting accidentally mixed in with validation/testing samples.

**Reason 5**: Data augmentation mechanism. Data augmentation is usually done only on training set and not on validation set (as for the dropout regularization), and this may lead to a validation set containing "easier" cases to predict than those in the training set.

**Reason 6**: There is also the possibility that there is a bug in the code which makes it possible that training has not converged to the optimal soluion on the training set. 

#### What are some tips and tricks to train a deep neural network?

*  Use standard architectures and transfer learning if your problem allows you to, such as ResNet-50 pre-trained on ImageNet.

* The learning rate might be the most important hyperparameter. There is no fixed learning rate for a neural network. It depends on the kind of problem you are working on, the dataset you are feeding to your network, and most importantly the structure of the network which varies from problem to problem because topology of loss landscape changes. You can use cyclical learning rates which allows learning rate to cyclically oscillate between the two bounds. Training with cyclical learning rates instead of fixed values achieves improved classification accuracy without a need to tune and often in fewer iterations (https://arxiv.org/pdf/1506.01186.pdf).
  
  ```
  local cycle = math.floor(1 + epochCounter / ( 2 ∗ stepsize))
  local x = math.abs(epochCounter / stepsize − 2∗ cycle + 1 )
  local lr = opt.LR + (maxLR − opt.LR) ∗ math.max(0 , (1−x ))
  ```
  
  where
  
  * `opt.LR` is the specified lower (i.e., base) learning rate.
  * `epochCounter` is the number of epochs of training.
  * `lr` is the computed learning rate.
  * `stepsize` is half the period or cycle length.
  * `max_lr` is the maximum learning rate boundary (upper bound).
 
  ![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/cyclical_learning_rate.png?raw=true)
  
  You can also use learning rate finder and Learning Rate Scheduler from this blogpost https://www.jeremyjordan.me/nn-learning-rate/
 
* Instead of using a fixed number of epochs, stop the training when validation accuracy stops improving (early stopping).

* Use the ReLU activation function with Kaiming (He) initialization.

* Use Adam as your default optimizer. 

* Do not use minibatches of size greater than 32.

* Try Dropout and Batch Normalization to regularize the model.

* Use data augmentation to increase the diversity of the data, which may act as a regularizer to avoid overfitting.

* Get more data if you can, which can lead to a performance improvement.

#### What is an activation layer in Keras?

As stated in the [docs](https://keras.io/activations/), the activation layer in Keras is equivalent to a dense layer with the same activation passed as an argument.

What is given below

```
x = Dense(64)(x)
x = Activation('relu')(x)
```

is equivalent to

```
x = Dense(8, activation='relu')(x)
```


## SQL

#### What is SQL?
SQL stands for Structured Query Language , and it is used to communicate with the Database. This is a standard language used to perform tasks such as retrieval, updation, insertion and deletion of data from a database.

#### What is Database?
Database is nothing but an organized form of data for easy access, storing, retrieval and managing of data. This is also known as structured form of data which can be accessed in many ways.

#### What are the different subsets of SQL?

* DDL (Data Definition Language) – It allows you to perform various operations on the database such as `CREATE`, `ALTER` and `DELETE` objects.
* DML ( Data Manipulation Language) – It allows you to access and manipulate data. It helps you to `INSERT`, `UPDATE`, `DELETE` AND retrieve data from the database.
* DCL ( Data Control Language) – It allows you to control access to the database. Example – Grant, Revoke access permissions.

#### What is a query?
A database query is a request for data or information from a database table or combination of tables. A database query can be either a select query or an action query.

#### What is subquery?

A subquery is a query within another query. The outer query is called as main query, and inner query is called subquery, also known as nested query. SubQuery is always executed first and one time, and the result of subquery is passed on to the main query.

#### What is a primary key and a foreign key?

A primary key is a special database table column or combination of columns (also called Composite PRIMARY KEY) designated to uniquely identify all table records. A primary key's main features are:
1. It must contain a unique value for each row of the data. 
2. It cannot contain null values (it has an implicit NOT NULL constraint).

A table in SQL is strictly restricted to have one and only one primary key, which is comprised of single or multiple fields (columns).

A primary key is either an existing table column or a column that is specifically generated by the database according to a defined sequence.

A foreign key is a column or group of columns in a relational database table that provides a link between data in two tables. It acts as a cross-reference between tables because it references the primary key of another table, thereby, establishing a link between them. A table can have multiple foreign keys.

The table with the foreign key constraint is labelled as the child table, and the table containing the candidate key is labelled as the referenced or parent table.

Depending on what role the foreign key plays in a relation:
1. It can not be NULL if this foreign key is also a key attribute.
2. It can be NULL, if this foreign key is a normal attribute.

It can also contain duplicates. Whether it is unique or not unique relates to whether the table has a one-one or a one-many relationship to the parent table.

An example for relation between primary key and foreign key:

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/primary_foregin_keys.png?raw=true)

Note that the orders table contains two keys: one for the order and one for the customer who placed that order. In scenarios when there are multiple keys in a table, the key that refers to the entity being described in that table is called the primary key (PK) and other key is called a foreign key (FK).

In our example, `order_id` is a primary key in the orders table, while `customer_id` is both a primary key in the customers table and a foreign key in the orders table. Primary and foreign keys are essential to describing relations between the tables, and in performing SQL joins.

To add a primary key constraint into an existing table, we use the following syntax:

```sql
ALTER TABLE tablename
ADD PRIMARY KEY (column_name);
```

or

```sql
ALTER TABLE table_name
ADD CONSTRAINT MyPrimaryKey PRIMARY KEY (column1, column2...);
```

The basic syntax of ALTER TABLE to DROP PRIMARY KEY constraint from a table is as follows −

```sql
ALTER TABLE table_name
DROP CONSTRAINT MyPrimaryKey;
```

Foreign keys are added into an existing table using the ALTER TABLE statement. The following syntax is used:

```sql
ALTER TABLE child_table
ADD CONSTRAINT constraint_name FOREIGN KEY (c1) REFERENCES parent_table (p1);
```
In the above syntax, the child_table is the table that will contain the foreign key while the parent table shall have the primary keys. C1 and p1 are the columns from the child_table and the parent_table columns respectively.

#### In which order do SQL queries happen?

Consider the SQL SELECT statement syntax:

{% highlight sql %}
SELECT DISTINCT <TOP_specification> <select_list>
FROM <left_table>
<join_type> JOIN <right_table>
ON <join_condition>
WHERE <where_condition>
GROUP BY <group_by_list>
HAVING <having_condition>
ORDER BY <order_by_list>
{% endhighlight %}

![](https://jvns.ca/images/sql-queries.jpeg)
Source: [https://jvns.ca/blog/2019/10/03/sql-queries-don-t-start-with-select/](https://jvns.ca/blog/2019/10/03/sql-queries-don-t-start-with-select/){:target="_blank"}

the order is:

1. `FROM/JOIN` and all the `ON` conditions
2. `WHERE`
3. `GROUP BY`
4. `HAVING`
5. `SELECT` (including window functions)
6. `DISTINCT`
6. `ORDER BY`
7. `LIMIT`

In practice this order of execution is most likely unchanged from above. With this information, we can fine-tune our queries for speed and performance.

#### What are the different operators available in SQL?

There are three operators available in SQL, namely:

1. Arithmetic Operators such as `+`, `-`, `*`, `/`, and `%`.
2. Bitwise Operators such as `&`, `|`, and `^`.
3. Comparison Operators such as `=`, `!=`, `>`, `<`, `>=`, `<=` and `<>`.
4. Logical Operators such as `AND`, `NOT`, `OR`, `ANY`, `BETWEEN`, `LIKE`, and `IN`.


#### What is the difference between UNION and UNION ALL?

UNION is a Set Operator, which combines data sets vertically. To implement a Set Operator like UNION, both queries must return the same number of columns. The corresponding columns in the queries must have compatible data types.

UNION removes duplicate records (where all columns in the results are the same), UNION ALL does not.

```sql
SELECT
   column_1,
   column_2
FROM
   tbl_name_1
UNION (UNION ALL)
SELECT
   column_1,
   column_2
FROM
   tbl_name_2;
```

If the number of columns are different, you can use a trick with the NULL to make the datasets UNION compatible.

```sql
Select Col1, Col2, Col3, Col4, Col5 from Table1
Union
Select Col1, Col2, Col3, Null as Col4, Null as Col5 from Table2
```

#### How to avoid duplicate records in a query?

The `SELECT DISTINCT` query is used to return only unique values. It eliminates all the duplicated values.

#### What's the difference between VARCHAR and CHAR?

`VARCHAR(x)` is variable-length, which can have up to x characters. `CHAR(x)` is fixed length, which can only have exactly x characters. `CHAR` always uses the same amount of storage space per entry, while `VARCHAR` only uses the amount necessary to store the actual text. If your content is a fixed size, you'll get better performance with `CHAR`.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Screen%20Shot%202019-12-21%20at%2010.56.33.png?raw=true)

#### How to insert rows into a table?

```sql
INSERT INTO mytable
(column, another_column, ...)
VALUES (value_or_expr, another_value_or_expr, ...),
      (value_or_expr_2, another_value_or_expr_2,  ...),
       ...;
```

#### How to insert rows from one table to another?

The `INSERT INTO SELECT` statement copies rows from one table and inserts it into another table.

```sql
INSERT INTO table2 (column1, column2, column3, ...)
SELECT column1, column2, column3, ...
FROM table1
WHERE condition;
```

In order to use above query, you must have already created `table2`. When `table2` is ready you can insert rows from `table1`. 

If you want to create a new table and insert values into it from another table at the same time, use query below:

```sql
CREATE TABLE table2 as
select column1, column2, column3, ... from table1 where condition;
```

#### How to copy a table definition?

You want to create a new table having the same set of columns as an existing table. You do not want to copy the rows, only the column structure of the table:

```sql
CREATE TABLE new_table
AS 
SELECT * FROM existing_table where 1=0
```

The subquery in above query will return no rows.

#### How to update rows in a table?

```sql
UPDATE mytable
SET column = value_or_expr, 
    other_column = another_value_or_expr, 
    ...
WHERE condition;
```

#### How to delete rows in a table?

The command below will remove specific records defined by WHERE clause:

```sql
DELETE FROM mytable
WHERE condition;
```

Commit and Rollback can be performed after delete statement. `DELETE` command is a DML command.

If you want to delete all the rows:

```sql
DELETE FROM mytable
```

The `TRUNCATE` command also removes all rows of a table. We cannot use a `WHERE` clause in this. This operation cannot be rolled back. This command is a DDL command.

```sql
TRUNCATE TABLE table_name;
```

To remove all data from multiple tables at once, you separate each table by a comma (,) as follows:

```sql
TRUNCATE TABLE table_name1, table_name2, ...
```

#### How to create a new database table?

```sql
CREATE TABLE IF NOT EXISTS mytable (
    column DataType TableConstraint DEFAULT default_value,
    another_column DataType TableConstraint DEFAULT default_value,
    ...
);
```

#### What are the data types in PostgreSQL?

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Screen%20Shot%202020-02-03%20at%2009.22.09.png?raw=true)

#### What are the constraints?

Constraints are additional requirements for acceptable values in addition to those provided by data types. They allow you to define narrower conditions for your data than those found in the general purpose data types.

These are often reflections on the specific characteristics of a field based on additional context provided by your applications. For example, an `age` field might use the `int` data type to store whole numbers. However, certain ranges of acceptable integers do not make sense as valid ages. For instance, negative integers would not be reasonable in this scenario. We can express this logical requirement in PostgreSQL using constraints.

When you create a table, you can create a constraint using the CREATE TABLE command's CONSTRAINT clause. There are two types of constraints: column constraints and table constraints. In other words, postgreSQL allows you to create constraints associated with a specific column or with a table in general.

* Column level constraint is declared at the time of creating a table but table level constraint is created after table is created.
* Composite primary key must be declared at table level.
* All the constraints can be created at table level but for table level NOT NULL is not allowed.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Screen%20Shot%202020-02-03%20at%2009.21.57.png?raw=true)

Let's give an example how to define a column constraint and a table constraint. A column constraint is defined by:

```sql
CREATE TABLE person (
    . . .
    age int CHECK (age >= 0),
    . . .
);
```

and table constraint by:

```sql
CREATE TABLE person (
    . . .
    age int,
    . . .
    CHECK (age >= 0)
);
```

Note that multiple column constraints are separated by a space. For instance:

```sql
CREATE TABLE mytable(name CHAR(10) NOT NULL,
        id INTEGER REFERENCES idtable(id),
        age INTEGER CHECK (age > 0));
```

The syntax to add constraints to an existing table column is as follows:

```sql
ALTER TABLE table_name
ADD constaint_type (column_name);
```

To remove (drop) a constraint you need to know its name. If the name is known, it is easy to drop. Else, you need to find out the system-generated name.

```sql
ALTER TABLE table_name DROP CONSTRAINT constraint_name;
```

#### What is CHECK constraint?

You normally use the `CHECK` constraint at the time of creating the table using the `CREATE TABLE` statement. The following statement defines an `employees` table.

```sql
CREATE TABLE employees (
   id serial PRIMARY KEY,
   first_name VARCHAR (50),
   last_name VARCHAR (50),
   birth_date DATE CHECK (birth_date > '1900-01-01'),
   joined_date DATE CHECK (joined_date > birth_date),
   salary numeric CHECK(salary > 0)
);
```

The `employees` table has three `CHECK` constraints:

1. First, the birth date (`birth_date`) of the employee must be greater than "01/01/1900". If you try to insert a birth date before 0"1/01/1900", you will receive an error message.
2. Second, the joined date (`joined_date`) must be greater than the birth date (`birth_date`). This check will prevent from updating invalid dates in terms of their semantic meanings.
3. Third, the salary must be greater than zero, which is obvious.

#### What is a DEFAULT constraint?

A `DEFAULT` constraint is used to include a default value in a column when no value is supplied at the time of inserting a record. For example;

```sql
CREATE TABLE Persons
(
P_Id int NOT NULL,
LastName varchar(255) NOT NULL,
FirstName varchar(255),
Address varchar(255),
City varchar(255) DEFAULT 'Sandnes'
)
```

The SQL code above will create `City` column whose default value is `Sandnes`.

#### What is a UNIQUE constraint?

A `UNIQUE` constraint ensures that all values in a column are different. This provides uniqueness for the column(s) and helps identify each row uniquely. Unlike primary key, there can be multiple unique constraints defined per table.

```sql
CREATE TABLE Students ( 	 /* Create table with a single field as unique */
    ID INT NOT NULL UNIQUE
    Name VARCHAR(255)
);

CREATE TABLE Students ( 	 /* Create table with multiple fields as unique */
    ID INT NOT NULL
    LastName VARCHAR(255)
    FirstName VARCHAR(255) NOT NULL
    CONSTRAINT PK_Student
    UNIQUE (ID, FirstName)
);

ALTER TABLE Students 	 /* Set a column as unique */
ADD UNIQUE (ID);

ALTER TABLE Students 	 /* Set multiple columns as unique */
ADD CONSTRAINT PK_Student 	 /* Naming a unique constraint */
UNIQUE (ID, FirstName);
```

#### How to drop a table?

If a table is dropped, all things associated with the tables are dropped as well. This includes - the relationships defined on the table with other tables, the integrity checks and constraints, access privileges and other grants that the table has. Therefore, this operation cannot be rolled back.

```sql
DROP TABLE IF EXISTS mytable;
```

#### How to add a new column to a table?

```sql
ALTER TABLE mytable
ADD column DataType OptionalTableConstraint 
    DEFAULT default_value;
 ```
 
#### How to remove a column from a table? 

```sql
ALTER TABLE mytable
DROP column_to_be_deleted;
```

#### How to rename a table?

```sql
ALTER TABLE mytable
RENAME TO new_table_name;
```

#### What is the difference between BETWEEN and IN operators in SQL?

The BETWEEN operator is used to fetch rows based on a range of values.
For example,

```sql
SELECT * FROM Students WHERE ROLL_NO BETWEEN 20 AND 30;
```
This query will select all those rows from the table Students where the value of the field ROLL_NO lies between 20 and 30.

The IN operator is used to check for values contained in specific sets.
For example,

```sql
SELECT * FROM Students WHERE ROLL_NO IN (20,21,23);
```

This query will select all those rows from the table Students where the value of the field ROLL_NO is either 20 or 21 or 23.

#### What is the difference between primary key and unique constraints?

Primary key cannot have NULL value, the unique constraints can have NULL values. There is only one primary key in a table, but there can be multiple unique constrains. The primary key creates the cluster index automatically but the Unique key does not.

#### What is the default ordering of data using the ORDER BY clause? How could it be changed?

The default sorting order is ascending. It can be changed using the DESC keyword, after the column name in the ORDER BY clause.

#### What are the aggregate functions?

An aggregate function performs a calculation on a set of values, and returns a single value. Except for COUNT, aggregate functions ignore null values. Aggregate functions are often used with the GROUP BY clause of the SELECT statement.

```sql
aggregate_function_name(DISTINCT | ALL expression)
```

In this syntax;

1. First, specify the name of an aggregate function that you want to use such as AVG, SUM, and MAX.
2. Second, use DISTINCT if you want only distinct values are considered in the calculation or ALL if all values are considered in the calculation. By default, ALL is used if you don’t specify any modifier.
3. Third, the expression can be a column of a table or an expression that consists of multiple columns with arithmetic operators.

Some aggregate functions are given below:

1. AVG – calculates the average of a set of values.
2. COUNT – counts rows in a specified table or view.
3. MIN – gets the minimum value in a set of values.
4. MAX – gets the maximum value in a set of values.
5. SUM – calculates the sum of values.

#### What Is an Equi and Non-equi Join in SQL?

In SQL, a join doesn’t have to be based on identical matches. Here, non-equi join uses "non-equal" operators to match records. Like a self join, a SQL non equi join doesn’t have a specific keyword; you’ll never see the words NON EQUI JOIN in anyone’s SQL code. Instead, they are defined by the type of operator in the join condition: anything but an equals sign means a non-equi join. Below, we have some non equi join operators and their meanings:

| Operator | Meaning |
|-|-|
| “>” | Greater than |
| “>=” | Greater than or equal to |
| “<” | Less than |
| “<=” | Less than or equal to |
| “!=” | Not equal to |
| ”<>” | Not equal to (ANSI Standard) |
| BETWEEN … AND | Values in a range between x and y |

it’s good to know that a SQL non equi join can only be used with one or two tables. An example of non-equi join can be seen below:

```sql
SELECT first_name, last_name, min_price, max_price, price, city 
FROM person JOIN apartment ON apartment.id != person.apartment_id
    AND price BETWEEN min_price AND max_price
ORDER BY last_name;
```

The majority of SQL joins are equi joins. An equi join is any JOIN operation that uses an equals sign and only an equals sign (`=`). Calling such “standard” joins an equi-joins is just a fancy way to name it. You will see queries that use more than one join condition; if one condition is an equals sign and the other isn’t, that’s a considered a non equi join in SQL. An example of equi join can be seen below:

```sql
SELECT first_name, last_name, price, city 
FROM person 
JOIN  apartment  ON   apartment.id = person.apartment_id ;
```

#### What is a join in SQL? What are the types of joins?

A SQL Join is SQL way of linking data from two or more tables based on a column shared between tables. It allows you to gather data from multiple tables using one query because it is very unlikely that you will work exclusively with one table.

There are four basic types of SQL joins: inner, left, right and full. The easiest and the most intuitive way to explain the difference between these four types is by using Venn diagram.

Let's say we have two sets of data in our relational database: table X and table Y with some sort of relation specified by primary and foreign keys. The extent of the overlap, if any, is determined by how many records in Table X match the records in Table Y. Depending on what subset of data we would like to select from the two tables, the seven join types can be used:

1. (INNER) JOIN: Returns records that have matching values in both tables.
![](https://tableplus.com/assets/images/sql-joins/inner-join.png)

```sql
SELECT *
FROM table1
INNER JOIN table2
ON table1.col1 = table2.col2;
```

2. LEFT (OUTER) JOIN: Returns all records from the left table and matches the records from the right table. 
![](https://tableplus.com/assets/images/sql-joins/left-join.png)

```sql
SELECT *
FROM table1
LEFT JOIN table2
ON table1.col1 = table2.col2;
```

3. RIGHT (OUTER) JOIN: Returns all records from the right table and matches records from the left table.
![](https://tableplus.com/assets/images/sql-joins/right-join.png)

```sql
SELECT *
FROM table1
RIGHT JOIN table2
ON table1.col1 = table2.col2;
```

4. FULL (OUTER) JOIN: Returns all the records that match the ON condition, no matter which table they are stored in. It can be from table1, table2, or both. 
![](https://tableplus.com/assets/images/sql-joins/full-join.png)

```sql
SELECT *
FROM table1
FULL JOIN table2
ON table1.col1 = table2.col2;
```

5. LEFT (OUTER) JOIN without Intersection: This join type is a variant of the basic left outer join. It returns all rows from the left-hand table specified in the ON condition that also meets the join condition but None of the rows from the right-hand table that matches the join condition. In plain term, it can be understood as (LEFT JOIN) - (INNER JOIN).
![](https://tableplus.com/assets/images/sql-joins/left-join-no-intersection.png)

```sql
SELECT *
FROM table1
LEFT JOIN table2
ON table1.col1 = table2.col2
WHERE table2.col2 IS NULL;
```

6. RIGHT (OUTER) JOIN without Intersection: This join type is a variant of the basic right outer join. It returns all rows from the right-hand table specified in the ON condition that also meets the join condition but None of the rows from the left-hand table that matches the join condition. In plain term, it can be understood as (RIGHT JOIN) - (INNER JOIN).
![](https://tableplus.com/assets/images/sql-joins/right-join-no-intersection.png)

```sql
SELECT *
FROM table1
RIGHT JOIN table2
ON table1.col1 = table2.col2
WHERE table1.col1 IS NULL;
```

7. FULL (OUTER) JOIN without Intersection: This variant of the full outer join (sometimes abbreviated to full join) clause returns all records that match the ON condition, excluding those are in common between two tables, or those records exist in both tables. In plain term, it can be understood as (OUTER JOIN) - (INNER JOIN).
![](https://tableplus.com/assets/images/sql-joins/full-outer-join-no-intersection.png)

```sql
SELECT *
FROM table1
FULL JOIN table2
ON table1.col1 = table2.col2
WHERE table1.col1 IS NULL
OR table2.col2 IS NULL;
```

8. SELF JOIN:  It is a mechanism of joining a table to itself. You would use self join when you want to create a result of set joining records in a table with same other records from the same table.

```sql
SELECT column_name(s)
FROM table1 T1, table1 T2
WHERE T1.id=T2.id;
```

9. CROSS JOIN: This type of join returns all rows for all possible combinations of two tables. It is also known as Cartesian Join.

```sql
SELECT * 
FROM table1 
CROSS JOIN table2;
```

The following statement is also equivalent to the CROSS JOIN above:

```sql
SELECT * 
FROM table1, table2;
```

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/cross-join-round.png?raw=true)

#### What is the difference between a Fact Table and a Dimension Table?

The Fact Table and Dimension Table, are the essential factors to create a schema. In Data Warehouse Modeling, a **star schema** consists of Fact and Dimension tables.

Star schema is a mature modeling approach widely adopted by relational data warehouses. It requires modelers to classify their model tables as either dimension or fact. Fact table has numbers/measures/facts. Dimension tables give more context to a Fact table

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/star-schema-example2.png?raw=true)

**Dimension Tables**
* They provides descriptive information for all the measurements recorded in fact table. 
* They are relatively very small as comparison of fact table.
* Commonly used dimensions are people, products, place and time.
* Each dimension table contains its primary key. Every dimension table must have a primary key that uniquely identifies each record of the table.
* Dimension tables need to be created first.
* Dimension table contains more attributes and less records.
* Dimension table grows horizontally.

**Fact Tables**
* Fact tables store observations or events.
* It also contains all the primary keys of the dimension and associated facts or measures(is a property on which calculations can be made) like quantity sold, amount sold and average sales.
* The dimension key columns determine the dimensionality of a fact table, while the dimension key values determine the granularity of a fact table. For example, consider a fact table designed to store sale targets that has two dimension key columns Date and ProductKey. It's easy to understand that the table has two dimensions. The granularity, however, can't be determined without considering the dimension key values. In this example, consider that the values stored in the Date column are the first day of each month. In this case, the granularity is at month-product level.
* Fact table contains a primary key which is a concatenation of primary keys of all dimension table. The concatenated key of fact table must uniquely identify the row in a fact table.
* Fact table can be created only when dimension tables are completed.
* Fact table contains less attributes and more records.
Generally, dimension tables contain a relatively small number of rows. Fact tables, on the other hand, can contain a very large number of rows and continue to grow over time.
* Fact table grows vertically.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/022218_0758_StarandSnow1.png?raw=true)

Dimension table then could be say a Product table consisting of attributes like Product Name, Price, Date of Expiry etc along with the surrogate key for each  product in the store etc. This table contains values which are more likely to remain constant for a product.

Fact table would be, say Sales table which would contain the quantity, price, date of sale etc for each product sold along with the surrogate key for that product (foreign key). Fact tables store transaction data.

So basically your fact table will have much more data than the dimension table and data in dimension table changes less frequently compared to Fact tables. Surrogate keys from dimension tables are used in fact tables as foreign keys.

#### What is Data Normalization?

Normalization is a database design technique which organizes tables in a manner that reduces redundancy and dependency of data. It divides larger tables to  a number of smaller tables and links them using relationships. This process automatically reduces duplicate data and also automatically avoids insertion, update, deletion problems.

The most used data normalization technique is 3rd Normal Form (3NF). It states that all columns in referenced data that are not dependent on the primary key should be removed.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/3NF.gif?raw=true)

#### What is a View?

In SQL, a view is a virtual table based on the result-set of an SQL statement. A view contains rows and columns, just like a real table. The fields in a view are fields from one or more real tables in the database. 

It does not hold the actual data. It does not use the physical memory, only the query is stored in the data dictionary.

If you could make change to a view, then it will change the actual table, if the view is not updatable (READ-ONLY), then you can't make change to it.

A view refers to a logical snapshot based on a table or another view. It is used for the following reasons:

The view has primarily two purposes:

1. Simplify the complex SQL queries.
2. Provide restriction to users from accessing sensitive data 
3. to restrict access to hide data complexity
4. Ensure data independence

```sql
CREATE VIEW view_name AS
SELECT column1, column2, ...
FROM table_name
WHERE condition;
```

#### How to drop a view?

```sql
DROP VIEW IF EXISTS View_table; 
```

#### Which operator is used in query for pattern matching?

LIKE operator is used for pattern matching, and it can be used as -.

* % - Matches zero or more characters.
* \_ (Underscore) – Matching exactly one character.

```sql
Select * from Student where studentname like 'a%'
```

```sql
Select * from Student where studentname like 'ami_'
```

### What is an Index?

An index refers to a performance tuning method of allowing faster retrieval of records from the table. An index creates an entry for each value and hence it will be faster to retrieve data.

There are three types of index namely:

1. **Unique Index**: This index does not allow the field to have duplicate values if the column is unique indexed. If a primary key is defined, a unique index can be applied automatically.
  ```sql
  CREATE UNIQUE INDEX myIndex
  ON students (enroll_no);
  ```
2. **Clustered Index**: This index reorders the physical order of the table and searches based on the basis of key values. Each table can only have one clustered index. Note that PostgreSQL does not have a clustered index
3. **Non-Clustered Index**: Non-Clustered Index does not alter the physical order of the table and maintains a logical order of the data. Each table can have many nonclustered indexes.

#### What are the different types of relationships in SQL?

* **One-to-One** : This can be defined as the relationship between two tables where each record in one table is associated with the maximum of one record in the other table.
* **One-to-Many** and **Many-to-One** : This is the most commonly used relationship where a record in a table is associated with multiple records in the other table.
* **Many-to-Many** : This is used in cases when multiple instances on both sides are needed for defining a relationship.
* **Self Referencing Relationships** : This is used when a table needs to define a relationship with itself.

#### What is an Alias in SQL?

An alias is a temporary name assigned to the table or table column for the purpose of a particular SQL query. In addition, aliasing can be employed as an obfuscation technique to secure the real names of database fields. A table alias is also called a _correlation name_.

An alias is represented explicitly by the `AS` keyword but in some cases the same can be performed without it as well. Nevertheless, using the `AS` keyword is always a good practice.

#### How to get random records from a table?

```sql
SELECT col1, col1 from table1 order by random() limit 5;
```

This line of code will pick 5 records randomly every time you call.

#### How to concatenate two string columns?

```sql
select ename || ' WORKS AS A ' || job from emp where deptno = 10;
```

Or

```sql
select CONCAT(ename, ' WORKS AS ', job)  from emp where deptno = 10;
```

#### How to split a string?

The PostgreSQL `split_part` function is used to split a given string based on a delimiter and pick out the desired field from the string, start from the left of the string.

```sql
split_part(<string>,<delimiter>, <field_number>)
```

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/postgresql-split_part-function.png?raw=true)

```sql
select split_part('MUSTAFA MURAT ARAT', ' ', 1); --- MUSTAFA
select split_part('MUSTAFA MURAT ARAT', ' ', 2); --- MURAT
select split_part('MUSTAFA MURAT ARAT', ' ', 3); --- ARAT

SELECT split_part('Hello, My name is murat!', ' ', 5) --- murat!

SELECT split_part('Hello, My name is murat!', ',', 2) --- " My name is murat!"
```

#### How to retrieve values from one table that do not exists in another table?

`EXCEPT` query will return rows from the upper query (query before the `EXCEPT`) that do not exists in the lower query (query after the `EXCEPT`). 

```sql
select deptno from dept
except 
select deptno from emp;
```

will return the value of `deptno` which exists in dept table but not in emp table.

`EXCEPT` clause will not return duplicates. 

#### How does EXTRACT and DATE_PART work in PostgreSQL?

The PostgreSQL `EXTRACT()` function retrieves a field such as a year, month, and day from a date/time value.

```sql
EXTRACT(field FROM source)
```

The `DATE_PART()` function extracts a subfield from a date or time value. 

```sql
DATE_PART(field,source)
```

The `field` is an identifier that determines what field to extract from the `source`. The values of the `field` must be in a list of permitted values: century, decade, year, month, day, hour, minute, second, microseconds, milliseconds, dow, doy, epoch, isodow, isoyear, timezone, timezone_hour, timezone_minute

```sql
SELECT date_part('year',TIMESTAMP '2017-02-01 13:30:15'); --- 2017
SELECT extract('year' from TIMESTAMP '2017-02-01 13:30:15'); --- 2017

SELECT date_part('month',TIMESTAMP '2017-02-01 13:30:15'); --- 2
SELECT extract('month' from TIMESTAMP '2017-02-01 13:30:15'); --- 2

SELECT date_part('day',TIMESTAMP '2017-02-01 13:30:15'); --- 1
SELECT extract('day' from TIMESTAMP '2017-02-01 13:30:15'); --- 1

SELECT date_part('hour',TIMESTAMP '2017-02-01 13:30:15'); --- 13
SELECT extract('hour' from TIMESTAMP '2017-02-01 13:30:15'); --- 13

SELECT date_part('minute',TIMESTAMP '2017-02-01 13:30:15'); --- 30
SELECT extract('minute' from TIMESTAMP '2017-02-01 13:30:15'); --- 30

SELECT date_part('decade',TIMESTAMP '2017-02-01 13:30:15'); --- 201
SELECT extract('decade' from TIMESTAMP '2017-02-01 13:30:15'); --- 201

--- The number of seconds since 1970-01-01 00:00:00 UTC
SELECT date_part('epoch',TIMESTAMP '2017-02-01 13:30:15'); --- 1485955815
SELECT extract('epoch' from TIMESTAMP '2017-02-01 13:30:15'); --- 1485955815

SELECT EXTRACT(YEAR FROM INTERVAL '6 years 5 months 4 days 3 hours 2 minutes 1 second' ); --- 6
SELECT EXTRACT(MONTH FROM INTERVAL '6 years 5 months 4 days 3 hours 2 minutes 1 second' ); --- 5

SELECT DATE_PART('YEAR', INTERVAL '6 years 5 months 4 days 3 hours 2 minutes 1 second' ); --- 6
SELECT DATE_PART('MONTH', INTERVAL '6 years 5 months 4 days 3 hours 2 minutes 1 second' ); --- 5
```

#### What is the difference between EXTRACT and DATE_PART in PostgreSQL?

They both allow you to retrieve subfields e.g., year, month, week from a date or time value.

The `EXTRACT` syntax ends up as a call to the internal `date_part(...)` function. If SQL-portability is not a concern, calling `date_part()` directly should be a bit quicker.

#### How to convert from  12 hours timestamp format to 24 hours timestamp or other way around?

You can use `TO_CHAR()` function. The end result will be text.

```sql
select to_char(TIMESTAMP '2016-07-01 01:12:22 PM', 'yyyy-mm-dd hh24:mi:ss') --- 2016-07-01 13:12:22
```

Or from 24 hours to 12 hours, you can use Meridiem indicator (PM or AM), what are given below are equal:

```sql
select to_char(TIMESTAMP '2016-07-01 13:12:22', 'yyyy-mm-dd hh12:mi:ss PM') --- "2016-07-01 01:12:22 PM"
select to_char(TIMESTAMP '2016-07-01 11:12:22', 'yyyy-mm-dd hh12:mi:ss PM') --- "2016-07-01 11:12:22 AM"

select to_char(TIMESTAMP '2016-07-01 13:12:22', 'yyyy-mm-dd hh12:mi:ss AM') --- "2016-07-01 01:12:22 PM"
select to_char(TIMESTAMP '2016-07-01 11:12:22', 'yyyy-mm-dd hh12:mi:ss AM') --- "2016-07-01 11:12:22 AM"
```

#### How does AGE function work in PostgreSQL?

The `age()` function subtract arguments, producing a "symbolic" result that uses years and months.

```sql
age(timestamp, timestamp)
--- OR

age(timestamp)
```

```sql
SELECT age(timestamp '2015-01-15', timestamp '1972-12-28'); --- "42 years 18 days"
```

```sql
SELECT AGE(CURRENT_DATE, DATE '1989-09-18') --- "30 years 5 mons 5 days"
```

If you do not define the other time stamp, Postgresql finds the age between current date and the date as specified in the argument.

```sql
SELECT AGE(CURRENT_DATE, DATE '1989-09-18') --- "30 years 5 mons 5 days"
```

#### How to get yesterday's date?

```sql
SELECT TIMESTAMP 'yesterday' --- "2020-02-22 00:00:00"
SELECT current_date - 1 as Yesterday --- "2020-02-22"
select (current_date - interval '1 day')::date as Yesterday --- "2020-02-22"
select current_timestamp - interval '1 day' ---- "2020-02-22 13:49:32.615199-05"

select (DATE '2019-07-05' - Interval '1 Day')::Date --- "2019-07-04" (assuming 2019-07-05 is today)
```

#### How to get current date, time, timestamp?

The different options vary in precision. 

```sql
select NOW() --- "2020-02-23 13:50:25.058092-05" NOTE: THIS HAS TIMEZONE
select NOW()::TIMESTAMP --- "2020-02-23 13:56:45.638871"
--- The cast converts the timestamp to the current timestamp of your time zone. 
--- That's also how the standard SQL function LOCALTIMESTAMP is implemented in Postgres.


select current_date --- "2020-02-23"
select current_time --- "13:50:44.069109-05:00"  NOTE: THIS HAS TIMEZONE
select current_timestamp --- "2020-02-23 13:50:57.99141-05" NOTE: THIS HAS TIMEZONE
select timestamp 'NOW' --- "2020-02-23 13:51:19.856137"
select timeofday() --- "Sun Feb 23 13:52:06.027367 2020 EST" NOTE: THIS IS TEXT NOT TIME WITH TIMEZONE
```

`CURRENT_TIME` and `CURRENT_TIMESTAMP` deliver values with time zone; `LOCALTIME` and `LOCALTIMESTAMP` deliver values without time zone.

```sql
SELECT LOCALTIMESTAMP --- "2020-02-23 13:54:50.047158"
SELECT LOCALTIME --- "13:54:56.317438"
```

Timezones are computed from Coordinated Universal Time (or UTC). Right now, I am on Eastern Standard Time, which is 5 hours further from UTC so PostgreSQL shows `-05` or `-05:00`.

You can also set the timezone different than yours and compute the time:

```sql
SET TIMEZONE='US/Eastern';
select NOW() --- "2020-02-23 14:03:23.057357-05"

SET TIMEZONE='US/Pacific';
select NOW() --- "2020-02-23 11:03:41.265654-08"
```

In order to get current timezone,

```sql
show timezone; --- "US/Pacific"
```

```sql
SELECT * FROM pg_timezone_names;
```

will give the timezone names. Complete time-zones are given in [here](https://www.postgresql.org/docs/7.2/timezones.html).


To get 2 hours 30 minutes ago, you use the minus (-) operator as follows:

```sql
SELECT NOW() - interval '2 Hours 30 Minutes' --- "2020-02-23 11:40:07.788344-05"
```

To get 1 hour from now:
 
```sql
SELECT (NOW() + interval '1 hour') AS an_hour_later; --- "2020-02-23 15:10:38.840118-05"
```

#### How to use ROW_NUMBER(), RANK(), DENSE_RANK() window functions?

* ROW_NUMBER(): This one generates a new row number for every row, regardless of duplicates within a partition.
* RANK(): This one generates a new row number for every distinct row, leaving gaps between groups of duplicates within a partition.
* DENSE_RANK(): This one generates a new row number for every distinct row, leaving no gaps between groups of duplicates within a partition.

```sql
CREATE TABLE t AS
SELECT 'a' v UNION ALL
SELECT 'a'   UNION ALL
SELECT 'a'   UNION ALL
SELECT 'b'   UNION ALL
SELECT 'c'   UNION ALL
SELECT 'c'   UNION ALL
SELECT 'd'   UNION ALL
SELECT 'e';

SELECT
  v,
  ROW_NUMBER() OVER (ORDER BY v) row_number,
  RANK()       OVER (ORDER BY v) rank,
  DENSE_RANK() OVER (ORDER BY v) dense_rank
FROM t;
```

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Screen%20Shot%202020-02-21%20at%2008.49.35.png?raw=true)

As you can see clearly from the output:

* The first, second and third rows receive the same rank because they have the same value A.
* The fourth row receives the rank 4 because the `RANK()` functions the rank 2 and 3. The fifth and sixth rows receive the rank 5 because of the same reason as before and so on...


The `RANK`, `DENSE_RANK` and `ROW_NUMBER` Functions have the following similarities:
1. All of them require an `ORDER BY` clause.
2. All of them return an increasing integer with a base value of 1.
3. When combined with a `PARTITION BY` clause, all of these functions reset the returned integer value to 1.
4. If there are no duplicated values in the column used by the `ORDER BY` clause, these functions return the same output.

#### How to find modulus?

The PostgreSQL `MOD()` function performs the modulo operation that returns the remainder after division of the first argument by the second one.

```sql
MOD(x,y)
```

```sql
SELECT  MOD(15,4) --- 3

SELECT MOD(15,-4) ---3

SELECT MOD(-15,4); --- -3

SELECT MOD(-15,-4); --- -3
```

#### How to use DATE_TRUNC Function?

The `date_trunc` function truncates a TIMESTAMP or an  INTERVAL value based on a specified date part e.g., hour, week, or month and returns the truncated timestamp or interval with a level of precision.

```sql
date_trunc('datepart', field)
```

```sql
SELECT date_trunc('hour', TIMESTAMP '2017-03-17 02:09:30'); --- "2017-03-17 02:00:00"
SELECT date_trunc('minute', TIMESTAMP '2017-03-17 02:09:30'); --- "2017-03-17 02:09:00"

SELECT date_trunc('year', INTERVAL '6 years 5 months 4 days 3 hours 2 minutes 1 second'); --- "6 years"
SELECT date_trunc('month', INTERVAL '6 years 5 months 4 days 3 hours 2 minutes 1 second'); --- "6 years 5 mons"
SELECT date_trunc('day', INTERVAL '6 years 5 months 4 days 3 hours 2 minutes 1 second'); --- "6 years 5 mons 4 days"
SELECT date_trunc('hour', INTERVAL '6 years 5 months 4 days 3 hours 2 minutes 1 second'); --- "6 years 5 mons 4 days 03:00:00"
SELECT date_trunc('minute', INTERVAL '6 years 5 months 4 days 3 hours 2 minutes 1 second'); --- "6 years 5 mons 4 days 03:02:00"
SELECT date_trunc('second', INTERVAL '6 years 5 months 4 days 3 hours 2 minutes 1 second'); --- "6 years 5 mons 4 days 03:02:01"
```

#### How to use REPLACE and TRANSLATE functions? 

Sometimes, you want to search and replace a string in a column with a new one such as replacing outdated phone numbers, broken URLs, and spelling mistakes.

To search and replace all occurrences of a string with a new one, you use the REPLACE() function.

```sql
REPLACE(source, old_text, new_text );
```

This function is case-sensitive

```sql
SELECT REPLACE ('ABC AA', 'A', 'Z'); --- "ZBC ZZ"

SELECT REPLACE ('Mustafa Murat ARAT', 'a', 'p');
```

You can also use `TRANSLATE` function. The PostgreSQL `TRANSLATE()` function performs several single-character, **one-to-one** translation in one operation.

```sql
TRANSLATE(string, from, to)
```

There are some cases where both functions will return the same result. Like this:

```sql
SELECT REPLACE('ABC AA', 'A', 'Z'); --- "ZBC ZZ"
SELECT REPLACE ('Mustafa Murat ARAT', 'a', 'p'); --- "Mustpfp Murpt ARAT"

SELECT TRANSLATE('ABC AA', 'A', 'Z'); --- "ZBC ZZ"
SELECT TRANSLATE('Mustafa Murat ARAT', 'a', 'p'); --- "Mustpfp Murpt ARAT"
```

Now for an example that demonstrates one of the differences between `TRANSLATE()` and `REPLACE()`:

```sql
--- First Example
SELECT REPLACE('Mustafa Murat ARAT', 'Murat', 'Jack'); --- "Mustafa Jack ARAT"
SELECT TRANSLATE('Mustafa Murat ARAT', 'Murat', 'Jack'); --- "Jaskfk Jack ARAT"

--- Second Example
SELECT REPLACE('Mustafa Murat ARAT', 'urat', 'ack'); --- "Mustafa Mack ARAT"
SELECT TRANSLATE('Mustafa Murat ARAT', 'urat', 'ack'); --- "Maskfk Mack ARAT"

--- Third Example
SELECT REPLACE('Mustafa Murat ARAT', 'Tarum', 'Jack'); --- "Mustafa Murat ARAT"
SELECT TRANSLATE('Mustafa Murat ARAT', 'Tarum', 'Jack'); --- "Mkstafa Mkcat ARAJ"
```

For the third example, `REPLACE()` has no effect (it returns the original string) because the second argument is not an exact match for the first argument (or a substring within it, like second example). Even though the second argument contains the correct characters, they are not in the same order as the first argument, and therefore, the whole string doesn't match.

`TRANSLATE()` does have an effect because each character in the second argument is present in the first argument. It doesn’t matter that they are in a different order, because each character is translated one by one. PostgreSQL translates the first character, then the second, then the third, and so on.

Similar to the previous example, you can also get different results when the first argument contains the characters in the second argument, but they are non-contiguous:

```sql
SELECT REPLACE('Mustafa Murat ARAT', 'MARAT', 'Jack'); --- "Mustafa Murat ARAT"
SELECT TRANSLATE('Mustafa Murat ARAT', 'MARAT', 'Jack'); --- "Justafa Jurat aca"
```

As you can see, `REPLACE` function returned the same string because `MARAT` is not contained in the first argument, however, `TRANSLATE` function returned one-by-one translation. From this example, we cann see another feature of `TRANSLATE()` function, that it removes the extra character in the string 'MARAT', which is 'T', from the string 'Jack'.

Another example for non-Contiguous Strings:

```sql
SELECT 
    REPLACE('1car23', '123', '456') AS Replace, --- "1car23"
    TRANSLATE('1car23', '123', '456') AS Translate; --- "4car56"
```

#### How to pad a string on left or right?

The PostgreSQL `LPAD()` function pads a string on the left to a specified length with a sequence of characters.

```sql
LPAD(string, length[, fill])    
```

`length` is an positive integer that specifies the length of the result string **after padding**.

Similarly, `RPAD()` function pads a string on the right.

```sql
SELECT LPAD('PostgreSQL',15,'*'); --- "*****PostgreSQL"

SELECT RPAD('PostgreSQL',15,'*'); --- "PostgreSQL*****"
```

In this example, the length of the PostgreSQL string is 10, the result string should have the length 15, therefore, the `LPAD()` function pads 5 character `*` on the left of the string. Similarly, `RPAD()` function pads 5 character `*` on the right of the string.

#### How to convert a value of one data type into another?

There are many cases that you want to convert a value of one data type into another. PostgreSQL provides you with the `CAST` operator that allows you to do this.

```sql
CAST ( expression AS target_type );
```

For example:

```sql
SELECT '2019-06-15 14:30:20'::timestamp;
```

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Screen%20Shot%202020-02-24%20at%2008.19.33.png?raw=true)

```sql
SELECT '15 minute'::interval,
 '2 hour'::interval,
 '1 day'::interval,
 '2 week'::interval,
 '3 month'::interval;
```

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Screen%20Shot%202020-02-24%20at%2008.20.10.png?raw=true)

```sql
SELECT
   CAST ('2015-01-01' AS DATE),
   CAST ('01-OCT-2015' AS DATE);
```

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Screen%20Shot%202020-02-24%20at%2008.21.30.png?raw=true)

```sql
SELECT
   CAST ('100' AS INTEGER);
```

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Screen%20Shot%202020-02-24%20at%2008.22.15.png?raw=true)


Besides the type CAST syntax, you can use the following syntax to convert a value of one type into another:

```sql
expression::type
```

For example,

```sql
SELECT
  '100'::INTEGER, --- 100
  '01-OCT-2015'::DATE; --- "2015-10-01"
```

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Screen%20Shot%202020-02-24%20at%2008.24.02.png?raw=true)

#### How to find the position of a substring in a string?

The PostgreSQL `POSITION()` function returns the location of a substring in a string.

```sql
POSITION(substring in string)
```

The `POSITION()` function returns an integer that represents the location of the substring within the string.

The `POSITION()` function returns zero (0) if the substring is not found in the string. It returns null if either substring or string argument is null.

Note that the `POSITION()` function searches for the substring case-sensitively.

```sql
SELECT POSITION('Murat' IN 'Mustafa Murat ARAT') --- 9

SELECT POSITION('arat' IN 'Mustafa Murat ARAT') --- 0 Case-sensitive

SELECT POSITION(Null IN 'Mustafa Murat ARAT') --- [null]
```

#### How to replace Null in PostgreSQL?

Occasionally, you will end up with a dataset that has some nulls that you'd prefer to contain actual values. This happens frequently in numerical data (displaying nulls as 0 is often preferable), and when performing outer joins that result in some unmatched rows. In cases like this, you can use `COALESCE` to replace the null values.

SQL Server supports `ISNULL` function that replaces NULL with a specified replacement value:

```sql
ISNULL(expression, replacement)
```

If the `expression` is NULL, then the `ISNULL` function returns the `replacement`. Otherwise, it returns the result of the `expression`.

PostgreSQL does not have the `ISNULL` function. However, you can use the `COALESCE` function which provides the similar functionality. Note that the `COALESCE` function returns the first non-null argument, so the following syntax has the similar effect as the `ISNULL` function above:

```sql
COALESCE(expression,replacement)
```

```sql
SELECT COALESCE(Null, 2) --- will return 2
```

In addition to `COALESCE` function, you can use the `CASE` expression:

```sql
SELECT 
    CASE WHEN expression IS NULL 
            THEN replacement 
            ELSE expression 
    END AS column_alias;
```

#### How to find date differences?

```sql
select '2020-01-12'::date - '2020-01-01'::date;
select DATE '2020-01-12' - DATE '2020-01-01';
select extract('Day' from DATE '2020-01-12') - extract('Day' from DATE '2020-01-01')
select DATE_PART('Day', DATE '2020-01-12') - DATE_PART('Day', DATE '2020-01-01')
```

#### How to extract n number of characters specified in the argument from the left or right of a given string?

The PostgreSQL `LEFT()` function returns the first *n* characters in the string.

```sql
LEFT(string, n) 
```

For example, let's extract the first 2 letters from the string 'murat':

```sql
SELECT left('murat', 2) --- mu
```

Simirlary, `RIGHT()` function returns the last *n* characters in the string.

```sql
RIGHT(string, n) 
```

For example, let's extract the last 2 letters from the string 'murat':

```sql
SELECT right('murat', 2) --- at
```

#### How to extract a part of string?

The `SUBSTRING` function returns a part of string. 

```sql
SUBSTRING ( string, start_position , length)
```

For example:

```sql
SELECT
   SUBSTRING ('PostgreSQL', 1, 8); -- PostgreS
   
SELECT
   SUBSTRING ('PostgreSQL', 8); -- SQL # We do not need to submit length parameter
   
SELECT
   SUBSTRING ('PostgreSQL', length('PostgreSQL')-2, 3) -- SQL
```

#### How to remove (trim) characters from the beginning, end or both sides of a string?

The syntax of LTRIM() and RTRIM() function are as follows:

```sql
LTRIM(string, [character]);
RTRIM(string, [character]);
BTRIM(string, [character]);
```

This is equivalent to the following syntax of the TRIM() function:

```sql
TRIM(LEADING character FROM string); -- LTRIM(string,character)
TRIM(TRAILING character FROM string); -- RTRIM(string,character)
TRIM(BOTH character FROM string); -- BTRIM(string,character)
```

he TRIM function takes 3 arguments. First, you have to specify whether you want to remove characters from the beginning ('leading'), the end ('trailing'), or both ('both'). Next you must specify all characters to be trimmed. Any characters included in the single quotes will be removed from both beginning, end, or both sides of the string. Finally, you must specify the text you want to trim using `FROM`.

```sql
SELECT LTRIM('enterprise', 'e'); --- "nterprise"
SELECT TRIM(LEADING 'e' FROM 'enterprise') --- "nterprise"

SELECT RTRIM('enterprise', 'e'); --- "enterpris"
SELECT TRIM(TRAILING 'e' FROM 'enterprise') --- "enterpris"

SELECT BTRIM('enterprise', 'e'); --- "nterpris"
SELECT TRIM(BOTH 'e' FROM 'enterprise') --- "nterpris"
```

#### How to randomly select a row?

```sql
SELECT column FROM table
ORDER BY RANDOM()
ORDER BY 1
```

But this is not the best solution. What happens when you run such a query? Let’s say you run this query on a table with 10000 rows, than the SQL server generates 10000 random numbers, scans this numbers for the smallest one and gives you this row. Generating random numbers is relatively expensive operation, scaning them for the lowest one (if you have LIMIT 10, it will need to find 10 smallest numbers) is also not so fast (if quote is text it’s slower, if it’s something with fixed size it is faster, maybe because of need to create temporary table).


Therefore a faster way is given below:

```sql
select * from mytable offset floor(random() * (select count(*) from mytable)) limit 1 ;
```

The reason why we use `FLOOR()` function is that `FLOOR(RANDOM()*N)` where N is the number of records in the table, is guaranteed to be $0 .. . N-1$ and never $N$. Because `RANDOM()` returns a completely random number >= 0 and <1 and we will have always have 1 observation left after `OFFSET`.

#### What is the Dual Table? How is Oracle to PostgreSQL conversion?

Oracle uses the table DUAL for selects where actually no table name would be necessary, since the FROM clause in Oracle is mandatory. In PostgreSQL we can omit the FROM clause at all (PostgreSQL has implicit DUAL table). This table can be created in postgres as a view to ease porting problems. This allows code to remain somewhat compatible with Oracle SQL without annoying the Postgres parser.

In any case when migrating, if possible, just remove the "FROM DUAL" clause from the statement. Joins with dual are really rare - and peculiar.

For example, in Oracle, 

```sql
SELECT UPPER('hello') FROM DUAL
```

But in PostgreSQL, what's given below is sufficient:

```sql
SELECT UPPER('hello')
```

You can just create one , if you really want to:

```sql
CREATE TABLE dual();
```


## Miscellaneous

#### What is a good code?

Properties of a good code are:

1. Readable
2. Time efficiency
2. Space (Memory) efficiency

#### What is a data structure?

At the backbone of every program or piece of software there are two entities: data and algorithms. Algorithms transform data into something a program can effectively use. Data structure is a specialized format for organizing, processing, retrieving and storing data. This "format" allows a data structure to be efficient in some operations and inefficient in others.

#### What are the commonly used Data Structures?

The most commonly used data structures are

1. Arrays
2. Stacks
3. Queues (Circular Queues)
4. Linked Lists (Singly-linked list, Doubly linked list, Circular linked list, Circular double linked list)
5. Trees (and varianst: N-ary Tree, Balanced Tree, Binary Tree, Binary Search Tree, AVL Tree, Red Black Tree, 2–3 Tree, Heap)
6. Graphs
7. Tries (Prefix Trees)
8. Hash Tables
9. Skip Lists

Each of these data structures has its own advantages and disadvantages.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/cheatdiagram.png?raw=true)

#### What is a big-O notation?

In simplest terms, Big O notation is a way to measure performance of an operation in terms of space (memory) and time, based on the input size, known as n.

#### What are the different big-O notation measures?

There are a number of common Big O notations which we need to be familiar with.

Let’s consider $n$ to be the size of the input collection. In terms of time complexity:

1. **O(1)**: (also known as Constant time.) No matter how big your collection is, the time it takes to perform an operation is constant. it means upperbounded by a constant, i.e a function that is bounded above by a constant. In other words, it is not proportional to the length/size/magnitude of the input (independent of the input size). i.e., for any input, it can be computed in the same amount of time (even if that amount of time is really long).
2. **O(log n)**: When the size of a collection increases, the time it takes to perform an operation increases logarithmically. This is the logarithmic time complexity notation. Potentially optimised searching algorithms are O(log n).
3. **O(n)**: The time it takes to perform an operation is directly and linearly proportional to the number of items in the collection. This is the linear time complexity notation. This is some-what in-between or medium in terms of performance. As an instance, if we want to sum all of the items in a collection then we would have to iterate over the collection. Hence the iteration of a collection is an O(n) operation.
4. **(n log n)**: Where the performance of performing an operation is a quasilinear function of the number of items in the collection. This is known as the quasilinear time complexity notation. Time complexity of optimised sorting algorithm is usually n(log n).
5. **O($n^{2}$)**: When the time it takes to perform an operation is proportional to the square of the items in the collection. This is known as the quadratic time complexity notation.
6. **(n!)**: When every single permutation of a collection is computed in an operation and hence the time it takes to perform an operation is factorial of the size of the items in the collection. This is known as factorial time complexity notation. It is very slow.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/big_o_order.png?raw=true)

#### What is space complexity and time complexity?

Sometimes, there are more than one way to solve a problem. We need to learn how to compare the performance different algorithms and choose the best one to solve a particular problem. While analyzing an algorithm, we mostly consider **time complexity** and **space complexity**. **Time complexity** of an algorithm quantifies the amount of time taken by an algorithm to run as a function of the length of the input. Similarly, **Space complexity** of an algorithm quantifies the amount of space or memory taken by an algorithm to run as a function of the length of the input.

Time and space complexity depends on lots of things like hardware, operating system, processors, etc. However, we don't consider any of these factors while analyzing the algorithm. We will only consider the execution time of an algorithm.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Screen%20Shot%202019-10-13%20at%2021.25.05.png?raw=true)

#### What are the best, average, worst case scenarios in big-O notation?

1. **Best case scenario**: As the name implies, these are the scenarios when the data structures and the items in the collection along with the parameters are at their optimum state. As an instance, imagine we want to find an item in a collection. If that item happens to be the first item of the collection then it is the best-case scenario for the operation.
2. **Average case scenario** is when we define the complexity based on the distribution of the values of the input.
3. **Worst case scenario** could be an operation that requires finding an item that is positioned as the last item in a large-sized collection such as a list and the algorithm iterates over the collection from the very first item.

#### What are the built-in data types in Python?

Python has the following data types built-in by default, in these categories:

1. **Text Type**:	str
2. **Numeric Types**:	int, float, complex
3. **Sequence Types**:	list, tuple, range
4. **Mapping Type**:	dict
5. **Set Types**:	set, frozenset
6. **Boolean Type**:	bool
7. **Binary Types**:	bytes, bytearray, memoryview

In Python, the data type is set when you assign a value to a variable:

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/data_types_python.png?raw=true)

If you want to specify the data type, you can use the following constructor functions:

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/setting_data_types.png?raw=true)

#### What are the data structures in Python?

There are quite a few data structures available. The built-in data structures are: lists, tuples, dictionaries, strings, sets and frozensets.

Lists, strings and tuples are ordered sequences of objects. Unlike strings that contain only characters, list and tuples can contain any type of objects. Lists and tuples are like arrays. Tuples like strings are immutables. Lists are mutables so they can be extended or reduced at will. Sets are mutable unordered sequence of unique elements whereas frozensets are immutable sets.

In Python, dynamic arrays are lists. Linked lists do not exist in Python. Python lists are dynamic arrays, not linked lists. Python also uses lists as stacks. There is no direct support for graphs, too, in Python. 

But we can create our own data structures by using those already built-in Python structures.

#### What is `*args` and `**kwargs` in Python?

it is not necessary to write *args or `**kwargs`. Only the `*` (asterisk) is necessary. You could have also written `*var` and `**vars`. Writing *args and `**kwargs` is just a convention. `*args` and `**kwargs` are mostly used in function definitions. `*args` and `**kwargs` allow you to pass a variable number of arguments to a function. What variable means here is that you do not know beforehand how many arguments can be passed to your function by the user so in this case you use these two keywords. `*args` is used to send a non-keyworded variable length argument list to the function. `**kwargs` allows you to pass keyworded variable length of arguments to a function. You should use `**kwargs` if you want to handle named arguments in a function. 

{% highlight python %}
def multiply(x, y):
    print (x * y)
    
multiply(5, 4)
# 20

multiply(5, 4, 3)
# TypeError: multiply() takes 2 positional arguments but 3 were given

def multiply(*args):
    z = 1
    for num in args:
        z *= num
    print(z)

multiply(4, 5)
multiply(10, 9)
multiply(2, 3, 4)
multiply(3, 5, 10, 6)

# 20
# 90
# 24
# 900

def print_values(**kwargs):
    for key, value in kwargs.items():
        print("The value of {} is {}".format(key, value))

print_values(
            name_1="Alex",
            name_2="Gray",
            name_3="Harper",
            name_4="Phoenix",
            name_5="Remy",
            name_6="Val"
        )
# The value of name_1 is Alex
# The value of name_2 is Gray
# The value of name_3 is Harper
# The value of name_4 is Phoenix
# The value of name_5 is Remy
# The value of name_6 is Val
{% endhighlight %}

When ordering arguments within a function or function call, arguments need to occur in a particular order:

1. Formal positional arguments
2. `*args`
3. Keyword arguments
4. `**kwargs`

{% highlight python %}
def example2(arg_1, arg_2, *args, kw_1="shark", kw_2="blobfish", **kwargs):
...
{% endhighlight %}

#### What are the mutable and immutable objects in Python?

Every variable in python holds an instance of an object. There are two types of objects in python i.e. Mutable and Immutable objects. Whenever an object is instantiated, it is assigned a unique object id. The type of the object is defined at the runtime and it can't be changed afterwards. However, it’s state can be changed if it is a mutable object.

To summarise the difference, mutable objects can change their state or contents and immutable objects can’t change their state or content.

A mutable object can be changed after it is created, and an immutable object can not. 

1. **Mutable objects**: list, dict, set, byte array
2. **Immutable objects**: int, float, complex, string, tuple, frozen set (immutable version of set), bytes

* Mutable and immutable objects are handled differently in python. Immutable objects are quicker to access and are expensive to change because it involves the creation of a copy. Whereas mutable objects are easy to change.
* Use of mutable objects is recommended when there is a need to change the size or content of the object.
* Exception : However, there is an exception in immutability as well. We know that tuple in python is immutable. But the tuple consists of a sequence of names with unchangeable bindings to objects. Consider a tuple

  ```
  tup = ([3, 4, 5], 'myname')
  ```

  The tuple consists of a string and a list. Strings are immutable so we can’t change its value. But the contents of the list can change. The tuple itself isn't mutable but contain items that are mutable.

#### What is the difference between linked list and array?

**Array**
1. Array is a collection of elements of similar data type. Arrays are one of the oldest, most commonly used data structures.
2. Array supports Random Access, which means elements can be accessed directly using their index, like arr[0] for 1st element, arr[6] for 7th element etc. Hence, accessing elements in an array is fast with a constant time complexity of O(1).
3. In array, each element is independent and can be accessed using it's index value. Index is most commonly 0 based.
4. Array can single dimensional, two dimensional or multidimensional
5. Size of the array must be specified at time of array decalaration.
6. Array gets memory allocated in the Stack section.
7. In an array, elements are stored in contiguous memory location or consecutive manner in the memory.
8. Memory is allocated as soon as the array is declared, at compile time. It's also known as Static Memory Allocation.
9. In array, Insertion and Deletion operations take more time, as the memory locations are consecutive and fixed and you are shifting indexes while running those operations.

Linear arrays, or one dimensional arrays, are the most basic. They are static in size, meaning that they are declared with a fixed size. Dynamic arrays are like one dimensional arrays, but have reserved space for additional elements. If a dynamic array is full, it copies its contents to a larger array. Multi dimensional arrays nested arrays that allow for multiple dimensions such as an array of arrays providing a 2 dimensional spacial representation via x, y coordinates.

Keep in mind that unless you're writing your own data structure (e.g. linked list in C), it can depend dramatically on the implementation of data structures in your language/framework of choice.

**Time Complexity for Arrays:**

* Access (Indexing): Linear array: O(1), Dynamic array: O(1)
* Search: Linear array: O(n), Dynamic array: O(n)
* Deletion: Linear array: O(n)
* Insertion: Linear array: O(n) Dynamic array: O(n)

There are two different types of access: *random access* and *sequential access*. Sequential access means reading the elements one by one, starting at the first element. Linked List can only do sequential access. If you want to read the 10th element of a linked list,  you have to read the first 9 elements and follow the links to the 10th element. Arrays can allow random access through indexing. This is the reason arrays are faster at reads. 

It takes O(n) time to find the element you want to delete. Then in order to delete it, you must shift all elements to the right of it one space to the left. The same case is valid for insertion. For searching, you have to traverse all the nodes. Indexing is O(1) because when you know the index of the element, you can access it easily.

**Linked List**
1. Linked list is considered as non-primitive data structure contains a collection of unordered linked elements known as nodes.
2. Linked List supports Sequential Access, which means to access any element/node in a linked list, we have to sequentially traverse the complete linked list, upto that element. To access nth element of a linked list, time complexity is O(n).
3. In case of a linked list, each node/element points to the next, previous, or maybe both nodes.
4. Linked list can be Linear (Singly), Doubly or Circular linked list. Doubly linked list has nodes that also reference the previous node. Circularly linked list is simple linked list whose tail, the last node, references the head, the first node.
5. Size of a Linked list is variable. Linked lists are dynamic and flexible and can expand and contract its size. It grows at runtime, as more nodes are added to it.
6. Whereas, linked list gets memory allocated in Heap section.
7. In a linked list, new elements can be stored anywhere in the memory. Address of the memory location allocated to the new element is stored in the previous node of linked list, hence formaing a link between the two nodes/elements.
8. Memory is allocated at runtime/execution time, as and when a new node is added. It's also known as Dynamic Memory Allocation. It has poor locality, the memory used for linked list is scattered around in a mess. In contrast with, arrays which uses a contiguous addresses in memory. Arrays (slightly) benefits from processor caching since they are all near each other
9. Insertion and Deletion operations are fast in linked list.

Keep in mind that unless you're writing your own data structure (e.g. linked list in C), it can depend dramatically on the implementation of data structures in your language/framework of choice.

**Time Complexity for Linked List:**
* Access (Indexing): Singly Linked Lists: O(n), Doubly Linked Lists: O(n)
* Search: Linked Lists: O(n), Doubly Linked Lists: O(n)
* Insertion: Linked Lists: O(1), Doubly Linked Lists: O(1)
* Deletion: Linked Lists: O(1), Doubly Linked Lists: O(1)

There are two different types of access: *random access* and *sequential access*. Sequential access means reading the elements one by one, starting at the first element. Linked List can only do sequential access. If you want to read the 10th element of a linked list,  you have to read the first 9 elements and follow the links to the 10th element. Arrays can allow random access through indexing. This is the reason arrays are faster at reads. 

A linked list can typically only be accessed via its head node. From there you can only traverse from node to node until you reach the node you seek. Thus access is O(n).

Searching for a given value in a linked list similarly requires traversing all the elements until you find that value. Thus search is O(n).

Inserting into a linked list requires re-pointing the previous node (the node before the insertion point) to the inserted node, and pointing the newly-inserted node to the next node. Thus insertion is O(1).

Deleting from a linked list requires re-pointing the previous node (the node before the deleted node) to the next node (the node after the deleted node). Thus deletion is O(1).

It is worth mentioning that insertions and deletions are O(1) time only in Linked Lists, if you can instantly access the element to be deleted. It is common practice to keep track of the first and last items in a Linked List. 

Sometimes, append and prepend can also be used. Append means appending a new element at the tail and prepend means adding a new head. Append is a O(1) operation because we have the tail and there is no need to traverse through the entire linked list. Similarly, prepend is a O(1) operation.

#### What is the difference between stack and queue?

Stack is a linear data structure.

A stack is an ordered list where you can insert or delete only the last added element. A real-life example of Stack could be a pile of books placed in a vertical order. In order to get the book that’s somewhere in the middle, you will need to remove all the books placed on top of it. This is how the LIFO (Last In First Out) method works. A stack is a limited access data structure - elements can be added and removed from the stack only at the top. push adds an item to the top of the stack, pop removes the item from the top. A stack is a recursive data structure. 

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/stack_pop.jpg?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/stack_push.jpg?raw=true)

{% highlight python %} 
letters = []

# Let's push some letters into our list
#append adds an element at the end of an existing list
letters.append('c')
letters.append('a')
letters.append('t')
letters.append('g')

# Now let's pop letters, we should get 'g'
last_item = letters.pop()
print(last_item)
#g

# If we pop again we'll get 't'
last_item = letters.pop()
print(last_item)
#t

# 'c' and 'a' remain
print(letters) 
#['c', 'a']
{% endhighlight %}

We can create a class for this data structure in Python:

{% highlight python %} 
class Stack():
    def __init__(self):
        self.stack = []
        self.length = 0
        
    def __str__(self):
        return str(self.__dict__)
    
    def pop(self):
        if len(self.stack) < 1:
            return None
        self.length -= 1
        return self.stack.pop()
    
    def push(self, item):
        self.length += 1
        return self.stack.append(item)
    
    def size(self):
        return len(self.stack)
    #peek at the last element
    def peek(self):
        return self.stack[self.length - 1]
    
stack_example = Stack()

stack_example.push('c')
print(stack_example.length)
#1

stack_example.push('a')
print(stack_example.length)
#2

stack_example.push('t')
print(stack_example.length)
#3

stack_example.push('g')
print(stack_example.length)
#4

stack_example.size()
#4

print(stack_example)
#{'stack': ['c', 'a', 't', 'g'], 'length': 4}

last_item = stack_example.pop()
print(last_item)
#g
print(stack_example.length)
#3

last_item = stack_example.pop()
print(last_item)
#t
print(stack_example.length)
#2

stack_example.size()
#2

print(stack_example)
#{'stack': ['c', 'a'], 'length': 2}

print(stack_example.peek())
#a
{% endhighlight %}

Stack has O(n) lookup, O(1) pop, O(1) push, and O(1) peek.

A queue is an ordered list where you can delete the first added element (at the "front" of the queue) and insert an element at the "rear" of the queue. The only significant difference between Stack and Queue is that instead of using the LIFO method, Queue implements the FIFO method, which is short for First in First Out.  In the queue only two operations are allowed **enqueue** and **dequeue**. Enqueue means to insert an item into the back of the queue, dequeue means removing the front item. A perfect real-life example of Queue: a line of people waiting at a ticket booth. If a new person comes, they will join the line from the end, not from the start — and the person standing at the front will be the first to get the ticket and hence leave the line.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/queue_enqueue.jpg?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/queue_dequeue.jpg?raw=true)

{% highlight python %} 
fruits = []

# Let's enqueue some fruits into our list
fruits.append('banana')
fruits.append('grapes')
fruits.append('mango')
fruits.append('orange')

# Now let's dequeue our fruits, we should get 'banana'
first_item = fruits.pop(0)
print(first_item)
#banana 

# If we dequeue again we'll get 'grapes'
second_item = fruits.pop(0)
print(second_item)
#grapes

# 'mango' and 'orange' remain
print(fruits) 
#['mango', 'orange']
{% endhighlight %}

We can create a class for this data structure in Python:

{% highlight python %} 
#A queue that only has enqueue and dequeue operations
class Queue():

    def __init__(self):
        self.queue = []
        self.length = 0
        
    def __str__(self):
        return str(self.__dict__)

    def enqueue(self, item):
        self.length +=1
        self.queue.append(item)

    def dequeue(self):
        if len(self.queue) < 1:
            return None
        self.length -=1
        return self.queue.pop(0)

    def size(self):
        return len(self.queue) 
    
    #Peek: Get the top most element of the queue. i.e, the element at the front position.
    def peek(self):
        return self.queue[0]
    
queue_example = Queue()

# Let's enqueue some fruits into our list
queue_example.enqueue('banana')
queue_example.enqueue('grapes')
queue_example.enqueue('mango')
queue_example.enqueue('orange')

print(queue_example)
#{'queue': ['banana', 'grapes', 'mango', 'orange'], 'length': 4}

# Now let's dequeue our fruits, we should get 'banana'
first_item = queue_example.dequeue()
print(first_item)
#banana

print(queue_example)
#{'queue': ['grapes', 'mango', 'orange'], 'length': 3}

# If we dequeue again we'll get 'grapes'
second_item = queue_example.dequeue()
print(second_item)
#grapes

print(queue_example)
#{'queue': ['mango', 'orange'], 'length': 2}

# 'mango' and 'orange' remain
queue_example.size()
#2

print(queue_example.peek())
#mango
{% endhighlight %}

Queue has O(n) lookup, O(1) enqueue, O(1) dequeue, and O(1) peek.

**Additional remarks...**

Python has a `deque` (pronounced 'deck') library that provides a sequence with efficient methods to work as a stack or a queue. `deque` is short for Double Ended Queue - a generalized queue that can get the first or last element that's stored. It has two ends, a front and a rear. What makes a deque different is the unrestrictive nature of adding and removing items. New items can be added at either the front or the rear. Likewise, existing items can be removed from either end. In a sense, this hybrid linear structure provides all the capabilities of stacks and queues in a single data structure.

{% highlight python %} 
from collections import deque

# you can initialize a deque with a list 
numbers = deque()

# Use append like before to add elements
numbers.append(99)
numbers.append(15)
numbers.append(82)
numbers.append(50)
numbers.append(47)

# You can pop like a stack
last_item = numbers.pop()
print(last_item) # 47
print(numbers) # deque([99, 15, 82, 50])

# You can dequeue like a queue
first_item = numbers.popleft()
print(first_item) # 99
print(numbers) # deque([15, 82, 50])
{% endhighlight %}

It can also be implemented easily from scratch:

{% highlight python %} 
class Deque():
    
    def __init__(self):
        self.items = []
        
    def add_front(self, item):
        return self.items.append(item)
    
    def add_read(self, item):
        self.items.insert(0, item)
        
    def remove_front(self):
        return self.items.pop()
    
    def remove_rear(self):
        return self.items.pop(0)
    
    def size(self):
        return len(self.items)
{% endhighlight %}


##### Let's now look at the big-O notations of Stacks and Queues:

* **Access**: O(n) for Stacks; O(n) for Queues.
* **Search***: O(n) for Stacks; O(n) for Queues.
* **Insert**: O(1) for Stacks since we insert the element at the end; O(1) for Queues since we insert an element at the "rear" of the queue.
* **Delete**: O(1) for Stacks since we delete the element at the end; O(1) for Queues since we delete the first added element (at the "front" of the queue).

##### Comparing it with priority queues
Stacks and queues may be modeled as particular kinds of priority queues. As a reminder, here is how stacks and queues behave:

* stack – elements are pulled in last-in first-out-order (e.g., a stack of papers)
* queue – elements are pulled in first-in first-out-order (e.g., a line in a cafeteria)

In a stack, the priority of each inserted element is monotonically increasing; thus, the last element inserted is always the first retrieved. In a queue, the priority of each inserted element is monotonically decreasing; thus, the first element inserted is always the first retrieved.


#### What is the idea of a hashtable?

If the keys were integers, you could implement a Map using an array of values: given key $k$, store the associated value in the array cell with index $k$. Then adding and looking up entries would be $O(1)$, since you now have random access based on the array index.   

But of course, the keys are not always integers.  The idea of a hashtable is to try to make a lookup table for arbitrary key/value pairs that "acts" as though you had random access into an array.  The ideas are as follows:

1. From each key, compute an integer, called its hash code, and then use the integer as an index into an array of values. 
2. Take the hash code modulo the array size to get an index into an array. (You want the array to be bounded in size ‐ ideally it would be not much bigger than the total number of entries).
3. Store the key and the value in a linked list of entries at the array index.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Screen%20Shot%202020-02-10%20at%2010.50.16.png?raw=true)

The point of item (3) is that it is possible that two different keys end up with the same hash code, or that
two different hash codes result in the same array index.  That means you’ll try to store two values at the same array index.  This is called a collision. Irrespective of how good a hash function is, collisions are bound to occur. Therefore, to maintain the performance of a hash table, it is important to manage collisions through various collision resolution techniques. One of the solutions is based on the idea of putting the keys that collide in a linked list. Using a linked list of entries allows you to store multiple values at the same array index.  In this case, a hastable then is an array of lists. This tecnique is called a *separate chaining collision* resolution. Traditionally these lists are called buckets.  

#### Explain Class, Object (Instance), Instance Attribute, Class Attribute, Instance Method with an example.

While the class is the blueprint, an instance is a copy of the class with actual values, literally an object belonging to a specific class.

All classes create objects, and all objects contain characteristics called attributes (referred to as properties in the opening paragraph). Use the `__init__()` method to initialize (e.g., specify) an object’s initial attributes by giving them their default value (or state). This method must have at least one argument as well as the self variable, which refers to the object itself.

While instance attributes are specific to each object, class attributes are the same for all instances (objects).

{% highlight python %}
class Robot():
    types = "Electronic" #Class attribute
    def __init__(self, name, color, weight):
        self.name = name #Instance attribute
        self.color = color #Instance attribute
        self.weight = weight #Instance attribute
        
    def IntroduceYourself(self): #Instance Method
        print("My name is " + self.name)
        
    def MakeSound(self, sound): #Instance Method with additional parameter
        print("My name is " + self.name + sound)
        
r1 = Robot(name = "Tom", color = "red", weight = 30) # An object (instance)
#<__main__.Robot at 0x10a1d3048>

r2 = Robot(name = "Jerry", color = "blue", weight = 40) # Another object (instance)
#<__main__.Robot at 0x10a1d38d0>

r1.IntroduceYourself()
# My name is Tom

r2.IntroduceYourself()
# My name is Jerry

r1.MakeSound(' YAAAYYY!')
#My name is Tom YAAAYYY!
r2.MakeSound(' HURRRAAAY!')
#My name is Jerry HURRRAAAY!
{% endhighlight %}

#### How to create a JSON file? How to load a JSON file?

JSON (JavaScript Object Notation) is a popular data format used for representing structured data.

The built-in `json` package of Python has the magic code that transforms your Python dict object in to the serialized JSON string.

{% highlight python %}
import json

data = {}
data['people'] = []
data['people'].append({
    'name': 'Scott',
    'website': 'stackabuse.com',
    'from': 'Nebraska'
})
data['people'].append({
    'name': 'Larry',
    'website': 'google.com',
    'from': 'Michigan'
})
data['people'].append({
    'name': 'Tim',
    'website': 'apple.com',
    'from': 'Alabama'
})

data 
# {'people': [{'name': 'Scott', 'website': 'stackabuse.com', 'from': 'Nebraska'},
#             {'name': 'Larry', 'website': 'google.com', 'from': 'Michigan'},
#             {'name': 'Tim', 'website': 'apple.com', 'from': 'Alabama'}]}

with open('data.txt', 'w') as outfile:
    json.dump(data, outfile)
    
with open('data.txt') as json_file:
    data = json.load(json_file)
    for p in data['people']:
        print('Name: ' + p['name'])
        print('Website: ' + p['website'])
        print('From: ' + p['from'])
        print('')
        
# Name: Scott
# Website: stackabuse.com
# From: Nebraska

# Name: Larry
# Website: google.com
# From: Michigan

# Name: Tim
# Website: apple.com
# From: Alabama
{% endhighlight %}

To analyze and debug JSON data, we may need to print it in a more readable format. This can be done by passing additional parameters `indent` and `sort_keys` to `json.dumps()` and `json.dump()` method.

#### What is the time complexity for Binary Search Tree?

Keep in mind that unless you're writing your own data structure (e.g. linked list in C), it can depend dramatically on the implementation of data structures in your language/framework of choice.

Binary Search Trees (BST) has different time complexities for best and worst cases. In the best cases, access, search, insert and deletion operations have O(log n) time where n is the number of nodes in the balanced binary search tree. If you do $n$ searches in the binary tree, hence the total complexity is O(nlog(n)).

However, for all of these traversals (preorder, inorder, postorder, and level order) - whether done recursively or iteratively - you'll have to visit every node in the binary tree. That means that you’ll get a runtime complexity of 𝑂(𝑛) - where $n$ is the number of nodes in the binary tree.

![](https://i.stack.imgur.com/SulR5.png)

However, the time complexity for these operations is O(n) in the worst case when the tree becomes unbalanced as in given above. Because in the absolute worst case, a binary tree with $n$ elements would be like a linked list and this is why we need to balance the trees to achieve O(log N) search. Besides, similarly, if you are doing $n$ searches in the unbalanced tree, the total complexity will turn out to be O(n^2).

#### Where does the log of O(log n) come from?

You repeatedly divide n by two until you reach 1 or less. Therefore your stop condition is (roughly) expressed by the equation n/2^k = 1 <=> n = 2^k <=> log_2 n = k ( / is an integer division in your code). Roughness is allowed because we are dealing with O( ) and constants will disappear anyway.

https://stackoverflow.com/questions/9152890/what-would-cause-an-algorithm-to-have-olog-n-complexity/9153420#9153420

#### What is Big Data?

Big data analytics can be time-consuming, complicated, and computationally demanding, without the proper tools, frameworks, and techniques. When the volume of data is too high to process and analyze on a single machine, Apache Spark and Apache Hadoop can simplify the task through parallel processing and distributed processing. To understand the need for parallel processing and distributed processing in big data analytics, it is important to first understand what “big data” is. The high-velocity at which big data is generated requires that the data also be processed very quickly and the variety of big data means it contains various types of data, including structured, semi-structured, and unstructured data. The volume, velocity, and variety of big data calls for new, innovative techniques and frameworks for collecting, storing, and processing the data, which is why Apache Hadoop and Apache Spark were created.


# SOME NOTES

- Basically, `pip` comes with python itself.Therefore it carries no meaning for using pip itself to install or upgrade python. `pip` is designed to upgrade python packages and not to upgrade python itself. `pip` shouldn't try to upgrade python when you ask it to do so. Don't type `pip install python` but use an installer instead.

- `python3 --version` will print out the version of Python3 on Terminal window. `which -a python python3` will print out all the Python versions you have.

- `pip` is associated with Python 2.7 and `pip3` is with Python 3. So if you want to upgrade `pip3`, try running `pip3 install --upgrade pip`.

- You can list all the installed packages with pip either using `pip list` or `pip3 list`. 

- How to know which Python is running in Jupyter notebook? 
 ``` python
 import sys
print(sys.executable)
print(sys.version)
print(sys.version_info)
```
will give you the interpreter. You can select the interpreter you want when you create a new notebook.

- You can just specify the python version when running a program: 
  for python 2: 
  `python filename.py` 
  
  and 
  
  for python 3:
  `python3 filename.py`
  
  Note that this works if you do not add an alias to python3 (see below)
  
- In order to switch Python versions in Terminal, the simplest way would be to add an alias to python3 to always point to the native python installed. Add this line to the `.bash_profile` file in your `$HOME` directory at the last and source `.bash_profile` by doing `source ~/.bash_profile`. Doing so makes the changes to be reflected on every interactive shell opened.

- Some Python installations come with Apple MacOSX. The version of Python that ships with OS X is great for learning, but it’s not good for development. The version shipped with OS X may be out of date from the official current Python release, which is considered the stable production version.

  Items in `/usr/bin` should always be or link to files supplied by Apple in OS X, unless someone has been ill-advisedly changing things there. 

  ```shell
Arat-MacBook-Pro:~ mustafamuratarat$ /usr/bin/python

WARNING: Python 2.7 is not recommended. 
This version is included in macOS for compatibility with legacy software. 
Future versions of macOS will not include Python 2.7. 
Instead, it is recommended that you transition to using 'python3' from within Terminal.

Python 2.7.16 (default, Dec 21 2020, 23:00:36) 
[GCC Apple LLVM 12.0.0 (clang-1200.0.30.4) [+internal-os, ptrauth-isa=sign+stri on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> quit()
Arat-MacBook-Pro:~ mustafamuratarat$ /usr/bin/python2

WARNING: Python 2.7 is not recommended. 
This version is included in macOS for compatibility with legacy software. 
Future versions of macOS will not include Python 2.7. 
Instead, it is recommended that you transition to using 'python3' from within Terminal.

Python 2.7.16 (default, Dec 21 2020, 23:00:36) 
[GCC Apple LLVM 12.0.0 (clang-1200.0.30.4) [+internal-os, ptrauth-isa=sign+stri on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> quit()
Arat-MacBook-Pro:~ mustafamuratarat$ /usr/bin/python2.7

WARNING: Python 2.7 is not recommended. 
This version is included in macOS for compatibility with legacy software. 
Future versions of macOS will not include Python 2.7. 
Instead, it is recommended that you transition to using 'python3' from within Terminal.

Python 2.7.16 (default, Dec 21 2020, 23:00:36) 
[GCC Apple LLVM 12.0.0 (clang-1200.0.30.4) [+internal-os, ptrauth-isa=sign+stri on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> quit()
Arat-MacBook-Pro:~ mustafamuratarat$ /usr/bin/python3
Python 3.7.3 (default, Mar  6 2020, 22:34:30) 
[Clang 11.0.3 (clang-1103.0.32.29)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> quit()
  ```
- Each pixel typically consists of 8 bits (1 byte) for a Black and White (B&W) image or 24 bits (3 bytes) for a color image-- one byte each for Red, Green, and Blue. 8 bits represents 28 = 256 tonal levels (0-255).
