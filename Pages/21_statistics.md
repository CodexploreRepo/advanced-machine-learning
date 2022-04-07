# Statistics

# Table of contents
- [Table of contents](#table-of-contents)
- [1. Probability vs Likelihood](#1-probability-vs-likelihood)
  - [1.1. Example for Probability vs Likelihood](#11-example-for-probability-vs-likelihood)
  - [1.2. PDF vs Likelihood Function](#12-pdf-vs-likelihood-function)
  - [1.3. Maximum Likelihood Estimate (MLE)](#13-maximum-likelihood-estimate)
  - [1.4. Maximum A Posteriori (MAP) Estimation](#14-maximum-a-posteriori-estimation)

# 1. Probability vs Likelihood
- The terms “probability” and “likelihood” are often used interchangeably in the English language, but they have very different meanings in statistics.
- Given a statistical model with some parameters θ,
  - **Probability** is used to describe how plausible a future outcome x is (knowing the parameter values θ)
  - **Likelihood** is used to describe how plausible a particular set of parameter values θ are, after the outcome x is known.

## 1.1. Example for Probability vs Likelihood
- Consider a 1D mixture model of two Gaussian distributions centered (mean) at **–4** and **+1**. For simplicity, this toy model has a single parameter `θ` that controls the *standard deviations* of both distributions
- The *top-left&* contour plot in below Figure shows the entire model `f(x; θ)` as a function of both `x` and `θ`.
  - To estimate the probability distribution of a future outcome x, you need to set the model parameter θ.
  - For example, if you **set θ to 1.3** (the horizontal line), you get the probability density function `f(x; θ=1.3)` shown in the lower-left plot.
- The *lower-left* plot shows the probability density function (PDF) `f(x; θ=1.3)` 
  - Say you want to estimate the probability that x will fall between –2 and +2. 
  - You must calculate the integral of the PDF on this range (i.e., the shaded region). 
- Question: What if you don’t know θ, and instead if you have observed a single instance x=2.5 (the vertical line in the upper-left plot) ?
- The *upper-right* plot: In this case, we get the likelihood function `ℒ(θ|x=2.5)=f(x=2.5; θ)`, represented in the upper-right plot.

<p align="center">
<img src="https://user-images.githubusercontent.com/64508435/162111908-6441eec1-67d1-4685-a386-bb01a693e17d.png" width="600" /><br>A model’s parametric function (top left), a PDF with θ=1.3  (lower left),<br> a likelihood function (top right) when single instance x=2.5 , and a log likelihood function (lower right)
</p>

## 1.2. PDF vs Likelihood Function
- **Probability Density Function (PDF)** is a function of x (with θ fixed)
  - if you integrate a probability distribution over all possible values of x, you always get 1.
- **Likelihood Function** is a function of θ (with x fixed). 
  - It is important to understand that the *likelihood function is not a probability distribution*: if you integrate the likelihood function over all possible values of θ, the result can be any positive value.
 
 ## 1.3. Maximum Likelihood Estimate
- Given a dataset X, a common task is to try to estimate the most likely values for the model parameters. 
- To do this, you must find the values that maximize the likelihood function, given X. 
  - In the above example, if you have observed a single instance x=2.5, the maximum likelihood estimate (MLE) of θ=-2.15 when θ=1.5.
- Notice that maximizing the likelihood function is equivalent to maximizing its **logarithm**
  - Indeed the logarithm is a strictly increasing function, so if θ maximizes the log likelihood, it also maximizes the likelihood. It turns out that it is generally easier to maximize the log likelihood.
  
## 1.4. Maximum A Posteriori Estimation
- Maximum a-Posteriori (MAP) estimation: If a prior probability distribution **g** over θ exists, it is possible to take it into account by maximizing `ℒ(θ|x)g(θ)` rather than just maximizing `ℒ(θ|x)`. 
- Since MAP constrains the parameter values, you can think of it as a regularized version of MLE.

[(Back to top)](#table-of-contents)
