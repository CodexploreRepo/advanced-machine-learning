# 3. Bayes Learning

# Table of contents
- [Table of contents](#table-of-contents)
- [1. Introduction](#1-introduction.md)
- [2. Maximum-Likelihood Estimation (MLE)](#2-maximum-likelihood-estimation)


# 1. Introduction 

# 2. Maximum-Likelihood Estimation
## 2.1. MLE Introducion
- **Likelihood**: probability of observing sample under distribution d , which, given the independence assumption is
<p align="center">
  <img width="400" src="https://user-images.githubusercontent.com/64508435/148676323-c241763f-236d-4150-8b8f-31160b2ef983.png" />
</p>

- **Maximum Likelihood Estimation (MLE)** is a method that determines values for the parameters of a model. 
  - Select a distribution maximizing the sample probability
  - The parameter values are found such that they maximise the likelihood that the process described by the model produced the data that were actually observed.
  <p align="center">
    <img width="400" src="https://user-images.githubusercontent.com/64508435/148676422-f0feafc1-3d30-4973-b6e5-7706de817986.png" />
  </p>

- For example, Let’s suppose we have observed 10 data points from some process. For example, each data point could represent the length of time in seconds that it takes a student to answer a specific exam question. These 10 data points are shown in the figure below
  - For these data we’ll assume that the data generation process can be adequately described by a Gaussian (normal) distribution
  - **Gaussian distribution has 2 parameters**. The mean, `μ`, and the standard deviation, `σ`.
  - Different values of these parameters result in different curves. 
  - We want to know which curve was most likely responsible for creating the data points that we observed? (See figure below). 
  - *Maximum likelihood estimation is a method that will find the values of μ and σ that result in the curve that best fits the data*.

<p align="center">
  <img width="500" src="https://user-images.githubusercontent.com/64508435/148675089-44e7be47-4898-4979-9460-bea8ecd51de0.png" />
</p>

## 2.2. MLE Calculation
- [Reference](https://towardsdatascience.com/probability-concepts-explained-maximum-likelihood-estimation-c7b4342fdbb1) Suppose we have three data points this time and we assume that they have been generated from a process that is adequately described by a Gaussian distribution. These points are 9, 9.5 and 11. 
- How do we calculate the maximum likelihood estimates of the parameter values of the Gaussian distribution μ and σ?
- The probability density of observing a single data point x, that is generated from a Gaussian distribution is given by:
  - The semi colon used in the notation P(x; μ, σ) is there to emphasise that the symbols that appear after it are parameters of the probability distribution.
<p align="center">
  <img width="400" src="https://user-images.githubusercontent.com/64508435/148675856-7e9e6174-2290-4e4c-9faa-4546f509c7dc.png" />
</p>
  
  - In our example the total (joint) probability density of observing the three data points is given by

<p align="center">
  <img width="600" src="https://user-images.githubusercontent.com/64508435/148676091-ed45a5fa-d528-48b1-bf88-e119abe2a8af.png" />
</p>
  
  - A technique that can help us find maxima (and minima) of functions. It’s called differentiation.

### 2.2.1. Log Likelihood
- The above expression for the total probability is actually quite a pain to differentiate, so it is almost always simplified by taking the natural logarithm of the expression.
- That is fine because natural logarithm is a monotonically increasing function. 
  - This means that if the value on the x-axis increases, the value on the y-axis also increases 
<p align="center">
  <img width="600" src="https://user-images.githubusercontent.com/64508435/148676559-089c868a-68e7-4446-b391-79d785eb8198.png" />
</p>

- This expression can be differentiated to find the maximum. In this example we’ll find the MLE of the mean, μ. To do this we take the partial derivative of the function with respect to μ, giving
<p align="center">
  <img width="400" src="https://user-images.githubusercontent.com/64508435/148676684-028d2ec9-5eb2-4d30-8227-32e9fb374a0b.png" />
</p>

- Set the LHS = 0, so μ = 9.833



[(Back to top)](#table-of-contents)
