# 3. Bayes Learning

# Table of contents
- [Table of contents](#table-of-contents)
- [1. Introduction](#1-introduction.md)
- [2. Maximum-Likelihood Estimation (MLE)](#2-maximum-likelihood-estimation)
  - [2.1. MLE Examples](#21-mle-examples) 
  - [2.2. Maximum Likelihood Estimation Definition](#22-maximum-likelihood-estimation-definition)


# 1. Introduction 

# 2. Maximum-Likelihood Estimation
- There are many methods for estimating unknown parameters from data. We will first consider the `maximum likelihood estimate (MLE)`, which answers the question:
  - For which parameter value does the observed data have the biggest probability?
## 2.1. MLE Examples
- We will explain the MLE through a series of examples.
<p align="center">
  <img width="650" src="https://user-images.githubusercontent.com/64508435/149057110-a375832d-230d-48fd-84b6-eb84e521e204.png" />
</p>

### 2.1.1. Log likelihood
- If is often easier to work with the **natural log of the likelihood function**. For short this is simply called the `log likelihood`. Since ln(x) is an increasing function, the maxima of the likelihood and log likelihood coincide.

<p align="center">
  <img width="650" src="https://user-images.githubusercontent.com/64508435/149059337-aed113bb-bf17-4772-9e5a-ecca737ac13e.png" />
</p>

- The following example illustrates how we can use the method of maximum likelihood to estimate multiple parameters at once.
<p align="center">
  <img width="700" src="https://user-images.githubusercontent.com/64508435/149060302-cb376d90-37d6-43ff-bc03-0164d233b19c.png" />
  <img width="700" src="https://user-images.githubusercontent.com/64508435/149060584-9f8dfb72-d6d8-4637-ae07-649cf41aaea7.png" />
</p>

### 2.2. Maximum Likelihood Estimation Definition
- **Likelihood** `P(data | p)`; where p are parameters of a distribution/hypothesis `h`
- Given data, the maximum likelihood estimate (MLE) for the parameter p is the value of p that maximizes **likelihood** `P(data|p)`. 
  - i.e: MLE is the value of p for which the data is most likely.
- **In general**, given training data D, MLE is to find the best hypothesis h that maximizes the likelihood of the training data
<p align="center">
    <img width="400" src="https://user-images.githubusercontent.com/64508435/149061051-1d8d85e3-18ab-494c-9ecd-54c0f1a355c7.png" />
</p>

- If  H = all possible 1-dimensional Normal (Gaussian) distributions (like example 4 above), 
<p align="center">
    <img width="400" src="https://user-images.githubusercontent.com/64508435/149061429-c0dc9599-079a-4acc-a5e0-5965c9235b96.png" />
</p>


[(Back to top)](#table-of-contents)
