# 3. Bayes Learning

# Table of contents
- [Table of contents](#table-of-contents)
- [1. Introduction](#1-introduction.md)
- [2. Maximum-Likelihood Estimation (MLE)](#2-maximum-likelihood-estimation)
  - [2.1. MLE Examples](#21-mle-examples) 
  - [2.2. Maximum Likelihood Estimation Definition](#22-maximum-likelihood-estimation-definition)
- [3. Bayes Theorem](#3-bayes-theorem)
- [4. Maximum A Posterior (MAP)](#4-maximum-a-posterior)
  - [4.1. Maximum A Posterior (MAP) Definition](#41-maximum-a-posterior-definition)
  - [4.2. MLE vs MAP](#42-mle-vs-map)
- [5. Probabilistic Generative Models](#5-probabilistic-generative-models)
- [6. Naive Bayes Classifier](#6-naive-bayes-classifier)
  - [6.1. Limitation of Naive Bayes](#61-limitation-of-naive-bayes)
  - [6.2. Gaussian Naive Bayes](#62-gaussian-naive-bayes) 
  - [6.3. Multinomial Naive Bayes](#63-multinomial-naive-bayes)


# 1. Introduction 

[(Back to top)](#table-of-contents)

# 2. Maximum-Likelihood Estimation
- There are many methods for estimating unknown parameters from data. We will first consider the `maximum likelihood estimate (MLE)`, which answers the question:
  - For which parameter value does the observed data have the biggest probability?
## 2.1. MLE Examples
- We will explain the MLE through a series of examples.
<p align="center">
  <img width="650" src="https://user-images.githubusercontent.com/64508435/149057110-a375832d-230d-48fd-84b6-eb84e521e204.png" />
</p>

### 2.1.1. Log likelihood
- If is often easier to work with the **natural log of the likelihood function**. For short this is simply called the `log likelihood`. 
  - Since ln(x) is an increasing function, the maxima of the likelihood and log likelihood coincide.

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

# 3. Bayes Theorem
- Bayes: When we receive the new data D, we can update our belief/hypothesis `P(h|D)` provided that everything on the RHS is known.

<p align="center">
    <img width="800" src="https://user-images.githubusercontent.com/64508435/149085193-619e7e50-b550-403a-adcc-9a5ab6e9622a.png" />
</p>

- `P(h|D)`: Posterior - probability of h After seeing D
- `P(h)`: Prior - probability of h Before seeing D
  - For example: after observing the data, we know that the hypothesis `h (µ, σ)` has `µ` must only `<1`, so it will affect `P(h|D)`.

[(Back to top)](#table-of-contents)

# 4. Maximum A Posterior 
## 4.1. Maximum A Posterior Definition
- Unlike MLE, when we already have prior idea about how the hypothesis will happen, we will use Maximum A Posterior (MAP).
- Comparing the equation of `MAP` with `MLE`, we can see that the only difference is that MAP **includes prior in the formula**
  - which means that the **likelihood** `P(D|h)` is weighted by the **prior** `P(h)` in MAP, where `P(h)` is choosen according to our prior knowledge about the learning task
-  *Step 1*: For each hypothesis h in H, calculate  `posterior` probability
  -  the purpose is to find **argmax(h)** of `P(h|D) = P(D|h) x P(h) / P(D)` and `P(D)` is not dependent of h. 
  -  Hence argmax(h) of P(h|D) is the same as argmax of P(D|h) x P(h).
<p align="center">
    <img width="400" alt="Screenshot 2022-01-12 at 16 07 37" src="https://user-images.githubusercontent.com/64508435/149088143-e9f789a3-d863-4204-b3e6-0fb6735d3a97.png">
</p>

-  *Step 2*: Output the hypothesis h with the highest posterior probability
<p align="center">
<img width="400" alt="Screenshot 2022-01-12 at 16 09 10" src="https://user-images.githubusercontent.com/64508435/149088364-e09e9f37-63e7-4ca9-ac62-07dcfbc38dfb.png">
</p>

## 4.2. MLE vs MAP
- The difference between MLE and MAP is that MAP is used with consideration of the prior knowledge / preference while MLE is used without that.
<p align="center">
<img width="600" alt="Screenshot 2022-01-12 at 16 09 10" src="https://user-images.githubusercontent.com/64508435/149089867-5bc31065-691e-4b81-b255-47f1719fdb0b.jpeg">
</p>

- **Note**: For uniform prior, p(h) is the same for all possible h. 
  - Hence the maximum value of p(h|D) = p(D|h) x p(h) only depends on p(D|h) as p(h) is the same for all h(s). 
  - Therefore, MLE can concide with MAP

[(Back to top)](#table-of-contents)

# 5. Probabilistic Generative Models

<p align="center">
<img width="600" alt="Screenshot 2022-01-12 at 16 09 10" src="https://user-images.githubusercontent.com/64508435/150098908-fe03ee3f-1ca1-47e4-b2d9-873eb9c1395b.jpeg">
<img width="600" alt="Screenshot 2022-01-12 at 16 09 10" src="https://user-images.githubusercontent.com/64508435/150098837-b7bb9062-3e91-41bd-8093-beafba150de8.jpeg">
<img width="600" alt="Screenshot 2022-01-12 at 16 09 10" src="https://user-images.githubusercontent.com/64508435/150099338-296b0a9a-c25d-4053-ab72-78a3ae30fa1c.jpeg">
</p>

# 6. Naive Bayes Classifier
- A Naive Bayes classifier is a probabilistic generative model with the “naive” assumption of **conditional independence between every pair of features** given the value of the class variable. 
- **Practical insight**: Nearly all probabilistic models are “inaccurate”, but many are, nonetheless, *useful when trained with sufficient data*.

<p align="center">
<img width="600" alt="Screenshot 2022-01-12 at 16 09 10" src="https://user-images.githubusercontent.com/64508435/150103107-680a295b-1163-47a8-8259-59c7a4ba0575.jpeg">
</p>

- Example of Naive Bayes Classifier:

<p align="center">
<img width="600" alt="Screenshot 2022-01-12 at 16 09 10" src="https://user-images.githubusercontent.com/64508435/150103938-0dc211a2-cdcd-4bf2-a759-59950b1335e3.jpeg">
<img width="600" alt="Screenshot 2022-01-12 at 16 09 10" src="https://user-images.githubusercontent.com/64508435/150103975-671eb804-287a-4885-8a53-692a0b74d805.jpeg">
</p>

### Types of Naive Bayes Classifier
<p align="center">
<img width="600" alt="Screenshot 2022-01-12 at 16 09 10" src="https://user-images.githubusercontent.com/64508435/150463871-4cc1c377-c9c8-4358-9f36-0e64cde50d7a.png">
</p>


# 6.1. Limitation of Naive Bayes
- Although NB makes computation possible & Yields optimal classifiers when satisfied with fairly good empirical results
- Naive Bayes is seldom satisfied in practice, as attributes (variables) are often correlated
- Solution to overcome this limitation: `Bayesian networks`, that combine Bayesian reasoning with causal relationships between attributes

# 6.2. Gaussian Naive Bayes
- When the predictors take up a **continuous value** and are not discrete, we assume that these values are sampled from a gaussian distribution. For example: Iris Flower.

<p align="center">
<img width="600" alt="Screenshot 2022-01-12 at 16 09 10" src="https://user-images.githubusercontent.com/64508435/150101627-e4d5932b-64c2-4f7f-8cf7-23c2b3a29514.jpeg">
</p>

## 6.3. Multinomial Naive Bayes
- There are 2 common distribution for discrete values: **Bi-nomial**: throw a coin (Head and Tail), **Multinomial**: throw a dice (1,2..6)
- Multinomial Naive Bayes is used for **discrete values** of the input, mostly for document classification problem, i.e whether a document belongs to the category of sports, politics, technology etc. The features/predictors used by the classifier are the frequency of the words present in the document.

<p align="center">
<img width="600" alt="Screenshot 2022-01-12 at 16 09 10" src="https://user-images.githubusercontent.com/64508435/150105901-4b715341-6500-4b2f-8910-3c24df681c10.jpeg">
</p>

- **Parameter Estimation:**
<p align="center">
<img width="600" alt="Screenshot 2022-01-12 at 16 09 10" src="https://user-images.githubusercontent.com/64508435/150106377-907e4364-ae2a-4a90-a5bf-28b0e517c3b5.jpeg">
</p>

- In order to avoid zero probability, we add the smoothing factor `alpha = 1`

<p align="center">
<img width="600" alt="Screenshot 2022-01-12 at 16 09 10" src="https://user-images.githubusercontent.com/64508435/150106155-8173d555-384c-4f43-8b5e-ad1b879d8204.jpeg">
<img width="600" alt="Screenshot 2022-01-12 at 16 09 10" src="https://user-images.githubusercontent.com/64508435/150106159-7ecb3fbb-4261-49e5-8dca-b2d0992b9188.jpeg">
</p>



[(Back to top)](#table-of-contents)
