# Gaussian Mixture Model (GMM) 

# Table of contents
- [Table of contents](#table-of-contents)
- [1. Introduction](#1-introduction)
- [2. GMM Example](#2-gmm-example)
- [3. Expectation-Maximization Process](#3-expectation-maximization-process)
  - [3.1. Maximum Log-Likelihood ](#31-maximum-log-likelihood) 
  - [3.2. EM Algorithm for Clustering](#32-em-algorithm-for-clustering)
- [Resources](#resources)

# 1. Introduction
- Assume that data are generated from a mixture of Gaussian distributions.
  - For each Gaussian Distribution (Cluster) j: Center `μ`, Variance `Σ`
  - For each data point, determine membership i: how much belongs data point `i` belongs to the `j`th cluster
- **Gaussian mixture model (GMM)** is a probabilistic model that assumes that the instances were generated from a mixture of several Gaussian distributions whose parameters (`μ`, `Σ`) are unknown
  - All the instances generated from a single Gaussian distribution form a cluster that typically looks like an *ellipsoid* (Figure 1). 
  - Each cluster can have a different **ellipsoidal shape, size, density, and orientation**
<p align="center">
<img src="https://user-images.githubusercontent.com/64508435/162021995-f2bfaff0-f879-4ca1-9c99-2e038b4f56ef.png" width="400" /><br>Figure 1: ellipsoidal clusters
</p>

- There are several GMM variants. In the simplest variant, implemented in the GaussianMixture class, you **must know in advance the number k** of Gaussian distributions.
## 1.1. Hard vs Soft Clustering
<p align="center">
<img src="https://user-images.githubusercontent.com/64508435/162028394-59838624-418b-47dc-879f-70ed206d4242.png" width="400" />
</p>



[(Back to top)](#table-of-contents)

# 2. GMM Example
- The Gaussian mixture model (GMM) is classically trained by an optimization procedure named the *Expectation-Maximization* (or EM in short)
- For example: in iris classification problem, we have 150 Iris flowers divided between 3 classes. For each of them, we have the sepal length and width, the petal length and width, and the class.
  - The Gaussian Mixture model tries to describe the data as if it originated from a mixture of Gaussian distributions. 
  - **GMM in 1-Dimension**, say the petal width and try to fit 3 different Gaussians, we would end up with something like this:
<p align="center">
<img src="https://user-images.githubusercontent.com/64508435/162023276-b2bdc6ff-8b50-463d-9906-478a60ffd8ed.png" width="400" />
</p>

  - The algorithm found that the mixture that is most likely to represent the data generation process is made of the three following normal distributions:
    - The setosa petal widths are much more concentrated with a mean of 0.26 and a variance of 0.04. The other two classes are comparably more spread out but with different locations. 
<p align="center">
<img src="https://user-images.githubusercontent.com/64508435/162023597-371ba52e-c3a2-430a-91f9-12e780944b4d.png" width="800" />
</p>

  - **GMM in 2-Dimension**: say petal width and petal length and now the constituents of the mixture are the following:
<p align="center">
<img src="https://user-images.githubusercontent.com/64508435/162024102-8afd53a9-4691-4050-8b76-9f8d709aed46.png" width="400" />
<img src="https://user-images.githubusercontent.com/64508435/162024215-108a7365-e2d7-4473-b1b7-0ff9368ada08.png" width="800" />
</p>
[(Back to top)](#table-of-contents)

# 3. Expectation-Maximization Process
## 3.1. Maximum Log-likelihood 
- So how does the algorithm finds the best set of parameters to describe the mixture? Well, we start by defining the probability model. 
- The probability of observing any observation, that is the **probability density**, is a weighted sum of K Gaussian distributions.
<p align="center">
<img src="https://user-images.githubusercontent.com/64508435/162025066-803573df-69f0-41a6-b02a-e1ef1da1d71b.png" width="240" />
</p>

- In the case of 3 clusters, the set of parameters will be as follows and then we want to find parameter values that maximize the likelihood of the dataset.  
<p align="center">
<img src="https://user-images.githubusercontent.com/64508435/162025524-892ab830-29c1-4095-92c9-936d3d518a95.png" width="350" /><br>
<img src="https://user-images.githubusercontent.com/64508435/162025576-b898ea7f-99d1-4a81-b360-a671a3a0c26e.png" width="550" />
</p>
  
  - where `k`: number of clusters and `i` are all observing data points 
- Instead of the likelihood, we usually maximize the log-likelihood, in part because it turns the product of probabilities into a sum (simpler to work with).
<p align="center">
<img src="https://user-images.githubusercontent.com/64508435/162026136-33088aec-60b1-454e-a8a0-2c110df07b51.png" width="600" />
</p>

## 3.2. EM Algorithm for Clustering
- Expectation-Maximization (EM) algorithm, which has many similarities with the K-Means algorithm: it also initializes the cluster parameters randomly, &#8594; it repeats two steps until convergence, first assigning instances to clusters (this is called the expectation step) &#8594; updating the clusters (this is called the maximization step).
- Unlike K-Means, though, EM uses *soft cluster assignments*, not hard assignments. 
  - For each instance, during the expectation step, the algorithm estimates the probability that it belongs to each cluster (based on the current cluster parameters). 
  - Then, during the maximization step, each cluster is updated using **all the instances in the dataset**, with each instance weighted by the estimated probability that it belongs to that cluster. 
  - These probabilities are called the *responsibilities of the clusters* for the instances.
  - During the maximization step, each cluster’s update will mostly be impacted by the instances it is most responsible for.
```
Repeat
  - E-Step: Estimate membership of each data points 
  - M-Step: Estimate the cluster centers (and prior) 
Until convergence
```

<p align="center">
<img src="https://miro.medium.com/max/1210/1*RIHS9QCl8p-NEmUwxFaK-A.gif" width="500" />
</p>

### 3.2.1. E-Step
- E-Step (Expectation step): to estimate membership
  - Initially, we can randomly assign the cluster for each membership
  - The probability of the observation belonging to the cluster `E(Zij) = p(Kj|xi)`, would be the ratio between the Gaussian value and the sum of all the Gaussians (Z). 
<p align="center">
<img src="https://user-images.githubusercontent.com/64508435/162027354-a37aa048-3f85-478c-b570-eaf2996dff04.png" width="600" />
</p>

### 3.2.2. M-Step
- M-Step (Maximization step): Estimate cluster parameters
<p align="center">
<img src="https://user-images.githubusercontent.com/64508435/162029156-6a6cfecb-3d8d-4d80-9138-5244e8f12914.png" width="600" />
</p>



# Resources


[(Back to top)](#table-of-contents)
