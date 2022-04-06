# Gaussian Mixture Model (GMM) 

# Table of contents
- [Table of contents](#table-of-contents)
- [1. Introduction](#1-introduction)
- [2. Expectation-Maximization Process](#2-expectation-maximization-process)
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

# 2. Expectation-Maximization Process



# Resources


[(Back to top)](#table-of-contents)
