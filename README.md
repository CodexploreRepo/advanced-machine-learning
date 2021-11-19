# Advanced Machine Learning

# Table of contents
- [Table of contents](#table-of-contents)
- [1. Introduction](#1-introduction)
  - [1.1. Why study Machine Learning](#11-why-study-machine-learning)
  - [1.2. Niches for Machine Learning](#12-niches-for-machine-learning)
  - [1.3. Machine Learning Concepts](#13-machine-learning-concepts)


# 1. Introduction
## 1.1. Why study Machine Learning
- Automating automation
- Getting computers to program themselves
- Writing software is the bottleneck
- Let the data do the work instead!

## 1.2. Niches for Machine Learning
- **Data Mining**: use historical data to improve decisions
  - e.g. Medical records &#8594; Medical knowledge
    - *Given*: 9147 patient records, each describing pregnancy and birth; Each patient contains 215 features
    - *Task*: Classes of future patients at high risk for *Emergency Cesarean Section* (Sinh Mổ)
    - *Learned rules*: (one out of 18 rules) If no previous vaginal delivery abnormal 2nd Trimester Ultrasound Malpresentation at admission  &#8594;  Then probability of Emergency C-Section is 0.6
  - e.g. Credit Risk Analysis
    - *Learned rules*: 
      - If Other-Delinquent-Account > 2 Number-Delinquent-Billing-Cycles > 1 &#8594; Profitable-Customer ? = no
      - If Other-Delinquent-Account = 0 (Income > $30K or Years-of-Credit > 3) &#8594; Then Profitable-Customer ? = yes
- **Software Applications** that are hard to program by hand: Speech recognition, Image classification, Autonomous driving
  - e.g. ALVINN: an autonomous land vehicle in a neural network (1989); Google’s self-driving car; Image Tagging/Captioning
- **User Modeling**: Automatic Recommender Systems

## 1.3. Machine Learning Concepts

<p align="center"><img width="350" alt="Screenshot 2021-09-08 at 22 32 46" src="https://user-images.githubusercontent.com/64508435/142563823-c9718200-2490-493b-92a9-5a0ea89a9d9c.png"></p>

### 1.3.1. Designing a Learning System
- Choose the *training experience* &#8594; Choose exactly what is to be learned, i.e. the `target function` &#8594; Choose how to represent the target function &#8594; Choose a *learning algorithm* to infer the target function from the experience
### 1.3.2. Machine Learning Algorithm
- **Every machine learning algorithm** has three components: 
  - *Representation*:
    - Decision trees
    - Sets of rules / Logic programs
    - Instances
    - Graphical models (Bayes/Markov nets)
    - Neural networks
    - Support vector machines
    - Model ensembles
  - *Evaluation*:
    - Accuracy
    - Precision and recall
    - Squared error
    - Likelihood
    - Posterior probability
    - Cost / Utility
    - Margin
    - Entropy
    - K-L divergence
  - *Optimization*:
    - Combinatorial optimization
      - E.g.: Greedy search
    - Convex optimization
      - E.g.: Gradient descent
    - Constrained optimization
      - E.g.: Linear programming
### 1.3.3. Types of Machine Learning
- **Supervised (inductive) learning**: Training data includes desired outputs
  - *Classification*: discrete output
    - Binary Classification: input x, find y in {-1, +1}
    - Multi-class classification: input x, find y in {1, ..., k}
      - E.g.: Digit Recognition: Map each image x to one of ten digits `[0,...,9]` 
  - *Regression*: continuous output
    - Given input x, find y in real-valued space R (R^d)
    - **Linear Regression**: assume linear dependence
    - **Non-linear Regression**: Time series forecasting
- **Unsupervised learning**: Training data does not include desired outputs - Given (input, ~~correct output~~), (input, ?)
  - *Clustering*: Find a set of prototypes representing the data
    - E.g.: Marketing segmentation, group of insurance interests, web news, pictures, city-planning 
  - *Dimension Reduction / Principal Components*: Find a subspace representing the data
    - E.g.: Principal Component Analysis (PCA), Non-Linear Embedding
  - *Independent components / dictionary learning*: Find (small) set of factors for observation
  - *Novelty/Anomaly detection*: Identification of new or unknown patterns
    - Parametric approach
    - Non-parametric approach
- **Semi-supervised learning**: Training data includes a few desired outputs
- **Reinforcement learning**: Take an action, environment responds, take new action
  - E.g.: Game playing, Self-driving cars, Autonomous plane flight 

[(Back to top)](#table-of-contents)
