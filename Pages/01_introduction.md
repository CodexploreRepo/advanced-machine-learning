# Introduction

# Table of contents
- [Table of contents](#table-of-contents)
- [1.1. Machine Learning Introduction](#11-machine-learning-introduction)
- [1.2. Niches for Machine Learning](#12-niches-for-machine-learning)
- [1.3. Machine Learning Concepts](#13-machine-learning-concepts)
  - [1.3.1. Designing a Learning System](#131-designing-a-learning-system)
  - [1.3.2. Machine Learning Algorithm](#132-machine-learning-algorithm)
  - [1.3.3. Types of Machine Learning](#133-types-of-machine-learning)
  - [1.3.4. Batch and Online Learning](#134-batch-and-online Learning)
  - [1.3.5. Important Issues in Machine Learning](#135-important-issues-in-machine-learning)
- [1.4. Deep Learning](#14-deep-learning)

## 1.1. Machine Learning Introduction
### 1.1.1. What is Machine Learning
- Machine Learning is the field of study that gives computers the ability to learn without being explicitly programmed. (Arthur Samuel, 1959)

### 1.1.2. Why study Machine Learning
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
#### 1.3.1.1. Evaluation of Learning System
- **Experimental**
  - Conduct controlled cross-validation experiments to compare various methods on a variety of benchmark datasets.
  - Gather data on their performance, e.g. test accuracy, training- time, testing-time.
  - Analyze differences for statistical significance.
- **Theoretical**: Analyze algorithms mathematically and prove theorems about their
  - Computational complexity
  - Ability to fit training data
  - Sample complexity (number of training examples needed to learn an accurate function)

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
  - In supervised learning, the training set you feed to the algorithm includes the desired solutions, called labels 
  - *Classification*: discrete output
    - Binary Classification: input x, find y in {-1, +1}
    - Multi-class classification: input x, find y in {1, ..., k}
      - E.g.: Digit Recognition: Map each image x to one of ten digits `[0,...,9]` 
  - *Regression*: continuous output
    - Given input x, find y in real-valued space R (R^d)
    - **Linear Regression**: assume linear dependence
    - **Non-linear Regression**: Time series forecasting
    - Note: some regression algorithms can be used for classification as well, and vice versa. For example, Logistic Regression is commonly used for classification, as it can output a value that corresponds to the probability of belonging to a given class (e.g., 20% chance of being spam).
  - Most important supervised learning algorithms:
    - k-Nearest Neighbors
    - Linear Regression
    - Logistic Regression
    - Support Vector Machines (SVMs)
    - Decision Trees and Random Forests
    - Neural networks
- **Unsupervised learning**: Training data does not include desired outputs - (i.e:  the training data is unlabeled)
  - *Clustering*: Find a set of prototypes representing the data
    - K-Means
    - DBSCAN
    - Hierarchical Cluster Analysis (HCA)
    - E.g.: Marketing segmentation, group of insurance interests, web news, pictures, city-planning 
  - *Dimension Reduction / Principal Components*: Find a subspace representing the data
    - Principal Component Analysis (PCA)
    - Kernel PCA
    - Locally Linear Embedding (LLE)
    - t-Distributed Stochastic Neighbor Embedding (t-SNE)
    - E.g.: Principal Component Analysis (PCA), Non-Linear Embedding, simplify the data without losing too much information. One way to do this is to merge several correlated features into one. For example, a car’s mileage may be strongly correlated with its age, so the dimensionality reduction algorithm will merge them into one feature that represents the car’s wear and tear.
    - **Tips**:  good idea to try to reduce the dimension of your training data using a dimensionality reduction algorithm before you feed it to another Machine Learning algorithm (such as a supervised learning algorithm). It will run much faster, the data will take up less disk and memory space, and in some cases it may also perform better
  - *Independent components / dictionary learning*: Find (small) set of factors for observation
  - *Novelty/Anomaly detection*: Identification of new or unknown patterns
    - One-class SVM
    - Isolation Forest
    - E.g.: detecting unusual credit card transactions to prevent fraud, catching manufacturing defects, or automatically removing outliers from a dataset before feeding it to another learning algorithm. 
- **Semi-supervised learning**: 
  - Since labeling data is usually time-consuming and costly, you will often have plenty of unlabeled instances, and few labeled instances. Some algorithms can deal with data that’s partially labeled. This is called `semi-supervised learning`
    - E.g.: Some photo-hosting services, such as Google Photos, are good examples of this. Once you upload all your family photos to the service, it automatically recognizes that the same person A shows up in photos 1, 5, and 11, while another person B shows up in photos 2, 5, and 7. This is the unsupervised part of the algorithm (clustering). Now all the system needs is for you to tell it who these people are. Just add one label per person4 and it is able to name everyone in every photo, which is useful for searching photos.
  - Most semisupervised learning algorithms are combinations of unsupervised and supervised algorithms. 
- **Reinforcement learning**: Take an action, environment responds, take new action
  - Reinforcement Learning is a very different beast. The learning system, called an `agent` in this context, can observe the environment, select and perform actions, and get rewards in return (or penalties in the form of negative rewards, as shown in Figure 1-12). 
  - It must then learn by itself what is the **best strategy**, called a `policy`, to get the most reward over time. 
  - A `policy` defines what action the agent should choose when it is in a given situation.
    - E.g.: Game playing, Self-driving cars, Autonomous plane flight 
    - Many robots implement Reinforcement Learning algorithms to learn how to walk. DeepMind’s AlphaGo program is also a good example of Reinforcement Learning: it made the headlines in May 2017 when it beat the world champion Ke Jie at the game of Go. It learned its winning policy by analyzing millions of games, and then playing many games against itself. Note that learning was turned off during the games against the champion; AlphaGo was just applying the policy it had learned.
<p align="center"><img width="450" alt="Screenshot 2021-09-08 at 22 32 46" src="https://user-images.githubusercontent.com/64508435/145361790-008cd94d-96d2-400d-b3f9-021ed2f29e72.png"></p>

### 1.3.4. Batch and Online Learning
- Another criterion used to classify Machine Learning systems is whether or not the system can learn incrementally from a stream of incoming data.
#### Batch Learning
- In `batch learning`, the system is incapable of learning incrementally: it must be trained using all the available data.
- This will generally take a lot of time and computing resources, so it is typically done offline. 
- First the system is trained, and then it is launched into production and runs without learning anymore; it just applies what it has learned. 
- This is called **offline learning**.

##### Drawback of Batch Learning
- If you want a batch learning system to know about new data (such as a new type of spam), you need to train a new version of the system from scratch on the full dataset (not just the new data, but also the old data), then stop the old system and replace it with the new one.

### 1.3.5. Important Issues in Machine Learning
- **Obtaining experience**
  - How to obtain experience? Supervised learning vs. Unsupervised learning
  - How many examples are enough? PAC learning theory
- **Learning algorithms**
  - What algorithm can approximate function well, when?
  - How does the complexity of learning algorithms impact the learning accuracy?
  - Whether the target function is learnable?
- **Representing inputs**
  - How to represent the inputs?
  - How to remove the irrelevant information from the input representation?
  - How to reduce the redundancy of the input representation?


## 1.4. Deep Learning
### 1.4.1. Shallow Learning vs Deep Learning
<p align="center"><img width="450" alt="Screenshot 2021-09-08 at 22 32 46" src="https://user-images.githubusercontent.com/64508435/142584149-bc990035-e2e6-4dab-80a0-82119ecc1e99.png"></p>

### 1.4.2. Why Deep Learning ?
- A family of machine learning algorithms based on multi-layer networks
- Inspired by the biological architecture of brain in neuroscience
<p align="center"><img width="600" alt="Screenshot 2021-11-19 at 15 43 24" src="https://user-images.githubusercontent.com/64508435/142585189-0b477b90-5159-4152-bef1-3a478a5ce70c.png"></p>


[(Back to top)](#table-of-contents)
