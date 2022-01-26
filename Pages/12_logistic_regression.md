# Logistic Regression

# Table of contents
- [Table of contents](#table-of-contents)
- [1. Introduction](#1-introduction)
  - [1.1. Odds](#11-odds) 
  - [1.2. Logistic Regression](#12-logistic-regression)
  - [1.3. Objective Function for Logistic Regression](#13-objective-function-for-logistic-regression)


# 1. Introduction
- Can we use regression model for classification ? YES
- Logistic Regression (also called Logit Regression) is commonly used to estimate the probability that an instance belongs to a particular class (e.g., what is the probability that this email is spam?). If the estimated probability is greater than 50%, then the model predicts that the instance belongs to that class (called the positive class, labeled “1”), and otherwise it predicts that it does not (i.e., it belongs to the negative class, labeled “0”). This makes it a binary classifier.
## 1.1 Odds
- We defined the Odds below formula and then **model** it to exponential function as
  - exp function has the range also from 0 to infinity
  - y_pred = `wT*x` large, exp(y_pred) becomes very large => P(1|x) ~ 1, which is belong to Class "1"
  - y_pred = `wT*x` small or negative, exp(y_pred) becomes very small => P(-1|x) ~ 1, which is belong to Class "0"

<p align="center">
<img width="600" alt="Screenshot 2022-01-18 at 12 43 11" src="https://user-images.githubusercontent.com/64508435/151121195-bdae3517-397b-4761-aaaf-3855476259c7.jpeg"></p>

## 1.2. Logistic Regression
<p align="center">
<img width="600" alt="Screenshot 2022-01-18 at 12 43 11" src="https://user-images.githubusercontent.com/64508435/151122148-918f21d9-2a19-46cd-a949-f1cbe494e8f6.jpeg"></p>

## 1.3. Objective Function for Logistic Regression
- Unlike Linear regression where the **objective function** is MSE, here we will try to Maximise the Log Likelihood (MLE) of the training data
<p align="center">
<img width="600" alt="Screenshot 2022-01-18 at 12 43 11" src="https://user-images.githubusercontent.com/64508435/151122463-13f98403-18ad-48b9-8e19-e264a5a786f7.jpeg">
</p>


[(Back to top)](#table-of-contents)
