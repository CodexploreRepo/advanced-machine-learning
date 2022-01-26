# Logistic Regression

# Table of contents
- [Table of contents](#table-of-contents)
- [1. Introduction](#1-introduction)
  - [1.1. Odds](#11-odds) 
  - [1.2. Logistic Regression](#12-logistic-regression)
  - [1.3. Maximum Likelihood for Logistic Regression](#13-maximum-likelihood-for-logistic-regression)
- [2. Logistic Regression Implementation](#12-logistic-regression-implementation)

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

## 1.3. Maximum Likelihood for Logistic Regression
- Unlike Linear regression where the **objective function** is MSE, here we will try to Maximise the Log Likelihood (MLE) of the training data
<p align="center">
<img width="600" alt="Screenshot 2022-01-18 at 12 43 11" src="https://user-images.githubusercontent.com/64508435/151122463-13f98403-18ad-48b9-8e19-e264a5a786f7.jpeg">
</p>
<p align="center">
<img width="450" alt="Screenshot 2022-01-18 at 12 43 11" src="https://user-images.githubusercontent.com/64508435/151123217-74c0b726-9d68-45b4-81bb-2733275359e5.jpeg">
<img width="450" alt="Screenshot 2022-01-18 at 12 43 11" src="https://user-images.githubusercontent.com/64508435/151123199-b135765a-d039-45dd-9bf9-79fe3c9378bb.jpeg">
</p>

- Since there is no close-form (analytical) solution for the above objective function, so we will use iterative approach which is Gradient Descent
<p align="center">
<img width="600" alt="Screenshot 2022-01-18 at 12 43 11" src="https://user-images.githubusercontent.com/64508435/151123912-46f09e5f-72e9-44b1-a9da-0b89ece6a129.jpeg">
</p>

# 2. Logistic Regression Implementation
- The hyperparameter controlling the regularization strength of a Scikit-Learn `LogisticRegression` model is **not alpha** (as in other linear models), but its inverse: C. The higher the value of `C`, the less the model is regularized.

```Python
# build an estimator with smaller C
log_reg = linear_model.LogisticRegression(solver = 'lbfgs', C = 0.001) #C = [0.0001, 0.01, 0.1, 1]
log_reg.fit(x_train, y_train)

# Look at model’s estimated probabilities
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)
plt.plot(X_new, y_proba[:, 1], "g-", label="Iris virginica")
plt.plot(X_new, y_proba[:, 0], "b--", label="Not Iris virginica")
# + more Matplotlib code to make the image look pretty
```

[(Back to top)](#table-of-contents)
