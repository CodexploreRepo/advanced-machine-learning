# Linear Regression

# Table of contents
- [Table of contents](#table-of-contents)
- [1. Statistics in Linear Model](#1-statistics-in-linear-model)
  - [1.1. Sum of Squares](#11-sum-of-squares)
  - [1.2. R-squared](#12-r-squared)
  - [1.3. Explained Variance](#13-explained-variance)
  - [1.4. Mean Absolute Error](#14-mean-absolute-error)
- [2. Linear Regression](#2-linear-regression)
  - [2.1. Gradient Descent](#21-gradient-descent)
    - [2.1.1. Batch GD](#211-batch-gd) 
- [3. Polynomial Regression](#3-polynomial-regression)
- [4. Bias-Variance Tradeoff](#4-bias-variance-tradeoff)
- [5. Regularized Linear Model](#5-regularized-linear-model)
  - [5.1. Ridge Regression](#51-ridge-regression) 
  - [5.2. Lasso Regression](#52-lasso-regression)
  - [5.3. Elastic Net](#53-elastic-net)

# 1. Statistics in Linear Model
## 1.1. Sum of Squares
- **Sum of squares** (SS) is a statistical tool that is used to identify the dispersion of data as well as how well the data can fit the model in regression analysis. 
- The general rule is that a smaller sum of squares indicates a better model, as there is less variation in the data.
- **3 main types of sum of squares**: 
  - `SST` total sum of squares: Variance that already exists in the data
  - `SSR` regression sum of squares: Try to explain the variance by using the regression model. (i.e: How far away, your predicted values from the mean)
  - `SSE` residual sum of squares: The un-explain part of the model. How far away the predicted value from the actual one.

<p align="center">
  <img width="650" src="https://user-images.githubusercontent.com/64508435/149864125-e42aa3ec-7f6b-4664-9803-3ec271b17ac0.png" />
</p>

- Proof for total sum of squares
<p align="center">
  <img width="650" src="https://user-images.githubusercontent.com/64508435/149864335-07fdf58c-50d6-482d-88f8-5d7fa1748552.png" />
</p>

## 1.2. R-squared
- R-squared is a statistical measure that represents the goodness of fit of a regression model. The ideal value for r-square is 1. The closer the value of r-square to 1, the better is the model fitted.
- `R-square = SSR/SST = 1 - SSE/SST`: how you use the model to explain the data

<p align="center">
  <img width="400" src="https://user-images.githubusercontent.com/64508435/149864618-6676f14d-806d-4ed1-a038-16626dc4db30.png" />
  <img width="400" src="https://user-images.githubusercontent.com/64508435/149864722-53a82063-0f36-4793-9a3c-103c0c95add8.png" />
</p>

<p align="center">
<img width="658" alt="Screenshot 2022-01-18 at 12 43 11" src="https://user-images.githubusercontent.com/64508435/149872313-f5ed987e-5a64-48a6-975b-0fd5cc9082ed.png"></p>

## 1.3. Explained Variance
-  `Var[y-y_pred]` and `Var[y]` are variance of prediction errors and actual values respectively. 

<p align="center">
<img width="400" alt="Screenshot 2022-01-18 at 12 43 11" src="https://user-images.githubusercontent.com/64508435/149874565-cca7c0ed-1313-4799-b241-5eabd49dc0bd.png"></p>


## 1.4. Mean Absolute Error
- Easy to understand the model error via MAE as they are on the same scale.

[(Back to top)](#table-of-contents)

# 2. Linear Regression
- Both the `Normal Equation` and Scikit-Learn’s LinearRegression class (which is based on `Singular Value Decomposition (SVD) approach`) get **very slow when the number of features (`n`) grows large** (e.g. 100,000). 
- On the positive side, both are linear with regard to the number of instances in the training set (they are O(m)), 
    - so they **handle large training sets efficiently (`m`)**, provided they can fit in memory.
```Python
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
lin_reg.intercept_, lin_reg.coef_
```
- To evaluate the performance of Linear Regression
```Python
from sklearn import metrics
y_pred = lin_reg.predict(x_test)
print(metrics.explained_variance_score(y_test, y_pred))
print(metrics.mean_absolute_error(y_test, y_pred))
print(metrics.mean_squared_error(y_test, y_pred))
```

## 2.1. Gradient Descent
### Gradient Descent introduction
- `Objective Function` of Linear Regression is **MSE**, which is also called as Cost Function, so our goal (objective) is to minimize the object function. 
  - Apart from analytical approach (i.e: `Normal Equation` and Scikit-Learn’s LinearRegression class (which is based on `Singular Value Decomposition (SVD)`), we also can consider Iterative approach which is Gradient Descent.

<p align="center">
<img width="850" alt="Screenshot 2022-01-20 at 16 48 48" src="https://user-images.githubusercontent.com/64508435/151113192-26b12f5e-db1a-4258-88f2-23eb20daeb19.png"></p>


- **Gradient Descent** is to measures the local gradient of the cost function with regard to the parameter vector θ, and it goes in the direction of descending gradient. Once the gradient is zero, you have reached a minimum!
-  There are 3 types of GD: `Batch GD`, `Stochastic GD`, `Mini-Batch GD`

### Convex Function
- **Convex Function**: *Implies that all local minima are global minima*. It means that, no matter where you start, if the cost function is a convex function, you will end up at the same point, which is the global optima.

<p align="center">
<img width="500" alt="Screenshot 2022-01-20 at 16 48 48" src="https://user-images.githubusercontent.com/64508435/151113847-3c28ddde-370e-47fd-8b00-344e28550266.jpeg">
<img width="350" alt="Screenshot 2022-01-20 at 16 48 48" src="https://user-images.githubusercontent.com/64508435/151114298-44a963c5-3f21-4548-9ed6-498c96776ad2.jpeg"></p>

### 2.1.1. Batch GD


# 3. Polynomial Regression
- What if your data is more complex than a straight line? Surprisingly, you can use a linear model to fit nonlinear data. 
- A simple way to do this is to add powers of each feature as new features, then train a linear model on this extended set of features. 
- This technique is called **Polynomial Regression**.
- Usually, we will set degree up to 4 as to reduce `overfitting`. This is also known as to `regularize` the model (i.e., **to constrain it**)
  - The fewer degrees of freedom it has, the harder it will be for it to overfit the data. A simple way to regularize a polynomial model is to reduce the number of polynomial degrees.
```Python
poly2 = preprocessing.PolynomialFeatures(degree = 2, include_bias = False)
x2 = poly2.fit_transform(x)

x2_train, x2_test, y2_train, y2_test = model_selection.train_test_split(x2, y, test_size = 0.2, random_state = 2022)

regr2 = linear_model.LinearRegression()
regr2.fit(x2_train, y2_train)

print('R^2 score: %.6f' % regr2.score(x2_test, y2_test))
print(regr2.coef_)
print(poly2.powers_)

#R^2 score: 0.994592
#[  0.          69.26585136  93.05262999 -23.7832548  -15.10744814
# -12.51933868]
#[[0 0] => A^0 * B^0 = 1
#[1 0]  => A^1 * B^0 = A
#[0 1]  => A^0 * B^1 = B
#[2 0]
#[1 1]
#[0 2]]
```

[(Back to top)](#table-of-contents)

# 4. Bias-Variance Tradeoff
- An important theoretical result of statistics and Machine Learning is the fact that a model’s generalization error can be expressed as the sum of three very different errors:
### Bias
- This part of the generalization error is due to wrong assumptions, such as assuming that the data is linear when it is actually quadratic. A high-bias model is most likely to underfit the training data. 
### Variance
- This part is due to the model’s excessive sensitivity to small variations in the training data. A model with many degrees of freedom (such as a high-degree polynomial model) is likely to have high variance and thus overfit the training data.
### Irreducible error
- This part is due to the noisiness of the data itself. The only way to reduce this part of the error is to clean up the data (e.g., fix the data sources, such as broken sensors, or detect and remove outliers).
- Increasing a model’s complexity will typically increase its variance and reduce its bias. Conversely, reducing a model’s complexity increases its bias and reduces its variance. This is why it is called a trade-off.

# 5. Regularized Linear Model
- In order to address overfitting, we can use **Regularization**
  - Incorporate model complexity for optimization, penalize complex models using prior knowledge
  - Keep all the features, but reduce magnitude/values of model parameters
  - Works well when we have a lot of features, each of which contributes a bit to the prediction
- For a linear model, regularization is typically achieved by constraining the weights of the model.

<p align="center">
<img width="598" alt="Screenshot 2022-01-20 at 16 48 48" src="https://user-images.githubusercontent.com/64508435/150304872-a520602e-73a5-4f80-b865-a561607566de.png"></p>

### When to use plain Linear Regression (i.e., without any regularization), Ridge, Lasso, or Elastic Net?

- A model with some regularization typically performs better than a model without any regularization, so you should generally prefer Ridge Regression over plain Linear Regression.
- Lasso Regression uses an `ℓ1` penalty, which tends to push the weights down to exactly zero. This leads to sparse models, where all weights are zero except for the most important weights. This is a way to perform feature selection automatically, which is good if you suspect that only a few features actually matter. When you are not sure, you should prefer Ridge Regression.
- Elastic Net is generally preferred over Lasso since Lasso may behave erratically in some cases (when several features are strongly correlated or when there are more features than training instances). However, it does add an extra hyperparameter to tune. If you want Lasso without the erratic behavior, you can just use Elastic Net with an `l1_ratio` close to 1.

## 5.1. Ridge Regression
- *Ridge Regression* (also called Tikhonov regularization) is a regularized version of Linear Regression: a `regularization term` is added to the cost function.
- The hyperparameter `α` controls how much you want to regularize the model. 
    - If `α = 0`, then Ridge Regression is just Linear Regression. If α is very large, then all weights end up very close to zero and the result is a flat line going through the data’s mean. 

```Python
for alpha in [1, 0.1, 0.01, 0.001]:
    ridge = linear_model.Ridge(alpha = alpha)
    ridge.fit(x_train, y_train)
    print('score:', ridge.score(x_test, y_test))
    print(ridge.coef_)
```

## 5.2. Lasso Regression
- **Least Absolute Shrinkage and Selection Operator** (Lasso) Regression (usually simply called Lasso Regression) is another regularized version of Linear Regression
- An important characteristic of Lasso Regression is that it tends to eliminate the weights of the least important features (i.e., set them to zero). 
- Lasso Regression **automatically performs feature selection and outputs a sparse model** (i.e., with few non-zero feature weights).
```Python
from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
lasso_reg.predict([[1.5]])
```
## 5.3. Elastic Net
The regularization term is a simple mix of both Ridge and Lasso’s regularization terms, and you can control the mix ratio r. 
- When `r = 0`, Elastic Net is equivalent to Ridge Regression
- When `r = 1`, it is equivalent to Lasso Regression
```Python
from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5) #l1_ratio corresponds to the mix ratio r
elastic_net.fit(X, y)
elastic_net.predict([[1.5]])
```
