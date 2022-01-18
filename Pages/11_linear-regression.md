# Linear Regression

# Table of contents
- [Table of contents](#table-of-contents)
- [1. Statistics in Linear Model](#1-statistics-in-linear-model)
  - [1.1. Sum of Squares](#11-sum-of-squares)
  - [1.2. R-squared](#12-r-squared)
  - [1.3. Explained Variance](#13-explained-variance)

# 1. Statistics in Linear Model
## 1.1. Sum of Squares
- **Sum of squares** (SS) is a statistical tool that is used to identify the dispersion of data as well as how well the data can fit the model in regression analysis. 
- The general rule is that a smaller sum of squares indicates a better model, as there is less variation in the data.
- **3 main types of sum of squares**: total sum of squares, regression sum of squares, and residual sum of squares.

<p align="center">
  <img width="650" src="https://user-images.githubusercontent.com/64508435/149864125-e42aa3ec-7f6b-4664-9803-3ec271b17ac0.png" />
</p>

- Proof for total sum of squares
<p align="center">
  <img width="650" src="https://user-images.githubusercontent.com/64508435/149864335-07fdf58c-50d6-482d-88f8-5d7fa1748552.png" />
</p>

## 1.2. R-squared
- R-squared is a statistical measure that represents the goodness of fit of a regression model. The ideal value for r-square is 1. The closer the value of r-square to 1, the better is the model fitted.

<p align="center">
  <img width="400" src="https://user-images.githubusercontent.com/64508435/149864618-6676f14d-806d-4ed1-a038-16626dc4db30.png" />
  <img width="400" src="https://user-images.githubusercontent.com/64508435/149864722-53a82063-0f36-4793-9a3c-103c0c95add8.png" />
</p>

<p align="center">
<img width="658" alt="Screenshot 2022-01-18 at 12 43 11" src="https://user-images.githubusercontent.com/64508435/149872313-f5ed987e-5a64-48a6-975b-0fd5cc9082ed.png"></p>

## 1.3. Explained Variance
-  `Var[y-y_pred]` and `Var[y]` are variance of prediction errors and actual values respectively. 

<p align="center">
<img width="658" alt="Screenshot 2022-01-18 at 12 43 11" src="https://user-images.githubusercontent.com/64508435/149874565-cca7c0ed-1313-4799-b241-5eabd49dc0bd.png"></p>


[(Back to top)](#table-of-contents)
