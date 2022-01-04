# 1. Ensemble Method

# Table of contents
- [Table of contents](#table-of-contents)
- [1. Ensemble Method Introduction](#1-ensemble-method-introduction)
- [2. Random Forests](#2-random-forests)
- [3. Gradient Boosting](#3-gradient-boosting)
  - [3.1. XGBoost](#31-xgboost) 


# 1. Ensemble Method Introduction
- The group (or “ensemble”) will often perform better than the best individual model (just like Random Forests perform better than the individual Decision Trees they rely on), especially if the individual models make very different types of errors. (more on [Chapter 7](https://www.oreilly.com/library/view/hands-on-machine-learning/9781491962282/ch07.html#ensembles_chapter))
- The goal of `ensemble methods` is to combine the predictions of several base estimators built with a given learning algorithm in order to improve generalizability / robustness over a single estimator (**for classification, regression and anomaly detection**)
- Two families of ensemble methods:
  - In **averaging methods**, the driving principle is to build several estimators independently and then to average their predictions. On average, the combined estimator is usually better than any of the single base estimator because its variance is reduced.
    - *Examples*: Bagging methods, Forests of randomized trees, etc.
  - In **boosting methods**, base estimators are built sequentially and one tries to reduce the bias of the combined estimator. The motivation is to combine several weak models to produce a powerful ensemble.
    - *Examples*: AdaBoost, Gradient Tree Boosting, etc.

# 2. Random Forests
- Random forest method, which achieves better performance than a single decision tree simply by averaging the predictions of many decision trees. We refer to the random forest method as an "ensemble method". By definition, ensemble methods combine the predictions of several models (e.g., several trees, in the case of random forests).
- Decision trees leave you with a difficult decision. 
  - A deep tree with lots of leaves will overfit because each prediction is coming from historical data from only the few data at its leaf. 
  - But a shallow tree with few leaves will perform poorly because it fails to capture as many distinctions in the raw data.
- The random forest uses many trees, and it makes a prediction by averaging the predictions of each component tree. 
- It generally has much better predictive accuracy than a single decision tree and it works well with default parameters.

# 3. Gradient Boosting
- **Gradient boosting** is the method dominates many Kaggle competitions and achieves state-of-the-art results on a variety of datasets.
- Gradient boosting is a method that goes through cycles to iteratively add models into an ensemble.
- It begins by initializing the ensemble with a single model, whose predictions can be pretty naive. (Even if its predictions are wildly inaccurate, subsequent additions to the ensemble will address those errors.)
- *"Gradient"* in "gradient boosting" refers to the fact that we'll use `gradient descent` on the loss function to determine the parameters in this new model.)
![image](https://user-images.githubusercontent.com/64508435/144753278-7e6573ec-4a2e-45e9-aa2b-b713529366d0.png)


```Python
#Example of Gradient Boosting - Regressor
from sklearn.ensemble import GradientBoostingRegressor

gbm_model = GradientBoostingRegressor(random_state=1, n_estimators=500)
gbm_model.fit(train_X, train_y)
gbm_val_predictions = gbm_model.predict(val_X)
gbm_val_rmse = np.sqrt(mean_squared_error(gbm_val_predictions, val_y))
```

## 3.1. XGBoost 
- `XGBoost` stands for **extreme gradient boosting**, which is an implementation of gradient boosting with several additional features focused on performance and speed.
- XGBoost has a few parameters that can dramatically affect accuracy and training speed:
  - `n_estimators`: (typically range from **100-1000**, though this depends a lot on the learning_rate parameter) specifies how many times to go through the modeling cycle described above. It is equal to the number of models that we include in the ensemble.
    - Too low a value causes *underfitting*, which leads to inaccurate predictions on both training data and test data.
    - Too high a value causes *overfitting*, which causes accurate predictions on training data, but inaccurate predictions on test data (which is what we care about).
  - `learning_rate`: (default: learning_rate=0.1) Instead of getting predictions by simply adding up the predictions from each component model, we can multiply the predictions from each model by a small number (known as the **learning rate**) before adding them in.
  - `n_jobs`: equal to the number of cores on your machine
    - On smaller datasets, the resulting model won't be any better, so micro-optimizing for fitting time is typically nothing but a distraction.
    - On larger datasets where runtime is a consideration, you can use parallelism to build your models faster; otherwise spend a long time waiting during the `fit` command.
```Python
from xgboost import XGBRegressor

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4, random_state = 42)

my_model.fit(X_train, y_train, 
             #Setting early_stopping_rounds=5 is a reasonable choice. In this case, we stop after 5 straight rounds of deteriorating validation scores
             early_stopping_rounds=5, 
             #When using early_stopping_rounds, you also need to set aside some data for calculating the validation scores - this is done by setting the eval_set parameter.
             eval_set=[(X_valid, y_valid)], 
             verbose=False)
```
  - `early_stopping_rounds`: offers a way to automatically find the ideal value for n_estimators. Early stopping causes the model to stop iterating when the validation score stops improving, even if we aren't at the hard stop for n_estimators. It's smart to set a high value for n_estimators and then use early_stopping_rounds to find the optimal time to stop iterating.
    - Setting **early_stopping_rounds=5** is a reasonable choice. In this case, we stop after 5 straight rounds of deteriorating validation scores.





[(Back to top)](#table-of-contents)
