# Model Fine-Tuning

Let’s assume that you now have a shortlist of promising models. You now need to fine-tune them.

# Table of contents
- [Table of contents](#table-of-contents)
- [1. Grid Search](#1-grid-search)
- [2. Randomized Search](#2-randomized-search)
- [3. Analyze the Best Models and Their Errors](#3-analyze-the-best-models-and-their-errors)
- [4. Evaluate Your System on the Test Set](#4-evaluate-your-system-on-the-test-set)
- [5. Launch, Monitor, and Maintain Your System](#5-launch-monitor-and-maintain-your-system)




# 1. Grid Search
- One way to do that would be to fiddle with the hyperparameters manually, until you find a great combination of hyperparameter
- Scikit-Learn’s `GridSearchCV` to search which hyperparameters you want it to experiment with, and what values to try out 
- It will evaluate all the possible combinations of hyperparameter values, using cross-validation

```Python
from sklearn.model_selection import GridSearchCV

param_grid = [
    #first evaluate all 3 × 4 = 12 combinations of n_estimators and max_features hyperparameter values specified in the first dict
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    #all 2 × 3 = 6 combinations of hyperparameter values in the second dict
    #this time with the bootstrap hyperparameter set to False instead of True (which is the default)
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor()

#All in all, the grid search will explore 12 + 6 = 18 combinations of RandomForestRegressor hyperparameter values
#Train each model five times (since we are using five-fold cross validation)
#All in all, there will be 18 × 5 = 90 rounds of training
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)

grid_search.fit(housing_prepared, housing_labels)
```
-  Tip: Since 8 and 30 are the maximum values that were evaluated, you should probably try searching again with higher values, since the score may continue to improve.
```Python
#get the best combination of parameters
grid_search.best_params_
```
- If `GridSearchCV` is initialized with `refit=True` (which is the default), then once it finds the best estimator using cross-validation, it retrains it on the whole training set. This is usually a good idea since feeding it more data will likely improve its performance.

```Python
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

#50036.32733962357 {'max_features': 8, 'n_estimators': 30}
#61747.39782442657 {'bootstrap': False, 'max_features': 2, 'n_estimators': 3}
```

# 2. Randomized Search
- When the hyperparameter search space is large, it is often preferable to use `RandomizedSearchCV` instead. 
- Instead of trying out all possible combinations like `GridSearchCV`, it evaluates a given number of random combinations by selecting a random value for each hyperparameter at every iteration. 
- This approach has two main benefits:
  - If you let the randomized search run for, say, 1,000 iterations, this approach will explore 1,000 different values for each hyperparameter (instead of just a few values per hyperparameter with the grid search approach).
  - You have more control over the computing budget you want to allocate to hyperparameter search, simply by setting the number of iterations.

```Python
from sklearn.model_selection import RandomizedSearchCV

# Setup random seed
np.random.seed(42)

param_grid = [
    #first evaluate all 3 × 4 = 12 combinations of n_estimators and max_features hyperparameter values specified in the first dict
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    #all 2 × 3 = 6 combinations of hyperparameter values in the second dict
    #this time with the bootstrap hyperparameter set to False instead of True (which is the default)
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

# Setup random hyperparameter search for RandomForestClassifier
randomized_search = RandomizedSearchCV(forest_reg, 
                           param_distributions=param_grid,
                           cv=5,
                           n_iter=10,
                           scoring='neg_mean_squared_error',
                           return_train_score=True,
                                      verbose=True)

# Fit random hyperparameter search model for RandomForestClassifier()
randomized_search.fit(housing_prepared, housing_labels)
```
- To get the best combination of parameters
```Python
#get the best combination of parameters
randomized_search.best_params_
#{'n_estimators': 30, 'max_features': 8}

#get the score for each combination
cvres = randomized_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

#65029.23239964716 {'n_estimators': 3, 'max_features': 2}
#55289.066755389285 {'n_estimators': 10, 'max_features': 2}
```

# 3. Analyze the Best Models and Their Errors
- You will gain good insights on the problem by inspecting the best models. 
    - For example, the `RandomForestRegressor` can indicate the relative importance of each attribute for making accurate predictions
```Python
#To get the feature importances of the best estimator
feature_importances = randomized_search.best_estimator_.feature_importances_

extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])

attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)

"""
[(0.34494467079972574, 'median_income'),
 (0.17270875334179428, 'INLAND'),
 (0.11156438666412408, 'pop_per_hhold'),
 (0.0699146551283051, 'bedrooms_per_room'),
 (0.0680788476897446, 'longitude'),
 (0.06437338799728325, 'latitude'),
 (0.05229664262496094, 'rooms_per_hhold'),
 (0.04300665520810458, 'housing_median_age'),
 (0.01624609185547273, 'total_rooms'),
 (0.015168543028308725, 'population'),
 (0.014458910047585575, 'total_bedrooms'),
 (0.014129424198288248, 'households'),
 (0.007866351833047272, '<1H OCEAN'),
 (0.0030872145999579397, 'NEAR OCEAN'),
 (0.0020850563416243396, 'NEAR BAY'),
 (7.0408641672674e-05, 'ISLAND')]
"""
```
- With this information, you may want to try dropping some of the less useful features (e.g., apparently only one `ocean_proximity ( 'INLAND')` category is really useful, so you could try dropping the others `('<1H OCEAN','ISLAND','NEAR BAY', 'NEAR OCEAN')`).

# 4. Evaluate Your System on the Test Set
- After tweaking your models for a while, you eventually have a system that performs sufficiently well. 
- Now is the time to evaluate the final model on the test set.
- get the predictors and the labels from your test set, run your full_pipeline to transform the data (call transform(), not fit_transform(), you do not want to fit the test set!), and evaluate the final model on the test set:

```Python
final_model = randomized_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)   # => evaluates to 48006
final_rmse
```

# 5. Launch, Monitor, and Maintain Your System
- Now, You need to get your solution ready for production, in particular by plugging the production input data sources into your system and writing tests.
- You also need to write monitoring code to check your system’s live performance at regular intervals and trigger alerts when it drops.
  - This is important to catch not only sudden breakage, but also performance degradation. 
- You should also make sure you evaluate the system’s input data quality.
  - Sometimes performance will degrade slightly because of a poor quality signal (e.g., a malfunctioning sensor sending random values, or another team’s output becoming stale)
- You will generally want to train your models on a regular basis using fresh data. You should automate this process as much as possible. 
  - If your system is an online learning system, you should make sure you save snapshots of its state at regular intervals so you can easily roll back to a previously working state.

[(Back to top)](#table-of-contents)

