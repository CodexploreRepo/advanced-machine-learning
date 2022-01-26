# Evaluation Metrics

# Table of contents
- [Table of contents](#table-of-contents)
- [1. Underfitting and Overfitting](#1-underfitting-and-overfitting)
- [2. Cross-Validation](#2-cross-validation)
  - [2.1. Stratified k-fold](#21-stratified-k-fold) 
- [3. Metrics for Regression](#3-metrics-for-regression)
- [4. Metrics for Classification](#4-metrics-for-classification)
  - [4.1. Confusion Matrix](#41-confusion-matrix)
  - [4.2. Precision and Recall](#42-precision-and-recall)
  - [4.3. F1 Score](#43-f1-score)
  - [4.4. ROC](#44-roc)

# 1. Underfitting and Overfitting
Models can suffer from either:
- **Overfitting**: capturing spurious patterns that won't recur in the future, leading to less accurate predictions
  - Where a model matches the training data almost perfectly, but does poorly in validation and other new data.  
- **Underfitting**: failing to capture relevant patterns, again leading to less accurate predictions.
  - When a model fails to capture important distinctions and patterns in the data, so it performs poorly even in training data 
## 1.1. Methods to avoid Underfitting and Overfitting
### Example 1: DecisionTreeRegressor Model
- `max_leaf_nodes` argument provides a very sensible way to control overfitting vs underfitting. The more leaves we allow the model to make, the more we move from the underfitting area in the above graph to the overfitting area.
<p align="center"><img src="https://user-images.githubusercontent.com/64508435/129434680-b30efd3e-ab04-4871-85ce-03b53027c0e7.png" height="320px" /></p>

- We can use a utility function to help compare MAE scores from different values for `max_leaf_nodes`:
```Python
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)
```
- Call the *get_mae* function on each value of max_leaf_nodes. Store the output in some way that allows you to select the value of `max_leaf_nodes` that gives the most accurate model on your data.
```Python
scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}
best_tree_size = min(scores.keys(), key=(lambda k: scores[k]))
```

[(Back to top)](#table-of-contents)

# 2. Cross-Validation
- **For small dataset &#8594; Cross-validation**, we run our modeling process on different subsets of the data to get multiple measures of model quality.
- **For large dataset &#8594; Hold-out**: when `training data > 100K or 1M`, we will hold-out 5-10% data as a validation set.

  ```Python
  from sklearn.model_selection import cross_val_score
  
  def get_score(n_estimators):
      my_pipeline = Pipeline(steps=[
          ('preprocessor', SimpleImputer()),
          ('model', RandomForestRegressor(n_estimators=n_estimators, random_state=0))
      ])
      
      # Multiply by -1 since sklearn calculates *negative* MAE
      scores = -1 * cross_val_score(my_pipeline, X, y,
                                cv=5, #This is 5-fold cross-validation
                                scoring='neg_mean_absolute_error')
      #Since cross_val_score return 5 MAE for each fold, so take mean()
      return scores.mean() 
  
  #Evaluate the model performance corresponding to eight different values for the number of trees (n_estimators) in the random forest: 50, 100, 150, ..., 300, 350, 400.
  results = {n_estimators: get_score(n_estimators) for n_estimators in range(50, 450, 50)} 
  
  plt.plot(results.keys(), results.values())
  plt.show()
  ```
  <img width="414" alt="Screenshot 2021-12-02 at 16 42 21" src="https://user-images.githubusercontent.com/64508435/144396996-81ae36c5-98c1-4a50-b9df-dee77eaf44cd.png"> 
  
## 2.1. Stratified k-fold
Stratified k-fold cross-validation is same as just k-fold cross-validation, but in Stratified k-fold cross-validation, it does stratified sampling instead of random sampling.
  - For example, the US population is composed of 51.3% female and 48.7% male, so a well-conducted survey in the US would try to maintain this ratio in the sample: 513 female and 487 male. This is called `stratified sampling`: the population is divided into homogeneous subgroups called `strata`, and the right number of instances is sampled from each stratum to guarantee that the test set is representative of the overall population. If they used purely random sampling, there would be about 12% chance of sampling a skewed test set with either less than 49% female or more than 54% female. Either way, the survey results would be significantly biased.
    - Hence, Stratified k-fold keeps the same ratio of classes in each fold in comparison with the ratio of the original training data.
    - **Classification** problem: can apply Stratified k-fold directly
    - **Regression** problem: need to convert `Y` into `1+log2(N)` bins (Sturge’s Rule) and then Stratified k-fold  will split accordingly.
  ![image](https://user-images.githubusercontent.com/64508435/144378824-53f0db43-38f1-47cf-a0c2-15bf74f9d2ab.png)
  
  ```Python
  from sklearn.model_selection import StratifiedKFold
  from sklearn.base import clone

  skfolds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

  for train_index, test_index in skfolds.split(X_train, y_train_5):
      #train_index = [array of indices]; test_index = [array of indices]
      #print(len(train_index), len(test_index)) #40000 20000

      #Clone does a deep copy of the model in an estimator without actually copying attached data. 
      #It returns a new estimator with the same parameters that has not been fitted on any data.
      clone_clf = clone(sgd_clf)

      X_train_folds = X_train.iloc[list(train_index)]
      y_train_folds = y_train_5.iloc[list(train_index)]
      X_test_fold = X_train.iloc[list(test_index)]
      y_test_fold = y_train_5.iloc[list(test_index)]

      clone_clf.fit(X_train_folds, y_train_folds)
      y_pred = clone_clf.predict(X_test_fold)
      n_correct = sum(y_pred == y_test_fold)
      print(n_correct / len(y_pred))  # prints 0.9502, 0.96565 and 0.96495
  ```
  - At each iteration the code creates a clone of the classifier, trains that clone on the training folds, and makes predictions on the test fold. Then it counts the number of correct predictions and outputs the ratio of correct predictions
  - Let’s use the `cross_val_score()` function to evaluate your `SGDClassifier` model using K-fold cross-validation, with three folds. 
  - One possible reason for the difference between `cross_val_score` and `StratifiedKFold` is that `cross_val_score` uses `StratifiedKFold` with the default `shuffle=False` parameter, whereas in your manual cross-validation using `StratifiedKFold` you have passed `shuffle=True`.
  - Remember that K-fold cross-validation means splitting the training set into K-folds (in this case, three), then making predictions and evaluating them on each fold using a model trained on the remaining folds 
  
  ```Python
  from sklearn.model_selection import cross_val_score
  #cross_val_score uses StratifiedKFold with the default shuffle=False parameter
  cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
  array([0.95035, 0.96035, 0.9604 ])
  ```

[(Back to top)](#table-of-contents)

# 3. Metrics for Regression
## 3.1 Mean Absolute Error (MAE)
```Python
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_pred, y_test)
```
## 3.2 Root Mean Squared Error (RMSE)
```Python
import numpy as np
from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(y_pred, y_test))
```

# 4. Metrics for Classification
- A naive classifier that always classify the image NOT number "5" in the dataset that contains only 10% is "5" images, still can achieve Accuracy of 90%. 
- This is simply because only about 10% of the images are 5s, so if you always guess that an image is not a 5, you will be right about 90% of the time. 
- This demonstrates why accuracy is generally not the preferred performance measure for classifiers, especially when you are dealing with skewed datasets (i.e., when some classes are much more frequent than others).
## 4.1. Confusion Matrix
- A much better way to evaluate the performance of a classifier is to look at the confusion matrix. The general idea is to count the number of times instances of class A are classified as class B. 
- To compute the confusion matrix, you first need to have a set of predictions so that they can be compared to the actual targets. You could make predictions on the test set, but let’s keep it untouched for now (remember that you want to use the test set only at the very end of your project, once you have a classifier that you are ready to launch). Instead, you can use the `cross_val_predict()` function:
```Python
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
```

- Just like the `cross_val_score()` function, `cross_val_predict()` performs K-fold cross-validation, but instead of returning the evaluation scores, it returns the predictions made on each test fold. 
- This means that you get a clean prediction for each instance in the training set (“clean” meaning that the prediction is made by a model that never saw the data during training).
- Now you are ready to get the confusion matrix using the `confusion_matrix()` function. Just pass it the target classes (`y_train_5`) and the predicted classes (`y_train_pred`):

```Python
from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_5, y_train_pred)

#array([[53892,   687],
#       [ 1891,  3530]])
```
#### Understanding Confusion Matrix
- Each **row** in a confusion matrix represents an `actual class`, 
- Each **column** represents a `predicted class`. 
- A perfect classifier would have only true positives and true negatives, so its confusion matrix would have nonzero values only on its **main diagonal (top left to bottom right)**

<p align="center"><img width="500" alt="Screenshot 2021-08-15 at 16 25 45" src="https://user-images.githubusercontent.com/64508435/148041146-2995ca33-faca-4461-b7b8-29b022b44e96.png"></p>


```Python
y_train_perfect_predictions = y_train_5  # pretend we reached perfection
confusion_matrix(y_train_5, y_train_perfect_predictions)

#array([[54579,     0],
#      [    0,  5421]])
```

## 4.2. Precision and Recall
- **Precision**: How precise is the positive predictions.
- **Recall**: How many positive cases are deteceted.
- Exampl of Precision & Recall: For Covid detection, we want to have High Recall, which is to capture as much Positive Cases as Possible by lowering down the threshod. Although, this migh cause Low Precision (i.e: might have quite a number of False Positive cases).
<p align="center"><img width="250" alt="Screenshot 2021-08-15 at 16 25 45" src="https://user-images.githubusercontent.com/64508435/148773812-229be486-9c96-4800-85d2-730621d2441c.png"></p>
<p align="center"><img width="250" alt="Screenshot 2021-08-15 at 16 25 45" src="https://user-images.githubusercontent.com/64508435/148773944-5d15d210-5a4c-4d3c-8947-c6aa681f4b95.png"></p>

```Python
>>> from sklearn.metrics import precision_score, recall_score
>>> precision_score(y_train_5, y_train_pred) # == 4096 / (4096 + 1522)
0.7290850836596654
>>> recall_score(y_train_5, y_train_pred) # == 4096 / (4096 + 1325)
0.7555801512636044
```
- In this case, when the classifer claims an image represents a 5, it is correct only 72.9% of the time. (Precision)
- Moreover, it only detects 75.5% of the 5s. (Recall)

## 4.3. F1 Score
- It is often convenient to combine precision and recall into a single metric called the F1 score, in particular if you need a simple way to compare two classifiers. 
- The F1 score is the harmonic mean of precision and recall. 
- Whereas the regular mean treats all values equally, the harmonic mean gives much more weight to low values. 
- As a result, the classifier will only get a high F1 score if both recall and precision are high.
```Python
>>> from sklearn.metrics import f1_score
>>> f1_score(y_train_5, y_train_pred)
0.7420962043663375
```
<p align="center"><img width="450" alt="Screenshot 2021-08-15 at 16 25 45" src="https://user-images.githubusercontent.com/64508435/148774640-5e2378e3-c4f7-4796-bb66-b379a6daf930.png"></p>

- The F1 score favors classifiers that have similar precision and recall. This is not always what you want: in some contexts you mostly care about precision, and in other contexts you really care about recall. 
  - For example, if you trained a classifier to detect videos that are safe for kids, you would probably prefer a classifier that rejects many good videos (low recall) but keeps only safe ones (high precision), rather than a classifier that has a much higher recall but lets a few really bad videos show up in your product (in such cases, you may even want to add a human pipeline to check the classifier’s video selection). 
  - On the other hand, suppose you train a classifier to detect shoplifters in surveillance images: it is probably fine if your classifier has only 30% precision as long as it has 99% recall (sure, the security guards will get a few false alerts, but almost all shoplifters will get caught).

<p align="center"><img width="650" alt="Screenshot 2021-08-15 at 16 25 45" src="https://user-images.githubusercontent.com/64508435/151130205-bb186ed2-e1a5-42ef-99e8-6c37c8ab8816.jpeg"></p>

## 4.4. ROC
- When `thredshod = 0.8`, we can achieve very high precision, but we might left out those positive class but the prob < 0.8, will cause low recall.
- When `thredshod = 0.2`, we can achieve very high recall (able to detect all positive case), but we might include also those belong to negative class as Positive, causing the low precision.

<p align="center"><img width="650" alt="Screenshot 2021-08-15 at 16 25 45" src="https://user-images.githubusercontent.com/64508435/151132178-4b0971a2-6472-4445-b226-bf8e2d77a609.jpeg"></p>

- The receiver operating characteristic (ROC) curve is another common tool used with binary classifiers. It is very similar to the precision/recall curve, but instead of plotting precision versus recall, the ROC curve plots the `true positive rate` (another name for recall) against the `false positive rate` (FPR) at **various threshold**. 
  - The FPR is the ratio of negative instances that are incorrectly classified as positive.

<p align="center"><img width="650" alt="Screenshot 2021-08-15 at 16 25 45" src="https://user-images.githubusercontent.com/64508435/151135435-cb43a832-a37c-4aec-84f6-35d94310e0fb.jpeg"></p>

- One way to compare classifiers is to measure the area under the curve (AUC). A perfect classifier will have a ROC AUC equal to 1, whereas a purely random classifier will have a ROC AUC equal to 0.5. Scikit-Learn provides a function to compute the ROC AUC:

<p align="center"><img width="650" alt="Screenshot 2021-08-15 at 16 25 45" src="https://user-images.githubusercontent.com/64508435/151135716-68ac607f-66e9-475d-9daf-a11a1fd0f92a.jpeg"></p>

