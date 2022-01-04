# 3. Transformation Pipeline

# Table of contents
- [Table of contents](#table-of-contents)



# Example of Pipeline
- **Pipelines** are a simple way to keep your data preprocessing and modeling code organized.
- Specifically, a pipeline bundles preprocessing and modeling steps so you can use the whole bundle as if it were a single step.
- Construct the full pipeline in three steps:
  - **Step 1: Define Preprocessing Steps**
    - A pipeline bundles together preprocessing and modeling steps, we use the `ColumnTransformer` class to bundle together different preprocessing steps. 
      - imputes missing values in numerical data, and
      - imputes missing values and applies a one-hot encoding to categorical data.
    ```Python
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder

    # Preprocessing for numerical data
    numerical_transformer = SimpleImputer(strategy='constant')

    # Preprocessing for categorical data using Pipeline class
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle preprocessing for numerical and categorical data using ColumnTransformer class
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    ```
  - **Step 2: Define the Model**
    ```Python
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import GradientBoostingRegressor

    model1 = RandomForestRegressor(n_estimators=100, random_state=0)
    model2 = GradientBoostingRegressor(n_estimators=500, random_state = 42)
    ```
  - **Step 3: Create and Evaluate the Pipeline**
    - Finally, we use the Pipeline class to define a pipeline that bundles the preprocessing and modeling steps. There are a few important things to notice:
      - With the pipeline, we preprocess the training data and fit the model in a single line of code. *(In contrast, without a pipeline, we have to do imputation, one-hot encoding, and model training in separate steps. This becomes especially messy if we have to deal with both numerical and categorical variables!)*
      - With the pipeline, we supply the unprocessed features in X_valid to the predict() command, and the pipeline automatically preprocesses the features before generating predictions. *(However, without a pipeline, we have to remember to preprocess the validation data before making predictions.)*
    ```Python
    from sklearn.ensemble import GradientBoostingRegressor
    my_pipeline1 = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('rf', model1)
                                 ])
    my_pipeline2 = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('gbm', model2)
                                 ])

    # Preprocessing of training data, fit model 
    my_pipeline1.fit(X_train, y_train)
    # Preprocessing of validation data, get predictions
    preds1 = my_pipeline1.predict(X_valid)

    my_pipeline2.fit(X_train, y_train)
    preds2 = my_pipeline2.predict(X_valid)

    preds = (preds1 + preds2)/2
    # Evaluate the model
    score = mean_absolute_error(y_valid, preds)
    print('MAE:', score)
    ```
[(Back to top)](#table-of-contents)


