# Data Pre-Processing

# Table of contents
- [Table of contents](#table-of-contents)
- [2. Data Pre-Processing](#2-data-pre-processing)
  - [2.1. Read and Split Data](#21-read-and-split-data) 
  - [2.2. Missing Values](#22-missing-values)
  - [2.3. Categorical variable](#23-categorical-variable)

# 2. Data Pre-Processing
## 2.1. Read and Split Data
- **Step 1**: Read the data
```Python
X_full = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

# Obtain target and predictors
y = X_full["target"]

X = X_full[:-1].copy() #X will not include last column, which is "target" column
X_test = X_test_full.copy()
```
- **Step 2**: Investigate and filter Numeric & Categorical Data
  - Note 1: Some features although they are numerical, but there data type is object, and vice versa. Hence, **need to spend time to investigate on the real type of the features**, *convert them into correct data type before performing the below commands*.
  ```Python
  # Select numeric columns only
  numeric_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]
  # Categorical columns in the training data
  object_cols = [col for col in X.columns if X[col].dtype == "object"]
  ```
- **Step 3**: Break off validation set from training data `X`
```Python
# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
```
- **Step 4**: Comparing different models
```Python
models = [model_1, model_2, model_3, model_4, model_5]

# Function for comparing different models
def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)

for i in range(0, len(models)):
    mae = score_model(models[i])
    print("Model %d MAE: %d" % (i+1, mae))
```
[(Back to top)](#table-of-contents)

## 2.2. Missing Values
- This part is to handle missing values for both `Numerical` & `Categorical` Data
```Python
# Get names of columns with missing values
cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]

# Number of missing values in each column of training data
missing_val_count_by_column = (X_train.isnull().sum())
```
- **Method 1**: Drop Columns with Missing Values 
- **Method 2**: Imputation
- **Method 3**: Extension To Imputation

### 2.2.1. Method 1: Drop Columns with Missing Values
<img width="889" alt="Screenshot 2021-08-20 at 10 53 33" src="https://user-images.githubusercontent.com/64508435/130171794-186b7922-3464-4057-9004-87111c6ea44f.png">

```Python
# Drop columns in training and validation data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)
```
### 2.2.2. Method 2: Imputation
- `Imputation` fills in the missing values with some number.
- `strategy = “mean”, "median"` for numerical column
- `strategy = “most_frequent”` for object (categorical) column
<img width="889" alt="Screenshot 2021-08-20 at 10 56 11" src="https://user-images.githubusercontent.com/64508435/130172082-479fbb77-03f9-4438-b8bc-97bcbe3e0d1e.png">

```Python
from sklearn.impute import SimpleImputer

# Imputation
my_imputer = SimpleImputer(missing_values=np.nan, strategy="mean") 
#Only fit on training data
my_imputer.fit(X_train) 

imputed_X_train = pd.DataFrame(my_imputer.transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# Fill in the lines below: imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns
```
### 2.2.2. Method 3: Extension To Imputation
- Imputation is the standard approach, and it usually works well. 
- However, imputed values may be systematically above or below their actual values (which weren't collected in the dataset). Or rows with missing values may be unique in some other way. 
- In that case, your model would make better predictions by considering which values were originally missing.

<img width="889" alt="Screenshot 2021-08-20 at 11 36 52" src="https://user-images.githubusercontent.com/64508435/130175336-a19e86d8-cba1-489a-87cb-c33c9378f8c0.png">

- **Note**: In some cases, this will meaningfully improve results. In other cases, it doesn't help at all.
```Python
# Make copy to avoid changing original data (when imputing)
X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()

# Make new columns indicating what will be imputed
for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

# Imputation
my_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))

# Imputation removed column names; put them back
imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_valid_plus.columns = X_valid_plus.columns
```

[(Back to top)](#table-of-contents)

## 2.3. Categorical variable
- There are 4 types of Categorical variable
  - `Nominal`: non-order variables like "Honda", "Toyota", and "Ford"
  - `Ordinal`: the order is important 
    - For tree-based models (like decision trees and random forests), you can expect ordinal encoding to work well with ordinal variables
    - `Label Encoder` &#8594; can map to 1,2,3,4, etc &#8594; Use **Tree-based Models: Random Forest, GBM, XGBoost**
    - `Binary Encoder` &#8594; binary-presentation vectors of 1,2,3,4, etc values &#8594; Use **Logistic and Linear Regression, SVM**
  - `Binary`: only have 2 values (Female, Male)
  - `Cyclic`: Monday, Tuesday, Wednesday, Thursday
- Determine Categorical Columns:
```Python
# Categorical columns in the training data 
object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]
```
- **Filter Good & Problematic Categorical Columns** which will affect Encoding Procedure:
  - For example: Unique values in Train Data are different from Unique values in Valid Data &#8594; Solution: ensure values in `Valid Data` is a subset of values in `Train Data`
  - The simplest approach, however, is to drop the problematic categorical columns.
```Python
# Categorical columns in the training data
object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

# Columns that can be safely ordinal encoded
good_label_cols = [col for col in object_cols if 
                   set(X_valid[col]).issubset(set(X_train[col]))]
        
# Problematic columns that will be dropped from the dataset
bad_label_cols = list(set(object_cols)-set(good_label_cols))
        
print('Categorical columns that will be ordinal encoded:', good_label_cols)
print('\nCategorical columns that will be dropped from the dataset:', bad_label_cols)
```
  - The simplest approach, however, is to drop the problematic categorical columns.
```Python
# Drop categorical columns that will not be encoded
label_X_train = X_train.drop(bad_label_cols, axis=1)
label_X_valid = X_valid.drop(bad_label_cols, axis=1)
```
There are 5 methods to encode Categorical variables 
- **Method 1**: Drop Categorical Variables 
- **Method 2**: Ordinal Encoding
- **Method 3**: Label Encoding (Same as Ordinal Encoder but NOT care about the order)
- **Method 4**: One-Hot Encoding
  - If a categorical attribute has a large number of possible categories, then one-hot encoding will result in a large number of input features. This may slow down training and degrade performance.
  - If this happens, you will want to produce denser representations called `embeddings`, but this requires a good understanding of neural networks (see [Chapter 14](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781491962282/ch14.html#rnn_chapter) for more details).
- **Method 5**: Entity Embedding (Need to learn from Video: https://youtu.be/EATAM3BOD_E)

### 2.3.1. Method 1: Drop Categorical Variables 
- This approach will only work well if the columns did not contain useful information.
```Python
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])
```
### 2.3.2. Method 2: Ordinal Encoding
<img width="764" alt="Screenshot 2021-08-22 at 18 00 58" src="https://user-images.githubusercontent.com/64508435/130351069-8cd904d8-f59d-4c6e-a454-1b636c81c2e2.png">

- This approach assumes an ordering of the categories: "Never" (0) < "Rarely" (1) < "Most days" (2) < "Every day" (3).

```Python
from sklearn.preprocessing import OrdinalEncoder

# Apply ordinal encoder 
ordinal_encoder = OrdinalEncoder() # Your code here
ordinal_encoder.fit(label_X_train[good_label_cols])

label_X_train[good_label_cols] = ordinal_encoder.transform(label_X_train[good_label_cols])
label_X_valid[good_label_cols] = ordinal_encoder.transform(label_X_valid[good_label_cols])
```
### 2.3.3. Method 3: Label Encoding
- Same as Ordinal Encoder but NOT care about the order, but follow by Alphabet of the values
- `Label Encoder` need to **fit in each column separately**
```Python
from sklearn.preprocessing import LabelEncoder

# Drop categorical columns that will not be encoded
label_X_train = X_train.drop(bad_label_cols, axis=1)
label_X_valid = X_valid.drop(bad_label_cols, axis=1)


for c in good_label_cols:
    label_encoder = LabelEncoder()
    label_encoder.fit(label_X_train[c])
    label_X_train[c] = label_encoder.transform(label_X_train[c])
    label_X_valid[c] = label_encoder.transform(label_X_valid[c])
```
### 2.3.4. Method 4: One-Hot Encoding
#### Investigating Cardinality
- `Cardinality`: # of unique entries of a categorical variable
  - For instance, the `Street` column in the training data has two unique values: `Grvl` and `Pave`, the `Street` col has cardinality 2
- For large datasets with many rows, one-hot encoding can greatly expand the size of the dataset. 
- Hence, we typically will only one-hot encode columns with relatively `low cardinality`. 
- `High cardinality` columns can either be dropped from the dataset, or we can use ordinal encoding.
```Python
# Columns that will be one-hot encoded
low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]

# Columns that will be dropped from the dataset
high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))
```
#### One-Hot Encoding
- One-hot encoding generally does NOT perform well if the categorical variable has `cardinality >= 15` as One-Hot encoder will expand the original training data with increasing columns

<img width="764" alt="Screenshot 2021-08-22 at 18 33 33" src="https://user-images.githubusercontent.com/64508435/130351973-e54a71c1-c010-4233-a282-37e5528eaccd.png">

- Set `handle_unknown='ignore'` to avoid errors when the validation data contains classes that aren't represented in the training data, and
- Set `sparse=False` ensures that the encoded columns are returned as a numpy array (instead of a sparse matrix).

```Python
from sklearn.preprocessing import OneHotEncoder

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_encoder.fit(X_train[low_cardinality_cols])
OH_cols_train = pd.DataFrame(OH_encoder.transform(X_train[low_cardinality_cols])) #Convert back to Pandas DataFrame from Numpy Array
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[low_cardinality_cols]))  

# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remove categorical columns in the original datasets (will replace with one-hot encoding columns)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)
```



[(Back to top)](#table-of-contents)
