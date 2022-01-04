# Model Deployment

# Table of contents
- [Table of contents](#table-of-contents)
- [1. Save Model & Kaggle Submission](#1-save-model-and-kaggle-submission)




# 1. Save Model and Kaggle Submission
## 1.1. Save Model
- You can easily save Scikit-Learn models by using Python’s `pickle` module, or using `sklearn.externals.joblib`, which is more efficient at serializing large NumPy arrays:

```Python
from sklearn.externals import joblib

joblib.dump(my_model, "my_model.pkl")

# and later...
my_model_loaded = joblib.load("my_model.pkl")
```

## 1.2. Kaggle Submission

```Python
predictions = model.predict(X_test)
output = pd.DataFrame({'id': test_data.id, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
```

[(Back to top)](#table-of-contents)
