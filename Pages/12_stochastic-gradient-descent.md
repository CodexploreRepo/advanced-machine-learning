# Stochastic Gradient Descent

# Table of contents
- [Table of contents](#table-of-contents)
- [1. Stochastic Gradient Descent (SGD) Introduction](#1-stochastic-gradient-descent-introduction) 

# 1. Stochastic Gradient Descent Introduction
- `SGDClassifier` relies on randomness during training (hence the name “stochastic”)
- This classifier has the advantage of being capable of handling very large datasets efficiently. This is in part because SGD deals with training instances independently ((which also makes SGD well suited for online learning)

```Python
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
```

[(Back to top)](#table-of-contents)
