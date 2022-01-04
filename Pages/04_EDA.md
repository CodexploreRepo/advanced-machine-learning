# 4. EDA

# Table of contents
- [Table of contents](#table-of-contents)
- [1. Graph](#1-graph)
  - [1.1 Label](#11-label) 



# 1. Graph
## 1.1. Label
### 1.1.1. Overwrite Label:
```Python
# Get current yticks: An array of the values displayed on the y-axis (150, 175, 200, etc.)
ticks = ax.get_yticks()
# Format those values into strings beginning with dollar sign
new_labels = [f"${int(tick)}" for tick in ticks]
# Set the new labels
ax.set_yticklabels(new_labels)
```
[(Back to top)](#table-of-contents)

# Resources
- Resource: [Hands-On Machine Learning with Scikit-Learn and TensorFlow](https://github.com/ageron/handson-ml)

[(Back to top)](#table-of-contents)
