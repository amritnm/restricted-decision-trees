# Restricted Decision Tree

A Python implementation of **Restricted Decision Trees** - a novel tree-based machine learning algorithm that balances between the rigidity of oblivious trees and the flexibility of standard decision trees.

## 🌳 What is a Restricted Decision Tree?

In a **Restricted Decision Tree**, all nodes at the same depth use the **same feature** for splitting, but each node can have a **different threshold**. This constraint provides:

- **Structure**: More interpretable than standard decision trees
- **Flexibility**: More adaptable than oblivious trees (which use the same feature AND threshold)
- **Performance**: Competitive accuracy with better generalization

### Tree Type Comparison

| Tree Type | Feature Selection | Threshold Selection | Structure |
|-----------|------------------|-------------------|-----------|
| **Standard Decision Tree** | Different feature per node | Different threshold per node | Unrestricted |
| **Restricted Decision Tree** | **Same feature per depth** | **Different threshold per node** | Semi-structured |
| **Oblivious Decision Tree** | Same feature per depth | Same threshold per depth | Highly structured |

## ✨ Features

### Two Implementations

This package provides **two variants** of the Restricted Decision Tree:

1. **`RestrictedDecisionTree` (Greedy)**
   - **Fast**: Chooses feature based on first node at each depth
   - **Efficient**: Lower computational cost
   - **Good**: Provides strong performance in most cases

2. **`RestrictedDecisionTreeFull` (Optimal)**
   - **Optimal**: Uses two-pass algorithm to evaluate all nodes at each depth
   - **Accurate**: Truly maximizes total improvement
   - **Thorough**: Considers all nodes before choosing feature

### Core Capabilities

✅ **Classification** (Binary & Multi-class)  
✅ **Regression** (Continuous targets)  
✅ **Categorical Features** (CatBoost-style ordered encoding)  
✅ **Interactive D3 Visualization** (Built-in HTML generation)  
✅ **Multiple Criteria** (Gini, Entropy, MSE, MAE)  
✅ **Probability Predictions** (For classification)  
✅ **JSON Export** (For custom visualizations)

## 🚀 Quick Start

### Installation

Simply copy the files to your project directory:

```python
from restricted_decision_tree import RestrictedDecisionTree
from restricted_decision_tree_full import RestrictedDecisionTreeFull
```

### Basic Classification Example

```python
import numpy as np
from restricted_decision_tree import RestrictedDecisionTree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Create and train tree
tree = RestrictedDecisionTree(
    task='classification',
    criterion='gini',
    max_depth=4,
    min_samples_split=5
)

tree.fit(X_train, y_train)

# Make predictions
y_pred = tree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# View tree structure
tree.print_tree()
```

### Basic Regression Example

```python
from restricted_decision_tree_full import RestrictedDecisionTreeFull
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

# Generate data
X, y = make_regression(n_samples=200, n_features=5, noise=10.0)

# Create and train tree
tree = RestrictedDecisionTreeFull(
    task='regression',
    criterion='mse',
    max_depth=5
)

tree.fit(X, y)

# Make predictions
y_pred = tree.predict(X)
rmse = np.sqrt(mean_squared_error(y, y_pred))
print(f"RMSE: {rmse:.4f}")
```

## 📊 Advanced Features

### 1. Categorical Features Support

Both implementations support categorical features using **CatBoost-style ordered target statistics**:

```python
import numpy as np

# Dataset with categorical features
X = np.array([
    ['NYC', 50000, 'Bachelor'],      # City, Income, Education
    ['LA', 60000, 'Master'],
    ['NYC', 55000, 'Bachelor'],
    ['Chicago', 45000, 'High School']
])
y = np.array([1, 1, 1, 0])

# Specify which features are categorical
tree = RestrictedDecisionTree(
    task='classification',
    cat_features=[0, 2],  # City and Education are categorical
    prior_weight=1.0      # Smoothing parameter
)

tree.fit(X, y)

# Categorical encodings are learned automatically
print(tree.cat_encodings_)
# {0: {'NYC': 0.833, 'LA': 0.75, ...}, 2: {'Bachelor': 0.70, ...}}
```

**Key Features of Categorical Support:**
- ✅ **Ordered encoding** prevents target leakage
- ✅ **Automatic smoothing** for rare categories
- ✅ **Handles unseen categories** gracefully (uses global prior)
- ✅ Works for both **classification and regression**

### 2. Interactive D3 Visualizations

Generate **interactive HTML visualizations** with a single method call:

```python
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target

tree = RestrictedDecisionTree(task='classification', max_depth=4)
tree.fit(X, y)

# Create interactive D3.js visualization
tree.visualize_d3_observable(
    y=y,
    feature_names=iris.feature_names.tolist(),
    class_names=iris.target_names.tolist(),
    filename='my_tree.html',
    width=1400,
    height=900
)
```

Open `my_tree.html` in a browser to:
- 🖱️ **Click nodes** to expand/collapse subtrees
- 🔍 **Zoom and pan** with mouse
- 📊 **View statistics** for each node
- 🌲 **Explore** tree structure interactively

### 3. Probability Predictions

Get class probabilities for classification tasks:

```python
tree = RestrictedDecisionTree(task='classification')
tree.fit(X_train, y_train)

# Get probability predictions
probas = tree.predict_proba(X_test)
print(probas.shape)  # (n_samples, n_classes)
print(probas[0])     # [0.1, 0.7, 0.2] - probabilities for first sample
```

### 4. Export to JSON

Export tree structure for custom visualizations:

```python
import json

tree_dict = tree.export_tree_dict(
    y=y,
    feature_names=['Feature1', 'Feature2'],
    class_names=['Class 0', 'Class 1']
)

print(json.dumps(tree_dict, indent=2))
# Use in Observable, custom D3 charts, web apps, etc.
```

## 🔧 API Reference

### RestrictedDecisionTree

**Parameters:**

- `task` : str, default='classification'
  - Task type: 'classification' or 'regression'

- `criterion` : str, optional
  - Splitting criterion:
    - Classification: 'gini' (default) or 'entropy'
    - Regression: 'mse' (default) or 'mae'

- `max_depth` : int, default=5
  - Maximum tree depth

- `min_samples_split` : int, default=2
  - Minimum samples required to split a node

- `min_samples_leaf` : int, default=1
  - Minimum samples required in a leaf

- `cat_features` : list, optional
  - Indices of categorical features

- `prior_weight` : float, default=1.0
  - Smoothing weight for categorical encoding

**Methods:**

- `fit(X, y)` - Build the tree from training data
- `predict(X)` - Predict target values
- `predict_proba(X)` - Predict class probabilities (classification only)
- `print_tree()` - Print tree structure
- `visualize_d3_observable(...)` - Create D3 visualization
- `export_tree_dict(...)` - Export tree as nested dictionary

**Attributes (after fitting):**

- `depth_features_` : dict - Feature index used at each depth
- `n_features_` : int - Number of features
- `classes_` : array - Unique class labels (classification)
- `cat_encodings_` : dict - Categorical feature encodings

### RestrictedDecisionTreeFull

Same API as `RestrictedDecisionTree`, but uses optimal two-pass algorithm for feature selection.

## 📚 How It Works

### Algorithm Overview

#### Greedy Variant (RestrictedDecisionTree)

```
For each depth level:
  1. Choose feature based on first node's best split
  2. Each node finds its optimal threshold for chosen feature
  3. Split nodes and move to next depth
```

#### Optimal Variant (RestrictedDecisionTreeFull)

```
For each depth level:
  1. Collect all nodes at current depth
  2. For each potential feature:
     - Find best threshold for each node
     - Calculate total improvement across all nodes
  3. Choose feature with maximum total improvement
  4. Split all nodes using their optimal thresholds
```

### Feature Selection

At each depth level, the algorithm:
1. Evaluates each feature
2. For each feature, finds the best threshold for each node
3. Calculates the total improvement (information gain)
4. Selects the feature that maximizes total improvement

This ensures the **same feature** is used at each depth, but allows **different thresholds** per node.

## 📖 Examples

### Example 1: Compare Greedy vs Full

```python
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=500, n_features=10, random_state=42)

# Greedy variant
tree_greedy = RestrictedDecisionTree(max_depth=4)
tree_greedy.fit(X, y)
acc_greedy = accuracy_score(y, tree_greedy.predict(X))

# Full/Optimal variant
tree_full = RestrictedDecisionTreeFull(max_depth=4)
tree_full.fit(X, y)
acc_full = accuracy_score(y, tree_full.predict(X))

print(f"Greedy accuracy: {acc_greedy:.4f}")
print(f"Full accuracy:   {acc_full:.4f}")
```

### Example 2: Categorical Features

```python
# Car price prediction
X = np.array([
    ['Toyota', 50000, 'New'],
    ['Honda', 30000, 'Used'],
    ['BMW', 70000, 'Certified']
])
y = np.array([25000, 18000, 35000])

tree = RestrictedDecisionTree(
    task='regression',
    cat_features=[0, 2],  # Brand and Condition
    max_depth=3
)

tree.fit(X, y)
predictions = tree.predict(X)
```

### Example 3: D3 Visualization

```python
tree = RestrictedDecisionTree(task='classification', max_depth=4)
tree.fit(X_train, y_train)

# Generate interactive visualization
tree.visualize_d3_observable(
    y=y_train,
    feature_names=['Feature 1', 'Feature 2', 'Feature 3'],
    class_names=['Class A', 'Class B'],
    filename='interactive_tree.html'
)
```

## 🎯 Use Cases

### When to Use Restricted Decision Trees

✅ **Interpretability is important** - Depth-based feature selection is easy to explain  
✅ **Feature importance needed** - Features used at earlier depths are more important  
✅ **Moderate-sized datasets** - Works well with hundreds to thousands of samples  
✅ **Mixed feature types** - Handles numerical and categorical features seamlessly  
✅ **Need visualization** - Built-in D3 visualizations for exploration  

### When to Use Full vs Greedy

**Use Greedy (`RestrictedDecisionTree`) when:**
- Speed is important
- Dataset is large
- Quick prototyping needed

**Use Full (`RestrictedDecisionTreeFull`) when:**
- Accuracy is critical
- Dataset is moderate-sized
- Optimal feature selection desired

## 📁 Package Structure

```
restricted_decision_tree_package/
├── README.md                          # This file
├── restricted_decision_tree.py        # Greedy implementation
├── restricted_decision_tree_full.py   # Optimal implementation
├── examples/
│   ├── basic_usage.py                # Quick start examples
│   ├── d3_visualization_demo.py      # Visualization examples
│   └── categorical_features_demo.py  # Categorical features examples
├── tests/
│   ├── test_rdt.py                   # Tests for greedy variant
│   └── test_rdt_full.py              # Tests for optimal variant
└── docs/
    ├── CATEGORICAL_FEATURES.md        # Categorical features guide
    └── ALGORITHM_DETAILS.md           # Algorithm documentation
```

## 🔬 Technical Details

### Categorical Encoding

Uses **ordered target statistics** (CatBoost-style):
```
encoded_value = (sum_previous_targets + prior_weight * global_prior) / 
                (count_previous + prior_weight)
```

For sample `i`, only samples `0` to `i-1` are used for encoding, preventing target leakage.

### Splitting Criteria

**Classification:**
- **Gini Impurity**: `1 - Σ(p_i²)`
- **Entropy**: `-Σ(p_i * log₂(p_i))`

**Regression:**
- **MSE**: `mean((y - ȳ)²)`
- **MAE**: `mean(|y - median(y)|)`

### Time Complexity

**Greedy Variant:**
- Training: `O(n * m * d * log(n))` where n=samples, m=features, d=depth

**Optimal Variant:**
- Training: `O(n * m² * d * log(n))` - evaluates all features at each depth

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional splitting criteria
- Pruning strategies
- Parallel computation
- GPU acceleration
- More visualization options

## 📄 License

MIT License - feel free to use in your projects!

## 🙏 Acknowledgments

- Inspired by **Oblivious Decision Trees** used in CatBoost
- Categorical encoding based on **CatBoost's ordered target statistics**
- D3.js for interactive visualizations

## 📞 Contact & Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- See examples in the `examples/` directory
- Check documentation in `docs/` directory

## 🚦 Quick Reference

```python
# Import
from restricted_decision_tree import RestrictedDecisionTree
from restricted_decision_tree_full import RestrictedDecisionTreeFull

# Create
tree = RestrictedDecisionTree(task='classification', max_depth=4)

# Fit
tree.fit(X_train, y_train)

# Predict
y_pred = tree.predict(X_test)
y_proba = tree.predict_proba(X_test)  # Classification only

# Visualize
tree.print_tree()
tree.visualize_d3_observable(y=y_train, filename='tree.html')

# Export
tree_dict = tree.export_tree_dict(y=y_train)
```

---

**Happy Tree Building! 🌲**
