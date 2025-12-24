# Categorical Features Support in Restricted Decision Tree

## Summary

**YES**, the Restricted Decision Tree **fully supports categorical features** with the exact same implementation as the Oblivious Decision Tree.

## Implementation Details

### 1. CatBoost-Style Ordered Target Statistics

Both trees use the same sophisticated categorical encoding method:

```python
tree = RestrictedDecisionTree(
    task='classification',
    cat_features=[0, 2],  # Specify which features are categorical
    prior_weight=1.0      # Smoothing parameter
)
```

### 2. Key Features of the Encoding

| Feature | Description |
|---------|-------------|
| **Ordered Approach** | For sample i, only uses samples 0 to i-1 for encoding |
| **Target Leakage Prevention** | Current sample's target is NOT used in its own encoding |
| **Smoothing** | Uses global prior for rare/unseen categories |
| **Formula** | `(sum_previous_targets + prior_weight * global_prior) / (count + prior_weight)` |

### 3. Code Implementation

The restricted tree includes these methods (identical to oblivious tree):

```python
class RestrictedDecisionTree:
    def __init__(self, ..., cat_features=None, prior_weight=1.0):
        self.cat_features = cat_features  # List of categorical feature indices
        self.prior_weight = prior_weight  # Smoothing parameter
        
    def _identify_categorical_features(self, X):
        """Identify which features are categorical"""
        
    def _encode_categorical_features(self, X, y):
        """Encode categorical features using ordered target statistics"""
        # For each categorical feature:
        #   For each sample i:
        #     Use only samples 0 to i-1 for encoding
        #     Apply smoothing with global prior
        
    def _encode_categorical_features_predict(self, X):
        """Encode categorical features for prediction using stored encodings"""
        # Use encodings learned during training
        # Fall back to global prior for unseen categories
```

## Example Usage

### Classification with Categorical Features

```python
from restricted_decision_tree import RestrictedDecisionTree
import numpy as np

# Dataset with mixed features
X = np.array([
    ['NYC', 50000, 'Bachelor'],      # City (cat), Income (num), Education (cat)
    ['LA', 60000, 'Master'],
    ['NYC', 55000, 'Bachelor'],
    ['Chicago', 45000, 'High School'],
    # ... more samples
])

y = np.array([1, 1, 1, 0, ...])  # Target

# Create tree with categorical features specified
tree = RestrictedDecisionTree(
    task='classification',
    cat_features=[0, 2],  # Features 0 and 2 are categorical
    max_depth=4,
    prior_weight=1.0
)

# Fit - categorical encoding happens automatically
tree.fit(X, y)

# View encodings
print(tree.cat_encodings_)
# {0: {'NYC': 0.8333, 'LA': 0.75, 'Chicago': 0.4},
#  2: {'Bachelor': 0.7, 'Master': 0.8, 'High School': 0.3}}

# Predict - handles unseen categories gracefully
X_new = np.array([['Boston', 55000, 'PhD']])  # Unseen categories
predictions = tree.predict(X_new)  # Uses global prior for 'Boston' and 'PhD'
```

### Regression with Categorical Features

```python
# Car price prediction
X = np.array([
    ['Toyota', 50000, 'New'],      # Brand, Mileage, Condition
    ['Honda', 30000, 'Used'],
    ['Ford', 70000, 'Certified'],
    # ... more samples
])

y = np.array([25000, 18000, 20000, ...])  # Prices

tree = RestrictedDecisionTree(
    task='regression',
    cat_features=[0, 2],  # Brand and Condition are categorical
    criterion='mse'
)

tree.fit(X, y)
predictions = tree.predict(X)
```

## How It Works

### During Training (fit)

1. **Identify categorical features** based on `cat_features` parameter
2. **Calculate global prior** (mean of target values)
3. **Encode each categorical feature** using ordered target statistics:
   ```
   For each sample i:
     If i == 0:
       encoding = global_prior
     Else:
       Find previous samples (0 to i-1) with same category
       encoding = (sum_of_their_targets + prior_weight * global_prior) / 
                  (count + prior_weight)
   ```
4. **Store encodings** for each category in `cat_encodings_`
5. **Build tree** using the encoded numerical values

### During Prediction

1. **Encode categorical features** using stored encodings from training
2. For **unseen categories**: use global prior
3. **Navigate tree** with encoded values to make predictions

## Comparison with Oblivious Tree

| Aspect | Oblivious Tree | Restricted Tree |
|--------|----------------|-----------------|
| Categorical encoding method | ✓ Ordered Target Statistics | ✓ Ordered Target Statistics |
| Prevents target leakage | ✓ Yes | ✓ Yes |
| Handles unseen categories | ✓ Yes (global prior) | ✓ Yes (global prior) |
| Smoothing parameter | ✓ Yes | ✓ Yes |
| Implementation | Same | **Identical** |

**The categorical feature handling is EXACTLY THE SAME in both implementations.**

## Advantages

1. **No preprocessing needed**: Just specify which features are categorical
2. **Prevents overfitting**: Ordered approach prevents target leakage
3. **Robust to unseen data**: Uses global prior for new categories
4. **Works with both tasks**: Classification and regression
5. **Automatic encoding**: Happens transparently during `fit()`

## Testing

See `test_restricted_categorical.py` for comprehensive tests including:
- Classification with mixed features
- Regression with categorical variables
- Handling of unseen categories
- Encoding visualization

Run tests:
```bash
python test_restricted_categorical.py
```

## Technical Notes

### Parameters

- `cat_features`: List of indices (e.g., `[0, 2, 5]`) indicating which features are categorical
- `prior_weight`: Smoothing parameter (default: 1.0)
  - Higher values → more smoothing (more weight to global prior)
  - Lower values → less smoothing (more weight to category statistics)

### Storage

- `cat_features_`: Validated list of categorical feature indices
- `cat_encodings_`: Dictionary mapping `{feature_idx: {category: encoding}}`
- `global_prior_`: Mean of target values used for smoothing

### Encoding Formula

```
encoded_value = (sum_previous_targets + prior_weight * global_prior) / 
                (count_previous + prior_weight)
```

This is equivalent to:
- Bayesian smoothing with global prior
- Shrinkage estimator towards the mean
- Prevents overfitting to rare categories

## Conclusion

The Restricted Decision Tree has **complete and identical categorical feature support** as the Oblivious Decision Tree. The only difference between the two implementations is in the tree structure (same feature but different thresholds vs. same feature and same threshold at each depth). The categorical encoding is exactly the same.
