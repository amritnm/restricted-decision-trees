# Restricted Decision Tree - Algorithm Details

## Overview

This document provides an in-depth explanation of the Restricted Decision Tree algorithm, including both the **Greedy** and **Full/Optimal** variants.

## Core Concept

### Restriction Constraint

In a Restricted Decision Tree:
- **All nodes at depth `d` use the SAME feature for splitting**
- **Each node at depth `d` can use DIFFERENT thresholds**

This is the key distinction from:
- **Standard Decision Trees**: Each node uses any feature and threshold
- **Oblivious Decision Trees**: All nodes at depth `d` use same feature AND same threshold

### Benefits

1. **Interpretability**: Easier to understand which features matter at each depth
2. **Regularization**: Constraint prevents overfitting compared to standard trees
3. **Flexibility**: More adaptive than oblivious trees
4. **Feature Importance**: Natural ranking by depth

## Algorithm Variants

### 1. Greedy Variant (`RestrictedDecisionTree`)

#### Algorithm

```
function BuildRestrictedTree_Greedy(X, y, max_depth):
    root = Node(indices=all_samples)
    
    for depth = 0 to max_depth-1:
        if no nodes to split:
            break
        
        // Choose feature based on first/representative node
        best_feature = FindBestFeatureForFirstNode(depth)
        
        for each node at depth:
            // Find optimal threshold for this node using best_feature
            threshold = FindBestThreshold(node, best_feature)
            
            // Split node
            node.left, node.right = Split(node, best_feature, threshold)
    
    return root
```

#### Feature Selection

For each depth:
1. Take the first node at that depth (or a representative node)
2. Find the feature that gives the best split for that node
3. Use this feature for ALL nodes at this depth

#### Characteristics

- **Time Complexity**: O(n × m × d × log(n))
  - n = number of samples
  - m = number of features
  - d = tree depth

- **Advantages**:
  - Fast computation
  - Simple implementation
  - Good performance in practice

- **Disadvantages**:
  - Not guaranteed to be optimal
  - Feature choice based on single node

### 2. Full/Optimal Variant (`RestrictedDecisionTreeFull`)

#### Algorithm

```
function BuildRestrictedTree_Optimal(X, y, max_depth):
    root = Node(indices=all_samples)
    current_level = [root]
    
    for depth = 0 to max_depth-1:
        if current_level is empty:
            break
        
        // TWO-PASS ALGORITHM
        
        // PASS 1: Evaluate all features
        best_feature = None
        best_total_improvement = -∞
        best_thresholds = {}
        
        for each feature in features:
            total_improvement = 0
            thresholds = {}
            
            // For each node, find best threshold for this feature
            for each node in current_level:
                threshold, improvement = FindBestThresholdForFeature(
                    node, feature
                )
                total_improvement += improvement
                thresholds[node.id] = threshold
            
            // Check if this feature is better than current best
            if total_improvement > best_total_improvement:
                best_total_improvement = total_improvement
                best_feature = feature
                best_thresholds = thresholds
        
        // PASS 2: Split all nodes using best feature and their thresholds
        next_level = []
        for each node in current_level:
            threshold = best_thresholds[node.id]
            node.left, node.right = Split(node, best_feature, threshold)
            next_level.append(node.left, node.right)
        
        current_level = next_level
    
    return root
```

#### Feature Selection

For each depth:
1. **Collect** all nodes at current depth
2. **For each feature**:
   - Find optimal threshold for each node
   - Calculate total improvement across all nodes
3. **Choose** feature that maximizes total improvement
4. **Split** all nodes using their optimal thresholds

#### Characteristics

- **Time Complexity**: O(n × m² × d × log(n))
  - Evaluates all m features at each depth
  - For each feature, evaluates all nodes

- **Advantages**:
  - Truly optimal feature selection at each depth
  - Maximizes total improvement
  - Better accuracy potential

- **Disadvantages**:
  - Slower than greedy variant
  - Higher computational cost

## Splitting Process

### 1. Finding Best Feature (Optimal Variant)

```python
def find_optimal_feature_for_depth(nodes, X, y):
    best_feature = None
    best_total_improvement = -∞
    best_node_thresholds = {}
    
    for feature_idx in range(n_features):
        total_improvement = 0
        node_thresholds = {}
        
        for node in nodes:
            # Find best threshold for this feature at this node
            threshold, improvement = find_best_threshold(
                X[node.indices, feature_idx],
                y[node.indices]
            )
            
            if threshold is not None:
                total_improvement += improvement
                node_thresholds[node.id] = threshold
        
        if total_improvement > best_total_improvement:
            best_total_improvement = total_improvement
            best_feature = feature_idx
            best_node_thresholds = node_thresholds
    
    return best_feature, best_node_thresholds
```

### 2. Finding Best Threshold for a Node

```python
def find_best_threshold(X_feature, y):
    # Get unique values and compute candidate thresholds
    unique_values = sorted(set(X_feature))
    thresholds = [(unique_values[i] + unique_values[i+1]) / 2 
                  for i in range(len(unique_values) - 1)]
    
    best_improvement = -∞
    best_threshold = None
    
    for threshold in thresholds:
        # Split samples
        left_mask = X_feature <= threshold
        y_left = y[left_mask]
        y_right = y[~left_mask]
        
        # Check minimum samples constraint
        if len(y_left) < min_samples_leaf or len(y_right) < min_samples_leaf:
            continue
        
        # Calculate improvement
        improvement = calculate_improvement(y, y_left, y_right)
        
        if improvement > best_improvement:
            best_improvement = improvement
            best_threshold = threshold
    
    return best_threshold, best_improvement
```

### 3. Calculating Improvement

```python
def calculate_improvement(y_parent, y_left, y_right):
    n_parent = len(y_parent)
    n_left = len(y_left)
    n_right = len(y_right)
    
    # Parent criterion value
    parent_criterion = calculate_criterion(y_parent)
    
    # Weighted child criterion
    left_criterion = calculate_criterion(y_left)
    right_criterion = calculate_criterion(y_right)
    child_criterion = (n_left / n_parent) * left_criterion + \
                      (n_right / n_parent) * right_criterion
    
    # Improvement = reduction in criterion
    improvement = parent_criterion - child_criterion
    
    return improvement
```

## Splitting Criteria

### Classification

#### Gini Impurity

```
Gini(S) = 1 - Σ(p_i²)
```

where `p_i` is the proportion of class `i` in set `S`.

**Properties:**
- Range: [0, 0.5] for binary classification
- 0 = pure node (all same class)
- Maximum at equal proportions

#### Entropy (Information Gain)

```
Entropy(S) = -Σ(p_i × log₂(p_i))
```

**Properties:**
- Range: [0, log₂(n_classes)]
- 0 = pure node
- Maximum at equal proportions
- More sensitive to probability changes than Gini

### Regression

#### Mean Squared Error (MSE)

```
MSE(S) = (1/|S|) × Σ((y_i - ȳ)²)
```

where `ȳ` is the mean of targets in `S`.

**Properties:**
- Variance of the target values
- Penalizes large deviations
- Leads to mean prediction in leaves

#### Mean Absolute Error (MAE)

```
MAE(S) = (1/|S|) × Σ(|y_i - median(y)|)
```

where `median(y)` is the median of targets in `S`.

**Properties:**
- Robust to outliers
- Leads to median prediction in leaves
- Less sensitive to extreme values

## Tree Construction Flow

### Greedy Variant

```
1. Initialize root with all samples
2. For each depth level:
   a. Find best feature using first node
   b. For each node at this depth:
      - Find best threshold for chosen feature
      - Split node if valid
   c. Move to next depth
3. Make remaining nodes leaves
```

### Optimal Variant

```
1. Initialize root with all samples
2. current_level = [root]
3. For each depth level:
   a. Filter nodes that can be split
   b. For each feature:
      - For each node: find best threshold
      - Calculate total improvement
   c. Choose feature with max total improvement
   d. Split all nodes with their optimal thresholds
   e. current_level = all child nodes
4. Make remaining nodes leaves
```

## Leaf Node Prediction

### Classification

```python
def get_leaf_value(y):
    # Return most common class
    class_counts = Counter(y)
    most_common = class_counts.most_common(1)[0][0]
    return most_common
```

### Regression

```python
def get_leaf_value(y):
    # Return mean (for MSE) or median (for MAE)
    if criterion == 'mse':
        return np.mean(y)
    else:  # mae
        return np.median(y)
```

## Prediction

```python
def predict_sample(x, node):
    if node.is_leaf():
        return node.value
    
    # Navigate based on feature and threshold
    feature_idx = node.split_info['feature']
    threshold = node.split_info['threshold']
    
    if x[feature_idx] <= threshold:
        return predict_sample(x, node.left)
    else:
        return predict_sample(x, node.right)
```

## Stopping Criteria

Tree growth stops when:

1. **Max depth reached**: `depth >= max_depth`
2. **Insufficient samples**: `n_samples < min_samples_split`
3. **Pure node**: All samples have same target value
4. **No valid split**: Cannot find split satisfying constraints

## Comparison: Greedy vs Optimal

| Aspect | Greedy | Optimal |
|--------|--------|---------|
| Feature selection | Based on single node | Based on all nodes |
| Time complexity | O(n×m×d×log n) | O(n×m²×d×log n) |
| Optimality | Local | Global (at each depth) |
| Speed | Fast | Slower |
| Accuracy | Good | Potentially better |
| Use case | Large datasets | Moderate datasets |

## Implementation Notes

### Data Structures

```python
class Node:
    indices: List[int]           # Sample indices
    split_info: Dict             # {'feature': int, 'threshold': float}
    left: Node                   # Left child
    right: Node                  # Right child
    value: Union[int, float]     # Prediction (for leaves)
    class_counts: Dict[int, int] # For probability predictions
```

### Key Attributes

```python
class RestrictedDecisionTree:
    depth_features_: Dict[int, int]  # {depth: feature_idx}
    n_features_: int
    classes_: np.ndarray             # (classification only)
    cat_encodings_: Dict             # Categorical encodings
```

## Advantages and Limitations

### Advantages

✅ **Interpretable**: Easy to see which features matter at each depth  
✅ **Structured**: Depth-based feature selection provides natural organization  
✅ **Flexible**: More adaptive than oblivious trees  
✅ **Regularized**: Constraint helps prevent overfitting  
✅ **Fast inference**: No different from standard trees  

### Limitations

❌ **Training time**: Optimal variant slower than standard trees  
❌ **Not always optimal**: Greedy variant may miss global optimum  
❌ **Depth-based**: Feature must be useful across all nodes at depth  
❌ **Fixed structure**: Less flexible than unrestricted trees  

## References and Related Work

- **Oblivious Decision Trees**: Used in CatBoost, complete depth-based structure
- **Standard CART**: Unrestricted feature and threshold selection
- **Feature-based Partitioning**: Related to axis-aligned splits in decision trees
- **Regularization**: Structural constraints as implicit regularization

---

This document provides the theoretical foundation for understanding and implementing Restricted Decision Trees.
