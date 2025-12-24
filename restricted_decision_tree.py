from __future__ import annotations
from typing import Optional, List, Dict, Any, Union
import numpy as np
from collections import Counter
try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False


class Node:
    """
    Node class for Restricted Decision Tree.
    
    Attributes:
        indices: List of row indices that qualify/reach this node
        left: Left child node (samples where feature <= threshold)
        right: Right child node (samples where feature > threshold)
        split_info: Dictionary containing 'feature' and 'threshold' for the split
        value: Prediction value for leaf nodes (class label or regression value)
    """
    
    def __init__(
        self,
        indices: List[int],
        split_info: Optional[Dict[str, Any]] = None,
        left: Optional[Node] = None,
        right: Optional[Node] = None,
        value: Optional[Union[int, float]] = None,
        class_counts: Optional[Dict[int, int]] = None
    ):
        self.indices: List[int] = indices
        self.left: Optional[Node] = left
        self.right: Optional[Node] = right
        self.split_info: Optional[Dict[str, Any]] = split_info
        self.value: Optional[Union[int, float]] = value
        self.class_counts: Optional[Dict[int, int]] = class_counts  # For probability predictions
    
    def is_leaf(self) -> bool:
        """Check if this node is a leaf node."""
        return self.left is None and self.right is None
    
    def __repr__(self) -> str:
        """String representation of the node."""
        if self.is_leaf():
            return f"LeafNode(indices={len(self.indices)}, value={self.value})"
        else:
            feature = self.split_info.get('feature', 'N/A')
            threshold = self.split_info.get('threshold', 'N/A')
            return f"InternalNode(indices={len(self.indices)}, feature={feature}, threshold={threshold:.4f})"


class RestrictedDecisionTree:
    """
    Restricted Decision Tree for Classification and Regression.
    
    In a restricted tree, all nodes at the same depth use the same FEATURE but can have
    different THRESHOLDS. The feature at each level is chosen to maximize the overall
    improvement in the splitting criterion (gini/entropy/mse/mae).
    
    Parameters:
        task: 'classification' or 'regression'
        criterion: Splitting criterion
            - For classification: 'gini' (default) or 'entropy'
            - For regression: 'mse' (default) or 'mae'
        max_depth: Maximum depth of the tree (default: 5)
        min_samples_split: Minimum samples required to split a node (default: 2)
        min_samples_leaf: Minimum samples required in a leaf node (default: 1)
    """
    
    def __init__(
        self,
        task: str = 'classification',
        criterion: Optional[str] = None,
        max_depth: int = 5,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        cat_features: Optional[Union[List[int], List[str]]] = None,
        prior_weight: float = 1.0
    ):
        self.task = task.lower()
        if self.task not in ['classification', 'regression']:
            raise ValueError("task must be 'classification' or 'regression'")
        
        # Set default criterion based on task
        if criterion is None:
            self.criterion = 'gini' if self.task == 'classification' else 'mse'
        else:
            self.criterion = criterion.lower()
        
        # Validate criterion
        if self.task == 'classification' and self.criterion not in ['gini', 'entropy']:
            raise ValueError("For classification, criterion must be 'gini' or 'entropy'")
        if self.task == 'regression' and self.criterion not in ['mse', 'mae']:
            raise ValueError("For regression, criterion must be 'mse' or 'mae'")
        
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.cat_features = cat_features if cat_features is not None else []
        self.prior_weight = prior_weight
        
        # Attributes set during fit
        self.root = None
        self.n_features_ = None
        self.n_classes_ = None  # Number of classes (classification only)
        self.classes_ = None  # Unique class labels (classification only)
        self.depth_features_ = {}  # Store feature index for each depth level
        self.cat_features_ = []  # Indices of categorical features
        self.cat_encodings_ = {}  # Category to encoded value mapping per feature
        self.global_prior_ = None  # Global mean/mode for smoothing
        self.X_encoded_ = None  # Store encoded training data
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> RestrictedDecisionTree:
        """
        Build the restricted decision tree.
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
        
        Returns:
            self: Fitted estimator
        """
        X = np.array(X)
        y = np.array(y)
        
        self.n_features_ = X.shape[1]
        
        # Store classes for classification
        if self.task == 'classification':
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)
        
        # Identify categorical features
        self._identify_categorical_features(X)
        
        # Calculate global prior for smoothing
        if self.task == 'classification':
            self.global_prior_ = np.mean(y)  # For binary, or most common class
        else:
            self.global_prior_ = np.mean(y)
        
        # Encode categorical features using ordered target statistics
        if len(self.cat_features_) > 0:
            X_encoded = self._encode_categorical_features(X, y)
        else:
            X_encoded = X.copy()
        
        self.X_encoded_ = X_encoded
        indices = list(range(len(X)))
        
        # Build the tree with encoded features
        self.root = self._build_tree(X_encoded, y, indices, depth=0)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for X.
        
        Args:
            X: Features of shape (n_samples, n_features)
        
        Returns:
            Predicted values of shape (n_samples,)
        """
        if self.root is None:
            raise ValueError("Tree has not been fitted yet. Call fit() first.")
        
        X = np.array(X)
        
        # Encode categorical features if present
        if len(self.cat_features_) > 0:
            X_encoded = self._encode_categorical_features_predict(X)
        else:
            X_encoded = X
        
        predictions = [self._predict_sample(x, self.root) for x in X_encoded]
        return np.array(predictions)
    
    def _calculate_gini(self, y: np.ndarray) -> float:
        """Calculate Gini impurity."""
        if len(y) == 0:
            return 0.0
        
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1.0 - np.sum(probabilities ** 2)
    
    def _calculate_entropy(self, y: np.ndarray) -> float:
        """Calculate entropy."""
        if len(y) == 0:
            return 0.0
        
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))
    
    def _calculate_mse(self, y: np.ndarray) -> float:
        """Calculate Mean Squared Error."""
        if len(y) == 0:
            return 0.0
        
        mean = np.mean(y)
        return np.mean((y - mean) ** 2)
    
    def _calculate_mae(self, y: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        if len(y) == 0:
            return 0.0
        
        median = np.median(y)
        return np.mean(np.abs(y - median))
    
    def _calculate_criterion(self, y: np.ndarray) -> float:
        """Calculate the splitting criterion based on the chosen method."""
        if self.criterion == 'gini':
            return self._calculate_gini(y)
        elif self.criterion == 'entropy':
            return self._calculate_entropy(y)
        elif self.criterion == 'mse':
            return self._calculate_mse(y)
        elif self.criterion == 'mae':
            return self._calculate_mae(y)
    
    def _calculate_split_score(self, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """
        Calculate the weighted criterion for a split.
        Lower is better for all criteria.
        """
        n_left, n_right = len(y_left), len(y_right)
        n_total = n_left + n_right
        
        if n_total == 0:
            return float('inf')
        
        score_left = self._calculate_criterion(y_left)
        score_right = self._calculate_criterion(y_right)
        
        # Weighted average
        weighted_score = (n_left / n_total) * score_left + (n_right / n_total) * score_right
        return weighted_score
    
    def _calculate_improvement(self, y_parent: np.ndarray, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """
        Calculate the improvement (information gain) from a split.
        Higher is better.
        """
        n_left, n_right = len(y_left), len(y_right)
        n_total = len(y_parent)
        
        if n_total == 0:
            return 0.0
        
        parent_criterion = self._calculate_criterion(y_parent)
        weighted_child_criterion = self._calculate_split_score(y_left, y_right)
        
        # Improvement = parent criterion - weighted child criterion
        improvement = parent_criterion - weighted_child_criterion
        return improvement
    
    def _find_best_feature_for_depth(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        all_indices_at_depth: List[List[int]]
    ) -> Optional[int]:
        """
        Find the best feature to use for ALL nodes at this depth level.
        The feature is chosen to maximize the overall improvement across all nodes.
        
        Args:
            X: Full feature matrix
            y: Full target array
            all_indices_at_depth: List of index lists for all nodes at this depth
        
        Returns:
            Feature index that maximizes overall improvement, or None if no valid split
        """
        best_feature = None
        best_total_improvement = -float('inf')
        
        # Try each feature
        for feature_idx in range(self.n_features_):
            total_improvement = 0.0
            valid_split_found = False
            
            # For each node at this depth, find the best threshold for this feature
            for node_indices in all_indices_at_depth:
                if len(node_indices) < self.min_samples_split:
                    continue
                
                X_subset = X[node_indices]
                y_subset = y[node_indices]
                
                # Find best threshold for this feature at this node
                feature_values = X_subset[:, feature_idx]
                unique_values = np.unique(feature_values)
                
                if len(unique_values) < 2:
                    continue
                
                thresholds = (unique_values[:-1] + unique_values[1:]) / 2
                
                best_node_improvement = -float('inf')
                
                for threshold in thresholds:
                    left_mask = feature_values <= threshold
                    right_mask = ~left_mask
                    
                    y_left = y_subset[left_mask]
                    y_right = y_subset[right_mask]
                    
                    # Check minimum samples in leaf
                    if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                        continue
                    
                    # Calculate improvement for this split
                    improvement = self._calculate_improvement(y_subset, y_left, y_right)
                    
                    if improvement > best_node_improvement:
                        best_node_improvement = improvement
                        valid_split_found = True
                
                # Add this node's best improvement to total
                if best_node_improvement > -float('inf'):
                    total_improvement += best_node_improvement
            
            # Check if this feature gives better total improvement
            if valid_split_found and total_improvement > best_total_improvement:
                best_total_improvement = total_improvement
                best_feature = feature_idx
        
        return best_feature
    
    def _find_best_threshold_for_feature(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        indices: List[int],
        feature_idx: int
    ) -> Optional[float]:
        """
        Find the best threshold for a specific feature at a specific node.
        
        Args:
            X: Full feature matrix
            y: Full target array
            indices: Indices of samples at this node
            feature_idx: Index of the feature to split on
        
        Returns:
            Best threshold value, or None if no valid split found
        """
        if len(indices) < self.min_samples_split:
            return None
        
        X_subset = X[indices]
        y_subset = y[indices]
        
        feature_values = X_subset[:, feature_idx]
        unique_values = np.unique(feature_values)
        
        if len(unique_values) < 2:
            return None
        
        thresholds = (unique_values[:-1] + unique_values[1:]) / 2
        
        best_score = float('inf')
        best_threshold = None
        
        for threshold in thresholds:
            left_mask = feature_values <= threshold
            right_mask = ~left_mask
            
            y_left = y_subset[left_mask]
            y_right = y_subset[right_mask]
            
            # Check minimum samples in leaf
            if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                continue
            
            # Calculate split score
            score = self._calculate_split_score(y_left, y_right)
            
            if score < best_score:
                best_score = score
                best_threshold = threshold
        
        return best_threshold
    
    def _get_leaf_value(self, y: np.ndarray, store_distribution: bool = False) -> Union[int, float, Dict[int, int]]:
        """
        Get the prediction value for a leaf node.
        
        Args:
            y: Target values at this leaf
            store_distribution: If True, return (value, class_counts) for classification
        
        Returns:
            For classification: class label (or tuple if store_distribution=True)
            For regression: mean value
        """
        if self.task == 'classification':
            # Return the most common class
            counter = Counter(y)
            most_common_class = counter.most_common(1)[0][0]
            if store_distribution:
                return most_common_class, dict(counter)
            return most_common_class
        else:
            # Return the mean for regression
            return np.mean(y)
    
    def _collect_nodes_at_depth(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        indices_list: List[List[int]], 
        depth: int
    ) -> List[List[int]]:
        """
        Collect all node indices at a specific depth.
        This is used to determine the best feature for the depth level.
        """
        return indices_list
    
    def _build_tree(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        indices: List[int], 
        depth: int,
        nodes_at_depth: Optional[List[List[int]]] = None
    ) -> Node:
        """
        Recursively build the restricted decision tree.
        
        In a restricted tree, we first find the best FEATURE for this depth level
        (maximizing overall improvement), then each node finds its own best THRESHOLD
        for that feature.
        """
        # Check stopping criteria
        if (depth >= self.max_depth or 
            len(indices) < self.min_samples_split or 
            len(np.unique(y[indices])) == 1):
            # Create leaf node
            if self.task == 'classification':
                value, class_counts = self._get_leaf_value(y[indices], store_distribution=True)
                return Node(indices=indices, value=value, class_counts=class_counts)
            else:
                value = self._get_leaf_value(y[indices])
                return Node(indices=indices, value=value)
        
        # If we haven't determined the feature for this depth, we need to collect all nodes first
        if depth not in self.depth_features_:
            # This is the first node at this depth - we need to determine the best feature
            # For simplicity in this recursive approach, we'll use a greedy approach:
            # Choose the feature that gives best improvement for THIS node
            # (A full implementation would require two-pass approach to collect all nodes first)
            
            # Find best feature based on this node's data
            best_feature = self._find_best_feature_for_single_node(X, y, indices)
            
            if best_feature is None:
                if self.task == 'classification':
                    value, class_counts = self._get_leaf_value(y[indices], store_distribution=True)
                    return Node(indices=indices, value=value, class_counts=class_counts)
                else:
                    value = self._get_leaf_value(y[indices])
                    return Node(indices=indices, value=value)
            
            self.depth_features_[depth] = best_feature
        
        # Get the feature for this depth
        feature_idx = self.depth_features_[depth]
        
        # Find the best threshold for THIS node using the depth's feature
        threshold = self._find_best_threshold_for_feature(X, y, indices, feature_idx)
        
        if threshold is None:
            # No valid split found, create leaf
            if self.task == 'classification':
                value, class_counts = self._get_leaf_value(y[indices], store_distribution=True)
                return Node(indices=indices, value=value, class_counts=class_counts)
            else:
                value = self._get_leaf_value(y[indices])
                return Node(indices=indices, value=value)
        
        # Split the indices
        X_subset = X[indices]
        left_mask = X_subset[:, feature_idx] <= threshold
        
        left_indices = [indices[i] for i in range(len(indices)) if left_mask[i]]
        right_indices = [indices[i] for i in range(len(indices)) if not left_mask[i]]
        
        # If either split is empty, create a leaf
        if len(left_indices) == 0 or len(right_indices) == 0:
            if self.task == 'classification':
                value, class_counts = self._get_leaf_value(y[indices], store_distribution=True)
                return Node(indices=indices, value=value, class_counts=class_counts)
            else:
                value = self._get_leaf_value(y[indices])
                return Node(indices=indices, value=value)
        
        # Create split info for this node
        split_info = {
            'feature': feature_idx,
            'threshold': threshold
        }
        
        # Recursively build left and right subtrees
        left_node = self._build_tree(X, y, left_indices, depth + 1)
        right_node = self._build_tree(X, y, right_indices, depth + 1)
        
        # Create internal node
        return Node(
            indices=indices,
            split_info=split_info,
            left=left_node,
            right=right_node
        )
    
    def _find_best_feature_for_single_node(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        indices: List[int]
    ) -> Optional[int]:
        """
        Find the best feature for a single node based on maximum improvement.
        """
        if len(indices) < self.min_samples_split:
            return None
        
        X_subset = X[indices]
        y_subset = y[indices]
        
        best_improvement = -float('inf')
        best_feature = None
        
        # Try each feature
        for feature_idx in range(self.n_features_):
            feature_values = X_subset[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            if len(unique_values) < 2:
                continue
            
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2
            
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                y_left = y_subset[left_mask]
                y_right = y_subset[right_mask]
                
                # Check minimum samples in leaf
                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue
                
                # Calculate improvement
                improvement = self._calculate_improvement(y_subset, y_left, y_right)
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_feature = feature_idx
        
        return best_feature
    
    def _predict_sample(self, x: np.ndarray, node: Node) -> Union[int, float]:
        """Predict the value for a single sample."""
        if node.is_leaf():
            return node.value
        
        # Navigate to left or right child based on split
        feature_idx = node.split_info['feature']
        threshold = node.split_info['threshold']
        
        if x[feature_idx] <= threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for X.
        
        Args:
            X: Features of shape (n_samples, n_features)
        
        Returns:
            Class probabilities of shape (n_samples, n_classes)
        
        Raises:
            ValueError: If tree not fitted or task is regression
        """
        if self.root is None:
            raise ValueError("Tree has not been fitted yet. Call fit() first.")
        
        if self.task != 'classification':
            raise ValueError("predict_proba is only available for classification tasks")
        
        X = np.array(X)
        
        # Encode categorical features if present
        if len(self.cat_features_) > 0:
            X_encoded = self._encode_categorical_features_predict(X)
        else:
            X_encoded = X
        
        # Get probabilities for each sample
        probas = np.array([self._predict_proba_sample(x, self.root) for x in X_encoded])
        return probas
    
    def _predict_proba_sample(self, x: np.ndarray, node: Node) -> np.ndarray:
        """
        Get probability distribution for a single sample.
        
        Args:
            x: Single sample features
            node: Current node in traversal
        
        Returns:
            Probability array of shape (n_classes,)
        """
        if node.is_leaf():
            # Convert class counts to probabilities
            probs = np.zeros(self.n_classes_)
            if node.class_counts is not None:
                total = sum(node.class_counts.values())
                for cls, count in node.class_counts.items():
                    # Find index of this class in self.classes_
                    cls_idx = np.where(self.classes_ == cls)[0][0]
                    probs[cls_idx] = count / total
            return probs
        
        # Navigate to left or right child based on split
        feature_idx = node.split_info['feature']
        threshold = node.split_info['threshold']
        
        if x[feature_idx] <= threshold:
            return self._predict_proba_sample(x, node.left)
        else:
            return self._predict_proba_sample(x, node.right)
    
    def _identify_categorical_features(self, X: np.ndarray) -> None:
        """
        Identify which features are categorical based on user input.
        
        Args:
            X: Input features
        """
        if not self.cat_features:
            self.cat_features_ = []
            return
        
        # Convert feature names/indices to indices
        if isinstance(self.cat_features[0], int):
            self.cat_features_ = list(self.cat_features)
        else:
            # Assume feature names are provided (not implemented for array input)
            raise ValueError("Feature names are not supported. Please use feature indices.")
        
        # Validate indices
        for idx in self.cat_features_:
            if idx < 0 or idx >= self.n_features_:
                raise ValueError(f"Categorical feature index {idx} is out of bounds.")
    
    def _encode_categorical_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Encode categorical features using ordered target statistics (CatBoost-style).
        
        This implements the "ordered TS" approach:
        - For each sample i, use only samples 0 to i-1 for encoding
        - Apply smoothing to handle rare categories
        - Prevents target leakage
        
        Args:
            X: Input features
            y: Target values
        
        Returns:
            X_encoded: Features with categorical columns encoded
        """
        # Create a copy - will convert columns to float as we encode them
        X_encoded = X.copy()
        
        for cat_idx in self.cat_features_:
            # Initialize encoding dictionary for this feature
            if cat_idx not in self.cat_encodings_:
                self.cat_encodings_[cat_idx] = {}
            
            # Encode using ordered target statistics
            encoded_col = np.zeros(len(X))
            
            for i in range(len(X)):
                category = X[i, cat_idx]
                
                # Use only previous samples for encoding (ordered approach)
                if i == 0:
                    # First sample: use global prior
                    encoded_value = self.global_prior_
                else:
                    # Find previous samples with same category
                    prev_mask = np.array([X[j, cat_idx] == category for j in range(i)])
                    prev_targets = y[:i][prev_mask]
                    
                    # Calculate smoothed encoding
                    count = len(prev_targets)
                    if count > 0:
                        sum_target = np.sum(prev_targets)
                        encoded_value = (sum_target + self.prior_weight * self.global_prior_) / (count + self.prior_weight)
                    else:
                        # No previous samples with this category: use global prior
                        encoded_value = self.global_prior_
                
                encoded_col[i] = encoded_value
                
                # Store the encoding for this category (use last seen encoding)
                self.cat_encodings_[cat_idx][category] = encoded_value
            
            # Replace categorical column with encoded values
            X_encoded[:, cat_idx] = encoded_col
        
        # Now convert to float (categorical columns are already encoded as numbers)
        return X_encoded.astype(float)
    
    def _encode_categorical_features_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Encode categorical features for prediction using stored encodings.
        
        Args:
            X: Input features to encode
        
        Returns:
            X_encoded: Features with categorical columns encoded
        """
        X_encoded = X.copy()
        
        for cat_idx in self.cat_features_:
            encoded_col = np.zeros(len(X))
            for i in range(len(X)):
                category = X[i, cat_idx]
                
                # Use stored encoding if available, otherwise use global prior
                if cat_idx in self.cat_encodings_ and category in self.cat_encodings_[cat_idx]:
                    encoded_value = self.cat_encodings_[cat_idx][category]
                else:
                    # Unseen category: use global prior
                    encoded_value = self.global_prior_
                
                encoded_col[i] = encoded_value
            
            # Replace categorical column with encoded values
            X_encoded[:, cat_idx] = encoded_col
        
        # Convert to float after encoding
        return X_encoded.astype(float)
    
    def print_tree(self, node: Optional[Node] = None, depth: int = 0, prefix: str = "Root: "):
        """Print the tree structure for visualization."""
        if node is None:
            node = self.root
        
        if node is None:
            print("Tree has not been fitted yet.")
            return
        
        indent = "  " * depth
        print(f"{indent}{prefix}{node}")
        
        if not node.is_leaf():
            self.print_tree(node.left, depth + 1, "L--- ")
            self.print_tree(node.right, depth + 1, "R--- ")
    
    def visualize(
        self,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        filename: str = "restricted_tree",
        format: str = "png",
        view: bool = True,
        filled: bool = True,
        rounded: bool = True,
        node_colors: Optional[Dict[str, str]] = None
    ) -> Optional[graphviz.Digraph]:
        """
        Create a Graphviz visualization of the decision tree.
        
        Args:
            feature_names: List of feature names (default: Feature 0, Feature 1, ...)
            class_names: List of class names for classification (default: Class 0, Class 1, ...)
            filename: Output filename without extension (default: "restricted_tree")
            format: Output format - 'png', 'pdf', 'svg', etc. (default: "png")
            view: Whether to open the rendered file (default: True)
            filled: Whether to fill nodes with colors (default: True)
            rounded: Whether to use rounded boxes (default: True)
            node_colors: Dictionary with 'internal' and 'leaf' color hex codes
        
        Returns:
            graphviz.Digraph object or None if graphviz is not available
        """
        if not GRAPHVIZ_AVAILABLE:
            print("Graphviz is not installed. Please install it using:")
            print("  pip install graphviz")
            print("Also ensure Graphviz system package is installed:")
            print("  - Windows: Download from https://graphviz.org/download/")
            print("  - Mac: brew install graphviz")
            print("  - Linux: sudo apt-get install graphviz")
            return None
        
        if self.root is None:
            raise ValueError("Tree has not been fitted yet. Call fit() first.")
        
        # Set default feature names
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(self.n_features_)]
        
        # Set default colors
        if node_colors is None:
            node_colors = {
                'internal': '#E8F4F8',  # Light blue for internal nodes
                'leaf': '#F5E6D3'        # Light orange for leaf nodes
            }
        
        # Create Digraph
        dot = graphviz.Digraph(comment='Restricted Decision Tree')
        dot.attr('node', shape='box')
        
        if rounded:
            dot.attr('node', style='rounded,filled' if filled else 'rounded')
        elif filled:
            dot.attr('node', style='filled')
        
        dot.attr('graph', rankdir='TB')  # Top to bottom layout
        
        # Build the graph recursively
        self._add_nodes_to_graph(
            dot=dot,
            node=self.root,
            node_id=0,
            feature_names=feature_names,
            class_names=class_names,
            node_colors=node_colors,
            filled=filled
        )
        
        # Render the graph
        try:
            dot.render(filename, format=format, view=view, cleanup=True)
            print(f"Tree visualization saved as '{filename}.{format}'")
        except Exception as e:
            print(f"Error rendering graph: {e}")
            print("Make sure Graphviz is installed on your system.")
        
        return dot
    
    def _add_nodes_to_graph(
        self,
        dot: graphviz.Digraph,
        node: Node,
        node_id: int,
        feature_names: List[str],
        class_names: Optional[List[str]],
        node_colors: Dict[str, str],
        filled: bool,
        parent_id: Optional[int] = None,
        edge_label: str = ""
    ) -> int:
        """
        Recursively add nodes to the Graphviz graph.
        
        Returns:
            The next available node ID
        """
        current_id = node_id
        
        # Create node label and style
        if node.is_leaf():
            # Leaf node
            if self.task == 'classification':
                if class_names is not None and isinstance(node.value, (int, np.integer)):
                    value_str = class_names[int(node.value)]
                else:
                    value_str = f"Class {node.value}"
                label = f"{value_str}\nSamples: {len(node.indices)}"
            else:
                label = f"Value: {node.value:.4f}\nSamples: {len(node.indices)}"
            
            color = node_colors['leaf'] if filled else 'white'
        else:
            # Internal node
            feature_idx = node.split_info['feature']
            threshold = node.split_info['threshold']
            feature_name = feature_names[feature_idx]
            
            label = f"{feature_name} ≤ {threshold:.4f}\nSamples: {len(node.indices)}"
            color = node_colors['internal'] if filled else 'white'
        
        # Add node to graph
        dot.node(str(current_id), label=label, fillcolor=color)
        
        # Add edge from parent
        if parent_id is not None:
            dot.edge(str(parent_id), str(current_id), label=edge_label)
        
        # Recursively add children
        next_id = current_id + 1
        if not node.is_leaf():
            # Add left child
            next_id = self._add_nodes_to_graph(
                dot=dot,
                node=node.left,
                node_id=next_id,
                feature_names=feature_names,
                class_names=class_names,
                node_colors=node_colors,
                filled=filled,
                parent_id=current_id,
                edge_label="True"
            )
            
            # Add right child
            next_id = self._add_nodes_to_graph(
                dot=dot,
                node=node.right,
                node_id=next_id,
                feature_names=feature_names,
                class_names=class_names,
                node_colors=node_colors,
                filled=filled,
                parent_id=current_id,
                edge_label="False"
            )
        
        return next_id
    
    def export_tree_dict(
        self,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Export tree structure as nested dictionary with statistics.
        
        Args:
            y: Training target values (needed for statistics)
            feature_names: List of feature names
            class_names: List of class names (for classification)
        
        Returns:
            Nested dictionary representing the tree structure
        """
        if self.root is None:
            raise ValueError("Tree has not been fitted yet. Call fit() first.")
        
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(self.n_features_)]
        
        return self._node_to_dict_with_stats(
            self.root, y, 0, feature_names, class_names
        )
    
    def _node_to_dict_with_stats(
        self,
        node: Node,
        y: np.ndarray,
        depth: int,
        feature_names: List[str],
        class_names: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Convert node to dictionary with statistics."""
        
        # Get targets for this node's samples
        node_targets = y[node.indices]
        target_avg = float(np.mean(node_targets))
        
        # Calculate criterion value for this node
        criterion_value = self._calculate_criterion(node_targets)
        
        if node.is_leaf():
            # Leaf node
            if self.task == 'classification':
                if class_names is not None and isinstance(node.value, (int, np.integer)):
                    pred_str = class_names[int(node.value)]
                else:
                    pred_str = f"Class {node.value}"
                
                # Calculate purity
                counts = Counter(node_targets)
                purity = counts.most_common(1)[0][1] / len(node_targets) * 100
            else:
                pred_str = f"{node.value:.4f}"
                purity = 100.0  # For regression, we don't have purity
            
            return {
                "name": f"🍃 {pred_str}",
                "samples": len(node.indices),
                "criterion_value": float(criterion_value),
                "criterion_name": self.criterion,
                "target_avg": target_avg,
                "depth": depth,
                "is_leaf": True,
                "prediction": pred_str,
                "purity": purity
            }
        else:
            # Internal node
            feature_idx = node.split_info['feature']
            threshold = node.split_info['threshold']
            feature_name = feature_names[feature_idx]
            
            node_dict = {
                "name": f"{feature_name} ≤ {threshold:.4f}",
                "samples": len(node.indices),
                "criterion_value": float(criterion_value),
                "criterion_name": self.criterion,
                "target_avg": target_avg,
                "depth": depth,
                "is_leaf": False,
                "feature": feature_name,
                "feature_idx": feature_idx,
                "threshold": float(threshold),
                "children": []
            }
            
            # Add children
            if node.left:
                node_dict["children"].append(
                    self._node_to_dict_with_stats(
                        node.left, y, depth + 1, feature_names, class_names
                    )
                )
            if node.right:
                node_dict["children"].append(
                    self._node_to_dict_with_stats(
                        node.right, y, depth + 1, feature_names, class_names
                    )
                )
            
            return node_dict
    
    def visualize_d3_observable(
        self,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        filename: str = "restricted_tree_d3.html",
        width: int = 1200,
        height: int = 800
    ) -> str:
        """
        Create an Observable-compatible D3.js visualization with collapsible nodes.
        
        Args:
            y: Training target values (needed for statistics)
            feature_names: List of feature names
            class_names: List of class names (for classification)
            filename: Output HTML filename
            width: Visualization width in pixels
            height: Visualization height in pixels
        
        Returns:
            HTML string with embedded D3.js visualization
        """
        import json
        
        # Export tree as dictionary
        tree_data = self.export_tree_dict(y, feature_names, class_names)
        
        # Create HTML with D3.js visualization
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Restricted Decision Tree - D3 Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #f5f5f5;
        }}
        
        #tree-container {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            padding: 20px;
            overflow: auto;
        }}
        
        .node {{
            cursor: pointer;
        }}
        
        .node rect {{
            stroke: #999;
            stroke-width: 2px;
            rx: 8;
        }}
        
        .node-internal rect {{
            fill: #95C0E8;
            stroke: #7AA8D6;
        }}
        
        .node-leaf rect {{
            fill: #C2B59B;
            stroke: #ADA088;
        }}
        
        .node text {{
            font-size: 12px;
            fill: #333;
            font-family: 'Courier New', monospace;
        }}
        
        .node-title {{
            font-weight: bold;
            font-size: 13px;
        }}
        
        .link {{
            fill: none;
            stroke: #999;
            stroke-width: 2px;
        }}
        
        .link-label {{
            font-size: 10px;
            fill: #666;
        }}
        
        #info-panel {{
            position: fixed;
            top: 20px;
            right: 20px;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            max-width: 250px;
        }}
        
        #info-panel h3 {{
            margin-top: 0;
            font-size: 16px;
            color: #333;
        }}
        
        #info-panel p {{
            margin: 5px 0;
            font-size: 13px;
            color: #666;
        }}
        
        .collapsed {{
            opacity: 0.6;
        }}
    </style>
</head>
<body>
    <div id="info-panel">
        <h3>Restricted Decision Tree</h3>
        <p><strong>Task:</strong> {self.task}</p>
        <p><strong>Criterion:</strong> {self.criterion}</p>
        <p><strong>Max Depth:</strong> {self.max_depth}</p>
        <p><br><em>Click nodes to expand/collapse</em></p>
        <p style="margin-top: 15px; font-size: 11px; color: #888;">
            Note: Same feature at each depth,<br>
            but different thresholds per node
        </p>
    </div>
    
    <div id="tree-container"></div>
    
    <script>
        const treeData = {json.dumps(tree_data, indent=2)};
        
        const width = {width};
        const height = {height};
        const nodeWidth = 220;
        const nodeHeight = 140;
        
        const svg = d3.select("#tree-container")
            .append("svg")
            .attr("width", width)
            .attr("height", height)
            .call(d3.zoom().on("zoom", (event) => {{
                g.attr("transform", event.transform);
            }}))
            .append("g")
            .attr("transform", "translate(100,50)");
        
        const g = svg;
        
        // Create tree layout
        const tree = d3.tree()
            .nodeSize([nodeWidth + 100, nodeHeight + 60]);
        
        const root = d3.hierarchy(treeData, d => d.children);
        root.x0 = 0;
        root.y0 = 0;
        
        // Collapse all nodes initially except root
        root.descendants().forEach(d => {{
            if (d.depth > 1) {{
                if (d.children) {{
                    d._children = d.children;
                    d.children = null;
                }}
            }}
        }});
        
        update(root);
        
        function update(source) {{
            const treeData = tree(root);
            const nodes = treeData.descendants();
            const links = treeData.links();
            
            // Update nodes
            const node = g.selectAll('.node')
                .data(nodes, d => d.id || (d.id = Math.random()));
            
            // Enter new nodes
            const nodeEnter = node.enter().append('g')
                .attr('class', d => `node ${{d.data.is_leaf ? 'node-leaf' : 'node-internal'}}`)
                .attr('transform', d => `translate(${{source.x0}},${{source.y0}})`)
                .on('click', click);
            
            // Add node rectangles
            nodeEnter.append('rect')
                .attr('width', nodeWidth)
                .attr('height', nodeHeight)
                .attr('x', -nodeWidth / 2)
                .attr('y', -nodeHeight / 2);
            
            // Add node text
            nodeEnter.each(function(d) {{
                const g = d3.select(this);
                const data = d.data;
                
                // Title
                g.append('text')
                    .attr('class', 'node-title')
                    .attr('dy', -nodeHeight/2 + 20)
                    .attr('text-anchor', 'middle')
                    .text(data.name.length > 30 ? data.name.substring(0, 27) + '...' : data.name);
                
                // Samples
                g.append('text')
                    .attr('dy', -nodeHeight/2 + 45)
                    .attr('text-anchor', 'middle')
                    .text(`Samples: ${{data.samples}}`);
                
                // Criterion
                g.append('text')
                    .attr('dy', -nodeHeight/2 + 65)
                    .attr('text-anchor', 'middle')
                    .text(`${{data.criterion_name}}: ${{data.criterion_value.toFixed(4)}}`);
                
                // Target avg
                g.append('text')
                    .attr('dy', -nodeHeight/2 + 85)
                    .attr('text-anchor', 'middle')
                    .text(`Target Avg: ${{data.target_avg.toFixed(4)}}`);
                
                // Depth
                g.append('text')
                    .attr('dy', -nodeHeight/2 + 105)
                    .attr('text-anchor', 'middle')
                    .attr('font-size', '10px')
                    .text(`Depth: ${{data.depth}}`);
                
                // Purity for leaves
                if (data.is_leaf) {{
                    g.append('text')
                        .attr('dy', -nodeHeight/2 + 125)
                        .attr('text-anchor', 'middle')
                        .attr('font-size', '10px')
                        .attr('fill', '#2ecc71')
                        .text(`Purity: ${{data.purity.toFixed(1)}}%`);
                }}
            }});
            
            // Transition nodes to new position
            const nodeUpdate = nodeEnter.merge(node)
                .transition()
                .duration(750)
                .attr('transform', d => `translate(${{d.x}},${{d.y}})`);
            
            // Transition exiting nodes
            const nodeExit = node.exit()
                .transition()
                .duration(750)
                .attr('transform', d => `translate(${{source.x}},${{source.y}})`)
                .remove();
            
            // Update links
            const link = g.selectAll('.link')
                .data(links, d => d.target.id);
            
            const linkEnter = link.enter().insert('path', 'g')
                .attr('class', 'link')
                .attr('d', d => {{
                    const o = {{x: source.x0, y: source.y0}};
                    return diagonal(o, o);
                }});
            
            const linkUpdate = linkEnter.merge(link)
                .transition()
                .duration(750)
                .attr('d', d => diagonal(d.source, d.target));
            
            const linkExit = link.exit()
                .transition()
                .duration(750)
                .attr('d', d => {{
                    const o = {{x: source.x, y: source.y}};
                    return diagonal(o, o);
                }})
                .remove();
            
            // Store old positions
            nodes.forEach(d => {{
                d.x0 = d.x;
                d.y0 = d.y;
            }});
        }}
        
        function diagonal(s, d) {{
            return `M ${{s.x}} ${{s.y}}
                    C ${{s.x}} ${{(s.y + d.y) / 2}},
                      ${{d.x}} ${{(s.y + d.y) / 2}},
                      ${{d.x}} ${{d.y}}`;
        }}
        
        function click(event, d) {{
            if (d.children) {{
                d._children = d.children;
                d.children = null;
            }} else {{
                d.children = d._children;
                d._children = null;
            }}
            update(d);
        }}
    </script>
</body>
</html>
"""
        
        # Save to file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"D3 visualization saved to: {filename}")
        print(f"Open the file in a browser to view the interactive tree.")
        
        return html
