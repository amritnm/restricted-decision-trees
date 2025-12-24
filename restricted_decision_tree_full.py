from __future__ import annotations
from typing import Optional, List, Dict, Any, Union, Tuple
import numpy as np
from collections import Counter
try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False


class Node:
    """
    Node class for Restricted Decision Tree (Full/Optimal variant).
    
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


class RestrictedDecisionTreeFull:
    """
    Restricted Decision Tree (Full/Optimal variant) for Classification and Regression.
    
    This is the FULL/OPTIMAL implementation that uses a two-pass algorithm:
    1. First pass: Collect all nodes at each depth
    2. Second pass: For each feature, evaluate all nodes with their best thresholds
    3. Choose feature that maximizes total improvement across all nodes
    
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
        self.depth_node_thresholds_ = {}  # Store {depth: {node_id: threshold}}
        self.cat_features_ = []
        self.cat_encodings_ = {}
        self.global_prior_ = None
        self.X_encoded_ = None
        self._node_id_counter = 0  # For tracking nodes across passes
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> RestrictedDecisionTreeFull:
        """Build the optimal restricted decision tree using two-pass algorithm."""
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
            self.global_prior_ = np.mean(y)
        else:
            self.global_prior_ = np.mean(y)
        
        # Encode categorical features
        if len(self.cat_features_) > 0:
            X_encoded = self._encode_categorical_features(X, y)
        else:
            X_encoded = X.copy()
        
        self.X_encoded_ = X_encoded
        indices = list(range(len(X)))
        
        # Build tree level by level (breadth-first approach for optimal feature selection)
        self.root = self._build_tree_optimal(X_encoded, y, indices)
        return self
    
    def _build_tree_optimal(self, X: np.ndarray, y: np.ndarray, root_indices: List[int]) -> Node:
        """
        Build tree using optimal two-pass algorithm.
        Process level by level to ensure optimal feature selection at each depth.
        """
        # Initialize root
        root = Node(indices=root_indices)
        
        # Process tree level by level
        current_level = [root]
        
        for depth in range(self.max_depth):
            if not current_level:
                break
            
            # Filter out nodes that should be leaves
            nodes_to_split = []
            for node in current_level:
                if self._should_split(node, y, depth):
                    nodes_to_split.append(node)
                else:
                    if self.task == 'classification':
                        node.value, node.class_counts = self._get_leaf_value(y[node.indices], store_distribution=True)
                    else:
                        node.value = self._get_leaf_value(y[node.indices])
            
            if not nodes_to_split:
                break
            
            # OPTIMAL: Find best feature for this depth considering ALL nodes
            best_feature, node_best_thresholds = self._find_optimal_feature_for_depth(
                X, y, nodes_to_split
            )
            
            if best_feature is None:
                # No valid split found, make all remaining nodes leaves
                for node in nodes_to_split:
                    if self.task == 'classification':
                        node.value, node.class_counts = self._get_leaf_value(y[node.indices], store_distribution=True)
                    else:
                        node.value = self._get_leaf_value(y[node.indices])
                break
            
            # Store the feature for this depth
            self.depth_features_[depth] = best_feature
            
            # Split all nodes using their optimal thresholds
            next_level = []
            for node in nodes_to_split:
                threshold = node_best_thresholds.get(id(node))
                if threshold is None:
                    if self.task == 'classification':
                        node.value, node.class_counts = self._get_leaf_value(y[node.indices], store_distribution=True)
                    else:
                        node.value = self._get_leaf_value(y[node.indices])
                    continue
                
                # Split this node
                left_indices, right_indices = self._split_node(
                    X, node.indices, best_feature, threshold
                )
                
                if len(left_indices) == 0 or len(right_indices) == 0:
                    if self.task == 'classification':
                        node.value, node.class_counts = self._get_leaf_value(y[node.indices], store_distribution=True)
                    else:
                        node.value = self._get_leaf_value(y[node.indices])
                    continue
                
                # Create children
                node.split_info = {'feature': best_feature, 'threshold': threshold}
                node.left = Node(indices=left_indices)
                node.right = Node(indices=right_indices)
                
                next_level.append(node.left)
                next_level.append(node.right)
            
            current_level = next_level
        
        # Make any remaining nodes leaves
        for node in current_level:
            if node.value is None:  # Changed condition
                if self.task == 'classification':
                    node.value, node.class_counts = self._get_leaf_value(y[node.indices], store_distribution=True)
                else:
                    node.value = self._get_leaf_value(y[node.indices])
        
        return root
    
    def _should_split(self, node: Node, y: np.ndarray, depth: int) -> bool:
        """Check if a node should be split."""
        if depth >= self.max_depth:
            return False
        if len(node.indices) < self.min_samples_split:
            return False
        if len(np.unique(y[node.indices])) == 1:
            return False
        return True
    
    def _find_optimal_feature_for_depth(
        self,
        X: np.ndarray,
        y: np.ndarray,
        nodes: List[Node]
    ) -> Tuple[Optional[int], Dict[int, float]]:
        """
        Find the optimal feature for all nodes at this depth.
        
        For each feature:
            1. Find best threshold for each node
            2. Calculate total improvement across all nodes
        Choose feature with maximum total improvement.
        
        Returns:
            (best_feature_idx, {node_id: best_threshold})
        """
        best_feature = None
        best_total_improvement = -float('inf')
        best_node_thresholds = {}
        
        # Try each feature
        for feature_idx in range(self.n_features_):
            total_improvement = 0.0
            node_thresholds = {}
            valid_split_found = False
            
            # For each node, find its best threshold for this feature
            for node in nodes:
                threshold, improvement = self._find_best_threshold_for_node_feature(
                    X, y, node.indices, feature_idx
                )
                
                if threshold is not None:
                    total_improvement += improvement
                    node_thresholds[id(node)] = threshold
                    valid_split_found = True
            
            # Check if this feature gives better total improvement
            if valid_split_found and total_improvement > best_total_improvement:
                best_total_improvement = total_improvement
                best_feature = feature_idx
                best_node_thresholds = node_thresholds
        
        return best_feature, best_node_thresholds
    
    def _find_best_threshold_for_node_feature(
        self,
        X: np.ndarray,
        y: np.ndarray,
        indices: List[int],
        feature_idx: int
    ) -> Tuple[Optional[float], float]:
        """
        Find the best threshold for a specific feature at a specific node.
        
        Returns:
            (best_threshold, best_improvement)
        """
        if len(indices) < self.min_samples_split:
            return None, 0.0
        
        X_subset = X[indices]
        y_subset = y[indices]
        
        feature_values = X_subset[:, feature_idx]
        unique_values = np.unique(feature_values)
        
        if len(unique_values) < 2:
            return None, 0.0
        
        thresholds = (unique_values[:-1] + unique_values[1:]) / 2
        
        best_improvement = -float('inf')
        best_threshold = None
        
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
                best_threshold = threshold
        
        if best_threshold is None:
            return None, 0.0
        
        return best_threshold, best_improvement
    
    def _split_node(
        self,
        X: np.ndarray,
        indices: List[int],
        feature_idx: int,
        threshold: float
    ) -> Tuple[List[int], List[int]]:
        """Split node indices based on feature and threshold."""
        X_subset = X[indices]
        left_mask = X_subset[:, feature_idx] <= threshold
        
        left_indices = [indices[i] for i in range(len(indices)) if left_mask[i]]
        right_indices = [indices[i] for i in range(len(indices)) if not left_mask[i]]
        
        return left_indices, right_indices
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for X."""
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
        """Calculate the weighted criterion for a split."""
        n_left, n_right = len(y_left), len(y_right)
        n_total = n_left + n_right
        
        if n_total == 0:
            return float('inf')
        
        score_left = self._calculate_criterion(y_left)
        score_right = self._calculate_criterion(y_right)
        
        weighted_score = (n_left / n_total) * score_left + (n_right / n_total) * score_right
        return weighted_score
    
    def _calculate_improvement(self, y_parent: np.ndarray, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """Calculate the improvement (information gain) from a split."""
        n_left, n_right = len(y_left), len(y_right)
        n_total = len(y_parent)
        
        if n_total == 0:
            return 0.0
        
        parent_criterion = self._calculate_criterion(y_parent)
        weighted_child_criterion = self._calculate_split_score(y_left, y_right)
        
        improvement = parent_criterion - weighted_child_criterion
        return improvement
    
    def _get_leaf_value(self, y: np.ndarray, store_distribution: bool = False) -> Union[int, float, Tuple[Union[int, float], Dict[int, int]]]:
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
    
    def _identify_categorical_features(self, X: np.ndarray) -> None:
        """Identify which features are categorical based on user input."""
        if not self.cat_features:
            self.cat_features_ = []
            return
        
        if isinstance(self.cat_features[0], int):
            self.cat_features_ = list(self.cat_features)
        else:
            raise ValueError("Feature names are not supported. Please use feature indices.")
        
        for idx in self.cat_features_:
            if idx < 0 or idx >= self.n_features_:
                raise ValueError(f"Categorical feature index {idx} is out of bounds.")
    
    def _encode_categorical_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Encode categorical features using ordered target statistics (CatBoost-style)."""
        X_encoded = X.copy()
        
        for cat_idx in self.cat_features_:
            if cat_idx not in self.cat_encodings_:
                self.cat_encodings_[cat_idx] = {}
            
            encoded_col = np.zeros(len(X))
            
            for i in range(len(X)):
                category = X[i, cat_idx]
                
                if i == 0:
                    encoded_value = self.global_prior_
                else:
                    prev_mask = np.array([X[j, cat_idx] == category for j in range(i)])
                    prev_targets = y[:i][prev_mask]
                    
                    count = len(prev_targets)
                    if count > 0:
                        sum_target = np.sum(prev_targets)
                        encoded_value = (sum_target + self.prior_weight * self.global_prior_) / (count + self.prior_weight)
                    else:
                        encoded_value = self.global_prior_
                
                encoded_col[i] = encoded_value
                self.cat_encodings_[cat_idx][category] = encoded_value
            
            X_encoded[:, cat_idx] = encoded_col
        
        return X_encoded.astype(float)
    
    def _encode_categorical_features_predict(self, X: np.ndarray) -> np.ndarray:
        """Encode categorical features for prediction using stored encodings."""
        X_encoded = X.copy()
        
        for cat_idx in self.cat_features_:
            encoded_col = np.zeros(len(X))
            for i in range(len(X)):
                category = X[i, cat_idx]
                
                if cat_idx in self.cat_encodings_ and category in self.cat_encodings_[cat_idx]:
                    encoded_value = self.cat_encodings_[cat_idx][category]
                else:
                    encoded_value = self.global_prior_
                
                encoded_col[i] = encoded_value
            
            X_encoded[:, cat_idx] = encoded_col
        
        return X_encoded.astype(float)
    
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
