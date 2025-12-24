"""
Basic usage examples for Restricted Decision Tree implementations.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from restricted_decision_tree import RestrictedDecisionTree
from restricted_decision_tree_full import RestrictedDecisionTreeFull
from sklearn.datasets import load_iris, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error


def example_classification_basic():
    """Basic classification example using Iris dataset."""
    print("="*80)
    print("EXAMPLE 1: BASIC CLASSIFICATION")
    print("="*80)
    
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print("\nUsing RestrictedDecisionTree (Greedy):")
    print("-" * 40)
    
    # Create and train tree
    tree = RestrictedDecisionTree(
        task='classification',
        criterion='gini',
        max_depth=4,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    tree.fit(X_train, y_train)
    
    # Make predictions
    y_pred = tree.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Get probability predictions
    y_proba = tree.predict_proba(X_test)
    print(f"\nProbability predictions shape: {y_proba.shape}")
    print(f"First prediction probabilities: {y_proba[0]}")
    
    # View tree structure
    print("\nTree structure:")
    tree.print_tree()
    
    print("\nFeatures used at each depth:")
    for depth, feat_idx in tree.depth_features_.items():
        print(f"  Depth {depth}: {iris.feature_names[feat_idx]}")
    print()


def example_classification_full():
    """Classification using Full/Optimal variant."""
    print("="*80)
    print("EXAMPLE 2: CLASSIFICATION WITH FULL VARIANT")
    print("="*80)
    
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print("\nUsing RestrictedDecisionTreeFull (Optimal):")
    print("-" * 40)
    
    # Create and train tree
    tree = RestrictedDecisionTreeFull(
        task='classification',
        criterion='entropy',  # Using entropy instead of gini
        max_depth=4,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    tree.fit(X_train, y_train)
    
    # Make predictions
    y_pred = tree.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Criterion: {tree.criterion}")
    
    print("\nFeatures used at each depth:")
    for depth, feat_idx in tree.depth_features_.items():
        print(f"  Depth {depth}: {iris.feature_names[feat_idx]}")
    print()


def example_regression():
    """Regression example."""
    print("="*80)
    print("EXAMPLE 3: REGRESSION")
    print("="*80)
    
    # Generate synthetic data
    np.random.seed(42)
    X, y = make_regression(
        n_samples=200,
        n_features=5,
        n_informative=4,
        noise=10.0,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print("\nUsing RestrictedDecisionTree for regression:")
    print("-" * 40)
    
    # Create and train tree
    tree = RestrictedDecisionTree(
        task='regression',
        criterion='mse',
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5
    )
    
    tree.fit(X_train, y_train)
    
    # Make predictions
    y_pred = tree.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Mean target value: {np.mean(y_test):.4f}")
    
    print("\nFeatures used at each depth:")
    for depth, feat_idx in tree.depth_features_.items():
        print(f"  Depth {depth}: Feature {feat_idx}")
    print()


def example_comparison():
    """Compare greedy vs full implementations."""
    print("="*80)
    print("EXAMPLE 4: GREEDY VS FULL COMPARISON")
    print("="*80)
    
    # Generate data
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=300,
        n_features=8,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Greedy variant
    tree_greedy = RestrictedDecisionTree(
        task='classification',
        max_depth=4,
        min_samples_split=10
    )
    tree_greedy.fit(X_train, y_train)
    y_pred_greedy = tree_greedy.predict(X_test)
    acc_greedy = accuracy_score(y_test, y_pred_greedy)
    
    # Full variant
    tree_full = RestrictedDecisionTreeFull(
        task='classification',
        max_depth=4,
        min_samples_split=10
    )
    tree_full.fit(X_train, y_train)
    y_pred_full = tree_full.predict(X_test)
    acc_full = accuracy_score(y_test, y_pred_full)
    
    print(f"\nGreedy variant accuracy: {acc_greedy:.4f}")
    print(f"Full variant accuracy:   {acc_full:.4f}")
    print(f"Improvement: {(acc_full - acc_greedy):.4f}")
    
    print("\nFeature selection comparison:")
    max_depth = max(len(tree_greedy.depth_features_), len(tree_full.depth_features_))
    for depth in range(max_depth):
        feat_greedy = tree_greedy.depth_features_.get(depth, 'N/A')
        feat_full = tree_full.depth_features_.get(depth, 'N/A')
        match = "✓" if feat_greedy == feat_full else "✗"
        print(f"  Depth {depth}: Greedy={feat_greedy}, Full={feat_full} {match}")
    print()


def main():
    """Run all basic examples."""
    print("\n" + "#"*80)
    print("# RESTRICTED DECISION TREE - BASIC USAGE EXAMPLES")
    print("#"*80)
    print()
    
    example_classification_basic()
    example_classification_full()
    example_regression()
    example_comparison()
    
    print("#"*80)
    print("# ALL EXAMPLES COMPLETED!")
    print("#"*80)
    print("\nKey Points:")
    print("  • RestrictedDecisionTree: Greedy, faster")
    print("  • RestrictedDecisionTreeFull: Optimal, more accurate")
    print("  • Both support classification and regression")
    print("  • Same feature at each depth, different thresholds per node")
    print()


if __name__ == "__main__":
    main()
