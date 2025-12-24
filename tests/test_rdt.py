"""
Test and example usage of the Restricted Decision Tree implementation.
Demonstrates the key difference from oblivious trees: same feature at each depth,
but different thresholds for each node.
"""

import numpy as np
from restricted_decision_tree import Node, RestrictedDecisionTree
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score


def test_classification():
    """Test Restricted Decision Tree for classification tasks."""
    print("="*80)
    print("CLASSIFICATION TEST - RESTRICTED DECISION TREE")
    print("="*80)
    
    # Generate synthetic classification data
    X, y = make_classification(
        n_samples=200,
        n_features=4,
        n_informative=3,
        n_redundant=1,
        n_classes=3,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"\nDataset: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    print(f"Features: {X_train.shape[1]}, Classes: {len(np.unique(y))}")
    
    # Test with Gini criterion
    print("\n" + "-"*80)
    print("Classification with Gini Impurity")
    print("-"*80)
    
    clf_gini = RestrictedDecisionTree(
        task='classification',
        criterion='gini',
        max_depth=4,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    clf_gini.fit(X_train, y_train)
    y_pred_gini = clf_gini.predict(X_test)
    accuracy_gini = accuracy_score(y_test, y_pred_gini)
    
    print(f"Accuracy (Gini): {accuracy_gini:.4f}")
    print(f"\nTree structure:")
    clf_gini.print_tree()
    
    # Test with Entropy criterion
    print("\n" + "-"*80)
    print("Classification with Entropy (Information Gain)")
    print("-"*80)
    
    clf_entropy = RestrictedDecisionTree(
        task='classification',
        criterion='entropy',
        max_depth=4,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    clf_entropy.fit(X_train, y_train)
    y_pred_entropy = clf_entropy.predict(X_test)
    accuracy_entropy = accuracy_score(y_test, y_pred_entropy)
    
    print(f"Accuracy (Entropy): {accuracy_entropy:.4f}")
    print(f"\nTree structure:")
    clf_entropy.print_tree()
    
    print(f"\n{'='*80}")
    print("CLASSIFICATION SUMMARY")
    print(f"{'='*80}")
    print(f"Gini Accuracy:    {accuracy_gini:.4f}")
    print(f"Entropy Accuracy: {accuracy_entropy:.4f}")
    print()


def test_regression():
    """Test Restricted Decision Tree for regression tasks."""
    print("="*80)
    print("REGRESSION TEST - RESTRICTED DECISION TREE")
    print("="*80)
    
    # Generate synthetic regression data
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
    
    print(f"\nDataset: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    print(f"Features: {X_train.shape[1]}")
    
    # Test with MSE criterion
    print("\n" + "-"*80)
    print("Regression with MSE (Mean Squared Error)")
    print("-"*80)
    
    reg_mse = RestrictedDecisionTree(
        task='regression',
        criterion='mse',
        max_depth=4,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    reg_mse.fit(X_train, y_train)
    y_pred_mse = reg_mse.predict(X_test)
    mse_score = mean_squared_error(y_test, y_pred_mse)
    r2_mse = r2_score(y_test, y_pred_mse)
    
    print(f"MSE: {mse_score:.4f}")
    print(f"R² Score: {r2_mse:.4f}")
    print(f"\nTree structure:")
    reg_mse.print_tree()
    
    # Test with MAE criterion
    print("\n" + "-"*80)
    print("Regression with MAE (Mean Absolute Error)")
    print("-"*80)
    
    reg_mae = RestrictedDecisionTree(
        task='regression',
        criterion='mae',
        max_depth=4,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    reg_mae.fit(X_train, y_train)
    y_pred_mae = reg_mae.predict(X_test)
    mse_mae = mean_squared_error(y_test, y_pred_mae)
    r2_mae = r2_score(y_test, y_pred_mae)
    
    print(f"MSE: {mse_mae:.4f}")
    print(f"R² Score: {r2_mae:.4f}")
    print(f"\nTree structure:")
    reg_mae.print_tree()
    
    print(f"\n{'='*80}")
    print("REGRESSION SUMMARY")
    print(f"{'='*80}")
    print(f"MSE criterion - MSE: {mse_score:.4f}, R²: {r2_mse:.4f}")
    print(f"MAE criterion - MSE: {mse_mae:.4f}, R²: {r2_mae:.4f}")
    print()


def demonstrate_restricted_property():
    """Demonstrate the restricted property of the tree."""
    print("="*80)
    print("DEMONSTRATING RESTRICTED TREE PROPERTY")
    print("="*80)
    print("\nIn a RESTRICTED tree:")
    print("  - All nodes at the SAME DEPTH use the SAME FEATURE")
    print("  - BUT each node can have a DIFFERENT THRESHOLD")
    print("  - Feature is chosen to maximize overall improvement at that depth")
    print("\nThis differs from:")
    print("  - OBLIVIOUS tree: same feature AND same threshold at each depth")
    print("  - STANDARD tree: different features and thresholds for each node")
    
    # Create a simple dataset
    np.random.seed(42)
    X = np.array([
        [1.0, 5.0],
        [2.0, 6.0],
        [3.0, 4.0],
        [4.0, 3.0],
        [5.0, 2.0],
        [6.0, 1.0],
        [1.5, 5.5],
        [2.5, 5.8],
        [3.5, 3.5],
        [4.5, 2.5]
    ])
    y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1])
    
    tree = RestrictedDecisionTree(
        task='classification',
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1
    )
    
    tree.fit(X, y)
    
    print("\n" + "-"*80)
    print("Tree Structure:")
    print("-"*80)
    tree.print_tree()
    
    print("\n" + "-"*80)
    print("Feature at each depth level:")
    print("-"*80)
    for depth, feature_idx in tree.depth_features_.items():
        print(f"Depth {depth}: Feature {feature_idx}")
    
    print("\n" + "-"*80)
    print("Key Observation:")
    print("-"*80)
    print("Notice: All nodes at the same depth use the same FEATURE,")
    print("        but you'll see DIFFERENT THRESHOLDS in the tree structure above.")
    print("        This allows more flexibility than oblivious trees while")
    print("        maintaining some structure.")
    print()


def compare_with_manual_example():
    """Compare restricted tree behavior with a manual example."""
    print("="*80)
    print("DETAILED EXAMPLE: How Restricted Trees Work")
    print("="*80)
    
    # Create a simple 2D dataset
    np.random.seed(123)
    X = np.array([
        [1.0, 1.0],
        [1.5, 1.2],
        [2.0, 1.5],
        [5.0, 5.0],
        [5.5, 5.2],
        [6.0, 5.5],
        [1.0, 5.0],
        [1.5, 5.5],
        [5.0, 1.0],
        [5.5, 1.5]
    ])
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 3, 3])
    
    print("\nDataset (10 samples, 2 features, 4 classes):")
    print("X:")
    for i, (x, label) in enumerate(zip(X, y)):
        print(f"  Sample {i}: {x} -> Class {label}")
    
    tree = RestrictedDecisionTree(
        task='classification',
        max_depth=2,
        min_samples_split=2,
        min_samples_leaf=1
    )
    
    tree.fit(X, y)
    
    print("\n" + "-"*80)
    print("Fitted Tree:")
    print("-"*80)
    tree.print_tree()
    
    print("\n" + "-"*80)
    print("Analysis:")
    print("-"*80)
    print("Depth 0 (root): Uses Feature", tree.depth_features_.get(0, "N/A"))
    if 1 in tree.depth_features_:
        print("Depth 1: Uses Feature", tree.depth_features_[1])
        print("\nAt depth 1, both left and right children use the SAME feature,")
        print("but each finds its own optimal threshold for its own subset of data.")
    
    # Make predictions
    predictions = tree.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"\nTraining Accuracy: {accuracy:.4f}")
    print()


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("RESTRICTED DECISION TREE - COMPREHENSIVE TEST SUITE")
    print("="*80)
    print()
    
    # Demonstrate the restricted property
    demonstrate_restricted_property()
    
    # Detailed example
    compare_with_manual_example()
    
    # Test classification
    test_classification()
    
    # Test regression
    test_regression()
    
    print("="*80)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nKey Differences Summary:")
    print("  - OBLIVIOUS TREE: Same feature + same threshold at each depth")
    print("  - RESTRICTED TREE: Same feature + different thresholds at each depth")
    print("  - STANDARD TREE: Different features + different thresholds")
    print("\nRestricted trees offer a middle ground between the structure of")
    print("oblivious trees and the flexibility of standard decision trees.")
    print("="*80)


if __name__ == "__main__":
    main()
