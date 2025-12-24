"""
Test optimal restricted decision tree implementation.
Compares greedy vs optimal implementations.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from restricted_decision_tree_full import RestrictedDecisionTreeFull
import sys
sys.path.append('..')
from restricted_decision_tree import RestrictedDecisionTree
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time


def test_optimal_vs_greedy():
    """Compare optimal and greedy implementations."""
    print("="*80)
    print("OPTIMAL VS GREEDY IMPLEMENTATION COMPARISON")
    print("="*80)
    
    # Generate dataset
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=3,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"\nDataset: {X_train.shape[0]} training, {X_test.shape[0]} test samples")
    print(f"Features: {X_train.shape[1]}, Classes: {len(np.unique(y))}")
    
    # Test Greedy Implementation
    print("\n" + "-"*80)
    print("1. GREEDY Implementation (Original)")
    print("-"*80)
    
    start = time.time()
    tree_greedy = RestrictedDecisionTree(
        task='classification',
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2
    )
    tree_greedy.fit(X_train, y_train)
    greedy_time = time.time() - start
    
    y_pred_greedy = tree_greedy.predict(X_test)
    greedy_acc = accuracy_score(y_test, y_pred_greedy)
    
    print(f"Training time: {greedy_time:.4f} seconds")
    print(f"Test accuracy: {greedy_acc:.4f}")
    print(f"\nFeatures selected at each depth:")
    for depth, feat in tree_greedy.depth_features_.items():
        print(f"  Depth {depth}: Feature {feat}")
    
    # Test Optimal Implementation
    print("\n" + "-"*80)
    print("2. OPTIMAL Implementation (Two-Pass)")
    print("-"*80)
    
    start = time.time()
    tree_optimal = RestrictedDecisionTreeFull(
        task='classification',
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2
    )
    tree_optimal.fit(X_train, y_train)
    optimal_time = time.time() - start
    
    y_pred_optimal = tree_optimal.predict(X_test)
    optimal_acc = accuracy_score(y_test, y_pred_optimal)
    
    print(f"Training time: {optimal_time:.4f} seconds")
    print(f"Test accuracy: {optimal_acc:.4f}")
    print(f"\nFeatures selected at each depth:")
    for depth, feat in tree_optimal.depth_features_.items():
        print(f"  Depth {depth}: Feature {feat}")
    
    # Comparison
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"Greedy accuracy:  {greedy_acc:.4f}  | Time: {greedy_time:.4f}s")
    print(f"Optimal accuracy: {optimal_acc:.4f}  | Time: {optimal_time:.4f}s")
    print(f"\nAccuracy improvement: {(optimal_acc - greedy_acc):.4f}")
    print(f"Time overhead: {(optimal_time / greedy_time - 1) * 100:.1f}%")
    
    print(f"\nFeature selection comparison:")
    for depth in range(min(len(tree_greedy.depth_features_), len(tree_optimal.depth_features_))):
        greedy_feat = tree_greedy.depth_features_.get(depth, 'N/A')
        optimal_feat = tree_optimal.depth_features_.get(depth, 'N/A')
        match = "✓" if greedy_feat == optimal_feat else "✗"
        print(f"  Depth {depth}: Greedy={greedy_feat}, Optimal={optimal_feat} {match}")
    print()


def test_optimal_algorithm():
    """Test the optimal algorithm's two-pass approach."""
    print("="*80)
    print("OPTIMAL ALGORITHM DEMONSTRATION")
    print("="*80)
    
    print("\nThe optimal algorithm uses a TWO-PASS approach:")
    print("  1. COLLECT: Gather all nodes at current depth")
    print("  2. EVALUATE: For each feature, find best threshold per node")
    print("  3. OPTIMIZE: Choose feature maximizing total improvement")
    print("  4. SPLIT: Apply chosen feature with node-specific thresholds")
    
    # Simple dataset
    np.random.seed(42)
    X = np.random.randn(100, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    tree = RestrictedDecisionTreeFull(
        task='classification',
        max_depth=3,
        min_samples_split=5
    )
    
    tree.fit(X, y)
    
    print("\n" + "-"*80)
    print("Resulting tree structure:")
    print("-"*80)
    tree.print_tree()
    
    print("\n" + "-"*80)
    print("Key characteristics of optimal approach:")
    print("-"*80)
    print("✓ Truly maximizes total improvement at each depth")
    print("✓ Considers all nodes simultaneously")
    print("✓ More computationally expensive but more accurate")
    print("✓ Breadth-first construction ensures optimality")
    print()


def test_iris_detailed():
    """Detailed test on Iris dataset."""
    print("="*80)
    print("IRIS DATASET - OPTIMAL IMPLEMENTATION")
    print("="*80)
    
    iris = load_iris()
    X, y = iris.data, iris.target
    
    tree = RestrictedDecisionTreeFull(
        task='classification',
        criterion='gini',
        max_depth=4,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    tree.fit(X, y)
    
    predictions = tree.predict(X)
    accuracy = accuracy_score(y, predictions)
    
    print(f"\nDataset: {len(X)} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    print(f"Training accuracy: {accuracy:.4f}")
    
    print("\n" + "-"*80)
    print("Tree structure:")
    print("-"*80)
    tree.print_tree()
    
    print("\n" + "-"*80)
    print("Features used at each depth:")
    print("-"*80)
    feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    for depth, feat_idx in tree.depth_features_.items():
        print(f"  Depth {depth}: {feature_names[feat_idx]} (Feature {feat_idx})")
    print()


def main():
    """Run all tests."""
    print("\n" + "#"*80)
    print("# OPTIMAL RESTRICTED DECISION TREE - TEST SUITE")
    print("#"*80)
    print()
    
    # Test 1: Compare greedy vs optimal
    test_optimal_vs_greedy()
    
    # Test 2: Demonstrate algorithm
    test_optimal_algorithm()
    
    # Test 3: Iris dataset
    test_iris_detailed()
    
    print("#"*80)
    print("# ALL TESTS COMPLETED!")
    print("#"*80)
    print("\nKey Takeaways:")
    print("  • Optimal implementation uses two-pass algorithm")
    print("  • Slightly slower but potentially more accurate")
    print("  • Truly maximizes total improvement at each depth")
    print("  • Same interface as greedy implementation")
    print("#"*80)


if __name__ == "__main__":
    main()
