"""
D3.js Interactive Visualization Examples for Restricted Decision Trees.

Both RestrictedDecisionTree and RestrictedDecisionTreeFull include built-in
D3.js visualization capabilities through the visualize_d3_observable() method.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from restricted_decision_tree import RestrictedDecisionTree
from restricted_decision_tree_full import RestrictedDecisionTreeFull
from sklearn.datasets import load_iris, make_regression


def example_d3_classification():
    """Create interactive D3 visualization for classification tree."""
    print("="*80)
    print("D3 VISUALIZATION - CLASSIFICATION")
    print("="*80)
    
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Create and train tree
    tree = RestrictedDecisionTree(
        task='classification',
        criterion='gini',
        max_depth=4,
        min_samples_split=5
    )
    
    tree.fit(X, y)
    
    print(f"\nDataset: {len(X)} samples, {X.shape[1]} features")
    print(f"Classes: {', '.join(iris.target_names)}")
    
    # Create D3 visualization
    print("\nGenerating interactive D3.js visualization...")
    tree.visualize_d3_observable(
        y=y,
        feature_names=iris.feature_names.tolist(),
        class_names=iris.target_names.tolist(),
        filename='restricted_tree_iris_d3.html',
        width=1400,
        height=900
    )
    
    print("\n✓ Visualization created: restricted_tree_iris_d3.html")
    print("\nFeatures:")
    print("  • Click nodes to expand/collapse subtrees")
    print("  • Zoom and pan with mouse/trackpad")
    print("  • View detailed statistics for each node")
    print("  • See feature splits and sample counts")
    print()


def example_d3_regression():
    """Create interactive D3 visualization for regression tree."""
    print("="*80)
    print("D3 VISUALIZATION - REGRESSION")
    print("="*80)
    
    # Generate regression data
    np.random.seed(42)
    X, y = make_regression(
        n_samples=200,
        n_features=5,
        n_informative=4,
        noise=10.0,
        random_state=42
    )
    
    # Create and train tree
    tree = RestrictedDecisionTreeFull(
        task='regression',
        criterion='mse',
        max_depth=4,
        min_samples_split=10
    )
    
    tree.fit(X, y)
    
    print(f"\nDataset: {len(X)} samples, {X.shape[1]} features")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    
    # Create D3 visualization
    print("\nGenerating interactive D3.js visualization...")
    tree.visualize_d3_observable(
        y=y,
        feature_names=[f'Feature_{i}' for i in range(X.shape[1])],
        filename='restricted_tree_regression_d3.html',
        width=1400,
        height=900
    )
    
    print("\n✓ Visualization created: restricted_tree_regression_d3.html")
    print()


def example_d3_comparison():
    """Create visualizations comparing greedy vs full implementations."""
    print("="*80)
    print("D3 VISUALIZATION - GREEDY VS FULL COMPARISON")
    print("="*80)
    
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Greedy implementation
    print("\n1. Creating visualization for Greedy variant...")
    tree_greedy = RestrictedDecisionTree(
        task='classification',
        max_depth=3,
        min_samples_split=10
    )
    tree_greedy.fit(X, y)
    tree_greedy.visualize_d3_observable(
        y=y,
        feature_names=iris.feature_names.tolist(),
        class_names=iris.target_names.tolist(),
        filename='restricted_tree_greedy_d3.html',
        width=1200,
        height=800
    )
    
    # Full/Optimal implementation
    print("2. Creating visualization for Full/Optimal variant...")
    tree_full = RestrictedDecisionTreeFull(
        task='classification',
        max_depth=3,
        min_samples_split=10
    )
    tree_full.fit(X, y)
    tree_full.visualize_d3_observable(
        y=y,
        feature_names=iris.feature_names.tolist(),
        class_names=iris.target_names.tolist(),
        filename='restricted_tree_full_d3.html',
        width=1200,
        height=800
    )
    
    print("\n✓ Created two visualizations for comparison:")
    print("  • restricted_tree_greedy_d3.html")
    print("  • restricted_tree_full_d3.html")
    print("\nOpen both in browser to compare feature selection!")
    print()


def example_export_json():
    """Example of exporting tree data as JSON for custom visualizations."""
    print("="*80)
    print("EXPORT TREE AS JSON")
    print("="*80)
    
    # Simple dataset
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
    y = np.array([0, 0, 0, 1, 1, 1])
    
    tree = RestrictedDecisionTree(task='classification', max_depth=2)
    tree.fit(X, y)
    
    # Export as dictionary
    import json
    tree_dict = tree.export_tree_dict(
        y=y,
        feature_names=['X1', 'X2'],
        class_names=['Class A', 'Class B']
    )
    
    print("\nTree structure as JSON:")
    print(json.dumps(tree_dict, indent=2))
    
    print("\nThis JSON can be used in:")
    print("  • Custom D3 visualizations")
    print("  • Observable notebooks")
    print("  • Web applications")
    print("  • Data analysis tools")
    print()


def main():
    """Run all D3 visualization examples."""
    print("\n" + "#"*80)
    print("# RESTRICTED DECISION TREE - D3 VISUALIZATION EXAMPLES")
    print("#"*80)
    print()
    
    # Example 1: Classification visualization
    example_d3_classification()
    
    # Example 2: Regression visualization
    example_d3_regression()
    
    # Example 3: Compare greedy vs full
    example_d3_comparison()
    
    # Example 4: JSON export
    example_export_json()
    
    print("#"*80)
    print("# ALL D3 VISUALIZATIONS CREATED!")
    print("#"*80)
    print("\nGenerated HTML files:")
    print("  1. restricted_tree_iris_d3.html - Iris classification")
    print("  2. restricted_tree_regression_d3.html - Regression tree")
    print("  3. restricted_tree_greedy_d3.html - Greedy variant")
    print("  4. restricted_tree_full_d3.html - Full/Optimal variant")
    print("\nTo view:")
    print("  • Open any .html file in a web browser")
    print("  • Click nodes to expand/collapse")
    print("  • Use mouse to zoom and pan")
    print()


if __name__ == "__main__":
    main()
