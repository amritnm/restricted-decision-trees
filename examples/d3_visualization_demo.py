"""Examples of interactive D3.js visualization of a fitted RDT."""

import json

import numpy as np
from sklearn.datasets import load_iris, make_regression

from rdt import RDT
from rdt.visualize import export_tree_dict, visualize_d3_observable


def example_classification_d3():
    """D3 visualization for a classification tree."""
    print("=" * 80)
    print("CLASSIFICATION TREE - D3 INTERACTIVE VISUALIZATION")
    print("=" * 80)

    iris = load_iris()
    X, y = iris.data, iris.target.astype(float)

    tree = RDT(task="classification", criterion="gini", max_depth=4, min_samples_split=5)
    tree.fit(X, y)

    print(f"\nDataset: {len(X)} samples, {X.shape[1]} features")
    print(f"Classes: {tree.n_classes_}")

    print("\nGenerating D3.js visualization...")
    visualize_d3_observable(
        tree,
        y,
        feature_names=list(iris.feature_names),
        class_names=list(iris.target_names),
        filename="iris_tree_d3.html",
        width=1400,
        height=900,
    )
    print()


def example_regression_d3():
    """D3 visualization for a regression tree."""
    print("=" * 80)
    print("REGRESSION TREE - D3 INTERACTIVE VISUALIZATION")
    print("=" * 80)

    np.random.seed(42)
    X, y = make_regression(n_samples=200, n_features=5, n_informative=4, noise=10.0, random_state=42)

    tree = RDT(task="regression", criterion="mse", max_depth=4, min_samples_split=5)
    tree.fit(X, y)

    print(f"\nDataset: {len(X)} samples, {X.shape[1]} features")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")

    print("\nGenerating D3.js visualization...")
    visualize_d3_observable(
        tree,
        y,
        feature_names=[f"Feature_{i}" for i in range(X.shape[1])],
        filename="regression_tree_d3.html",
        width=1400,
        height=900,
    )
    print()


def example_export_json():
    """Export a fitted tree's structure as JSON (for custom visualizations)."""
    print("=" * 80)
    print("EXPORT TREE AS JSON")
    print("=" * 80)

    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1], dtype=float)

    tree = RDT(task="classification", max_depth=2)
    tree.fit(X, y)

    tree_dict = export_tree_dict(tree, y, feature_names=["X1", "X2"], class_names=["Class A", "Class B"])

    print("\nTree structure as JSON:")
    print(json.dumps(tree_dict, indent=2))
    print()


def main():
    print("\n" + "#" * 80)
    print("# RESTRICTED DECISION TREE - D3.JS VISUALIZATION EXAMPLES")
    print("#" * 80 + "\n")

    example_classification_d3()
    example_regression_d3()
    example_export_json()

    print("#" * 80)
    print("# ALL D3 VISUALIZATIONS CREATED!")
    print("#" * 80)
    print("\nGenerated files:")
    print("  1. iris_tree_d3.html - Iris classification tree")
    print("  2. regression_tree_d3.html - Regression tree")
    print("\nOpen either file in a browser. Click nodes to expand/collapse,")
    print("scroll/drag to zoom and pan.")
    print()


if __name__ == "__main__":
    main()
