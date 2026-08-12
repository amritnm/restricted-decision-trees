"""Basic usage examples for the restricted-decision-trees package."""

import numpy as np
from sklearn.datasets import load_iris, make_regression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

from rdt import RDT
from rdt.visualize import print_tree


def example_classification():
    """Classification on the Iris dataset."""
    print("=" * 80)
    print("EXAMPLE 1: CLASSIFICATION")
    print("=" * 80)

    iris = load_iris()
    X, y = iris.data, iris.target.astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    tree = RDT(task="classification", criterion="gini", max_depth=4, min_samples_split=5, min_samples_leaf=2)
    tree.fit(X_train, y_train)

    y_pred = tree.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Test accuracy: {accuracy:.4f}")

    y_proba = tree.predict_proba(X_test)
    print(f"\nProbability predictions shape: {y_proba.shape}")
    print(f"First prediction probabilities: {y_proba[0]}")

    print("\nTree structure:")
    print_tree(tree, feature_names=list(iris.feature_names))

    print("\nFeature chosen at each depth:")
    for depth, feat_idx in tree.depth_features_.items():
        print(f"  Depth {depth}: {iris.feature_names[feat_idx]}")
    print()


def example_regression():
    """Regression on synthetic data."""
    print("=" * 80)
    print("EXAMPLE 2: REGRESSION")
    print("=" * 80)

    np.random.seed(42)
    X, y = make_regression(n_samples=200, n_features=5, n_informative=4, noise=10.0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    tree = RDT(task="regression", criterion="mse", max_depth=5, min_samples_split=10, min_samples_leaf=5)
    tree.fit(X_train, y_train)

    y_pred = tree.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Mean target value: {np.mean(y_test):.4f}")

    print("\nFeature chosen at each depth:")
    for depth, feat_idx in tree.depth_features_.items():
        print(f"  Depth {depth}: Feature {feat_idx}")
    print()


def example_criteria_comparison():
    """Compare gini vs entropy on the same classification problem."""
    print("=" * 80)
    print("EXAMPLE 3: GINI VS ENTROPY")
    print("=" * 80)

    iris = load_iris()
    X, y = iris.data, iris.target.astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    for criterion in ("gini", "entropy"):
        tree = RDT(task="classification", criterion=criterion, max_depth=4, min_samples_split=5)
        tree.fit(X_train, y_train)
        acc = accuracy_score(y_test, tree.predict(X_test))
        print(f"  {criterion:8s}: accuracy = {acc:.4f}, depth_features_ = {tree.depth_features_}")
    print()


def main():
    print("\n" + "#" * 80)
    print("# RESTRICTED DECISION TREE - BASIC USAGE EXAMPLES")
    print("#" * 80 + "\n")

    example_classification()
    example_regression()
    example_criteria_comparison()

    print("#" * 80)
    print("# ALL EXAMPLES COMPLETED!")
    print("#" * 80)
    print("\nKey points:")
    print("  - Every node at a given depth splits on the same feature;")
    print("    each node still picks its own threshold.")
    print("  - Supports classification (gini/entropy) and regression (mse/mae).")
    print("  - depth_features_ tells you which feature was chosen at each depth.")
    print()


if __name__ == "__main__":
    main()
