"""Random Forest and Gradient Boosting ensembles of RDTs."""

import numpy as np
from sklearn.datasets import load_iris, make_regression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

from rdt.ensemble import GradientBoostingRDT, RandomForestRDT


def example_random_forest():
    print("=" * 80)
    print("RANDOM FOREST")
    print("=" * 80)

    iris = load_iris()
    X, y = iris.data, iris.target.astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    rf = RandomForestRDT(
        n_estimators=100, max_depth=4, max_features="sqrt",
        oob_score=True, random_state=42, n_jobs=-1,  # n_jobs=-1: use all available threads
    )
    rf.fit(X_train, y_train)

    acc = accuracy_score(y_test, rf.predict(X_test))
    print(f"\nTest accuracy: {acc:.4f}")
    print(f"OOB score: {rf.oob_score_:.4f}")
    print(f"Feature importances: {dict(zip(iris.feature_names, rf.feature_importances_.round(3)))}")
    print(f"Number of trees: {len(rf.estimators_)}")
    print()


def example_gradient_boosting_binary():
    print("=" * 80)
    print("GRADIENT BOOSTING - BINARY CLASSIFICATION")
    print("=" * 80)

    iris = load_iris()
    mask = iris.target < 2  # binary subset
    X, y = iris.data[mask], iris.target[mask].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    gb = GradientBoostingRDT(
        n_estimators=100, learning_rate=0.1, max_depth=3,
        validation_fraction=0.15, n_iter_no_change=10,  # early stopping
        random_state=42,
    )
    gb.fit(X_train, y_train)

    acc = accuracy_score(y_test, gb.predict(X_test))
    print(f"\nTest accuracy: {acc:.4f}")
    print(f"Rounds actually fitted: {gb.n_estimators_} (n_estimators was {gb.n_estimators})")
    print(f"Training loss: round 0 = {gb.train_score_[0]:.4f}, last = {gb.train_score_[-1]:.4f}")
    print()


def example_gradient_boosting_regression():
    print("=" * 80)
    print("GRADIENT BOOSTING - REGRESSION")
    print("=" * 80)

    X, y = make_regression(n_samples=300, n_features=5, n_informative=4, noise=10.0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    gb = GradientBoostingRDT(
        task="regression", loss="squared_error", n_estimators=100,
        learning_rate=0.1, max_depth=3, subsample=0.8, random_state=42,
    )
    gb.fit(X_train, y_train)

    rmse = np.sqrt(mean_squared_error(y_test, gb.predict(X_test)))
    print(f"\nTest RMSE: {rmse:.4f}")
    print()


def main():
    example_random_forest()
    example_gradient_boosting_binary()
    example_gradient_boosting_regression()


if __name__ == "__main__":
    main()
