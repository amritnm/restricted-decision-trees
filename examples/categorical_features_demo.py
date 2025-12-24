"""
Categorical Features Support Examples for Restricted Decision Trees.

Both implementations support categorical features using CatBoost-style
ordered target statistics encoding.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from restricted_decision_tree import RestrictedDecisionTree
from restricted_decision_tree_full import RestrictedDecisionTreeFull
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error


def example_categorical_classification():
    """Classification with categorical features."""
    print("="*80)
    print("CATEGORICAL FEATURES - CLASSIFICATION")
    print("="*80)
    
    # Create dataset with categorical features
    np.random.seed(42)
    n_samples = 200
    
    # Features: City(cat), Income(num), Education(cat)
    cities = np.random.choice(['NYC', 'LA', 'Chicago', 'Houston'], n_samples)
    incomes = np.random.randint(30000, 120000, n_samples)
    education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples)
    
    X = np.column_stack([cities, incomes, education])
    
    # Target: High earner classification (influenced by education and city)
    y = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        score = 0
        if education[i] in ['Master', 'PhD']:
            score += 2
        if cities[i] in ['NYC', 'LA']:
            score += 1
        if incomes[i] > 70000:
            score += 1
        y[i] = 1 if score >= 2 else 0
    
    # Add some noise
    noise_idx = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    y[noise_idx] = 1 - y[noise_idx]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"\nDataset: {len(X_train)} training, {len(X_test)} test samples")
    print(f"Features: 0=City(cat), 1=Income(num), 2=Education(cat)")
    print(f"Classes: {np.unique(y)}")
    
    # Create tree with categorical features specified
    print("\nTraining RestrictedDecisionTree with categorical features...")
    tree = RestrictedDecisionTree(
        task='classification',
        cat_features=[0, 2],  # City and Education are categorical
        max_depth=4,
        min_samples_split=10,
        prior_weight=1.0  # Smoothing parameter
    )
    
    tree.fit(X_train, y_train)
    
    # Make predictions
    y_pred = tree.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test accuracy: {accuracy:.4f}")
    
    # View categorical encodings
    print("\nCategorical encodings learned:")
    print("-" * 40)
    if 0 in tree.cat_encodings_:
        print("City encodings:")
        for cat, enc in tree.cat_encodings_[0].items():
            print(f"  {cat}: {enc:.4f}")
    
    if 2 in tree.cat_encodings_:
        print("\nEducation encodings:")
        for cat, enc in tree.cat_encodings_[2].items():
            print(f"  {cat}: {enc:.4f}")
    
    print("\nFeatures used at each depth:")
    feature_names = ['City', 'Income', 'Education']
    for depth, feat_idx in tree.depth_features_.items():
        print(f"  Depth {depth}: {feature_names[feat_idx]}")
    print()


def example_categorical_regression():
    """Regression with categorical features."""
    print("="*80)
    print("CATEGORICAL FEATURES - REGRESSION")
    print("="*80)
    
    # Car price prediction dataset
    np.random.seed(42)
    n_samples = 300
    
    # Features: Brand(cat), Mileage(num), Condition(cat)
    brands = np.random.choice(['Toyota', 'Honda', 'Ford', 'BMW', 'Audi'], n_samples)
    mileage = np.random.randint(0, 150000, n_samples)
    condition = np.random.choice(['New', 'Used', 'Certified'], n_samples)
    
    X = np.column_stack([brands, mileage, condition])
    
    # Target: Price (influenced by brand, mileage, and condition)
    y = np.zeros(n_samples)
    for i in range(n_samples):
        base_price = 20000
        
        # Brand adjustment
        if brands[i] in ['BMW', 'Audi']:
            base_price += 15000
        elif brands[i] in ['Toyota', 'Honda']:
            base_price += 5000
        
        # Mileage adjustment (negative impact)
        base_price -= mileage[i] * 0.1
        
        # Condition adjustment
        if condition[i] == 'New':
            base_price += 10000
        elif condition[i] == 'Certified':
            base_price += 5000
        
        y[i] = max(5000, base_price + np.random.randn() * 2000)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"\nDataset: {len(X_train)} training, {len(X_test)} test samples")
    print(f"Features: 0=Brand(cat), 1=Mileage(num), 2=Condition(cat)")
    print(f"Target: Price (${y.min():.0f} - ${y.max():.0f})")
    
    # Create tree with categorical features
    print("\nTraining RestrictedDecisionTreeFull with categorical features...")
    tree = RestrictedDecisionTreeFull(
        task='regression',
        cat_features=[0, 2],  # Brand and Condition are categorical
        max_depth=5,
        min_samples_split=15,
        prior_weight=2.0  # More smoothing for regression
    )
    
    tree.fit(X_train, y_train)
    
    # Make predictions
    y_pred = tree.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"Test RMSE: ${rmse:.2f}")
    print(f"Mean price: ${np.mean(y_test):.2f}")
    
    # View categorical encodings
    print("\nCategorical encodings learned:")
    print("-" * 40)
    if 0 in tree.cat_encodings_:
        print("Brand encodings:")
        for cat, enc in tree.cat_encodings_[0].items():
            print(f"  {cat}: ${enc:.2f}")
    
    if 2 in tree.cat_encodings_:
        print("\nCondition encodings:")
        for cat, enc in tree.cat_encodings_[2].items():
            print(f"  {cat}: ${enc:.2f}")
    print()


def example_unseen_categories():
    """Demonstrate handling of unseen categories."""
    print("="*80)
    print("HANDLING UNSEEN CATEGORIES")
    print("="*80)
    
    # Training data
    X_train = np.array([
        ['Red', 10, 'A'],
        ['Blue', 20, 'B'],
        ['Red', 15, 'A'],
        ['Green', 25, 'C'],
        ['Blue', 30, 'B'],
        ['Red', 12, 'A']
    ])
    y_train = np.array([1, 0, 1, 0, 0, 1])
    
    # Test data with unseen categories
    X_test = np.array([
        ['Red', 11, 'A'],      # Seen categories
        ['Yellow', 22, 'D'],   # Completely unseen
        ['Blue', 18, 'A'],     # Partially unseen
    ])
    
    print("\nTraining data categories:")
    print(f"  Feature 0: {np.unique(X_train[:, 0])}")
    print(f"  Feature 2: {np.unique(X_train[:, 2])}")
    
    print("\nTest data categories:")
    print(f"  Feature 0: {np.unique(X_test[:, 0])} (Yellow is new!)")
    print(f"  Feature 2: {np.unique(X_test[:, 2])} (D is new!)")
    
    # Train tree
    tree = RestrictedDecisionTree(
        task='classification',
        cat_features=[0, 2],
        max_depth=3
    )
    
    tree.fit(X_train, y_train)
    
    print(f"\nGlobal prior (fallback for unseen): {tree.global_prior_:.4f}")
    
    print("\nLearned encodings:")
    for feat_idx in [0, 2]:
        print(f"  Feature {feat_idx}:")
        for cat, enc in tree.cat_encodings_[feat_idx].items():
            print(f"    {cat}: {enc:.4f}")
    
    # Predict - unseen categories use global prior
    predictions = tree.predict(X_test)
    
    print("\nPredictions on test data:")
    for i, (sample, pred) in enumerate(zip(X_test, predictions)):
        print(f"  Sample {i}: {sample} -> Class {pred}")
    
    print("\nNote: Unseen categories ('Yellow', 'D') use global prior")
    print("This prevents errors and provides reasonable defaults")
    print()


def example_ordered_target_statistics():
    """Demonstrate ordered target statistics encoding."""
    print("="*80)
    print("ORDERED TARGET STATISTICS ENCODING")
    print("="*80)
    
    print("\nThis prevents target leakage by using only PREVIOUS samples")
    print("for encoding each category.")
    
    # Simple example
    X = np.array([
        ['A'], ['B'], ['A'], ['B'], ['A'], ['B']
    ])
    y = np.array([1, 0, 1, 0, 1, 0])
    
    tree = RestrictedDecisionTree(
        task='classification',
        cat_features=[0],
        prior_weight=1.0
    )
    
    print("\nEncoding process (feature='Category', prior_weight=1.0):")
    print("-" * 60)
    
    global_prior = np.mean(y)
    print(f"Global prior (mean of all targets): {global_prior:.4f}")
    
    print("\nSample-by-sample encoding:")
    for i in range(len(X)):
        cat = X[i, 0]
        if i == 0:
            enc = global_prior
            print(f"  i={i}: {cat} -> {enc:.4f} (first sample, use prior)")
        else:
            prev_y = y[:i]
            prev_X = X[:i, 0]
            prev_same = prev_y[prev_X == cat]
            
            if len(prev_same) > 0:
                enc = (sum(prev_same) + 1.0 * global_prior) / (len(prev_same) + 1.0)
                print(f"  i={i}: {cat} -> {enc:.4f} (prev {cat}: {prev_same}, count={len(prev_same)})")
            else:
                enc = global_prior
                print(f"  i={i}: {cat} -> {enc:.4f} (no prev {cat}, use prior)")
    
    tree.fit(X, y)
    
    print("\nFinal encodings stored:")
    for cat, enc in tree.cat_encodings_[0].items():
        print(f"  '{cat}': {enc:.4f}")
    
    print("\nKey points:")
    print("  ✓ Sample i uses only samples 0 to i-1")
    print("  ✓ Prevents target leakage")
    print("  ✓ Smoothing with global prior prevents overfitting")
    print()


def main():
    """Run all categorical features examples."""
    print("\n" + "#"*80)
    print("# CATEGORICAL FEATURES SUPPORT - EXAMPLES")
    print("#"*80)
    print()
    
    example_categorical_classification()
    example_categorical_regression()
    example_unseen_categories()
    example_ordered_target_statistics()
    
    print("#"*80)
    print("# ALL CATEGORICAL EXAMPLES COMPLETED!")
    print("#"*80)
    print("\nKey Takeaways:")
    print("  • Both implementations support categorical features")
    print("  • Uses CatBoost-style ordered target statistics")
    print("  • Prevents target leakage")
    print("  • Handles unseen categories gracefully")
    print("  • Works for both classification and regression")
    print("  • Just specify cat_features=[...] indices")
    print()


if __name__ == "__main__":
    main()
