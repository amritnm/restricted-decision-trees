#pragma once

#include <stdexcept>
#include <string>

namespace rdt {

enum class Task { Classification, Regression };

enum class Criterion { Gini, Entropy, MSE, MAE };

inline Criterion default_criterion(Task task) {
    return task == Task::Classification ? Criterion::Gini : Criterion::MSE;
}

inline Criterion criterion_from_string(const std::string& name) {
    if (name == "gini") return Criterion::Gini;
    if (name == "entropy") return Criterion::Entropy;
    if (name == "mse") return Criterion::MSE;
    if (name == "mae") return Criterion::MAE;
    throw std::invalid_argument("Unknown criterion: " + name);
}

inline std::string to_string(Criterion c) {
    switch (c) {
        case Criterion::Gini: return "gini";
        case Criterion::Entropy: return "entropy";
        case Criterion::MSE: return "mse";
        case Criterion::MAE: return "mae";
    }
    return "";
}

struct TreeParams {
    Task task = Task::Classification;
    Criterion criterion = Criterion::Gini;
    int max_depth = 5;
    int min_samples_split = 2;
    int min_samples_leaf = 1;
    // A depth's winning feature is only committed if its total improvement
    // meets this threshold; otherwise every node at that depth becomes a
    // leaf. sklearn calls this min_impurity_decrease; XGBoost/LightGBM call
    // the equivalent gamma/min_split_loss.
    double min_impurity_decrease = 0.0;

    // Implementation choices that do not change which splits are chosen.
    //
    // presort: score every node of a depth in one pass over each feature's
    // presorted rows (presort.hpp) instead of sorting each (node, feature)
    // pair's samples at every depth. Both searches pick the same features and
    // thresholds; the only possible difference is floating-point summation
    // order *within a run of tied feature values* (whose internal order the
    // per-node std::sort leaves unspecified). MAE always uses the per-node
    // search. Exposed mainly so tests can compare the two.
    bool presort = true;
    // Keep each Node's `indices` (the training rows that reached it) after
    // fit. rdt.visualize's export_tree_dict needs them for a standalone tree;
    // the ensembles turn this off, since they would otherwise retain
    // ~4 * n_samples * (max_depth + 1) bytes per fitted tree.
    bool retain_indices = true;
};

} // namespace rdt
