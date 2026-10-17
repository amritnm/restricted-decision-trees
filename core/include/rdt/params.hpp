#pragma once

#include <stdexcept>
#include <string>

namespace rdt {

enum class Task { Classification, Regression };

// Newton: XGBoost's second-order gain on per-row gradients / hessians
// (gradient boosting with split_criterion='newton'; regression task only).
enum class Criterion { Gini, Entropy, MSE, MAE, Newton };

inline Criterion default_criterion(Task task) {
    return task == Task::Classification ? Criterion::Gini : Criterion::MSE;
}

inline Criterion criterion_from_string(const std::string& name) {
    if (name == "gini") return Criterion::Gini;
    if (name == "entropy") return Criterion::Entropy;
    if (name == "mse") return Criterion::MSE;
    if (name == "mae") return Criterion::MAE;
    if (name == "newton") return Criterion::Newton;
    throw std::invalid_argument("Unknown criterion: " + name);
}

inline std::string to_string(Criterion c) {
    switch (c) {
        case Criterion::Gini: return "gini";
        case Criterion::Entropy: return "entropy";
        case Criterion::MSE: return "mse";
        case Criterion::MAE: return "mae";
        case Criterion::Newton: return "newton";
    }
    return "";
}

// XGBoost's split / leaf parameters for Criterion::Newton, with XGBoost's
// meaning and defaults. A node's score is score(G, H) = T(G)^2 / (H + lambda),
// T(G) = sign(G) * max(|G| - alpha, 0) (XGBoost's code scale: no 1/2); a split's
// gain is score(left) + score(right) - score(node).
struct NewtonParams {
    double reg_lambda = 1.0;
    double reg_alpha = 0.0;
    double gamma = 0.0;            // min_split_loss: a node splits only if its gain >= gamma
    double min_child_weight = 1.0; // minimum hessian sum in each child
    double max_delta_step = 0.0;   // > 0: leaf weights clipped to +-max_delta_step
};

struct TreeParams {
    Task task = Task::Classification;
    Criterion criterion = Criterion::Gini;
    int max_depth = 5;
    int min_samples_split = 2;
    int min_samples_leaf = 1;
    // A depth's winning feature is only committed if its total weighted
    // impurity decrease -- sum over the depth's nodes of
    // N_t/N * (node improvement), N = total training weight -- plus machine
    // epsilon reaches this threshold; otherwise every node at that depth
    // becomes a leaf. Same quantity and comparison as sklearn's
    // min_impurity_decrease (which gates each node separately rather than
    // each depth); XGBoost/LightGBM's closest equivalent is
    // gamma/min_split_loss, on a different (unnormalised gain) scale.
    double min_impurity_decrease = 0.0;
    // Criterion::Newton only (min_impurity_decrease is not used then: gamma
    // gates each node's split instead).
    NewtonParams newton;

    // Implementation choices that do not change which splits are chosen.
    //
    // presort: score every node of a depth in one pass over each feature's
    // presorted rows (presort.hpp) instead of sorting each (node, feature)
    // pair's samples at every depth. Both searches perform the same
    // floating-point operations in the same order (tied feature values are
    // visited in row order by both), so they produce identical trees. MAE
    // always uses the per-node search.
    bool presort = true;
    // Keep each Node's `indices` (the training rows that reached it) after
    // fit. rdt.visualize's export_tree_dict needs them. Gradient boosting and
    // bootstrapped forests turn this off -- they would otherwise retain
    // ~4 * n_samples * (max_depth + 1) bytes per fitted tree, and a bootstrap
    // tree's indices are positions in its bootstrap sample, not training rows.
    bool retain_indices = true;
};

} // namespace rdt
