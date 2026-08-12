#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace rdt {

// Mirrors optimal_rdt_cython.py's Node: a lightweight node holding the
// sample indices that reach it, and (once split) its children.
struct Node {
    std::vector<std::uint32_t> indices;

    std::unique_ptr<Node> left;
    std::unique_ptr<Node> right;

    bool is_split = false;
    int split_feature = -1;
    double split_threshold = 0.0;

    bool has_value = false;
    double value = 0.0; // majority class label (classification) or mean (regression)

    // Class counts for predict_proba, aligned with RDTModel::classes(). Only
    // populated on leaves for classification tasks.
    std::vector<double> class_counts;

    bool is_leaf() const { return left == nullptr && right == nullptr; }
};

} // namespace rdt
