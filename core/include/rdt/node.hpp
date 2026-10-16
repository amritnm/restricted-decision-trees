#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace rdt {

// Mirrors optimal_rdt_cython.py's Node: a lightweight node holding the
// sample indices that reach it, and (once split) its children.
struct Node {
    // Training rows that reached this node, ascending. Released after fit
    // (has_indices == false) when TreeParams::retain_indices is false -- the
    // boosting trees and bootstrapped forest trees; n_samples always keeps
    // the count.
    std::vector<std::uint32_t> indices;
    std::size_t n_samples = 0;
    bool has_indices = true;

    std::unique_ptr<Node> left;
    std::unique_ptr<Node> right;

    bool is_split = false;
    int split_feature = -1;
    double split_threshold = 0.0;
    // Hist trees: rows whose bin of split_feature is <= split_bin go left --
    // the same rows as x <= split_threshold (the bin's upper cut). -1 otherwise.
    std::int32_t split_bin = -1;

    bool has_value = false;
    double value = 0.0; // majority class label (classification) or mean (regression)

    // Class counts for predict_proba, aligned with RDTModel::classes(). Only
    // populated on leaves for classification tasks.
    std::vector<double> class_counts;

    bool is_leaf() const { return left == nullptr && right == nullptr; }
};

} // namespace rdt
