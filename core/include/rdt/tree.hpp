#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "rdt/criteria.hpp"
#include "rdt/matrix.hpp"
#include "rdt/node.hpp"
#include "rdt/params.hpp"

// Port of rdt_optimal_cython/optimal_rdt_cython.py's OptimalRDTCython.
//
// Per-depth greedy search: for each depth, every candidate feature is
// scored across all currently-active nodes at that depth (summing each
// node's best achievable impurity improvement for the feature); the feature
// with the highest total wins and is locked in for the whole depth, and the
// per-node thresholds already found while scoring it are reused directly as
// the final split points (see the plan's "two-pass" note: this is
// result-equivalent to the paper's Algorithm 1, not an approximation of it).
namespace rdt {

class RDTModel {
public:
    explicit RDTModel(TreeParams params) : params_(params) {}

    void fit(const MatrixView& X, const VectorView& y) {
        n_features_ = X.n_cols();
        depth_features_.clear();
        classes_.clear();

        if (params_.task == Task::Classification) {
            std::vector<double> labels(y.size());
            for (std::size_t i = 0; i < y.size(); ++i) labels[i] = y[i];
            std::sort(labels.begin(), labels.end());
            labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
            classes_ = std::move(labels);
        }

        std::vector<std::uint32_t> root_indices(X.n_rows());
        for (std::size_t i = 0; i < root_indices.size(); ++i) root_indices[i] = static_cast<std::uint32_t>(i);

        root_ = build_tree(X, y, std::move(root_indices));
    }

    std::vector<double> predict(const MatrixView& X) const {
        std::vector<double> out(X.n_rows());
        for (std::size_t i = 0; i < X.n_rows(); ++i) out[i] = predict_one(X, i, root_.get());
        return out;
    }

    // n_samples x n_classes, classes ordered per classes().
    std::vector<std::vector<double>> predict_proba(const MatrixView& X) const {
        std::vector<std::vector<double>> out(X.n_rows(), std::vector<double>(classes_.size(), 0.0));
        for (std::size_t i = 0; i < X.n_rows(); ++i) out[i] = predict_proba_one(X, i, root_.get());
        return out;
    }

    const std::vector<double>& classes() const { return classes_; }
    std::size_t n_classes() const { return classes_.size(); }
    const std::map<int, int>& depth_features() const { return depth_features_; }
    const Node* root() const { return root_.get(); }
    const TreeParams& params() const { return params_; }

private:
    TreeParams params_;
    std::unique_ptr<Node> root_;
    std::vector<double> classes_;
    std::map<int, int> depth_features_;
    std::size_t n_features_ = 0;

    static bool all_same_label(const VectorView& y, const std::vector<std::uint32_t>& indices) {
        if (indices.empty()) return true;
        double first = y[indices[0]];
        for (auto idx : indices) {
            if (y[idx] != first) return false;
        }
        return true;
    }

    bool should_split(const Node& node, int depth) const {
        if (depth >= params_.max_depth) return false;
        if (static_cast<int>(node.indices.size()) < params_.min_samples_split) return false;
        return true; // caller checks all_same_label separately (needs y)
    }

    void make_leaf(Node& node, const VectorView& y) const {
        node.has_value = true;
        if (params_.task == Task::Classification) {
            std::vector<double> seen_labels;
            std::unordered_map<double, std::size_t> label_to_pos;
            std::vector<long long> counts;
            for (auto idx : node.indices) {
                double v = y[idx];
                auto it = label_to_pos.find(v);
                if (it == label_to_pos.end()) {
                    label_to_pos.emplace(v, seen_labels.size());
                    seen_labels.push_back(v);
                    counts.push_back(1);
                } else {
                    counts[it->second]++;
                }
            }
            std::size_t best_pos = 0;
            long long best_count = counts.empty() ? 0 : counts[0];
            for (std::size_t i = 1; i < counts.size(); ++i) {
                if (counts[i] > best_count) {
                    best_count = counts[i];
                    best_pos = i;
                }
            }
            node.value = seen_labels.empty() ? 0.0 : seen_labels[best_pos];

            node.class_counts.assign(classes_.size(), 0.0);
            for (std::size_t i = 0; i < seen_labels.size(); ++i) {
                auto it = std::lower_bound(classes_.begin(), classes_.end(), seen_labels[i]);
                std::size_t cls_idx = static_cast<std::size_t>(it - classes_.begin());
                node.class_counts[cls_idx] = static_cast<double>(counts[i]);
            }
        } else {
            double sum = 0.0;
            for (auto idx : node.indices) sum += y[idx];
            node.value = node.indices.empty() ? 0.0 : sum / static_cast<double>(node.indices.size());
        }
    }

    // Returns (best_feature or -1, per-node thresholds keyed by Node*).
    std::pair<int, std::unordered_map<Node*, double>> find_optimal_feature_for_depth(
        const MatrixView& X, const VectorView& y, const std::vector<Node*>& nodes) const {

        int best_feature = -1;
        double best_total_improvement = -std::numeric_limits<double>::infinity();
        std::unordered_map<Node*, double> best_node_thresholds;

        for (int feature_idx = 0; feature_idx < static_cast<int>(n_features_); ++feature_idx) {
            double total_improvement = 0.0;
            std::unordered_map<Node*, double> node_thresholds;
            bool valid_split_found = false;

            for (Node* node : nodes) {
                ThresholdResult r = find_best_threshold(
                    X, y, node->indices, feature_idx, params_.criterion,
                    params_.min_samples_split, params_.min_samples_leaf);
                if (r.found) {
                    total_improvement += r.improvement;
                    node_thresholds[node] = r.threshold;
                    valid_split_found = true;
                }
            }

            if (valid_split_found && total_improvement > best_total_improvement) {
                best_total_improvement = total_improvement;
                best_feature = feature_idx;
                best_node_thresholds = std::move(node_thresholds);
            }
        }

        return {best_feature, std::move(best_node_thresholds)};
    }

    std::unique_ptr<Node> build_tree(const MatrixView& X, const VectorView& y, std::vector<std::uint32_t> root_indices) {
        auto root = std::make_unique<Node>();
        root->indices = std::move(root_indices);

        std::vector<Node*> current_level = {root.get()};

        for (int depth = 0; depth < params_.max_depth; ++depth) {
            if (current_level.empty()) break;

            std::vector<Node*> nodes_to_split;
            for (Node* node : current_level) {
                if (should_split(*node, depth) && !all_same_label(y, node->indices)) {
                    nodes_to_split.push_back(node);
                } else {
                    make_leaf(*node, y);
                }
            }

            if (nodes_to_split.empty()) break;

            auto [best_feature, node_thresholds] = find_optimal_feature_for_depth(X, y, nodes_to_split);

            if (best_feature < 0) {
                for (Node* node : nodes_to_split) make_leaf(*node, y);
                break;
            }

            depth_features_[depth] = best_feature;

            std::vector<Node*> next_level;
            for (Node* node : nodes_to_split) {
                auto it = node_thresholds.find(node);
                if (it == node_thresholds.end()) {
                    make_leaf(*node, y);
                    continue;
                }
                double threshold = it->second;

                std::vector<std::uint32_t> left_indices, right_indices;
                split_indices(X, node->indices, best_feature, threshold, left_indices, right_indices);

                if (left_indices.empty() || right_indices.empty()) {
                    make_leaf(*node, y);
                    continue;
                }

                node->is_split = true;
                node->split_feature = best_feature;
                node->split_threshold = threshold;

                node->left = std::make_unique<Node>();
                node->left->indices = std::move(left_indices);
                node->right = std::make_unique<Node>();
                node->right->indices = std::move(right_indices);

                next_level.push_back(node->left.get());
                next_level.push_back(node->right.get());
            }

            current_level = std::move(next_level);
        }

        for (Node* node : current_level) {
            if (!node->has_value) make_leaf(*node, y);
        }

        return root;
    }

    double predict_one(const MatrixView& X, std::size_t row, const Node* node) const {
        while (!node->is_leaf()) {
            node = (X(row, node->split_feature) <= node->split_threshold) ? node->left.get() : node->right.get();
        }
        return node->value;
    }

    std::vector<double> predict_proba_one(const MatrixView& X, std::size_t row, const Node* node) const {
        while (!node->is_leaf()) {
            node = (X(row, node->split_feature) <= node->split_threshold) ? node->left.get() : node->right.get();
        }
        std::vector<double> probs(classes_.size(), 0.0);
        double total = 0.0;
        for (double c : node->class_counts) total += c;
        if (total > 0.0) {
            for (std::size_t i = 0; i < probs.size(); ++i) probs[i] = node->class_counts[i] / total;
        }
        return probs;
    }
};

} // namespace rdt
