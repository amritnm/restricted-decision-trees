#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "rdt/matrix.hpp"
#include "rdt/params.hpp"

// Impurity criteria and the per-(node, feature) best-threshold search.
//
// This mirrors rdt_optimal_cython/optimal_rdt_cython.py's
// _find_best_threshold_for_node_feature / _calculate_criterion faithfully in
// terms of *what* is computed (same candidate thresholds, same ascending
// scan order, same strict '>' tie-break so the first-encountered threshold
// among ties wins) but computes it more efficiently: samples are sorted by
// feature value once per (node, feature) call and class counts / sum /
// sum-of-squares are updated incrementally while sweeping the split point,
// instead of re-scanning all n samples with a boolean mask for every
// candidate threshold the way the Python reference does.
namespace rdt {

inline double gini_from_counts(const std::vector<long long>& counts, std::size_t n) {
    if (n == 0) return 0.0;
    double sum_sq = 0.0;
    for (long long c : counts) {
        double p = static_cast<double>(c) / static_cast<double>(n);
        sum_sq += p * p;
    }
    return 1.0 - sum_sq;
}

inline double entropy_from_counts(const std::vector<long long>& counts, std::size_t n) {
    if (n == 0) return 0.0;
    double h = 0.0;
    for (long long c : counts) {
        double p = static_cast<double>(c) / static_cast<double>(n);
        h -= p * std::log2(p + 1e-10);
    }
    return h;
}

inline double mse_from_sums(double sum, double sum_sq, std::size_t n) {
    if (n == 0) return 0.0;
    double mean = sum / static_cast<double>(n);
    return sum_sq / static_cast<double>(n) - mean * mean;
}

inline double mae_from_values(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    std::size_t n = values.size();
    double median = (n % 2 == 1)
        ? values[n / 2]
        : (values[n / 2 - 1] + values[n / 2]) / 2.0;
    double total = 0.0;
    for (double v : values) total += std::abs(v - median);
    return total / static_cast<double>(n);
}

// Public, testable impurity functions operating on an arbitrary index list.
// Not used in the hot threshold-search loop (see find_best_threshold below),
// which maintains equivalent running aggregates incrementally instead.
inline double gini(const VectorView& y, const std::vector<std::uint32_t>& indices) {
    std::unordered_map<double, long long> class_counts;
    for (auto idx : indices) class_counts[y[idx]]++;
    std::vector<long long> counts;
    counts.reserve(class_counts.size());
    for (auto& kv : class_counts) counts.push_back(kv.second);
    return gini_from_counts(counts, indices.size());
}

inline double entropy(const VectorView& y, const std::vector<std::uint32_t>& indices) {
    std::unordered_map<double, long long> class_counts;
    for (auto idx : indices) class_counts[y[idx]]++;
    std::vector<long long> counts;
    counts.reserve(class_counts.size());
    for (auto& kv : class_counts) counts.push_back(kv.second);
    return entropy_from_counts(counts, indices.size());
}

inline double mse(const VectorView& y, const std::vector<std::uint32_t>& indices) {
    double sum = 0.0, sum_sq = 0.0;
    for (auto idx : indices) {
        double v = y[idx];
        sum += v;
        sum_sq += v * v;
    }
    return mse_from_sums(sum, sum_sq, indices.size());
}

inline double mae(const VectorView& y, const std::vector<std::uint32_t>& indices) {
    std::vector<double> values;
    values.reserve(indices.size());
    for (auto idx : indices) values.push_back(y[idx]);
    return mae_from_values(std::move(values));
}

inline double impurity(Criterion c, const VectorView& y, const std::vector<std::uint32_t>& indices) {
    switch (c) {
        case Criterion::Gini: return gini(y, indices);
        case Criterion::Entropy: return entropy(y, indices);
        case Criterion::MSE: return mse(y, indices);
        case Criterion::MAE: return mae(y, indices);
    }
    return 0.0;
}

struct ThresholdResult {
    bool found = false;
    double threshold = 0.0;
    double improvement = 0.0;
};

// Best split threshold (and its impurity improvement) for one feature at one
// node. `indices` are the sample indices belonging to the node.
inline ThresholdResult find_best_threshold(
    const MatrixView& X,
    const VectorView& y,
    const std::vector<std::uint32_t>& indices,
    int feature_idx,
    Criterion criterion,
    int min_samples_split,
    int min_samples_leaf) {

    ThresholdResult result;
    const std::size_t n = indices.size();
    if (static_cast<int>(n) < min_samples_split) return result;

    std::vector<std::pair<double, double>> pairs; // (feature_value, y_value)
    pairs.reserve(n);
    for (auto idx : indices) pairs.emplace_back(X(idx, feature_idx), y[idx]);
    std::sort(pairs.begin(), pairs.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    if (pairs.front().first == pairs.back().first) return result; // < 2 unique values

    double best_improvement = -std::numeric_limits<double>::infinity();
    double best_threshold = 0.0;
    bool any_valid = false;

    if (criterion == Criterion::Gini || criterion == Criterion::Entropy) {
        std::unordered_map<double, int> class_index;
        std::vector<int> sample_class(n);
        std::vector<long long> total_counts;
        for (std::size_t i = 0; i < n; ++i) {
            double label = pairs[i].second;
            auto it = class_index.find(label);
            int idx;
            if (it == class_index.end()) {
                idx = static_cast<int>(total_counts.size());
                class_index.emplace(label, idx);
                total_counts.push_back(0);
            } else {
                idx = it->second;
            }
            sample_class[i] = idx;
            total_counts[idx]++;
        }

        const double parent_impurity = (criterion == Criterion::Gini)
            ? gini_from_counts(total_counts, n)
            : entropy_from_counts(total_counts, n);

        std::vector<long long> left_counts(total_counts.size(), 0);
        std::vector<long long> right_counts = total_counts;

        for (std::size_t i = 0; i + 1 < n; ++i) {
            int cls = sample_class[i];
            left_counts[cls]++;
            right_counts[cls]--;

            if (pairs[i].first == pairs[i + 1].first) continue; // not a real boundary

            std::size_t n_left = i + 1;
            std::size_t n_right = n - n_left;
            if (static_cast<int>(n_left) < min_samples_leaf ||
                static_cast<int>(n_right) < min_samples_leaf) continue;

            double left_imp = (criterion == Criterion::Gini)
                ? gini_from_counts(left_counts, n_left)
                : entropy_from_counts(left_counts, n_left);
            double right_imp = (criterion == Criterion::Gini)
                ? gini_from_counts(right_counts, n_right)
                : entropy_from_counts(right_counts, n_right);

            double weighted = (static_cast<double>(n_left) / n) * left_imp +
                               (static_cast<double>(n_right) / n) * right_imp;
            double improvement = parent_impurity - weighted;

            if (improvement > best_improvement) {
                best_improvement = improvement;
                best_threshold = (pairs[i].first + pairs[i + 1].first) / 2.0;
                any_valid = true;
            }
        }
    } else if (criterion == Criterion::MSE) {
        double total_sum = 0.0, total_sum_sq = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            total_sum += pairs[i].second;
            total_sum_sq += pairs[i].second * pairs[i].second;
        }
        const double parent_impurity = mse_from_sums(total_sum, total_sum_sq, n);

        double left_sum = 0.0, left_sum_sq = 0.0;
        for (std::size_t i = 0; i + 1 < n; ++i) {
            left_sum += pairs[i].second;
            left_sum_sq += pairs[i].second * pairs[i].second;

            if (pairs[i].first == pairs[i + 1].first) continue;

            std::size_t n_left = i + 1;
            std::size_t n_right = n - n_left;
            if (static_cast<int>(n_left) < min_samples_leaf ||
                static_cast<int>(n_right) < min_samples_leaf) continue;

            double right_sum = total_sum - left_sum;
            double right_sum_sq = total_sum_sq - left_sum_sq;

            double left_imp = mse_from_sums(left_sum, left_sum_sq, n_left);
            double right_imp = mse_from_sums(right_sum, right_sum_sq, n_right);

            double weighted = (static_cast<double>(n_left) / n) * left_imp +
                               (static_cast<double>(n_right) / n) * right_imp;
            double improvement = parent_impurity - weighted;

            if (improvement > best_improvement) {
                best_improvement = improvement;
                best_threshold = (pairs[i].first + pairs[i + 1].first) / 2.0;
                any_valid = true;
            }
        }
    } else { // MAE: no cheap incremental update for the median; slice + recompute.
        std::vector<double> all_y(n);
        for (std::size_t i = 0; i < n; ++i) all_y[i] = pairs[i].second;
        const double parent_impurity = mae_from_values(all_y);

        for (std::size_t i = 0; i + 1 < n; ++i) {
            if (pairs[i].first == pairs[i + 1].first) continue;

            std::size_t n_left = i + 1;
            std::size_t n_right = n - n_left;
            if (static_cast<int>(n_left) < min_samples_leaf ||
                static_cast<int>(n_right) < min_samples_leaf) continue;

            std::vector<double> left_vals(all_y.begin(), all_y.begin() + n_left);
            std::vector<double> right_vals(all_y.begin() + n_left, all_y.end());

            double left_imp = mae_from_values(std::move(left_vals));
            double right_imp = mae_from_values(std::move(right_vals));

            double weighted = (static_cast<double>(n_left) / n) * left_imp +
                               (static_cast<double>(n_right) / n) * right_imp;
            double improvement = parent_impurity - weighted;

            if (improvement > best_improvement) {
                best_improvement = improvement;
                best_threshold = (pairs[i].first + pairs[i + 1].first) / 2.0;
                any_valid = true;
            }
        }
    }

    if (!any_valid) return result;
    result.found = true;
    result.threshold = best_threshold;
    result.improvement = best_improvement;
    return result;
}

// Partition `indices` (in their original relative order) into left/right
// children for a chosen (feature, threshold), matching the Python
// reference's order-preserving boolean-mask split exactly.
inline void split_indices(
    const MatrixView& X,
    const std::vector<std::uint32_t>& indices,
    int feature_idx,
    double threshold,
    std::vector<std::uint32_t>& left,
    std::vector<std::uint32_t>& right) {
    left.clear();
    right.clear();
    for (auto idx : indices) {
        if (X(idx, feature_idx) <= threshold) left.push_back(idx);
        else right.push_back(idx);
    }
}

} // namespace rdt
