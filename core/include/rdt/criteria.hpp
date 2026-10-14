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
//
// Every criterion function accepts an optional `weights` (nullptr = every
// sample has weight 1.0, which is arithmetically identical to the original
// unweighted formulas -- verified by the existing Catch2/pytest suites
// passing unchanged). `min_samples_split`/`min_samples_leaf` stay literal
// sample *counts*, unaffected by weights, matching sklearn's convention
// (weighted-count leaf constraints are a separate, not-yet-added parameter).
namespace rdt {

// `weighted_counts` are summed in the order given; callers that need results
// bit-identical to each other must present the classes in the same order
// (find_best_threshold and the presorted search both use first-encountered
// order along the feature-sorted samples).
inline double gini_from_counts(const double* weighted_counts, std::size_t n_counts, double total_weight) {
    if (total_weight <= 0.0) return 0.0;
    double sum_sq = 0.0;
    for (std::size_t i = 0; i < n_counts; ++i) {
        double p = weighted_counts[i] / total_weight;
        sum_sq += p * p;
    }
    return 1.0 - sum_sq;
}

inline double entropy_from_counts(const double* weighted_counts, std::size_t n_counts, double total_weight) {
    if (total_weight <= 0.0) return 0.0;
    double h = 0.0;
    for (std::size_t i = 0; i < n_counts; ++i) {
        double p = weighted_counts[i] / total_weight;
        h -= p * std::log2(p + 1e-10);
    }
    return h;
}

inline double gini_from_counts(const std::vector<double>& weighted_counts, double total_weight) {
    return gini_from_counts(weighted_counts.data(), weighted_counts.size(), total_weight);
}

inline double entropy_from_counts(const std::vector<double>& weighted_counts, double total_weight) {
    return entropy_from_counts(weighted_counts.data(), weighted_counts.size(), total_weight);
}

inline double mse_from_sums(double weighted_sum, double weighted_sum_sq, double total_weight) {
    if (total_weight <= 0.0) return 0.0;
    double mean = weighted_sum / total_weight;
    return weighted_sum_sq / total_weight - mean * mean;
}

// `value_weight` need not be pre-sorted; sorted internally by value.
// Reduces exactly to the unweighted median (average-the-two-middle-values
// for even n) when every weight is 1.0.
inline double mae_from_weighted_values(std::vector<std::pair<double, double>> value_weight) {
    if (value_weight.empty()) return 0.0;
    std::sort(value_weight.begin(), value_weight.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    double total_weight = 0.0;
    for (const auto& vw : value_weight) total_weight += vw.second;
    if (total_weight <= 0.0) return 0.0;

    const double half = total_weight / 2.0;
    double median = value_weight.back().first;
    double cum = 0.0;
    for (std::size_t i = 0; i < value_weight.size(); ++i) {
        cum += value_weight[i].second;
        if (cum > half) {
            median = value_weight[i].first;
            break;
        }
        if (cum == half) {
            median = (i + 1 < value_weight.size())
                ? (value_weight[i].first + value_weight[i + 1].first) / 2.0
                : value_weight[i].first;
            break;
        }
    }

    double total = 0.0;
    for (const auto& vw : value_weight) total += vw.second * std::abs(vw.first - median);
    return total / total_weight;
}

// Public, testable impurity functions operating on an arbitrary index list.
// Not used in the hot threshold-search loop (see find_best_threshold below),
// which maintains equivalent running aggregates incrementally instead.
inline double gini(const VectorView& y, const std::vector<std::uint32_t>& indices,
                    const VectorView* weights = nullptr) {
    std::unordered_map<double, double> class_counts;
    for (auto idx : indices) class_counts[y[idx]] += weights ? (*weights)[idx] : 1.0;
    std::vector<double> counts;
    counts.reserve(class_counts.size());
    double total = 0.0;
    for (auto& kv : class_counts) { counts.push_back(kv.second); total += kv.second; }
    return gini_from_counts(counts, total);
}

inline double entropy(const VectorView& y, const std::vector<std::uint32_t>& indices,
                       const VectorView* weights = nullptr) {
    std::unordered_map<double, double> class_counts;
    for (auto idx : indices) class_counts[y[idx]] += weights ? (*weights)[idx] : 1.0;
    std::vector<double> counts;
    counts.reserve(class_counts.size());
    double total = 0.0;
    for (auto& kv : class_counts) { counts.push_back(kv.second); total += kv.second; }
    return entropy_from_counts(counts, total);
}

inline double mse(const VectorView& y, const std::vector<std::uint32_t>& indices,
                   const VectorView* weights = nullptr) {
    double sum = 0.0, sum_sq = 0.0, total_weight = 0.0;
    for (auto idx : indices) {
        double v = y[idx];
        double w = weights ? (*weights)[idx] : 1.0;
        sum += w * v;
        sum_sq += w * v * v;
        total_weight += w;
    }
    return mse_from_sums(sum, sum_sq, total_weight);
}

inline double mae(const VectorView& y, const std::vector<std::uint32_t>& indices,
                   const VectorView* weights = nullptr) {
    std::vector<std::pair<double, double>> value_weight;
    value_weight.reserve(indices.size());
    for (auto idx : indices) value_weight.emplace_back(y[idx], weights ? (*weights)[idx] : 1.0);
    return mae_from_weighted_values(std::move(value_weight));
}

inline double impurity(Criterion c, const VectorView& y, const std::vector<std::uint32_t>& indices,
                        const VectorView* weights = nullptr) {
    switch (c) {
        case Criterion::Gini: return gini(y, indices, weights);
        case Criterion::Entropy: return entropy(y, indices, weights);
        case Criterion::MSE: return mse(y, indices, weights);
        case Criterion::MAE: return mae(y, indices, weights);
    }
    return 0.0;
}

struct ThresholdResult {
    bool found = false;
    double threshold = 0.0;
    double improvement = 0.0;
};

// Best split threshold (and its impurity improvement) for one feature at one
// node. `indices` are the sample indices belonging to the node. `weights`
// (nullptr = uniform 1.0) are indexed the same way as `y`/`X`, i.e. by the
// original sample index, not by position within `indices`.
//
// This per-(node, feature) sort is the reference algorithm. RDTModel normally
// scores all nodes of a depth at once from presorted columns instead (see
// presort.hpp), which yields the same thresholds and improvements; this path
// is still used for MAE and when presorting is disabled. `Matrix` is
// MatrixView or IndexedMatrixView.
template <class Matrix>
ThresholdResult find_best_threshold(
    const Matrix& X,
    const VectorView& y,
    const std::vector<std::uint32_t>& indices,
    int feature_idx,
    Criterion criterion,
    int min_samples_split,
    int min_samples_leaf,
    const VectorView* weights = nullptr) {

    ThresholdResult result;
    const std::size_t n = indices.size();
    if (static_cast<int>(n) < min_samples_split) return result;

    struct SamplePoint { double x, y, w; };
    std::vector<SamplePoint> pts;
    pts.reserve(n);
    for (auto idx : indices) {
        pts.push_back({X(idx, feature_idx), y[idx], weights ? (*weights)[idx] : 1.0});
    }
    std::sort(pts.begin(), pts.end(), [](const auto& a, const auto& b) { return a.x < b.x; });

    if (pts.front().x == pts.back().x) return result; // < 2 unique values

    double best_improvement = -std::numeric_limits<double>::infinity();
    double best_threshold = 0.0;
    bool any_valid = false;

    if (criterion == Criterion::Gini || criterion == Criterion::Entropy) {
        std::unordered_map<double, int> class_index;
        std::vector<int> sample_class(n);
        std::vector<double> total_counts;
        for (std::size_t i = 0; i < n; ++i) {
            double label = pts[i].y;
            auto it = class_index.find(label);
            int idx;
            if (it == class_index.end()) {
                idx = static_cast<int>(total_counts.size());
                class_index.emplace(label, idx);
                total_counts.push_back(0.0);
            } else {
                idx = it->second;
            }
            sample_class[i] = idx;
            total_counts[idx] += pts[i].w;
        }
        double total_weight = 0.0;
        for (double c : total_counts) total_weight += c;

        const double parent_impurity = (criterion == Criterion::Gini)
            ? gini_from_counts(total_counts, total_weight)
            : entropy_from_counts(total_counts, total_weight);

        std::vector<double> left_counts(total_counts.size(), 0.0);
        std::vector<double> right_counts = total_counts;
        double left_weight = 0.0, right_weight = total_weight;

        for (std::size_t i = 0; i + 1 < n; ++i) {
            int cls = sample_class[i];
            left_counts[cls] += pts[i].w;
            right_counts[cls] -= pts[i].w;
            left_weight += pts[i].w;
            right_weight -= pts[i].w;

            if (pts[i].x == pts[i + 1].x) continue; // not a real boundary

            std::size_t n_left = i + 1;
            std::size_t n_right = n - n_left;
            if (static_cast<int>(n_left) < min_samples_leaf ||
                static_cast<int>(n_right) < min_samples_leaf) continue;

            double left_imp = (criterion == Criterion::Gini)
                ? gini_from_counts(left_counts, left_weight)
                : entropy_from_counts(left_counts, left_weight);
            double right_imp = (criterion == Criterion::Gini)
                ? gini_from_counts(right_counts, right_weight)
                : entropy_from_counts(right_counts, right_weight);

            double weighted = (left_weight / total_weight) * left_imp +
                               (right_weight / total_weight) * right_imp;
            double improvement = parent_impurity - weighted;

            if (improvement > best_improvement) {
                best_improvement = improvement;
                best_threshold = (pts[i].x + pts[i + 1].x) / 2.0;
                any_valid = true;
            }
        }
    } else if (criterion == Criterion::MSE) {
        double total_sum = 0.0, total_sum_sq = 0.0, total_weight = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            total_sum += pts[i].w * pts[i].y;
            total_sum_sq += pts[i].w * pts[i].y * pts[i].y;
            total_weight += pts[i].w;
        }
        const double parent_impurity = mse_from_sums(total_sum, total_sum_sq, total_weight);

        double left_sum = 0.0, left_sum_sq = 0.0, left_weight = 0.0;
        for (std::size_t i = 0; i + 1 < n; ++i) {
            left_sum += pts[i].w * pts[i].y;
            left_sum_sq += pts[i].w * pts[i].y * pts[i].y;
            left_weight += pts[i].w;

            if (pts[i].x == pts[i + 1].x) continue;

            std::size_t n_left = i + 1;
            std::size_t n_right = n - n_left;
            if (static_cast<int>(n_left) < min_samples_leaf ||
                static_cast<int>(n_right) < min_samples_leaf) continue;

            double right_sum = total_sum - left_sum;
            double right_sum_sq = total_sum_sq - left_sum_sq;
            double right_weight = total_weight - left_weight;

            double left_imp = mse_from_sums(left_sum, left_sum_sq, left_weight);
            double right_imp = mse_from_sums(right_sum, right_sum_sq, right_weight);

            double weighted = (left_weight / total_weight) * left_imp +
                               (right_weight / total_weight) * right_imp;
            double improvement = parent_impurity - weighted;

            if (improvement > best_improvement) {
                best_improvement = improvement;
                best_threshold = (pts[i].x + pts[i + 1].x) / 2.0;
                any_valid = true;
            }
        }
    } else { // MAE: no cheap incremental update for the weighted median; slice + recompute.
        std::vector<std::pair<double, double>> all_vw(n);
        for (std::size_t i = 0; i < n; ++i) all_vw[i] = {pts[i].y, pts[i].w};
        const double parent_impurity = mae_from_weighted_values(all_vw);

        for (std::size_t i = 0; i + 1 < n; ++i) {
            if (pts[i].x == pts[i + 1].x) continue;

            std::size_t n_left = i + 1;
            std::size_t n_right = n - n_left;
            if (static_cast<int>(n_left) < min_samples_leaf ||
                static_cast<int>(n_right) < min_samples_leaf) continue;

            std::vector<std::pair<double, double>> left_vw(all_vw.begin(), all_vw.begin() + n_left);
            std::vector<std::pair<double, double>> right_vw(all_vw.begin() + n_left, all_vw.end());

            double left_weight = 0.0, right_weight = 0.0;
            for (const auto& vw : left_vw) left_weight += vw.second;
            for (const auto& vw : right_vw) right_weight += vw.second;
            double total_weight = left_weight + right_weight;

            double left_imp = mae_from_weighted_values(std::move(left_vw));
            double right_imp = mae_from_weighted_values(std::move(right_vw));

            double weighted = (left_weight / total_weight) * left_imp +
                               (right_weight / total_weight) * right_imp;
            double improvement = parent_impurity - weighted;

            if (improvement > best_improvement) {
                best_improvement = improvement;
                best_threshold = (pts[i].x + pts[i + 1].x) / 2.0;
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
template <class Matrix>
void split_indices(
    const Matrix& X,
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
