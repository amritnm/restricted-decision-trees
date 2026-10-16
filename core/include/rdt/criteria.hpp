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

// Pass sums of values *centred* near their mean (see find_best_threshold's
// MSE branch): E[y^2] - E[y]^2 on raw values cancels catastrophically once
// |mean| >> spread (targets around 1e8 with unit spread lose every digit).
inline double mse_from_sums(double weighted_sum, double weighted_sum_sq, double total_weight) {
    if (total_weight <= 0.0) return 0.0;
    double mean = weighted_sum / total_weight;
    return std::max(0.0, weighted_sum_sq / total_weight - mean * mean);
}

// Threshold for a split between consecutive distinct sorted values a < b,
// applied as `x <= threshold` -> left. Normally the midpoint (computed as
// a/2 + b/2 so it cannot overflow), but -- scikit-learn's rule -- when that
// is not strictly inside [a, b) (rounds up to b for adjacent doubles, or is
// non-finite) use a itself. Either way exactly the values <= a go left, so
// the partition committed by split_indices()/predict() is the one scored.
inline double split_threshold(double a, double b) {
    double t = a / 2.0 + b / 2.0;
    if (!(t >= a && t < b)) t = a;
    return t;
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
    double raw_sum = 0.0, total_weight = 0.0;
    for (auto idx : indices) {
        double w = weights ? (*weights)[idx] : 1.0;
        raw_sum += w * y[idx];
        total_weight += w;
    }
    const double center = total_weight > 0.0 ? raw_sum / total_weight : 0.0;
    double sum = 0.0, sum_sq = 0.0;
    for (auto idx : indices) {
        double d = y[idx] - center;
        double w = weights ? (*weights)[idx] : 1.0;
        sum += w * d;
        sum_sq += w * d * d;
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
        case Criterion::Newton: break; // not an impurity of y
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

    // Tied feature values are ordered by row index. std::sort leaves the
    // order inside a run of ties unspecified (it differs between standard
    // libraries and input sizes), and that order fixes the summation order of
    // the running sums -- which, once they are inexact (centred MSE targets,
    // real-valued weights), decides the last bits of an improvement and so
    // the winner between equal candidates. Sorting by (value, row) makes the
    // search deterministic across platforms.
    struct SamplePoint { double x, y, w; std::uint32_t idx; };
    std::vector<SamplePoint> pts;
    pts.reserve(n);
    for (auto idx : indices) {
        pts.push_back({X(idx, feature_idx), y[idx], weights ? (*weights)[idx] : 1.0, idx});
    }
    std::sort(pts.begin(), pts.end(), [](const auto& a, const auto& b) {
        return a.x < b.x || (a.x == b.x && a.idx < b.idx);
    });

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
                best_threshold = split_threshold(pts[i].x, pts[i + 1].x);
                any_valid = true;
            }
        }
    } else if (criterion == Criterion::MSE) {
        // Running sums are kept on targets centred at the node's weighted
        // mean (two passes): variance is shift-invariant, and centring
        // removes the large-offset cancellation in mse_from_sums.
        double raw_sum = 0.0, total_weight = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            raw_sum += pts[i].w * pts[i].y;
            total_weight += pts[i].w;
        }
        const double center = total_weight > 0.0 ? raw_sum / total_weight : 0.0;
        for (auto& p : pts) p.y -= center;

        double total_sum = 0.0, total_sum_sq = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            total_sum += pts[i].w * pts[i].y;
            total_sum_sq += pts[i].w * pts[i].y * pts[i].y;
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
                best_threshold = split_threshold(pts[i].x, pts[i + 1].x);
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
                best_threshold = split_threshold(pts[i].x, pts[i + 1].x);
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

// ---------------------------------------------------------------- Newton gain
// XGBoost's second-order split scoring (src/tree/param.h, split_evaluator.h),
// on per-row gradients g = dL/dF and hessians h (already multiplied by the
// sample weight). G and H are a node's (or child's) sums.

// XGBoost's floor on a split's gain (kRtEps): a node splits only if its gain
// is > this (and >= gamma).
constexpr double kNewtonMinGain = 1e-6;

// T_alpha(G) = sign(G) * max(|G| - alpha, 0) (XGBoost's ThresholdL1).
inline double newton_threshold_l1(double G, double alpha) {
    if (G > alpha) return G - alpha;
    if (G < -alpha) return G + alpha;
    return 0.0;
}

// Leaf weight -T(G) / (H + lambda), clipped to +-max_delta_step when that is
// > 0; 0 when H <= 0 (XGBoost's CalcWeight). Not yet multiplied by the
// learning rate.
inline double newton_leaf_weight(double G, double H, const NewtonParams& p) {
    if (H <= 0.0) return 0.0;
    double w = -newton_threshold_l1(G, p.reg_alpha) / (H + p.reg_lambda);
    if (p.max_delta_step != 0.0 && std::abs(w) > p.max_delta_step) w = std::copysign(p.max_delta_step, w);
    return w;
}

// score(G, H) = T(G)^2 / (H + lambda) -- the loss reduction of the node's
// optimal weight. With max_delta_step > 0 the weight may be clipped, so (as
// XGBoost's CalcGain does) the score is evaluated at the clipped weight w:
// -(2 G w + (H + lambda) w^2 + 2 alpha |w|), which equals T(G)^2 / (H + lambda)
// when w is not clipped. 0 when H <= 0.
inline double newton_score(double G, double H, const NewtonParams& p) {
    if (H <= 0.0) return 0.0;
    if (p.max_delta_step == 0.0) {
        const double t = newton_threshold_l1(G, p.reg_alpha);
        return t * t / (H + p.reg_lambda);
    }
    const double w = newton_leaf_weight(G, H, p);
    return -(2.0 * G * w + (H + p.reg_lambda) * w * w + 2.0 * p.reg_alpha * std::abs(w));
}

// Both children need a positive hessian sum of at least min_child_weight
// (XGBoost's IsValidSplit).
inline bool newton_children_valid(double left_h, double right_h, const NewtonParams& p) {
    return left_h > 0.0 && right_h > 0.0 && left_h >= p.min_child_weight && right_h >= p.min_child_weight;
}

// Gain of splitting a node with totals (G, H) into left (left_g, left_h) and
// the rest: score(left) + score(right) - parent_score. The one expression both
// searches (find_best_threshold_newton and the presorted DepthScanner) use.
inline double newton_split_gain(double left_g, double left_h, double G, double H, double parent_score,
                                const NewtonParams& p) {
    return newton_score(left_g, left_h, p) + newton_score(G - left_g, H - left_h, p) - parent_score;
}

// Best threshold of one node on one feature by Newton gain: the candidate
// boundaries, guarded midpoint, tie order and min_samples_split/leaf counts of
// find_best_threshold, plus the hessian constraint; `improvement` is the best
// gain. Whether that gain is large enough to split (>= gamma, > kNewtonMinGain)
// is decided by the caller. `g`/`h` are indexed like `X`'s rows; G/H are the
// node's totals (summed in row order by the caller, so they do not depend on
// the feature). Candidates whose gain is not finite are skipped.
template <class Matrix>
ThresholdResult find_best_threshold_newton(
    const Matrix& X,
    const VectorView& g,
    const VectorView& h,
    const std::vector<std::uint32_t>& indices,
    int feature_idx,
    double G,
    double H,
    int min_samples_split,
    int min_samples_leaf,
    const NewtonParams& p) {

    ThresholdResult result;
    const std::size_t n = indices.size();
    if (static_cast<int>(n) < min_samples_split) return result;

    struct SamplePoint { double x, g, h; std::uint32_t idx; };
    std::vector<SamplePoint> pts;
    pts.reserve(n);
    for (auto idx : indices) pts.push_back({X(idx, feature_idx), g[idx], h[idx], idx});
    std::sort(pts.begin(), pts.end(), [](const auto& a, const auto& b) {
        return a.x < b.x || (a.x == b.x && a.idx < b.idx);
    });
    if (pts.front().x == pts.back().x) return result;

    const double parent_score = newton_score(G, H, p);
    double best_gain = -std::numeric_limits<double>::infinity();
    double best_threshold = 0.0;
    bool any_valid = false;
    double left_g = 0.0, left_h = 0.0;
    for (std::size_t i = 0; i + 1 < n; ++i) {
        left_g += pts[i].g;
        left_h += pts[i].h;
        if (pts[i].x == pts[i + 1].x) continue;

        std::size_t n_left = i + 1;
        std::size_t n_right = n - n_left;
        if (static_cast<int>(n_left) < min_samples_leaf ||
            static_cast<int>(n_right) < min_samples_leaf) continue;
        if (!newton_children_valid(left_h, H - left_h, p)) continue;

        const double gain = newton_split_gain(left_g, left_h, G, H, parent_score, p);
        if (std::isfinite(gain) && gain > best_gain) {
            best_gain = gain;
            best_threshold = split_threshold(pts[i].x, pts[i + 1].x);
            any_valid = true;
        }
    }
    if (!any_valid) return result;
    result.found = true;
    result.threshold = best_threshold;
    result.improvement = best_gain;
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
