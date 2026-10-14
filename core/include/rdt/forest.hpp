#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "rdt/matrix.hpp"
#include "rdt/params.hpp"
#include "rdt/rng.hpp"
#include "rdt/thread_pool.hpp"
#include "rdt/tree.hpp"

// Random Forest of RDTs.
//
// RDT's structural constraint (every node at a depth splits on the same
// feature) means column subsampling (max_features/colsample_bytree) can't
// work sklearn's usual way -- a fresh random subset *per node*. Instead,
// each tree gets ONE random column subset chosen before its fit() call
// (matching the project's earlier Python ensemble/random_forest_rdt.py),
// which means every fitted sub-tree's split_feature/depth_features_ are
// indices into *that tree's own reduced column space*, not the original
// feature space -- predict(), predict_proba(), and feature_importances_ all
// have to remap through each tree's feature_subsets_[t] to get back to
// original column indices. This is the one subtlety in this file; get the
// remapping right and everything else is a fairly ordinary bagging loop.
namespace rdt {

enum class MaxFeaturesKind { Sqrt, Log2, All, Int, Float };

struct MaxFeaturesSpec {
    MaxFeaturesKind kind = MaxFeaturesKind::Sqrt;
    int int_value = 0;
    double float_value = 0.0;

    static MaxFeaturesSpec Sqrt() { return {MaxFeaturesKind::Sqrt, 0, 0.0}; }
    static MaxFeaturesSpec Log2() { return {MaxFeaturesKind::Log2, 0, 0.0}; }
    static MaxFeaturesSpec All() { return {MaxFeaturesKind::All, 0, 0.0}; }
    static MaxFeaturesSpec FromInt(int v) { return {MaxFeaturesKind::Int, v, 0.0}; }
    static MaxFeaturesSpec FromFloat(double v) { return {MaxFeaturesKind::Float, 0, v}; }
};

inline std::uint32_t resolve_max_features(const MaxFeaturesSpec& spec, std::uint32_t n_features) {
    double result = 1.0;
    switch (spec.kind) {
        case MaxFeaturesKind::Sqrt: result = std::sqrt(static_cast<double>(n_features)); break;
        case MaxFeaturesKind::Log2: result = std::log2(static_cast<double>(n_features)); break;
        case MaxFeaturesKind::All: result = static_cast<double>(n_features); break;
        case MaxFeaturesKind::Int: result = static_cast<double>(spec.int_value); break;
        case MaxFeaturesKind::Float: result = spec.float_value * n_features; break;
    }
    std::uint32_t k = static_cast<std::uint32_t>(std::max(1.0, std::floor(result)));
    return std::max<std::uint32_t>(1, std::min(k, n_features));
}

// Materializes an owning contiguous buffer for X[row_indices, col_indices]
// (row_indices == nullptr means "all rows"). Used at both fit time (bagging)
// and predict time (each tree needs the same column subset it was trained
// on) so there's exactly one place this logic lives.
struct SubMatrix {
    std::vector<double> buffer;
    MatrixView view;
};

inline SubMatrix extract_submatrix(const MatrixView& X, const std::vector<std::uint32_t>* row_indices,
                                    const std::vector<std::uint32_t>& col_indices) {
    std::size_t n_rows = row_indices ? row_indices->size() : X.n_rows();
    std::size_t n_cols = col_indices.size();
    SubMatrix out;
    out.buffer.resize(n_rows * n_cols);
    for (std::size_t r = 0; r < n_rows; ++r) {
        std::uint32_t orig_row = row_indices ? (*row_indices)[r] : static_cast<std::uint32_t>(r);
        for (std::size_t c = 0; c < n_cols; ++c) {
            out.buffer[r * n_cols + c] = X(orig_row, col_indices[c]);
        }
    }
    out.view = MatrixView(out.buffer.data(), n_rows, n_cols, /*row_major=*/true);
    return out;
}

// A bootstrap sample containing only zero-weight rows is redrawn at most
// this many times in total before RandomForestModel::fit gives up.
constexpr int kMaxBootstrapDraws = 100;

struct ForestParams {
    TreeParams tree_params;
    int n_estimators = 100;
    MaxFeaturesSpec max_features = MaxFeaturesSpec::Sqrt();
    bool bootstrap = true;
    double max_samples = 1.0; // fraction of n_samples per bootstrap draw (only used if bootstrap)
    bool oob_score = false;
    std::uint64_t random_state = 0;
    int n_jobs = 1; // sklearn convention: caller passes 1 for "None", -1 for "all cores"
};

class RandomForestModel {
public:
    explicit RandomForestModel(ForestParams params) : params_(params) {}

    RandomForestModel(const RandomForestModel&) = delete;
    RandomForestModel& operator=(const RandomForestModel&) = delete;
    RandomForestModel(RandomForestModel&&) = default;
    RandomForestModel& operator=(RandomForestModel&&) = default;

    void fit(const MatrixView& X, const VectorView& y, const VectorView* sample_weight = nullptr) {
        n_features_ = X.n_cols();
        n_samples_ = X.n_rows();
        classes_.clear();

        if (params_.tree_params.task == Task::Classification) {
            std::vector<double> labels(y.size());
            for (std::size_t i = 0; i < y.size(); ++i) labels[i] = y[i];
            std::sort(labels.begin(), labels.end());
            labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
            classes_ = std::move(labels);
        }

        const std::uint32_t k = resolve_max_features(params_.max_features, static_cast<std::uint32_t>(n_features_));
        const std::uint32_t bootstrap_size = params_.bootstrap
            ? static_cast<std::uint32_t>(std::max(1.0, params_.max_samples * static_cast<double>(n_samples_)))
            : static_cast<std::uint32_t>(n_samples_);

        const std::size_t n_est = static_cast<std::size_t>(params_.n_estimators);
        trees_.clear();
        trees_.reserve(n_est);
        for (std::size_t i = 0; i < n_est; ++i) trees_.emplace_back(params_.tree_params);
        feature_subsets_.assign(n_est, {});
        std::vector<std::vector<std::uint32_t>> oob_row_indices(n_est);
        // Trees whose bootstrap never drew a positively weighted row; reported
        // after the pool has finished (an exception thrown inside a pool task
        // would leave the other tasks running against this frame).
        std::vector<char> no_weighted_rows(n_est, 0);

        ThreadPool pool(resolve_n_jobs(params_.n_jobs));
        pool.parallel_for(n_est, [&](std::size_t t) {
            Rng rng(derive_seed(params_.random_state, t));

            std::vector<std::uint32_t> row_indices;
            if (params_.bootstrap) {
                row_indices = rng.sample_with_replacement(static_cast<std::uint32_t>(n_samples_), bootstrap_size);
                // A sample made only of zero-weight rows carries no information
                // (the tree would be a single leaf with zero class counts, i.e.
                // a predict_proba row of zeros, or value 0): redraw it from the
                // same per-tree generator, so the result stays deterministic.
                if (sample_weight) {
                    int draws = 1;
                    while (!has_positive_weight(*sample_weight, row_indices) && draws < kMaxBootstrapDraws) {
                        row_indices = rng.sample_with_replacement(static_cast<std::uint32_t>(n_samples_), bootstrap_size);
                        ++draws;
                    }
                    if (!has_positive_weight(*sample_weight, row_indices)) {
                        no_weighted_rows[t] = 1;
                        return;
                    }
                }
                if (params_.oob_score) {
                    std::vector<bool> in_bag(n_samples_, false);
                    for (auto idx : row_indices) in_bag[idx] = true;
                    for (std::uint32_t i = 0; i < n_samples_; ++i) {
                        if (!in_bag[i]) oob_row_indices[t].push_back(i);
                    }
                }
            } else {
                row_indices.resize(n_samples_);
                for (std::uint32_t i = 0; i < n_samples_; ++i) row_indices[i] = i;
            }

            std::vector<std::uint32_t> col_indices;
            if (k < n_features_) {
                col_indices = rng.sample_without_replacement(static_cast<std::uint32_t>(n_features_), k);
            } else {
                col_indices.resize(n_features_);
                for (std::uint32_t i = 0; i < n_features_; ++i) col_indices[i] = i;
            }
            feature_subsets_[t] = col_indices;

            SubMatrix sub = extract_submatrix(X, &row_indices, col_indices);
            std::vector<double> sub_y(row_indices.size());
            std::vector<double> sub_w;
            if (sample_weight) sub_w.resize(row_indices.size());
            for (std::size_t r = 0; r < row_indices.size(); ++r) {
                sub_y[r] = y[row_indices[r]];
                if (sample_weight) sub_w[r] = (*sample_weight)[row_indices[r]];
            }
            VectorView sub_Y(sub_y.data(), sub_y.size());

            if (sample_weight) {
                VectorView sub_W(sub_w.data(), sub_w.size());
                trees_[t].fit(sub.view, sub_Y, &sub_W);
            } else {
                trees_[t].fit(sub.view, sub_Y);
            }
        });

        for (std::size_t t = 0; t < n_est; ++t) {
            if (no_weighted_rows[t]) {
                throw std::invalid_argument(
                    "every bootstrap sample drawn for a tree (" + std::to_string(kMaxBootstrapDraws) +
                    " attempts) contained only rows with zero sample_weight; increase max_samples or give "
                    "more rows a positive weight");
            }
        }

        compute_feature_importances();
        if (params_.oob_score) compute_oob_score(X, y, oob_row_indices);
    }

    std::vector<double> predict(const MatrixView& X) const {
        if (params_.tree_params.task == Task::Classification) {
            auto proba = predict_proba(X);
            std::vector<double> out(proba.size());
            for (std::size_t i = 0; i < proba.size(); ++i) out[i] = classes_[argmax(proba[i])];
            return out;
        }
        std::vector<double> sums(X.n_rows(), 0.0);
        for (std::size_t t = 0; t < trees_.size(); ++t) {
            SubMatrix sub = extract_submatrix(X, nullptr, feature_subsets_[t]);
            auto preds = trees_[t].predict(sub.view);
            for (std::size_t i = 0; i < preds.size(); ++i) sums[i] += preds[i];
        }
        for (double& s : sums) s /= static_cast<double>(trees_.size());
        return sums;
    }

    std::vector<std::vector<double>> predict_proba(const MatrixView& X) const {
        std::vector<std::vector<double>> out(X.n_rows(), std::vector<double>(classes_.size(), 0.0));
        for (std::size_t t = 0; t < trees_.size(); ++t) {
            SubMatrix sub = extract_submatrix(X, nullptr, feature_subsets_[t]);
            auto proba = trees_[t].predict_proba(sub.view);
            for (std::size_t i = 0; i < proba.size(); ++i) {
                for (std::size_t c = 0; c < classes_.size(); ++c) out[i][c] += proba[i][c];
            }
        }
        double denom = static_cast<double>(trees_.size());
        for (auto& row : out) for (double& v : row) v /= denom;
        return out;
    }

    const std::vector<double>& classes() const { return classes_; }
    std::size_t n_classes() const { return classes_.size(); }
    const std::vector<RDTModel>& estimators() const { return trees_; }
    const std::vector<std::vector<std::uint32_t>>& feature_subsets() const { return feature_subsets_; }
    const std::vector<double>& feature_importances() const { return feature_importances_; }
    double oob_score() const { return oob_score_; }
    bool has_oob_score() const { return params_.oob_score; }

private:
    ForestParams params_;
    std::vector<RDTModel> trees_;
    std::vector<std::vector<std::uint32_t>> feature_subsets_;
    std::vector<double> classes_;
    std::vector<double> feature_importances_;
    double oob_score_ = 0.0;
    std::size_t n_features_ = 0;
    std::size_t n_samples_ = 0;

    static bool has_positive_weight(const VectorView& w, const std::vector<std::uint32_t>& rows) {
        for (auto r : rows) {
            if (w[r] > 0.0) return true;
        }
        return false;
    }

    static std::size_t argmax(const std::vector<double>& v) {
        std::size_t best = 0;
        for (std::size_t i = 1; i < v.size(); ++i) if (v[i] > v[best]) best = i;
        return best;
    }

    void compute_feature_importances() {
        feature_importances_.assign(n_features_, 0.0);
        for (std::size_t t = 0; t < trees_.size(); ++t) {
            const auto& depth_features = trees_[t].depth_features();
            const auto& depth_improvements = trees_[t].depth_improvements();
            double tree_total = 0.0;
            std::vector<double> tree_importance(n_features_, 0.0);
            for (const auto& kv : depth_features) {
                int depth = kv.first;
                int local_feature = kv.second;
                std::uint32_t orig_feature = feature_subsets_[t][static_cast<std::size_t>(local_feature)];
                double improvement = depth_improvements.at(depth);
                tree_importance[orig_feature] += improvement;
                tree_total += improvement;
            }
            if (tree_total > 0.0) {
                for (std::size_t f = 0; f < n_features_; ++f) feature_importances_[f] += tree_importance[f] / tree_total;
            }
        }
        double denom = static_cast<double>(trees_.size());
        double sum = 0.0;
        for (double& v : feature_importances_) { v /= denom; sum += v; }
        if (sum > 0.0) for (double& v : feature_importances_) v /= sum;
    }

    void compute_oob_score(const MatrixView& X, const VectorView& y,
                            const std::vector<std::vector<std::uint32_t>>& oob_row_indices) {
        const bool classification = params_.tree_params.task == Task::Classification;
        std::vector<std::vector<double>> vote_sums(n_samples_,
            std::vector<double>(classification ? classes_.size() : 1, 0.0));
        std::vector<int> vote_counts(n_samples_, 0);

        for (std::size_t t = 0; t < trees_.size(); ++t) {
            const auto& oob = oob_row_indices[t];
            if (oob.empty()) continue;
            SubMatrix sub = extract_submatrix(X, &oob, feature_subsets_[t]);
            if (classification) {
                auto proba = trees_[t].predict_proba(sub.view);
                for (std::size_t i = 0; i < oob.size(); ++i) {
                    for (std::size_t c = 0; c < classes_.size(); ++c) vote_sums[oob[i]][c] += proba[i][c];
                    vote_counts[oob[i]]++;
                }
            } else {
                auto preds = trees_[t].predict(sub.view);
                for (std::size_t i = 0; i < oob.size(); ++i) {
                    vote_sums[oob[i]][0] += preds[i];
                    vote_counts[oob[i]]++;
                }
            }
        }

        if (classification) {
            std::size_t correct = 0, total = 0;
            for (std::size_t i = 0; i < n_samples_; ++i) {
                if (vote_counts[i] == 0) continue;
                std::size_t predicted = argmax(vote_sums[i]);
                if (classes_[predicted] == y[i]) ++correct;
                ++total;
            }
            oob_score_ = total > 0 ? static_cast<double>(correct) / static_cast<double>(total) : 0.0;
        } else {
            double y_mean = 0.0;
            for (std::size_t i = 0; i < n_samples_; ++i) y_mean += y[i];
            y_mean /= static_cast<double>(n_samples_);

            double ss_res = 0.0, ss_tot = 0.0;
            std::size_t total = 0;
            for (std::size_t i = 0; i < n_samples_; ++i) {
                if (vote_counts[i] == 0) continue;
                double pred = vote_sums[i][0] / vote_counts[i];
                ss_res += (y[i] - pred) * (y[i] - pred);
                ss_tot += (y[i] - y_mean) * (y[i] - y_mean);
                ++total;
            }
            oob_score_ = (total > 0 && ss_tot > 0.0) ? (1.0 - ss_res / ss_tot) : 0.0;
        }
    }
};

} // namespace rdt
