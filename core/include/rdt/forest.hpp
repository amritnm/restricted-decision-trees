#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
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

// Rows per block when an ensemble predicts: every tree is applied to a block
// of rows before moving to the next block, so a block's rows of X stay in
// cache across trees. Per-row accumulation order over trees is unchanged.
constexpr std::size_t kPredictRowBlock = 256;

// A bootstrap sample containing only zero-weight rows is redrawn at most
// this many times in total before RandomForestModel::fit gives up.
constexpr int kMaxBootstrapDraws = 100;

struct ForestParams {
    TreeParams tree_params;
    int n_estimators = 100;
    MaxFeaturesSpec max_features = MaxFeaturesSpec::Sqrt();
    bool bootstrap = true;
    double max_samples = 1.0; // fraction of n_samples per bootstrap draw (only used if bootstrap)
    // Absolute bootstrap draw size; when > 0 it overrides max_samples
    // (scikit-learn's integer max_samples).
    std::uint32_t max_samples_count = 0;
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
        const std::uint32_t bootstrap_size = !params_.bootstrap
            ? static_cast<std::uint32_t>(n_samples_)
            : params_.max_samples_count > 0
                ? params_.max_samples_count
                : static_cast<std::uint32_t>(std::max(1.0, params_.max_samples * static_cast<double>(n_samples_)));

        const std::size_t n_est = static_cast<std::size_t>(params_.n_estimators);
        TreeParams tree_params = params_.tree_params;
        // Without bootstrap a tree's rows are the training rows themselves, so
        // its node indices stay meaningful (rdt.visualize); with bootstrap they
        // would be positions in the sample. See TreeParams::retain_indices.
        tree_params.retain_indices = !params_.bootstrap;
        trees_.clear();
        trees_.reserve(n_est);
        for (std::size_t i = 0; i < n_est; ++i) trees_.emplace_back(tree_params);
        feature_subsets_.assign(n_est, {});
        std::vector<std::vector<std::uint32_t>> oob_row_indices(n_est);
        // Trees whose bootstrap never drew a positively weighted row; reported
        // after the pool has finished, for the lowest such tree, so the error
        // does not depend on thread scheduling.
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

            // The tree's rows/columns as a view of X: nothing is copied.
            IndexedMatrixView sub(X, row_indices.data(), row_indices.size(), col_indices.data(), col_indices.size());
            std::vector<double> sub_y(row_indices.size());
            std::vector<double> sub_w;
            if (sample_weight) sub_w.resize(row_indices.size());
            for (std::size_t r = 0; r < row_indices.size(); ++r) {
                sub_y[r] = y[row_indices[r]];
                if (sample_weight) sub_w[r] = (*sample_weight)[row_indices[r]];
            }
            VectorView sub_Y(sub_y.data(), sub_y.size());

            // Every tree is aligned to the forest's classes_, not just the
            // labels its bootstrap sample happens to contain, so predict_proba
            // and the OOB vote can add tree rows column-by-column.
            const std::vector<double>* tree_classes =
                params_.tree_params.task == Task::Classification ? &classes_ : nullptr;
            if (sample_weight) {
                VectorView sub_W(sub_w.data(), sub_w.size());
                trees_[t].fit(sub, sub_Y, &sub_W, nullptr, tree_classes);
            } else {
                trees_[t].fit(sub, sub_Y, nullptr, nullptr, tree_classes);
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
        const std::size_t n = X.n_rows();
        if (params_.tree_params.task == Task::Classification) {
            const std::size_t k = classes_.size();
            std::vector<double> proba(n * k);
            predict_proba_into(X, proba.data());
            std::vector<double> out(n);
            for (std::size_t i = 0; i < n; ++i) out[i] = classes_[argmax(proba.data() + i * k, k)];
            return out;
        }
        std::vector<double> sums(n, 0.0);
        for (std::size_t b = 0; b < n; b += kPredictRowBlock) {
            const std::size_t e = std::min(n, b + kPredictRowBlock);
            for (std::size_t t = 0; t < trees_.size(); ++t) {
                const std::uint32_t* cols = feature_subsets_[t].data();
                for (std::size_t i = b; i < e; ++i) sums[i] += trees_[t].find_leaf(X, i, cols)->value;
            }
        }
        for (double& s : sums) s /= static_cast<double>(trees_.size());
        return sums;
    }

    std::vector<std::vector<double>> predict_proba(const MatrixView& X) const {
        const std::size_t k = classes_.size();
        std::vector<double> flat(X.n_rows() * k);
        predict_proba_into(X, flat.data());
        std::vector<std::vector<double>> out(X.n_rows());
        for (std::size_t i = 0; i < out.size(); ++i) out[i].assign(flat.begin() + i * k, flat.begin() + (i + 1) * k);
        return out;
    }

    // predict_proba into a row-major n_samples x n_classes buffer.
    void predict_proba_into(const MatrixView& X, double* out) const {
        const std::size_t n = X.n_rows();
        const std::size_t k = classes_.size();
        std::fill(out, out + n * k, 0.0);
        std::vector<double> tree_proba(k);
        for (std::size_t b = 0; b < n; b += kPredictRowBlock) {
            const std::size_t e = std::min(n, b + kPredictRowBlock);
            for (std::size_t t = 0; t < trees_.size(); ++t) {
                const std::uint32_t* cols = feature_subsets_[t].data();
                for (std::size_t i = b; i < e; ++i) {
                    add_tree_proba(trees_[t], trees_[t].find_leaf(X, i, cols), tree_proba, out + i * k);
                }
            }
        }
        double denom = static_cast<double>(trees_.size());
        for (std::size_t j = 0; j < n * k; ++j) out[j] /= denom;
    }

    const std::vector<double>& classes() const { return classes_; }
    std::size_t n_classes() const { return classes_.size(); }
    const std::vector<RDTModel>& estimators() const { return trees_; }
    const std::vector<std::vector<std::uint32_t>>& feature_subsets() const { return feature_subsets_; }
    const std::vector<double>& feature_importances() const { return feature_importances_; }
    // Per tree, the feature chosen at each depth as a column of the X passed
    // to fit (each tree's own column indices mapped through its column subset).
    std::vector<std::map<int, int>> depth_features() const {
        std::vector<std::map<int, int>> out(trees_.size());
        for (std::size_t t = 0; t < trees_.size(); ++t) {
            for (const auto& kv : trees_[t].depth_features()) {
                out[t][kv.first] = static_cast<int>(feature_subsets_[t][static_cast<std::size_t>(kv.second)]);
            }
        }
        return out;
    }
    double oob_score() const { return oob_score_; }
    bool has_oob_score() const { return params_.oob_score; }
    std::size_t n_features() const { return n_features_; }

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

    static std::size_t argmax(const double* v, std::size_t n) {
        std::size_t best = 0;
        for (std::size_t i = 1; i < n; ++i) if (v[i] > v[best]) best = i;
        return best;
    }

    static std::size_t argmax(const std::vector<double>& v) { return argmax(v.data(), v.size()); }

    // Adds one tree's class probabilities for a row into `acc` (forest
    // classes() order). Every classification tree is fitted with the forest's
    // classes_ (see fit), so its columns are the forest's columns; the min()
    // only guards against a tree with fewer columns, which v0.2.0 produced
    // when a bootstrap sample missed a class (and then read out of bounds).
    void add_tree_proba(const RDTModel& tree, const Node* leaf, std::vector<double>& buf, double* acc) const {
        const std::size_t tree_k = tree.n_classes();
        tree.leaf_proba(leaf, buf.data());
        const std::size_t k = std::min(classes_.size(), tree_k);
        for (std::size_t c = 0; c < k; ++c) acc[c] += buf[c];
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

        std::vector<double> tree_proba(classes_.size());
        for (std::size_t t = 0; t < trees_.size(); ++t) {
            const auto& oob = oob_row_indices[t];
            if (oob.empty()) continue;
            const std::uint32_t* cols = feature_subsets_[t].data();
            if (classification) {
                for (std::size_t i = 0; i < oob.size(); ++i) {
                    const Node* leaf = trees_[t].find_leaf(X, oob[i], cols);
                    add_tree_proba(trees_[t], leaf, tree_proba, vote_sums[oob[i]].data());
                    vote_counts[oob[i]]++;
                }
            } else {
                for (std::size_t i = 0; i < oob.size(); ++i) {
                    vote_sums[oob[i]][0] += trees_[t].find_leaf(X, oob[i], cols)->value;
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
