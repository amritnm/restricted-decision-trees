#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

#include "rdt/forest.hpp" // MaxFeaturesSpec, resolve_max_features, extract_submatrix, SubMatrix
#include "rdt/matrix.hpp"
#include "rdt/params.hpp"
#include "rdt/rng.hpp"
#include "rdt/thread_pool.hpp"
#include "rdt/tree.hpp"

// Gradient Boosting of RDTs -- first-order (Friedman-style): each round fits
// a regression tree (MSE criterion) to the negative gradient of the chosen
// loss, then shrinks it by learning_rate. Not XGBoost's second-order
// (Newton/hessian-aware) mechanics -- see reg_alpha/reg_lambda below.
//
// Every round's tree(s) always fit MSE against the current negative
// gradient; for absolute_error/huber/quantile this means the *gradient
// formula* is loss-correct but the fitted leaf value is the tree's ordinary
// mean-of-residuals-in-leaf (not the loss-specific terminal-region-corrected
// constant sklearn's exact Friedman algorithm computes, e.g. the leaf
// median for LAD). A documented simplification, same spirit as
// reg_alpha/reg_lambda below.
//
// reg_alpha/reg_lambda are applied as a post-hoc shrink of each fitted
// tree's *predictions* (equivalent to shrinking its leaf values, since
// predict() just returns the landed-on leaf's value) -- soft-threshold for
// alpha, ridge-style denominator shrink for lambda. A real effect, but not
// XGBoost's exact Newton-derived leaf formula (which needs hessians this
// first-order engine doesn't compute).
namespace rdt {

enum class GBLoss {
    LogLoss,        // classification: binary + multiclass
    Exponential,     // classification: binary only (AdaBoost-style)
    SquaredError,    // regression
    AbsoluteError,   // regression
    Huber,           // regression
    Quantile         // regression
};

struct GBParams {
    TreeParams tree_params; // .task selects classification/regression; caller should default max_depth to 3
    int n_estimators = 100;
    double learning_rate = 0.1;
    double subsample = 1.0;              // row fraction per round, without replacement
    MaxFeaturesSpec max_features = MaxFeaturesSpec::All();
    GBLoss loss = GBLoss::LogLoss;        // caller defaults to SquaredError for regression
    double validation_fraction = 0.1;
    int n_iter_no_change = 0;             // <= 0 disables early stopping
    double tol = 1e-4;
    double reg_alpha = 0.0;
    double reg_lambda = 0.0;
    double huber_delta = 1.0;
    double quantile_alpha = 0.5;
    std::uint64_t random_state = 0;
    int n_jobs = 1;
};

class GradientBoostingModel {
public:
    explicit GradientBoostingModel(GBParams params) : params_(params) {}

    GradientBoostingModel(const GradientBoostingModel&) = delete;
    GradientBoostingModel& operator=(const GradientBoostingModel&) = delete;
    GradientBoostingModel(GradientBoostingModel&&) = default;
    GradientBoostingModel& operator=(GradientBoostingModel&&) = default;

    void fit(const MatrixView& X, const VectorView& y, const VectorView* sample_weight = nullptr) {
        n_features_ = X.n_cols();
        const std::size_t n = X.n_rows();
        classes_.clear();
        trees_.clear();
        feature_subsets_.clear();
        train_score_.clear();
        rounds_fitted_ = 0;

        bool is_classification = params_.tree_params.task == Task::Classification;
        if (is_classification) {
            std::vector<double> labels(n);
            for (std::size_t i = 0; i < n; ++i) labels[i] = y[i];
            std::sort(labels.begin(), labels.end());
            labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
            classes_ = std::move(labels);
        }
        const std::size_t n_outputs = (is_classification && classes_.size() > 2) ? classes_.size() : 1;
        n_outputs_ = n_outputs;

        // y encoded per-output as what each output's gradient formula needs:
        // multiclass -> one-hot; binary log_loss/exponential -> {0,1}/{-1,+1}; regression -> y itself.
        std::vector<std::vector<double>> y_enc = encode_targets(y, n, is_classification, n_outputs);

        Rng split_rng(derive_seed(params_.random_state, 0xF0));
        std::vector<std::uint32_t> train_rows, val_rows;
        bool early_stopping = params_.n_iter_no_change > 0 && params_.validation_fraction > 0.0;
        if (early_stopping) {
            split_train_val(static_cast<std::uint32_t>(n), params_.validation_fraction, split_rng, train_rows, val_rows);
        } else {
            train_rows.resize(n);
            for (std::uint32_t i = 0; i < n; ++i) train_rows[i] = i;
        }

        // F[k][i]: current raw score for output k, sample i (over ALL rows).
        std::vector<std::vector<double>> F(n_outputs, std::vector<double>(n, 0.0));
        init_scores_.assign(n_outputs, 0.0);
        for (std::size_t k = 0; k < n_outputs; ++k) {
            double init = compute_init_score(y_enc[k], train_rows);
            init_scores_[k] = init;
            for (std::size_t i = 0; i < n; ++i) F[k][i] = init;
        }

        double best_val_score = std::numeric_limits<double>::infinity();
        int rounds_without_improve = 0;
        const std::size_t n_est = static_cast<std::size_t>(params_.n_estimators);
        ThreadPool pool(resolve_n_jobs(params_.n_jobs));

        for (std::size_t round = 0; round < n_est; ++round) {
            std::vector<RDTModel> round_trees;
            round_trees.reserve(n_outputs);
            for (std::size_t k = 0; k < n_outputs; ++k) {
                TreeParams tp = params_.tree_params;
                tp.task = Task::Regression;
                tp.criterion = Criterion::MSE;
                round_trees.emplace_back(tp);
            }
            std::vector<std::vector<std::uint32_t>> round_col_subsets(n_outputs);
            std::vector<std::vector<double>> round_raw_preds(n_outputs);

            pool.parallel_for(n_outputs, [&](std::size_t k) {
                Rng rng(derive_seed(params_.random_state, round * 1000 + k + 1));

                std::vector<double> residual_train(train_rows.size());
                for (std::size_t r = 0; r < train_rows.size(); ++r) {
                    std::uint32_t i = train_rows[r];
                    residual_train[r] = negative_gradient(i, k, y_enc, F, n_outputs);
                }

                std::vector<std::uint32_t> fit_rows;
                if (params_.subsample < 1.0) {
                    std::uint32_t m = static_cast<std::uint32_t>(
                        std::max(1.0, params_.subsample * static_cast<double>(train_rows.size())));
                    auto local_idx = rng.sample_without_replacement(static_cast<std::uint32_t>(train_rows.size()), m);
                    fit_rows.reserve(local_idx.size());
                    for (auto li : local_idx) fit_rows.push_back(train_rows[li]);
                } else {
                    fit_rows = train_rows;
                }

                std::uint32_t k_features = resolve_max_features(params_.max_features, static_cast<std::uint32_t>(n_features_));
                std::vector<std::uint32_t> col_subset;
                if (k_features < n_features_) {
                    col_subset = rng.sample_without_replacement(static_cast<std::uint32_t>(n_features_), k_features);
                } else {
                    col_subset.resize(n_features_);
                    for (std::uint32_t f = 0; f < n_features_; ++f) col_subset[f] = f;
                }
                round_col_subsets[k] = col_subset;

                SubMatrix fit_sub = extract_submatrix(X, &fit_rows, col_subset);
                std::vector<double> fit_residual(fit_rows.size());
                std::vector<double> fit_weight;
                if (sample_weight) fit_weight.resize(fit_rows.size());
                // Build a row_indices->residual map restricted to fit_rows (a subset of train_rows).
                std::unordered_map<std::uint32_t, double> residual_by_row;
                residual_by_row.reserve(train_rows.size());
                for (std::size_t r = 0; r < train_rows.size(); ++r) residual_by_row[train_rows[r]] = residual_train[r];
                for (std::size_t r = 0; r < fit_rows.size(); ++r) {
                    fit_residual[r] = residual_by_row[fit_rows[r]];
                    if (sample_weight) fit_weight[r] = (*sample_weight)[fit_rows[r]];
                }
                VectorView residual_view(fit_residual.data(), fit_residual.size());

                if (sample_weight) {
                    VectorView w_view(fit_weight.data(), fit_weight.size());
                    round_trees[k].fit(fit_sub.view, residual_view, &w_view);
                } else {
                    round_trees[k].fit(fit_sub.view, residual_view);
                }

                SubMatrix all_sub = extract_submatrix(X, nullptr, col_subset);
                std::vector<double> raw_pred = round_trees[k].predict(all_sub.view);
                shrink_leaf_values(raw_pred);
                round_raw_preds[k] = std::move(raw_pred);
            });

            for (std::size_t k = 0; k < n_outputs; ++k) {
                for (std::size_t i = 0; i < n; ++i) F[k][i] += params_.learning_rate * round_raw_preds[k][i];
                trees_.push_back(std::move(round_trees[k]));
                feature_subsets_.push_back(std::move(round_col_subsets[k]));
            }
            rounds_fitted_ = round + 1;

            double train_loss = compute_loss(y_enc, F, train_rows, n_outputs);
            train_score_.push_back(train_loss);

            if (early_stopping) {
                double val_loss = compute_loss(y_enc, F, val_rows, n_outputs);
                if (val_loss < best_val_score - params_.tol) {
                    best_val_score = val_loss;
                    rounds_without_improve = 0;
                } else {
                    ++rounds_without_improve;
                }
                if (rounds_without_improve >= params_.n_iter_no_change) break;
            }
        }

        compute_feature_importances();
    }

    std::vector<double> predict(const MatrixView& X) const {
        auto F = accumulate_scores(X);
        if (params_.tree_params.task == Task::Regression) return F[0];

        std::vector<double> out(X.n_rows());
        if (n_outputs_ == 1) { // binary
            for (std::size_t i = 0; i < X.n_rows(); ++i) {
                double p = sigmoid_like(F[0][i]);
                out[i] = p >= 0.5 ? classes_[1] : classes_[0];
            }
        } else {
            for (std::size_t i = 0; i < X.n_rows(); ++i) {
                std::size_t best = 0;
                double best_val = F[0][i];
                for (std::size_t k = 1; k < n_outputs_; ++k) {
                    if (F[k][i] > best_val) { best_val = F[k][i]; best = k; }
                }
                out[i] = classes_[best];
            }
        }
        return out;
    }

    std::vector<std::vector<double>> predict_proba(const MatrixView& X) const {
        auto F = accumulate_scores(X);
        std::vector<std::vector<double>> out(X.n_rows(), std::vector<double>(classes_.size(), 0.0));
        if (n_outputs_ == 1) {
            for (std::size_t i = 0; i < X.n_rows(); ++i) {
                double p1 = sigmoid_like(F[0][i]);
                out[i][0] = 1.0 - p1;
                out[i][1] = p1;
            }
        } else {
            for (std::size_t i = 0; i < X.n_rows(); ++i) {
                double max_f = F[0][i];
                for (std::size_t k = 1; k < n_outputs_; ++k) max_f = std::max(max_f, F[k][i]);
                double sum = 0.0;
                std::vector<double> exp_vals(n_outputs_);
                for (std::size_t k = 0; k < n_outputs_; ++k) {
                    exp_vals[k] = std::exp(F[k][i] - max_f);
                    sum += exp_vals[k];
                }
                for (std::size_t k = 0; k < n_outputs_; ++k) out[i][k] = exp_vals[k] / sum;
            }
        }
        return out;
    }

    const std::vector<double>& classes() const { return classes_; }
    std::size_t n_classes() const { return classes_.size(); }
    const std::vector<double>& feature_importances() const { return feature_importances_; }
    const std::vector<double>& train_score() const { return train_score_; }
    std::size_t n_estimators_fitted() const { return rounds_fitted_; }

private:
    GBParams params_;
    std::vector<double> classes_;
    std::size_t n_features_ = 0;
    std::size_t n_outputs_ = 1;
    std::vector<RDTModel> trees_;                          // round-major: trees_[round*n_outputs_+k]
    std::vector<std::vector<std::uint32_t>> feature_subsets_;
    std::vector<double> init_scores_;
    std::vector<double> feature_importances_;
    std::vector<double> train_score_;
    std::size_t rounds_fitted_ = 0;

    std::vector<std::vector<double>> encode_targets(const VectorView& y, std::size_t n,
                                                      bool is_classification, std::size_t n_outputs) const {
        std::vector<std::vector<double>> enc(n_outputs, std::vector<double>(n, 0.0));
        if (!is_classification) {
            for (std::size_t i = 0; i < n; ++i) enc[0][i] = y[i];
            return enc;
        }
        if (n_outputs == 1) { // binary
            for (std::size_t i = 0; i < n; ++i) {
                bool positive = y[i] == classes_[1];
                enc[0][i] = (params_.loss == GBLoss::Exponential) ? (positive ? 1.0 : -1.0) : (positive ? 1.0 : 0.0);
            }
        } else { // multiclass one-hot
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t k = 0; k < n_outputs; ++k) enc[k][i] = (y[i] == classes_[k]) ? 1.0 : 0.0;
            }
        }
        return enc;
    }

    static void split_train_val(std::uint32_t n, double val_fraction, Rng& rng,
                                 std::vector<std::uint32_t>& train_rows, std::vector<std::uint32_t>& val_rows) {
        std::uint32_t n_val = static_cast<std::uint32_t>(std::max(1.0, val_fraction * n));
        n_val = std::min(n_val, n - 1);
        val_rows = rng.sample_without_replacement(n, n_val);
        std::vector<bool> is_val(n, false);
        for (auto v : val_rows) is_val[v] = true;
        train_rows.clear();
        for (std::uint32_t i = 0; i < n; ++i) if (!is_val[i]) train_rows.push_back(i);
    }

    double compute_init_score(const std::vector<double>& enc, const std::vector<std::uint32_t>& train_rows) const {
        double mean = 0.0;
        for (auto i : train_rows) mean += enc[i];
        mean /= static_cast<double>(train_rows.size());

        switch (params_.loss) {
            case GBLoss::LogLoss: {
                if (n_outputs_ > 1) return 0.0; // multiclass: start every class's raw score at 0
                double p = std::min(std::max(mean, 1e-7), 1.0 - 1e-7);
                return std::log(p / (1.0 - p));
            }
            case GBLoss::Exponential: {
                double p = std::min(std::max((mean + 1.0) / 2.0, 1e-7), 1.0 - 1e-7); // enc in {-1,+1}
                return 0.5 * std::log(p / (1.0 - p));
            }
            case GBLoss::AbsoluteError: {
                std::vector<double> vals;
                vals.reserve(train_rows.size());
                for (auto i : train_rows) vals.push_back(enc[i]);
                std::sort(vals.begin(), vals.end());
                return vals[vals.size() / 2];
            }
            case GBLoss::Quantile: {
                std::vector<double> vals;
                vals.reserve(train_rows.size());
                for (auto i : train_rows) vals.push_back(enc[i]);
                std::sort(vals.begin(), vals.end());
                std::size_t idx = static_cast<std::size_t>(params_.quantile_alpha * (vals.size() - 1));
                return vals[idx];
            }
            default: // SquaredError, Huber
                return mean;
        }
    }

    // Negative gradient of the loss w.r.t. output k's raw score, at sample i.
    // Multiclass log_loss needs every output's current score at i (softmax),
    // hence the full y_enc/F rather than just this output's target/score.
    double negative_gradient(std::size_t i, std::size_t k, const std::vector<std::vector<double>>& y_enc,
                              const std::vector<std::vector<double>>& F, std::size_t n_outputs) const {
        double target = y_enc[k][i];
        double f = F[k][i];
        switch (params_.loss) {
            case GBLoss::LogLoss:
                if (n_outputs == 1) return target - sigmoid_like(f);
                {
                    double max_f = F[0][i];
                    for (std::size_t j = 1; j < n_outputs; ++j) max_f = std::max(max_f, F[j][i]);
                    double sum = 0.0;
                    for (std::size_t j = 0; j < n_outputs; ++j) sum += std::exp(F[j][i] - max_f);
                    double softmax_k = std::exp(f - max_f) / sum;
                    return target - softmax_k;
                }
            case GBLoss::Exponential:
                return target * std::exp(-target * f);
            case GBLoss::SquaredError:
                return target - f;
            case GBLoss::AbsoluteError:
                return (target - f) > 0 ? 1.0 : ((target - f) < 0 ? -1.0 : 0.0);
            case GBLoss::Huber: {
                double diff = target - f;
                return std::abs(diff) <= params_.huber_delta ? diff : params_.huber_delta * (diff > 0 ? 1.0 : -1.0);
            }
            case GBLoss::Quantile:
                return (target - f) > 0 ? params_.quantile_alpha : (params_.quantile_alpha - 1.0);
        }
        return 0.0;
    }

    void shrink_leaf_values(std::vector<double>& values) const {
        if (params_.reg_lambda == 0.0 && params_.reg_alpha == 0.0) return;
        for (double& v : values) {
            double shrunk = v / (1.0 + params_.reg_lambda);
            double sign = shrunk > 0 ? 1.0 : (shrunk < 0 ? -1.0 : 0.0);
            v = sign * std::max(std::abs(shrunk) - params_.reg_alpha, 0.0);
        }
    }

    static double sigmoid_like(double f) { return 1.0 / (1.0 + std::exp(-f)); }

    double compute_loss(const std::vector<std::vector<double>>& y_enc, const std::vector<std::vector<double>>& F,
                         const std::vector<std::uint32_t>& rows, std::size_t n_outputs) const {
        if (rows.empty()) return 0.0;
        double total = 0.0;
        for (auto i : rows) {
            switch (params_.loss) {
                case GBLoss::LogLoss:
                    if (n_outputs == 1) {
                        double p = std::min(std::max(sigmoid_like(F[0][i]), 1e-12), 1.0 - 1e-12);
                        total += -(y_enc[0][i] * std::log(p) + (1 - y_enc[0][i]) * std::log(1 - p));
                    } else {
                        double max_f = F[0][i];
                        for (std::size_t k = 1; k < n_outputs; ++k) max_f = std::max(max_f, F[k][i]);
                        double sum = 0.0;
                        for (std::size_t k = 0; k < n_outputs; ++k) sum += std::exp(F[k][i] - max_f);
                        for (std::size_t k = 0; k < n_outputs; ++k) {
                            if (y_enc[k][i] > 0.5) total += -(F[k][i] - max_f - std::log(sum));
                        }
                    }
                    break;
                case GBLoss::Exponential:
                    total += std::exp(-y_enc[0][i] * F[0][i]);
                    break;
                case GBLoss::SquaredError:
                    total += (y_enc[0][i] - F[0][i]) * (y_enc[0][i] - F[0][i]);
                    break;
                case GBLoss::AbsoluteError:
                    total += std::abs(y_enc[0][i] - F[0][i]);
                    break;
                case GBLoss::Huber: {
                    double diff = y_enc[0][i] - F[0][i];
                    total += std::abs(diff) <= params_.huber_delta
                        ? 0.5 * diff * diff
                        : params_.huber_delta * (std::abs(diff) - 0.5 * params_.huber_delta);
                    break;
                }
                case GBLoss::Quantile: {
                    double diff = y_enc[0][i] - F[0][i];
                    total += diff >= 0 ? params_.quantile_alpha * diff : (params_.quantile_alpha - 1.0) * diff;
                    break;
                }
            }
        }
        return total / static_cast<double>(rows.size());
    }

    std::vector<std::vector<double>> accumulate_scores(const MatrixView& X) const {
        std::vector<std::vector<double>> F(n_outputs_, std::vector<double>(X.n_rows()));
        for (std::size_t k = 0; k < n_outputs_; ++k) {
            std::fill(F[k].begin(), F[k].end(), init_scores_[k]);
        }
        for (std::size_t t = 0; t < trees_.size(); ++t) {
            std::size_t k = t % n_outputs_;
            SubMatrix sub = extract_submatrix(X, nullptr, feature_subsets_[t]);
            auto raw = trees_[t].predict(sub.view);
            shrink_leaf_values(raw);
            for (std::size_t i = 0; i < raw.size(); ++i) F[k][i] += params_.learning_rate * raw[i];
        }
        return F;
    }

    void compute_feature_importances() {
        feature_importances_.assign(n_features_, 0.0);
        double grand_total = 0.0;
        for (std::size_t t = 0; t < trees_.size(); ++t) {
            const auto& depth_features = trees_[t].depth_features();
            const auto& depth_improvements = trees_[t].depth_improvements();
            for (const auto& kv : depth_features) {
                int local_feature = kv.second;
                std::uint32_t orig_feature = feature_subsets_[t][static_cast<std::size_t>(local_feature)];
                double improvement = depth_improvements.at(kv.first);
                feature_importances_[orig_feature] += improvement;
                grand_total += improvement;
            }
        }
        if (grand_total > 0.0) for (double& v : feature_importances_) v /= grand_total;
    }
};

} // namespace rdt
