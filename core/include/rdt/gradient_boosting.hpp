#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "rdt/forest.hpp" // MaxFeaturesSpec, resolve_max_features, kPredictRowBlock
#include "rdt/hist.hpp"
#include "rdt/matrix.hpp"
#include "rdt/params.hpp"
#include "rdt/presort.hpp"
#include "rdt/rng.hpp"
#include "rdt/thread_pool.hpp"
#include "rdt/tree.hpp"

// Gradient Boosting of RDTs -- Friedman's gradient boosting, as implemented
// by scikit-learn's GradientBoostingClassifier/Regressor: each round fits a
// regression tree (MSE criterion, sample-weighted) to the negative gradient
// of the chosen loss, then (step 5 of Friedman's Algorithm 1, sklearn's
// _update_terminal_regions) re-estimates every leaf's value for the loss
// itself over the in-bag rows reaching it, then shrinks by learning_rate:
//
// - squared_error: the leaf mean of the residuals is already the exact
//   minimiser, kept as is;
// - log_loss: one Newton step, sum(w*(y-p)) / sum(w*p*(1-p)); multiclass
//   (softmax, one tree per class per round) additionally scales the
//   numerator by (K-1)/K, as in Friedman's paper and sklearn;
// - exponential: one Newton step, sum(w*g) / sum(w*|g|) with
//   g = y~*exp(-y~*F), y~ in {-1,+1}; F estimates half the log-odds, so
//   probabilities are sigmoid(2F);
// - absolute_error/huber/quantile: the leaf mean of the (sign/clipped)
//   gradient -- a documented simplification of sklearn's leaf median /
//   quantile line search, same spirit as reg_alpha/reg_lambda below.
//
// The initial score is the loss's (weighted) constant minimiser: weighted
// log-odds of the positive class (half of it for exponential), weighted
// log class priors centred to sum to zero for multiclass, weighted
// mean/median/quantile for regression. sample_weight enters the tree fits,
// the init, the leaf values and the loss reported in train_score_ / used for
// early stopping.
//
// With the default split criterion (friedman_mse) splits are chosen
// first-order (MSE on the gradient), not with XGBoost's hessian-weighted gain,
// and reg_alpha/reg_lambda are applied as a post-hoc shrink of each fitted
// tree's *predictions* (equivalent to shrinking its leaf values, since
// predict() just returns the landed-on leaf's value) -- soft-threshold for
// alpha, ridge-style denominator shrink for lambda. A real effect, but not
// XGBoost's exact regularised leaf formula.
//
// split_criterion = newton grows each tree XGBoost's way instead (see
// Criterion::Newton in tree.hpp): per-row gradients g = dL/dF and hessians h
// (times the sample weight) -- log_loss: g = p - y, h = max(p(1-p), 1e-16)
// (binary), h = max(2p(1-p) w, 1e-16) (multiclass, per class, as XGBoost's
// softmax); exponential: g = -y~ e, h = e with e = exp(-y~ F); squared_error:
// g = F - y, h = 1 -- splits by the gain T(G)^2/(H+lambda) with gamma,
// min_child_weight (a hessian sum) and max_delta_step, and leaves
// -T(G)/(H+lambda) (clipped), shrunk by learning_rate. reg_alpha/reg_lambda
// enter only these formulas (no post-hoc shrink, no Friedman leaf step).
// Losses without a hessian (absolute_error, huber, quantile) are rejected.
namespace rdt {

enum class SplitCriterion { FriedmanMSE, Newton };

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
    SplitCriterion split_criterion = SplitCriterion::FriedmanMSE;
    // Newton only (XGBoost's meaning; gamma is tree_params.min_impurity_decrease):
    double min_child_weight = 1.0; // hessian sum per child
    double max_delta_step = 0.0;
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
        if (params_.tree_params.tree_method == TreeMethod::Hist) {
            fit_hist(X, y, sample_weight);
            return;
        }
        const bool presort = params_.tree_params.presort && X.n_rows() <= kPresortMaxRows;
        boost(X.n_rows(), X.n_cols(), y, sample_weight,
              [&](const std::vector<std::uint32_t>& train_rows, ThreadPool&) {
                  return ExactSource(X, presort, train_rows);
              });
    }

    // tree_method='hist': X (MatrixView or StridedMatrix<float/double>, read
    // without copying) is binned once -- cuts from the training rows (after the
    // early-stopping split) with positive weight, every row binned -- and is
    // not read again: the trees are fitted on the bins and the training scores
    // (validation rows included) are updated through them.
    template <class Matrix>
    void fit_hist(const Matrix& X, const VectorView& y, const VectorView* sample_weight = nullptr) {
        if (params_.tree_params.tree_method != TreeMethod::Hist) {
            throw std::invalid_argument("fit_hist needs tree_method='hist'");
        }
        boost(X.n_rows(), X.n_cols(), y, sample_weight,
              [&](const std::vector<std::uint32_t>& train_rows, ThreadPool& pool) {
                  std::vector<std::uint32_t> cut_rows;
                  cut_rows.reserve(train_rows.size());
                  for (auto r : train_rows) {
                      if (!sample_weight || (*sample_weight)[r] > 0.0) cut_rows.push_back(r);
                  }
                  return HistSource(X, cut_rows, params_.tree_params.max_bin, pool.size() > 1 ? &pool : nullptr);
              });
    }

private:
    // Where a round's trees get their features: the exact search on X (with
    // the per-fit presorted columns) or the histogram search on the bins.
    class ExactSource {
    public:
        static constexpr bool kFeatureParallel = false;

        ExactSource(const MatrixView& X, bool presort, const std::vector<std::uint32_t>& train_rows)
            : X_(X), presort_(presort), sorted_columns_(X, presort ? train_rows : std::vector<std::uint32_t>()) {}

        // Every round's trees are fitted on a subset of train_rows, so each
        // column's training rows are sorted once per fit -- lazily, the first
        // time a round draws the column -- and kept (4 bytes per (training
        // row, sorted column) for the duration of fit()); each round derives
        // its trees' presorted columns from these in O(n) per column
        // (presort.hpp).
        void prepare(const std::vector<std::uint32_t>& col_subset) {
            if (presort_) sorted_columns_.ensure(col_subset);
        }

        void fit_tree(RDTModel& tree, const std::vector<std::uint32_t>& fit_rows,
                      const std::vector<std::uint32_t>& col_subset, const VectorView& residual,
                      const VectorView* weight, const VectorView* hessian, ThreadPool*, HistArena*) const {
            // The tree sees X[fit_rows, col_subset] as a view (no copy of X).
            const std::size_t n = X_.n_rows();
            const std::size_t m = fit_rows.size();
            IndexedMatrixView X_sub(X_, fit_rows.data(), m, col_subset.data(), col_subset.size());
            // This round's presorted columns: the per-fit ones for
            // col_subset, restricted to fit_rows and renumbered to the
            // tree's row ids 0..m-1 (fit_rows is ascending, so tied
            // values stay in row order).
            SortedColumns round_sorted;
            PresortedView presorted;
            if (presort_) {
                presorted = sorted_columns_.view(col_subset);
                const bool all_rows = m == n; // fit_rows == 0..n-1: ids already match
                if (!all_rows) {
                    std::vector<std::int32_t> new_id(n, -1);
                    for (std::size_t r = 0; r < m; ++r) new_id[fit_rows[r]] = static_cast<std::int32_t>(r);
                    round_sorted = SortedColumns(presorted, new_id.data(), m);
                    presorted = round_sorted.view();
                }
            }
            tree.fit(X_sub, residual, weight, presort_ ? &presorted : nullptr, nullptr, hessian);
        }

        Node* leaf(RDTModel& tree, std::size_t row, const std::uint32_t* cols) const {
            return tree.leaf_for(X_, row, cols);
        }

    private:
        MatrixView X_;
        bool presort_;
        LazySortedColumns sorted_columns_;
    };

    class HistSource {
    public:
        static constexpr bool kFeatureParallel = true;

        template <class Matrix>
        HistSource(const Matrix& X, const std::vector<std::uint32_t>& cut_rows, int max_bin, ThreadPool* pool)
            : bins_(X, cut_rows, max_bin, pool) {}

        void prepare(const std::vector<std::uint32_t>&) {}

        void fit_tree(RDTModel& tree, const std::vector<std::uint32_t>& fit_rows,
                      const std::vector<std::uint32_t>& col_subset, const VectorView& residual,
                      const VectorView* weight, const VectorView* hessian, ThreadPool* pool,
                      HistArena* arena) const {
            BinnedView view(bins_, fit_rows.data(), fit_rows.size(), col_subset.data(), col_subset.size());
            tree.fit_binned(view, residual, weight, nullptr, hessian, pool, arena);
        }

        Node* leaf(RDTModel& tree, std::size_t row, const std::uint32_t* cols) const {
            return tree.leaf_for_binned(bins_, row, cols);
        }

    private:
        BinnedMatrix bins_;
    };

    // The boosting loop. make_source(train_rows, pool) builds the Source
    // (ExactSource / HistSource) once the training rows are known.
    template <class MakeSource>
    void boost(std::size_t n, std::size_t n_cols, const VectorView& y, const VectorView* sample_weight,
               MakeSource&& make_source) {
        ++fit_generation_;
        n_features_ = n_cols;
        classes_.clear();
        trees_.clear();
        feature_subsets_.clear();
        train_score_.clear();
        rounds_fitted_ = 0;

        if (newton() && !(params_.loss == GBLoss::LogLoss || params_.loss == GBLoss::Exponential ||
                          params_.loss == GBLoss::SquaredError)) {
            throw std::invalid_argument(
                "split_criterion='newton' needs a loss with a hessian (log_loss, exponential, squared_error)");
        }
        bool is_classification = params_.tree_params.task == Task::Classification;
        if (is_classification) {
            std::vector<double> labels(n);
            for (std::size_t i = 0; i < n; ++i) labels[i] = y[i];
            std::sort(labels.begin(), labels.end());
            labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
            classes_ = std::move(labels);
            if (classes_.size() < 2) {
                throw std::invalid_argument("gradient boosting classification needs at least 2 classes in y");
            }
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

        // Both sides of the early-stopping split must carry weight: with no
        // weight on the training rows there is nothing to fit (the init would
        // be a clipped prior of no data), and with none on the holdout the
        // validation loss is 0/0 and early stopping fires after
        // n_iter_no_change + 1 rounds regardless of the fit. scikit-learn
        // raises in both cases too.
        if (sample_weight && total_weight(*sample_weight, train_rows) <= 0.0) {
            throw std::invalid_argument(
                "the training rows (after the validation_fraction split) have zero total sample_weight");
        }
        if (sample_weight && early_stopping && total_weight(*sample_weight, val_rows) <= 0.0) {
            throw std::invalid_argument(
                "the early-stopping validation rows (validation_fraction) have zero total sample_weight; "
                "increase validation_fraction or give more rows a positive weight");
        }

        // F[k][i]: current raw score for output k, sample i (over ALL rows).
        std::vector<std::vector<double>> F(n_outputs, std::vector<double>(n, 0.0));
        init_scores_ = compute_init_scores(y_enc, train_rows, sample_weight);
        for (std::size_t k = 0; k < n_outputs; ++k) {
            for (std::size_t i = 0; i < n; ++i) F[k][i] = init_scores_[k];
        }

        double best_val_score = std::numeric_limits<double>::infinity();
        int rounds_without_improve = 0;
        const std::size_t n_est = static_cast<std::size_t>(params_.n_estimators);
        ThreadPool pool(resolve_n_jobs(params_.n_jobs));
        auto source = make_source(train_rows, pool);
        using Source = decltype(source);

        // Negative gradient per output, indexed by sample (only train_rows
        // are written/read); reused across rounds. Replaces v0.2.0's per-round
        // unordered_map from row to residual.
        std::vector<std::vector<double>> residual(n_outputs, std::vector<double>(n, 0.0));
        // Newton: hessian per output, indexed like residual (which then holds
        // the gradient dL/dF rather than its negative).
        std::vector<std::vector<double>> hessian(newton() ? n_outputs : 0, std::vector<double>(n, 0.0));
        std::vector<double> loss_terms; // compute_loss's per-row scratch
        // The subtraction caches (hist), reused round after round: one arena
        // per tree being fitted at the same time -- a multiclass round's
        // trees take one from this free list and give it back, so its kept
        // histograms take at most max_hist_cache_bytes per tree in flight
        // (min(n_jobs, n_outputs)), not per output. Which arena a tree gets
        // does not change the tree.
        std::mutex arena_mutex;
        std::vector<std::unique_ptr<HistArena>> free_arenas;
        auto take_arena = [&]() -> std::unique_ptr<HistArena> {
            std::lock_guard<std::mutex> lock(arena_mutex);
            if (free_arenas.empty()) return std::make_unique<HistArena>();
            std::unique_ptr<HistArena> a = std::move(free_arenas.back());
            free_arenas.pop_back();
            return a;
        };
        auto give_arena = [&](std::unique_ptr<HistArena> a) {
            std::lock_guard<std::mutex> lock(arena_mutex);
            free_arenas.push_back(std::move(a));
        };

        for (std::size_t round = 0; round < n_est; ++round) {
            std::vector<RDTModel> round_trees;
            round_trees.reserve(n_outputs);
            for (std::size_t k = 0; k < n_outputs; ++k) {
                TreeParams tp = params_.tree_params;
                tp.task = Task::Regression;
                tp.criterion = newton() ? Criterion::Newton : Criterion::MSE;
                if (newton()) {
                    tp.newton.reg_lambda = params_.reg_lambda;
                    tp.newton.reg_alpha = params_.reg_alpha;
                    tp.newton.gamma = params_.tree_params.min_impurity_decrease;
                    tp.newton.min_child_weight = params_.min_child_weight;
                    tp.newton.max_delta_step = params_.max_delta_step;
                }
                tp.retain_indices = false; // see TreeParams::retain_indices
                // colsample_bylevel: the tree's own generator, seeded from the
                // (round, output) seed before any thread starts.
                tp.bylevel_seed = derive_seed(derive_seed(params_.random_state, round * 1000 + k + 1), kBylevelSeedIndex);
                round_trees.emplace_back(tp);
            }
            std::vector<std::vector<std::uint32_t>> round_fit_rows(n_outputs);
            std::vector<std::vector<std::uint32_t>> round_col_subsets(n_outputs);
            std::vector<std::vector<double>> round_raw_preds(n_outputs);

            // Row / column subsets. Each (round, output) has its own seeded
            // Rng, so drawing them here, sequentially and before the
            // residuals, gives the same subsets as drawing them inside the
            // parallel per-output work; it lets the columns the round needs be
            // presorted up front, outside the threads.
            for (std::size_t k = 0; k < n_outputs; ++k) {
                Rng rng(derive_seed(params_.random_state, round * 1000 + k + 1));

                std::vector<std::uint32_t>& fit_rows = round_fit_rows[k];
                if (params_.subsample < 1.0) {
                    std::uint32_t m = static_cast<std::uint32_t>(
                        std::max(1.0, params_.subsample * static_cast<double>(train_rows.size())));
                    auto local_idx = rng.sample_without_replacement(static_cast<std::uint32_t>(train_rows.size()), m);
                    fit_rows.reserve(local_idx.size());
                    for (auto li : local_idx) fit_rows.push_back(train_rows[li]);
                } else {
                    fit_rows = train_rows;
                }

                std::uint32_t k_features =
                    resolve_max_features(params_.max_features, static_cast<std::uint32_t>(n_features_));
                std::vector<std::uint32_t>& col_subset = round_col_subsets[k];
                if (k_features < n_features_) {
                    col_subset = rng.sample_without_replacement(static_cast<std::uint32_t>(n_features_), k_features);
                } else {
                    col_subset.resize(n_features_);
                    for (std::uint32_t f = 0; f < n_features_; ++f) col_subset[f] = f;
                }
                source.prepare(col_subset);
            }

            // One output's work for this round. tree_pool: threads for the
            // tree's own feature search (hist), only when the round has a
            // single tree; a multiclass round's trees run in parallel instead.
            // The per-row passes below (gradients, gathers, routing every row
            // to its leaf) run in row blocks on tree_pool.
            auto fit_output = [&](std::size_t k, ThreadPool* tree_pool) {
                std::vector<double>& residual_k = residual[k];
                if (newton()) {
                    std::vector<double>& hessian_k = hessian[k];
                    for_blocks(tree_pool, train_rows.size(), [&](std::size_t r0, std::size_t r1) {
                        for (std::size_t r = r0; r < r1; ++r) {
                            std::uint32_t i = train_rows[r];
                            const double w = sample_weight ? (*sample_weight)[i] : 1.0;
                            gradient_hessian(i, k, y_enc, F, n_outputs, w, residual_k[i], hessian_k[i]);
                        }
                    });
                } else {
                    for_blocks(tree_pool, train_rows.size(), [&](std::size_t r0, std::size_t r1) {
                        for (std::size_t r = r0; r < r1; ++r) {
                            std::uint32_t i = train_rows[r];
                            residual_k[i] = negative_gradient(i, k, y_enc, F, n_outputs);
                        }
                    });
                }
                const std::vector<std::uint32_t>& fit_rows = round_fit_rows[k];
                const std::vector<std::uint32_t>& col_subset = round_col_subsets[k];

                // Only the fit rows' residuals / weights are gathered.
                const std::size_t m = fit_rows.size();
                std::vector<double> fit_residual(m);
                std::vector<double> fit_weight, fit_hessian;
                if (sample_weight) fit_weight.resize(m);
                if (newton()) fit_hessian.resize(m);
                for_blocks(tree_pool, m, [&](std::size_t r0, std::size_t r1) {
                    for (std::size_t r = r0; r < r1; ++r) {
                        fit_residual[r] = residual_k[fit_rows[r]];
                        if (sample_weight) fit_weight[r] = (*sample_weight)[fit_rows[r]];
                        if (newton()) fit_hessian[r] = hessian[k][fit_rows[r]];
                    }
                });
                VectorView residual_view(fit_residual.data(), m);
                VectorView w_view(fit_weight.data(), fit_weight.size());
                VectorView h_view(fit_hessian.data(), fit_hessian.size());

                // Newton trees set their own leaf values (and take the
                // sample weights only to leave zero-weight rows out).
                std::unique_ptr<HistArena> arena = take_arena();
                source.fit_tree(round_trees[k], fit_rows, col_subset, residual_view,
                                sample_weight ? &w_view : nullptr, newton() ? &h_view : nullptr, tree_pool,
                                arena.get());
                give_arena(std::move(arena));
                // Rows are routed through the source (X and the tree's column
                // subset, or the bins), which evaluates exactly what the fit
                // partitioned on.
                auto leaf_of = [&](std::size_t row) { return source.leaf(round_trees[k], row, col_subset.data()); };
                if (!newton() && has_newton_leaf_step()) {
                    update_leaf_values(leaf_of, fit_rows, fit_residual, sample_weight ? &fit_weight : nullptr,
                                       y_enc[k], n_outputs);
                }

                std::vector<double> raw_pred(n);
                for_blocks(tree_pool, n, [&](std::size_t i0, std::size_t i1) {
                    for (std::size_t i = i0; i < i1; ++i) raw_pred[i] = shrink_leaf_value(leaf_of(i)->value);
                });
                round_raw_preds[k] = std::move(raw_pred);
            };
            if (Source::kFeatureParallel && n_outputs == 1) {
                fit_output(0, pool.size() > 1 ? &pool : nullptr);
            } else {
                pool.parallel_for(n_outputs, [&](std::size_t k) { fit_output(k, nullptr); });
            }

            ThreadPool* row_pool = pool.size() > 1 ? &pool : nullptr;
            for (std::size_t k = 0; k < n_outputs; ++k) {
                const std::vector<double>& raw = round_raw_preds[k];
                std::vector<double>& Fk = F[k];
                for_blocks(row_pool, n, [&](std::size_t i0, std::size_t i1) {
                    for (std::size_t i = i0; i < i1; ++i) Fk[i] += params_.learning_rate * raw[i];
                });
                trees_.push_back(std::move(round_trees[k]));
                feature_subsets_.push_back(std::move(round_col_subsets[k]));
            }
            rounds_fitted_ = round + 1;

            double train_loss = compute_loss(y_enc, F, train_rows, n_outputs, sample_weight, row_pool, loss_terms);
            train_score_.push_back(train_loss);

            if (early_stopping) {
                double val_loss = compute_loss(y_enc, F, val_rows, n_outputs, sample_weight, row_pool, loss_terms);
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

public:
    std::vector<double> predict(const MatrixView& X) const {
        std::vector<double> out(X.n_rows());
        scores_to_labels(accumulate_scores(X), X.n_rows(), out.data());
        return out;
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
        scores_to_proba(accumulate_scores(X), X.n_rows(), out);
    }

    // Raw scores (the init plus every round), row-major n_samples x
    // n_outputs() -- scikit-learn's decision_function.
    void decision_function_into(const MatrixView& X, double* out) const {
        scores_to_decision(accumulate_scores(X), X.n_rows(), out);
    }

    // Staged prediction, one round at a time: F starts at
    // init_score_matrix(n) and add_round(X, r, F) adds round r's trees for
    // r = 0, 1, ..., n_estimators_fitted() - 1. Each F[k][i] receives exactly
    // the operations accumulate_scores() performs, in the same order, so the
    // last stage equals predict()/predict_proba() bit for bit.
    std::vector<std::vector<double>> init_score_matrix(std::size_t n) const {
        std::vector<std::vector<double>> F(n_outputs_, std::vector<double>(n));
        for (std::size_t k = 0; k < n_outputs_; ++k) std::fill(F[k].begin(), F[k].end(), init_scores_[k]);
        return F;
    }

    void add_round(const MatrixView& X, std::size_t round, std::vector<std::vector<double>>& F) const {
        if (round >= rounds_fitted_) throw std::out_of_range("add_round: round out of range");
        const std::size_t n = X.n_rows();
        auto pool = predict_pool(n * n_outputs_);
        for (std::size_t k = 0; k < n_outputs_; ++k) {
            const std::size_t t = round * n_outputs_ + k;
            for_blocks(pool.get(), n, [&](std::size_t i0, std::size_t i1) {
                for (std::size_t i = i0; i < i1; ++i) add_tree_score(t, X, i, F[k][i]);
            });
        }
    }

    // Raw scores F (n_outputs x n) -> row-major n x n_classes probabilities.
    void scores_to_proba(const std::vector<std::vector<double>>& F, std::size_t n, double* out) const {
        const std::size_t n_cls = classes_.size();
        if (n_outputs_ == 1) {
            for (std::size_t i = 0; i < n; ++i) {
                double p1 = positive_proba(F[0][i]);
                // v0.2.0 also wrote out[i][1] for single-class y, past the end of the row
                // (fit now refuses fewer than 2 classes; the guards stay as cheap insurance).
                if (n_cls > 0) out[i * n_cls + 0] = 1.0 - p1;
                if (n_cls > 1) out[i * n_cls + 1] = p1;
            }
        } else {
            std::vector<double> exp_vals(n_outputs_);
            for (std::size_t i = 0; i < n; ++i) {
                double max_f = F[0][i];
                for (std::size_t k = 1; k < n_outputs_; ++k) max_f = std::max(max_f, F[k][i]);
                double sum = 0.0;
                for (std::size_t k = 0; k < n_outputs_; ++k) {
                    exp_vals[k] = std::exp(F[k][i] - max_f);
                    sum += exp_vals[k];
                }
                for (std::size_t k = 0; k < n_outputs_; ++k) out[i * n_cls + k] = exp_vals[k] / sum;
            }
        }
    }

    // Raw scores F -> predict()'s labels (classification) or values (regression), out[0..n).
    void scores_to_labels(const std::vector<std::vector<double>>& F, std::size_t n, double* out) const {
        if (params_.tree_params.task == Task::Regression) {
            std::copy(F[0].begin(), F[0].begin() + static_cast<std::ptrdiff_t>(n), out);
            return;
        }
        if (n_outputs_ == 1) { // binary
            for (std::size_t i = 0; i < n; ++i) {
                double p = positive_proba(F[0][i]);
                out[i] = p >= 0.5 ? classes_[1] : classes_[0];
            }
        } else {
            for (std::size_t i = 0; i < n; ++i) {
                std::size_t best = 0;
                double best_val = F[0][i];
                for (std::size_t k = 1; k < n_outputs_; ++k) {
                    if (F[k][i] > best_val) { best_val = F[k][i]; best = k; }
                }
                out[i] = classes_[best];
            }
        }
    }

    // Raw scores F (n_outputs x n) -> row-major n x n_outputs.
    void scores_to_decision(const std::vector<std::vector<double>>& F, std::size_t n, double* out) const {
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t k = 0; k < n_outputs_; ++k) out[i * n_outputs_ + k] = F[k][i];
        }
    }

    const std::vector<double>& classes() const { return classes_; }
    std::size_t n_classes() const { return classes_.size(); }
    const std::vector<double>& feature_importances() const { return feature_importances_; }
    const std::vector<double>& train_score() const { return train_score_; }
    std::size_t n_estimators_fitted() const { return rounds_fitted_; }
    std::size_t n_features() const { return n_features_; }
    // Trees per round: 1 (binary, regression) or the number of classes.
    std::size_t n_outputs() const { return n_outputs_; }
    const std::vector<double>& init_scores() const { return init_scores_; }
    // Incremented by every fit(); lets a staged iterator detect a refit.
    std::uint64_t fit_generation() const { return fit_generation_; }

    // Per tree (round-major), the candidate columns of each searched depth
    // (colsample_bylevel) as columns of X, replayed from each tree's seed.
    std::vector<std::map<int, std::vector<std::uint32_t>>> level_features() const {
        return map_level_features(trees_, feature_subsets_);
    }

    // Per tree (round-major, as trees are fitted), the feature chosen at
    // each depth as a column of the X passed to fit (each tree's own column
    // indices mapped through its column subset).
    // The fitted trees, round-major (trees()[round * n_outputs() + k]); a
    // tree's features are indices into its column subset.
    const std::vector<RDTModel>& trees() const { return trees_; }

    // Each tree's depth_improvements() (Newton: the depth's total gain), in
    // the order of depth_features().
    std::vector<std::map<int, double>> depth_improvements() const {
        std::vector<std::map<int, double>> out;
        out.reserve(trees_.size());
        for (const auto& tree : trees_) out.push_back(tree.depth_improvements());
        return out;
    }

    std::vector<std::map<int, int>> depth_features() const {
        std::vector<std::map<int, int>> out(trees_.size());
        for (std::size_t t = 0; t < trees_.size(); ++t) {
            for (const auto& kv : trees_[t].depth_features()) {
                out[t][kv.first] = static_cast<int>(feature_subsets_[t][static_cast<std::size_t>(kv.second)]);
            }
        }
        return out;
    }

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
    std::uint64_t fit_generation_ = 0;

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

    static double weighted_mean(const std::vector<double>& enc, const std::vector<std::uint32_t>& rows,
                                const VectorView* sample_weight) {
        double sum = 0.0, total_weight = 0.0;
        for (auto i : rows) {
            double w = sample_weight ? (*sample_weight)[i] : 1.0;
            sum += w * enc[i];
            total_weight += w;
        }
        return total_weight > 0.0 ? sum / total_weight : 0.0;
    }

    // The element of enc[rows] at 0-based sorted position `rank`, where
    // weights act as replication counts rescaled to sum to rows.size():
    // the first value whose cumulative (rescaled) weight exceeds `rank`.
    // Unweighted it is exactly sorted[floor(rank)].
    static double order_statistic(const std::vector<double>& enc, const std::vector<std::uint32_t>& rows,
                                  const VectorView* sample_weight, double rank) {
        if (!sample_weight) {
            std::vector<double> vals;
            vals.reserve(rows.size());
            for (auto i : rows) vals.push_back(enc[i]);
            std::sort(vals.begin(), vals.end());
            return vals[std::min(static_cast<std::size_t>(rank), vals.size() - 1)];
        }
        std::vector<std::pair<double, double>> vw;
        vw.reserve(rows.size());
        double total_weight = 0.0;
        for (auto i : rows) {
            vw.emplace_back(enc[i], (*sample_weight)[i]);
            total_weight += (*sample_weight)[i];
        }
        std::sort(vw.begin(), vw.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        const double n = static_cast<double>(rows.size());
        double cum = 0.0;
        for (const auto& v : vw) {
            cum += v.second;
            if (cum * n > rank * total_weight) return v.first;
        }
        return vw.back().first;
    }

    // Priors are clipped to [eps, 1 - eps] before the logit/log, with
    // scikit-learn's eps (float64 machine epsilon).
    static constexpr double kProbaEps = std::numeric_limits<double>::epsilon();

    static double total_weight(const VectorView& w, const std::vector<std::uint32_t>& rows) {
        double total = 0.0;
        for (auto i : rows) total += w[i];
        return total;
    }

    // Constant initial raw score per output: the (weighted) minimiser of the
    // loss over the training rows, as scikit-learn's default init does.
    std::vector<double> compute_init_scores(const std::vector<std::vector<double>>& y_enc,
                                             const std::vector<std::uint32_t>& rows,
                                             const VectorView* sample_weight) const {
        std::vector<double> init(n_outputs_, 0.0);
        if (params_.loss == GBLoss::LogLoss && n_outputs_ > 1) {
            // Multiclass: log of each class's weighted prior, centred to sum
            // to zero (softmax is shift-invariant; sklearn's symmetric link).
            double mean_log = 0.0;
            for (std::size_t k = 0; k < n_outputs_; ++k) {
                double p = std::min(std::max(weighted_mean(y_enc[k], rows, sample_weight), kProbaEps), 1.0 - kProbaEps);
                init[k] = std::log(p);
                mean_log += init[k] / static_cast<double>(n_outputs_);
            }
            for (double& v : init) v -= mean_log;
            return init;
        }
        const std::vector<double>& enc = y_enc[0];
        const double n = static_cast<double>(rows.size());
        switch (params_.loss) {
            case GBLoss::LogLoss: {
                double p = std::min(std::max(weighted_mean(enc, rows, sample_weight), kProbaEps), 1.0 - kProbaEps);
                init[0] = std::log(p / (1.0 - p));
                break;
            }
            case GBLoss::Exponential: {
                double mean = weighted_mean(enc, rows, sample_weight); // enc in {-1,+1}
                double p = std::min(std::max((mean + 1.0) / 2.0, kProbaEps), 1.0 - kProbaEps);
                init[0] = 0.5 * std::log(p / (1.0 - p));
                break;
            }
            case GBLoss::AbsoluteError:
                init[0] = order_statistic(enc, rows, sample_weight, n / 2.0);
                break;
            case GBLoss::Quantile:
                init[0] = order_statistic(enc, rows, sample_weight, params_.quantile_alpha * (n - 1.0));
                break;
            default: // SquaredError, Huber
                init[0] = weighted_mean(enc, rows, sample_weight);
        }
        return init;
    }

    bool newton() const { return params_.split_criterion == SplitCriterion::Newton; }

    // Newton: gradient dL/dF and hessian of the loss w.r.t. output k's raw
    // score at sample i, times the sample weight w, as XGBoost computes them
    // (binary:logistic, multi:softprob, reg:squarederror; exponential by the
    // same rules).
    void gradient_hessian(std::size_t i, std::size_t k, const std::vector<std::vector<double>>& y_enc,
                          const std::vector<std::vector<double>>& F, std::size_t n_outputs, double w,
                          double& g, double& h) const {
        constexpr double kHessEps = 1e-16; // XGBoost's floor on logistic / softmax hessians
        const double target = y_enc[k][i];
        const double f = F[k][i];
        switch (params_.loss) {
            case GBLoss::LogLoss:
                if (n_outputs == 1) {
                    const double p = sigmoid_like(f);
                    g = (p - target) * w;
                    h = std::max(p * (1.0 - p), kHessEps) * w;
                } else {
                    double max_f = F[0][i];
                    for (std::size_t j = 1; j < n_outputs; ++j) max_f = std::max(max_f, F[j][i]);
                    double sum = 0.0;
                    for (std::size_t j = 0; j < n_outputs; ++j) sum += std::exp(F[j][i] - max_f);
                    const double p = std::exp(f - max_f) / sum;
                    g = (p - target) * w;
                    h = std::max(2.0 * p * (1.0 - p) * w, kHessEps);
                }
                return;
            case GBLoss::Exponential: {
                const double e = std::exp(-target * f); // target in {-1, +1}
                g = -target * e * w;
                h = e * w;
                return;
            }
            case GBLoss::SquaredError:
                g = (f - target) * w;
                h = w;
                return;
            default:
                throw std::logic_error("gradient_hessian: loss has no hessian");
        }
    }

    bool has_newton_leaf_step() const {
        return params_.loss == GBLoss::LogLoss || params_.loss == GBLoss::Exponential;
    }

    // Friedman's terminal-region update (scikit-learn's
    // _update_terminal_regions): replace each leaf's mean-of-gradient value
    // by one Newton step of the loss, sum(w*g) / sum(w*h), over the in-bag
    // rows reaching that leaf. g is the negative gradient the tree was fit
    // to; the hessian follows from it: p*(1-p) with p = y - g for log_loss,
    // |g| = exp(-y~*F) for exponential. Multiclass scales the numerator by
    // (K-1)/K. A leaf whose hessian sum is ~0 (all p in {0,1}) gets 0.
    // `leaf_of(row)` is the (mutable) leaf the row reaches.
    template <class LeafOf>
    void update_leaf_values(LeafOf&& leaf_of, const std::vector<std::uint32_t>& fit_rows,
                            const std::vector<double>& fit_residual, const std::vector<double>* fit_weight,
                            const std::vector<double>& enc, std::size_t n_outputs) const {
        std::unordered_map<Node*, std::pair<double, double>> sums; // leaf -> (sum w*g, sum w*h)
        for (std::size_t r = 0; r < fit_rows.size(); ++r) {
            const double g = fit_residual[r];
            double h;
            if (params_.loss == GBLoss::Exponential) {
                h = std::abs(g);
            } else {
                const double p = enc[fit_rows[r]] - g;
                h = p * (1.0 - p);
            }
            const double w = fit_weight ? (*fit_weight)[r] : 1.0;
            auto& acc = sums[leaf_of(fit_rows[r])];
            acc.first += w * g;
            acc.second += w * h;
        }
        const double scale = (params_.loss == GBLoss::LogLoss && n_outputs > 1)
            ? static_cast<double>(n_outputs - 1) / static_cast<double>(n_outputs) : 1.0;
        for (auto& kv : sums) {
            const double den = kv.second.second;
            kv.first->value = std::abs(den) < 1e-150 ? 0.0 : scale * kv.second.first / den;
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

    // Friedman mode's post-hoc reg_lambda / reg_alpha shrink (newton leaves
    // are already regularised).
    double shrink_leaf_value(double v) const {
        if (newton() || (params_.reg_lambda == 0.0 && params_.reg_alpha == 0.0)) return v;
        double shrunk = v / (1.0 + params_.reg_lambda);
        double sign = shrunk > 0 ? 1.0 : (shrunk < 0 ? -1.0 : 0.0);
        return sign * std::max(std::abs(shrunk) - params_.reg_alpha, 0.0);
    }

    static double sigmoid_like(double f) { return 1.0 / (1.0 + std::exp(-f)); }

    // P(positive class) from a binary raw score. The exponential loss's raw
    // score estimates half the log-odds (its init is 0.5*logit(p)), so its
    // link is sigmoid(2F), as in scikit-learn.
    double positive_proba(double f) const {
        return sigmoid_like(params_.loss == GBLoss::Exponential ? 2.0 * f : f);
    }

    // Mean loss over `rows`, sample-weighted when weights are given -- the
    // same objective the trees and leaf values optimise, for train_score_
    // and the early-stopping validation loss. Each row's loss is computed in
    // row blocks on `pool` (into `terms`), and the weighted sum is then taken
    // serially in row order, so the result does not depend on the pool.
    double compute_loss(const std::vector<std::vector<double>>& y_enc, const std::vector<std::vector<double>>& F,
                         const std::vector<std::uint32_t>& rows, std::size_t n_outputs,
                         const VectorView* sample_weight, ThreadPool* pool, std::vector<double>& terms) const {
        if (rows.empty()) return 0.0;
        terms.resize(rows.size());
        for_blocks(pool, rows.size(), [&](std::size_t r0, std::size_t r1) {
            for (std::size_t r = r0; r < r1; ++r) terms[r] = row_loss(y_enc, F, rows[r], n_outputs);
        });
        double weighted_total = 0.0, total_weight = 0.0;
        for (std::size_t r = 0; r < rows.size(); ++r) {
            const double w = sample_weight ? (*sample_weight)[rows[r]] : 1.0;
            weighted_total += w * terms[r];
            total_weight += w;
        }
        return total_weight > 0.0 ? weighted_total / total_weight : 0.0;
    }

    // The loss of row i (summed over the outputs of a multiclass model).
    double row_loss(const std::vector<std::vector<double>>& y_enc, const std::vector<std::vector<double>>& F,
                    std::size_t i, std::size_t n_outputs) const {
        double total = 0.0;
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
        return total;
    }

    // F += learning_rate * (tree t's raw score for row i of X), predicting
    // through the tree's column subset (no copy of X). The one place a tree's
    // contribution is added, shared by predict and the staged predictions.
    void add_tree_score(std::size_t t, const MatrixView& X, std::size_t i, double& F) const {
        double raw = shrink_leaf_value(trees_[t].find_leaf(X, i, feature_subsets_[t].data())->value);
        F += params_.learning_rate * raw;
    }

    // Threads to predict with (n_jobs), when `work` (rows x trees) is large
    // enough to be worth starting them; nullptr otherwise.
    std::unique_ptr<ThreadPool> predict_pool(std::size_t work) const {
        constexpr std::size_t kMinParallelWork = std::size_t{1} << 18;
        const std::size_t threads = resolve_n_jobs(params_.n_jobs);
        if (threads < 2 || work < kMinParallelWork) return nullptr;
        return std::make_unique<ThreadPool>(threads);
    }

    // Raw scores of every fitted tree. Rows are processed in blocks across
    // all trees (blocks in parallel, n_jobs); each F[k][i] still accumulates
    // trees in round order.
    std::vector<std::vector<double>> accumulate_scores(const MatrixView& X) const {
        const std::size_t n = X.n_rows();
        std::vector<std::vector<double>> F = init_score_matrix(n);
        auto pool = predict_pool(n * trees_.size());
        for_blocks(pool.get(), n, [&](std::size_t r0, std::size_t r1) {
            for (std::size_t b = r0; b < r1; b += kPredictRowBlock) {
                const std::size_t e = std::min(r1, b + kPredictRowBlock);
                for (std::size_t t = 0; t < trees_.size(); ++t) {
                    std::size_t k = t % n_outputs_;
                    for (std::size_t i = b; i < e; ++i) add_tree_score(t, X, i, F[k][i]);
                }
            }
        });
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
