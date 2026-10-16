#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "rdt/criteria.hpp"
#include "rdt/hist.hpp"
#include "rdt/matrix.hpp"
#include "rdt/node.hpp"
#include "rdt/params.hpp"
#include "rdt/presort.hpp"

// Port of rdt_optimal_cython/optimal_rdt_cython.py's OptimalRDTCython.
//
// Per-depth greedy search: for each depth, every candidate feature is
// scored across all currently-active nodes at that depth (summing each
// node's best achievable impurity improvement for the feature, weighted by
// the node's share of the training weight -- scikit-learn's weighted impurity
// decrease N_t/N * (impurity - N_tL/N_t * left - N_tR/N_t * right)); the
// feature with the highest total wins and is locked in for the whole depth, and the
// per-node thresholds already found while scoring it are reused directly as
// the final split points (see the plan's "two-pass" note: this is
// result-equivalent to the paper's Algorithm 1, not an approximation of it).
//
// `sample_weight` (nullptr = uniform 1.0) and `min_impurity_decrease` exist
// primarily to support the ensemble layer (class_weight, sample_weight,
// gamma/min_split_loss pruning) built on top of this core; passing neither
// reproduces the original unweighted, unpruned behavior exactly.
//
// The per-depth feature scoring normally runs on presorted columns (see
// presort.hpp and TreeParams::presort); the per-(node, feature) sort in
// find_best_threshold remains for MAE and as the reference implementation.
//
// Criterion::Newton (gradient boosting's split_criterion='newton') grows the
// tree XGBoost's way on per-row gradients `y` and hessians `hessian`: a node's
// best split maximises the gain score(L) + score(R) - score(node) (see
// NewtonParams), and is kept only if the gain is >= gamma and > 1e-6 with both
// children's hessian sums >= min_child_weight (nodes without such a split
// become leaves as the tree grows). A depth's feature maximises the sum over
// its splitting nodes of (gain - gamma); depth_improvements() holds the sum of
// the gains (XGBoost's total_gain). Leaves hold -T(G) / (H + lambda), clipped
// by max_delta_step. The node purity test does not apply.
namespace rdt {

class RDTModel {
public:
    explicit RDTModel(TreeParams params) : params_(params) {}

    RDTModel(const RDTModel&) = delete;
    RDTModel& operator=(const RDTModel&) = delete;
    RDTModel(RDTModel&&) = default;
    RDTModel& operator=(RDTModel&&) = default;

    void fit(const MatrixView& X, const VectorView& y, const VectorView* sample_weight = nullptr,
             const std::vector<double>* classes = nullptr, const VectorView* hessian = nullptr) {
        if (params_.tree_method == TreeMethod::Hist) {
            fit_hist(X, y, sample_weight, classes, hessian);
            return;
        }
        fit(IndexedMatrixView(X), y, sample_weight, nullptr, classes, hessian);
    }

    // tree_method='hist' on a whole matrix (MatrixView or
    // StridedMatrix<float/double>, read without copying): bins it (cuts from
    // the rows with positive weight, see hist.hpp), fits on the bins with
    // TreeParams::n_jobs threads, and drops the bins.
    template <class Matrix>
    void fit_hist(const Matrix& X, const VectorView& y, const VectorView* sample_weight = nullptr,
                  const std::vector<double>* classes = nullptr, const VectorView* hessian = nullptr) {
        if (params_.criterion == Criterion::MAE) {
            throw std::invalid_argument("tree_method='hist' does not support criterion='mae'");
        }
        std::vector<std::uint32_t> cut_rows;
        cut_rows.reserve(X.n_rows());
        for (std::size_t i = 0; i < X.n_rows(); ++i) {
            if (!sample_weight || (*sample_weight)[i] > 0.0) cut_rows.push_back(static_cast<std::uint32_t>(i));
        }
        const std::size_t threads = resolve_n_jobs(params_.n_jobs);
        std::unique_ptr<ThreadPool> pool;
        if (threads > 1) pool = std::make_unique<ThreadPool>(threads);
        BinnedMatrix bins(X, cut_rows, params_.max_bin, pool.get());
        fit_binned(BinnedView(bins), y, sample_weight, classes, hessian, pool.get());
    }

    // Fit on a row/column-mapped view of a matrix (e.g. a bootstrap sample
    // restricted to a column subset) without copying it; y / sample_weight
    // are indexed by the view's rows. `presorted`, if given, holds the view's
    // rows presorted by each of its columns (presort.hpp), for callers that
    // fit many trees on related rows (boosting rounds) and sort only once;
    // otherwise the tree presorts its rows itself when TreeParams::presort is
    // set.
    //
    // `classes` (classification only; nullptr = the distinct labels of y)
    // fixes the class list that classes(), leaf class_counts and
    // predict_proba() columns are aligned to. An ensemble passes its full
    // class list so a tree whose bootstrap sample lacks some class still
    // produces probability rows in the ensemble's column order. It must be
    // sorted, unique, and contain every label in y (std::invalid_argument
    // otherwise).
    //
    // Criterion::Newton: y holds the per-row gradients and `hessian` (required)
    // the hessians, both already multiplied by the sample weight; rows whose
    // sample_weight is 0 are still left out.
    void fit(const IndexedMatrixView& X, const VectorView& y, const VectorView* sample_weight = nullptr,
             const PresortedView* presorted = nullptr, const std::vector<double>* classes = nullptr,
             const VectorView* hessian = nullptr) {
        std::vector<std::uint32_t> root_indices = begin_fit(X.n_cols(), X.n_rows(), y, sample_weight, classes, hessian);

        if (presorted) {
            bool ok = presorted->n_entries == X.n_rows() && presorted->columns.size() == X.n_cols();
            // Every column holds the same rows; checking the first one's ids is O(n).
            if (ok && !presorted->columns.empty()) {
                const std::uint32_t* col = presorted->columns[0];
                for (std::size_t i = 0; ok && i < presorted->n_entries; ++i) {
                    ok = (col[i] & kPresortRowMask) < X.n_rows();
                }
            }
            if (!ok) throw std::invalid_argument("presorted columns must cover exactly X's rows and columns");
        }

        const bool presort = params_.presort && DepthScanner::supports(params_.criterion) &&
                             X.n_rows() <= kPresortMaxRows;
        SortedColumns own_sorted;
        PresortedView own_view;
        if (presort && presorted == nullptr) {
            own_sorted = SortedColumns(X, root_indices);
            own_view = own_sorted.view();
            presorted = &own_view;
        }
        root_ = build_tree(X, X.n_rows(), y, sample_weight, std::move(root_indices), presort ? presorted : nullptr);
        hessian_ = nullptr;
    }

    // Fit on binned features (tree_method='hist', hist.hpp): `X` maps the
    // tree's rows / columns to a BinnedMatrix (no copy); y, sample_weight,
    // classes and hessian as in fit(). Each node's threshold is the cut after
    // its split bin, so predicting with the raw X routes every row as its
    // bins did. `pool` (may be nullptr) scores a depth's features in parallel;
    // the result does not depend on it. Not available for MAE.
    void fit_binned(const BinnedView& X, const VectorView& y, const VectorView* sample_weight = nullptr,
                    const std::vector<double>* classes = nullptr, const VectorView* hessian = nullptr,
                    ThreadPool* pool = nullptr) {
        if (params_.criterion == Criterion::MAE) {
            throw std::invalid_argument("tree_method='hist' does not support criterion='mae'");
        }
        std::vector<std::uint32_t> root_indices = begin_fit(X.n_cols, X.n_rows, y, sample_weight, classes, hessian);
        root_ = build_tree(IndexedMatrixView(), X.n_rows, y, sample_weight, std::move(root_indices), nullptr, &X, pool);
        hessian_ = nullptr;
    }

    std::vector<double> predict(const MatrixView& X) const {
        std::vector<double> out(X.n_rows());
        for (std::size_t i = 0; i < X.n_rows(); ++i) out[i] = find_leaf(X, i)->value;
        return out;
    }

    // n_samples x n_classes, classes ordered per classes().
    std::vector<std::vector<double>> predict_proba(const MatrixView& X) const {
        std::vector<std::vector<double>> out(X.n_rows(), std::vector<double>(classes_.size(), 0.0));
        for (std::size_t i = 0; i < X.n_rows(); ++i) leaf_proba(find_leaf(X, i), out[i].data());
        return out;
    }

    // Same as predict_proba, into a row-major n_samples x n_classes buffer.
    void predict_proba_into(const MatrixView& X, double* out) const {
        const std::size_t k = classes_.size();
        for (std::size_t i = 0; i < X.n_rows(); ++i) leaf_proba(find_leaf(X, i), out + i * k);
    }

    // Leaf reached by row `row` of X. `col_map` (nullptr = identity) maps
    // this tree's feature indices to X's columns -- how the ensembles predict
    // through a tree fitted on a column subset without copying X.
    const Node* find_leaf(const MatrixView& X, std::size_t row, const std::uint32_t* col_map = nullptr) const {
        const Node* node = root_.get();
        while (!node->is_leaf()) {
            const std::size_t col = col_map ? static_cast<std::size_t>(col_map[node->split_feature])
                                            : static_cast<std::size_t>(node->split_feature);
            node = (X(row, col) <= node->split_threshold) ? node->left.get() : node->right.get();
        }
        return node;
    }

    // Leaf reached by row `row` of a BinnedMatrix through the split bins (a
    // hist tree; `col_map` as for find_leaf). Same leaf as find_leaf on the
    // raw row.
    const Node* find_leaf_binned(const BinnedMatrix& bins, std::size_t row, const std::uint32_t* col_map = nullptr) const {
        const Node* node = root_.get();
        while (!node->is_leaf()) {
            const std::size_t col = col_map ? static_cast<std::size_t>(col_map[node->split_feature])
                                            : static_cast<std::size_t>(node->split_feature);
            node = bins.bin(row, col) <= static_cast<std::uint32_t>(node->split_bin) ? node->left.get()
                                                                                       : node->right.get();
        }
        return node;
    }

    Node* leaf_for_binned(const BinnedMatrix& bins, std::size_t row, const std::uint32_t* col_map = nullptr) {
        return const_cast<Node*>(find_leaf_binned(bins, row, col_map));
    }

    // Mutable find_leaf, so gradient boosting can overwrite leaf values after
    // the fit (its per-leaf line search); the tree structure itself must not
    // be modified.
    Node* leaf_for(const MatrixView& X, std::size_t row, const std::uint32_t* col_map = nullptr) {
        return const_cast<Node*>(find_leaf(X, row, col_map));
    }

    // A leaf's class probabilities (classes() order) into out[0..n_classes()).
    void leaf_proba(const Node* leaf, double* out) const {
        double total = 0.0;
        for (double c : leaf->class_counts) total += c;
        for (std::size_t i = 0; i < classes_.size(); ++i) out[i] = 0.0;
        if (total > 0.0) {
            for (std::size_t i = 0; i < classes_.size(); ++i) out[i] = leaf->class_counts[i] / total;
        }
    }

    const std::vector<double>& classes() const { return classes_; }
    std::size_t n_classes() const { return classes_.size(); }
    const std::map<int, int>& depth_features() const { return depth_features_; }
    // Weighted impurity decrease achieved by the winning feature at each
    // depth, sum over the depth's split nodes of N_t/N * (node improvement)
    // (N = total training weight) -- scikit-learn's per-split importance
    // definition, and the basis for feature_importances_ in the ensembles.
    const std::map<int, double>& depth_improvements() const { return depth_improvements_; }
    const Node* root() const { return root_.get(); }
    const TreeParams& params() const { return params_; }
    std::size_t n_features() const { return n_features_; }

private:
    TreeParams params_;
    std::unique_ptr<Node> root_;
    std::vector<double> classes_;
    std::map<int, int> depth_features_;
    std::map<int, double> depth_improvements_;
    std::size_t n_features_ = 0;
    const VectorView* hessian_ = nullptr; // Newton: the hessians, during fit only

    bool newton() const { return params_.criterion == Criterion::Newton; }

    // Shared start of fit / fit_binned: checks, classes, and the root's rows
    // (every row with a non-zero weight).
    std::vector<std::uint32_t> begin_fit(std::size_t n_cols, std::size_t n_rows, const VectorView& y,
                                         const VectorView* sample_weight, const std::vector<double>* classes,
                                         const VectorView* hessian) {
        if (params_.criterion == Criterion::Newton) {
            if (params_.task != Task::Regression) {
                throw std::invalid_argument("RDTModel::fit: criterion 'newton' needs task='regression'");
            }
            if (!hessian || hessian->size() != y.size()) {
                throw std::invalid_argument("RDTModel::fit: criterion 'newton' needs a hessian per row");
            }
        }
        hessian_ = params_.criterion == Criterion::Newton ? hessian : nullptr;
        n_features_ = n_cols;
        depth_features_.clear();
        depth_improvements_.clear();
        classes_.clear();

        if (params_.task == Task::Classification && classes) {
            classes_ = *classes;
            if (!std::is_sorted(classes_.begin(), classes_.end()) ||
                std::adjacent_find(classes_.begin(), classes_.end()) != classes_.end()) {
                throw std::invalid_argument("RDTModel::fit: classes must be sorted and unique");
            }
            for (std::size_t i = 0; i < y.size(); ++i) {
                if (!std::binary_search(classes_.begin(), classes_.end(), y[i])) {
                    throw std::invalid_argument("RDTModel::fit: y contains a label that is not in classes");
                }
            }
        } else if (params_.task == Task::Classification) {
            std::vector<double> labels(y.size());
            for (std::size_t i = 0; i < y.size(); ++i) labels[i] = y[i];
            std::sort(labels.begin(), labels.end());
            labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
            classes_ = std::move(labels);
        }

        // Zero-weight samples carry no information. Like scikit-learn's
        // splitter, leave them out of the tree entirely (equivalent to
        // deleting those rows): otherwise a split could isolate them into a
        // leaf with zero total weight, i.e. all-zero class counts and a
        // predict_proba row summing to 0, and they would shift thresholds and
        // count towards min_samples_split/min_samples_leaf. (With presorted
        // columns supplied by the caller they are simply never routed to a
        // node, see build_tree.)
        std::vector<std::uint32_t> root_indices;
        root_indices.reserve(n_rows);
        for (std::size_t i = 0; i < n_rows; ++i) {
            if (sample_weight && (*sample_weight)[i] == 0.0) continue;
            root_indices.push_back(static_cast<std::uint32_t>(i));
        }
        return root_indices;
    }

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

    void make_leaf(Node& node, const VectorView& y, const VectorView* sample_weight) const {
        node.has_value = true;
        if (newton()) {
            double G = 0.0, H = 0.0;
            node_sums(node, y, G, H);
            node.value = newton_leaf_weight(G, H, params_.newton);
            return;
        }
        if (params_.task == Task::Classification) {
            std::vector<double> seen_labels;
            std::unordered_map<double, std::size_t> label_to_pos;
            std::vector<double> weighted_counts;
            for (auto idx : node.indices) {
                double v = y[idx];
                double w = sample_weight ? (*sample_weight)[idx] : 1.0;
                auto it = label_to_pos.find(v);
                if (it == label_to_pos.end()) {
                    label_to_pos.emplace(v, seen_labels.size());
                    seen_labels.push_back(v);
                    weighted_counts.push_back(w);
                } else {
                    weighted_counts[it->second] += w;
                }
            }
            std::size_t best_pos = 0;
            double best_count = weighted_counts.empty() ? 0.0 : weighted_counts[0];
            for (std::size_t i = 1; i < weighted_counts.size(); ++i) {
                if (weighted_counts[i] > best_count) {
                    best_count = weighted_counts[i];
                    best_pos = i;
                }
            }
            node.value = seen_labels.empty() ? 0.0 : seen_labels[best_pos];

            node.class_counts.assign(classes_.size(), 0.0);
            for (std::size_t i = 0; i < seen_labels.size(); ++i) {
                auto it = std::lower_bound(classes_.begin(), classes_.end(), seen_labels[i]);
                std::size_t cls_idx = static_cast<std::size_t>(it - classes_.begin());
                node.class_counts[cls_idx] = weighted_counts[i];
            }
        } else {
            double sum = 0.0, total_weight = 0.0;
            for (auto idx : node.indices) {
                double w = sample_weight ? (*sample_weight)[idx] : 1.0;
                sum += w * y[idx];
                total_weight += w;
            }
            node.value = total_weight > 0.0 ? sum / total_weight : 0.0;
        }
    }

    // Winning feature of a depth: per-node thresholds, aligned with the
    // `nodes` passed in (found[k] == 0: node k has no valid split on it).
    struct DepthChoice {
        int feature = -1;
        double total_improvement = -std::numeric_limits<double>::infinity();
        double gain_sum = 0.0; // Newton: sum of the splitting nodes' gains
        std::vector<char> found;
        std::vector<double> threshold;
        std::vector<std::uint32_t> split_bin; // hist: node k's rows with bin <= split_bin[k] go left
    };

    // Newton: a node's gradient / hessian totals, summed over its rows in
    // row order (independent of the feature being scored).
    void node_sums(const Node& node, const VectorView& g, double& G, double& H) const {
        for (auto idx : node.indices) {
            G += g[idx];
            H += (*hessian_)[idx];
        }
    }

    // Adds node k's best split on a feature (`found`, `improvement`) to the
    // feature's score for the depth; returns whether node k splits on it.
    // Impurity criteria: the node's improvement weighted by its share of the
    // training weight. Newton: the node splits only if its gain is >= gamma
    // and > kNewtonMinGain, and adds gain - gamma.
    bool add_node_score(bool found, double improvement, double share, double& total, double& gain_sum) const {
        if (!found) return false;
        if (newton()) {
            const double gamma = params_.newton.gamma;
            if (!(improvement >= gamma && improvement > kNewtonMinGain)) return false;
            total += improvement - gamma;
            gain_sum += improvement;
            return true;
        }
        total += share * improvement;
        return true;
    }

    // Reference search: sort each (node, feature) pair (find_best_threshold).
    // A node's improvement is weighted by its share of the total training
    // weight (`node_share[k]`, see build_tree), so a feature that helps a tiny
    // node a lot can't outscore one that helps a large node -- the per-node
    // improvement is a within-node quantity and is not comparable across
    // nodes otherwise.
    DepthChoice find_optimal_feature_for_depth(
        const IndexedMatrixView& X, const VectorView& y, const VectorView* sample_weight,
        const std::vector<Node*>& nodes, const std::vector<double>& node_share,
        const std::vector<double>& node_g, const std::vector<double>& node_h) const {

        DepthChoice best;
        std::vector<char> found(nodes.size());
        std::vector<double> thresholds(nodes.size());

        for (int feature_idx = 0; feature_idx < static_cast<int>(n_features_); ++feature_idx) {
            double total_improvement = 0.0, gain_sum = 0.0;
            bool valid_split_found = false;

            for (std::size_t k = 0; k < nodes.size(); ++k) {
                ThresholdResult r = newton()
                    ? find_best_threshold_newton(X, y, *hessian_, nodes[k]->indices, feature_idx, node_g[k],
                                                 node_h[k], params_.min_samples_split, params_.min_samples_leaf,
                                                 params_.newton)
                    : find_best_threshold(X, y, nodes[k]->indices, feature_idx, params_.criterion,
                                          params_.min_samples_split, params_.min_samples_leaf, sample_weight);
                found[k] = add_node_score(r.found, r.improvement, node_share[k], total_improvement, gain_sum);
                thresholds[k] = r.threshold;
                if (found[k]) valid_split_found = true;
            }

            if (valid_split_found && total_improvement > best.total_improvement) {
                best.total_improvement = total_improvement;
                best.gain_sum = gain_sum;
                best.feature = feature_idx;
                best.found = found;
                best.threshold = thresholds;
            }
        }
        return best;
    }

    // Presorted search: the same choice, scoring each feature for all nodes
    // in one pass over its presorted rows. Requires scan_rows[i].node == k
    // for every row i of nodes[k] and -1 for every other row.
    DepthChoice find_optimal_feature_presorted(
        const IndexedMatrixView& X, const std::vector<Node*>& nodes, const std::vector<double>& node_share,
        const std::vector<double>& node_g, const std::vector<double>& node_h,
        const PresortedView& presorted, const std::vector<ScanRow>& scan_rows, DepthScanner& scanner) const {

        std::vector<std::size_t> node_sizes(nodes.size());
        for (std::size_t k = 0; k < nodes.size(); ++k) node_sizes[k] = nodes[k]->indices.size();
        scanner.begin_depth(node_sizes, &node_g, &node_h);

        DepthChoice best;
        std::vector<NodeSplit> splits, best_splits;
        for (int feature_idx = 0; feature_idx < static_cast<int>(n_features_); ++feature_idx) {
            scanner.score_feature(presorted.columns[static_cast<std::size_t>(feature_idx)], presorted.n_entries,
                                  scan_rows.data(), splits);
            double total_improvement = 0.0, gain_sum = 0.0;
            bool valid_split_found = false;
            for (std::size_t k = 0; k < splits.size(); ++k) {
                splits[k].found = add_node_score(splits[k].found, splits[k].improvement, node_share[k],
                                                 total_improvement, gain_sum);
                if (splits[k].found) valid_split_found = true;
            }
            if (valid_split_found && total_improvement > best.total_improvement) {
                best.total_improvement = total_improvement;
                best.gain_sum = gain_sum;
                best.feature = feature_idx;
                best_splits = splits;
            }
        }
        if (best.feature >= 0) {
            best.found.assign(nodes.size(), 0);
            best.threshold.assign(nodes.size(), 0.0);
            const std::size_t f = static_cast<std::size_t>(best.feature);
            for (std::size_t k = 0; k < nodes.size(); ++k) {
                if (!best_splits[k].found) continue;
                best.found[k] = 1;
                best.threshold[k] = split_threshold(X(best_splits[k].lo_row, f), X(best_splits[k].hi_row, f));
            }
        }
        return best;
    }

    // Histogram search (tree_method='hist', hist.hpp): the same choice rule as
    // the other searches. Each feature's per-node results are computed by
    // one task (features are split into contiguous chunks across `pool`) and
    // reduced here in feature order, so the result does not depend on the
    // number of threads.
    DepthChoice find_optimal_feature_hist(
        const BinnedView& X, const VectorView& y, const VectorView* sample_weight,
        const std::vector<Node*>& nodes, const std::vector<double>& node_share,
        const std::vector<double>& node_g, const std::vector<double>& node_h,
        const std::vector<std::int32_t>& cls, HistScanner& scanner, ThreadPool* pool) const {

        std::vector<const std::vector<std::uint32_t>*> node_rows(nodes.size());
        for (std::size_t k = 0; k < nodes.size(); ++k) node_rows[k] = &nodes[k]->indices;
        scanner.begin_depth(X, node_rows, y, newton() ? nullptr : sample_weight, hessian_, cls.data(), &node_g, &node_h);

        const std::size_t n_feat = n_features_;
        std::vector<std::vector<HistSplit>> results(n_feat);
        const std::size_t n_chunks = pool ? std::min(n_feat, 4 * pool->size()) : 1;
        run_tasks(pool, n_chunks, [&](std::size_t c) {
            HistScanner::Scratch scratch;
            const std::size_t f0 = n_feat * c / n_chunks, f1 = n_feat * (c + 1) / n_chunks;
            for (std::size_t f = f0; f < f1; ++f) scanner.score_feature(X, f, results[f], scratch);
        });

        DepthChoice best;
        std::size_t best_f = 0;
        for (std::size_t f = 0; f < n_feat; ++f) {
            std::vector<HistSplit>& splits = results[f];
            double total_improvement = 0.0, gain_sum = 0.0;
            bool valid_split_found = false;
            for (std::size_t k = 0; k < splits.size(); ++k) {
                splits[k].found = add_node_score(splits[k].found, splits[k].improvement, node_share[k],
                                                 total_improvement, gain_sum);
                if (splits[k].found) valid_split_found = true;
            }
            if (valid_split_found && total_improvement > best.total_improvement) {
                best.total_improvement = total_improvement;
                best.gain_sum = gain_sum;
                best.feature = static_cast<int>(f);
                best_f = f;
            }
        }
        if (best.feature >= 0) {
            best.found.assign(nodes.size(), 0);
            best.threshold.assign(nodes.size(), 0.0);
            best.split_bin.assign(nodes.size(), 0);
            for (std::size_t k = 0; k < nodes.size(); ++k) {
                const HistSplit& sp = results[best_f][k];
                if (!sp.found) continue;
                best.found[k] = 1;
                const auto point = X.m->split_point(X.col(best_f), sp.lo_bin, sp.hi_bin);
                best.threshold[k] = point.first;
                best.split_bin[k] = point.second;
            }
        }
        return best;
    }

    // split_indices by bins: rows of `indices` with bin <= split_bin go left.
    static void split_indices_binned(const BinnedView& X, const std::vector<std::uint32_t>& indices, int feature,
                                     std::uint32_t split_bin, std::vector<std::uint32_t>& left,
                                     std::vector<std::uint32_t>& right) {
        left.clear();
        right.clear();
        for (auto idx : indices) {
            if (X.bin(idx, static_cast<std::size_t>(feature)) <= split_bin) left.push_back(idx);
            else right.push_back(idx);
        }
    }

    void release_indices(Node& node) const {
        if (params_.retain_indices) return;
        std::vector<std::uint32_t>().swap(node.indices);
        node.has_indices = false;
    }

    void finish_leaf(Node& node, const VectorView& y, const VectorView* sample_weight) const {
        make_leaf(node, y, sample_weight);
        release_indices(node);
    }

    static double total_weight(const Node& node, const VectorView* sample_weight) {
        if (!sample_weight) return static_cast<double>(node.indices.size());
        double w = 0.0;
        for (auto idx : node.indices) w += (*sample_weight)[idx];
        return w;
    }

    // `X` (exact search) or `binned` (hist search) supplies the features of
    // the n_rows tree rows.
    std::unique_ptr<Node> build_tree(const IndexedMatrixView& X, std::size_t n_rows, const VectorView& y,
                                      const VectorView* sample_weight,
                                      std::vector<std::uint32_t> root_indices,
                                      const PresortedView* presorted,
                                      const BinnedView* binned = nullptr, ThreadPool* pool = nullptr) {
        auto root = std::make_unique<Node>();
        root->n_samples = root_indices.size();
        root->indices = std::move(root_indices);
        const double root_weight = total_weight(*root, sample_weight);

        // Labels that cannot be located in classes_ (NaN) have no class
        // index for the presorted search; use the per-node search instead.
        if (presorted && params_.task == Task::Classification) {
            for (auto idx : root->indices) {
                auto it = std::lower_bound(classes_.begin(), classes_.end(), y[idx]);
                if (it == classes_.end() || !(*it == y[idx])) {
                    presorted = nullptr;
                    break;
                }
            }
        }

        // Presorted search state: per-row target/weight/class and the index
        // of the node being split that the row currently belongs to.
        std::vector<ScanRow> scan_rows;
        DepthScanner scanner(params_.criterion, params_.min_samples_leaf, classes_.size(), params_.newton);
        if (presorted) {
            scan_rows.resize(n_rows);
            for (auto idx : root->indices) {
                ScanRow& r = scan_rows[idx];
                r.y = y[idx];
                r.w = newton() ? (*hessian_)[idx] : sample_weight ? (*sample_weight)[idx] : 1.0;
                if (params_.task == Task::Classification) {
                    r.cls = static_cast<std::int32_t>(
                        std::lower_bound(classes_.begin(), classes_.end(), y[idx]) - classes_.begin());
                }
            }
        }

        // Hist search state: the scanner and each row's class index.
        std::unique_ptr<HistScanner> hist_scanner;
        std::vector<std::int32_t> row_class;
        if (binned) {
            hist_scanner = std::make_unique<HistScanner>(params_.criterion, params_.min_samples_leaf, classes_.size(),
                                                         params_.newton);
            if (params_.task == Task::Classification) {
                row_class.assign(n_rows, 0);
                for (auto idx : root->indices) {
                    auto it = std::lower_bound(classes_.begin(), classes_.end(), y[idx]);
                    if (it == classes_.end() || !(*it == y[idx])) {
                        throw std::invalid_argument("RDTModel::fit_binned: y contains a label that is not a class");
                    }
                    row_class[idx] = static_cast<std::int32_t>(it - classes_.begin());
                }
            }
        }

        std::vector<Node*> current_level = {root.get()};

        for (int depth = 0; depth < params_.max_depth; ++depth) {
            if (current_level.empty()) break;

            std::vector<Node*> nodes_to_split;
            for (Node* node : current_level) {
                if (should_split(*node, depth) && (newton() || !all_same_label(y, node->indices))) {
                    nodes_to_split.push_back(node);
                } else {
                    finish_leaf(*node, y, sample_weight);
                }
            }

            if (nodes_to_split.empty()) break;

            // Each node's share of the total training weight (both searches
            // weight a node's improvement by it).
            std::vector<double> node_share(nodes_to_split.size(), 0.0);
            for (std::size_t k = 0; k < nodes_to_split.size(); ++k) {
                node_share[k] = root_weight > 0.0 ? total_weight(*nodes_to_split[k], sample_weight) / root_weight : 0.0;
            }
            // Newton: each node's gradient / hessian totals.
            std::vector<double> node_g, node_h;
            if (newton()) {
                node_g.assign(nodes_to_split.size(), 0.0);
                node_h.assign(nodes_to_split.size(), 0.0);
                for (std::size_t k = 0; k < nodes_to_split.size(); ++k) node_sums(*nodes_to_split[k], y, node_g[k], node_h[k]);
            }

            DepthChoice choice;
            if (binned) {
                choice = find_optimal_feature_hist(*binned, y, sample_weight, nodes_to_split, node_share, node_g, node_h,
                                                   row_class, *hist_scanner, pool);
            } else if (presorted) {
                for (std::size_t k = 0; k < nodes_to_split.size(); ++k) {
                    for (auto idx : nodes_to_split[k]->indices) scan_rows[idx].node = static_cast<std::int32_t>(k);
                }
                choice = find_optimal_feature_presorted(X, nodes_to_split, node_share, node_g, node_h, *presorted,
                                                        scan_rows, scanner);
                for (Node* node : nodes_to_split) {
                    for (auto idx : node->indices) scan_rows[idx].node = -1;
                }
            } else {
                choice = find_optimal_feature_for_depth(X, y, sample_weight, nodes_to_split, node_share, node_g, node_h);
            }
            const int best_feature = choice.feature;

            // scikit-learn's rule: split unless improvement + EPSILON < min_impurity_decrease.
            // (Newton: gamma has already gated each node.)
            if (best_feature < 0 ||
                (!newton() && choice.total_improvement + std::numeric_limits<double>::epsilon() <
                                  params_.min_impurity_decrease)) {
                for (Node* node : nodes_to_split) finish_leaf(*node, y, sample_weight);
                break;
            }

            depth_features_[depth] = best_feature;
            depth_improvements_[depth] = newton() ? choice.gain_sum : choice.total_improvement;

            std::vector<Node*> next_level;
            for (std::size_t k = 0; k < nodes_to_split.size(); ++k) {
                Node* node = nodes_to_split[k];
                if (!choice.found[k]) {
                    finish_leaf(*node, y, sample_weight);
                    continue;
                }
                double threshold = choice.threshold[k];

                std::vector<std::uint32_t> left_indices, right_indices;
                if (binned) {
                    split_indices_binned(*binned, node->indices, best_feature, choice.split_bin[k], left_indices,
                                         right_indices);
                    node->split_bin = static_cast<std::int32_t>(choice.split_bin[k]);
                } else {
                    split_indices(X, node->indices, best_feature, threshold, left_indices, right_indices);
                }

                if (left_indices.empty() || right_indices.empty()) {
                    finish_leaf(*node, y, sample_weight);
                    continue;
                }

                node->is_split = true;
                node->split_feature = best_feature;
                node->split_threshold = threshold;

                node->left = std::make_unique<Node>();
                node->left->n_samples = left_indices.size();
                node->left->indices = std::move(left_indices);
                node->right = std::make_unique<Node>();
                node->right->n_samples = right_indices.size();
                node->right->indices = std::move(right_indices);
                release_indices(*node);

                next_level.push_back(node->left.get());
                next_level.push_back(node->right.get());
            }

            current_level = std::move(next_level);
        }

        for (Node* node : current_level) {
            if (!node->has_value) finish_leaf(*node, y, sample_weight);
        }

        return root;
    }
};

} // namespace rdt
