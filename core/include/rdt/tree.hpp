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
#include "rdt/rng.hpp"

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
        root_ = build_tree_hist(X, y, sample_weight, std::move(root_indices), pool);
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
    // Number of depths at which the last fit searched for a split (the
    // depths in depth_features(), plus a final one that found none).
    std::size_t n_searched_depths() const { return n_searched_depths_; }
    // The candidate columns (this tree's indices) of each searched depth,
    // replayed from TreeParams::bylevel_seed: the same draws the fit made.
    std::vector<std::vector<std::uint32_t>> level_candidates() const {
        std::vector<std::vector<std::uint32_t>> out;
        LevelSampler sampler(params_, n_features_);
        for (std::size_t d = 0; d < n_searched_depths_; ++d) out.push_back(sampler.next());
        return out;
    }
    std::size_t n_features() const { return n_features_; }

private:
    TreeParams params_;
    std::unique_ptr<Node> root_;
    std::vector<double> classes_;
    std::map<int, int> depth_features_;
    std::map<int, double> depth_improvements_;
    std::size_t n_features_ = 0;
    const VectorView* hessian_ = nullptr; // Newton: the hessians, during fit only
    std::size_t n_searched_depths_ = 0;

    // Draws each searched depth's candidate columns (colsample_bylevel); with
    // colsample_bylevel == 1 every column, without an Rng.
    class LevelSampler {
    public:
        LevelSampler(const TreeParams& p, std::size_t n_features) : n_(static_cast<std::uint32_t>(n_features)) {
            if (p.colsample_bylevel < 1.0) {
                k_ = static_cast<std::uint32_t>(std::max(1.0, std::floor(p.colsample_bylevel * static_cast<double>(n_))));
                rng_ = std::make_unique<Rng>(p.bylevel_seed);
            }
        }
        std::vector<std::uint32_t> next() {
            if (rng_ && k_ < n_) return rng_->sample_without_replacement(n_, k_);
            std::vector<std::uint32_t> all(n_);
            for (std::uint32_t f = 0; f < n_; ++f) all[f] = f;
            return all;
        }

    private:
        std::uint32_t n_;
        std::uint32_t k_ = 0;
        std::unique_ptr<Rng> rng_;
    };

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
        n_searched_depths_ = 0;
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

    static bool all_same_label(const VectorView& y, const std::uint32_t* indices, std::size_t n) {
        if (n == 0) return true;
        double first = y[indices[0]];
        for (std::size_t i = 0; i < n; ++i) {
            if (y[indices[i]] != first) return false;
        }
        return true;
    }

    static bool all_same_label(const VectorView& y, const std::vector<std::uint32_t>& indices) {
        return all_same_label(y, indices.data(), indices.size());
    }

    bool should_split(const Node& node, int depth) const {
        if (depth >= params_.max_depth) return false;
        if (static_cast<int>(node.indices.size()) < params_.min_samples_split) return false;
        return true; // caller checks all_same_label separately (needs y)
    }

    void make_leaf(Node& node, const VectorView& y, const VectorView* sample_weight) const {
        make_leaf(node, node.indices.data(), node.indices.size(), y, sample_weight);
    }

    // make_leaf from the node's rows indices[0..n) (ascending).
    void make_leaf(Node& node, const std::uint32_t* indices, std::size_t n, const VectorView& y,
                   const VectorView* sample_weight) const {
        node.has_value = true;
        if (newton()) {
            double G = 0.0, H = 0.0;
            node_sums(indices, n, y, G, H);
            node.value = newton_leaf_weight(G, H, params_.newton);
            return;
        }
        if (params_.task == Task::Classification) {
            std::vector<double> seen_labels;
            std::unordered_map<double, std::size_t> label_to_pos;
            std::vector<double> weighted_counts;
            for (std::size_t i = 0; i < n; ++i) {
                const std::uint32_t idx = indices[i];
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
            for (std::size_t i = 0; i < n; ++i) {
                const std::uint32_t idx = indices[i];
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
    };

    // Newton: a node's gradient / hessian totals, summed over its rows in
    // row order (independent of the feature being scored).
    void node_sums(const Node& node, const VectorView& g, double& G, double& H) const {
        node_sums(node.indices.data(), node.indices.size(), g, G, H);
    }

    void node_sums(const std::uint32_t* indices, std::size_t n, const VectorView& g, double& G, double& H) const {
        for (std::size_t i = 0; i < n; ++i) {
            G += g[indices[i]];
            H += (*hessian_)[indices[i]];
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
        const std::vector<double>& node_g, const std::vector<double>& node_h,
        const std::vector<std::uint32_t>& candidates) const {

        DepthChoice best;
        std::vector<char> found(nodes.size());
        std::vector<double> thresholds(nodes.size());

        for (std::uint32_t candidate : candidates) {
            const int feature_idx = static_cast<int>(candidate);
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
        const PresortedView& presorted, const std::vector<ScanRow>& scan_rows, DepthScanner& scanner,
        const std::vector<std::uint32_t>& candidates) const {

        std::vector<std::size_t> node_sizes(nodes.size());
        for (std::size_t k = 0; k < nodes.size(); ++k) node_sizes[k] = nodes[k]->indices.size();
        scanner.begin_depth(node_sizes, &node_g, &node_h);

        DepthChoice best;
        std::vector<NodeSplit> splits, best_splits;
        for (std::uint32_t candidate : candidates) {
            const int feature_idx = static_cast<int>(candidate);
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
        return total_weight(node.indices.data(), node.indices.size(), sample_weight);
    }

    static double total_weight(const std::uint32_t* indices, std::size_t n, const VectorView* sample_weight) {
        if (!sample_weight) return static_cast<double>(n);
        double w = 0.0;
        for (std::size_t i = 0; i < n; ++i) w += (*sample_weight)[indices[i]];
        return w;
    }

    // The exact search's tree (presorted or per-node); `X` supplies the
    // features of the n_rows tree rows.
    std::unique_ptr<Node> build_tree(const IndexedMatrixView& X, std::size_t n_rows, const VectorView& y,
                                      const VectorView* sample_weight,
                                      std::vector<std::uint32_t> root_indices,
                                      const PresortedView* presorted) {
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

        std::vector<Node*> current_level = {root.get()};
        // Each searched depth's candidate columns (colsample_bylevel).
        LevelSampler level_sampler(params_, n_features_);

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
            const std::vector<std::uint32_t> candidates = level_sampler.next();
            ++n_searched_depths_;

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
            if (presorted) {
                for (std::size_t k = 0; k < nodes_to_split.size(); ++k) {
                    for (auto idx : nodes_to_split[k]->indices) scan_rows[idx].node = static_cast<std::int32_t>(k);
                }
                choice = find_optimal_feature_presorted(X, nodes_to_split, node_share, node_g, node_h, *presorted,
                                                        scan_rows, scanner, candidates);
                for (Node* node : nodes_to_split) {
                    for (auto idx : node->indices) scan_rows[idx].node = -1;
                }
            } else {
                choice = find_optimal_feature_for_depth(X, y, sample_weight, nodes_to_split, node_share, node_g, node_h,
                                                        candidates);
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
                split_indices(X, node->indices, best_feature, threshold, left_indices, right_indices);

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

    // Entries [begin, end) of item `item` (a node), for per-row parallel work.
    struct RowBlock {
        std::size_t item, begin, end;
    };
    static constexpr std::size_t kHistRowBlock = 8192;

    // The entries [range(i).first, range(i).second) of items i = 0..n-1, in
    // blocks of at most kHistRowBlock entries, in item and entry order.
    template <class Range>
    static std::vector<RowBlock> row_blocks(std::size_t n, Range&& range) {
        std::vector<RowBlock> out;
        for (std::size_t i = 0; i < n; ++i) {
            const std::pair<std::size_t, std::size_t> r = range(i);
            for (std::size_t b = r.first; b < r.second; b += kHistRowBlock) {
                out.push_back({i, b, std::min(r.second, b + kHistRowBlock)});
            }
        }
        return out;
    }

    // The histogram search's tree (fit_binned): the same greedy choice per
    // depth and the same node arithmetic as build_tree, on the bins.
    //
    // The tree's rows are kept partitioned by node: each node of the current
    // depth owns a contiguous range of the row arrays (its view rows and their
    // bin-matrix rows), ascending -- the order node->indices has in
    // build_tree -- so every per-node sum runs in the same order as there. A
    // split node's range becomes its children's ranges by a stable partition.
    // Per-node work (leaf values, totals) runs in parallel over nodes, per-row
    // work (statistics, the partition) over fixed blocks of rows, and the
    // features are scored in parallel (HistScanner); none of it depends on the
    // number of threads.
    std::unique_ptr<Node> build_tree_hist(const BinnedView& X, const VectorView& y, const VectorView* sample_weight,
                                          std::vector<std::uint32_t> root_indices, ThreadPool* pool) {
        const std::size_t m = root_indices.size();
        auto root = std::make_unique<Node>();
        root->n_samples = m;
        const double root_weight = total_weight(root_indices.data(), m, sample_weight);
        const bool classification = params_.task == Task::Classification;
        const bool mse = params_.criterion == Criterion::MSE;
        const bool subtraction = params_.hist_subtraction;

        // Each row's class index (classification).
        std::vector<std::int32_t> row_class;
        if (classification) {
            row_class.assign(X.n_rows, 0);
            for (auto idx : root_indices) {
                auto it = std::lower_bound(classes_.begin(), classes_.end(), y[idx]);
                if (it == classes_.end() || !(*it == y[idx])) {
                    throw std::invalid_argument("RDTModel::fit_binned: y contains a label that is not a class");
                }
                row_class[idx] = static_cast<std::int32_t>(it - classes_.begin());
            }
        }
        HistScanner scanner(params_.criterion, params_.min_samples_leaf, classes_.size(), params_.newton);
        const std::size_t S = scanner.n_stats(), RS = scanner.row_stats();

        // The row arrays, double-buffered across depths: each entry's view
        // row and bin-matrix row; and the depth's per-entry statistics.
        std::vector<std::uint32_t> vrow[2], mrow[2];
        vrow[0] = std::move(root_indices);
        mrow[0].resize(m);
        for_blocks(pool, m, [&](std::size_t e0, std::size_t e1) {
            for (std::size_t e = e0; e < e1; ++e) mrow[0][e] = static_cast<std::uint32_t>(X.row(vrow[0][e]));
        });
        vrow[1].resize(m);
        mrow[1].resize(m);
        int cur = 0;
        std::vector<double> stat(m * RS);
        std::vector<std::int32_t> cls(classification ? m : 0);

        struct LevelNode {
            Node* node;
            std::size_t begin, end; // entries of the row arrays
            std::int32_t parent;    // index of its parent among the previous depth's scored nodes
            double parent_center;   // MSE: the parent's weighted mean
        };
        auto keep_indices = [&](Node& node, const std::uint32_t* rows, std::size_t n) {
            node.has_indices = params_.retain_indices;
            if (params_.retain_indices) node.indices.assign(rows, rows + n);
        };
        keep_indices(*root, vrow[0].data(), m);
        std::vector<LevelNode> level = {{root.get(), 0, m, -1, 0.0}};
        auto level_size = [&](std::size_t l) { return level[l].end - level[l].begin; };
        auto leaf = [&](const LevelNode& ln) {
            make_leaf(*ln.node, vrow[cur].data() + ln.begin, ln.end - ln.begin, y, sample_weight);
        };

        LevelSampler level_sampler(params_, n_features_);
        // Histogram subtraction: the next depth's candidates (drawn one depth
        // early, from the same sequence of draws) and the cached histograms of
        // the previous depth's scored nodes.
        std::vector<std::uint32_t> next_candidates;
        bool have_next = false;
        HistCache prev_cache;
        std::vector<std::vector<HistSplit>> results;
        for (int depth = 0; depth < params_.max_depth; ++depth) {
            if (level.empty()) break;
            const std::uint32_t* V = vrow[cur].data();

            // Nodes too small to split, or (impurity criteria) of a single
            // label, become leaves; the others are scored.
            std::vector<char> is_scored(level.size(), 0);
            run_balanced(pool, level.size(), level_size, [&](std::size_t l) {
                const LevelNode& ln = level[l];
                const std::size_t n = ln.end - ln.begin;
                if (static_cast<int>(n) >= params_.min_samples_split &&
                    (newton() || !all_same_label(y, V + ln.begin, n))) {
                    is_scored[l] = 1;
                } else {
                    leaf(ln);
                }
            });
            std::vector<std::size_t> scored; // level index of each scored node
            std::vector<std::int32_t> scored_index(level.size(), -1);
            for (std::size_t l = 0; l < level.size(); ++l) {
                if (!is_scored[l]) continue;
                scored_index[l] = static_cast<std::int32_t>(scored.size());
                scored.push_back(l);
            }
            if (scored.empty()) break;
            const std::vector<std::uint32_t> candidates = have_next ? std::move(next_candidates) : level_sampler.next();
            have_next = false;
            ++n_searched_depths_;
            const std::size_t K = scored.size();
            auto scored_size = [&](std::size_t k) { return level_size(scored[k]); };
            // Subtraction below the root: statistics centred at the parent's
            // mean (MSE), and computed for unscored siblings as well, which a
            // group may build to derive its scored child.
            const bool sub_depth = subtraction && depth > 0;

            // The scored nodes' per-entry statistics (as HistDepth describes
            // them) and per-node totals, both in row order, and each node's
            // share of the training weight.
            std::vector<double> center(mse ? K : 0, 0.0);
            if (mse) {
                // Targets are centred at each node's weighted mean, as the
                // exact search centres them.
                run_balanced(pool, K, scored_size, [&](std::size_t k) {
                    const LevelNode& ln = level[scored[k]];
                    double raw = 0.0, wsum = 0.0;
                    for (std::size_t e = ln.begin; e < ln.end; ++e) {
                        const double wi = sample_weight ? (*sample_weight)[V[e]] : 1.0;
                        raw += wi * y[V[e]];
                        wsum += wi;
                    }
                    center[k] = wsum > 0.0 ? raw / wsum : 0.0;
                });
            }
            // The level nodes whose entries get statistics: the scored ones,
            // and with subtraction every node (siblings of scored ones).
            std::vector<std::size_t> stat_nodes;
            for (std::size_t l = 0; l < level.size(); ++l) {
                if (scored_index[l] >= 0 || sub_depth) stat_nodes.push_back(l);
            }
            // MSE: the centre of each level node's statistics.
            std::vector<double> stat_center(mse ? level.size() : 0, 0.0);
            for (std::size_t l = 0; mse && l < level.size(); ++l) {
                stat_center[l] = sub_depth ? level[l].parent_center : scored_index[l] >= 0 ? center[scored_index[l]] : 0.0;
            }
            const std::vector<RowBlock> stat_blocks = row_blocks(stat_nodes.size(), [&](std::size_t i) {
                return std::make_pair(level[stat_nodes[i]].begin, level[stat_nodes[i]].end);
            });
            run_balanced(pool, stat_blocks.size(), [&](std::size_t b) { return stat_blocks[b].end - stat_blocks[b].begin; },
                         [&](std::size_t b) {
                const RowBlock& rb = stat_blocks[b];
                for (std::size_t e = rb.begin; e < rb.end; ++e) {
                    const std::uint32_t i = V[e];
                    if (newton()) {
                        stat[2 * e] = y[i];
                        stat[2 * e + 1] = (*hessian_)[i];
                    } else if (mse) {
                        const double wi = sample_weight ? (*sample_weight)[i] : 1.0;
                        const double d = y[i] - stat_center[stat_nodes[rb.item]];
                        stat[3 * e] = wi * d;
                        stat[3 * e + 1] = wi * d * d;
                        stat[3 * e + 2] = wi;
                    } else {
                        stat[e] = sample_weight ? (*sample_weight)[i] : 1.0;
                        cls[e] = row_class[i];
                    }
                }
            });
            std::vector<double> totals(K * S, 0.0), node_share(K, 0.0);
            std::vector<std::size_t> sizes(K);
            run_balanced(pool, K, scored_size, [&](std::size_t k) {
                const LevelNode& ln = level[scored[k]];
                double* t = totals.data() + k * S;
                if (classification) {
                    for (std::size_t e = ln.begin; e < ln.end; ++e) t[static_cast<std::size_t>(cls[e])] += stat[e];
                } else if (mse && sub_depth) {
                    // The statistics are centred at the parent's mean; the
                    // totals at the node's own, as without subtraction.
                    for (std::size_t e = ln.begin; e < ln.end; ++e) {
                        const double wi = sample_weight ? (*sample_weight)[V[e]] : 1.0;
                        const double d = y[V[e]] - center[k];
                        const double s[3] = {wi * d, wi * d * d, wi};
                        for (std::size_t si = 0; si < 3; ++si) t[si] += s[si];
                    }
                } else {
                    for (std::size_t e = ln.begin; e < ln.end; ++e) {
                        for (std::size_t s = 0; s < S; ++s) t[s] += stat[e * S + s];
                    }
                }
                sizes[k] = ln.end - ln.begin;
                node_share[k] =
                    root_weight > 0.0 ? total_weight(V + ln.begin, ln.end - ln.begin, sample_weight) / root_weight : 0.0;
            });

            // Score every candidate for every scored node.
            HistDepth hd;
            hd.mrow = mrow[cur].data();
            hd.stat = stat.data();
            hd.cls = cls.data();
            hd.n_nodes = K;
            for (std::size_t l = 0; l < level.size(); l += (depth == 0 ? 1 : 2)) {
                HistGroup g;
                g.parent = level[l].parent;
                for (std::size_t c = 0; c < (depth == 0 ? 1u : 2u); ++c) {
                    g.begin[c] = level[l + c].begin;
                    g.end[c] = level[l + c].end;
                    g.node[c] = scored_index[l + c];
                }
                if (g.node[0] >= 0 || g.node[1] >= 0) hd.groups.push_back(g);
            }
            if (mse && sub_depth) {
                hd.recenter.resize(K);
                for (std::size_t k = 0; k < K; ++k) hd.recenter[k] = center[k] - level[scored[k]].parent_center;
            }
            // The histograms to keep for the next depth: candidates that are
            // also candidates there, in candidate order, while the two depths'
            // caches together fit in the budget.
            HistCache next_cache;
            const bool keep = subtraction && depth + 1 < params_.max_depth;
            if (keep) {
                next_candidates = level_sampler.next();
                have_next = true;
                next_cache = HistCache(n_features_, K, S);
                std::vector<char> is_next(n_features_, 0);
                for (std::uint32_t f : next_candidates) is_next[f] = 1;
                const std::size_t budget = params_.max_hist_cache_bytes;
                std::size_t used = prev_cache.bytes();
                for (std::uint32_t f : candidates) {
                    const std::size_t nb = X.m->n_bins(X.col(f));
                    const std::size_t b = HistCache::bytes_for(K, nb, S);
                    if (!is_next[f] || nb < 2 || used > budget || b > budget - used) continue;
                    next_cache.add(f, nb);
                    used += b;
                }
                next_cache.allocate();
            }
            scanner.begin_depth(std::move(sizes), std::move(totals));
            scanner.score(X, hd, candidates, sub_depth ? &prev_cache : nullptr, keep ? &next_cache : nullptr, results,
                          pool);
            prev_cache = std::move(next_cache);

            // The depth's choice, reduced in candidate order.
            DepthChoice best;
            std::size_t best_i = 0;
            for (std::size_t i = 0; i < candidates.size(); ++i) {
                std::vector<HistSplit>& splits = results[i];
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
                    best.feature = static_cast<int>(candidates[i]);
                    best_i = i;
                }
            }
            // scikit-learn's rule: split unless improvement + EPSILON < min_impurity_decrease.
            // (Newton: gamma has already gated each node.)
            if (best.feature < 0 || (!newton() && best.total_improvement + std::numeric_limits<double>::epsilon() <
                                                      params_.min_impurity_decrease)) {
                run_balanced(pool, K, scored_size, [&](std::size_t k) { leaf(level[scored[k]]); });
                break;
            }
            depth_features_[depth] = best.feature;
            depth_improvements_[depth] = newton() ? best.gain_sum : best.total_improvement;

            // Split: a stable partition of each splitting node's entries
            // (bin <= split_bin first), in two passes over fixed row blocks.
            const std::size_t best_col = X.col(static_cast<std::size_t>(best.feature));
            const BinnedMatrix::ColumnReader column = X.m->column(best_col);
            std::vector<char> splits_node(K, 0);
            std::vector<std::uint32_t> split_bin(K, 0);
            std::vector<double> threshold(K, 0.0);
            for (std::size_t k = 0; k < K; ++k) {
                const HistSplit& sp = results[best_i][k];
                if (!sp.found) continue;
                splits_node[k] = 1;
                const auto point = X.m->split_point(best_col, sp.lo_bin, sp.hi_bin);
                threshold[k] = point.first;
                split_bin[k] = point.second;
            }
            const std::vector<RowBlock> part_blocks = row_blocks(K, [&](std::size_t k) {
                return splits_node[k] ? std::make_pair(level[scored[k]].begin, level[scored[k]].end)
                                      : std::make_pair(std::size_t{0}, std::size_t{0});
            });
            auto block_size = [&](std::size_t b) { return part_blocks[b].end - part_blocks[b].begin; };
            std::vector<std::size_t> block_left(part_blocks.size(), 0);
            const std::uint32_t* M = mrow[cur].data();
            run_balanced(pool, part_blocks.size(), block_size, [&](std::size_t b) {
                const RowBlock& rb = part_blocks[b];
                const std::uint32_t sb = split_bin[rb.item];
                std::size_t n_left = 0;
                for (std::size_t e = rb.begin; e < rb.end; ++e) n_left += column(M[e]) <= sb ? 1 : 0;
                block_left[b] = n_left;
            });
            std::vector<std::size_t> node_left(K, 0), left_pos(part_blocks.size()), right_pos(part_blocks.size());
            for (std::size_t b = 0; b < part_blocks.size(); ++b) node_left[part_blocks[b].item] += block_left[b];
            {
                std::vector<std::size_t> next_left(K), next_right(K);
                for (std::size_t k = 0; k < K; ++k) {
                    next_left[k] = level[scored[k]].begin;
                    next_right[k] = level[scored[k]].begin + node_left[k];
                }
                for (std::size_t b = 0; b < part_blocks.size(); ++b) {
                    const std::size_t k = part_blocks[b].item;
                    left_pos[b] = next_left[k];
                    right_pos[b] = next_right[k];
                    next_left[k] += block_left[b];
                    next_right[k] += block_size(b) - block_left[b];
                }
            }
            const int nxt = 1 - cur;
            run_balanced(pool, part_blocks.size(), block_size, [&](std::size_t b) {
                const RowBlock& rb = part_blocks[b];
                const std::uint32_t sb = split_bin[rb.item];
                std::size_t l = left_pos[b], r = right_pos[b];
                for (std::size_t e = rb.begin; e < rb.end; ++e) {
                    const std::size_t to = column(M[e]) <= sb ? l++ : r++;
                    vrow[nxt][to] = V[e];
                    mrow[nxt][to] = M[e];
                }
            });

            std::vector<LevelNode> next_level;
            std::vector<std::size_t> leaves;
            for (std::size_t k = 0; k < K; ++k) {
                const LevelNode& ln = level[scored[k]];
                Node* node = ln.node;
                if (!splits_node[k]) {
                    leaves.push_back(k);
                    continue;
                }
                node->split_bin = static_cast<std::int32_t>(split_bin[k]);
                const std::size_t n = ln.end - ln.begin, n_left = node_left[k];
                if (n_left == 0 || n_left == n) {
                    leaves.push_back(k);
                    continue;
                }
                node->is_split = true;
                node->split_feature = best.feature;
                node->split_threshold = threshold[k];
                node->left = std::make_unique<Node>();
                node->left->n_samples = n_left;
                node->right = std::make_unique<Node>();
                node->right->n_samples = n - n_left;
                const double c = mse ? center[k] : 0.0;
                next_level.push_back({node->left.get(), ln.begin, ln.begin + n_left, static_cast<std::int32_t>(k), c});
                next_level.push_back({node->right.get(), ln.begin + n_left, ln.end, static_cast<std::int32_t>(k), c});
            }
            run_balanced(pool, leaves.size(), [&](std::size_t i) { return scored_size(leaves[i]); },
                         [&](std::size_t i) { leaf(level[scored[leaves[i]]]); });
            cur = nxt;
            level = std::move(next_level);
            run_balanced(pool, level.size(), level_size, [&](std::size_t l) {
                keep_indices(*level[l].node, vrow[cur].data() + level[l].begin, level[l].end - level[l].begin);
            });
        }

        run_balanced(pool, level.size(), level_size, [&](std::size_t l) {
            if (!level[l].node->has_value) leaf(level[l]);
        });
        return root;
    }
};

} // namespace rdt
