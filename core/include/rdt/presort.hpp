#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "rdt/criteria.hpp"
#include "rdt/matrix.hpp"
#include "rdt/params.hpp"

// Presorted split search.
//
// The RDT constraint means every node at a depth is scored on the same
// candidate feature, so instead of sorting each (node, feature) pair's
// samples at every depth (find_best_threshold), each column is sorted once
// per fit and every depth's candidate feature is scored for *all* of that
// depth's nodes in a single pass over the column's presorted rows: each row
// is routed to its node's running aggregates, and a node's candidate
// boundaries are the points where its own subsequence of values changes.
// Cost per depth: O(n_features * n) instead of
// O(n_features * sum_nodes(n_node * log n_node)), plus a one-off
// O(n_features * n log n) presort (shared by every boosting round).
//
// Exactness. For each node this reproduces find_best_threshold's arithmetic
// operation by operation: the same candidate boundaries (between distinct
// adjacent values only), the same guarded midpoint, the same min_samples_leaf
// test, the same centring of MSE targets at the node's weighted mean, node
// totals accumulated in the same feature-sorted order, the same running
// left/right updates, classes numbered in the same first-encountered order,
// and the same strict '>' tie-breaking, with the expressions written
// identically so the compiler contracts them (FMA) the same way. Samples
// inside a run of tied feature values are visited in ascending row id here
// (a stable presort) and find_best_threshold sorts by (value, row id) for the
// same reason, so the two searches agree bit for bit on every input. The
// node-size weighting of the per-node improvements and the zero-weight row
// exclusion happen in RDTModel, outside the scanner, identically for both.
namespace rdt {

// A presorted column entry: the row id in the low 31 bits; the high bit is
// set when the row's value differs from the previous entry's (operator==,
// so -0.0 == 0.0 and NaN never equals anything).
constexpr std::uint32_t kPresortRowMask = 0x7FFFFFFFu;
constexpr std::uint32_t kPresortNewValue = 0x80000000u;
// Row ids must fit in the low 31 bits.
constexpr std::size_t kPresortMaxRows = static_cast<std::size_t>(kPresortRowMask);

// Order-preserving map from double to uint64 (-0.0 and 0.0 map together;
// NaN sorts after +inf). Sorting by (key, row) is a strict total order even
// with NaN in X, unlike operator< on the doubles themselves.
inline std::uint64_t presort_key(double x) {
    if (std::isnan(x)) return ~std::uint64_t{0};
    if (x == 0.0) x = 0.0;
    std::uint64_t u;
    std::memcpy(&u, &x, sizeof u);
    return (u >> 63) ? ~u : (u | 0x8000000000000000ULL);
}

// Read-only per-feature presorted orders as handed to RDTModel::fit:
// columns[f] holds the n_entries entries (one per row the tree is fitted on)
// for the tree's feature f.
struct PresortedView {
    std::vector<const std::uint32_t*> columns;
    std::size_t n_entries = 0;
};

// Sorts `rows` by each column cols[j] of X (value ascending, NaN last, row
// id ascending) and writes the encoded entries to out[j][0..rows.size()).
// For wide X, columns are gathered a block at a time so a row-major X is read
// a cache line at a time rather than one strided double at a time. The
// transient buffer is 16 bytes per (row, column of a block); the block is at
// most X.n_cols() / 4 columns (so the buffer never exceeds the 4 bytes per
// (row, column) of the result) and at most 8.
template <class Matrix>
void presort_columns(const Matrix& X, const std::vector<std::uint32_t>& rows,
                     const std::vector<std::size_t>& cols, const std::vector<std::uint32_t*>& out) {
    struct KeyRow {
        std::uint64_t key;
        std::uint32_t row;
    };
    constexpr std::uint64_t kNaNKey = ~std::uint64_t{0};
    const std::size_t block = std::min<std::size_t>(8, std::max<std::size_t>(1, X.n_cols() / 4));
    const std::size_t n = rows.size();
    std::vector<std::vector<KeyRow>> buf(std::min(block, cols.size()));
    for (auto& b : buf) b.resize(n);
    for (std::size_t c0 = 0; c0 < cols.size(); c0 += block) {
        const std::size_t nb = std::min(block, cols.size() - c0);
        for (std::size_t i = 0; i < n; ++i) {
            const std::uint32_t r = rows[i];
            for (std::size_t j = 0; j < nb; ++j) buf[j][i] = {presort_key(X(r, cols[c0 + j])), r};
        }
        for (std::size_t j = 0; j < nb; ++j) {
            auto& v = buf[j];
            std::sort(v.begin(), v.end(), [](const KeyRow& a, const KeyRow& b) {
                return a.key < b.key || (a.key == b.key && a.row < b.row);
            });
            std::uint32_t* o = out[c0 + j];
            for (std::size_t k = 0; k < n; ++k) {
                const bool new_value = k == 0 || v[k].key != v[k - 1].key || v[k].key == kNaNKey;
                o[k] = v[k].row | (new_value ? kPresortNewValue : 0u);
            }
        }
    }
}

// Owning presorted orders: for every column of X, the given rows sorted by
// (value ascending, NaN last, row id ascending), encoded as above.
// 4 bytes per (row, column).
class SortedColumns {
public:
    SortedColumns() = default;

    template <class Matrix>
    SortedColumns(const Matrix& X, const std::vector<std::uint32_t>& rows) {
        build(X, rows);
    }

    // `src` restricted to the rows with new_id[row] >= 0, renumbered to
    // new_id[row] (which must preserve the rows' relative order, so ties
    // stay ordered by row). Used by boosting to derive each round's
    // presorted columns from the per-fit ones in O(n) per column.
    SortedColumns(const PresortedView& src, const std::int32_t* new_id, std::size_t n_kept) {
        n_entries_ = n_kept;
        n_cols_ = src.columns.size();
        data_.assign(n_entries_ * n_cols_ + 1, 0u);
        for (std::size_t c = 0; c < n_cols_; ++c) {
            const std::uint32_t* in = src.columns[c];
            std::uint32_t* out = data_.data() + c * n_entries_;
            std::size_t k = 0;
            std::uint32_t pending = 0; // a new value started since the last kept entry
            for (std::size_t i = 0; i < src.n_entries; ++i) {
                // Branch-free: every entry is written at out[k], and k only
                // advances past kept ones (data_ has one slot of slack).
                const std::uint32_t e = in[i];
                pending |= e & kPresortNewValue;
                const std::int32_t id = new_id[e & kPresortRowMask];
                const std::uint32_t keep = id >= 0 ? 1u : 0u;
                out[k] = static_cast<std::uint32_t>(id) | pending;
                k += keep;
                pending &= keep - 1u;
            }
        }
    }

    std::size_t n_entries() const { return n_entries_; }
    std::size_t n_cols() const { return n_cols_; }
    const std::uint32_t* column(std::size_t col) const { return data_.data() + col * n_entries_; }

    // Columns for the given subset of this object's columns, in that order
    // (nullptr = all columns).
    PresortedView view(const std::vector<std::uint32_t>* cols = nullptr) const {
        PresortedView v;
        v.n_entries = n_entries_;
        if (cols) {
            v.columns.reserve(cols->size());
            for (auto c : *cols) v.columns.push_back(column(c));
        } else {
            v.columns.reserve(n_cols_);
            for (std::size_t c = 0; c < n_cols_; ++c) v.columns.push_back(column(c));
        }
        return v;
    }

private:
    std::vector<std::uint32_t> data_;
    std::size_t n_entries_ = 0;
    std::size_t n_cols_ = 0;

    template <class Matrix>
    void build(const Matrix& X, const std::vector<std::uint32_t>& rows) {
        n_entries_ = rows.size();
        n_cols_ = X.n_cols();
        data_.assign(n_entries_ * n_cols_, 0u);
        std::vector<std::size_t> cols(n_cols_);
        std::vector<std::uint32_t*> out(n_cols_);
        for (std::size_t c = 0; c < n_cols_; ++c) {
            cols[c] = c;
            out[c] = data_.data() + c * n_entries_;
        }
        presort_columns(X, rows, cols, out);
    }
};

// Presorted orders of a fixed set of rows, computed per column on first
// use. Gradient boosting with column subsampling then sorts (and keeps) only
// the columns its rounds actually draw. ensure() is not thread-safe; views
// of ensured columns may be read concurrently.
class LazySortedColumns {
public:
    LazySortedColumns(const MatrixView& X, std::vector<std::uint32_t> rows)
        : X_(X), rows_(std::move(rows)), cols_(X.n_cols()) {}

    void ensure(const std::vector<std::uint32_t>& cols) {
        std::vector<std::size_t> todo;
        std::vector<std::uint32_t*> out;
        for (auto c : cols) {
            if (!cols_[c].empty() || rows_.empty()) continue;
            cols_[c].resize(rows_.size());
            todo.push_back(c);
            out.push_back(cols_[c].data());
        }
        if (!todo.empty()) presort_columns(X_, rows_, todo, out);
    }

    // Columns `cols` (all ensured) in that order.
    PresortedView view(const std::vector<std::uint32_t>& cols) const {
        PresortedView v;
        v.n_entries = rows_.size();
        v.columns.reserve(cols.size());
        for (auto c : cols) v.columns.push_back(cols_[c].data());
        return v;
    }

private:
    MatrixView X_;
    std::vector<std::uint32_t> rows_;
    std::vector<std::vector<std::uint32_t>> cols_;
};

// What the presorted search needs per row, indexed by row id.
struct ScanRow {
    double y = 0.0;
    double w = 1.0;
    std::int32_t node = -1; // index into the depth's nodes being split; -1 = skip this row
    std::int32_t cls = 0;   // index into RDTModel::classes() (classification only)
};

// Best split of one node on one feature, as found by DepthScanner. The
// threshold is split_threshold() of rows lo_row's and hi_row's values (their
// guarded midpoint).
struct NodeSplit {
    bool found = false;
    double improvement = 0.0;
    std::uint32_t lo_row = 0;
    std::uint32_t hi_row = 0;
};

// Scores one feature for every node of a depth. One pass over the
// feature's presorted rows gathers each node's rows into a contiguous
// segment (in sorted order); each segment is then swept twice -- totals,
// then candidate boundaries -- with the running aggregates in registers.
// Gini, Entropy and MSE only (MAE keeps the per-node path).
class DepthScanner {
public:
    DepthScanner(Criterion criterion, int min_samples_leaf, std::size_t n_classes)
        : criterion_(criterion), min_samples_leaf_(min_samples_leaf), n_classes_(n_classes) {}

    static bool supports(Criterion c) { return c != Criterion::MAE; }

    // node_sizes[k] = number of rows routed to node k (ScanRow::node == k).
    void begin_depth(const std::vector<std::size_t>& node_sizes) {
        node_begin_.assign(node_sizes.size() + 1, 0);
        for (std::size_t k = 0; k < node_sizes.size(); ++k) node_begin_[k + 1] = node_begin_[k] + node_sizes[k];
        items_.resize(node_begin_.back());
        cursor_.resize(node_sizes.size());
        if (criterion_ == Criterion::MSE) {
            raw_sum_.resize(node_sizes.size());
            weight_sum_.resize(node_sizes.size());
        } else {
            local_.assign(n_classes_, -1);
            total_.assign(n_classes_, 0.0);
            left_.assign(n_classes_, 0.0);
            right_.assign(n_classes_, 0.0);
        }
    }

    // `sorted`: the feature's presorted entries (presort.hpp) covering at
    // least every row with rows[row].node >= 0.
    void score_feature(const std::uint32_t* sorted, std::size_t n_entries, const ScanRow* rows,
                       std::vector<NodeSplit>& out) {
        const std::size_t n_nodes = cursor_.size();
        out.assign(n_nodes, NodeSplit{});
        gather(sorted, n_entries, rows);
        for (std::size_t k = 0; k < n_nodes; ++k) {
            const Item* b = items_.data() + node_begin_[k];
            const Item* e = items_.data() + node_begin_[k + 1];
            if (criterion_ == Criterion::MSE) {
                score_mse(b, e, raw_sum_[k], weight_sum_[k], out[k]);
            } else {
                score_classification(b, e, out[k]);
            }
        }
    }

private:
    struct Item {
        union {
            double y;          // regression (MSE)
            std::int64_t cls;  // classification: index into the classes
        };
        double w;
        std::uint32_t row;
        std::uint32_t run; // index of the row's run of equal values in the sorted column
    };

    Criterion criterion_;
    int min_samples_leaf_;
    std::size_t n_classes_;
    std::vector<std::size_t> node_begin_; // node k's items are [node_begin_[k], node_begin_[k + 1])
    std::vector<std::size_t> cursor_;
    std::vector<Item> items_;
    std::vector<std::int32_t> local_;     // class -> first-encounter position within the node
    // MSE: each node's sum(w*y) and sum(w), accumulated while gathering, in the
    // node's sorted order (the order score_mse would otherwise re-read them in).
    std::vector<double> raw_sum_, weight_sum_;
    std::vector<double> total_, left_, right_; // per first-encounter position

    bool leaf_sizes_ok(std::size_t n_left, std::size_t n_total) const {
        const std::size_t n_right = n_total - n_left;
        return !(static_cast<int>(n_left) < min_samples_leaf_ ||
                 static_cast<int>(n_right) < min_samples_leaf_);
    }

    void gather(const std::uint32_t* sorted, std::size_t n_entries, const ScanRow* rows) {
        const bool mse = criterion_ == Criterion::MSE;
        for (std::size_t k = 0; k < cursor_.size(); ++k) cursor_[k] = node_begin_[k];
        if (mse) {
            std::fill(raw_sum_.begin(), raw_sum_.end(), 0.0);
            std::fill(weight_sum_.begin(), weight_sum_.end(), 0.0);
        }
        std::uint32_t run = 0;
        for (std::size_t i = 0; i < n_entries; ++i) {
            const std::uint32_t e = sorted[i];
            run += e >> 31;
            const std::uint32_t row = e & kPresortRowMask;
            const ScanRow& r = rows[row];
            if (r.node < 0) continue;
            const std::size_t k = static_cast<std::size_t>(r.node);
            Item& item = items_[cursor_[k]++];
            if (mse) {
                item.y = r.y;
                raw_sum_[k] += r.w * r.y;
                weight_sum_[k] += r.w;
            } else {
                item.cls = r.cls;
            }
            item.w = r.w;
            item.row = row;
            item.run = run;
        }
    }

    // One node's items in sorted order. A boundary between the node's
    // previous row and the current one is evaluated (with the previous rows
    // on the left) when a new run of values started in between.
    //
    // Like find_best_threshold, the running sums are kept on targets centred
    // at the node's weighted mean, with every centred value rounded to a
    // double before use, exactly as find_best_threshold stores them -- so both
    // searches perform the same operations. The mean's sums (`raw_sum`,
    // `total_weight`) were accumulated by gather() in this same sorted order,
    // which saves a pass over the node's items per feature.
    void score_mse(const Item* begin, const Item* end, double raw_sum, double total_weight, NodeSplit& out) const {
        const std::size_t n = static_cast<std::size_t>(end - begin);
        const double center = total_weight > 0.0 ? raw_sum / total_weight : 0.0;

        double total_sum = 0.0, total_sum_sq = 0.0;
        for (const Item* p = begin; p != end; ++p) {
            const double d = p->y - center;
            total_sum += p->w * d;
            total_sum_sq += p->w * d * d;
        }
        const double parent_impurity = mse_from_sums(total_sum, total_sum_sq, total_weight);

        double best_improvement = -std::numeric_limits<double>::infinity();
        double left_sum = 0.0, left_sum_sq = 0.0, left_weight = 0.0;
        std::size_t n_left = 0;
        for (const Item* p = begin; p != end; ++p) {
            if (n_left > 0 && p[-1].run != p->run && leaf_sizes_ok(n_left, n)) {
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
                    out.found = true;
                    out.improvement = improvement;
                    out.lo_row = p[-1].row;
                    out.hi_row = p->row;
                }
            }
            const double d = p->y - center;
            left_sum += p->w * d;
            left_sum_sq += p->w * d * d;
            left_weight += p->w;
            ++n_left;
        }
    }

    double class_impurity(const double* counts, std::size_t n, double weight) const {
        return (criterion_ == Criterion::Gini) ? gini_from_counts(counts, n, weight)
                                               : entropy_from_counts(counts, n, weight);
    }

    void score_classification(const Item* begin, const Item* end, NodeSplit& out) {
        const std::size_t n = static_cast<std::size_t>(end - begin);
        // Class totals in sorted order, classes numbered by first encounter
        // (find_best_threshold's order, which fixes the summation order
        // inside the impurity functions).
        std::size_t n_seen = 0;
        for (const Item* p = begin; p != end; ++p) {
            std::int32_t& pos = local_[static_cast<std::size_t>(p->cls)];
            if (pos < 0) {
                pos = static_cast<std::int32_t>(n_seen++);
                total_[static_cast<std::size_t>(pos)] = 0.0;
            }
            total_[static_cast<std::size_t>(pos)] += p->w;
        }
        double total_weight = 0.0;
        for (std::size_t c = 0; c < n_seen; ++c) total_weight += total_[c];
        const double parent_impurity = class_impurity(total_.data(), n_seen, total_weight);
        for (std::size_t c = 0; c < n_seen; ++c) {
            left_[c] = 0.0;
            right_[c] = total_[c];
        }

        double best_improvement = -std::numeric_limits<double>::infinity();
        double left_weight = 0.0, right_weight = total_weight;
        std::size_t n_left = 0;
        for (const Item* p = begin; p != end; ++p) {
            if (n_left > 0 && p[-1].run != p->run && leaf_sizes_ok(n_left, n)) {
                double left_imp = class_impurity(left_.data(), n_seen, left_weight);
                double right_imp = class_impurity(right_.data(), n_seen, right_weight);

                double weighted = (left_weight / total_weight) * left_imp +
                                   (right_weight / total_weight) * right_imp;
                double improvement = parent_impurity - weighted;

                if (improvement > best_improvement) {
                    best_improvement = improvement;
                    out.found = true;
                    out.improvement = improvement;
                    out.lo_row = p[-1].row;
                    out.hi_row = p->row;
                }
            }
            const std::size_t pos = static_cast<std::size_t>(local_[static_cast<std::size_t>(p->cls)]);
            left_[pos] += p->w;
            right_[pos] -= p->w;
            left_weight += p->w;
            right_weight -= p->w;
            ++n_left;
        }
        for (const Item* p = begin; p != end; ++p) local_[static_cast<std::size_t>(p->cls)] = -1;
    }
};

} // namespace rdt
