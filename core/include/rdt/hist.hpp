#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "rdt/criteria.hpp"
#include "rdt/matrix.hpp"
#include "rdt/params.hpp"
#include "rdt/thread_pool.hpp"

// Histogram split search (tree_method='hist').
//
// Binning. Once per fit, every column is binned from the rows the model trains
// on (the training rows with positive weight): a column with at most max_bin
// distinct values gets one bin per value; otherwise bins of approximately
// equal row counts (boundaries only between distinct values, so heavy ties
// may leave fewer bins). Bin b's upper cut is the guarded midpoint
// (split_threshold) between the largest training value in bin b and the
// smallest in bin b + 1, and a value's bin is the number of cuts below it.
// Every row of X (validation and zero-weight rows included) is binned with
// these cuts and stored column-major as uint8 (every column as uint16 when
// any column has more than 256 bins); each bin also records the extent of
// its training values and of all of its values. X is read through a typed,
// strided view (float32 or float64, any order, memory-mapped or not) a block
// of columns at a time, without copying it, and never read again after
// binning; computing the cuts sorts one column's training values at a time
// per thread (8 bytes per training row; a few adjacent columns at once while
// they fit in kBinningBufferBytes).
//
// Search. Per depth, each candidate feature's histograms are built for all of
// the depth's nodes in one pass over its bin column (rows in ascending
// order), then each node's bins are swept left to right with the criterion
// arithmetic of the exact search (the same boundary rule, min_samples_leaf
// counts, MSE centring, impurity / Newton gain expressions and strict '>' tie
// order, lowest boundary first); a node's candidate boundaries lie between
// its consecutive non-empty bins. Node totals are summed once per depth in
// row order. Features may be scored in parallel; each feature's result is
// computed by one thread and the depth's choice is reduced in feature order,
// so results do not depend on the number of threads.
//
// Thresholds. A split between a node's adjacent non-empty bins lo < hi gets
// the exact search's threshold, the guarded midpoint of the largest training
// value of bin lo and the smallest of bin hi, so with one bin per value (and
// no other values in X) a hist tree is the exact tree up to tie-breaking:
// both searches score the same candidates, but round differently (here a
// bin's rows are summed first and bins are added in turn; the exact search
// adds rows one at a time in feature order), so two candidates whose scores
// are equal in exact arithmetic may be ranked differently. If a row outside the
// training rows (validation, zero weight) has a value that this threshold
// would route differently from its bin, the threshold is moved to the nearest
// bin boundary instead (split_point). Either way every row of X satisfies
// x <= threshold exactly when bin(x) <= split_bin: training scores updated
// through the bins equal predictions on X.
namespace rdt {

// Read-only 2D view with signed element strides over float or double data.
template <class T>
class StridedMatrix {
public:
    StridedMatrix(const T* data, std::size_t n_rows, std::size_t n_cols, std::ptrdiff_t row_stride,
                  std::ptrdiff_t col_stride)
        : data_(data), n_rows_(n_rows), n_cols_(n_cols), rs_(row_stride), cs_(col_stride) {}

    double operator()(std::size_t row, std::size_t col) const {
        return static_cast<double>(
            data_[static_cast<std::ptrdiff_t>(row) * rs_ + static_cast<std::ptrdiff_t>(col) * cs_]);
    }
    std::size_t n_rows() const { return n_rows_; }
    std::size_t n_cols() const { return n_cols_; }
    // Columns per cache line when a row's columns are adjacent, else 1.
    std::size_t column_block() const {
        return (cs_ == 1 || cs_ == -1) && n_cols_ > 1 ? 64 / sizeof(T) : 1;
    }

private:
    const T* data_;
    std::size_t n_rows_, n_cols_;
    std::ptrdiff_t rs_, cs_;
};

inline std::size_t column_block(const MatrixView& X) { return X.n_cols() > 1 ? 8 : 1; }
template <class T>
std::size_t column_block(const StridedMatrix<T>& X) { return X.column_block(); }

// Binning's sort buffers per task: adjacent columns are gathered together
// (cache-friendly on row-major X) only while their cut-row values fit here.
constexpr std::size_t kBinningBufferBytes = std::size_t{8} << 20;

// Runs f(0..n-1) on the pool when there is one with more than one thread.
template <class F>
void run_tasks(ThreadPool* pool, std::size_t n, F&& f) {
    if (pool && n > 1) {
        pool->parallel_for(n, f);
    } else {
        for (std::size_t i = 0; i < n; ++i) f(i);
    }
}

// Cuts of one column from its (sorted) values, and each bin's smallest and
// largest value; see the file comment. Works on `sorted` in place: the extra
// memory is O(max_bin), not O(rows).
inline void column_cuts(const std::vector<double>& sorted, int max_bin, std::vector<double>& cuts,
                        std::vector<double>& bin_lo, std::vector<double>& bin_hi) {
    cuts.clear();
    bin_lo.clear();
    bin_hi.clear();
    const std::size_t n = sorted.size();
    if (n == 0) {
        bin_lo.push_back(0.0);
        bin_hi.push_back(0.0);
        return;
    }
    const auto mb = static_cast<std::size_t>(max_bin);
    std::size_t d = 1; // distinct values, counted up to max_bin + 1
    for (std::size_t i = 1; i < n && d <= mb; ++i) d += sorted[i] != sorted[i - 1] ? 1 : 0;
    // ends[b]: one past bin b's last row in `sorted` (the number of values <= its largest value).
    std::vector<std::size_t> ends;
    if (d <= mb) {
        for (std::size_t i = 1; i < n; ++i) {
            if (sorted[i] != sorted[i - 1]) ends.push_back(i);
        }
    } else {
        for (std::uint64_t b = 1; b < static_cast<std::uint64_t>(mb); ++b) {
            // Bin b - 1 ends with the smallest value whose count of values <= it reaches
            // ceil(b * n / max_bin): sorted[target - 1].
            const std::uint64_t target = (b * n + mb - 1) / mb;
            const auto end = static_cast<std::size_t>(
                std::upper_bound(sorted.begin(), sorted.end(), sorted[static_cast<std::size_t>(target) - 1]) -
                sorted.begin());
            if (end >= n || (!ends.empty() && end <= ends.back())) continue;
            ends.push_back(end);
        }
    }
    ends.push_back(n);
    std::size_t first = 0;
    for (std::size_t b = 0; b < ends.size(); ++b) {
        bin_lo.push_back(sorted[first]);
        bin_hi.push_back(sorted[ends[b] - 1]);
        if (b + 1 < ends.size()) cuts.push_back(split_threshold(sorted[ends[b] - 1], sorted[ends[b]]));
        first = ends[b];
    }
}

class BinnedMatrix {
public:
    BinnedMatrix() = default;

    // Bins every row of X with cuts computed from X's `cut_rows`.
    template <class Matrix>
    BinnedMatrix(const Matrix& X, const std::vector<std::uint32_t>& cut_rows, int max_bin, ThreadPool* pool = nullptr) {
        if (max_bin < kMinMaxBin || max_bin > kMaxMaxBin) throw std::invalid_argument("max_bin must be in [2, 65536]");
        n_rows_ = X.n_rows();
        n_cols_ = X.n_cols();
        const std::size_t block = std::max<std::size_t>(1, column_block(X));
        const std::size_t n_blocks = (n_cols_ + block - 1) / block;

        // Pass 1: each column's cuts (and its bins' training-value extents)
        // from its sorted cut-row values. Each task (one per thread) owns one
        // sort buffer of a block of adjacent columns -- as many as fit in
        // kBinningBufferBytes, at least one (8 bytes per cut row) -- reused
        // for all of its columns.
        std::vector<std::vector<double>> col_cuts(n_cols_), col_lo(n_cols_), col_hi(n_cols_);
        const std::size_t column_bytes = std::max<std::size_t>(1, cut_rows.size() * sizeof(double));
        const std::size_t sort_block = std::max<std::size_t>(1, std::min(block, kBinningBufferBytes / column_bytes));
        const std::size_t n_sort_blocks = (n_cols_ + sort_block - 1) / sort_block;
        const std::size_t n_tasks = pool ? std::min(n_sort_blocks, pool->size()) : 1;
        run_tasks(pool, n_tasks, [&](std::size_t task) {
            std::vector<std::vector<double>> vals(std::min(sort_block, n_cols_));
            for (auto& v : vals) v.resize(cut_rows.size());
            for (std::size_t blk = task; blk < n_sort_blocks; blk += n_tasks) { // round robin: balanced
                const std::size_t c0 = blk * sort_block;
                const std::size_t nb = std::min(sort_block, n_cols_ - c0);
                for (std::size_t i = 0; i < cut_rows.size(); ++i) {
                    const std::size_t r = cut_rows[i];
                    for (std::size_t j = 0; j < nb; ++j) vals[j][i] = X(r, c0 + j);
                }
                for (std::size_t j = 0; j < nb; ++j) {
                    std::sort(vals[j].begin(), vals[j].end());
                    column_cuts(vals[j], max_bin, col_cuts[c0 + j], col_lo[c0 + j], col_hi[c0 + j]);
                }
            }
        });

        cut_offset_.resize(n_cols_ + 1, 0);
        bin_offset_.resize(n_cols_ + 1, 0);
        n_bins_.resize(n_cols_);
        for (std::size_t c = 0; c < n_cols_; ++c) {
            n_bins_[c] = static_cast<std::uint32_t>(col_cuts[c].size() + 1);
            cut_offset_[c + 1] = cut_offset_[c] + col_cuts[c].size();
            bin_offset_[c + 1] = bin_offset_[c] + n_bins_[c];
            wide_ = wide_ || n_bins_[c] > 256;
        }
        cuts_.reserve(cut_offset_.back());
        train_lo_.reserve(bin_offset_.back());
        train_hi_.reserve(bin_offset_.back());
        for (std::size_t c = 0; c < n_cols_; ++c) {
            cuts_.insert(cuts_.end(), col_cuts[c].begin(), col_cuts[c].end());
            train_lo_.insert(train_lo_.end(), col_lo[c].begin(), col_lo[c].end());
            train_hi_.insert(train_hi_.end(), col_hi[c].begin(), col_hi[c].end());
        }
        all_lo_ = train_lo_;
        all_hi_ = train_hi_;
        if (wide_) {
            data16_.resize(n_rows_ * n_cols_);
        } else {
            data8_.resize(n_rows_ * n_cols_);
        }

        // Pass 2: every row's bin, a block of columns at a time (rows in
        // order), widening each bin's all-row extent by values outside the
        // cut rows.
        run_tasks(pool, n_blocks, [&](std::size_t blk) {
            const std::size_t c0 = blk * block;
            const std::size_t nb = std::min(block, n_cols_ - c0);
            for (std::size_t r = 0; r < n_rows_; ++r) {
                for (std::size_t j = 0; j < nb; ++j) {
                    const std::size_t c = c0 + j;
                    const double x = X(r, c);
                    const double* cb = cuts_.data() + cut_offset_[c];
                    const double* ce = cuts_.data() + cut_offset_[c + 1];
                    const auto b = static_cast<std::uint32_t>(std::lower_bound(cb, ce, x) - cb);
                    double& lo = all_lo_[bin_offset_[c] + b];
                    double& hi = all_hi_[bin_offset_[c] + b];
                    if (x < lo) lo = x;
                    if (x > hi) hi = x;
                    if (wide_) {
                        data16_[c * n_rows_ + r] = static_cast<std::uint16_t>(b);
                    } else {
                        data8_[c * n_rows_ + r] = static_cast<std::uint8_t>(b);
                    }
                }
            }
        });
    }

    std::size_t n_rows() const { return n_rows_; }
    std::size_t n_cols() const { return n_cols_; }
    bool wide() const { return wide_; }
    std::uint32_t n_bins(std::size_t col) const { return n_bins_[col]; }
    std::uint32_t bin(std::size_t row, std::size_t col) const {
        return wide_ ? data16_[col * n_rows_ + row] : data8_[col * n_rows_ + row];
    }
    const std::uint8_t* column8(std::size_t col) const { return data8_.data() + col * n_rows_; }
    const std::uint16_t* column16(std::size_t col) const { return data16_.data() + col * n_rows_; }
    // x <= cut(col, b) exactly when bin(x) <= b, for b < n_bins(col) - 1.
    double cut(std::size_t col, std::uint32_t b) const { return cuts_[cut_offset_[col] + b]; }
    const double* cuts(std::size_t col) const { return cuts_.data() + cut_offset_[col]; }

    // Threshold and split bin of a split between a node's adjacent non-empty
    // bins lo < hi of column `col` (see the file comment): the guarded
    // midpoint of bin lo's largest and bin hi's smallest training value,
    // moved to a bin boundary if some binned row would otherwise route
    // differently from its bin. The result satisfies, for every binned row,
    // x <= threshold exactly when bin(x) <= split_bin, with lo <= split_bin < hi.
    std::pair<double, std::uint32_t> split_point(std::size_t col, std::uint32_t lo, std::uint32_t hi) const {
        const double* alo = all_lo_.data() + bin_offset_[col];
        const double* ahi = all_hi_.data() + bin_offset_[col];
        const double t = split_threshold(train_hi_[bin_offset_[col] + lo], train_lo_[bin_offset_[col] + hi]);
        if (t >= alo[hi]) return {ahi[hi - 1], hi - 1};
        std::uint32_t s = lo;
        for (std::uint32_t m = lo; m < hi; ++m) {
            if (alo[m] <= t && t < ahi[m]) return {ahi[m], m};
            if (ahi[m] <= t) s = m;
        }
        return {t, s};
    }

private:
    std::size_t n_rows_ = 0, n_cols_ = 0;
    bool wide_ = false;
    std::vector<std::uint8_t> data8_;
    std::vector<std::uint16_t> data16_;
    std::vector<std::uint32_t> n_bins_;
    std::vector<std::size_t> cut_offset_, bin_offset_;
    std::vector<double> cuts_;
    // Per bin: the smallest / largest training (cut-row) value, and of all rows.
    std::vector<double> train_lo_, train_hi_, all_lo_, all_hi_;
};

// A tree's rows / columns of a BinnedMatrix (nullptr maps = identity), as
// IndexedMatrixView is for X: bootstrap rows, boosting subsamples and column
// subsets are mapped, not copied.
struct BinnedView {
    const BinnedMatrix* m = nullptr;
    const std::uint32_t* rows = nullptr;
    std::size_t n_rows = 0;
    const std::uint32_t* cols = nullptr;
    std::size_t n_cols = 0;

    BinnedView() = default;
    explicit BinnedView(const BinnedMatrix& b) : m(&b), n_rows(b.n_rows()), n_cols(b.n_cols()) {}
    BinnedView(const BinnedMatrix& b, const std::uint32_t* rows_, std::size_t n_rows_, const std::uint32_t* cols_,
               std::size_t n_cols_)
        : m(&b), rows(rows_), n_rows(n_rows_), cols(cols_), n_cols(n_cols_) {}

    std::size_t row(std::size_t i) const { return rows ? rows[i] : i; }
    std::size_t col(std::size_t f) const { return cols ? cols[f] : f; }
    std::uint32_t bin(std::size_t i, std::size_t f) const { return m->bin(row(i), col(f)); }
    double cut(std::size_t f, std::uint32_t b) const { return m->cut(col(f), b); }
};

// Best split of one node on one feature by the histogram search: between the
// node's adjacent non-empty bins lo_bin < hi_bin (BinnedMatrix::split_point
// gives its threshold and split bin).
struct HistSplit {
    bool found = false;
    double improvement = 0.0;
    std::uint32_t lo_bin = 0;
    std::uint32_t hi_bin = 0;
};

class HistScanner {
public:
    HistScanner(Criterion criterion, int min_samples_leaf, std::size_t n_classes, const NewtonParams& newton)
        : criterion_(criterion), min_samples_leaf_(min_samples_leaf), n_classes_(n_classes), newton_(newton) {
        if (criterion == Criterion::MAE) throw std::invalid_argument("tree_method='hist' does not support MAE");
        n_stats_ = criterion == Criterion::MSE ? 3 : criterion == Criterion::Newton ? 2 : n_classes;
    }

    // The depth's nodes (their rows, ascending view-row ids) and per-row
    // inputs, indexed by view row: y (Newton: gradient), weights (nullptr = 1;
    // Newton: unused), hessian (Newton), cls (classification: class index).
    // Newton: node_g / node_h are the nodes' totals (summed in row order).
    void begin_depth(const BinnedView& X, const std::vector<const std::vector<std::uint32_t>*>& nodes,
                     const VectorView& y, const VectorView* w, const VectorView* hessian, const std::int32_t* cls,
                     const std::vector<double>* node_g, const std::vector<double>* node_h) {
        const std::size_t n_nodes = nodes.size();
        node_size_.assign(n_nodes, 0);
        std::vector<std::int32_t> node_of(X.n_rows, -1);
        for (std::size_t k = 0; k < n_nodes; ++k) {
            node_size_[k] = nodes[k]->size();
            for (auto idx : *nodes[k]) node_of[idx] = static_cast<std::int32_t>(k);
        }
        std::size_t n_entries = 0;
        for (std::size_t k = 0; k < n_nodes; ++k) n_entries += node_size_[k];
        row_.clear();
        node_.clear();
        stat_.clear();
        row_.reserve(n_entries);
        node_.reserve(n_entries);
        stat_.reserve(n_entries * n_stats_);
        cls_.clear();
        total_.assign(n_nodes * n_stats_, 0.0);
        total_w_.assign(n_nodes, 0.0);

        std::vector<double> center;
        if (criterion_ == Criterion::MSE) {
            // Targets are centred at each node's weighted mean (row order),
            // as the exact search centres them.
            std::vector<double> raw(n_nodes, 0.0), wsum(n_nodes, 0.0);
            for (std::size_t i = 0; i < X.n_rows; ++i) {
                if (node_of[i] < 0) continue;
                const double wi = w ? (*w)[i] : 1.0;
                raw[static_cast<std::size_t>(node_of[i])] += wi * y[i];
                wsum[static_cast<std::size_t>(node_of[i])] += wi;
            }
            center.resize(n_nodes);
            for (std::size_t k = 0; k < n_nodes; ++k) center[k] = wsum[k] > 0.0 ? raw[k] / wsum[k] : 0.0;
        }
        for (std::size_t i = 0; i < X.n_rows; ++i) {
            if (node_of[i] < 0) continue;
            const auto k = static_cast<std::size_t>(node_of[i]);
            row_.push_back(static_cast<std::uint32_t>(X.row(i)));
            node_.push_back(static_cast<std::uint32_t>(k));
            double* t = total_.data() + k * n_stats_;
            if (criterion_ == Criterion::MSE) {
                const double wi = w ? (*w)[i] : 1.0;
                const double d = y[i] - center[k];
                const double s[3] = {wi * d, wi * d * d, wi};
                for (std::size_t s_i = 0; s_i < 3; ++s_i) {
                    stat_.push_back(s[s_i]);
                    t[s_i] += s[s_i];
                }
            } else if (criterion_ == Criterion::Newton) {
                stat_.push_back(y[i]);
                stat_.push_back((*hessian)[i]);
            } else {
                const double wi = w ? (*w)[i] : 1.0;
                stat_.push_back(wi);
                cls_.push_back(cls[i]);
                t[static_cast<std::size_t>(cls[i])] += wi;
            }
        }
        if (criterion_ == Criterion::Newton) {
            for (std::size_t k = 0; k < n_nodes; ++k) {
                total_[k * 2] = (*node_g)[k];
                total_[k * 2 + 1] = (*node_h)[k];
            }
        } else if (criterion_ != Criterion::MSE) {
            for (std::size_t k = 0; k < n_nodes; ++k) {
                double tw = 0.0;
                for (std::size_t c = 0; c < n_classes_; ++c) tw += total_[k * n_stats_ + c];
                total_w_[k] = tw;
            }
        }
    }

    struct Scratch {
        std::vector<double> hist;
        std::vector<std::uint32_t> count;
        std::vector<double> left, right;
    };

    // Feature f's best split for every node of the depth into out[k].
    void score_feature(const BinnedView& X, std::size_t f, std::vector<HistSplit>& out, Scratch& sc) const {
        const std::size_t n_nodes = node_size_.size();
        out.assign(n_nodes, HistSplit{});
        const std::size_t col = X.col(f);
        const std::size_t n_bins = X.m->n_bins(col);
        if (n_bins < 2) return;
        if (X.m->wide()) {
            build(X.m->column16(col), n_bins, sc);
        } else {
            build(X.m->column8(col), n_bins, sc);
        }
        for (std::size_t k = 0; k < n_nodes; ++k) {
            const double* h = sc.hist.data() + k * n_bins * n_stats_;
            const std::uint32_t* cnt = sc.count.data() + k * n_bins;
            if (criterion_ == Criterion::MSE) {
                sweep_mse(h, cnt, n_bins, k, out[k]);
            } else if (criterion_ == Criterion::Newton) {
                sweep_newton(h, cnt, n_bins, k, out[k]);
            } else {
                sweep_classification(h, cnt, n_bins, k, out[k], sc);
            }
        }
    }

private:
    Criterion criterion_;
    int min_samples_leaf_;
    std::size_t n_classes_;
    NewtonParams newton_;
    std::size_t n_stats_ = 0;
    // The depth's rows (ascending view-row order): bin-matrix row, node, statistics.
    std::vector<std::uint32_t> row_, node_;
    std::vector<double> stat_;
    std::vector<std::int32_t> cls_;
    std::vector<std::size_t> node_size_;
    std::vector<double> total_;   // per node: n_stats_ totals (MSE: sum w*d, w*d^2, w; Newton: G, H; classes)
    std::vector<double> total_w_; // classification: per-node total weight

    bool leaf_sizes_ok(std::size_t n_left, std::size_t n_total) const {
        const std::size_t n_right = n_total - n_left;
        return !(static_cast<int>(n_left) < min_samples_leaf_ || static_cast<int>(n_right) < min_samples_leaf_);
    }

    // Histograms of all nodes in one pass over the rows. A bin's statistics
    // are set by its first row and summed from then on, so only the counts
    // need clearing.
    template <class Bin>
    void build(const Bin* column, std::size_t n_bins, Scratch& sc) const {
        const std::size_t n_nodes = node_size_.size();
        const std::size_t S = n_stats_;
        sc.count.assign(n_nodes * n_bins, 0u);
        if (sc.hist.size() < n_nodes * n_bins * S) sc.hist.resize(n_nodes * n_bins * S);
        const bool classification = criterion_ != Criterion::MSE && criterion_ != Criterion::Newton;
        for (std::size_t e = 0; e < row_.size(); ++e) {
            const std::size_t slot = static_cast<std::size_t>(node_[e]) * n_bins + column[row_[e]];
            double* h = sc.hist.data() + slot * S;
            if (classification) {
                if (sc.count[slot]++ == 0) std::fill(h, h + S, 0.0);
                h[static_cast<std::size_t>(cls_[e])] += stat_[e];
            } else if (sc.count[slot]++ == 0) {
                for (std::size_t s = 0; s < S; ++s) h[s] = stat_[e * S + s];
            } else {
                for (std::size_t s = 0; s < S; ++s) h[s] += stat_[e * S + s];
            }
        }
    }

    void sweep_mse(const double* h, const std::uint32_t* cnt, std::size_t n_bins, std::size_t k, HistSplit& out) const {
        const std::size_t n = node_size_[k];
        const double total_sum = total_[k * 3], total_sum_sq = total_[k * 3 + 1], total_weight = total_[k * 3 + 2];
        const double parent_impurity = mse_from_sums(total_sum, total_sum_sq, total_weight);
        double best_improvement = -std::numeric_limits<double>::infinity();
        double left_sum = 0.0, left_sum_sq = 0.0, left_weight = 0.0;
        std::size_t n_left = 0, prev = 0;
        for (std::size_t b = 0; b < n_bins; ++b) {
            if (cnt[b] == 0) continue;
            if (n_left > 0 && leaf_sizes_ok(n_left, n)) {
                double right_sum = total_sum - left_sum;
                double right_sum_sq = total_sum_sq - left_sum_sq;
                double right_weight = total_weight - left_weight;
                double left_imp = mse_from_sums(left_sum, left_sum_sq, left_weight);
                double right_imp = mse_from_sums(right_sum, right_sum_sq, right_weight);
                double weighted = (left_weight / total_weight) * left_imp + (right_weight / total_weight) * right_imp;
                double improvement = parent_impurity - weighted;
                if (improvement > best_improvement) {
                    best_improvement = improvement;
                    out.found = true;
                    out.improvement = improvement;
                    out.lo_bin = static_cast<std::uint32_t>(prev);
                    out.hi_bin = static_cast<std::uint32_t>(b);
                }
            }
            left_sum += h[b * 3];
            left_sum_sq += h[b * 3 + 1];
            left_weight += h[b * 3 + 2];
            n_left += cnt[b];
            prev = b;
        }
    }

    void sweep_newton(const double* h, const std::uint32_t* cnt, std::size_t n_bins, std::size_t k,
                      HistSplit& out) const {
        const std::size_t n = node_size_[k];
        const double G = total_[k * 2], H = total_[k * 2 + 1];
        const double parent_score = newton_score(G, H, newton_);
        double best_gain = -std::numeric_limits<double>::infinity();
        double left_g = 0.0, left_h = 0.0;
        std::size_t n_left = 0, prev = 0;
        for (std::size_t b = 0; b < n_bins; ++b) {
            if (cnt[b] == 0) continue;
            if (n_left > 0 && leaf_sizes_ok(n_left, n) && newton_children_valid(left_h, H - left_h, newton_)) {
                const double gain = newton_split_gain(left_g, left_h, G, H, parent_score, newton_);
                if (std::isfinite(gain) && gain > best_gain) {
                    best_gain = gain;
                    out.found = true;
                    out.improvement = gain;
                    out.lo_bin = static_cast<std::uint32_t>(prev);
                    out.hi_bin = static_cast<std::uint32_t>(b);
                }
            }
            left_g += h[b * 2];
            left_h += h[b * 2 + 1];
            n_left += cnt[b];
            prev = b;
        }
    }

    double class_impurity(const double* counts, double weight) const {
        return criterion_ == Criterion::Gini ? gini_from_counts(counts, n_classes_, weight)
                                             : entropy_from_counts(counts, n_classes_, weight);
    }

    void sweep_classification(const double* h, const std::uint32_t* cnt, std::size_t n_bins, std::size_t k,
                              HistSplit& out, Scratch& sc) const {
        const std::size_t n = node_size_[k];
        const std::size_t K = n_classes_;
        const double* total = total_.data() + k * K;
        const double total_weight = total_w_[k];
        const double parent_impurity = class_impurity(total, total_weight);
        sc.left.assign(K, 0.0);
        sc.right.resize(K);
        double best_improvement = -std::numeric_limits<double>::infinity();
        double left_weight = 0.0;
        std::size_t n_left = 0, prev = 0;
        for (std::size_t b = 0; b < n_bins; ++b) {
            if (cnt[b] == 0) continue;
            if (n_left > 0 && leaf_sizes_ok(n_left, n)) {
                for (std::size_t c = 0; c < K; ++c) sc.right[c] = total[c] - sc.left[c];
                const double right_weight = total_weight - left_weight;
                double left_imp = class_impurity(sc.left.data(), left_weight);
                double right_imp = class_impurity(sc.right.data(), right_weight);
                double weighted = (left_weight / total_weight) * left_imp + (right_weight / total_weight) * right_imp;
                double improvement = parent_impurity - weighted;
                if (improvement > best_improvement) {
                    best_improvement = improvement;
                    out.found = true;
                    out.improvement = improvement;
                    out.lo_bin = static_cast<std::uint32_t>(prev);
                    out.hi_bin = static_cast<std::uint32_t>(b);
                }
            }
            const double* hb = h + b * K;
            for (std::size_t c = 0; c < K; ++c) {
                sc.left[c] += hb[c];
                left_weight += hb[c];
            }
            n_left += cnt[b];
            prev = b;
        }
    }
};

} // namespace rdt
