#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
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
// these cuts and stored as uint8 (uint16 when a column has more than 256
// bins) in row blocks: the columns are taken kBinBlock at a time, and a block
// holds each row's bins of its columns side by side. Each bin also records
// the extent of its training values and of all of its values. X is read
// through a typed, strided view (float32 or float64, any order, memory-mapped
// or not) a block of columns at a time, without copying it, and never read
// again after binning.
//
// Search. The tree keeps its rows partitioned by node (each node's rows
// contiguous and ascending, see RDTModel::build_tree_hist). Per depth, each
// candidate feature's histogram is built for one node at a time, over that
// node's rows in ascending order, for all of a bin block's candidate features
// in the same pass, so a node's histograms stay in cache and its rows' bins
// are read one cache line per row and block. A node's bins are then swept left
// to right with the criterion arithmetic of the exact search (the same
// boundary rule, min_samples_leaf counts, MSE centring, impurity / Newton gain
// expressions and strict '>' tie order, lowest boundary first); its candidate
// boundaries lie between its consecutive non-empty bins. Node totals are
// summed in row order. Features are scored in parallel; each (feature, node)
// result is computed by one thread and the depth's choice is reduced in
// feature order, so results do not depend on the number of threads.
//
// Subtraction (TreeParams::hist_subtraction). Below the root, a feature whose
// histograms of the previous depth were kept (HistCache) is built only for
// the smaller child of each split, and the sibling's histogram is the
// parent's minus that child's. Derived sums are rounded differently from
// directly built ones, so split scores may differ in their last bits from the
// direct build (exactly equal for unit-weight class counts).
//
// Thresholds. A split between a node's adjacent non-empty bins lo < hi gets
// the exact search's threshold, the guarded midpoint of the largest training
// value of bin lo and the smallest of bin hi, so with one bin per value (and
// no other values in X) a hist tree is the exact tree. If a row outside the
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

// Runs f(0..n-1) on the pool when there is one with more than one thread.
template <class F>
void run_tasks(ThreadPool* pool, std::size_t n, F&& f) {
    if (pool && pool->size() > 1 && n > 1) {
        pool->parallel_for(n, f);
    } else {
        for (std::size_t i = 0; i < n; ++i) f(i);
    }
}

// Runs f(k) for every k in [0, n): contiguous ranges of k, of about equal
// total weight(k) (e.g. a node's row count), are run as one task each on the
// pool. f(k) must not depend on which other k share its task.
template <class W, class F>
void run_balanced(ThreadPool* pool, std::size_t n, W&& weight, F&& f) {
    if (!pool || pool->size() < 2 || n < 2) {
        for (std::size_t k = 0; k < n; ++k) f(k);
        return;
    }
    double total = 0.0;
    for (std::size_t k = 0; k < n; ++k) total += static_cast<double>(weight(k)) + 1.0;
    const double target = std::max(total / static_cast<double>(4 * pool->size()), 4096.0);
    std::vector<std::size_t> bounds = {0};
    double acc = 0.0;
    for (std::size_t k = 0; k < n; ++k) {
        acc += static_cast<double>(weight(k)) + 1.0;
        if (acc >= target && k + 1 < n) {
            bounds.push_back(k + 1);
            acc = 0.0;
        }
    }
    bounds.push_back(n);
    run_tasks(pool, bounds.size() - 1, [&](std::size_t t) {
        for (std::size_t k = bounds[t]; k < bounds[t + 1]; ++k) f(k);
    });
}

// Prefetch the cache line at p for reading (a no-op where unsupported).
inline void prefetch_read(const void* p) {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(p, 0, 3);
#else
    (void)p;
#endif
}

// Cuts of one column from its (sorted) values, and each bin's smallest and
// largest value; see the file comment.
inline void column_cuts(const std::vector<double>& sorted, int max_bin, std::vector<double>& cuts,
                        std::vector<double>& bin_lo, std::vector<double>& bin_hi) {
    cuts.clear();
    bin_lo.clear();
    bin_hi.clear();
    std::vector<double> vals;
    std::vector<std::size_t> cum; // number of values <= vals[j]
    for (std::size_t i = 0; i < sorted.size(); ++i) {
        if (i == 0 || sorted[i] != sorted[i - 1]) {
            vals.push_back(sorted[i]);
            cum.push_back(0);
        }
        cum.back() = i + 1;
    }
    const std::size_t d = vals.size();
    if (d == 0) {
        bin_lo.push_back(0.0);
        bin_hi.push_back(0.0);
        return;
    }
    std::vector<std::size_t> last_of_bin; // index into vals of each bin's largest value
    if (d <= static_cast<std::size_t>(max_bin)) {
        for (std::size_t j = 0; j < d; ++j) last_of_bin.push_back(j);
    } else {
        const std::uint64_t n = sorted.size();
        for (std::uint64_t b = 1; b < static_cast<std::uint64_t>(max_bin); ++b) {
            const std::uint64_t target =
                (b * n + static_cast<std::uint64_t>(max_bin) - 1) / static_cast<std::uint64_t>(max_bin);
            const std::size_t j =
                static_cast<std::size_t>(std::lower_bound(cum.begin(), cum.end(), target) - cum.begin());
            if (j + 1 >= d || (!last_of_bin.empty() && j <= last_of_bin.back())) continue;
            last_of_bin.push_back(j);
        }
        last_of_bin.push_back(d - 1);
    }
    std::size_t first = 0;
    for (std::size_t b = 0; b < last_of_bin.size(); ++b) {
        const std::size_t last = last_of_bin[b];
        bin_lo.push_back(vals[first]);
        bin_hi.push_back(vals[last]);
        if (b + 1 < last_of_bin.size()) cuts.push_back(split_threshold(vals[last], vals[last + 1]));
        first = last + 1;
    }
}

// Columns per row block of a BinnedMatrix (see the file comment): a node's
// pass over its rows reads one row's bins of a block (16 or 32 bytes) at a
// time, and builds that many features' histograms at once.
constexpr std::size_t kBinBlock = 16;

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
        // from its sorted cut-row values.
        std::vector<std::vector<double>> col_cuts(n_cols_), col_lo(n_cols_), col_hi(n_cols_);
        run_tasks(pool, n_blocks, [&](std::size_t blk) {
            const std::size_t c0 = blk * block;
            const std::size_t nb = std::min(block, n_cols_ - c0);
            std::vector<std::vector<double>> vals(nb, std::vector<double>(cut_rows.size()));
            for (std::size_t i = 0; i < cut_rows.size(); ++i) {
                const std::size_t r = cut_rows[i];
                for (std::size_t j = 0; j < nb; ++j) vals[j][i] = X(r, c0 + j);
            }
            for (std::size_t j = 0; j < nb; ++j) {
                std::sort(vals[j].begin(), vals[j].end());
                column_cuts(vals[j], max_bin, col_cuts[c0 + j], col_lo[c0 + j], col_hi[c0 + j]);
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

        // Pass 2: every row's bin, a row block of columns at a time (rows in
        // order), widening each bin's all-row extent by values outside the
        // cut rows.
        run_tasks(pool, n_bin_blocks(), [&](std::size_t blk) {
            const std::size_t c0 = blk * kBinBlock;
            const std::size_t w = block_width(blk);
            const std::size_t off = block_offset(blk);
            for (std::size_t r = 0; r < n_rows_; ++r) {
                for (std::size_t j = 0; j < w; ++j) {
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
                        data16_[off + r * w + j] = static_cast<std::uint16_t>(b);
                    } else {
                        data8_[off + r * w + j] = static_cast<std::uint8_t>(b);
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
        const std::size_t blk = col / kBinBlock;
        const std::size_t i = block_offset(blk) + row * block_width(blk) + (col - blk * kBinBlock);
        return wide_ ? data16_[i] : data8_[i];
    }
    // Row blocks: block `blk` holds columns [blk * kBinBlock, + block_width(blk)),
    // row r's bins at block8/16(blk) + r * block_width(blk).
    std::size_t n_bin_blocks() const { return (n_cols_ + kBinBlock - 1) / kBinBlock; }
    std::size_t block_width(std::size_t blk) const { return std::min(kBinBlock, n_cols_ - blk * kBinBlock); }
    const std::uint8_t* block8(std::size_t blk) const { return data8_.data() + block_offset(blk); }
    const std::uint16_t* block16(std::size_t blk) const { return data16_.data() + block_offset(blk); }
    // Reads one column's bins: column(col)(row) == bin(row, col).
    struct ColumnReader {
        const std::uint8_t* p8;
        const std::uint16_t* p16;
        std::size_t width;
        std::uint32_t operator()(std::size_t row) const {
            return p16 ? p16[row * width] : p8[row * width];
        }
    };
    ColumnReader column(std::size_t col) const {
        const std::size_t blk = col / kBinBlock, j = col - blk * kBinBlock;
        return wide_ ? ColumnReader{nullptr, block16(blk) + j, block_width(blk)}
                     : ColumnReader{block8(blk) + j, nullptr, block_width(blk)};
    }
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

    std::size_t block_offset(std::size_t blk) const { return blk * kBinBlock * n_rows_; }
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

// The children of one node split at the previous depth (at depth 0, the root
// alone). Child c holds entries [begin[c], end[c]) of the depth's row arrays
// (HistDepth); node[c] is its index among the depth's scored nodes, or -1 when
// it is not scored (it became a leaf).
struct HistGroup {
    std::int32_t parent = -1; // index among the previous depth's scored nodes (-1: the root)
    std::size_t begin[2] = {0, 0};
    std::size_t end[2] = {0, 0};
    std::int32_t node[2] = {-1, -1};
};

// One depth's rows as the tree prepares them (RDTModel::build_tree_hist):
// entry e is bin-matrix row mrow[e] with statistics stat[e * row_stats() ..]
// (Newton: g, h; MSE: w*d, w*d^2, w with d = y - c; classification: the
// weight, and the class in cls[e]). MSE centres c at the node's own weighted
// mean, or -- with histogram subtraction, below the root -- at its parent's,
// and recenter[k] (when not empty) is node k's mean minus its parent's.
struct HistDepth {
    const std::uint32_t* mrow = nullptr;
    const double* stat = nullptr;
    const std::int32_t* cls = nullptr;
    std::vector<HistGroup> groups;
    std::size_t n_nodes = 0;
    std::vector<double> recenter;
};

// Histogram subtraction (TreeParams::hist_subtraction): one depth's scored
// nodes' histograms of some features, kept so the next depth can derive a
// child's histogram as parent - sibling. The features are added first, then
// allocate() makes one buffer for all of them (so freeing it returns the
// memory); each feature's histograms are filled by the task that scores it.
class HistCache {
public:
    HistCache() = default;
    HistCache(std::size_t n_features, std::size_t n_nodes, std::size_t n_stats)
        : slot_(n_features, -1), n_nodes_(n_nodes), n_stats_(n_stats) {}

    // Bytes the histograms of n_nodes nodes of a feature with n_bins bins take.
    static std::size_t bytes_for(std::size_t n_nodes, std::size_t n_bins, std::size_t n_stats) {
        return n_nodes * n_bins * (n_stats * sizeof(double) + sizeof(std::uint32_t));
    }
    void add(std::size_t feature, std::size_t n_bins) {
        slot_[feature] = static_cast<std::int32_t>(entries_.size());
        // Each feature's histograms start on their own cache line: the
        // scoring threads write different features concurrently.
        const std::size_t bytes = (bytes_for(n_nodes_, n_bins, n_stats_) + kAlign - 1) / kAlign * kAlign;
        entries_.push_back(Entry{n_bins, bytes_, nullptr, nullptr});
        bytes_ += bytes;
    }
    void allocate() {
        if (bytes_ == 0) return;
        buffer_.reset(static_cast<unsigned char*>(::operator new(bytes_, std::align_val_t{kAlign})));
        for (Entry& e : entries_) {
            e.hist = reinterpret_cast<double*>(buffer_.get() + e.offset);
            e.count = reinterpret_cast<std::uint32_t*>(buffer_.get() + e.offset +
                                                       n_nodes_ * e.n_bins * n_stats_ * sizeof(double));
        }
    }
    // The feature's slot, or -1 when it is not kept.
    std::int32_t slot(std::size_t feature) const { return slot_.empty() ? -1 : slot_[feature]; }
    std::size_t bytes() const { return bytes_; }
    double* hist(std::int32_t slot, std::size_t node) const {
        const Entry& e = entries_[static_cast<std::size_t>(slot)];
        return e.hist + node * e.n_bins * n_stats_;
    }
    std::uint32_t* count(std::int32_t slot, std::size_t node) const {
        const Entry& e = entries_[static_cast<std::size_t>(slot)];
        return e.count + node * e.n_bins;
    }

private:
    static constexpr std::size_t kAlign = 128; // a cache line (two on x86)
    struct AlignedFree {
        void operator()(unsigned char* p) const { ::operator delete(p, std::align_val_t{kAlign}); }
    };
    struct Entry {
        std::size_t n_bins;
        std::size_t offset;
        double* hist;
        std::uint32_t* count;
    };
    std::vector<std::int32_t> slot_; // per feature of the tree's view
    std::vector<Entry> entries_;
    std::unique_ptr<unsigned char, AlignedFree> buffer_;
    std::size_t n_nodes_ = 0, n_stats_ = 0, bytes_ = 0;
};

class HistScanner {
public:
    HistScanner(Criterion criterion, int min_samples_leaf, std::size_t n_classes, const NewtonParams& newton)
        : criterion_(criterion), min_samples_leaf_(min_samples_leaf), n_classes_(n_classes), newton_(newton) {
        if (criterion == Criterion::MAE) throw std::invalid_argument("tree_method='hist' does not support MAE");
        n_stats_ = criterion == Criterion::MSE ? 3 : criterion == Criterion::Newton ? 2 : n_classes;
    }

    bool classification() const { return criterion_ != Criterion::MSE && criterion_ != Criterion::Newton; }
    // Statistics per bin, and per row of HistDepth::stat.
    std::size_t n_stats() const { return n_stats_; }
    std::size_t row_stats() const { return classification() ? 1 : n_stats_; }

    // The depth's scored nodes: their row counts and totals (n_stats() per
    // node, summed in row order: MSE sum w*d, w*d^2, w; Newton G, H; class
    // weights).
    void begin_depth(std::vector<std::size_t> node_size, std::vector<double> totals) {
        node_size_ = std::move(node_size);
        total_ = std::move(totals);
        total_w_.assign(node_size_.size(), 0.0);
        if (classification()) {
            for (std::size_t k = 0; k < node_size_.size(); ++k) {
                double tw = 0.0;
                for (std::size_t c = 0; c < n_classes_; ++c) tw += total_[k * n_stats_ + c];
                total_w_[k] = tw;
            }
        }
    }

    // Every candidate's best split for every scored node of the depth:
    // results[i][k] for candidates[i] (a column of X's view) and node k.
    // `prev` (may be nullptr): the previous depth's cache -- a feature it
    // keeps gets each group's smaller child built and the sibling derived;
    // `next` (may be nullptr): the features to keep for the next depth.
    void score(const BinnedView& X, const HistDepth& depth, const std::vector<std::uint32_t>& candidates,
               const HistCache* prev, HistCache* next, std::vector<std::vector<HistSplit>>& results,
               ThreadPool* pool) const {
        const std::size_t n_cand = candidates.size();
        results.resize(n_cand);
        // Candidates grouped by bin block, in candidate order within a block
        // (a column with a single bin has no split).
        std::vector<std::vector<std::size_t>> by_block(X.m->n_bin_blocks());
        for (std::size_t i = 0; i < n_cand; ++i) {
            const std::size_t col = X.col(candidates[i]);
            if (X.m->n_bins(col) >= 2) {
                by_block[col / kBinBlock].push_back(i);
            } else {
                results[i].assign(depth.n_nodes, HistSplit{});
            }
        }
        // Tasks: a block's candidates, split further when there are too few
        // blocks to keep the threads busy. Each (candidate, node) result is
        // computed by one task, so the split does not change results.
        std::vector<std::vector<std::size_t>> tasks;
        std::size_t n_used = 0;
        for (const auto& b : by_block) n_used += b.empty() ? 0 : 1;
        const std::size_t want = pool && pool->size() > 1 ? 2 * pool->size() : 1;
        const std::size_t parts = n_used > 0 && n_used < want ? (want + n_used - 1) / n_used : 1;
        for (auto& b : by_block) {
            if (b.empty()) continue;
            const std::size_t np = std::min(parts, b.size());
            for (std::size_t q = 0; q < np; ++q) {
                tasks.emplace_back(b.begin() + static_cast<std::ptrdiff_t>(b.size() * q / np),
                                   b.begin() + static_cast<std::ptrdiff_t>(b.size() * (q + 1) / np));
            }
        }
        run_tasks(pool, tasks.size(), [&](std::size_t t) {
            if (X.m->wide()) {
                score_block<std::uint16_t>(X, depth, candidates, tasks[t], prev, next, results);
            } else {
                score_block<std::uint8_t>(X, depth, candidates, tasks[t], prev, next, results);
            }
        });
    }

private:
    Criterion criterion_;
    int min_samples_leaf_;
    std::size_t n_classes_;
    NewtonParams newton_;
    std::size_t n_stats_ = 0;
    std::vector<std::size_t> node_size_;
    std::vector<double> total_;   // per node: n_stats_ totals (MSE: sum w*d, w*d^2, w; Newton: G, H; classes)
    std::vector<double> total_w_; // classification: per-node total weight

    // One feature's histogram for one node: n_bins x n_stats statistics and
    // n_bins row counts.
    struct HistRef {
        double* h;
        std::uint32_t* c;
    };

    // A block task's per-feature state.
    struct BlockFeature {
        std::size_t cand;      // index into candidates / results
        std::uint32_t j;       // position in the bin block
        std::size_t n_bins;
        std::size_t scratch;   // offset of its two histograms in the task's scratch
        std::int32_t prev;     // slot in the previous depth's cache, or -1
        std::int32_t next;     // slot in the next depth's cache, or -1
    };

    struct SweepScratch {
        std::vector<double> left, right;
        std::vector<std::uint32_t> bins; // a node's non-empty bins, ascending
    };

    // The non-empty bins of a histogram, ascending, into sc.bins; returns
    // their number. (Branch-free: which bins are empty is unpredictable.)
    static std::size_t non_empty_bins(const std::uint32_t* cnt, std::size_t n_bins, SweepScratch& sc) {
        if (sc.bins.size() < n_bins) sc.bins.resize(n_bins);
        std::uint32_t* out = sc.bins.data();
        std::size_t n = 0;
        for (std::size_t b = 0; b < n_bins; ++b) {
            out[n] = static_cast<std::uint32_t>(b);
            n += cnt[b] != 0 ? 1 : 0;
        }
        return n;
    }

    template <class Bin>
    static const Bin* block_data(const BinnedMatrix& m, std::size_t blk);

    // Histograms of the features `feats` (positions j in the bin block) for
    // the entries [e0, e1): a bin's statistics are added in entry order.
    // hist[q] must be zero on entry.
    template <class Bin>
    void accumulate(const Bin* block, std::size_t width, const HistDepth& depth, std::size_t e0, std::size_t e1,
                    const std::uint32_t* js, const HistRef* hist, std::size_t m) const {
        constexpr std::size_t kAhead = 16; // rows of bin lines prefetched ahead
        const std::uint32_t* mrow = depth.mrow;
        if (criterion_ == Criterion::Newton) {
            const double* st = depth.stat;
            for (std::size_t e = e0; e < e1; ++e) {
                if (e + kAhead < e1) prefetch_read(block + static_cast<std::size_t>(mrow[e + kAhead]) * width);
                const Bin* rb = block + static_cast<std::size_t>(mrow[e]) * width;
                const double g = st[2 * e], h = st[2 * e + 1];
                for (std::size_t q = 0; q < m; ++q) {
                    const std::size_t b = rb[js[q]];
                    double* hb = hist[q].h + 2 * b;
                    hb[0] += g;
                    hb[1] += h;
                    ++hist[q].c[b];
                }
            }
        } else if (criterion_ == Criterion::MSE) {
            const double* st = depth.stat;
            for (std::size_t e = e0; e < e1; ++e) {
                if (e + kAhead < e1) prefetch_read(block + static_cast<std::size_t>(mrow[e + kAhead]) * width);
                const Bin* rb = block + static_cast<std::size_t>(mrow[e]) * width;
                const double s0 = st[3 * e], s1 = st[3 * e + 1], s2 = st[3 * e + 2];
                for (std::size_t q = 0; q < m; ++q) {
                    const std::size_t b = rb[js[q]];
                    double* hb = hist[q].h + 3 * b;
                    hb[0] += s0;
                    hb[1] += s1;
                    hb[2] += s2;
                    ++hist[q].c[b];
                }
            }
        } else {
            const std::size_t K = n_classes_;
            const double* st = depth.stat;
            const std::int32_t* cls = depth.cls;
            for (std::size_t e = e0; e < e1; ++e) {
                if (e + kAhead < e1) prefetch_read(block + static_cast<std::size_t>(mrow[e + kAhead]) * width);
                const Bin* rb = block + static_cast<std::size_t>(mrow[e]) * width;
                const double w = st[e];
                const std::size_t c = static_cast<std::size_t>(cls[e]);
                for (std::size_t q = 0; q < m; ++q) {
                    const std::size_t b = rb[js[q]];
                    hist[q].h[b * K + c] += w;
                    ++hist[q].c[b];
                }
            }
        }
    }

    void sweep(const HistRef& hr, std::size_t n_bins, std::size_t k, HistSplit& out, SweepScratch& sc) const {
        if (criterion_ == Criterion::MSE) {
            sweep_mse(hr.h, hr.c, n_bins, k, out, sc);
        } else if (criterion_ == Criterion::Newton) {
            sweep_newton(hr.h, hr.c, n_bins, k, out, sc);
        } else {
            sweep_classification(hr.h, hr.c, n_bins, k, out, sc);
        }
    }

    // dst = parent - built, bin by bin (histogram subtraction).
    void subtract(const HistRef& parent, const HistRef& built, const HistRef& dst, std::size_t n_bins) const {
        const std::size_t n = n_bins * n_stats_;
        for (std::size_t i = 0; i < n; ++i) dst.h[i] = parent.h[i] - built.h[i];
        for (std::size_t b = 0; b < n_bins; ++b) dst.c[b] = parent.c[b] - built.c[b];
    }

    // MSE statistics centred at c, re-centred at c + delta.
    static void recenter(const HistRef& hr, std::size_t n_bins, double delta) {
        for (std::size_t b = 0; b < n_bins; ++b) {
            double* h = hr.h + 3 * b;
            const double s = h[0], w = h[2];
            h[0] = s - delta * w;
            h[1] = h[1] - 2.0 * delta * s + delta * delta * w;
        }
    }

    // One task: the candidates `cand_idx` (all in one bin block), for every
    // scored node of the depth, a group of siblings at a time. A candidate
    // kept in `prev` builds the group's smaller child (or the only scored
    // one, when it is the smaller) and derives the other from the parent; any
    // other candidate builds each scored child.
    template <class Bin>
    void score_block(const BinnedView& X, const HistDepth& depth, const std::vector<std::uint32_t>& candidates,
                     const std::vector<std::size_t>& cand_idx, const HistCache* prev, HistCache* next,
                     std::vector<std::vector<HistSplit>>& results) const {
        const std::size_t S = n_stats_;
        const std::size_t blk = X.col(candidates[cand_idx[0]]) / kBinBlock;
        const std::size_t width = X.m->block_width(blk);
        const Bin* block = block_data<Bin>(*X.m, blk);
        const std::size_t m = cand_idx.size();

        std::vector<BlockFeature> feats(m);
        std::size_t scratch_size = 0;
        bool any_prev = false;
        for (std::size_t q = 0; q < m; ++q) {
            const std::size_t f = candidates[cand_idx[q]];
            const std::size_t col = X.col(f);
            feats[q] = {cand_idx[q], static_cast<std::uint32_t>(col - blk * kBinBlock), X.m->n_bins(col), scratch_size,
                        prev ? prev->slot(f) : -1, next ? next->slot(f) : -1};
            any_prev = any_prev || feats[q].prev >= 0;
            scratch_size += 2 * feats[q].n_bins;
            results[cand_idx[q]].assign(depth.n_nodes, HistSplit{});
        }
        std::vector<double> hist(scratch_size * S);
        std::vector<std::uint32_t> count(scratch_size);
        auto scratch = [&](std::size_t q, int side) {
            const std::size_t off = feats[q].scratch + static_cast<std::size_t>(side) * feats[q].n_bins;
            return HistRef{hist.data() + off * S, count.data() + off};
        };
        std::vector<HistRef> dest[2] = {std::vector<HistRef>(m), std::vector<HistRef>(m)};
        std::vector<std::uint32_t> js(m);
        std::vector<HistRef> refs(m);
        SweepScratch sc;

        for (const HistGroup& g : depth.groups) {
            const std::size_t n_side[2] = {g.end[0] - g.begin[0], g.end[1] - g.begin[1]};
            // The child the subtracted candidates build (-1: none).
            int build = -1;
            if (any_prev && g.parent >= 0) {
                if (g.node[0] >= 0 && g.node[1] >= 0) {
                    build = n_side[1] < n_side[0] ? 1 : 0;
                } else {
                    const int s = g.node[0] >= 0 ? 0 : 1;
                    build = n_side[1 - s] < n_side[s] ? 1 - s : s;
                }
            }
            for (int side = 0; side < 2; ++side) {
                for (std::size_t q = 0; q < m; ++q) {
                    dest[side][q] = g.node[side] >= 0 && feats[q].next >= 0
                                        ? HistRef{next->hist(feats[q].next, static_cast<std::size_t>(g.node[side])),
                                                  next->count(feats[q].next, static_cast<std::size_t>(g.node[side]))}
                                        : scratch(q, side);
                }
            }
            // Build: a scored child for the direct candidates, the chosen
            // child for the subtracted ones.
            for (int side = 0; side < 2; ++side) {
                std::size_t n_build = 0;
                for (std::size_t q = 0; q < m; ++q) {
                    const bool subtracted = build >= 0 && feats[q].prev >= 0;
                    if (subtracted ? side == build : g.node[side] >= 0) {
                        js[n_build] = feats[q].j;
                        refs[n_build] = dest[side][q];
                        std::fill(refs[n_build].h, refs[n_build].h + feats[q].n_bins * S, 0.0);
                        std::fill(refs[n_build].c, refs[n_build].c + feats[q].n_bins, 0u);
                        ++n_build;
                    }
                }
                if (n_build > 0 && n_side[side] > 0) {
                    accumulate<Bin>(block, width, depth, g.begin[side], g.end[side], js.data(), refs.data(), n_build);
                }
            }
            // Derive the other child, if it is scored.
            if (build >= 0 && g.node[1 - build] >= 0) {
                for (std::size_t q = 0; q < m; ++q) {
                    if (feats[q].prev < 0) continue;
                    const HistRef parent{prev->hist(feats[q].prev, static_cast<std::size_t>(g.parent)),
                                         prev->count(feats[q].prev, static_cast<std::size_t>(g.parent))};
                    subtract(parent, dest[build][q], dest[1 - build][q], feats[q].n_bins);
                }
            }
            // Sweep the scored children.
            for (int side = 0; side < 2; ++side) {
                const std::int32_t k = g.node[side];
                if (k < 0) continue;
                const std::size_t ku = static_cast<std::size_t>(k);
                for (std::size_t q = 0; q < m; ++q) {
                    if (!depth.recenter.empty()) recenter(dest[side][q], feats[q].n_bins, depth.recenter[ku]);
                    sweep(dest[side][q], feats[q].n_bins, ku, results[feats[q].cand][ku], sc);
                }
            }
        }
    }

    bool leaf_sizes_ok(std::size_t n_left, std::size_t n_total) const {
        const std::size_t n_right = n_total - n_left;
        return !(static_cast<int>(n_left) < min_samples_leaf_ || static_cast<int>(n_right) < min_samples_leaf_);
    }

    // The sweeps visit a node's non-empty bins left to right; each boundary
    // between consecutive ones is a candidate split.
    void sweep_mse(const double* h, const std::uint32_t* cnt, std::size_t n_bins, std::size_t k, HistSplit& out,
                   SweepScratch& sc) const {
        const std::size_t n_used = non_empty_bins(cnt, n_bins, sc);
        const std::uint32_t* used = sc.bins.data();
        const std::size_t n = node_size_[k];
        const double total_sum = total_[k * 3], total_sum_sq = total_[k * 3 + 1], total_weight = total_[k * 3 + 2];
        const double parent_impurity = mse_from_sums(total_sum, total_sum_sq, total_weight);
        double best_improvement = -std::numeric_limits<double>::infinity();
        double left_sum = 0.0, left_sum_sq = 0.0, left_weight = 0.0;
        std::size_t n_left = 0, prev = 0;
        for (std::size_t u = 0; u < n_used; ++u) {
            const std::size_t b = used[u];
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

    void sweep_newton(const double* h, const std::uint32_t* cnt, std::size_t n_bins, std::size_t k, HistSplit& out,
                      SweepScratch& sc) const {
        const std::size_t n_used = non_empty_bins(cnt, n_bins, sc);
        const std::uint32_t* used = sc.bins.data();
        const std::size_t n = node_size_[k];
        const double G = total_[k * 2], H = total_[k * 2 + 1];
        const double parent_score = newton_score(G, H, newton_);
        double best_gain = -std::numeric_limits<double>::infinity();
        double left_g = 0.0, left_h = 0.0;
        std::size_t n_left = 0, prev = 0;
        for (std::size_t u = 0; u < n_used; ++u) {
            const std::size_t b = used[u];
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
                              HistSplit& out, SweepScratch& sc) const {
        const std::size_t n = node_size_[k];
        const std::size_t K = n_classes_;
        const double* total = total_.data() + k * K;
        const double total_weight = total_w_[k];
        const double parent_impurity = class_impurity(total, total_weight);
        sc.left.assign(K, 0.0);
        sc.right.resize(K);
        const std::size_t n_used = non_empty_bins(cnt, n_bins, sc);
        const std::uint32_t* used = sc.bins.data();
        double best_improvement = -std::numeric_limits<double>::infinity();
        double left_weight = 0.0;
        std::size_t n_left = 0, prev = 0;
        for (std::size_t u = 0; u < n_used; ++u) {
            const std::size_t b = used[u];
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

template <>
inline const std::uint8_t* HistScanner::block_data<std::uint8_t>(const BinnedMatrix& m, std::size_t blk) {
    return m.block8(blk);
}
template <>
inline const std::uint16_t* HistScanner::block_data<std::uint16_t>(const BinnedMatrix& m, std::size_t blk) {
    return m.block16(blk);
}

} // namespace rdt
