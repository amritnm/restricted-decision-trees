#pragma once

#include <cstddef>

namespace rdt {

// Non-owning view over a contiguous 2D buffer of doubles. Supports both
// row-major (NumPy's default C order) and column-major (R's native storage)
// layouts so both language bindings can pass their arrays through with zero
// copies.
class MatrixView {
public:
    MatrixView() = default;

    MatrixView(const double* data, std::size_t n_rows, std::size_t n_cols, bool row_major)
        : data_(data), n_rows_(n_rows), n_cols_(n_cols), row_major_(row_major) {}

    double operator()(std::size_t row, std::size_t col) const {
        return row_major_ ? data_[row * n_cols_ + col] : data_[col * n_rows_ + row];
    }

    std::size_t n_rows() const { return n_rows_; }
    std::size_t n_cols() const { return n_cols_; }

private:
    const double* data_ = nullptr;
    std::size_t n_rows_ = 0;
    std::size_t n_cols_ = 0;
    bool row_major_ = true;
};

// Non-owning view over a contiguous 1D buffer of doubles (labels/targets).
class VectorView {
public:
    VectorView() = default;

    VectorView(const double* data, std::size_t n) : data_(data), n_(n) {}

    double operator[](std::size_t i) const { return data_[i]; }
    std::size_t size() const { return n_; }

private:
    const double* data_ = nullptr;
    std::size_t n_ = 0;
};

} // namespace rdt
