/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#pragma once

#include "util.hpp"
#include <vector>

template <typename T> struct perflibs_spmat_impl_t;

namespace perflibs::sparse {

template <typename T> struct perflibs_dense {
  perflibs_dense_layout layout;
  int64_t m;
  int64_t n;
  int64_t lda;
  bool matrix_is_zero =
      false; // a flag to state that the matrix contains all zero values (i.e.
             // has probably been materialized in place of a null_matrix input,
             // in order to store a result)

  // Our own copy
  std::vector<T> vals;

  // Either user's data or pointer to above
  const T *vals_ptr;

  // Set ints negative so that we know when to copy data in the copy operator
  perflibs_dense()
      : layout(PERFLIBS_COL_MAJOR), m(-1), n(-1), lda(-1), vals_ptr(nullptr) {}

  // Copy version
  perflibs_dense(perflibs_dense_layout layout, int64_t m, int64_t n,
                 int64_t lda, const T *vals, int64_t nnz)
      : layout(layout), m(m), n(n), vals(nnz), vals_ptr(this->vals.data()) {
    if (layout == PERFLIBS_COL_MAJOR) {
      this->lda = m;
      for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < m; j++) {
          this->vals[i * this->lda + j] = vals[i * lda + j];
        }
      }
    } else {
      this->lda = n;
      for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
          this->vals[i * this->lda + j] = vals[i * lda + j];
        }
      }
    }
  }

  // No-copy version
  perflibs_dense(perflibs_dense_layout layout, int64_t m, int64_t n,
                 int64_t lda, const T *vals)
      : layout(layout), m(m), n(n), lda(lda), vals_ptr(vals) {}

  void transpose() {
    layout =
        layout == PERFLIBS_COL_MAJOR ? PERFLIBS_ROW_MAJOR : PERFLIBS_COL_MAJOR;
    std::swap(m, n);
  }

  perflibs_dense &operator=(const perflibs_dense &other);
  perflibs_dense(const perflibs_dense &other) { *this = other; }
  perflibs_dense &operator=(perflibs_dense &&other) = default;
  perflibs_dense(perflibs_dense &&other) = default;

  void scale_matrix(enum perflibs_sparse_hint_value trans, T alpha);
  void make_writable() {
    if (vals.empty()) {
      scale_matrix(PERFLIBS_SPARSE_OPERATION_NOTRANS, (T)1.0);
    }
  };

  void printer() const {
    auto stride_row = layout == PERFLIBS_ROW_MAJOR ? lda : 1;
    auto stride_col = layout == PERFLIBS_COL_MAJOR ? lda : 1;
    for (perflibs_int_t r = 0; r < m; r++) {
      for (perflibs_int_t c = 0; c < n; c++) {
        if constexpr (std::is_same_v<float, T> || std::is_same_v<double, T>) {
          if (vals_ptr[stride_col * c + stride_row * r] >= 0.0)
            fprintf(stderr, " ");
          fprintf(stderr, "%.1f ", vals_ptr[stride_col * c + stride_row * r]);
        }
        if constexpr (std::is_same_v<std::complex<float>, T> ||
                      std::is_same_v<std::complex<double>, T>) {
          fprintf(stderr, "(");
          if (vals_ptr[stride_col * c + stride_row * r].real() >= 0.0)
            fprintf(stderr, " ");
          fprintf(stderr, "%.1f",
                  vals_ptr[stride_col * c + stride_row * r].real());
          fprintf(stderr, ",");
          if (vals_ptr[stride_col * c + stride_row * r].imag() >= 0.0)
            fprintf(stderr, " ");
          fprintf(stderr, "%.1f",
                  vals_ptr[stride_col * c + stride_row * r].imag());
          fprintf(stderr, ") ");
        }
      }
      fprintf(stderr, "\n");
    }
  }
};

template <typename T>
perflibs_status_t
fill_initial_data_dense(perflibs_spmat_t A, perflibs_dense_layout layout,
                        perflibs_int_t m, perflibs_int_t n, perflibs_int_t lda,
                        perflibs_int_t index_base, const T *vals, bool no_copy);

template <typename T>
std::vector<T> csr2dense(perflibs_int_t m, perflibs_int_t n, const T *vals,
                         const perflibs_int_t *col_indx,
                         const perflibs_int_t *row_ptr);

/// Convert special (i.e. null/identity) matrices into their dense equivalents
template <typename T>
std::vector<T> special2dense(perflibs_int_t m, perflibs_int_t n,
                             spmat_format_t type);

template <typename T>
perflibs_status_t spmv_gemv(perflibs_dense_layout layout,
                            perflibs_sparse_hint_value trans, perflibs_int_t m,
                            perflibs_int_t n, const T *A, perflibs_int_t lda,
                            T alpha, const T *x, T beta, T *y);

template <typename T>
perflibs_status_t
spmm_gemm(perflibs_dense_layout layout, perflibs_sparse_hint_value transA,
          perflibs_sparse_hint_value transB, perflibs_int_t m, perflibs_int_t n,
          perflibs_int_t k, T alpha, const T *A, perflibs_int_t lda, const T *B,
          perflibs_int_t ldb, T beta, T *C, perflibs_int_t ldc);

template <typename T>
perflibs_status_t
spsv_trsv(perflibs_dense_layout layout, sparse_hint_value_internal trans,
          sparse_hint_value_internal uplo, sparse_hint_value_internal diag,
          perflibs_int_t n, const T *A, perflibs_int_t lda, T *x);

// used in BSR
template <typename T>
perflibs_sparse_matrix_shape_t
get_shape_dense(perflibs_dense_layout layout, perflibs_int_t m,
                perflibs_int_t n, const T *vals, perflibs_int_t lda);

template <typename T>
void spnorm_inf_dense(const perflibs_dense<T> &dense,
                      perflibs::sparse::remove_complex_t<T> *result);

template <typename T>
perflibs_status_t
spmat_update_dense(perflibs_spmat_impl_t<T> *impl, perflibs_int_t n_updates,
                   const perflibs_int_t *row_indx,
                   const perflibs_int_t *col_indx, const T *vals);

template <typename T>
perflibs_status_t
spelmm_dense(perflibs_sparse_hint_value transA, const perflibs_dense<T> &A,
             perflibs_sparse_hint_value transB, const perflibs_dense<T> &B,
             perflibs_dense_layout layoutC, perflibs_int_t ldaC,
             perflibs_spmat_t AB);
} // namespace perflibs::sparse
