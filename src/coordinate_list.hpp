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

template <typename T> struct perflibs_coo {
  int64_t nnz;
  int64_t m;
  int64_t n;
  int64_t index_base;

  // When we're copying:
  std::vector<T> vals;
  std::vector<perflibs_int_t> row_indx;
  std::vector<perflibs_int_t> col_indx;

  // Pointers to either user data or the data under the vector copies
  const T *vals_ptr;
  const perflibs_int_t *row_indx_ptr;
  const perflibs_int_t *col_indx_ptr;

  int nthreads = perflibs::sparse::omp::get_max_threads();

  // Set ints negative so that we know when to copy data in the copy operator
  perflibs_coo()
      : nnz(-1), m(-1), n(-1), index_base(-1), vals_ptr(nullptr),
        row_indx_ptr(nullptr), col_indx_ptr(nullptr) {}
  perflibs_coo(int64_t m, int64_t n, int64_t nnz, int64_t index_base)
      : nnz(nnz), m(m), n(n), index_base(index_base), vals(nnz), row_indx(nnz),
        col_indx(nnz) {}
  perflibs_coo(int64_t m, int64_t n, int64_t nnz, int64_t index_base,
               const T *vals, const perflibs_int_t *row_indx,
               const perflibs_int_t *col_indx)
      : nnz(nnz), m(m), n(n), index_base(index_base), vals(vals, vals + nnz),
        row_indx(row_indx, row_indx + nnz), col_indx(col_indx, col_indx + nnz),
        vals_ptr(this->vals.data()), row_indx_ptr(this->row_indx.data()),
        col_indx_ptr(this->col_indx.data()) {}
  perflibs_coo(int64_t m, int64_t n, int64_t index_base, const T *vals,
               const perflibs_int_t *row_indx, const perflibs_int_t *col_indx,
               perflibs_int_t nnz)
      : nnz(nnz), m(m), n(n), index_base(index_base), vals_ptr(vals),
        row_indx_ptr(row_indx), col_indx_ptr(col_indx) {}

  perflibs_coo &operator=(const perflibs_coo &other);
  perflibs_coo(const perflibs_coo &other) { *this = other; }
  perflibs_coo &operator=(perflibs_coo &&other) = default;
  perflibs_coo(perflibs_coo &&other) = default;

  void scale_matrix(enum perflibs_sparse_hint_value trans, T alpha);
};

template <typename T>
perflibs_status_t fill_initial_data_coo(perflibs_spmat_t A, perflibs_int_t m,
                                        perflibs_int_t n, perflibs_int_t nnz,
                                        perflibs_int_t index_base,
                                        const perflibs_int_t *row_indx,
                                        const perflibs_int_t *col_indx,
                                        const T *vals, bool no_copy);

template <typename T>
perflibs_coo<T> csr2coo(perflibs_int_t m, perflibs_int_t n, const T *vals,
                        const perflibs_int_t *col_indx,
                        const perflibs_int_t *row_ptr);

template <typename T>
void spmv_coo(perflibs_coo<T> &coo, const sparse_hint_value_internal trans,
              const T *const x, T *const y, const T alpha, const T beta,
              perflibs_int_t index_base);

template <typename T>
void spnorm_inf_coo(const perflibs_coo<T> &coo,
                    perflibs::sparse::remove_complex_t<T> *result);

template <typename T>
perflibs_status_t
spmat_update_coo(perflibs_spmat_impl_t<T> *impl, perflibs_int_t n_updates,
                 const perflibs_int_t *row_indx, const perflibs_int_t *col_indx,
                 const T *vals);

template <typename T>
perflibs_status_t spelmm_coo(perflibs_sparse_hint_value transA,
                             const perflibs_coo<T> &A,
                             perflibs_sparse_hint_value transB,
                             const perflibs_coo<T> &B, perflibs_spmat_t AB);
} // namespace perflibs::sparse
