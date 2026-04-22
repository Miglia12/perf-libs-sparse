/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "special.hpp"
#include "cblas_wrappers.hpp"

#include <algorithm>
#include <cstring>

namespace perflibs::sparse {

template <typename T>
perflibs_status_t spmv_exec_identity(perflibs_sparse_hint_value trans, T alpha,
                                     perflibs_spmat_impl_t<T> *impl, const T *x,
                                     T beta, T *y) {
  // The total length of the y vector
  auto y_len = trans == PERFLIBS_SPARSE_OPERATION_NOTRANS ? impl->m : impl->n;
  // The length of the y vector that is affected by multiplication by the
  // diagonal
  auto y_mul_len = std::min(impl->m, impl->n);
  // The difference
  auto y_remain = y_len - y_mul_len;

  if (alpha == (T)1 && beta == (T)0) {
    std::memcpy((void *)y, (const void *)x, sizeof(T) * y_mul_len);
    std::memset((void *)&y[y_mul_len], 0, sizeof(T) * y_remain);
  }

  else {
    if (beta == (T)0) {
      for (perflibs_int_t i = 0; i < y_mul_len; i++) {
        y[i] = alpha * x[i];
      }
      std::memset((void *)&y[y_mul_len], 0, sizeof(T) * y_remain);
    } else {
      perflibs_int_t inc = 1;
      perflibs::sparse::cblas_axpby<T>(y_mul_len, alpha, x, inc, beta, y, inc);
      perflibs::sparse::cblas_scal<T>(y_remain, beta, &y[y_mul_len], inc);
    }
  }

  return PERFLIBS_STATUS_SUCCESS;
};
template perflibs_status_t
spmv_exec_identity<float>(perflibs_sparse_hint_value trans, float alpha,
                          perflibs_spmat_impl_t<float> *impl, const float *x,
                          float beta, float *y);
template perflibs_status_t
spmv_exec_identity<double>(perflibs_sparse_hint_value trans, double alpha,
                           perflibs_spmat_impl_t<double> *impl, const double *x,
                           double beta, double *y);
template perflibs_status_t spmv_exec_identity<std::complex<float>>(
    perflibs_sparse_hint_value trans, std::complex<float> alpha,
    perflibs_spmat_impl_t<std::complex<float>> *impl,
    const std::complex<float> *x, std::complex<float> beta,
    std::complex<float> *y);
template perflibs_status_t spmv_exec_identity<std::complex<double>>(
    perflibs_sparse_hint_value trans, std::complex<double> alpha,
    perflibs_spmat_impl_t<std::complex<double>> *impl,
    const std::complex<double> *x, std::complex<double> beta,
    std::complex<double> *y);

template <typename T>
perflibs_status_t spmv_exec_null(perflibs_sparse_hint_value trans,
                                 perflibs_spmat_impl_t<T> *impl, T beta, T *y) {
  perflibs_int_t y_len =
      trans == PERFLIBS_SPARSE_OPERATION_NOTRANS ? impl->m : impl->n;

  if (beta == (T)0) {
    std::memset((void *)y, 0, sizeof(T) * y_len);
  } else {
    perflibs_int_t inc = 1;
    perflibs::sparse::cblas_scal<T>(y_len, beta, y, inc);
  }
  return PERFLIBS_STATUS_SUCCESS;
};
template perflibs_status_t spmv_exec_null(perflibs_sparse_hint_value trans,
                                          perflibs_spmat_impl_t<float> *impl,
                                          float beta, float *y);
template perflibs_status_t spmv_exec_null(perflibs_sparse_hint_value trans,
                                          perflibs_spmat_impl_t<double> *impl,
                                          double beta, double *y);
template perflibs_status_t
spmv_exec_null(perflibs_sparse_hint_value trans,
               perflibs_spmat_impl_t<std::complex<float>> *impl,
               std::complex<float> beta, std::complex<float> *y);
template perflibs_status_t
spmv_exec_null(perflibs_sparse_hint_value trans,
               perflibs_spmat_impl_t<std::complex<double>> *impl,
               std::complex<double> beta, std::complex<double> *y);

} // namespace perflibs::sparse
