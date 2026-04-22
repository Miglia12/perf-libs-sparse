/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#pragma once

#include "int.hpp"
#define PERFLIBS_AVOID_SINGLECOMPLEX_RETURN
#define PERFLIBS_AVOID_DOUBLECOMPLEX_RETURN
#include <cassert>
#include <cblas.h>
#include <complex>

// Templated C++ functions calling into dense BLAS via cblas interfaces.
// Functions to be added on demand.

namespace perflibs::sparse {

// start of cblas_gemv
template <typename T>
using cblas_gemv_f = void (*)(CBLAS_LAYOUT, CBLAS_TRANSPOSE, perflibs_int_t,
                              perflibs_int_t, T, const T *, perflibs_int_t,
                              const T *, perflibs_int_t, T, T *,
                              perflibs_int_t);

template <typename T> extern cblas_gemv_f<T> cblas_gemv;

template <> inline cblas_gemv_f<float> cblas_gemv<float> = &cblas_sgemv;

template <> inline cblas_gemv_f<double> cblas_gemv<double> = &cblas_dgemv;

inline void cblas_cgemv_wrap(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans,
                             perflibs_int_t M, perflibs_int_t N,
                             std::complex<float> alpha,
                             const std::complex<float> *A, perflibs_int_t lda,
                             const std::complex<float> *X, perflibs_int_t incX,
                             std::complex<float> beta, std::complex<float> *Y,
                             perflibs_int_t incY) {
  cblas_cgemv(layout, trans, M, N, &alpha, A, lda, X, incX, &beta, Y, incY);
}

inline void cblas_zgemv_wrap(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans,
                             perflibs_int_t M, perflibs_int_t N,
                             std::complex<double> alpha,
                             const std::complex<double> *A, perflibs_int_t lda,
                             const std::complex<double> *X, perflibs_int_t incX,
                             std::complex<double> beta, std::complex<double> *Y,
                             perflibs_int_t incY) {
  cblas_zgemv(layout, trans, M, N, &alpha, A, lda, X, incX, &beta, Y, incY);
}

template <>
inline cblas_gemv_f<std::complex<float>> cblas_gemv<std::complex<float>> =
    &cblas_cgemv_wrap;

template <>
inline cblas_gemv_f<std::complex<double>> cblas_gemv<std::complex<double>> =
    &cblas_zgemv_wrap;
// end of cblas_gemv

// start of cblas_gemm
template <typename T>
using cblas_gemm_f = void (*)(CBLAS_LAYOUT, CBLAS_TRANSPOSE, CBLAS_TRANSPOSE,
                              perflibs_int_t, perflibs_int_t, perflibs_int_t, T,
                              const T *, perflibs_int_t, const T *,
                              perflibs_int_t, T, T *, perflibs_int_t);

template <typename T> extern cblas_gemm_f<T> cblas_gemm;

template <> inline cblas_gemm_f<float> cblas_gemm<float> = &cblas_sgemm;

template <> inline cblas_gemm_f<double> cblas_gemm<double> = &cblas_dgemm;

inline void cblas_cgemm_wrap(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE ta,
                             CBLAS_TRANSPOSE tb, perflibs_int_t M,
                             perflibs_int_t N, perflibs_int_t K,
                             std::complex<float> alpha,
                             const std::complex<float> *A, perflibs_int_t lda,
                             const std::complex<float> *B, perflibs_int_t ldb,
                             std::complex<float> beta, std::complex<float> *C,
                             perflibs_int_t ldc) {
  cblas_cgemm(layout, ta, tb, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

inline void cblas_zgemm_wrap(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE ta,
                             CBLAS_TRANSPOSE tb, perflibs_int_t M,
                             perflibs_int_t N, perflibs_int_t K,
                             std::complex<double> alpha,
                             const std::complex<double> *A, perflibs_int_t lda,
                             const std::complex<double> *B, perflibs_int_t ldb,
                             std::complex<double> beta, std::complex<double> *C,
                             perflibs_int_t ldc) {
  cblas_zgemm(layout, ta, tb, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

template <>
inline cblas_gemm_f<std::complex<float>> cblas_gemm<std::complex<float>> =
    &cblas_cgemm_wrap;

template <>
inline cblas_gemm_f<std::complex<double>> cblas_gemm<std::complex<double>> =
    &cblas_zgemm_wrap;
// end of cblas_gemm

// start of cblas_trsv
template <typename T>
using cblas_trsv_f = void (*)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE,
                              CBLAS_DIAG, perflibs_int_t, const T *,
                              perflibs_int_t, T *, perflibs_int_t);

template <typename T> extern cblas_trsv_f<T> cblas_trsv;

// Real specializations
template <> inline cblas_trsv_f<float> cblas_trsv<float> = &cblas_strsv;

template <> inline cblas_trsv_f<double> cblas_trsv<double> = &cblas_dtrsv;

inline void cblas_ctrsv_wrap(CBLAS_LAYOUT layout, CBLAS_UPLO uplo,
                             CBLAS_TRANSPOSE ta, CBLAS_DIAG diag,
                             perflibs_int_t N, const std::complex<float> *A,
                             perflibs_int_t lda, std::complex<float> *x,
                             perflibs_int_t incx) {
  cblas_ctrsv(layout, uplo, ta, diag, N, reinterpret_cast<const void *>(A), lda,
              reinterpret_cast<void *>(x), incx);
}

inline void cblas_ztrsv_wrap(CBLAS_LAYOUT layout, CBLAS_UPLO uplo,
                             CBLAS_TRANSPOSE ta, CBLAS_DIAG diag,
                             perflibs_int_t N, const std::complex<double> *A,
                             perflibs_int_t lda, std::complex<double> *x,
                             perflibs_int_t incx) {
  cblas_ztrsv(layout, uplo, ta, diag, N, reinterpret_cast<const void *>(A), lda,
              reinterpret_cast<void *>(x), incx);
}

template <>
inline cblas_trsv_f<std::complex<float>> cblas_trsv<std::complex<float>> =
    &cblas_ctrsv_wrap;

template <>
inline cblas_trsv_f<std::complex<double>> cblas_trsv<std::complex<double>> =
    &cblas_ztrsv_wrap;
// end of cblas_trsv

// start of cblas_axpby
template <typename T>
using cblas_axpby_f = void (*)(perflibs_int_t, T, const T *, perflibs_int_t, T,
                               T *, perflibs_int_t);

template <typename T> extern cblas_axpby_f<T> cblas_axpby;

template <> inline cblas_axpby_f<float> cblas_axpby<float> = &cblas_saxpby;

template <> inline cblas_axpby_f<double> cblas_axpby<double> = &cblas_daxpby;

inline void cblas_caxpby_wrap(perflibs_int_t N, std::complex<float> alpha,
                              const std::complex<float> *X, perflibs_int_t incX,
                              std::complex<float> beta, std::complex<float> *Y,
                              perflibs_int_t incY) {

  // Some libs fail for in-place complex axpby
  if (X == Y) {
    assert(incX == incY && incX == 1);
    for (perflibs_int_t i = 0; i < N; i++) {
      Y[i * incY] *= alpha;
    }
  } else {
    cblas_caxpby(N, &alpha, X, incX, &beta, Y, incY);
  }
}

inline void cblas_zaxpby_wrap(perflibs_int_t N, std::complex<double> alpha,
                              const std::complex<double> *X,
                              perflibs_int_t incX, std::complex<double> beta,
                              std::complex<double> *Y, perflibs_int_t incY) {
  // Some libs fail for in-place complex axpby
  if (X == Y) {
    assert(incX == incY && incX == 1);
    for (perflibs_int_t i = 0; i < N; i++) {
      Y[i * incY] *= alpha;
    }
  } else {
    cblas_zaxpby(N, &alpha, X, incX, &beta, Y, incY);
  }
}

template <>
inline cblas_axpby_f<std::complex<float>> cblas_axpby<std::complex<float>> =
    &cblas_caxpby_wrap;

template <>
inline cblas_axpby_f<std::complex<double>> cblas_axpby<std::complex<double>> =
    &cblas_zaxpby_wrap;
// end of cblas_axpby

// start of cblas_scal
template <typename T>
using cblas_scal_f = void (*)(perflibs_int_t, T, T *, perflibs_int_t);

template <typename T> extern cblas_scal_f<T> cblas_scal;

template <> inline cblas_scal_f<float> cblas_scal<float> = &cblas_sscal;

template <> inline cblas_scal_f<double> cblas_scal<double> = &cblas_dscal;

inline void cblas_cscal_wrap(perflibs_int_t N, std::complex<float> alpha,
                             std::complex<float> *X, perflibs_int_t incX) {
  cblas_cscal(N, &alpha, X, incX);
}

inline void cblas_zscal_wrap(perflibs_int_t N, std::complex<double> alpha,
                             std::complex<double> *X, perflibs_int_t incX) {
  cblas_zscal(N, &alpha, X, incX);
}

template <>
inline cblas_scal_f<std::complex<float>> cblas_scal<std::complex<float>> =
    &cblas_cscal_wrap;

template <>
inline cblas_scal_f<std::complex<double>> cblas_scal<std::complex<double>> =
    &cblas_zscal_wrap;
// end of cblas_scal

} // end namespace perflibs::sparse
