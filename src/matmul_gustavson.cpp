/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "matmul_gustavson.hpp"
#include "spec.hpp"
#include <arm_neon.h>

namespace perflibs::sparse {

// Default accumulator
template <typename T>
void spmm_csr_mm_gustavson_inner(T valA, perflibs_sparse_hint_value transB,
                                 perflibs_int_t index_base, perflibs_int_t rB,
                                 const perflibs_int_t *row_ptrB,
                                 const perflibs_int_t *col_indxB,
                                 const T *valsB, T *x, perflibs_int_t *xbool,
                                 perflibs_int_t &i_row_ptr_C,
                                 perflibs_int_t *col_indxC, perflibs_int_t i) {

  auto conjB = [&](T val) {
    return transB == PERFLIBS_SPARSE_OPERATION_CONJTRANS
               ? perflibs::sparse::conj(val)
               : val;
  };

  for (auto j = row_ptrB[rB] - index_base; j < row_ptrB[rB + 1] - index_base;
       j++) {
    auto cB =
        col_indxB[j] -
        index_base; // The column of B (and therefore col of C) we're looking at
    x[cB] += valA * conjB(valsB[j]);
    if (xbool[cB] != i) {
      xbool[cB] = i;
      col_indxC[i_row_ptr_C++] = cB + index_base;
    }
  }
}
template void spmm_csr_mm_gustavson_inner<float>(
    float valA, perflibs_sparse_hint_value transB, perflibs_int_t index_base,
    perflibs_int_t rB, const perflibs_int_t *row_ptrB,
    const perflibs_int_t *col_indxB, const float *valsB, float *x,
    perflibs_int_t *xbool, perflibs_int_t &i_row_ptr_C,
    perflibs_int_t *col_indxC, perflibs_int_t i);
template void spmm_csr_mm_gustavson_inner<double>(
    double valA, perflibs_sparse_hint_value transB, perflibs_int_t index_base,
    perflibs_int_t rB, const perflibs_int_t *row_ptrB,
    const perflibs_int_t *col_indxB, const double *valsB, double *x,
    perflibs_int_t *xbool, perflibs_int_t &i_row_ptr_C,
    perflibs_int_t *col_indxC, perflibs_int_t i);
template void spmm_csr_mm_gustavson_inner<std::complex<float>>(
    std::complex<float> valA, perflibs_sparse_hint_value transB,
    perflibs_int_t index_base, perflibs_int_t rB,
    const perflibs_int_t *row_ptrB, const perflibs_int_t *col_indxB,
    const std::complex<float> *valsB, std::complex<float> *x,
    perflibs_int_t *xbool, perflibs_int_t &i_row_ptr_C,
    perflibs_int_t *col_indxC, perflibs_int_t i);
template void spmm_csr_mm_gustavson_inner<std::complex<double>>(
    std::complex<double> valA, perflibs_sparse_hint_value transB,
    perflibs_int_t index_base, perflibs_int_t rB,
    const perflibs_int_t *row_ptrB, const perflibs_int_t *col_indxB,
    const std::complex<double> *valsB, std::complex<double> *x,
    perflibs_int_t *xbool, perflibs_int_t &i_row_ptr_C,
    perflibs_int_t *col_indxC, perflibs_int_t i);

template <typename T>
void spmm_csr_gustavson(enum perflibs_sparse_hint_value transA,
                        enum perflibs_sparse_hint_value transB,
                        perflibs_int_t m, perflibs_int_t n, T alpha,
                        const perflibs_int_t *row_ptrA,
                        const perflibs_int_t *col_indxA, const T *valsA,
                        const perflibs_int_t *row_ptrB,
                        const perflibs_int_t *col_indxB, const T *valsB,
                        std::vector<perflibs_int_t> &row_ptrC,
                        std::vector<perflibs_int_t> &col_indxC,
                        std::vector<T> &valsC) {

  auto kernel = get_gustavson_kernel(transA, transB, m, n, alpha);

  auto index_base = row_ptrA[0];
  assert(row_ptrB[0] == index_base);
  row_ptrC[0] = index_base;

  auto conjA = [&](T val) {
    return transA == PERFLIBS_SPARSE_OPERATION_CONJTRANS
               ? perflibs::sparse::conj(val)
               : val;
  };

  valsC.resize(2 * n);
  col_indxC.resize(2 * n);

  // Gustavson's algorithm

  perflibs_int_t i_row_ptr_C =
      0; // Pointer to where we're up to in the NNZ-length arrays; this is
         // effectively row_ptrC[i+1] - index_base
  // xbool is used to decide whether we've seen the current column in the
  // current row. Instead of resetting, the row number is written.
  std::vector<perflibs_int_t> xbool(n, -1);
  // x is the dense accumulator
  std::vector<T> x(n);
  for (perflibs_int_t i = 0; i < m; i++) {
    row_ptrC[i] = i_row_ptr_C + index_base;
    for (auto k = row_ptrA[i] - index_base; k < row_ptrA[i + 1] - index_base;
         k++) {
      auto rB = col_indxA[k] - index_base; // The row of B we're looking at
      // Accumulator
      kernel(conjA(valsA[k]), transB, index_base, rB, row_ptrB, col_indxB,
             valsB, x.data(), xbool.data(), i_row_ptr_C, col_indxC.data(), i);
    }
    if (valsC.size() <= (size_t)i_row_ptr_C + n) {
      valsC.resize(i_row_ptr_C + n);
    }
    if (col_indxC.size() <= (size_t)i_row_ptr_C + n) {
      col_indxC.resize(i_row_ptr_C + n);
    }
    for (perflibs_int_t j = row_ptrC[i] - index_base; j < i_row_ptr_C; j++) {
      valsC[j] = alpha * x[col_indxC[j] - index_base];
      // Reset so that we can always do fmla into x in the inner loop
      x[col_indxC[j] - index_base] = (T)0;
    }
  }
  row_ptrC[m] = i_row_ptr_C + index_base;
}

template void spmm_csr_gustavson<float>(
    enum perflibs_sparse_hint_value transA,
    enum perflibs_sparse_hint_value transB, perflibs_int_t m, perflibs_int_t n,
    float alpha, const perflibs_int_t *row_ptrA,
    const perflibs_int_t *col_indxA, const float *valsA,
    const perflibs_int_t *row_ptrB, const perflibs_int_t *col_indxB,
    const float *valsB, std::vector<perflibs_int_t> &row_ptrC,
    std::vector<perflibs_int_t> &col_indxC, std::vector<float> &valsC);
template void spmm_csr_gustavson<double>(
    enum perflibs_sparse_hint_value transA,
    enum perflibs_sparse_hint_value transB, perflibs_int_t m, perflibs_int_t n,
    double alpha, const perflibs_int_t *row_ptrA,
    const perflibs_int_t *col_indxA, const double *valsA,
    const perflibs_int_t *row_ptrB, const perflibs_int_t *col_indxB,
    const double *valsB, std::vector<perflibs_int_t> &row_ptrC,
    std::vector<perflibs_int_t> &col_indxC, std::vector<double> &valsC);
template void spmm_csr_gustavson<std::complex<float>>(
    enum perflibs_sparse_hint_value transA,
    enum perflibs_sparse_hint_value transB, perflibs_int_t m, perflibs_int_t n,
    std::complex<float> alpha, const perflibs_int_t *row_ptrA,
    const perflibs_int_t *col_indxA, const std::complex<float> *valsA,
    const perflibs_int_t *row_ptrB, const perflibs_int_t *col_indxB,
    const std::complex<float> *valsB, std::vector<perflibs_int_t> &row_ptrC,
    std::vector<perflibs_int_t> &col_indxC,
    std::vector<std::complex<float>> &valsC);
template void spmm_csr_gustavson<std::complex<double>>(
    enum perflibs_sparse_hint_value transA,
    enum perflibs_sparse_hint_value transB, perflibs_int_t m, perflibs_int_t n,
    std::complex<double> alpha, const perflibs_int_t *row_ptrA,
    const perflibs_int_t *col_indxA, const std::complex<double> *valsA,
    const perflibs_int_t *row_ptrB, const perflibs_int_t *col_indxB,
    const std::complex<double> *valsB, std::vector<perflibs_int_t> &row_ptrC,
    std::vector<perflibs_int_t> &col_indxC,
    std::vector<std::complex<double>> &valsC);

template <typename T>
void spmm_csr_mm_gustavson_inner_vals_only(
    T valA, perflibs_sparse_hint_value transB, perflibs_int_t index_base,
    perflibs_int_t rB, const perflibs_int_t *row_ptrB,
    const perflibs_int_t *col_indxB, const T *valsB, T *x) {

  auto conjB = [&](T val) {
    return transB == PERFLIBS_SPARSE_OPERATION_CONJTRANS
               ? perflibs::sparse::conj(val)
               : val;
  };

  for (auto j = row_ptrB[rB] - index_base; j < row_ptrB[rB + 1] - index_base;
       j++) {
    auto cB =
        col_indxB[j] -
        index_base; // The column of B (and therefore col of C) we're looking at
    x[cB] += valA * conjB(valsB[j]);
  }
}
template void spmm_csr_mm_gustavson_inner_vals_only<float>(
    float valA, perflibs_sparse_hint_value transB, perflibs_int_t index_base,
    perflibs_int_t rB, const perflibs_int_t *row_ptrB,
    const perflibs_int_t *col_indxB, const float *valsB, float *x);
template void spmm_csr_mm_gustavson_inner_vals_only<double>(
    double valA, perflibs_sparse_hint_value transB, perflibs_int_t index_base,
    perflibs_int_t rB, const perflibs_int_t *row_ptrB,
    const perflibs_int_t *col_indxB, const double *valsB, double *x);
template void spmm_csr_mm_gustavson_inner_vals_only<std::complex<float>>(
    std::complex<float> valA, perflibs_sparse_hint_value transB,
    perflibs_int_t index_base, perflibs_int_t rB,
    const perflibs_int_t *row_ptrB, const perflibs_int_t *col_indxB,
    const std::complex<float> *valsB, std::complex<float> *x);
template void spmm_csr_mm_gustavson_inner_vals_only<std::complex<double>>(
    std::complex<double> valA, perflibs_sparse_hint_value transB,
    perflibs_int_t index_base, perflibs_int_t rB,
    const perflibs_int_t *row_ptrB, const perflibs_int_t *col_indxB,
    const std::complex<double> *valsB, std::complex<double> *x);

template <typename T>
void spmm_csr_gustavson_noalloc_vals_only(
    enum perflibs_sparse_hint_value transA,
    enum perflibs_sparse_hint_value transB, perflibs_int_t m, perflibs_int_t n,
    T alpha, const perflibs_int_t *row_ptrA, const perflibs_int_t *col_indxA,
    const T *valsA, const perflibs_int_t *row_ptrB,
    const perflibs_int_t *col_indxB, const T *valsB,
    const perflibs_int_t *row_ptrC, const perflibs_int_t *col_indxC, T *valsC,
    sparse_matrix_statistics stats) {

  auto kernel = get_gustavson_kernel_vals_only(transA, transB, m, n, alpha,
                                               std::move(stats));

  auto index_base = row_ptrA[0];
  assert(row_ptrB[0] == index_base);
  assert(row_ptrC[0] == index_base);

  auto conjA = [&](T val) {
    return transA == PERFLIBS_SPARSE_OPERATION_CONJTRANS
               ? perflibs::sparse::conj(val)
               : val;
  };

// Gustavson's algorithm

// x is the dense accumulator
#pragma omp parallel
  {
    std::vector<T> x(n);
#pragma omp for schedule(static)
    for (perflibs_int_t i = 0; i < m; i++) {
      for (auto k = row_ptrA[i] - index_base; k < row_ptrA[i + 1] - index_base;
           k++) {
        auto rB = col_indxA[k] - index_base; // The row of B we're looking at
        // Accumulator
        kernel(conjA(valsA[k]), transB, index_base, rB, row_ptrB, col_indxB,
               valsB, x.data());
      }
      for (perflibs_int_t j = row_ptrC[i] - index_base;
           j < row_ptrC[i + 1] - index_base; j++) {
        valsC[j] = alpha * x[col_indxC[j] - index_base];
        // Reset so that we can always do fmla into x in the inner loop
        x[col_indxC[j] - index_base] = (T)0;
      }
    }
  }
}

template void spmm_csr_gustavson_noalloc_vals_only<float>(
    enum perflibs_sparse_hint_value transA,
    enum perflibs_sparse_hint_value transB, perflibs_int_t m, perflibs_int_t n,
    float alpha, const perflibs_int_t *row_ptrA,
    const perflibs_int_t *col_indxA, const float *valsA,
    const perflibs_int_t *row_ptrB, const perflibs_int_t *col_indxB,
    const float *valsB, const perflibs_int_t *row_ptrC,
    const perflibs_int_t *col_indxC, float *valsC,
    sparse_matrix_statistics stats);
template void spmm_csr_gustavson_noalloc_vals_only<double>(
    enum perflibs_sparse_hint_value transA,
    enum perflibs_sparse_hint_value transB, perflibs_int_t m, perflibs_int_t n,
    double alpha, const perflibs_int_t *row_ptrA,
    const perflibs_int_t *col_indxA, const double *valsA,
    const perflibs_int_t *row_ptrB, const perflibs_int_t *col_indxB,
    const double *valsB, const perflibs_int_t *row_ptrC,
    const perflibs_int_t *col_indxC, double *valsC,
    sparse_matrix_statistics stats);
template void spmm_csr_gustavson_noalloc_vals_only<std::complex<float>>(
    enum perflibs_sparse_hint_value transA,
    enum perflibs_sparse_hint_value transB, perflibs_int_t m, perflibs_int_t n,
    std::complex<float> alpha, const perflibs_int_t *row_ptrA,
    const perflibs_int_t *col_indxA, const std::complex<float> *valsA,
    const perflibs_int_t *row_ptrB, const perflibs_int_t *col_indxB,
    const std::complex<float> *valsB, const perflibs_int_t *row_ptrC,
    const perflibs_int_t *col_indxC, std::complex<float> *valsC,
    sparse_matrix_statistics stats);
template void spmm_csr_gustavson_noalloc_vals_only<std::complex<double>>(
    enum perflibs_sparse_hint_value transA,
    enum perflibs_sparse_hint_value transB, perflibs_int_t m, perflibs_int_t n,
    std::complex<double> alpha, const perflibs_int_t *row_ptrA,
    const perflibs_int_t *col_indxA, const std::complex<double> *valsA,
    const perflibs_int_t *row_ptrB, const perflibs_int_t *col_indxB,
    const std::complex<double> *valsB, const perflibs_int_t *row_ptrC,
    const perflibs_int_t *col_indxC, std::complex<double> *valsC,
    sparse_matrix_statistics stats);

template <typename T>
void spmm_csr_gustavson_noalloc(enum perflibs_sparse_hint_value transA,
                                enum perflibs_sparse_hint_value transB,
                                perflibs_int_t m, perflibs_int_t n, T alpha,
                                const perflibs_int_t *row_ptrA,
                                const perflibs_int_t *col_indxA, const T *valsA,
                                const perflibs_int_t *row_ptrB,
                                const perflibs_int_t *col_indxB, const T *valsB,
                                const perflibs_int_t *row_ptrC,
                                perflibs_int_t *col_indxC, T *valsC) {

  // This will be the same kernel as the serial function spmm_csr_gustavson
  auto kernel = get_gustavson_kernel(transA, transB, m, n, alpha);

  auto index_base = row_ptrA[0];
  assert(row_ptrB[0] == index_base);
  assert(row_ptrC[0] == index_base);

  auto conjA = [&](T val) {
    return transA == PERFLIBS_SPARSE_OPERATION_CONJTRANS
               ? perflibs::sparse::conj(val)
               : val;
  };

// Gustavson's algorithm

// x is the dense accumulator
#pragma omp parallel
  {
    std::vector<T> x(n);
    std::vector<perflibs_int_t> xbool(n, -1);
#pragma omp for schedule(static)
    for (perflibs_int_t i = 0; i < m; i++) {
      perflibs_int_t i_row_ptr_C =
          row_ptrC[i] -
          index_base; // Pointer to where we're up to in the NNZ-length arrays
      for (auto k = row_ptrA[i] - index_base; k < row_ptrA[i + 1] - index_base;
           k++) {
        auto rB = col_indxA[k] - index_base; // The row of B we're looking at
        // Accumulator
        kernel(conjA(valsA[k]), transB, index_base, rB, row_ptrB, col_indxB,
               valsB, x.data(), xbool.data(), i_row_ptr_C, col_indxC, i);
      }
      for (perflibs_int_t j = row_ptrC[i] - index_base;
           j < row_ptrC[i + 1] - index_base; j++) {
        valsC[j] = alpha * x[col_indxC[j] - index_base];
        // Reset so that we can always do fmla into x in the inner loop
        x[col_indxC[j] - index_base] = (T)0;
      }
    }
  }
}

template void spmm_csr_gustavson_noalloc<float>(
    enum perflibs_sparse_hint_value transA,
    enum perflibs_sparse_hint_value transB, perflibs_int_t m, perflibs_int_t n,
    float alpha, const perflibs_int_t *row_ptrA,
    const perflibs_int_t *col_indxA, const float *valsA,
    const perflibs_int_t *row_ptrB, const perflibs_int_t *col_indxB,
    const float *valsB, const perflibs_int_t *row_ptrC,
    perflibs_int_t *col_indxC, float *valsC);
template void spmm_csr_gustavson_noalloc<double>(
    enum perflibs_sparse_hint_value transA,
    enum perflibs_sparse_hint_value transB, perflibs_int_t m, perflibs_int_t n,
    double alpha, const perflibs_int_t *row_ptrA,
    const perflibs_int_t *col_indxA, const double *valsA,
    const perflibs_int_t *row_ptrB, const perflibs_int_t *col_indxB,
    const double *valsB, const perflibs_int_t *row_ptrC,
    perflibs_int_t *col_indxC, double *valsC);
template void spmm_csr_gustavson_noalloc<std::complex<float>>(
    enum perflibs_sparse_hint_value transA,
    enum perflibs_sparse_hint_value transB, perflibs_int_t m, perflibs_int_t n,
    std::complex<float> alpha, const perflibs_int_t *row_ptrA,
    const perflibs_int_t *col_indxA, const std::complex<float> *valsA,
    const perflibs_int_t *row_ptrB, const perflibs_int_t *col_indxB,
    const std::complex<float> *valsB, const perflibs_int_t *row_ptrC,
    perflibs_int_t *col_indxC, std::complex<float> *valsC);
template void spmm_csr_gustavson_noalloc<std::complex<double>>(
    enum perflibs_sparse_hint_value transA,
    enum perflibs_sparse_hint_value transB, perflibs_int_t m, perflibs_int_t n,
    std::complex<double> alpha, const perflibs_int_t *row_ptrA,
    const perflibs_int_t *col_indxA, const std::complex<double> *valsA,
    const perflibs_int_t *row_ptrB, const perflibs_int_t *col_indxB,
    const std::complex<double> *valsB, const perflibs_int_t *row_ptrC,
    perflibs_int_t *col_indxC, std::complex<double> *valsC);

// By default the neon version calls the generic one
template <typename T>
void spmm_csr_mm_gustavson_inner_neon(
    T valA, perflibs_sparse_hint_value transB, perflibs_int_t index_base,
    perflibs_int_t rB, const perflibs_int_t *row_ptrB,
    const perflibs_int_t *col_indxB, const T *valsB, T *x,
    perflibs_int_t *xbool, perflibs_int_t &i_row_ptr_C,
    perflibs_int_t *col_indxC, perflibs_int_t i) {
  spmm_csr_mm_gustavson_inner<T>(valA, transB, index_base, rB, row_ptrB,
                                 col_indxB, valsB, x, xbool, i_row_ptr_C,
                                 col_indxC, i);
}

template void spmm_csr_mm_gustavson_inner_neon<float>(
    float valA, perflibs_sparse_hint_value transB, perflibs_int_t index_base,
    perflibs_int_t rB, const perflibs_int_t *row_ptrB,
    const perflibs_int_t *col_indxB, const float *valsB, float *x,
    perflibs_int_t *xbool, perflibs_int_t &i_row_ptr_C,
    perflibs_int_t *col_indxC, perflibs_int_t i);
template void spmm_csr_mm_gustavson_inner_neon<std::complex<float>>(
    std::complex<float> valA, perflibs_sparse_hint_value transB,
    perflibs_int_t index_base, perflibs_int_t rB,
    const perflibs_int_t *row_ptrB, const perflibs_int_t *col_indxB,
    const std::complex<float> *valsB, std::complex<float> *x,
    perflibs_int_t *xbool, perflibs_int_t &i_row_ptr_C,
    perflibs_int_t *col_indxC, perflibs_int_t i);
template void spmm_csr_mm_gustavson_inner_neon<std::complex<double>>(
    std::complex<double> valA, perflibs_sparse_hint_value transB,
    perflibs_int_t index_base, perflibs_int_t rB,
    const perflibs_int_t *row_ptrB, const perflibs_int_t *col_indxB,
    const std::complex<double> *valsB, std::complex<double> *x,
    perflibs_int_t *xbool, perflibs_int_t &i_row_ptr_C,
    perflibs_int_t *col_indxC, perflibs_int_t i);

template <>
void spmm_csr_mm_gustavson_inner_neon(
    double valA, perflibs_sparse_hint_value transB, perflibs_int_t index_base,
    perflibs_int_t rB, const perflibs_int_t *row_ptrB,
    const perflibs_int_t *col_indxB, const double *valsB, double *x,
    perflibs_int_t *xbool, perflibs_int_t &i_row_ptr_C,
    perflibs_int_t *col_indxC, perflibs_int_t i) {
  float64x1_t vA = vld1_f64(&valA);

  constexpr int unrl = 2;
  auto j = row_ptrB[rB] - index_base;
  for (; j < row_ptrB[rB + 1] - index_base - (unrl - 1); j += unrl) {
    auto cB0 =
        col_indxB[j] -
        index_base; // The column of B (and therefore col of C) we're looking at
    auto cB1 = col_indxB[j + 1] - index_base;

    // x[cB0] += valA*valsB[j    ];
    // x[cB1] += valA*valsB[j + 1];
    float64x2_t vB = vld1q_f64(&valsB[j]);
    float64x2_t vC = {x[cB0], x[cB1]};
    vC = vfmaq_lane_f64(vC, vB, vA, 0);
    vst1q_lane_f64(&x[cB0], vC, 0);
    vst1q_lane_f64(&x[cB1], vC, 1);

    if (xbool[cB0] != i) {
      xbool[cB0] = i;
      col_indxC[i_row_ptr_C++] = cB0 + index_base;
    }
    if (xbool[cB1] != i) {
      xbool[cB1] = i;
      col_indxC[i_row_ptr_C++] = cB1 + index_base;
    }
  }

  // Tidy-up loop
  for (; j < row_ptrB[rB + 1] - index_base; j++) {
    auto cB = col_indxB[j] - index_base;
    x[cB] += valA * valsB[j];
    if (xbool[cB] != i) {
      xbool[cB] = i;
      col_indxC[i_row_ptr_C++] = cB + index_base;
    }
  }
}

template <typename T>
void spmm_csr_mm_gustavson_inner_vals_only_neon(
    T valA, perflibs_sparse_hint_value transB, perflibs_int_t index_base,
    perflibs_int_t rB, const perflibs_int_t *row_ptrB,
    const perflibs_int_t *col_indxB, const T *valsB, T *x) {
  spmm_csr_mm_gustavson_inner_vals_only<T>(valA, transB, index_base, rB,
                                           row_ptrB, col_indxB, valsB, x);
}

template void spmm_csr_mm_gustavson_inner_vals_only_neon<float>(
    float valA, perflibs_sparse_hint_value transB, perflibs_int_t index_base,
    perflibs_int_t rB, const perflibs_int_t *row_ptrB,
    const perflibs_int_t *col_indxB, const float *valsB, float *x);
template void spmm_csr_mm_gustavson_inner_vals_only_neon<std::complex<float>>(
    std::complex<float> valA, perflibs_sparse_hint_value transB,
    perflibs_int_t index_base, perflibs_int_t rB,
    const perflibs_int_t *row_ptrB, const perflibs_int_t *col_indxB,
    const std::complex<float> *valsB, std::complex<float> *x);
template void spmm_csr_mm_gustavson_inner_vals_only_neon<std::complex<double>>(
    std::complex<double> valA, perflibs_sparse_hint_value transB,
    perflibs_int_t index_base, perflibs_int_t rB,
    const perflibs_int_t *row_ptrB, const perflibs_int_t *col_indxB,
    const std::complex<double> *valsB, std::complex<double> *x);
template <>
void spmm_csr_mm_gustavson_inner_vals_only_neon(
    double valA, perflibs_sparse_hint_value transB, perflibs_int_t index_base,
    perflibs_int_t rB, const perflibs_int_t *row_ptrB,
    const perflibs_int_t *col_indxB, const double *valsB, double *x) {
  float64x1_t vA = vld1_f64(&valA);

  constexpr int unrl = 2;
  auto j = row_ptrB[rB] - index_base;
  for (; j < row_ptrB[rB + 1] - index_base - (unrl - 1); j += unrl) {
    auto cB0 =
        col_indxB[j] -
        index_base; // The column of B (and therefore col of C) we're looking at
    auto cB1 = col_indxB[j + 1] - index_base;

    float64x2_t vB = vld1q_f64(&valsB[j]);
    float64x2_t vC = {x[cB0], x[cB1]};
    vC = vfmaq_lane_f64(vC, vB, vA, 0);
    vst1q_lane_f64(&x[cB0], vC, 0);
    vst1q_lane_f64(&x[cB1], vC, 1);
  }

  // Tidy-up loop
  for (; j < row_ptrB[rB + 1] - index_base; j++) {
    auto cB = col_indxB[j] - index_base;
    x[cB] += valA * valsB[j];
  }
}

template <typename T>
gs_kernel_t<T> get_gustavson_kernel_default(perflibs_sparse_hint_value transA,
                                            perflibs_sparse_hint_value transB,
                                            perflibs_int_t m, perflibs_int_t n,
                                            T alpha) {
  return &spmm_csr_mm_gustavson_inner_neon<T>;
}
template gs_kernel_t<float> get_gustavson_kernel_default<float>(
    perflibs_sparse_hint_value transA, perflibs_sparse_hint_value transB,
    perflibs_int_t m, perflibs_int_t n, float alpha);
template gs_kernel_t<double> get_gustavson_kernel_default<double>(
    perflibs_sparse_hint_value transA, perflibs_sparse_hint_value transB,
    perflibs_int_t m, perflibs_int_t n, double alpha);
template gs_kernel_t<std::complex<float>>
get_gustavson_kernel_default<std::complex<float>>(
    perflibs_sparse_hint_value transA, perflibs_sparse_hint_value transB,
    perflibs_int_t m, perflibs_int_t n, std::complex<float> alpha);
template gs_kernel_t<std::complex<double>>
get_gustavson_kernel_default<std::complex<double>>(
    perflibs_sparse_hint_value transA, perflibs_sparse_hint_value transB,
    perflibs_int_t m, perflibs_int_t n, std::complex<double> alpha);

template <typename T>
gs_na_kernel_t<T> get_gustavson_kernel_vals_only_default(
    perflibs_sparse_hint_value transA, perflibs_sparse_hint_value transB,
    perflibs_int_t m, perflibs_int_t n, T alpha,
    sparse_matrix_statistics stats) {
  return &spmm_csr_mm_gustavson_inner_vals_only_neon<T>;
}
template gs_na_kernel_t<float> get_gustavson_kernel_vals_only_default<float>(
    perflibs_sparse_hint_value transA, perflibs_sparse_hint_value transB,
    perflibs_int_t m, perflibs_int_t n, float alpha,
    sparse_matrix_statistics stats);
template gs_na_kernel_t<double> get_gustavson_kernel_vals_only_default<double>(
    perflibs_sparse_hint_value transA, perflibs_sparse_hint_value transB,
    perflibs_int_t m, perflibs_int_t n, double alpha,
    sparse_matrix_statistics stats);
template gs_na_kernel_t<std::complex<float>>
get_gustavson_kernel_vals_only_default<std::complex<float>>(
    perflibs_sparse_hint_value transA, perflibs_sparse_hint_value transB,
    perflibs_int_t m, perflibs_int_t n, std::complex<float> alpha,
    sparse_matrix_statistics stats);
template gs_na_kernel_t<std::complex<double>>
get_gustavson_kernel_vals_only_default<std::complex<double>>(
    perflibs_sparse_hint_value transA, perflibs_sparse_hint_value transB,
    perflibs_int_t m, perflibs_int_t n, std::complex<double> alpha,
    sparse_matrix_statistics stats);

} // end namespace perflibs::sparse
