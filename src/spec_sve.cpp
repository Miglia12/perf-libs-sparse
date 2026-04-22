/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

// This file must be compiled in such a way that enables SVE, e.g.
// `-march=armv8-a+sve`
#include <arm_sve.h>

#include "matmul_gustavson.hpp"
#include "sell_c_sigma.hpp"
#include "spec_sve.hpp"

namespace perflibs::sparse {

inline int vector_size_bytes_sve() { return svcntb(); }

template <typename T> int vector_size_elems_sve() {
  return vector_size_bytes_sve() / sizeof(T);
}

template <typename T>
void spmm_csr_mm_gustavson_inner_sve(
    T valA, perflibs_sparse_hint_value transB, perflibs_int_t index_base,
    perflibs_int_t rB, const perflibs_int_t *row_ptrB,
    const perflibs_int_t *col_indxB, const T *valsB, T *x,
    perflibs_int_t *xbool, perflibs_int_t &i_row_ptr_C,
    perflibs_int_t *col_indxC, perflibs_int_t i) {
  spmm_csr_mm_gustavson_inner<T>(valA, transB, index_base, rB, row_ptrB,
                                 col_indxB, valsB, x, xbool, i_row_ptr_C,
                                 col_indxC, i);
}

template void spmm_csr_mm_gustavson_inner_sve<float>(
    float valA, perflibs_sparse_hint_value transB, perflibs_int_t index_base,
    perflibs_int_t rB, const perflibs_int_t *row_ptrB,
    const perflibs_int_t *col_indxB, const float *valsB, float *x,
    perflibs_int_t *xbool, perflibs_int_t &i_row_ptr_C,
    perflibs_int_t *col_indxC, perflibs_int_t i);
template void spmm_csr_mm_gustavson_inner_sve<double>(
    double valA, perflibs_sparse_hint_value transB, perflibs_int_t index_base,
    perflibs_int_t rB, const perflibs_int_t *row_ptrB,
    const perflibs_int_t *col_indxB, const double *valsB, double *x,
    perflibs_int_t *xbool, perflibs_int_t &i_row_ptr_C,
    perflibs_int_t *col_indxC, perflibs_int_t i);
template void spmm_csr_mm_gustavson_inner_sve<std::complex<float>>(
    std::complex<float> valA, perflibs_sparse_hint_value transB,
    perflibs_int_t index_base, perflibs_int_t rB,
    const perflibs_int_t *row_ptrB, const perflibs_int_t *col_indxB,
    const std::complex<float> *valsB, std::complex<float> *x,
    perflibs_int_t *xbool, perflibs_int_t &i_row_ptr_C,
    perflibs_int_t *col_indxC, perflibs_int_t i);
template void spmm_csr_mm_gustavson_inner_sve<std::complex<double>>(
    std::complex<double> valA, perflibs_sparse_hint_value transB,
    perflibs_int_t index_base, perflibs_int_t rB,
    const perflibs_int_t *row_ptrB, const perflibs_int_t *col_indxB,
    const std::complex<double> *valsB, std::complex<double> *x,
    perflibs_int_t *xbool, perflibs_int_t &i_row_ptr_C,
    perflibs_int_t *col_indxC, perflibs_int_t i);

template <typename T>
void spmm_csr_mm_gustavson_inner_vals_only_sve(
    T valA, perflibs_sparse_hint_value transB, perflibs_int_t index_base,
    perflibs_int_t rB, const perflibs_int_t *row_ptrB,
    const perflibs_int_t *col_indxB, const T *valsB, T *x) {
  spmm_csr_mm_gustavson_inner_vals_only<T>(valA, transB, index_base, rB,
                                           row_ptrB, col_indxB, valsB, x);
}

template void spmm_csr_mm_gustavson_inner_vals_only_sve<float>(
    float valA, perflibs_sparse_hint_value transB, perflibs_int_t index_base,
    perflibs_int_t rB, const perflibs_int_t *row_ptrB,
    const perflibs_int_t *col_indxB, const float *valsB, float *x);
template void spmm_csr_mm_gustavson_inner_vals_only_sve<double>(
    double valA, perflibs_sparse_hint_value transB, perflibs_int_t index_base,
    perflibs_int_t rB, const perflibs_int_t *row_ptrB,
    const perflibs_int_t *col_indxB, const double *valsB, double *x);
template void spmm_csr_mm_gustavson_inner_vals_only_sve<std::complex<float>>(
    std::complex<float> valA, perflibs_sparse_hint_value transB,
    perflibs_int_t index_base, perflibs_int_t rB,
    const perflibs_int_t *row_ptrB, const perflibs_int_t *col_indxB,
    const std::complex<float> *valsB, std::complex<float> *x);
template void spmm_csr_mm_gustavson_inner_vals_only_sve<std::complex<double>>(
    std::complex<double> valA, perflibs_sparse_hint_value transB,
    perflibs_int_t index_base, perflibs_int_t rB,
    const perflibs_int_t *row_ptrB, const perflibs_int_t *col_indxB,
    const std::complex<double> *valsB, std::complex<double> *x);

inline svint64_t get_indices(svbool_t pred, const int32_t *col_indx) {
  // Load 32-bit indices into zero-extended 64-bit vector for lp64 lib
  return svld1sw_s64(pred, col_indx);
}

inline svint64_t get_indices(svbool_t pred, const int64_t *col_indx) {
  // Straight load of 64-bit indices for ilp64 lib
  return svld1_s64(pred, col_indx);
}

void spmm_csr_mm_gustavson_inner_vals_only_sve_acle(
    double valA, perflibs_sparse_hint_value transB, perflibs_int_t index_base,
    perflibs_int_t rB, const perflibs_int_t *row_ptrB,
    const perflibs_int_t *col_indxB, const double *valsB, double *x) {

  auto j = row_ptrB[rB] - index_base;
  auto end = row_ptrB[rB + 1] - index_base;
  auto xp = x - index_base;
  svbool_t pred = svwhilelt_b64(j, end);
  while (svptest_any(svptrue_b64(), pred)) {
    // Contiguous load of B
    svfloat64_t vB = svld1_f64(pred, &valsB[j]);
    // Handle index loads differently for lp64, ilp64
    svint64_t cB = get_indices(pred, &col_indxB[j]);
    // Gather load of x into vC
    svfloat64_t vC = svld1_gather_s64index_f64(pred, xp, cB);
    // FMLA: vC += vB*valA
    vC = svmla_n_f64_m(pred, vC, vB, valA);
    // Scatter store of vC to x
    svst1_scatter_s64index_f64(pred, xp, cB, vC);
    // Increment loop counter
    j += svcntd();
    // Update predicate
    pred = svwhilelt_b64(j, end);
  }
}

template <typename T> bool get_if_mgmd_is_required_sve() { return false; }

template bool get_if_mgmd_is_required_sve<float>();
template bool get_if_mgmd_is_required_sve<std::complex<float>>();
template bool get_if_mgmd_is_required_sve<std::complex<double>>();

// This is the only case where we currently have an SVE vector kernel
template <> bool get_if_mgmd_is_required_sve<double>() { return true; }

template <typename T>
gs_kernel_t<T> get_gustavson_kernel_sve(perflibs_sparse_hint_value transA,
                                        perflibs_sparse_hint_value transB,
                                        perflibs_int_t m, perflibs_int_t n,
                                        T alpha) {
  return &spmm_csr_mm_gustavson_inner_sve<T>;
}
template gs_kernel_t<float> get_gustavson_kernel_sve<float>(
    perflibs_sparse_hint_value transA, perflibs_sparse_hint_value transB,
    perflibs_int_t m, perflibs_int_t n, float alpha);
template gs_kernel_t<double> get_gustavson_kernel_sve<double>(
    perflibs_sparse_hint_value transA, perflibs_sparse_hint_value transB,
    perflibs_int_t m, perflibs_int_t n, double alpha);
template gs_kernel_t<std::complex<float>>
get_gustavson_kernel_sve<std::complex<float>>(perflibs_sparse_hint_value transA,
                                              perflibs_sparse_hint_value transB,
                                              perflibs_int_t m,
                                              perflibs_int_t n,
                                              std::complex<float> alpha);
template gs_kernel_t<std::complex<double>>
get_gustavson_kernel_sve<std::complex<double>>(
    perflibs_sparse_hint_value transA, perflibs_sparse_hint_value transB,
    perflibs_int_t m, perflibs_int_t n, std::complex<double> alpha);

template <typename T>
gs_na_kernel_t<T>
get_gustavson_kernel_vals_only_sve(perflibs_sparse_hint_value transA,
                                   perflibs_sparse_hint_value transB,
                                   perflibs_int_t m, perflibs_int_t n, T alpha,
                                   sparse_matrix_statistics stats) {
  return &spmm_csr_mm_gustavson_inner_vals_only_sve<T>;
}
template gs_na_kernel_t<float> get_gustavson_kernel_vals_only_sve<float>(
    perflibs_sparse_hint_value transA, perflibs_sparse_hint_value transB,
    perflibs_int_t m, perflibs_int_t n, float alpha,
    sparse_matrix_statistics stats);
template gs_na_kernel_t<std::complex<float>>
get_gustavson_kernel_vals_only_sve<std::complex<float>>(
    perflibs_sparse_hint_value transA, perflibs_sparse_hint_value transB,
    perflibs_int_t m, perflibs_int_t n, std::complex<float> alpha,
    sparse_matrix_statistics stats);
template gs_na_kernel_t<std::complex<double>>
get_gustavson_kernel_vals_only_sve<std::complex<double>>(
    perflibs_sparse_hint_value transA, perflibs_sparse_hint_value transB,
    perflibs_int_t m, perflibs_int_t n, std::complex<double> alpha,
    sparse_matrix_statistics stats);

/// We have an ACLE option for double, real. It only pays to use this for
/// certain matrix types.
template <>
gs_na_kernel_t<double> get_gustavson_kernel_vals_only_sve(
    perflibs_sparse_hint_value transA, perflibs_sparse_hint_value transB,
    perflibs_int_t m, perflibs_int_t n, double alpha,
    sparse_matrix_statistics stats) {
  // mgmd is the mean value over rows of geomean diffs of column indices within
  // in each row. Profiling data shows that for values < 60 (i.e.
  // where we have tight clusters of non-zeros in rows) it pays off to use the
  // SVE vector kernel, otherwise just stick to the 'sve' option, which is
  // currently scalar compiled code.
  return stats.mgmd < 60.0 && stats.mgmd > 0.0
             ? &spmm_csr_mm_gustavson_inner_vals_only_sve_acle
             : &spmm_csr_mm_gustavson_inner_vals_only_sve<double>;
}

} // end namespace perflibs::sparse

extern "C" void d_spmv_scs_kernel_sve_C1(
    int64_t nchunks, const double *vals, const perflibs_int_t *col_indx,
    const double *x, double *y, const int64_t *cs, const int64_t *cl,
    const int64_t *row_permd2in, int64_t col_indx_bytes,
    const int64_t *col_indx_min, double alpha, double beta);

extern "C" void
s_spmv_scs_kernel_sve_C1(int64_t nchunks, const float *vals,
                         const perflibs_int_t *col_indx, const float *x,
                         float *y, const int64_t *cs, const int64_t *cl,
                         const int64_t *row_permd2in, int64_t col_indx_bytes,
                         const int64_t *col_indx_min, float alpha, float beta);

extern "C" decltype(d_spmv_scs_kernel_sve_C1) d_spmv_scs_kernel_sve_C2;
extern "C" decltype(d_spmv_scs_kernel_sve_C1) d_spmv_scs_kernel_sve_C3;
extern "C" decltype(d_spmv_scs_kernel_sve_C1) d_spmv_scs_kernel_sve_C4;
extern "C" decltype(d_spmv_scs_kernel_sve_C1) d_spmv_scs_kernel_sve_C5;
extern "C" decltype(d_spmv_scs_kernel_sve_C1) d_spmv_scs_kernel_sve_C6;
extern "C" decltype(d_spmv_scs_kernel_sve_C1) d_spmv_scs_kernel_sve_C7;
extern "C" decltype(d_spmv_scs_kernel_sve_C1) d_spmv_scs_kernel_sve_C8;

extern "C" decltype(d_spmv_scs_kernel_sve_C1) d_spmv_scs_kernel_sve_C1_sig0;
extern "C" decltype(d_spmv_scs_kernel_sve_C1) d_spmv_scs_kernel_sve_C2_sig0;
extern "C" decltype(d_spmv_scs_kernel_sve_C1) d_spmv_scs_kernel_sve_C3_sig0;
extern "C" decltype(d_spmv_scs_kernel_sve_C1) d_spmv_scs_kernel_sve_C4_sig0;
extern "C" decltype(d_spmv_scs_kernel_sve_C1) d_spmv_scs_kernel_sve_C5_sig0;
extern "C" decltype(d_spmv_scs_kernel_sve_C1) d_spmv_scs_kernel_sve_C6_sig0;
extern "C" decltype(d_spmv_scs_kernel_sve_C1) d_spmv_scs_kernel_sve_C7_sig0;
extern "C" decltype(d_spmv_scs_kernel_sve_C1) d_spmv_scs_kernel_sve_C8_sig0;

extern "C" decltype(s_spmv_scs_kernel_sve_C1) s_spmv_scs_kernel_sve_C2;
extern "C" decltype(s_spmv_scs_kernel_sve_C1) s_spmv_scs_kernel_sve_C3;
extern "C" decltype(s_spmv_scs_kernel_sve_C1) s_spmv_scs_kernel_sve_C4;
extern "C" decltype(s_spmv_scs_kernel_sve_C1) s_spmv_scs_kernel_sve_C5;
extern "C" decltype(s_spmv_scs_kernel_sve_C1) s_spmv_scs_kernel_sve_C6;
extern "C" decltype(s_spmv_scs_kernel_sve_C1) s_spmv_scs_kernel_sve_C7;
extern "C" decltype(s_spmv_scs_kernel_sve_C1) s_spmv_scs_kernel_sve_C8;

extern "C" decltype(s_spmv_scs_kernel_sve_C1) s_spmv_scs_kernel_sve_C1_sig0;
extern "C" decltype(s_spmv_scs_kernel_sve_C1) s_spmv_scs_kernel_sve_C2_sig0;
extern "C" decltype(s_spmv_scs_kernel_sve_C1) s_spmv_scs_kernel_sve_C3_sig0;
extern "C" decltype(s_spmv_scs_kernel_sve_C1) s_spmv_scs_kernel_sve_C4_sig0;
extern "C" decltype(s_spmv_scs_kernel_sve_C1) s_spmv_scs_kernel_sve_C5_sig0;
extern "C" decltype(s_spmv_scs_kernel_sve_C1) s_spmv_scs_kernel_sve_C6_sig0;
extern "C" decltype(s_spmv_scs_kernel_sve_C1) s_spmv_scs_kernel_sve_C7_sig0;
extern "C" decltype(s_spmv_scs_kernel_sve_C1) s_spmv_scs_kernel_sve_C8_sig0;

template <>
const std::vector<std::pair<int, perflibs::sparse::scs_spmv_kernels<double>>> &
perflibs::sparse::scs_get_valid_C_sve(double) {
  static auto cvals = [] {
    auto vl = vector_size_elems_sve<double>();
    std::vector<std::pair<int, scs_spmv_kernels<double>>> ret{
        {8 * vl, {d_spmv_scs_kernel_sve_C8, d_spmv_scs_kernel_sve_C8_sig0}},
        {7 * vl, {d_spmv_scs_kernel_sve_C7, d_spmv_scs_kernel_sve_C7_sig0}},
        {6 * vl, {d_spmv_scs_kernel_sve_C6, d_spmv_scs_kernel_sve_C6_sig0}},
        {5 * vl, {d_spmv_scs_kernel_sve_C5, d_spmv_scs_kernel_sve_C5_sig0}},
        {4 * vl, {d_spmv_scs_kernel_sve_C4, d_spmv_scs_kernel_sve_C4_sig0}},
        {3 * vl, {d_spmv_scs_kernel_sve_C3, d_spmv_scs_kernel_sve_C3_sig0}},
        {2 * vl, {d_spmv_scs_kernel_sve_C2, d_spmv_scs_kernel_sve_C2_sig0}},
        {1 * vl, {d_spmv_scs_kernel_sve_C1, d_spmv_scs_kernel_sve_C1_sig0}},
    };
    auto aux = scs_get_valid_C_default<double>(0.0);
    ret.insert(ret.end(), aux.begin(), aux.end());
    return ret;
  }();
  return cvals;
}

template <>
const std::vector<std::pair<int, perflibs::sparse::scs_spmv_kernels<float>>> &
perflibs::sparse::scs_get_valid_C_sve(float) {
  static auto cvals = [] {
    auto vl = vector_size_elems_sve<float>();
    std::vector<std::pair<int, scs_spmv_kernels<float>>> ret{
        {8 * vl, {s_spmv_scs_kernel_sve_C8, s_spmv_scs_kernel_sve_C8_sig0}},
        {7 * vl, {s_spmv_scs_kernel_sve_C7, s_spmv_scs_kernel_sve_C7_sig0}},
        {6 * vl, {s_spmv_scs_kernel_sve_C6, s_spmv_scs_kernel_sve_C6_sig0}},
        {5 * vl, {s_spmv_scs_kernel_sve_C5, s_spmv_scs_kernel_sve_C5_sig0}},
        {4 * vl, {s_spmv_scs_kernel_sve_C4, s_spmv_scs_kernel_sve_C4_sig0}},
        {3 * vl, {s_spmv_scs_kernel_sve_C3, s_spmv_scs_kernel_sve_C3_sig0}},
        {2 * vl, {s_spmv_scs_kernel_sve_C2, s_spmv_scs_kernel_sve_C2_sig0}},
        {1 * vl, {s_spmv_scs_kernel_sve_C1, s_spmv_scs_kernel_sve_C1_sig0}},
    };
    auto aux = scs_get_valid_C_default<float>(0.0);
    ret.insert(ret.end(), aux.begin(), aux.end());
    return ret;
  }();
  return cvals;
}

template const std::vector<
    std::pair<int, perflibs::sparse::scs_spmv_kernels<std::complex<float>>>> &
    perflibs::sparse::scs_get_valid_C_sve(std::complex<float>);
template const std::vector<
    std::pair<int, perflibs::sparse::scs_spmv_kernels<std::complex<double>>>> &
    perflibs::sparse::scs_get_valid_C_sve(std::complex<double>);
