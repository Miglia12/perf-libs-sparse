/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "c_api_complex_abi.hpp"
#include "matrix_state.hpp"
#include "object_helpers.hpp"
#include "types.hpp"
#include "util.hpp"
#include "vector_state.hpp"

perflibs_status_t perflibs_spvec_create_s(perflibs_spvec_t *x,
                                          perflibs_int_t index_base,
                                          perflibs_int_t n, perflibs_int_t nnz,
                                          const perflibs_int_t *indx,
                                          const float *vals,
                                          perflibs_int_t flags) {
  return perflibs::sparse::create_spvec_top(x, index_base, n, nnz, indx, vals,
                                            flags);
}

perflibs_status_t perflibs_spvec_create_d(perflibs_spvec_t *x,
                                          perflibs_int_t index_base,
                                          perflibs_int_t n, perflibs_int_t nnz,
                                          const perflibs_int_t *indx,
                                          const double *vals,
                                          perflibs_int_t flags) {
  return perflibs::sparse::create_spvec_top(x, index_base, n, nnz, indx, vals,
                                            flags);
}

perflibs_status_t perflibs_spvec_create_c(perflibs_spvec_t *x,
                                          perflibs_int_t index_base,
                                          perflibs_int_t n, perflibs_int_t nnz,
                                          const perflibs_int_t *indx,
                                          const perflibs_singlecomplex_t *vals,
                                          perflibs_int_t flags) {
  return perflibs::sparse::create_spvec_top(
      x, index_base, n, nnz, indx,
      reinterpret_cast<const std::complex<float> *>(vals), flags);
}

perflibs_status_t perflibs_spvec_create_z(perflibs_spvec_t *x,
                                          perflibs_int_t index_base,
                                          perflibs_int_t n, perflibs_int_t nnz,
                                          const perflibs_int_t *indx,
                                          const perflibs_doublecomplex_t *vals,
                                          perflibs_int_t flags) {
  return perflibs::sparse::create_spvec_top(
      x, index_base, n, nnz, indx,
      reinterpret_cast<const std::complex<double> *>(vals), flags);
}

perflibs_status_t perflibs_spvec_query(perflibs_spvec_t x,
                                       perflibs_int_t *index_base,
                                       perflibs_int_t *n, perflibs_int_t *nnz) {
  return perflibs::sparse::basic_query(x, index_base, n, nnz);
}

perflibs_status_t perflibs_spvec_destroy(perflibs_spvec_t x) {
  delete x;
  return PERFLIBS_STATUS_SUCCESS;
}

perflibs_status_t perflibs_spvec_export_s(perflibs_spvec_t x,
                                          perflibs_int_t *index_base,
                                          perflibs_int_t *n,
                                          perflibs_int_t *nnz,
                                          perflibs_int_t *indx, float *vals) {
  return perflibs::sparse::export_spvec(x, index_base, n, nnz, indx, vals);
}

perflibs_status_t perflibs_spvec_export_d(perflibs_spvec_t x,
                                          perflibs_int_t *index_base,
                                          perflibs_int_t *n,
                                          perflibs_int_t *nnz,
                                          perflibs_int_t *indx, double *vals) {
  return perflibs::sparse::export_spvec(x, index_base, n, nnz, indx, vals);
}

perflibs_status_t
perflibs_spvec_export_c(perflibs_spvec_t x, perflibs_int_t *index_base,
                        perflibs_int_t *n, perflibs_int_t *nnz,
                        perflibs_int_t *indx, perflibs_singlecomplex_t *vals) {
  return perflibs::sparse::export_spvec(
      x, index_base, n, nnz, indx,
      reinterpret_cast<std::complex<float> *>(vals));
}

perflibs_status_t
perflibs_spvec_export_z(perflibs_spvec_t x, perflibs_int_t *index_base,
                        perflibs_int_t *n, perflibs_int_t *nnz,
                        perflibs_int_t *indx, perflibs_doublecomplex_t *vals) {
  return perflibs::sparse::export_spvec(
      x, index_base, n, nnz, indx,
      reinterpret_cast<std::complex<double> *>(vals));
}

perflibs_status_t perflibs_spvec_gather_s(const float *x_d,
                                          perflibs_int_t index_base,
                                          perflibs_int_t n,
                                          perflibs_spvec_t *x_s,
                                          perflibs_int_t flags) {
  return perflibs::sparse::gather_spvec_top(x_d, index_base, n, x_s, flags);
}

perflibs_status_t perflibs_spvec_gather_d(const double *x_d,
                                          perflibs_int_t index_base,
                                          perflibs_int_t n,
                                          perflibs_spvec_t *x_s,
                                          perflibs_int_t flags) {
  return perflibs::sparse::gather_spvec_top(x_d, index_base, n, x_s, flags);
}

perflibs_status_t perflibs_spvec_gather_c(const perflibs_singlecomplex_t *x_d,
                                          perflibs_int_t index_base,
                                          perflibs_int_t n,
                                          perflibs_spvec_t *x_s,
                                          perflibs_int_t flags) {
  return perflibs::sparse::gather_spvec_top(
      reinterpret_cast<const std::complex<float> *>(x_d), index_base, n, x_s,
      flags);
}

perflibs_status_t perflibs_spvec_gather_z(const perflibs_doublecomplex_t *x_d,
                                          perflibs_int_t index_base,
                                          perflibs_int_t n,
                                          perflibs_spvec_t *x_s,
                                          perflibs_int_t flags) {
  return perflibs::sparse::gather_spvec_top(
      reinterpret_cast<const std::complex<double> *>(x_d), index_base, n, x_s,
      flags);
}

perflibs_status_t perflibs_spvec_scatter_s(perflibs_spvec_t x_s, float *x_d) {
  return perflibs::sparse::scatter_spvec_top(x_s, x_d);
}

perflibs_status_t perflibs_spvec_scatter_d(perflibs_spvec_t x_s, double *x_d) {
  return perflibs::sparse::scatter_spvec_top(x_s, x_d);
}

perflibs_status_t perflibs_spvec_scatter_c(perflibs_spvec_t x_s,
                                           perflibs_singlecomplex_t *x_d) {
  return perflibs::sparse::scatter_spvec_top(
      x_s, reinterpret_cast<std::complex<float> *>(x_d));
}

perflibs_status_t perflibs_spvec_scatter_z(perflibs_spvec_t x_s,
                                           perflibs_doublecomplex_t *x_d) {
  return perflibs::sparse::scatter_spvec_top(
      x_s, reinterpret_cast<std::complex<double> *>(x_d));
}

perflibs_status_t perflibs_spvec_update_s(perflibs_spvec_t x,
                                          perflibs_int_t n_updates,
                                          const perflibs_int_t *indx,
                                          const float *vals) {
  return perflibs::sparse::update_spvec(x, n_updates, indx, vals);
}

perflibs_status_t perflibs_spvec_update_d(perflibs_spvec_t x,
                                          perflibs_int_t n_updates,
                                          const perflibs_int_t *indx,
                                          const double *vals) {
  return perflibs::sparse::update_spvec(x, n_updates, indx, vals);
}

perflibs_status_t
perflibs_spvec_update_c(perflibs_spvec_t x, perflibs_int_t n_updates,
                        const perflibs_int_t *indx,
                        const perflibs_singlecomplex_t *vals) {
  return perflibs::sparse::update_spvec(
      x, n_updates, indx, reinterpret_cast<const std::complex<float> *>(vals));
}

perflibs_status_t
perflibs_spvec_update_z(perflibs_spvec_t x, perflibs_int_t n_updates,
                        const perflibs_int_t *indx,
                        const perflibs_doublecomplex_t *vals) {
  return perflibs::sparse::update_spvec(
      x, n_updates, indx, reinterpret_cast<const std::complex<double> *>(vals));
}

perflibs_status_t perflibs_spdot_exec_s(perflibs_spvec_t x, const float *y,
                                        float *result) {
  return perflibs::sparse::dot_exec_top<false>(x, y, result);
}

perflibs_status_t perflibs_spdot_exec_d(perflibs_spvec_t x, const double *y,
                                        double *result) {
  return perflibs::sparse::dot_exec_top<false>(x, y, result);
}

perflibs_status_t perflibs_spdotu_exec_c(perflibs_spvec_t x,
                                         const perflibs_singlecomplex_t *y,
                                         perflibs_singlecomplex_t *result) {
  return perflibs::sparse::dot_exec_top<false>(
      x, reinterpret_cast<const std::complex<float> *>(y),
      reinterpret_cast<std::complex<float> *>(result));
}

perflibs_status_t perflibs_spdotu_exec_z(perflibs_spvec_t x,
                                         const perflibs_doublecomplex_t *y,
                                         perflibs_doublecomplex_t *result) {
  return perflibs::sparse::dot_exec_top<false>(
      x, reinterpret_cast<const std::complex<double> *>(y),
      reinterpret_cast<std::complex<double> *>(result));
}

perflibs_status_t perflibs_spdotc_exec_c(perflibs_spvec_t x,
                                         const perflibs_singlecomplex_t *y,
                                         perflibs_singlecomplex_t *result) {
  return perflibs::sparse::dot_exec_top<true>(
      x, reinterpret_cast<const std::complex<float> *>(y),
      reinterpret_cast<std::complex<float> *>(result));
}

perflibs_status_t perflibs_spdotc_exec_z(perflibs_spvec_t x,
                                         const perflibs_doublecomplex_t *y,
                                         perflibs_doublecomplex_t *result) {
  return perflibs::sparse::dot_exec_top<true>(
      x, reinterpret_cast<const std::complex<double> *>(y),
      reinterpret_cast<std::complex<double> *>(result));
}

perflibs_status_t perflibs_spaxpby_exec_s(const float alpha, perflibs_spvec_t x,
                                          const float beta, float *y) {
  return perflibs::sparse::axpby_exec_top(alpha, x, beta, y);
}

perflibs_status_t perflibs_spaxpby_exec_d(const double alpha,
                                          perflibs_spvec_t x, const double beta,
                                          double *y) {
  return perflibs::sparse::axpby_exec_top(alpha, x, beta, y);
}

perflibs_status_t perflibs_spaxpby_exec_c(const perflibs_singlecomplex_t alpha,
                                          perflibs_spvec_t x,
                                          const perflibs_singlecomplex_t beta,
                                          perflibs_singlecomplex_t *y) {
  return perflibs::sparse::axpby_exec_top(
      perflibs::sparse::c_api::to_cpp_scalar(alpha), x,
      perflibs::sparse::c_api::to_cpp_scalar(beta),
      reinterpret_cast<std::complex<float> *>(y));
}

perflibs_status_t perflibs_spaxpby_exec_z(const perflibs_doublecomplex_t alpha,
                                          perflibs_spvec_t x,
                                          const perflibs_doublecomplex_t beta,
                                          perflibs_doublecomplex_t *y) {
  return perflibs::sparse::axpby_exec_top(
      perflibs::sparse::c_api::to_cpp_scalar(alpha), x,
      perflibs::sparse::c_api::to_cpp_scalar(beta),
      reinterpret_cast<std::complex<double> *>(y));
}

perflibs_status_t perflibs_spwaxpby_exec_s(const float alpha,
                                           perflibs_spvec_t x, const float beta,
                                           const float *y, float *w) {
  return perflibs::sparse::waxpby_exec_top(alpha, x, beta, y, w);
}

perflibs_status_t perflibs_spwaxpby_exec_d(const double alpha,
                                           perflibs_spvec_t x,
                                           const double beta, const double *y,
                                           double *w) {
  return perflibs::sparse::waxpby_exec_top(alpha, x, beta, y, w);
}

perflibs_status_t perflibs_spwaxpby_exec_c(const perflibs_singlecomplex_t alpha,
                                           perflibs_spvec_t x,
                                           const perflibs_singlecomplex_t beta,
                                           const perflibs_singlecomplex_t *y,
                                           perflibs_singlecomplex_t *w) {
  return perflibs::sparse::waxpby_exec_top(
      perflibs::sparse::c_api::to_cpp_scalar(alpha), x,
      perflibs::sparse::c_api::to_cpp_scalar(beta),
      reinterpret_cast<const std::complex<float> *>(y),
      reinterpret_cast<std::complex<float> *>(w));
}

perflibs_status_t perflibs_spwaxpby_exec_z(const perflibs_doublecomplex_t alpha,
                                           perflibs_spvec_t x,
                                           const perflibs_doublecomplex_t beta,
                                           const perflibs_doublecomplex_t *y,
                                           perflibs_doublecomplex_t *w) {
  return perflibs::sparse::waxpby_exec_top(
      perflibs::sparse::c_api::to_cpp_scalar(alpha), x,
      perflibs::sparse::c_api::to_cpp_scalar(beta),
      reinterpret_cast<const std::complex<double> *>(y),
      reinterpret_cast<std::complex<double> *>(w));
}

perflibs_status_t perflibs_sprot_exec_s(perflibs_spvec_t x, float *y, float c,
                                        float s) {
  return perflibs::sparse::rot_exec_top(x, y, c, s);
}

perflibs_status_t perflibs_sprot_exec_d(perflibs_spvec_t x, double *y, double c,
                                        double s) {
  return perflibs::sparse::rot_exec_top(x, y, c, s);
}

perflibs_status_t perflibs_sprot_exec_c(perflibs_spvec_t x,
                                        perflibs_singlecomplex_t *y, float c,
                                        perflibs_singlecomplex_t s) {
  return perflibs::sparse::rot_exec_top(
      x, reinterpret_cast<std::complex<float> *>(y), c,
      perflibs::sparse::c_api::to_cpp_scalar(s));
}

perflibs_status_t perflibs_sprot_exec_cs(perflibs_spvec_t x,
                                         perflibs_singlecomplex_t *y, float c,
                                         float s) {
  return perflibs::sparse::rot_exec_top(
      x, reinterpret_cast<std::complex<float> *>(y), c, s);
}

perflibs_status_t perflibs_sprot_exec_z(perflibs_spvec_t x,
                                        perflibs_doublecomplex_t *y, double c,
                                        perflibs_doublecomplex_t s) {
  return perflibs::sparse::rot_exec_top(
      x, reinterpret_cast<std::complex<double> *>(y), c,
      perflibs::sparse::c_api::to_cpp_scalar(s));
}

perflibs_status_t perflibs_sprot_exec_zd(perflibs_spvec_t x,
                                         perflibs_doublecomplex_t *y, double c,
                                         double s) {
  return perflibs::sparse::rot_exec_top(
      x, reinterpret_cast<std::complex<double> *>(y), c, s);
}

void perflibs_spvec_print_err(perflibs_spvec_t x) {
  perflibs::sparse::spvec_print_err(x);
}
