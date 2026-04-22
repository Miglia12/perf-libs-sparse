/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "vector_ops.hpp"
#include "cblas_wrappers.hpp"
#include "types.hpp"
#include "vector_storage.hpp"

namespace perflibs::sparse {

template <typename T>
perflibs_status_t check_gather_params(perflibs_spvec_impl_t<T> *impl,
                                      perflibs_int_t index_base,
                                      perflibs_int_t n) {
  /* Check input values */
  if (index_base != 0 && index_base != 1) {
    impl->error_handle.perflibs_error_type =
        PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    impl->error_handle.perflibs_error_code = 2;
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }
  if (n < 0) {
    impl->error_handle.perflibs_error_type =
        PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    impl->error_handle.perflibs_error_code = 3;
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }
  return PERFLIBS_STATUS_SUCCESS;
}

template <typename T>
perflibs_status_t gather_vec(const T *x_d, perflibs_int_t index_base,
                             perflibs_int_t n, perflibs_spvec_t x_s,
                             bool no_copy) {
  auto impl = reinterpret_cast<perflibs_spvec_impl_t<T> *>(x_s->impl);
  auto ret = check_gather_params(impl, index_base, n);
  if (ret != PERFLIBS_STATUS_SUCCESS) {
    return ret;
  }
  std::vector<T> vals;
  std::vector<perflibs_int_t> indx;
  perflibs_int_t nnz = 0;
  for (perflibs_int_t i = 0; i < n; ++i) {
    if (x_d[i] != (T)0) {
      vals.emplace_back(x_d[i]);
      indx.emplace_back(i + index_base);
      nnz++;
    }
  }
  impl->index_base = index_base;
  impl->n = n;
  impl->nnz = nnz;
  // We take copies of indx and vals in order to trigger a free on destruction
  impl->vec = perflibs_vec<T>(index_base, n, nnz, indx.data(), vals.data());
  return PERFLIBS_STATUS_SUCCESS;
}

template perflibs_status_t
gather_vec<float>(const float *x_d, perflibs_int_t index_base, perflibs_int_t n,
                  perflibs_spvec_t x_s, bool no_copy);
template perflibs_status_t
gather_vec<double>(const double *x_d, perflibs_int_t index_base,
                   perflibs_int_t n, perflibs_spvec_t x_s, bool no_copy);
template perflibs_status_t
gather_vec<std::complex<float>>(const std::complex<float> *x_d,
                                perflibs_int_t index_base, perflibs_int_t n,
                                perflibs_spvec_t x_s, bool no_copy);
template perflibs_status_t
gather_vec<std::complex<double>>(const std::complex<double> *x_d,
                                 perflibs_int_t index_base, perflibs_int_t n,
                                 perflibs_spvec_t x_s, bool no_copy);

template <typename T>
perflibs_status_t scatter_vec(perflibs_spvec_t x_s, T *x_d) {
  auto impl = reinterpret_cast<perflibs_spvec_impl_t<T> *>(x_s->impl);
  for (perflibs_int_t i = 0; i < impl->n; ++i) {
    x_d[i] = (T)0;
  }
  auto indx_ptr = impl->vec.indx_ptr;
  auto vals_ptr = impl->vec.vals_ptr;
  for (perflibs_int_t i = 0; i < impl->nnz; ++i) {
    auto indx = indx_ptr[i] - impl->index_base;
    auto vals = vals_ptr[i];
    x_d[indx] = vals;
  }
  return PERFLIBS_STATUS_SUCCESS;
}

template perflibs_status_t scatter_vec<float>(perflibs_spvec_t x_s, float *x_d);
template perflibs_status_t scatter_vec<double>(perflibs_spvec_t x_s,
                                               double *x_d);
template perflibs_status_t
scatter_vec<std::complex<float>>(perflibs_spvec_t x_s,
                                 std::complex<float> *x_d);
template perflibs_status_t
scatter_vec<std::complex<double>>(perflibs_spvec_t x_s,
                                  std::complex<double> *x_d);

template <bool IsConj, typename T>
perflibs_status_t dot_exec(perflibs_spvec_t x, const T *y, T *result) {
  auto impl = reinterpret_cast<perflibs_spvec_impl_t<T> *>(x->impl);
  auto nnz = impl->nnz;
  auto index_base = impl->index_base;
  auto indx_ptr = impl->vec.indx_ptr;
  auto vals_ptr = impl->vec.vals_ptr;
  *result = (T)0;
  for (perflibs_int_t i = 0; i < nnz; ++i) {
    auto idx = indx_ptr[i] - index_base;
    if constexpr (IsConj) {
      *result += conj(vals_ptr[i]) * y[idx];
    } else {
      *result += vals_ptr[i] * y[idx];
    }
  }
  return PERFLIBS_STATUS_SUCCESS;
}
template perflibs_status_t
dot_exec<false, float>(perflibs_spvec_t x, const float *y, float *result);
template perflibs_status_t
dot_exec<false, std::complex<float>>(perflibs_spvec_t x,
                                     const std::complex<float> *y,
                                     std::complex<float> *result);
template perflibs_status_t
dot_exec<false, std::complex<double>>(perflibs_spvec_t x,
                                      const std::complex<double> *y,
                                      std::complex<double> *result);
template perflibs_status_t
dot_exec<true, std::complex<float>>(perflibs_spvec_t x,
                                    const std::complex<float> *y,
                                    std::complex<float> *result);
template perflibs_status_t
dot_exec<true, std::complex<double>>(perflibs_spvec_t x,
                                     const std::complex<double> *y,
                                     std::complex<double> *result);
// For double precision use the ACLE function used to speed up CSR SpSV
template <>
perflibs_status_t dot_exec<false, double>(perflibs_spvec_t x, const double *y,
                                          double *result) {
  auto impl = reinterpret_cast<perflibs_spvec_impl_t<double> *>(x->impl);
  const perflibs_int_t nnz = impl->nnz;
  const perflibs_int_t base = impl->index_base;
  const perflibs_int_t *indx = impl->vec.indx_ptr;
  const double *vals = impl->vec.vals_ptr;

  *result = 0.0;
  if (nnz == 0) {
    return PERFLIBS_STATUS_SUCCESS;
  }

  spsv_dot_fp64_helper(*result, indx, vals, y, base, 0, nnz - 1);

  return PERFLIBS_STATUS_SUCCESS;
}

template <typename T>
perflibs_status_t waxpby_exec(const T alpha, perflibs_spvec_t x, const T beta,
                              const T *y, T *w) {
  auto impl = reinterpret_cast<perflibs_spvec_impl_t<T> *>(x->impl);
  auto n = impl->n;
  auto nnz = impl->nnz;
  auto index_base = impl->index_base;
  auto indx_ptr = impl->vec.indx_ptr;
  auto vals_ptr = impl->vec.vals_ptr;
  if (beta == T(0)) {
    std::memset((void *)w, 0, sizeof(T) * (n));
  } else if (beta == T(1) && w != y) {
    std::memcpy((void *)w, (const void *)y, sizeof(T) * (n));
  } else {
    // w = beta * y;
    const perflibs_int_t inc = 1;
    const T zero = T(0);
    perflibs::sparse::cblas_axpby<T>(n, beta, y, inc, zero, w, inc);
  }
  if (alpha != T(0)) {
    for (perflibs_int_t i = 0; i < nnz; ++i) {
      auto idx = indx_ptr[i] - index_base;
      w[idx] += alpha * vals_ptr[i];
    }
  }
  return PERFLIBS_STATUS_SUCCESS;
}
template perflibs_status_t waxpby_exec<float>(const float alpha,
                                              perflibs_spvec_t x,
                                              const float beta, const float *y,
                                              float *w);
template perflibs_status_t waxpby_exec<double>(const double alpha,
                                               perflibs_spvec_t x,
                                               const double beta,
                                               const double *y, double *w);
template perflibs_status_t waxpby_exec<std::complex<float>>(
    const std::complex<float> alpha, perflibs_spvec_t x,
    const std::complex<float> beta, const std::complex<float> *y,
    std::complex<float> *w);
template perflibs_status_t waxpby_exec<std::complex<double>>(
    const std::complex<double> alpha, perflibs_spvec_t x,
    const std::complex<double> beta, const std::complex<double> *y,
    std::complex<double> *w);

template <typename T1, typename T2>
perflibs_status_t rot_exec(perflibs_spvec_t x, T1 *y,
                           perflibs::sparse::remove_complex_t<T1> c, T2 s) {
  auto impl = reinterpret_cast<perflibs_spvec_impl_t<T1> *>(x->impl);
  auto nnz = impl->nnz;
  auto index_base = impl->index_base;
  auto indx_ptr = impl->vec.indx_ptr;
  auto vals_ptr = const_cast<T1 *>(impl->vec.vals_ptr);

  for (perflibs_int_t i = 0; i < nnz; ++i) {
    auto idx = indx_ptr[i] - index_base;
    auto temp = c * vals_ptr[i] + s * y[idx];
    if constexpr (perflibs::sparse::is_complex_v<T2>) {
      y[idx] = c * y[idx] - perflibs::sparse::conj(s) * vals_ptr[i];
    } else {
      y[idx] = c * y[idx] - s * vals_ptr[i];
    }
    vals_ptr[i] = temp;
  }

  return PERFLIBS_STATUS_SUCCESS;
}
template perflibs_status_t rot_exec<float, float>(perflibs_spvec_t x, float *y,
                                                  float c, float s);
template perflibs_status_t
rot_exec<double, double>(perflibs_spvec_t x, double *y, double c, double s);
template perflibs_status_t rot_exec<std::complex<float>, std::complex<float>>(
    perflibs_spvec_t x, std::complex<float> *y, float c, std::complex<float> s);
template perflibs_status_t
rot_exec<std::complex<float>, float>(perflibs_spvec_t x, std::complex<float> *y,
                                     float c, float s);
template perflibs_status_t rot_exec<std::complex<double>, std::complex<double>>(
    perflibs_spvec_t x, std::complex<double> *y, double c,
    std::complex<double> s);
template perflibs_status_t rot_exec<std::complex<double>, double>(
    perflibs_spvec_t x, std::complex<double> *y, double c, double s);

} // end namespace perflibs::sparse
