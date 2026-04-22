/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "vector_state.hpp"
#include "object_helpers.hpp"
#include "types.hpp"
#include "util.hpp"
#include "vector_ops.hpp"

#include <cstring>
#include <inttypes.h>
#include <string>

namespace perflibs::sparse {

// Function to be called to return an error when one of the values to update is
// not non-zero
inline perflibs_status_t update_error(sp_error_t &error_handle,
                                      perflibs_int_t i) {
  error_handle.perflibs_error_type = PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  error_handle.perflibs_error_code =
      1; // The vector is always the first parameter

  std::string str0 = "The element requested to be updated in position ";
  std::string str1 = std::to_string(i);
  std::string str2 = " of the parameters passed to the update function does "
                     "not correspond to a non-zero "
                     "value in the sparse vector. The update function can only "
                     "update existing values, it "
                     "cannot be used to introduce new non-zeros.";

  error_handle.err_msg = str0 + str1 + str2;

  return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
}

template <typename T>
perflibs_status_t export_spvec(perflibs_spvec_t x, perflibs_int_t *index_base,
                               perflibs_int_t *n, perflibs_int_t *nnz,
                               perflibs_int_t *indx, T *vals) {
  auto impl = reinterpret_cast<perflibs_spvec_impl_t<T> *>(x->impl);
  *index_base = impl->index_base;
  *n = impl->n;
  *nnz = impl->nnz;
  std::memcpy((void *)indx, (const void *)impl->vec.indx_ptr,
              sizeof(perflibs_int_t) * impl->nnz);
  std::memcpy((void *)vals, (const void *)impl->vec.vals_ptr,
              sizeof(T) * impl->nnz);
  return PERFLIBS_STATUS_SUCCESS;
}
template perflibs_status_t export_spvec(perflibs_spvec_t x,
                                        perflibs_int_t *index_base,
                                        perflibs_int_t *n, perflibs_int_t *nnz,
                                        perflibs_int_t *indx, float *vals);
template perflibs_status_t export_spvec(perflibs_spvec_t x,
                                        perflibs_int_t *index_base,
                                        perflibs_int_t *n, perflibs_int_t *nnz,
                                        perflibs_int_t *indx, double *vals);
template perflibs_status_t export_spvec(perflibs_spvec_t x,
                                        perflibs_int_t *index_base,
                                        perflibs_int_t *n, perflibs_int_t *nnz,
                                        perflibs_int_t *indx,
                                        std::complex<float> *vals);
template perflibs_status_t export_spvec(perflibs_spvec_t x,
                                        perflibs_int_t *index_base,
                                        perflibs_int_t *n, perflibs_int_t *nnz,
                                        perflibs_int_t *indx,
                                        std::complex<double> *vals);

template <typename T>
perflibs_status_t gather_spvec_top(const T *x_d, perflibs_int_t index_base,
                                   perflibs_int_t n, perflibs_spvec_t *x_s,
                                   perflibs_int_t flags) {
  auto xtop = create_new_vector<T>().release();
  *x_s = xtop;
  return perflibs::sparse::gather_vec(x_d, index_base, n, xtop,
                                      flags & PERFLIBS_SPARSE_CREATE_NOCOPY);
};
template perflibs_status_t gather_spvec_top<float>(const float *x_d,
                                                   perflibs_int_t index_base,
                                                   perflibs_int_t n,
                                                   perflibs_spvec_t *x_s,
                                                   perflibs_int_t flags);
template perflibs_status_t gather_spvec_top<double>(const double *x_d,
                                                    perflibs_int_t index_base,
                                                    perflibs_int_t n,
                                                    perflibs_spvec_t *x_s,
                                                    perflibs_int_t flags);
template perflibs_status_t gather_spvec_top<std::complex<float>>(
    const std::complex<float> *x_d, perflibs_int_t index_base, perflibs_int_t n,
    perflibs_spvec_t *x_s, perflibs_int_t flags);
template perflibs_status_t gather_spvec_top<std::complex<double>>(
    const std::complex<double> *x_d, perflibs_int_t index_base,
    perflibs_int_t n, perflibs_spvec_t *x_s, perflibs_int_t flags);

template <typename T>
perflibs_status_t scatter_spvec_top(perflibs_spvec_t x_s, T *x_d) {
  return perflibs::sparse::scatter_vec(x_s, x_d);
};
template perflibs_status_t scatter_spvec_top<float>(perflibs_spvec_t x_s,
                                                    float *x_d);
template perflibs_status_t scatter_spvec_top<double>(perflibs_spvec_t x_s,
                                                     double *x_d);
template perflibs_status_t
scatter_spvec_top<std::complex<float>>(perflibs_spvec_t x_s,
                                       std::complex<float> *x_d);
template perflibs_status_t
scatter_spvec_top<std::complex<double>>(perflibs_spvec_t x_s,
                                        std::complex<double> *x_d);

template <typename T>
perflibs_status_t update_spvec(perflibs_spvec_t x, perflibs_int_t n_updates,
                               const perflibs_int_t *indx, const T *vals) {
  if (n_updates == 0) {
    return PERFLIBS_STATUS_SUCCESS;
  }

  auto impl = reinterpret_cast<perflibs_spvec_impl_t<T> *>(x->impl);
  if (n_updates < 0) {
    impl->error_handle.perflibs_error_type =
        PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    impl->error_handle.perflibs_error_code = 1;
    impl->error_handle.err_msg = "The number of updates can not be negative.";
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }

  auto nnz = impl->nnz;
  auto indx_ptr_orig = impl->vec.indx_ptr;
  auto vals_orig = const_cast<T *>(impl->vec.vals_ptr);
  for (perflibs_int_t i = 0; i < n_updates; ++i) {
    bool success = false;
    for (perflibs_int_t j = 0; j < nnz; ++j) {
      if (indx_ptr_orig[j] == indx[i]) {
        vals_orig[j] = vals[i];
        success = true;
        break;
      }
    }
    // If we exit the inner loop above without updating an element
    // and continuing on to the next update then the element is not
    // a non-zero value
    if (!success) {
      return update_error(impl->error_handle, i + impl->index_base);
    }
  }
  return PERFLIBS_STATUS_SUCCESS;
};
template perflibs_status_t update_spvec<float>(perflibs_spvec_t x,
                                               perflibs_int_t n_updates,
                                               const perflibs_int_t *indx,
                                               const float *vals);
template perflibs_status_t update_spvec<double>(perflibs_spvec_t x,
                                                perflibs_int_t n_updates,
                                                const perflibs_int_t *indx,
                                                const double *vals);
template perflibs_status_t
update_spvec<std::complex<float>>(perflibs_spvec_t x, perflibs_int_t n_updates,
                                  const perflibs_int_t *indx,
                                  const std::complex<float> *vals);
template perflibs_status_t
update_spvec<std::complex<double>>(perflibs_spvec_t x, perflibs_int_t n_updates,
                                   const perflibs_int_t *indx,
                                   const std::complex<double> *vals);

template <bool IsConj, typename T>
perflibs_status_t dot_exec_top(perflibs_spvec_t x, const T *y, T *result) {
  return perflibs::sparse::dot_exec<IsConj>(x, y, result);
}
template perflibs_status_t
dot_exec_top<false, float>(perflibs_spvec_t x, const float *y, float *result);
template perflibs_status_t dot_exec_top<false, double>(perflibs_spvec_t x,
                                                       const double *y,
                                                       double *result);
template perflibs_status_t
dot_exec_top<false, std::complex<float>>(perflibs_spvec_t x,
                                         const std::complex<float> *y,
                                         std::complex<float> *result);
template perflibs_status_t
dot_exec_top<false, std::complex<double>>(perflibs_spvec_t x,
                                          const std::complex<double> *y,
                                          std::complex<double> *result);
template perflibs_status_t
dot_exec_top<true, std::complex<float>>(perflibs_spvec_t x,
                                        const std::complex<float> *y,
                                        std::complex<float> *result);
template perflibs_status_t
dot_exec_top<true, std::complex<double>>(perflibs_spvec_t x,
                                         const std::complex<double> *y,
                                         std::complex<double> *result);

template <typename T>
perflibs_status_t axpby_exec_top(const T alpha, perflibs_spvec_t x,
                                 const T beta, T *y) {
  return perflibs::sparse::waxpby_exec(alpha, x, beta, y, y);
}
template perflibs_status_t axpby_exec_top<float>(const float alpha,
                                                 perflibs_spvec_t x,
                                                 const float beta, float *y);
template perflibs_status_t axpby_exec_top<double>(const double alpha,
                                                  perflibs_spvec_t x,
                                                  const double beta, double *y);
template perflibs_status_t axpby_exec_top<std::complex<float>>(
    const std::complex<float> alpha, perflibs_spvec_t x,
    const std::complex<float> beta, std::complex<float> *y);
template perflibs_status_t axpby_exec_top<std::complex<double>>(
    const std::complex<double> alpha, perflibs_spvec_t x,
    const std::complex<double> beta, std::complex<double> *y);

template <typename T>
perflibs_status_t waxpby_exec_top(const T alpha, perflibs_spvec_t x,
                                  const T beta, const T *y, T *w) {
  return perflibs::sparse::waxpby_exec(alpha, x, beta, y, w);
}
template perflibs_status_t waxpby_exec_top<float>(const float alpha,
                                                  perflibs_spvec_t x,
                                                  const float beta,
                                                  const float *y, float *w);
template perflibs_status_t waxpby_exec_top<double>(const double alpha,
                                                   perflibs_spvec_t x,
                                                   const double beta,
                                                   const double *y, double *w);
template perflibs_status_t waxpby_exec_top<std::complex<float>>(
    const std::complex<float> alpha, perflibs_spvec_t x,
    const std::complex<float> beta, const std::complex<float> *y,
    std::complex<float> *w);
template perflibs_status_t waxpby_exec_top<std::complex<double>>(
    const std::complex<double> alpha, perflibs_spvec_t x,
    const std::complex<double> beta, const std::complex<double> *y,
    std::complex<double> *w);

template <typename T1, typename T2>
perflibs_status_t rot_exec_top(perflibs_spvec_t x, T1 *y,
                               perflibs::sparse::remove_complex_t<T1> c, T2 s) {
  return perflibs::sparse::rot_exec(x, y, c, s);
}
template perflibs_status_t
rot_exec_top<float, float>(perflibs_spvec_t x, float *y, float c, float s);
template perflibs_status_t
rot_exec_top<double, double>(perflibs_spvec_t x, double *y, double c, double s);
template perflibs_status_t
rot_exec_top<std::complex<float>, std::complex<float>>(perflibs_spvec_t x,
                                                       std::complex<float> *y,
                                                       float c,
                                                       std::complex<float> s);
template perflibs_status_t rot_exec_top<std::complex<float>, float>(
    perflibs_spvec_t x, std::complex<float> *y, float c, float s);
template perflibs_status_t
rot_exec_top<std::complex<double>, std::complex<double>>(
    perflibs_spvec_t x, std::complex<double> *y, double c,
    std::complex<double> s);
template perflibs_status_t rot_exec_top<std::complex<double>, double>(
    perflibs_spvec_t x, std::complex<double> *y, double c, double s);

inline void sparse_print_err(sp_error_t err) {
  auto type = err.perflibs_error_type;
  int64_t code = (int64_t)err.perflibs_error_code;
  auto msg = err.err_msg;

  const std::string fail =
      "Execution failed for an unexpected reason. This may occur if one of the "
      "sparse objects provided to the library has been corrupted.";

  switch (type) {
  case PERFLIBS_STATUS_EXECUTION_FAILURE:
    perflibs::sparse::print_wrap(fail, true);
    break;
  case PERFLIBS_STATUS_INPUT_PARAMETER_ERROR:
    // Parameter error report is similar to BLAS's xerbla - not calling
    // print_wrap for this due to the complication of formatting, but the same
    // banner printed.
    fprintf(stderr,
            "%sParameter error: parameter %1" PRId64 " caused an error.\n",
            perflibs::sparse::banner, code);
    if (!msg.empty()) {
      perflibs::sparse::print_wrap(msg, false);
    }
  }
}

void spmat_print_err(perflibs_spmat_t A) {
  // The underlying datatype doesn't matter in this case, so use float
  // arbitrarily
  auto impl = reinterpret_cast<perflibs_spmat_impl_t<float> *>(A->impl);
  sparse_print_err(impl->error_handle);
}

void spvec_print_err(perflibs_spvec_t x) {
  // The underlying datatype doesn't matter in this case, so use float
  // arbitrarily
  auto impl = reinterpret_cast<perflibs_spvec_impl_t<float> *>(x->impl);
  sparse_print_err(impl->error_handle);
}

} // namespace perflibs::sparse
