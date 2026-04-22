/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "vector_storage.hpp"
#include "object_helpers.hpp"
#include "types.hpp"

namespace perflibs::sparse {

template <typename T>
perflibs_vec<T> make_vec(perflibs_int_t index_base, perflibs_int_t n,
                         perflibs_int_t nnz, const perflibs_int_t *indx,
                         const T *vals, bool no_copy) {
  if (no_copy) {
    return perflibs_vec<T>(index_base, n, indx, vals, nnz);
  } else {
    return perflibs_vec<T>(index_base, n, nnz, indx, vals);
  }
}

template <typename T>
void init_vec(perflibs_spvec_impl_t<T> *impl, perflibs_int_t index_base,
              perflibs_int_t n, perflibs_int_t nnz) {
  impl->index_base = index_base;
  impl->n = n;
  impl->nnz = nnz;
}

template <typename T>
perflibs_status_t check_vec_params(perflibs_spvec_impl_t<T> *impl,
                                   perflibs_int_t index_base, perflibs_int_t n,
                                   perflibs_int_t nnz) {
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
  if (nnz < 0) {
    impl->error_handle.perflibs_error_type =
        PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    impl->error_handle.perflibs_error_code = 4;
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }
  return PERFLIBS_STATUS_SUCCESS;
}

template <typename T>
perflibs_status_t
fill_initial_data_vec(perflibs_spvec_top_t *x, perflibs_int_t index_base,
                      perflibs_int_t n, perflibs_int_t nnz,
                      const perflibs_int_t *indx, const T *vals, bool no_copy) {

  auto impl = reinterpret_cast<perflibs_spvec_impl_t<T> *>(x->impl);
  auto ret = check_vec_params(impl, index_base, n, nnz);
  if (ret == PERFLIBS_STATUS_SUCCESS) {
    init_vec(impl, index_base, n, nnz);
  } else {
    return ret;
  }

  impl->vec =
      make_vec<T>(impl->index_base, impl->n, impl->nnz, indx, vals, no_copy);

  return PERFLIBS_STATUS_SUCCESS;
}
template perflibs_status_t
fill_initial_data_vec<float>(perflibs_spvec_top_t *x, perflibs_int_t index_base,
                             perflibs_int_t n, perflibs_int_t nnz,
                             const perflibs_int_t *indx, const float *vals,
                             bool no_copy);
template perflibs_status_t
fill_initial_data_vec<double>(perflibs_spvec_top_t *x,
                              perflibs_int_t index_base, perflibs_int_t n,
                              perflibs_int_t nnz, const perflibs_int_t *indx,
                              const double *vals, bool no_copy);
template perflibs_status_t fill_initial_data_vec<std::complex<float>>(
    perflibs_spvec_top_t *x, perflibs_int_t index_base, perflibs_int_t n,
    perflibs_int_t nnz, const perflibs_int_t *indx,
    const std::complex<float> *vals, bool no_copy);
template perflibs_status_t fill_initial_data_vec<std::complex<double>>(
    perflibs_spvec_top_t *x, perflibs_int_t index_base, perflibs_int_t n,
    perflibs_int_t nnz, const perflibs_int_t *indx,
    const std::complex<double> *vals, bool no_copy);

template <typename T>
perflibs_vec<T> &perflibs_vec<T>::operator=(const perflibs_vec<T> &other) {
  if (&other == this) {
    return *this;
  }
  // Copy the vector variables
  index_base = other.index_base;
  n = other.n;
  nnz = other.nnz;
  if (n >= 0) {
    // If the vectors are populated make the const pointers point to them
    // otherwise construct a new vector
    copy_from_vector_or_ptr(&indx_ptr, indx, other.indx_ptr, other.indx, nnz);

    copy_from_vector_or_ptr(&vals_ptr, vals, other.vals_ptr, other.vals, nnz);
  }

  return *this;
}
template perflibs_vec<float> &
perflibs_vec<float>::operator=(const perflibs_vec<float> &other);
template perflibs_vec<double> &
perflibs_vec<double>::operator=(const perflibs_vec<double> &other);
template perflibs_vec<std::complex<float>> &
perflibs_vec<std::complex<float>>::operator=(
    const perflibs_vec<std::complex<float>> &other);
template perflibs_vec<std::complex<double>> &
perflibs_vec<std::complex<double>>::operator=(
    const perflibs_vec<std::complex<double>> &other);

} // end namespace perflibs::sparse
