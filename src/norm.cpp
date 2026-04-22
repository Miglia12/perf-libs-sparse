/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "norm.hpp"

#include "block_sparse_rows.hpp"
#include "compressed_sparse_columns.hpp"
#include "coordinate_list.hpp"
#include "dense.hpp"
#include "inf_norm_csr.hpp"
#include "spnorm_constants.hpp"
#include "types.hpp"

#include <algorithm>
#include <cmath>

namespace perflibs::sparse {
template <typename T> T spnorm_max(const std::vector<T> rowsums) {
  T max_rowsum = 0;
  for (auto row = 0; row < (int)rowsums.size(); row++) {
    auto rowsum = rowsums[row];
    if (rowsum > max_rowsum || perflibs::sparse::is_real_nan(rowsum)) {
      max_rowsum = rowsum;
    }
  }
  return max_rowsum;
}
template float spnorm_max<float>(const std::vector<float> rowsums);
template double spnorm_max<double>(const std::vector<double> rowsums);

template <typename T>
void spnorm_accumulate(T abs_val, T &a_big, T &a_med, T &a_small) {
  if (abs_val > t_big<T>) {
    // Scale down
    T scaled_val = abs_val * s_big<T>;
    a_big += scaled_val * scaled_val;
  } else if (abs_val < t_small<T>) {
    // Scale up
    T scaled_val = abs_val * s_small<T>;
    a_small += scaled_val * scaled_val;
  } else {
    // No scaling
    a_med += abs_val * abs_val;
  }
}

template void spnorm_accumulate<float>(float abs_val, float &a_big,
                                       float &a_med, float &a_small);
template void spnorm_accumulate<double>(double abs_val, double &a_big,
                                        double &a_med, double &a_small);

template <typename T>
perflibs_status_t spnorm_frb(perflibs_spmat_impl_t<T> *impl,
                             perflibs::sparse::remove_complex_t<T> *result) {
  // Frobenius norm implementation, avoiding over/underflow
  // Parameters for safe scaling, t_big, s_small, t_small and s_big, are defined
  // in spnorm_constants.hpp t_big and s_small are large positive numbers
  // t_small and s_big are small positive numbers

  using RT = perflibs::sparse::remove_complex_t<T>;

  auto format = impl->spmat_format;
  const auto vals = format == perflibs_format_coo     ? impl->coo.vals_ptr
                    : format == perflibs_format_csr   ? impl->csr.vals_ptr
                    : format == perflibs_format_csc   ? impl->csc.vals_ptr
                    : format == perflibs_format_bsr   ? impl->bsr.vals_ptr
                    : format == perflibs_format_dense ? impl->dense.vals_ptr
                                                      : nullptr;
  if (!vals) {
    // Exit early if the format is unsupported
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }

  // Accumulators
  // Values < t_small will be scaled by s_small
  RT a_small = 0;
  // Unscaled values
  RT a_med = 0;
  // Values > t_big will be scaled by s_big
  RT a_big = 0;

  auto process_val = [&](const auto &val) {
    if constexpr (perflibs::sparse::is_complex_v<T>) {
      auto abs_val_real = std::abs(val.real());
      auto abs_val_imag = std::abs(val.imag());
      spnorm_accumulate(abs_val_real, a_big, a_med, a_small);
      spnorm_accumulate(abs_val_imag, a_big, a_med, a_small);
    } else {
      RT abs_val = std::abs(val);
      spnorm_accumulate(abs_val, a_big, a_med, a_small);
    }
  };

  if (format == perflibs_format_dense) {
    for (auto i = 0; i < impl->dense.m; i++) {
      for (auto j = 0; j < impl->dense.n; j++) {
        auto idx = impl->dense.layout == PERFLIBS_COL_MAJOR
                       ? i + (j * impl->dense.lda)
                       : j + (i * impl->dense.lda);
        process_val(vals[idx]);
      }
    }
  } else {
    for (auto i = 0; i < impl->nnz; i++) {
      process_val(vals[i]);
    }
  }

  if (a_big > 0) {
    // Include contributions from a_med and propagate NaNs
    // Any contributions from a_small can be ignored
    if (a_med > 0 || perflibs::sparse::is_real_nan(a_med)) {
      a_big += (a_med * s_big<RT>)*s_big<RT>;
    }
    *result = std::sqrt(a_big) / s_big<RT>;
  } else if (a_small > 0) {
    RT small_res = std::sqrt(a_small) / s_small<RT>;
    if (a_med > 0 || perflibs::sparse::is_real_nan(a_med)) {
      // We cannot add the contribution of a_med by doing (a_med * s_small) *
      // s_small as this will likely overflow. The problem we are trying to
      // solve is: result = sqrt(med_res^2 + small_res^2) Take y_max =
      // max(med_res, small_res) and multiply each term by y_max / y_max: result
      // = y_max * sqrt(1 + (y_min/y_max)^2)
      RT med_res = std::sqrt(a_med);
      RT y_min = std::min(med_res, small_res);
      RT y_max = std::max(med_res, small_res);
      *result = y_max * std::sqrt(1 + ((y_min / y_max) * (y_min / y_max)));
    } else {
      *result = small_res;
    }
  } else {
    *result = std::sqrt(a_med);
  }

  return PERFLIBS_STATUS_SUCCESS;
}
template perflibs_status_t spnorm_frb<float>(perflibs_spmat_impl_t<float> *impl,
                                             float *result);
template perflibs_status_t
spnorm_frb<double>(perflibs_spmat_impl_t<double> *impl, double *result);
template perflibs_status_t spnorm_frb<std::complex<float>>(
    perflibs_spmat_impl_t<std::complex<float>> *impl, float *result);
template perflibs_status_t spnorm_frb<std::complex<double>>(
    perflibs_spmat_impl_t<std::complex<double>> *impl, double *result);

template <typename T>
perflibs_status_t spnorm_inf(perflibs_spmat_impl_t<T> *impl,
                             perflibs::sparse::remove_complex_t<T> *result) {
  auto format = impl->spmat_format;
  switch (format) {
  case (perflibs_format_csr):
    spnorm_inf_csr<T>(impl->csr, result);
    break;
  case (perflibs_format_csc):
    spnorm_inf_csc<T>(impl->csc, result);
    break;
  case (perflibs_format_coo):
    spnorm_inf_coo<T>(impl->coo, result);
    break;
  case (perflibs_format_bsr):
    spnorm_inf_bsr<T>(impl->bsr, result);
    break;
  case (perflibs_format_dense):
    spnorm_inf_dense<T>(impl->dense, result);
    break;
  default:
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }

  return PERFLIBS_STATUS_SUCCESS;
}
template perflibs_status_t spnorm_inf<float>(perflibs_spmat_impl_t<float> *impl,
                                             float *result);
template perflibs_status_t
spnorm_inf<double>(perflibs_spmat_impl_t<double> *impl, double *result);
template perflibs_status_t spnorm_inf<std::complex<float>>(
    perflibs_spmat_impl_t<std::complex<float>> *impl, float *result);
template perflibs_status_t spnorm_inf<std::complex<double>>(
    perflibs_spmat_impl_t<std::complex<double>> *impl, double *result);

template <typename T>
perflibs_status_t spnorm_exec(perflibs_spmat_top_t *A, perflibs_sparse_norm nrm,
                              perflibs::sparse::remove_complex_t<T> *result) {
  // Return early if the norm is unrecognized
  if (nrm != PERFLIBS_SPARSE_NORM_INF && nrm != PERFLIBS_SPARSE_NORM_FRB) {
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }

  auto impl = reinterpret_cast<perflibs_spmat_impl_t<T> *>(A->impl);

  // Return early if there are no non-zero values
  // This includes the perflibs_format_null format
  if (impl->nnz == 0) {
    *result = 0;
    return PERFLIBS_STATUS_SUCCESS;
  }

  if (impl->spmat_format == perflibs_format_identity) {
    *result = nrm == PERFLIBS_SPARSE_NORM_FRB ? std::sqrt(impl->n) : 1;
    return PERFLIBS_STATUS_SUCCESS;
  }

  if (nrm == PERFLIBS_SPARSE_NORM_FRB) {
    return spnorm_frb<T>(impl, result);
  }

  return spnorm_inf<T>(impl, result);
}
template perflibs_status_t spnorm_exec<float>(perflibs_spmat_top_t *A,
                                              perflibs_sparse_norm nrm,
                                              float *result);
template perflibs_status_t spnorm_exec<double>(perflibs_spmat_top_t *A,
                                               perflibs_sparse_norm nrm,
                                               double *result);
template perflibs_status_t
spnorm_exec<std::complex<float>>(perflibs_spmat_top_t *A,
                                 perflibs_sparse_norm nrm, float *result);
template perflibs_status_t
spnorm_exec<std::complex<double>>(perflibs_spmat_top_t *A,
                                  perflibs_sparse_norm nrm, double *result);

} // namespace perflibs::sparse
