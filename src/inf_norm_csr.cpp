/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "inf_norm_csr.hpp"

namespace perflibs::sparse {

template <typename T>
void spnorm_inf_csr(const perflibs_csr<T> &csr,
                    perflibs::sparse::remove_complex_t<T> *result) {
  using RT = perflibs::sparse::remove_complex_t<T>;

  RT rowsum_max = 0;
  auto index_base = csr.row_ptr_ptr[0];
  for (auto i = 0; i < csr.m; i++) {
    RT rowsum = 0;
    for (auto j = csr.row_ptr_ptr[i] - index_base;
         j < csr.row_ptr_ptr[i + 1] - index_base; j++) {
      rowsum += std::abs(csr.vals_ptr[j]);
    }
    if (rowsum > rowsum_max || perflibs::sparse::is_real_nan(rowsum)) {
      rowsum_max = rowsum;
    }
  }
  *result = rowsum_max;
}
template void spnorm_inf_csr<float>(const perflibs_csr<float> &csr,
                                    float *result);
template void spnorm_inf_csr<double>(const perflibs_csr<double> &csr,
                                     double *result);
template void spnorm_inf_csr<std::complex<float>>(
    const perflibs_csr<std::complex<float>> &csr, float *result);
template void spnorm_inf_csr<std::complex<double>>(
    const perflibs_csr<std::complex<double>> &csr, double *result);

} // namespace perflibs::sparse
