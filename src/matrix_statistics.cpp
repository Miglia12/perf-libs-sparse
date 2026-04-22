/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "spec.hpp"
#include "types.hpp"

namespace perflibs::sparse {

// By default, don't compute anything!
template <typename T> bool get_if_mgmd_is_required_default() { return false; }

template bool get_if_mgmd_is_required_default<float>();
template bool get_if_mgmd_is_required_default<double>();
template bool get_if_mgmd_is_required_default<std::complex<float>>();
template bool get_if_mgmd_is_required_default<std::complex<double>>();

} // end namespace perflibs::sparse

template <typename T> void perflibs_spmat_impl_t<T>::fill_matrix_stats_mgmd() {
  if (!perflibs::sparse::get_if_mgmd_is_required<T>()) {
    return;
  }

  auto impl_A = this;
  auto index_base = impl_A->index_base;

  if (impl_A->spmat_format == perflibs_format_csr) {
    auto m = impl_A->csr.m;
    auto row_ptr = impl_A->csr.row_ptr_ptr;
    auto col_indx = impl_A->csr.col_indx_ptr;

    // For each row of the matrix compute the geometric mean of
    // distances between successive non-zeros. We do this by accumulating
    // log values additively, rather than multipliying differences
    // in order to avoid running into overflow problems.
#if defined(_WIN32)
    // OpenMP parallel reduction gives an incorrect result on Windows.
    // Do reduction manually until HPCL3-1422 is resolved.
    auto nthreads = perflibs::sparse::omp::get_max_threads();
    std::vector<double> mgmd(nthreads, 0.0);
#pragma omp parallel shared(mgmd)
    {
      auto tid = perflibs::sparse::omp::get_thread_num();
#pragma omp for
      for (perflibs_int_t i = 0; i < m; i++) {
        auto len = row_ptr[i + 1] - row_ptr[i];
        if (len) {
          double diff = 0.0;
          auto indx_cur = col_indx[row_ptr[i] - index_base];
          perflibs_int_t indx_last;
          for (auto j = row_ptr[i] + 1; j < row_ptr[i + 1]; j++) {
            indx_last = indx_cur;
            indx_cur = col_indx[j - index_base];
            diff += std::log(std::abs(indx_cur - indx_last));
          }
          mgmd[tid] += std::exp(diff / len);
        }
      }
    }
    for (perflibs_int_t i = 1; i < nthreads; ++i) {
      mgmd[0] += mgmd[i];
    }
    mgmd[0] /= m;
    impl_A->stats.mgmd = mgmd[0];
#else
    double mgmd = 0.0;
#pragma omp parallel for reduction(+ : mgmd)
    for (perflibs_int_t i = 0; i < m; i++) {
      auto len = row_ptr[i + 1] - row_ptr[i];
      if (len) {
        double diff = 0.0;
        auto indx_cur = col_indx[row_ptr[i] - index_base];
        perflibs_int_t indx_last;
        for (auto j = row_ptr[i] + 1; j < row_ptr[i + 1]; j++) {
          indx_last = indx_cur;
          indx_cur = col_indx[j - index_base];
          diff += std::log(std::abs(indx_cur - indx_last));
        }
        mgmd += std::exp(diff / len);
      }
    }
    mgmd /= m;
    impl_A->stats.mgmd = mgmd;
#endif
  }
}

template void perflibs_spmat_impl_t<float>::fill_matrix_stats_mgmd();
template void perflibs_spmat_impl_t<double>::fill_matrix_stats_mgmd();
template void
perflibs_spmat_impl_t<std::complex<float>>::fill_matrix_stats_mgmd();
template void
perflibs_spmat_impl_t<std::complex<double>>::fill_matrix_stats_mgmd();
