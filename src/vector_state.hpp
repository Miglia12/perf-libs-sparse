/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#pragma once

#include "util.hpp"

namespace perflibs::sparse {

template <typename T>
perflibs_status_t export_spvec(perflibs_spvec_t x, perflibs_int_t *index_base,
                               perflibs_int_t *n, perflibs_int_t *nnz,
                               perflibs_int_t *indx, T *vals);

template <typename T>
perflibs_status_t gather_spvec_top(const T *x_d, perflibs_int_t index_base,
                                   perflibs_int_t n, perflibs_spvec_t *x_s,
                                   perflibs_int_t flags);

template <typename T>
perflibs_status_t scatter_spvec_top(perflibs_spvec_t x_s, T *x_d);

template <typename T>
perflibs_status_t update_spvec(perflibs_spvec_t x, perflibs_int_t n_updates,
                               const perflibs_int_t *indx, const T *vals);

template <bool IsConj, typename T>
perflibs_status_t dot_exec_top(perflibs_spvec_t x, const T *y, T *result);

template <typename T>
perflibs_status_t axpby_exec_top(const T alpha, perflibs_spvec_t x,
                                 const T beta, T *y);

template <typename T>
perflibs_status_t waxpby_exec_top(const T alpha, perflibs_spvec_t x,
                                  const T beta, const T *y, T *w);

template <typename T1, typename T2>
perflibs_status_t rot_exec_top(perflibs_spvec_t x, T1 *y,
                               perflibs::sparse::remove_complex_t<T1> c, T2 s);

void spvec_print_err(perflibs_spvec_t x);

} // namespace perflibs::sparse
