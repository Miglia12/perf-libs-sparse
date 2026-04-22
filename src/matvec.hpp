/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#pragma once

#include "util.hpp"

struct perflibs_spmat_top_t;
template <typename T> struct perflibs_spmat_impl_t;

namespace perflibs::sparse {

template <typename T>
perflibs_status_t spmv_optimize(perflibs_spmat_impl_t<T> *impl);

template <typename T>
perflibs_status_t spmv_exec(perflibs_sparse_hint_value trans, T alpha,
                            perflibs_spmat_top_t *A, const T *x, T beta, T *y);

template <typename T>
perflibs_status_t spmv_exec_impl(perflibs_sparse_hint_value trans, T alpha,
                                 perflibs_spmat_impl_t<T> *impl, const T *x,
                                 T beta, T *y);

template <typename T>
perflibs_status_t spmat_update(perflibs_spmat_top_t *A,
                               perflibs_int_t n_updates,
                               const perflibs_int_t *row_indx,
                               const perflibs_int_t *col_indx, const T *vals);

template <typename T>
void audition_spmv(perflibs_spmat_impl_t<T> *impl, perflibs_int_t C_force,
                   perflibs_int_t sigma_force);

template <typename T> void audition_spmv(perflibs_spmat_impl_t<T> *impl);

} // namespace perflibs::sparse
