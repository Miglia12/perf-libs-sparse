/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#pragma once

#include "types.hpp"

namespace perflibs::sparse {

template <typename T>
perflibs_status_t spmv_exec_identity(perflibs_sparse_hint_value trans, T alpha,
                                     perflibs_spmat_impl_t<T> *impl, const T *x,
                                     T beta, T *y);

template <typename T>
perflibs_status_t spmv_exec_null(perflibs_sparse_hint_value trans,
                                 perflibs_spmat_impl_t<T> *impl, T beta, T *y);

} // namespace perflibs::sparse
