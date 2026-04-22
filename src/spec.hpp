/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#pragma once
#include "matmul_gustavson.hpp"
#include "sell_c_sigma.hpp"

namespace perflibs::sparse {

/**
 * This function returns the valid C for the given type in the
 * order of preference to be used during auditioning.
 */
template <typename T>
const std::vector<std::pair<int, scs_spmv_kernels<T>>> &scs_get_valid_C(T vals);

/// Returns the function to use for the inner kernel of Gustavson's algorithm
template <typename T>
gs_kernel_t<T> get_gustavson_kernel(enum perflibs_sparse_hint_value transA,
                                    enum perflibs_sparse_hint_value transB,
                                    perflibs_int_t m, perflibs_int_t n,
                                    T alpha);

/// Returns the function to use for the inner kernel of Gustavson's algorithm
/// when we are only computing vals
template <typename T>
gs_na_kernel_t<T>
get_gustavson_kernel_vals_only(enum perflibs_sparse_hint_value transA,
                               enum perflibs_sparse_hint_value transB,
                               perflibs_int_t m, perflibs_int_t n, T alpha,
                               sparse_matrix_statistics stats);

/// Determine whether or not we really need to compute mgmd
template <typename T> bool get_if_mgmd_is_required();

} // namespace perflibs::sparse
