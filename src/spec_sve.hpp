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

template <typename T>
const std::vector<std::pair<int, scs_spmv_kernels<T>>> &
scs_get_valid_C_sve(T vals) {
  return scs_get_valid_C_default(vals);
}

// Forward declarations for specializations where we don't want the compiler to
// use the above template definition.
template <>
const std::vector<std::pair<int, scs_spmv_kernels<double>>> &
scs_get_valid_C_sve(double vals);
template <>
const std::vector<std::pair<int, scs_spmv_kernels<float>>> &
scs_get_valid_C_sve(float vals);

template <typename T>
gs_kernel_t<T> get_gustavson_kernel_sve(enum perflibs_sparse_hint_value transA,
                                        enum perflibs_sparse_hint_value transB,
                                        perflibs_int_t m, perflibs_int_t n,
                                        T alpha);

template <typename T>
gs_na_kernel_t<T>
get_gustavson_kernel_vals_only_sve(enum perflibs_sparse_hint_value transA,
                                   enum perflibs_sparse_hint_value transB,
                                   perflibs_int_t m, perflibs_int_t n, T alpha,
                                   sparse_matrix_statistics stats);

template <typename T> bool get_if_mgmd_is_required_sve();

} // end namespace perflibs::sparse
