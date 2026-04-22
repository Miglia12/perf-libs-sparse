/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#pragma once

#include "util.hpp"

#include <string_view>

struct perflibs_spmat_top_t;

namespace perflibs::sparse {

template <typename T>
perflibs_status_t spmm_optimize(enum perflibs_sparse_hint_value transA,
                                enum perflibs_sparse_hint_value transB,
                                perflibs_sparse_hint_value alpha,
                                perflibs_spmat_t A, perflibs_spmat_t B,
                                perflibs_sparse_hint_value beta,
                                perflibs_spmat_t C);

template <typename T>
perflibs_status_t spmm_exec(enum perflibs_sparse_hint_value transA,
                            enum perflibs_sparse_hint_value transB, T alpha,
                            perflibs_spmat_t A, perflibs_spmat_t B, T beta,
                            perflibs_spmat_t C);

template <typename T>
perflibs_status_t
spelmm_optimize(perflibs_sparse_hint_value transA,
                perflibs_sparse_hint_value transB,
                perflibs_sparse_hint_value alpha, perflibs_spmat_top_t *A,
                perflibs_spmat_top_t *B, perflibs_sparse_hint_value beta,
                perflibs_spmat_top_t *C);

template <typename T>
perflibs_status_t
sddmm_optimize(perflibs_sparse_hint_value transA,
               perflibs_sparse_hint_value transB,
               perflibs_sparse_hint_value alpha, perflibs_spmat_top_t *A,
               perflibs_spmat_top_t *B, perflibs_sparse_hint_value beta,
               perflibs_spmat_top_t *C);

template <typename T>
perflibs_status_t spelmm_exec(perflibs_sparse_hint_value transA,
                              perflibs_sparse_hint_value transB, T alpha,
                              perflibs_spmat_t A, perflibs_spmat_t B, T beta,
                              perflibs_spmat_t C);

template <typename T, bool use_gemm>
perflibs_status_t sddmm_exec(perflibs_sparse_hint_value transA,
                             perflibs_sparse_hint_value transB, T alpha,
                             perflibs_spmat_t A, perflibs_spmat_t B, T beta,
                             perflibs_spmat_t C);

template <typename T>
perflibs_status_t spmm_check_params(enum perflibs_sparse_hint_value transA,
                                    enum perflibs_sparse_hint_value transB,
                                    enum perflibs_sparse_hint_value alpha,
                                    perflibs_spmat_t A, perflibs_spmat_t B,
                                    enum perflibs_sparse_hint_value beta,
                                    perflibs_spmat_t C);

/**
 * Translate a string description of one of the SpMM strategies into its
 * corresponding hint value, useful in testing & benchmarking.
 * @param s [in]	The name of the the strategy represented as a string.
 */
perflibs_sparse_hint_value get_spmm_strategy_from_string(std::string_view s);

} // namespace perflibs::sparse
