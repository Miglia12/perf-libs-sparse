/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#pragma once

#include "types.hpp"
#include <memory>
#include <utility>

namespace perflibs::sparse {

template <typename T>
std::pair<perflibs_status_t, std::unique_ptr<perflibs_spmat_top_t>>
spadd_dispatch(perflibs_sparse_hint_value transa,
               perflibs_sparse_hint_value transb, T alpha, perflibs_spmat_t A,
               T beta, perflibs_spmat_t B, bool exec);

template <typename T>
perflibs_status_t spadd_exec(perflibs_sparse_hint_value transa,
                             perflibs_sparse_hint_value transb, T alpha,
                             perflibs_spmat_t A, T beta, perflibs_spmat_t B,
                             perflibs_spmat_t C);

template <typename T>
perflibs_status_t spadd_optimize(perflibs_sparse_hint_value transa,
                                 perflibs_sparse_hint_value transb,
                                 perflibs_sparse_hint_value alpha,
                                 perflibs_spmat_t A,
                                 perflibs_sparse_hint_value beta,
                                 perflibs_spmat_t B, perflibs_spmat_t C);

} // namespace perflibs::sparse
