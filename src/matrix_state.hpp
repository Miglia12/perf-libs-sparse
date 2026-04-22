/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#pragma once

#include "util.hpp"

#include <memory>

struct perflibs_spmat_top_t;
struct perflibs_spvec_top_t;
template <typename T> struct perflibs_spmat_impl_t;

namespace perflibs::sparse {

std::unique_ptr<perflibs_spmat_top_t> null_matrix(perflibs_int_t m,
                                                  perflibs_int_t n);

std::unique_ptr<perflibs_spmat_top_t> identity_matrix(perflibs_int_t n);

template <typename T>
perflibs_status_t scale_matrix(perflibs_sparse_hint_value trans, T alpha,
                               perflibs_spmat_t A);

perflibs_status_t basic_query(perflibs_spmat_top_t *A,
                              perflibs_int_t *index_base, perflibs_int_t *m,
                              perflibs_int_t *n, perflibs_int_t *nnz);

perflibs_status_t basic_query(perflibs_spvec_top_t *x,
                              perflibs_int_t *index_base, perflibs_int_t *n,
                              perflibs_int_t *nnz);

template <typename T>
perflibs_status_t set_hint(perflibs_spmat_impl_t<T> *impl,
                           perflibs_sparse_hint_type hint,
                           perflibs_sparse_hint_value value);

template <typename T> void set_time_limit(perflibs_spmat_top_t *A, double tl);

template <typename T> void set_C(perflibs_spmat_top_t *A, int C);

template <typename T> void set_sigma(perflibs_spmat_top_t *A, int sigma);

template <typename T> double get_time_limit();

void spmat_print_err(perflibs_spmat_t x);

} // namespace perflibs::sparse
