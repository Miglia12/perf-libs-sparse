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
perflibs_status_t spmat_export_csr(perflibs_const_spmat_t A,
                                   perflibs_int_t index_base, perflibs_int_t *m,
                                   perflibs_int_t *n, perflibs_int_t **row_ptr,
                                   perflibs_int_t **col_indx, T **vals);

template <typename T>
perflibs_status_t spmat_export_csc(perflibs_const_spmat_t A,
                                   perflibs_int_t index_base, perflibs_int_t *m,
                                   perflibs_int_t *n, perflibs_int_t **row_indx,
                                   perflibs_int_t **col_ptr, T **vals);

template <typename T>
perflibs_status_t spmat_export_coo(perflibs_const_spmat_t A, perflibs_int_t *m,
                                   perflibs_int_t *n, perflibs_int_t *nnz,
                                   perflibs_int_t **row_indx,
                                   perflibs_int_t **col_indx, T **vals);

template <typename T>
perflibs_status_t
spmat_export_dense(perflibs_const_spmat_t A, enum perflibs_dense_layout layout,
                   perflibs_int_t *m, perflibs_int_t *n, T **vals);

template <typename T>
perflibs_status_t
spmat_export_bsr(perflibs_const_spmat_t A,
                 enum perflibs_dense_layout block_layout,
                 perflibs_int_t index_base, perflibs_int_t *m,
                 perflibs_int_t *n, perflibs_int_t *block_size,
                 perflibs_int_t **row_ptr, perflibs_int_t **col_indx, T **vals);

} // namespace perflibs::sparse
