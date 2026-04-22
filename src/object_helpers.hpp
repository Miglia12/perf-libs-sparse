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

namespace perflibs::sparse {

template <typename T> perflibs_datatype get_datatype();

inline bool is_special(spmat_format_t fmt) {
  return fmt == perflibs_format_null || fmt == perflibs_format_identity;
}

/**
 * If the input vector \p other_vec is populated, copies this into output vector
 * \p vec. Otherwise instantiates \p vec using data pointed to by \p other_ptr
 * @param [out] ptr       At the end of the routine, points to the data stored
 * in \p vec
 * @param [out] vec       The vector to populate with data
 * @param [in] other_ptr  May point to data to copy in to \p vec. Must be
 * non-null in the case that \p other_vec is empty
 * @param [in] other_vec  May contain data to copy to \p vec. If this is empty,
 * data is copied from \c other_ptr instead
 * @param [in] len        The number of elements to copy from \p other_ptr
 */
template <typename T, template <typename...> typename V>
void copy_from_vector_or_ptr(const T **ptr, V<T> &vec, const T *other_ptr,
                             const V<T> &other_vec, perflibs_int_t len);

template <typename T>
std::unique_ptr<perflibs_spmat_top_t> spmat_copy(perflibs_spmat_t A);

template <typename T>
std::unique_ptr<perflibs_spmat_top_t> spmat_copy(perflibs_const_spmat_t A);

template <typename T>
perflibs_status_t create_spmat_top_csr(perflibs_spmat_top_t **A,
                                       perflibs_int_t m, perflibs_int_t n,
                                       const perflibs_int_t *row_ptr,
                                       const perflibs_int_t *col_indx,
                                       const T *vals, perflibs_int_t flags);
template <typename T>
perflibs_status_t create_spmat_top_csc(perflibs_spmat_top_t **A,
                                       perflibs_int_t m, perflibs_int_t n,
                                       const perflibs_int_t *row_indx,
                                       const perflibs_int_t *col_ptr,
                                       const T *vals, perflibs_int_t flags);
template <typename T>
perflibs_status_t
create_spmat_top_coo(perflibs_spmat_top_t **A, perflibs_int_t m,
                     perflibs_int_t n, perflibs_int_t nnz,
                     perflibs_int_t index_base, const perflibs_int_t *row_indx,
                     const perflibs_int_t *col_indx, const T *vals,
                     perflibs_int_t flags);
template <typename T>
perflibs_status_t
create_spmat_top_dense(perflibs_spmat_top_t **A, perflibs_dense_layout layout,
                       perflibs_int_t m, perflibs_int_t n, perflibs_int_t lda,
                       perflibs_int_t index_base, const T *vals,
                       perflibs_int_t flags);
template <typename T>
perflibs_status_t create_spmat_top_bsr(perflibs_spmat_top_t **A,
                                       perflibs_dense_layout block_layout,
                                       perflibs_int_t m, perflibs_int_t n,
                                       perflibs_int_t block_size,
                                       const perflibs_int_t *row_ptr,
                                       const perflibs_int_t *col_indx,
                                       const T *vals, perflibs_int_t flags);
template <typename T>
perflibs_status_t create_spmat_top_supernodal(
    perflibs_spmat_top_t **A, perflibs_int_t m, perflibs_int_t n,
    perflibs_int_t nsuper, perflibs_int_t nparts,
    const perflibs_int_t *super_row_ptr, const perflibs_int_t *super_col_indx,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_ptr,
    const T *vals, const perflibs_int_t *part_indx, perflibs_int_t flags);

template <typename T>
perflibs_status_t create_spvec_top(perflibs_spvec_top_t **x,
                                   perflibs_int_t index_base, perflibs_int_t n,
                                   perflibs_int_t nnz,
                                   const perflibs_int_t *indx, const T *vals,
                                   perflibs_int_t flags);

template <typename T> std::unique_ptr<perflibs_spmat_top_t> create_new_matrix();

template <typename T> std::unique_ptr<perflibs_spvec_top_t> create_new_vector();

// Used in testing
template <typename T>
perflibs_status_t
create_spmat_csr(perflibs_spmat_top_t **A, perflibs_int_t m, perflibs_int_t n,
                 const perflibs_int_t *row_ptr, const perflibs_int_t *col_indx,
                 const T *vals, perflibs_int_t flags);
template <typename T>
perflibs_status_t
create_spmat_csc(perflibs_spmat_top_t **A, perflibs_int_t m, perflibs_int_t n,
                 const perflibs_int_t *row_indx, const perflibs_int_t *col_ptr,
                 const T *vals, perflibs_int_t flags);
template <typename T>
perflibs_status_t
create_spmat_coo(perflibs_spmat_top_t **A, perflibs_int_t m, perflibs_int_t n,
                 perflibs_int_t nnz, perflibs_int_t index_base,
                 const perflibs_int_t *row_indx, const perflibs_int_t *col_indx,
                 const T *vals, perflibs_int_t flags);
template <typename T>
perflibs_status_t
create_spmat_dense(perflibs_spmat_top_t **A, perflibs_dense_layout layout,
                   perflibs_int_t m, perflibs_int_t n, perflibs_int_t lda,
                   const T *vals, perflibs_int_t flags);
template <typename T>
perflibs_status_t
create_spmat_bsr(perflibs_spmat_top_t **A, perflibs_dense_layout block_layout,
                 perflibs_int_t m, perflibs_int_t n, perflibs_int_t block_size,
                 const perflibs_int_t *row_ptr, const perflibs_int_t *col_indx,
                 const T *vals, perflibs_int_t flags);
template <typename T>
perflibs_status_t create_spmat_supernodal(
    perflibs_spmat_top_t **A, perflibs_int_t m, perflibs_int_t n,
    perflibs_int_t nsuper, perflibs_int_t nparts,
    const perflibs_int_t *super_row_ptr, const perflibs_int_t *super_col_indx,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_ptr,
    const T *vals, const perflibs_int_t *part_indx, perflibs_int_t flags);

template <typename T>
perflibs_status_t create_spvec(perflibs_spvec_top_t **x,
                               perflibs_int_t index_base, perflibs_int_t n,
                               perflibs_int_t nnz, perflibs_int_t *indx,
                               const T *vals, perflibs_int_t flags);

} // namespace perflibs::sparse
