/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#pragma once

#include "matrix_statistics.hpp"
#include "types.hpp"

namespace perflibs::sparse {

/**
 Perform the inner loop in Gustavson's algorithm. Iterate over a row of B
 multiplying by the fixed value of A.
 @param valA [in] fixed value of matrix A
 @param transB [in] transpose option for matrix B (used to communicate whether
 conjugate is required)
 @param index_base [in] base indexing, 0 or 1
 @param rB [in] the row of B we are looping over
 @param row_ptrB [in] the full CSR row_ptr array for B
 @param col_indxB [in] the full CSR col_indx array for B
 @param valsB [in] the full CSR vals array for B
 @param x [in,out] the dense accumulator in use
 @param xbool [in,out] array to mark whether a column has already been updated
 for this row
 @param i_row_ptr_C [in,out] index to current entry in CSR NNZ-length arrays
 @param col_indxC [in,out] the full CSR col_indx array for C
 @param i [in] the current row of A and C being processed
 */
template <typename T>
void spmm_csr_mm_gustavson_inner(T valA, perflibs_sparse_hint_value transB,
                                 perflibs_int_t index_base, perflibs_int_t rB,
                                 const perflibs_int_t *row_ptrB,
                                 const perflibs_int_t *col_indxB,
                                 const T *valsB, T *x, perflibs_int_t *xbool,
                                 perflibs_int_t &i_row_ptr_C,
                                 perflibs_int_t *col_indxC, perflibs_int_t i);

template <typename T>
void spmm_csr_mm_gustavson_inner_vals_only(
    T valA, perflibs_sparse_hint_value transB, perflibs_int_t index_base,
    perflibs_int_t rB, const perflibs_int_t *row_ptrB,
    const perflibs_int_t *col_indxB, const T *valsB, T *x);

// Inner kernel type for full allocating version of Gustavson's algorithm
template <typename T>
using gs_kernel_t = decltype(&spmm_csr_mm_gustavson_inner<T>);

template <typename T>
gs_kernel_t<T> get_gustavson_kernel_default(perflibs_sparse_hint_value transA,
                                            perflibs_sparse_hint_value transB,
                                            perflibs_int_t m, perflibs_int_t n,
                                            T alpha);

// Inner kernel type for non-allocating version of Gustavson's algorithm
template <typename T>
using gs_na_kernel_t = decltype(&spmm_csr_mm_gustavson_inner_vals_only<T>);

template <typename T>
gs_na_kernel_t<T>
get_gustavson_kernel_vals_only_default(perflibs_sparse_hint_value transA,
                                       perflibs_sparse_hint_value transB,
                                       perflibs_int_t m, perflibs_int_t n,
                                       T alpha, sparse_matrix_statistics stats);

/**
 * \f$C = \alpha*A*B\f$ for CSR matrices A, B and C. A and B must have a
 * matching base index and matching k (i.e. this performs
 * transA=N transB=N multiplication). Gustavson's algorithm, single phase.
 * Builds up all std::vector data structures from scratch, serial only.
 * std::vector is used in preference to perflibs::sparse::pod_vector for its
 * faster incremental memory management.
 * @param transA [in] transpose option for matrix A, used only for conjugation
 * @param transB [in] transpose option for matrix B, used only for conjugation
 * @param m [in] number of rows of C
 * @param n [in] number of columns of C
 * @param alpha [in] scalar multiplier
 * @param row_ptrA [in] pointer to array of row pointers for A
 * @param col_indxA [in] pointer to array of column indices for A
 * @param valsA [in] pointer to non-zero array of values of A
 * @param row_ptrB [in] pointer to array of row pointers for B
 * @param col_indxB [in] pointer to array of column indices for B
 * @param valsB [in] pointer to non-zero array of values of B
 * @param row_ptrC [in,out] std::vector of row pointers for C, empty on input
 * @param col_indxC [in,out] std::vector of column indices for C, empty on input
 * @param valsC [in,out] non-zero std::vector of values of C, empty on input
 */
template <typename T>
void spmm_csr_gustavson(enum perflibs_sparse_hint_value transA,
                        enum perflibs_sparse_hint_value transB,
                        perflibs_int_t m, perflibs_int_t n, T alpha,
                        const perflibs_int_t *row_ptrA,
                        const perflibs_int_t *col_indxA, const T *valsA,
                        const perflibs_int_t *row_ptrB,
                        const perflibs_int_t *col_indxB, const T *valsB,
                        std::vector<perflibs_int_t> &row_ptrC,
                        std::vector<perflibs_int_t> &col_indxC,
                        std::vector<T> &valsC);

/**
 * \f$C = \alpha*A*B\f$ for CSR matrices A, B and C. A and B must have a
 * matching base index and matching k (i.e. this performs
 * transA=N transB=N multiplication). Gustavson's algorithm, numeric phase. Data
 * structures are pre-allocated and row_ptr is filled in. This function computes
 * col_indx and vals.
 * @param transA [in] transpose option for matrix A, used only for conjugation
 * @param transB [in] transpose option for matrix B, used only for conjugation
 * @param m [in] number of rows of C
 * @param n [in] number of columns of C
 * @param alpha [in] scalar multiplier
 * @param row_ptrA [in] pointer to array of row pointers for A
 * @param col_indxA [in] pointer to array of column indices for A
 * @param valsA [in] pointer to non-zero array of values of A
 * @param row_ptrB [in] pointer to array of row pointers for B
 * @param col_indxB [in] pointer to array of column indices for B
 * @param valsB [in] pointer to non-zero array of values of B
 * @param row_ptrC [in] row pointers for C
 * @param col_indxC [in, out] column indices for C, set in this function
 * @param valsC [in,out] non-zero values of C, set in this function
 */
template <typename T>
void spmm_csr_gustavson_noalloc(enum perflibs_sparse_hint_value transA,
                                enum perflibs_sparse_hint_value transB,
                                perflibs_int_t m, perflibs_int_t n, T alpha,
                                const perflibs_int_t *row_ptrA,
                                const perflibs_int_t *col_indxA, const T *valsA,
                                const perflibs_int_t *row_ptrB,
                                const perflibs_int_t *col_indxB, const T *valsB,
                                const perflibs_int_t *row_ptrC,
                                perflibs_int_t *col_indxC, T *valsC);

/**
 * \f$C = \alpha*A*B\f$ for CSR matrices A, B and C. A and B must have a
 * matching base index and matching k (i.e. this performs
 * transA=N transB=N multiplication). Gustavson's algorithm, numeric phase. Data
 * structures are pre-allocated and row_ptr and col_indx are filled in. This
 * function computes vals only.
 * @param transA [in] transpose option for matrix A, used only for conjugation
 * @param transB [in] transpose option for matrix B, used only for conjugation
 * @param m [in] number of rows of C
 * @param n [in] number of columns of C
 * @param alpha [in] scalar multiplier
 * @param row_ptrA [in] pointer to array of row pointers for A
 * @param col_indxA [in] pointer to array of column indices for A
 * @param valsA [in] pointer to non-zero array of values of A
 * @param row_ptrB [in] pointer to array of row pointers for B
 * @param col_indxB [in] pointer to array of column indices for B
 * @param valsB [in] pointer to non-zero array of values of B
 * @param row_ptrC [in] row pointers for C
 * @param col_indxC [in] column indices for C
 * @param valsC [in,out] non-zero values of C, set in this function
 */
template <typename T>
void spmm_csr_gustavson_noalloc_vals_only(
    enum perflibs_sparse_hint_value transA,
    enum perflibs_sparse_hint_value transB, perflibs_int_t m, perflibs_int_t n,
    T alpha, const perflibs_int_t *row_ptrA, const perflibs_int_t *col_indxA,
    const T *valsA, const perflibs_int_t *row_ptrB,
    const perflibs_int_t *col_indxB, const T *valsB,
    const perflibs_int_t *row_ptrC, const perflibs_int_t *col_indxC, T *valsC,
    sparse_matrix_statistics stats);

} // end namespace perflibs::sparse
