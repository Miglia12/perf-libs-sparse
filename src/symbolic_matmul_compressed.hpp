/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#pragma once

#include "pod_vector.hpp"
#include "util.hpp"
#include <cstdint>
#include <utility>
#include <vector>

namespace perflibs::sparse {

/**
 * A structure to represent a sparse matrix in the compressed format described
 * in Multi-threaded Sparse Matrix-Matrix Multiplication for Many-Core and GPU
 * Architectures, Trott (Sanida) et. al. Turn the column indices into 64-bit
 * blocks of bitwise masks in the array cs, using a second array csi to record
 * the multiple of 64 which the block represents.
 */
struct symbolic_matmul_compressed {

  /// Set bits in cs indicate non-zero columns
  perflibs::sparse::pod_vector<uint64_t *> cs;
  /// csi[i] denotes the multiple of 64=8*sizeof(uint64_t) that cs[i] starts at
  perflibs::sparse::pod_vector<perflibs_int_t *> csi;

  perflibs::sparse::pod_vector<perflibs_int_t> cs_len;

  /// The memory backing the vectors of pointers above
  perflibs::sparse::pod_vector<uint64_t> cs_buf;
  perflibs::sparse::pod_vector<perflibs_int_t> csi_buf;

  symbolic_matmul_compressed() = default;
  symbolic_matmul_compressed(perflibs_int_t m, const perflibs_int_t *row_ptr,
                             const perflibs_int_t *col_indx);
};

/**
 * Check that the column indices within each row of the input matrix are
 * ordered. If so, construct a new symbolic_matmul_compressed object, otherwise
 * return false.
 *
 * @param [in] m		Number of rows of the matrix.
 * @param [in] row_ptr	Row pointers array.
 * @param [in] col_indx Column indices array.
 * @returns A pair containing a @p bool and a @p symbolic_matmul_compressed
 * object.
 * If the @p bool is true then the check passed and we successfully constructed
 * a symbolic_matmul_compressed object, otherwise the check failed and we
 * reutrned
 * an empty object.
 */
std::pair<bool, symbolic_matmul_compressed>
make_symbolic_matmul_compressed(perflibs_int_t m, const perflibs_int_t *row_ptr,
                                const perflibs_int_t *col_indx);

/**
 * Function to compute the structure of the output matrix C (=A*B), given the
 * row_ptr and col_indx arrays for A and the compressed form of B. This function
 * computes the row lengths of C (row_ptrC) and also the column indices of C
 * (col_indxC).
 *
 * @param [in] m 			Number of rows of A and C
 * @param [in] n 			Number of columns of A and C
 * @param [in] row_ptrA 	Row pointers array for A
 * @param [in] col_indxA 	Column indices for A
 * @param [in] comp_B		Compressed representation of B
 * @param [in] index_base_B	Base index of matrix B in the original
 * representation (0 or 1)
 * @param [out] row_ptrC	Row pointers array for C, an empty vector on
 * entry
 * @param [out] col_indxC	Column indices for C, to be allocated and
 * populated in this function
 */
void get_symbolic_mm_structure(
    perflibs_int_t m, perflibs_int_t n, const perflibs_int_t *row_ptrA,
    const perflibs_int_t *col_indxA, const symbolic_matmul_compressed &comp_B,
    perflibs_int_t index_base_B, std::vector<perflibs_int_t> &row_ptrC,
    perflibs::sparse::pod_vector<perflibs_int_t> &col_indxC);

/**
 * Function to compute the structure of the output matrix C (=A*B), given the
 * row_ptr and col_indx arrays for A and the compressed form of B. This function
 * computes the row lengths of C (row_ptrC) and allocates space for the column
 * indices of C, but does not fill in the indices.
 *
 * @param [in] m 			Number of rows of A and C
 * @param [in] n 			Number of columns of A and C
 * @param [in] row_ptrA 	Row pointers array for A
 * @param [in] col_indxA 	Column indices for A
 * @param [in] comp_B		Compressed representation of B
 * @param [in] index_base_B	Base index of matrix B in the original
 * representation (0 or 1)
 * @param [out] row_ptrC	Row pointers array for C, an empty vector on
 * entry
 * @param [out] col_indxC	Column indices for C, to be allocated and
 * populated in this function
 */
void get_symbolic_mm_structure_alloc_only(
    perflibs_int_t m, perflibs_int_t n, const perflibs_int_t *row_ptrA,
    const perflibs_int_t *col_indxA, const symbolic_matmul_compressed &comp_B,
    perflibs_int_t index_base_B, std::vector<perflibs_int_t> &row_ptrC,
    perflibs::sparse::pod_vector<perflibs_int_t> &col_indxC);

} // end namespace perflibs::sparse
