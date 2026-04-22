/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#pragma once

#include "pod_vector.hpp"
#include "util.hpp"

namespace perflibs::sparse {

// An internal type for sparse vector structures
template <typename T> struct perflibs_vec {
  /// The base index
  int64_t index_base;
  /// The number of elements in the vector
  int64_t n;
  /// The number of non-zero elements
  int64_t nnz;

  /// Indices of the non-zero values in the vector. This
  /// is populated when user data has been copied, otherwise it is empty.
  perflibs::sparse::pod_vector<perflibs_int_t> indx;
  /// Non-zero values in the vector. This is populated when user data
  /// has been copied, otherwise it is empty.
  perflibs::sparse::pod_vector<T> vals;

  /// Pointer to either an array of indices provided by a user,
  /// or to the base of #indx if user data has been copied.
  const perflibs_int_t *indx_ptr;
  /// Pointer to either an array of values in the vector provided by a user,
  /// or to the base of #vals if user data has been copied.
  const T *vals_ptr;

  /**
   * Constructor of an empty vector
   */
  perflibs_vec() : n(-1), indx_ptr(nullptr), vals_ptr(nullptr) {}

  /**
   * Data copying constructor: we take copies of user arrays
   * @param [in] index_base	Base index
   * @param [in] n		Number of rows
   * @param [in] nnz		Number of non-zeros
   * @param [in] vals		Array of non-zero values of the matrix
   * @param [in] indx		Array of indices for non-zero values
   */
  perflibs_vec(int64_t index_base, int64_t n, int64_t nnz,
               const perflibs_int_t *indx, const T *vals)
      : index_base(index_base), n(n), nnz(nnz), indx(indx, indx + nnz),
        vals(vals, vals + nnz), indx_ptr(this->indx.data()),
        vals_ptr(this->vals.data()) {}

  /**
   * Non-data-copying constructor: pointers to user arrays are assigned
   * @param [in] index_base	Base index
   * @param [in] n		Number of rows
   * @param [in] vals		Array of non-zero values of the matrix
   * @param [in] indx		Array of indices for non-zero values
   * @param [in] nnz		Number of non-zeros
   */
  perflibs_vec(int64_t index_base, int64_t n, const perflibs_int_t *indx,
               const T *vals, int64_t nnz)
      : index_base(index_base), n(n), nnz(nnz), indx_ptr(indx), vals_ptr(vals) {
  }

  perflibs_vec &operator=(const perflibs_vec &other);
  perflibs_vec(const perflibs_vec &other) { *this = other; };
  perflibs_vec &operator=(perflibs_vec &&other) = default;
  perflibs_vec(perflibs_vec &&other) = default;
};

/**
 * Populate x with new details after checking params. This version takes raw
 * pointers, which may be pointers to user data.
 * @param [inout] x		Sparse vector to populate
 * @param [in] index_base	base index - can be either 0 or 1
 * @param [in] n		Number of elements
 * @param [in] nnz		Number of non-zero values
 * @param [in] indx		Array of indices of non-zero positions
 * @param [in] vals		Array of non-zero values
 * @param [in] no_copy		Do we want to take a copy of the arrays, or just
 * set pointers?
 */
template <typename T>
perflibs_status_t
fill_initial_data_vec(perflibs_spvec_t x, perflibs_int_t index_base,
                      perflibs_int_t n, perflibs_int_t nnz,
                      const perflibs_int_t *indx, const T *vals, bool no_copy);

} // end namespace perflibs::sparse
