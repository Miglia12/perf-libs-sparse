/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "add.hpp"
#include "convert.hpp"
#include "matrix_state.hpp"
#include "object_helpers.hpp"
#include <algorithm>

namespace perflibs::sparse {

template <typename T>
perflibs_status_t spadd_check_params(enum perflibs_sparse_hint_value transA,
                                     enum perflibs_sparse_hint_value transB,
                                     enum perflibs_sparse_hint_value alpha,
                                     perflibs_spmat_t A,
                                     enum perflibs_sparse_hint_value beta,
                                     perflibs_spmat_t B, perflibs_spmat_t C) {

  auto impl_A = reinterpret_cast<perflibs_spmat_impl_t<T> *>(A->impl);
  auto impl_B = reinterpret_cast<perflibs_spmat_impl_t<T> *>(B->impl);
  auto impl_C = reinterpret_cast<perflibs_spmat_impl_t<T> *>(C->impl);

  if (!(transA == PERFLIBS_SPARSE_OPERATION_NOTRANS ||
        transA == PERFLIBS_SPARSE_OPERATION_TRANS ||
        transA == PERFLIBS_SPARSE_OPERATION_CONJTRANS)) {
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }

  if (!(transB == PERFLIBS_SPARSE_OPERATION_NOTRANS ||
        transB == PERFLIBS_SPARSE_OPERATION_TRANS ||
        transB == PERFLIBS_SPARSE_OPERATION_CONJTRANS)) {
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }

  // Check that the values of m and n for each matrix are non-negative
  // Even check C since it should be the null matrix
  if (impl_A->m < 0 || impl_A->n < 0 || impl_B->m < 0 || impl_B->n < 0 ||
      impl_C->m < 0 || impl_C->n < 0) {
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }

  perflibs_int_t m = 0, n = 0;

  // Check that the dimensions of A and B are compatible, given the transpose
  // options
  if (transA == PERFLIBS_SPARSE_OPERATION_NOTRANS &&
      transB == PERFLIBS_SPARSE_OPERATION_NOTRANS) {
    if (impl_A->m != impl_B->m || impl_A->n != impl_B->n) {
      return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    }
    m = impl_A->m;
    n = impl_A->n;
  } else if (transA == PERFLIBS_SPARSE_OPERATION_NOTRANS &&
             transB != PERFLIBS_SPARSE_OPERATION_NOTRANS) {
    if (impl_A->m != impl_B->n || impl_A->n != impl_B->m) {
      return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    }
    m = impl_A->m;
    n = impl_A->n;
  } else if (transA != PERFLIBS_SPARSE_OPERATION_NOTRANS &&
             transB == PERFLIBS_SPARSE_OPERATION_NOTRANS) {
    if (impl_A->n != impl_B->m || impl_A->m != impl_B->n) {
      return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    }
    m = impl_A->n;
    n = impl_A->m;
  } else if (transA != PERFLIBS_SPARSE_OPERATION_NOTRANS &&
             transB != PERFLIBS_SPARSE_OPERATION_NOTRANS) {
    if (impl_A->n != impl_B->n || impl_A->m != impl_B->m) {
      return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    }
    m = impl_A->n;
    n = impl_A->m;
  } else { // an invalid trans option has been provided
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }

  // Check that valid scalar values have been provided
  if (!(alpha == PERFLIBS_SPARSE_SCALAR_ONE ||
        alpha == PERFLIBS_SPARSE_SCALAR_ZERO ||
        alpha == PERFLIBS_SPARSE_SCALAR_ANY)) {
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }
  if (!(beta == PERFLIBS_SPARSE_SCALAR_ONE ||
        beta == PERFLIBS_SPARSE_SCALAR_ZERO ||
        beta == PERFLIBS_SPARSE_SCALAR_ANY)) {
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }

  // Check that the user has set the input matrix to the null matrix
  if (impl_C->spmat_format != perflibs_format_null || impl_C->m != m ||
      impl_C->n != n) {
    impl_C->error_handle.perflibs_error_type =
        PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    impl_C->error_handle.perflibs_error_code =
        7; // C is the 7th parameter in the interface
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }

  return PERFLIBS_STATUS_SUCCESS;
}

/**
 * \f$C = \alpha*A + \beta*B\f$ for CSR matrices A, B and C. A and B must have a
 * matching base index and matching dimensions m and n (i.e. this performs
 * transA=N transB=N addition).
 * @param transA [in] transpose option for matrix A, used only for conjugation
 * @param transB [in] transpose option for matrix B, used only for conjugation
 * @param m [in] number of rows of A and B
 * @param n [in] number of columns of A and B
 * @param alpha [in] scalar multiplier of A
 * @param row_ptrA [in] ptr to array of row pointers for A
 * @param col_indxA [in] ptr to array of column indices for A
 * @param valsA [in] ptr to non-zero array of values of A
 * @param beta [in] scalar multiplier of B
 * @param row_ptrB [in] ptr to array of row pointers for B
 * @param col_indxB [in] ptr to array of column indices for B
 * @param valsB [in] ptr to non-zero array of values of B
 * @param row_ptrC [out] std::vector of row pointers for C, empty on input
 * @param col_indxC [out] std::vector of column indices for C, empty on input
 * @param valsC [out] non-zero std::vector of values of C, empty on input
 */
template <typename T>
void spadd_csr_kernel(perflibs_sparse_hint_value transA,
                      perflibs_sparse_hint_value transB, perflibs_int_t m,
                      perflibs_int_t n, T alpha, const perflibs_int_t *row_ptrA,
                      const perflibs_int_t *col_indxA, const T *valsA, T beta,
                      const perflibs_int_t *row_ptrB,
                      const perflibs_int_t *col_indxB, const T *valsB,
                      std::vector<perflibs_int_t> &row_ptrC,
                      std::vector<perflibs_int_t> &col_indxC,
                      std::vector<T> &valsC) {

  auto index_base = row_ptrA[0];
  assert(row_ptrB[0] == index_base);

  auto conjA = [&](T val) {
    return transA == PERFLIBS_SPARSE_OPERATION_CONJTRANS
               ? perflibs::sparse::conj(val)
               : val;
  };

  auto conjB = [&](T val) {
    return transB == PERFLIBS_SPARSE_OPERATION_CONJTRANS
               ? perflibs::sparse::conj(val)
               : val;
  };

  /*
      Column indices of A and B are not guaranteed to be sorted.
      This algorithm iterates over elements of A within a row and for each
      element of A searches for the corresponding column index in the same row
     of B. Repeated searching over B is deemed more efficient than repeated data
     copying and sorting of both A and B.
  */

  // A vector used to indicate whether a matching index of B is found to go with
  // the element of A we're working on in the loop below
  std::vector<perflibs_int_t> b_ind(n, -1);

  // A pair of function pointers to swap in the loop
  for (perflibs_int_t i = 0; i < m; i++) {

    // Make sure that B is the matrix with the shorter row
    size_t lenA = row_ptrA[i + 1] - row_ptrA[i];
    size_t lenB = row_ptrB[i + 1] - row_ptrB[i];
    if (lenA < lenB) {
      std::swap(row_ptrA, row_ptrB);
      std::swap(col_indxA, col_indxB);
      std::swap(valsA, valsB);
      std::swap(transA, transB);
      std::swap(alpha, beta);
    }

    // Iterate over A and add the matching elements from B, noting them down in
    // b_ind;
    for (perflibs_int_t jA = row_ptrA[i] - index_base;
         jA < row_ptrA[i + 1] - index_base; jA++) {
      auto colA = col_indxA[jA];
      if (row_ptrB[i + 1] > row_ptrB[i]) {
        // Column indices are not necessarily in order, so search the
        // corresponding row of B for a matching index
        auto startB = &col_indxB[row_ptrB[i] - index_base];
        auto endB = &col_indxB[row_ptrB[i + 1] - index_base];
        auto colB = std::find(startB, endB, colA);
        // If found add elements of A and B
        if (colB != endB) {
          auto jB = (row_ptrB[i] - index_base) + (colB - startB);
          valsC.push_back(alpha * conjA(valsA[jA]) + beta * conjB(valsB[jB]));
          col_indxC.push_back(colA);
          row_ptrC[i + 1]++;
          b_ind[colA - index_base] = i;
        }
      }
      // If no matching element of B was found just write elem of A into C
      if (b_ind[colA - index_base] != i) {
        valsC.push_back(alpha * conjA(valsA[jA]));
        col_indxC.push_back(colA);
        row_ptrC[i + 1]++;
      }
    }

    // Now iterate over B copying in the elements that were not written above
    for (perflibs_int_t jB = row_ptrB[i] - index_base;
         jB < row_ptrB[i + 1] - index_base; jB++) {
      auto colB = col_indxB[jB];
      if (b_ind[colB - index_base] != i) {
        valsC.push_back(beta * conjB(valsB[jB]));
        col_indxC.push_back(colB);
        row_ptrC[i + 1]++;
      }
    }
  }

  row_ptrC[0] = index_base;
  for (perflibs_int_t i = 1; i < m + 1; i++) {
    row_ptrC[i] += row_ptrC[i - 1];
  }
}

/**
 * C = alpha*op(A) + beta*op(B)
 */
template <typename T>
std::pair<perflibs_status_t, std::unique_ptr<perflibs_spmat_top_t>>
spadd_dispatch(perflibs_sparse_hint_value transA,
               perflibs_sparse_hint_value transB, T alpha, perflibs_spmat_t A,
               T beta, perflibs_spmat_t B, bool exec) {

  auto impl_A = reinterpret_cast<perflibs_spmat_impl_t<T> *>(A->impl);
  auto impl_B = reinterpret_cast<perflibs_spmat_impl_t<T> *>(B->impl);

  // Assumption here is that we've come from spmm_exec or spadd_exec and A, B
  // aren't aliases of the same object
  assert(impl_A != impl_B);

  auto m = transA == PERFLIBS_SPARSE_OPERATION_NOTRANS ? impl_A->m : impl_A->n;
  auto n = transA == PERFLIBS_SPARSE_OPERATION_NOTRANS ? impl_A->n : impl_A->m;

  // No addition is required if both terms are zero or one of the dimensions is
  // zero
  if (((alpha == (T)0 || impl_A->spmat_format == perflibs_format_null) &&
       (beta == (T)0 || impl_B->spmat_format == perflibs_format_null)) ||
      m < 1 || n < 1) {
    return std::make_pair(PERFLIBS_STATUS_SUCCESS, null_matrix(m, n));
  }

  // Handle a straight-forward copy
  if (alpha == (T)1 && transA == PERFLIBS_SPARSE_OPERATION_NOTRANS &&
      (beta == (T)0 || impl_B->spmat_format == perflibs_format_null)) {
    auto C = spmat_copy<T>(A);
    return std::make_pair(PERFLIBS_STATUS_SUCCESS, std::move(C));
  }
  if ((alpha == (T)0 || impl_A->spmat_format == perflibs_format_null) &&
      beta == (T)1 && transB == PERFLIBS_SPARSE_OPERATION_NOTRANS) {
    auto C = spmat_copy<T>(B);
    return std::make_pair(PERFLIBS_STATUS_SUCCESS, std::move(C));
  }

  // If one matrix is null just scale the other
  if (alpha == (T)0 || impl_A->spmat_format == perflibs_format_null) {
    auto C = spmat_copy<T>(B);
    auto ret = scale_matrix(transB, beta, C.get());
    if (ret != PERFLIBS_STATUS_SUCCESS) {
      return std::make_pair(PERFLIBS_STATUS_EXECUTION_FAILURE, nullptr);
    }
    return std::make_pair(PERFLIBS_STATUS_SUCCESS, std::move(C));
  }
  if (beta == (T)0 || impl_B->spmat_format == perflibs_format_null) {
    auto C = spmat_copy<T>(A);
    auto ret = scale_matrix(transA, alpha, C.get());
    if (ret != PERFLIBS_STATUS_SUCCESS) {
      return std::make_pair(PERFLIBS_STATUS_EXECUTION_FAILURE, nullptr);
    }
    return std::make_pair(PERFLIBS_STATUS_SUCCESS, std::move(C));
  }

  auto target_format_A = transA == PERFLIBS_SPARSE_OPERATION_NOTRANS
                             ? perflibs_format_csr
                             : perflibs_format_csc;
  auto target_format_B = transB == PERFLIBS_SPARSE_OPERATION_NOTRANS
                             ? perflibs_format_csr
                             : perflibs_format_csc;

  // Run through convert to get CSR matrices
  auto ret = convert(target_format_A, impl_A);
  if (ret != PERFLIBS_STATUS_SUCCESS) {
    return std::make_pair(PERFLIBS_STATUS_EXECUTION_FAILURE, nullptr);
  }

  ret = convert(target_format_B, impl_B);
  if (ret != PERFLIBS_STATUS_SUCCESS) {
    return std::make_pair(PERFLIBS_STATUS_EXECUTION_FAILURE, nullptr);
  }

  assert(impl_A->spmat_format == perflibs_format_csr ||
         impl_A->spmat_format == perflibs_format_csc);
  assert(impl_B->spmat_format == perflibs_format_csr ||
         impl_B->spmat_format == perflibs_format_csc);

  if (!exec) { // Return early here if we're optimizing only
    return {PERFLIBS_STATUS_SUCCESS, nullptr};
  }

  auto row_ptrA = impl_A->spmat_format == perflibs_format_csr
                      ? impl_A->csr.row_ptr_ptr
                      : impl_A->csc.col_ptr_ptr;
  auto row_ptrB = impl_B->spmat_format == perflibs_format_csr
                      ? impl_B->csr.row_ptr_ptr
                      : impl_B->csc.col_ptr_ptr;
  auto col_indxA = impl_A->spmat_format == perflibs_format_csr
                       ? impl_A->csr.col_indx_ptr
                       : impl_A->csc.row_indx_ptr;
  auto col_indxB = impl_B->spmat_format == perflibs_format_csr
                       ? impl_B->csr.col_indx_ptr
                       : impl_B->csc.row_indx_ptr;
  auto valsA = impl_A->spmat_format == perflibs_format_csr
                   ? impl_A->csr.vals_ptr
                   : impl_A->csc.vals_ptr;
  auto valsB = impl_B->spmat_format == perflibs_format_csr
                   ? impl_B->csr.vals_ptr
                   : impl_B->csc.vals_ptr;

  auto len_ptrA =
      impl_A->spmat_format == perflibs_format_csr ? impl_A->m : impl_A->n;
  auto len_ptrB =
      impl_B->spmat_format == perflibs_format_csr ? impl_B->m : impl_B->n;

  perflibs::sparse::pod_vector<perflibs_int_t> index_copy;
  perflibs::sparse::pod_vector<perflibs_int_t> ptr_copy;
  // Optionally copy one of the index arrays to make the index_base the same in
  // both
  make_index_base_equal(len_ptrA, len_ptrB, impl_A->nnz, impl_B->nnz, &row_ptrA,
                        &row_ptrB, &col_indxA, &col_indxB, index_copy,
                        ptr_copy);

  // Create CSR arrays for C_out
  std::vector<T> vals_C_out;
  std::vector<perflibs_int_t> col_indx_C_out;
  std::vector<perflibs_int_t> row_ptr_C_out(m + 1);

  // Perform the addition
  spadd_csr_kernel(transA, transB, m, n, alpha, row_ptrA, col_indxA, valsA,
                   beta, row_ptrB, col_indxB, valsB, row_ptr_C_out,
                   col_indx_C_out, vals_C_out);

  // Fill C with C_out arrays
  auto C = create_new_matrix<T>();
  ret = fill_initial_data_csr(C.get(), m, n, row_ptr_C_out.data(),
                              col_indx_C_out.data(), vals_C_out.data(), 0);
  if (ret != PERFLIBS_STATUS_SUCCESS) {
    return std::make_pair(ret, nullptr);
  }
  return std::make_pair(PERFLIBS_STATUS_SUCCESS, std::move(C));
}
template std::pair<perflibs_status_t, std::unique_ptr<perflibs_spmat_top_t>>
spadd_dispatch<float>(perflibs_sparse_hint_value, perflibs_sparse_hint_value,
                      float, perflibs_spmat_t, float, perflibs_spmat_t, bool);
template std::pair<perflibs_status_t, std::unique_ptr<perflibs_spmat_top_t>>
spadd_dispatch<double>(perflibs_sparse_hint_value, perflibs_sparse_hint_value,
                       double, perflibs_spmat_t, double, perflibs_spmat_t,
                       bool);
template std::pair<perflibs_status_t, std::unique_ptr<perflibs_spmat_top_t>>
spadd_dispatch<std::complex<float>>(perflibs_sparse_hint_value,
                                    perflibs_sparse_hint_value,
                                    std::complex<float>, perflibs_spmat_t,
                                    std::complex<float>, perflibs_spmat_t,
                                    bool);
template std::pair<perflibs_status_t, std::unique_ptr<perflibs_spmat_top_t>>
spadd_dispatch<std::complex<double>>(perflibs_sparse_hint_value,
                                     perflibs_sparse_hint_value,
                                     std::complex<double>, perflibs_spmat_t,
                                     std::complex<double>, perflibs_spmat_t,
                                     bool);

template <typename T>
perflibs_status_t spadd_exec_checked(perflibs_sparse_hint_value transA,
                                     perflibs_sparse_hint_value transB, T alpha,
                                     perflibs_spmat_t A, T beta,
                                     perflibs_spmat_t B, perflibs_spmat_t C,
                                     bool exec) {

  auto impl_A = reinterpret_cast<perflibs_spmat_impl_t<T> *>(A->impl);
  auto impl_B = reinterpret_cast<perflibs_spmat_impl_t<T> *>(B->impl);

  // If A and B are aliases take a copy since we may need to transform in
  // spadd_dispatch
  std::unique_ptr<perflibs_spmat_top_t> B_copy;
  auto B_use = B;
  if (impl_A == impl_B) {
    if (!exec) { // If we're optimizing then just return - all bets are off for
                 // optimization if A==B!
      return PERFLIBS_STATUS_SUCCESS;
    }
    B_copy = spmat_copy<T>(B);
    B_use = B_copy.get();
  }

  auto [ret, C_out] =
      spadd_dispatch(transA, transB, alpha, A, beta, B_use, exec);

  if (ret != PERFLIBS_STATUS_SUCCESS || !exec) {
    return ret;
  }

  *C = std::move(*C_out);

  return ret;
}

template <typename T>
perflibs_status_t spadd_exec(perflibs_sparse_hint_value transA,
                             perflibs_sparse_hint_value transB, T alpha,
                             perflibs_spmat_t A, T beta, perflibs_spmat_t B,
                             perflibs_spmat_t C) {

  auto ret_p = spadd_check_params<T>(transA, transB, PERFLIBS_SPARSE_SCALAR_ANY,
                                     A, PERFLIBS_SPARSE_SCALAR_ANY, B, C);
  if (ret_p != PERFLIBS_STATUS_SUCCESS) {
    return ret_p;
  }

  return spadd_exec_checked(transA, transB, alpha, A, beta, B, C, true);
}
template perflibs_status_t spadd_exec(perflibs_sparse_hint_value transA,
                                      perflibs_sparse_hint_value transB,
                                      float alpha, perflibs_spmat_t A,
                                      float beta, perflibs_spmat_t B,
                                      perflibs_spmat_t C);
template perflibs_status_t spadd_exec(perflibs_sparse_hint_value transA,
                                      perflibs_sparse_hint_value transB,
                                      double alpha, perflibs_spmat_t A,
                                      double beta, perflibs_spmat_t B,
                                      perflibs_spmat_t C);
template perflibs_status_t
spadd_exec(perflibs_sparse_hint_value transA, perflibs_sparse_hint_value transB,
           std::complex<float> alpha, perflibs_spmat_t A,
           std::complex<float> beta, perflibs_spmat_t B, perflibs_spmat_t C);
template perflibs_status_t
spadd_exec(perflibs_sparse_hint_value transA, perflibs_sparse_hint_value transB,
           std::complex<double> alpha, perflibs_spmat_t A,
           std::complex<double> beta, perflibs_spmat_t B, perflibs_spmat_t C);

/// The optimize function currently calls the execute function, telling it to
/// prepare the matrices, but don't actually execute
template <typename T>
perflibs_status_t spadd_optimize(perflibs_sparse_hint_value transA,
                                 perflibs_sparse_hint_value transB,
                                 perflibs_sparse_hint_value alpha,
                                 perflibs_spmat_t A,
                                 perflibs_sparse_hint_value beta,
                                 perflibs_spmat_t B, perflibs_spmat_t C) {

  auto ret_p = spadd_check_params<T>(transA, transB, alpha, A, beta, B, C);
  if (ret_p != PERFLIBS_STATUS_SUCCESS) {
    return ret_p;
  }

  // Create some representative values for the optimization
  T alpha_ = alpha == PERFLIBS_SPARSE_SCALAR_ZERO  ? (T)0
             : alpha == PERFLIBS_SPARSE_SCALAR_ONE ? (T)1
                                                   : (T)1.5;
  T beta_ = beta == PERFLIBS_SPARSE_SCALAR_ZERO  ? (T)0
            : beta == PERFLIBS_SPARSE_SCALAR_ONE ? (T)1
                                                 : (T)1.5;

  return spadd_exec_checked(transA, transB, alpha_, A, beta_, B, C, false);
}
template perflibs_status_t spadd_optimize<float>(
    perflibs_sparse_hint_value transA, perflibs_sparse_hint_value transB,
    perflibs_sparse_hint_value alpha, perflibs_spmat_t A,
    perflibs_sparse_hint_value beta, perflibs_spmat_t B, perflibs_spmat_t C);
template perflibs_status_t spadd_optimize<double>(
    perflibs_sparse_hint_value transA, perflibs_sparse_hint_value transB,
    perflibs_sparse_hint_value alpha, perflibs_spmat_t A,
    perflibs_sparse_hint_value beta, perflibs_spmat_t B, perflibs_spmat_t C);
template perflibs_status_t spadd_optimize<std::complex<float>>(
    perflibs_sparse_hint_value transA, perflibs_sparse_hint_value transB,
    perflibs_sparse_hint_value alpha, perflibs_spmat_t A,
    perflibs_sparse_hint_value beta, perflibs_spmat_t B, perflibs_spmat_t C);
template perflibs_status_t spadd_optimize<std::complex<double>>(
    perflibs_sparse_hint_value transA, perflibs_sparse_hint_value transB,
    perflibs_sparse_hint_value alpha, perflibs_spmat_t A,
    perflibs_sparse_hint_value beta, perflibs_spmat_t B, perflibs_spmat_t C);

} // namespace perflibs::sparse
