/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "perflibs_sparse.h"
#include "test_utils.h"

#include <stdlib.h>
#include <string.h>

static perflibs_doublecomplex_t
access_with_op(const perflibs_doublecomplex_t *dense, perflibs_int_t n,
               enum perflibs_sparse_hint_value op, perflibs_int_t i,
               perflibs_int_t j) {
  const perflibs_doublecomplex_t v_notrans = dense[i * n + j];
  const perflibs_doublecomplex_t v_trans = dense[j * n + i];
  if (op == PERFLIBS_SPARSE_OPERATION_NOTRANS) {
    return v_notrans;
  }
  if (op == PERFLIBS_SPARSE_OPERATION_TRANS) {
    return v_trans;
  }
  if (op == PERFLIBS_SPARSE_OPERATION_CONJTRANS) {
    return test_conj(v_trans);
  }
  return v_notrans;
}

static void compute_expected_spelmm_z(const perflibs_doublecomplex_t *denseA,
                                      const perflibs_doublecomplex_t *denseB,
                                      perflibs_int_t n,
                                      enum perflibs_sparse_hint_value transA,
                                      enum perflibs_sparse_hint_value transB,
                                      perflibs_doublecomplex_t alpha,
                                      const perflibs_doublecomplex_t *c_init,
                                      perflibs_doublecomplex_t beta,
                                      perflibs_doublecomplex_t *out) {
  for (perflibs_int_t i = 0; i < n; ++i) {
    for (perflibs_int_t j = 0; j < n; ++j) {
      const perflibs_doublecomplex_t a =
          access_with_op(denseA, n, transA, i, j);
      const perflibs_doublecomplex_t b =
          access_with_op(denseB, n, transB, i, j);
      const perflibs_doublecomplex_t prev =
          c_init ? c_init[i * n + j] : TEST_Z(0.0, 0.0);
      out[i * n + j] =
          test_cadd(test_cmul(alpha, test_cmul(a, b)), test_cmul(beta, prev));
    }
  }
}

static int test_coo_spelmm_notrans() {
  const perflibs_int_t n = 3;
  const perflibs_int_t nnz = 5;
  const perflibs_int_t rowA[] = {0, 0, 1, 2, 2};
  const perflibs_int_t colA[] = {0, 2, 1, 0, 2};
  const perflibs_doublecomplex_t valA[] = {TEST_Z(1.0, 1.0), TEST_Z(2.0, -1.0),
                                           TEST_Z(-1.0, 2.0), TEST_Z(3.0, 0.0),
                                           TEST_Z(0.5, -0.5)};
  const perflibs_int_t rowB[] = {0, 0, 1, 2, 2};
  const perflibs_int_t colB[] = {0, 1, 1, 0, 2};
  const perflibs_doublecomplex_t valB[] = {TEST_Z(2.0, -1.0), TEST_Z(4.0, 0.0),
                                           TEST_Z(1.0, 1.0), TEST_Z(-1.0, 2.0),
                                           TEST_Z(3.0, 1.0)};
  const perflibs_doublecomplex_t expected[] = {
      TEST_Z(3.0, 1.0),  TEST_Z(0.0, 0.0),  TEST_Z(0.0, 0.0),
      TEST_Z(0.0, 0.0),  TEST_Z(-3.0, 1.0), TEST_Z(0.0, 0.0),
      TEST_Z(-3.0, 6.0), TEST_Z(0.0, 0.0),  TEST_Z(2.0, -1.0)};

  perflibs_spmat_t A = NULL;
  perflibs_spmat_t B = NULL;
  perflibs_spmat_t C = perflibs_spmat_create_null(n, n);
  perflibs_int_t m_out = -1;
  perflibs_int_t n_out = -1;
  perflibs_doublecomplex_t *dense_out = NULL;

  CHECK_TRUE(C != NULL, "spmat_create_null failed");
  CHECK_STATUS(perflibs_spmat_create_coo_z(&A, n, n, nnz, rowA, colA, valA, 0));
  CHECK_STATUS(perflibs_spmat_create_coo_z(&B, n, n, nnz, rowB, colB, valB, 0));

  CHECK_STATUS(perflibs_spelmm_optimize(
      PERFLIBS_SPARSE_OPERATION_NOTRANS, PERFLIBS_SPARSE_OPERATION_NOTRANS,
      PERFLIBS_SPARSE_SCALAR_ANY, A, B, PERFLIBS_SPARSE_SCALAR_ZERO, C));

  CHECK_STATUS(perflibs_spelmm_exec_z(
      PERFLIBS_SPARSE_OPERATION_NOTRANS, PERFLIBS_SPARSE_OPERATION_NOTRANS,
      TEST_Z(1.0, 0.0), A, B, TEST_Z(0.0, 0.0), C));
  CHECK_STATUS(perflibs_spmat_export_dense_z(C, PERFLIBS_ROW_MAJOR, &m_out,
                                             &n_out, &dense_out));
  CHECK_INT_EQ(m_out, n);
  CHECK_INT_EQ(n_out, n);
  CHECK_DOUBLECOMPLEX_ARRAY(dense_out, expected, n * n, 1e-12);

  free(dense_out);
  CHECK_STATUS(perflibs_spmat_destroy(A));
  CHECK_STATUS(perflibs_spmat_destroy(B));
  CHECK_STATUS(perflibs_spmat_destroy(C));
  return EXIT_SUCCESS;
}

static int test_coo_spelmm_transpose_conjugate_beta_dense() {
  const perflibs_int_t n = 3;
  const perflibs_int_t nnz = 5;
  const perflibs_int_t rowA[] = {0, 0, 1, 2, 2};
  const perflibs_int_t colA[] = {0, 2, 1, 0, 2};
  const perflibs_doublecomplex_t valA[] = {TEST_Z(1.0, 1.0), TEST_Z(2.0, -1.0),
                                           TEST_Z(-1.0, 2.0), TEST_Z(3.0, 0.0),
                                           TEST_Z(0.5, -0.5)};
  const perflibs_int_t rowB[] = {0, 0, 1, 2, 2};
  const perflibs_int_t colB[] = {0, 1, 1, 0, 2};
  const perflibs_doublecomplex_t valB[] = {TEST_Z(2.0, -1.0), TEST_Z(4.0, 0.0),
                                           TEST_Z(1.0, 1.0), TEST_Z(-1.0, 2.0),
                                           TEST_Z(3.0, 1.0)};
  perflibs_doublecomplex_t denseA[9];
  perflibs_doublecomplex_t denseB[9];
  perflibs_doublecomplex_t expected[9];

  perflibs_spmat_t A = NULL;
  perflibs_spmat_t B = NULL;
  perflibs_spmat_t C = perflibs_spmat_create_null(n, n);
  perflibs_int_t m_out = -1;
  perflibs_int_t n_out = -1;
  perflibs_doublecomplex_t *dense_out = NULL;

  memset(denseA, 0, sizeof(denseA));
  memset(denseB, 0, sizeof(denseB));
  for (perflibs_int_t i = 0; i < nnz; ++i) {
    denseA[rowA[i] * n + colA[i]] = valA[i];
    denseB[rowB[i] * n + colB[i]] = valB[i];
  }

  CHECK_TRUE(C != NULL, "spmat_create_null failed");
  CHECK_STATUS(perflibs_spmat_create_coo_z(&A, n, n, nnz, rowA, colA, valA, 0));
  CHECK_STATUS(perflibs_spmat_create_coo_z(&B, n, n, nnz, rowB, colB, valB, 0));

  const perflibs_doublecomplex_t alpha = TEST_Z(0.5, -0.25);
  compute_expected_spelmm_z(
      denseA, denseB, n, PERFLIBS_SPARSE_OPERATION_CONJTRANS,
      PERFLIBS_SPARSE_OPERATION_TRANS, alpha, NULL, TEST_Z(0.0, 0.0), expected);

  CHECK_STATUS(perflibs_spelmm_exec_z(PERFLIBS_SPARSE_OPERATION_CONJTRANS,
                                      PERFLIBS_SPARSE_OPERATION_TRANS, alpha, A,
                                      B, TEST_Z(0.0, 0.0), C));

  CHECK_STATUS(perflibs_spmat_export_dense_z(C, PERFLIBS_ROW_MAJOR, &m_out,
                                             &n_out, &dense_out));
  CHECK_INT_EQ(m_out, n);
  CHECK_INT_EQ(n_out, n);
  CHECK_DOUBLECOMPLEX_ARRAY(dense_out, expected, n * n, 1e-12);

  free(dense_out);
  CHECK_STATUS(perflibs_spmat_destroy(A));
  CHECK_STATUS(perflibs_spmat_destroy(B));
  CHECK_STATUS(perflibs_spmat_destroy(C));
  return EXIT_SUCCESS;
}

int main() {
  if (test_coo_spelmm_notrans() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (test_coo_spelmm_transpose_conjugate_beta_dense() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
