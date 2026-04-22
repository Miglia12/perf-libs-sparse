/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "perflibs_sparse.h"
#include "test_utils.h"

#include <cblas.h>
#include <stdlib.h>
#include <string.h>

#define SPMM_STRATEGY_Z_M 16
#define SPMM_STRATEGY_Z_K 24
#define SPMM_STRATEGY_Z_N 96
#define SPMM_STRATEGY_Z_NNZA (SPMM_STRATEGY_Z_M * 5)
#define SPMM_STRATEGY_Z_NNZB (SPMM_STRATEGY_Z_K * 5)

static void
build_complex_inputs(perflibs_int_t *row_ptrA, perflibs_int_t *col_indxA,
                     perflibs_doublecomplex_t *valsA, perflibs_int_t *row_ptrB,
                     perflibs_int_t *col_indxB, perflibs_doublecomplex_t *valsB,
                     perflibs_doublecomplex_t *denseA,
                     perflibs_doublecomplex_t *denseB) {
  const perflibs_int_t m = 16;
  const perflibs_int_t k = 24;
  const perflibs_int_t n = 96;
  const perflibs_int_t entries = 5;

  memset(denseA, 0, sizeof(perflibs_doublecomplex_t) * m * k);
  memset(denseB, 0, sizeof(perflibs_doublecomplex_t) * k * n);

  row_ptrA[0] = 0;
  for (perflibs_int_t i = 0; i < m; ++i) {
    row_ptrA[i + 1] = row_ptrA[i] + entries;
    perflibs_int_t base = (i * 3) % (k - 4);
    for (perflibs_int_t j = 0; j < entries; ++j) {
      perflibs_int_t idx = row_ptrA[i] + j;
      perflibs_int_t col = base + j;
      perflibs_doublecomplex_t val = TEST_Z((double)(i + 1), (double)(j + 2));
      col_indxA[idx] = col;
      valsA[idx] = val;
      denseA[i * k + col] = val;
    }
  }

  row_ptrB[0] = 0;
  for (perflibs_int_t i = 0; i < k; ++i) {
    row_ptrB[i + 1] = row_ptrB[i] + entries;
    perflibs_int_t base = (i * 5) % (n - 5);
    for (perflibs_int_t j = 0; j < entries; ++j) {
      perflibs_int_t idx = row_ptrB[i] + j;
      perflibs_int_t col = base + j;
      perflibs_doublecomplex_t val =
          TEST_Z((double)(i + 2) * (double)(j + 3), (double)(i + j + 1));
      col_indxB[idx] = col;
      valsB[idx] = val;
      denseB[i * n + col] = val;
    }
  }
}

static int
run_complex_symbolic_strategy(enum perflibs_sparse_hint_value strat) {
  perflibs_int_t row_ptrA[SPMM_STRATEGY_Z_M + 1];
  perflibs_int_t col_indxA[SPMM_STRATEGY_Z_NNZA];
  perflibs_doublecomplex_t valsA[SPMM_STRATEGY_Z_NNZA];
  perflibs_int_t row_ptrB[SPMM_STRATEGY_Z_K + 1];
  perflibs_int_t col_indxB[SPMM_STRATEGY_Z_NNZB];
  perflibs_doublecomplex_t valsB[SPMM_STRATEGY_Z_NNZB];
  perflibs_doublecomplex_t denseA[SPMM_STRATEGY_Z_M * SPMM_STRATEGY_Z_K];
  perflibs_doublecomplex_t denseB[SPMM_STRATEGY_Z_K * SPMM_STRATEGY_Z_N];
  perflibs_doublecomplex_t expected[SPMM_STRATEGY_Z_M * SPMM_STRATEGY_Z_N];
  const perflibs_doublecomplex_t alpha = TEST_Z(1.0, 0.0);
  const perflibs_doublecomplex_t beta = TEST_Z(0.0, 0.0);

  build_complex_inputs(row_ptrA, col_indxA, valsA, row_ptrB, col_indxB, valsB,
                       denseA, denseB);
  cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, SPMM_STRATEGY_Z_M,
              SPMM_STRATEGY_Z_N, SPMM_STRATEGY_Z_K, &alpha, denseA,
              SPMM_STRATEGY_Z_K, denseB, SPMM_STRATEGY_Z_N, &beta, expected,
              SPMM_STRATEGY_Z_N);

  perflibs_spmat_t A = NULL;
  perflibs_spmat_t B = NULL;
  perflibs_spmat_t C =
      perflibs_spmat_create_null(SPMM_STRATEGY_Z_M, SPMM_STRATEGY_Z_N);
  perflibs_doublecomplex_t *denseC = NULL;

  CHECK_TRUE(C != NULL, "spmm output creation failed");
  CHECK_STATUS(perflibs_spmat_create_csr_z(
      &A, SPMM_STRATEGY_Z_M, SPMM_STRATEGY_Z_K, row_ptrA, col_indxA, valsA, 0));
  CHECK_STATUS(perflibs_spmat_create_csr_z(
      &B, SPMM_STRATEGY_Z_K, SPMM_STRATEGY_Z_N, row_ptrB, col_indxB, valsB, 0));
  CHECK_STATUS(
      perflibs_spmat_hint(C, PERFLIBS_SPARSE_HINT_SPMM_STRATEGY, strat));
  CHECK_STATUS(perflibs_spmm_optimize(
      PERFLIBS_SPARSE_OPERATION_NOTRANS, PERFLIBS_SPARSE_OPERATION_NOTRANS,
      PERFLIBS_SPARSE_SCALAR_ANY, A, B, PERFLIBS_SPARSE_SCALAR_ZERO, C));

  perflibs_int_t m_out = -1;
  perflibs_int_t n_out = -1;
  perflibs_int_t *row_ptrC = NULL;
  perflibs_int_t *col_indxC = NULL;
  perflibs_doublecomplex_t *valsC = NULL;

  CHECK_STATUS(perflibs_spmat_export_csr_z(C, 0, &m_out, &n_out, &row_ptrC,
                                           &col_indxC, &valsC));
  CHECK_INT_EQ(m_out, SPMM_STRATEGY_Z_M);
  CHECK_INT_EQ(n_out, SPMM_STRATEGY_Z_N);
  CHECK_TRUE(row_ptrC[SPMM_STRATEGY_Z_M] > 0,
             "optimized complex strategy created empty C");
  free(row_ptrC);
  free(col_indxC);
  free(valsC);

  CHECK_STATUS(perflibs_spmm_exec_z(
      PERFLIBS_SPARSE_OPERATION_NOTRANS, PERFLIBS_SPARSE_OPERATION_NOTRANS,
      TEST_Z(1.0, 0.0), A, B, TEST_Z(0.0, 0.0), C));
  CHECK_STATUS(perflibs_spmat_export_dense_z(C, PERFLIBS_ROW_MAJOR, &m_out,
                                             &n_out, &denseC));
  CHECK_DOUBLECOMPLEX_ARRAY(denseC, expected,
                            SPMM_STRATEGY_Z_M * SPMM_STRATEGY_Z_N, 1e-12);
  free(denseC);

  CHECK_STATUS(perflibs_spmat_destroy(A));
  CHECK_STATUS(perflibs_spmat_destroy(B));
  CHECK_STATUS(perflibs_spmat_destroy(C));
  return EXIT_SUCCESS;
}

int main() {
  if (run_complex_symbolic_strategy(
          PERFLIBS_SPARSE_SPMM_STRAT_OPT_PART_STRUCT) != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (run_complex_symbolic_strategy(
          PERFLIBS_SPARSE_SPMM_STRAT_OPT_FULL_STRUCT) != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
