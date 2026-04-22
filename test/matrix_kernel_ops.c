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

static void compute_dense_mm(const double *A, const double *B, double *C) {
  const double alpha = 1.0;
  const double beta = 0.0;
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 4, 4, 4, alpha, A, 4,
              B, 4, beta, C, 4);
}

static void compute_dense_hadamard(const double *A, const double *B,
                                   double *C) {
  for (perflibs_int_t i = 0; i < 16; ++i) {
    C[i] = A[i] * B[i];
  }
}

static void apply_dense_mask(const double *mask, double *C) {
  for (perflibs_int_t i = 0; i < 16; ++i) {
    if (mask[i] == 0.0) {
      C[i] = 0.0;
    }
  }
}

static int test_spadd_spmm_spelmm_sddmm() {
  perflibs_spmat_t a = NULL;
  perflibs_spmat_t b = NULL;
  perflibs_spmat_t c = NULL;
  double *dense = NULL;
  perflibs_int_t m = -1, n = -1;
  const double expected_spadd[16] = {2.0, -4.0, 0.0, 0.0, -1.0, 6.0, 0.0, 0.0,
                                     0.0, 0.0,  9.0, 0.0, 0.0,  0.0, 0.0, 10.0};
  double expected_spmm[16];
  double expected_spelmm[16];
  double expected_sddmm[16];

  CHECK_STATUS(perflibs_spmat_create_csr_d(
      &a, 4, 4, (const perflibs_int_t[]){0, 1, 3, 4, 5},
      (const perflibs_int_t[]){0, 0, 1, 2, 3},
      (const double[]){1.0, 2.0, 3.0, 4.0, 6.0}, 0));
  CHECK_STATUS(perflibs_spmat_create_csr_d(
      &b, 4, 4, (const perflibs_int_t[]){0, 1, 2, 3, 4},
      (const perflibs_int_t[]){1, 0, 2, 3},
      (const double[]){4.0, 5.0, -1.0, 2.0}, 0));
  c = perflibs_spmat_create_null(4, 4);
  CHECK_TRUE(c != NULL, "spadd null output creation failed");
  CHECK_STATUS(perflibs_spadd_optimize(
      PERFLIBS_SPARSE_OPERATION_NOTRANS, PERFLIBS_SPARSE_OPERATION_NOTRANS,
      PERFLIBS_SPARSE_SCALAR_ANY, a, PERFLIBS_SPARSE_SCALAR_ANY, b, c));
  CHECK_STATUS(perflibs_spadd_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                                     PERFLIBS_SPARSE_OPERATION_NOTRANS, 2.0, a,
                                     -1.0, b, c));
  CHECK_STATUS(
      perflibs_spmat_export_dense_d(c, PERFLIBS_ROW_MAJOR, &m, &n, &dense));
  CHECK_DOUBLE_ARRAY(dense, expected_spadd, 16, 1e-12);
  free(dense);
  dense = NULL;
  CHECK_STATUS(perflibs_spmat_destroy(a));
  CHECK_STATUS(perflibs_spmat_destroy(b));
  CHECK_STATUS(perflibs_spmat_destroy(c));

  const double denseA_spmm[16] = {1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0,
                                  0.0, 0.0, 4.0, 1.0, 2.0, 0.0, 0.0, 5.0};
  const double denseB_spmm[16] = {1.0, 2.0, 0.0, 1.0, 3.0, 4.0, 0.0, 2.0,
                                  5.0, 6.0, 1.0, 0.0, 7.0, 8.0, 0.0, 3.0};
  compute_dense_mm(denseA_spmm, denseB_spmm, expected_spmm);
  CHECK_STATUS(perflibs_spmat_create_csr_d(
      &a, 4, 4, (const perflibs_int_t[]){0, 2, 3, 5, 7},
      (const perflibs_int_t[]){0, 2, 1, 2, 3, 0, 3},
      (const double[]){1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 5.0}, 0));
  CHECK_STATUS(perflibs_spmat_create_dense_d(
      &b, PERFLIBS_ROW_MAJOR, 4, 4, 4,
      (const double[]){1.0, 2.0, 0.0, 1.0, 3.0, 4.0, 0.0, 2.0, 5.0, 6.0, 1.0,
                       0.0, 7.0, 8.0, 0.0, 3.0},
      0));
  c = perflibs_spmat_create_null(4, 4);
  CHECK_TRUE(c != NULL, "spmm null output creation failed");
  CHECK_STATUS(perflibs_spmm_optimize(
      PERFLIBS_SPARSE_OPERATION_NOTRANS, PERFLIBS_SPARSE_OPERATION_NOTRANS,
      PERFLIBS_SPARSE_SCALAR_ANY, a, b, PERFLIBS_SPARSE_SCALAR_ZERO, c));
  CHECK_STATUS(perflibs_spmm_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                                    PERFLIBS_SPARSE_OPERATION_NOTRANS, 1.0, a,
                                    b, 0.0, c));
  CHECK_STATUS(
      perflibs_spmat_export_dense_d(c, PERFLIBS_ROW_MAJOR, &m, &n, &dense));
  CHECK_DOUBLE_ARRAY(dense, expected_spmm, 16, 1e-12);
  free(dense);
  dense = NULL;
  CHECK_STATUS(perflibs_spmat_destroy(a));
  CHECK_STATUS(perflibs_spmat_destroy(b));
  CHECK_STATUS(perflibs_spmat_destroy(c));

  const double denseA_spelmm[16] = {1.0, 0.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0,
                                    0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 5.0, 6.0};
  const double denseB_spelmm[16] = {4.0, 0.0, 0.0, 0.0, 5.0, 6.0, 0.0, 0.0,
                                    0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 8.0, 9.0};
  compute_dense_hadamard(denseA_spelmm, denseB_spelmm, expected_spelmm);
  CHECK_STATUS(perflibs_spmat_create_csr_d(
      &a, 4, 4, (const perflibs_int_t[]){0, 1, 3, 4, 6},
      (const perflibs_int_t[]){0, 0, 1, 2, 2, 3},
      (const double[]){1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, 0));
  CHECK_STATUS(perflibs_spmat_create_csr_d(
      &b, 4, 4, (const perflibs_int_t[]){0, 1, 3, 4, 6},
      (const perflibs_int_t[]){0, 0, 1, 2, 2, 3},
      (const double[]){4.0, 5.0, 6.0, 7.0, 8.0, 9.0}, 0));
  c = perflibs_spmat_create_null(4, 4);
  CHECK_TRUE(c != NULL, "spelmm null output creation failed");
  CHECK_STATUS(perflibs_spelmm_optimize(
      PERFLIBS_SPARSE_OPERATION_NOTRANS, PERFLIBS_SPARSE_OPERATION_NOTRANS,
      PERFLIBS_SPARSE_SCALAR_ANY, a, b, PERFLIBS_SPARSE_SCALAR_ZERO, c));
  CHECK_STATUS(perflibs_spelmm_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                                      PERFLIBS_SPARSE_OPERATION_NOTRANS, 1.0, a,
                                      b, 0.0, c));
  CHECK_STATUS(
      perflibs_spmat_export_dense_d(c, PERFLIBS_ROW_MAJOR, &m, &n, &dense));
  CHECK_DOUBLE_ARRAY(dense, expected_spelmm, 16, 1e-12);
  free(dense);
  dense = NULL;
  CHECK_STATUS(perflibs_spmat_destroy(a));
  CHECK_STATUS(perflibs_spmat_destroy(b));
  CHECK_STATUS(perflibs_spmat_destroy(c));

  const double denseA_sddmm[16] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,
                                   7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                                   13.0, 14.0, 15.0, 16.0};
  const double denseB_sddmm[16] = {2.0, 1.0, 0.0, 1.0, 3.0, 2.0, 1.0, 0.0,
                                   4.0, 3.0, 2.0, 1.0, 5.0, 4.0, 3.0, 2.0};
  const double mask_sddmm[16] = {1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0,
                                 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0};
  compute_dense_mm(denseA_sddmm, denseB_sddmm, expected_sddmm);
  apply_dense_mask(mask_sddmm, expected_sddmm);
  CHECK_STATUS(perflibs_spmat_create_dense_d(
      &a, PERFLIBS_ROW_MAJOR, 4, 4, 4,
      (const double[]){1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
                       12.0, 13.0, 14.0, 15.0, 16.0},
      0));
  CHECK_STATUS(perflibs_spmat_create_dense_d(
      &b, PERFLIBS_ROW_MAJOR, 4, 4, 4,
      (const double[]){2.0, 1.0, 0.0, 1.0, 3.0, 2.0, 1.0, 0.0, 4.0, 3.0, 2.0,
                       1.0, 5.0, 4.0, 3.0, 2.0},
      0));
  CHECK_STATUS(perflibs_spmat_create_dense_d(
      &c, PERFLIBS_ROW_MAJOR, 4, 4, 4,
      (const double[]){1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                       1.0, 1.0, 0.0, 1.0, 1.0},
      0));
  CHECK_STATUS(perflibs_sddmm_optimize(
      PERFLIBS_SPARSE_OPERATION_NOTRANS, PERFLIBS_SPARSE_OPERATION_NOTRANS,
      PERFLIBS_SPARSE_SCALAR_ANY, a, b, PERFLIBS_SPARSE_SCALAR_ZERO, c));
  CHECK_STATUS(perflibs_sddmm_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                                     PERFLIBS_SPARSE_OPERATION_NOTRANS, 1.0, a,
                                     b, 0.0, c));
  CHECK_STATUS(
      perflibs_spmat_export_dense_d(c, PERFLIBS_ROW_MAJOR, &m, &n, &dense));
  CHECK_DOUBLE_ARRAY(dense, expected_sddmm, 16, 1e-12);
  free(dense);
  dense = NULL;
  CHECK_STATUS(perflibs_spmat_destroy(a));
  CHECK_STATUS(perflibs_spmat_destroy(b));
  CHECK_STATUS(perflibs_spmat_destroy(c));

  return EXIT_SUCCESS;
}

static int test_spsv_real_and_complex_spmv() {
  perflibs_spmat_t triangular = NULL;
  perflibs_spmat_t complex_mat = NULL;
  double x[4] = {0.0, 0.0, 0.0, 0.0};
  const double y_notrans[4] = {1.0, 4.0, 9.0, 16.0};
  const double y_trans[4] = {5.0, 11.0, 19.0, 4.0};
  const double expected_x[4] = {1.0, 2.0, 3.0, 4.0};
  perflibs_doublecomplex_t zx[4] = {TEST_Z(1.0, -1.0), TEST_Z(2.0, 2.0),
                                    TEST_Z(-1.0, 0.5), TEST_Z(0.5, -2.0)};
  perflibs_doublecomplex_t zy[4] = {TEST_Z(0.0, 0.0), TEST_Z(0.0, 0.0),
                                    TEST_Z(0.0, 0.0), TEST_Z(0.0, 0.0)};

  CHECK_STATUS(perflibs_spmat_create_csr_d(
      &triangular, 4, 4, (const perflibs_int_t[]){0, 1, 3, 5, 7},
      (const perflibs_int_t[]){0, 0, 1, 1, 2, 2, 3},
      (const double[]){1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0}, 0));
  CHECK_STATUS(perflibs_spmat_hint(triangular, PERFLIBS_SPARSE_HINT_STRUCTURE,
                                   PERFLIBS_SPARSE_STRUCTURE_TRIANGULAR));
  CHECK_STATUS(perflibs_spmat_hint(triangular,
                                   PERFLIBS_SPARSE_HINT_SPSV_OPERATION,
                                   PERFLIBS_SPARSE_OPERATION_NOTRANS));
  CHECK_STATUS(perflibs_spmat_hint(triangular,
                                   PERFLIBS_SPARSE_HINT_SPSV_INVOCATIONS,
                                   PERFLIBS_SPARSE_INVOCATIONS_FEW));
  CHECK_STATUS(perflibs_spsv_optimize(triangular));
  CHECK_STATUS(perflibs_spsv_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                                    triangular, x, 1.0, y_notrans));
  CHECK_DOUBLE_ARRAY(x, expected_x, 4, 1e-12);
  x[0] = x[1] = x[2] = x[3] = 0.0;
  CHECK_STATUS(perflibs_spsv_exec_d(PERFLIBS_SPARSE_OPERATION_TRANS, triangular,
                                    x, 1.0, y_trans));
  CHECK_DOUBLE_ARRAY(x, expected_x, 4, 1e-12);
  CHECK_STATUS(perflibs_spmat_destroy(triangular));

  CHECK_STATUS(perflibs_spmat_create_csr_z(
      &complex_mat, 4, 4, (const perflibs_int_t[]){0, 1, 2, 3, 4},
      (const perflibs_int_t[]){0, 1, 2, 3},
      (const perflibs_doublecomplex_t[]){TEST_Z(1.0, 1.0), TEST_Z(2.0, -1.0),
                                         TEST_Z(-1.0, 2.0), TEST_Z(3.0, 0.5)},
      0));
  CHECK_STATUS(perflibs_spmv_exec_z(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                                    TEST_Z(1.0, 0.0), complex_mat, zx,
                                    TEST_Z(0.0, 0.0), zy));
  CHECK_DOUBLECOMPLEX_NEAR(zy[0], TEST_Z(2.0, 0.0), 1e-12);
  CHECK_DOUBLECOMPLEX_NEAR(zy[1], TEST_Z(6.0, 2.0), 1e-12);
  CHECK_DOUBLECOMPLEX_NEAR(zy[2], TEST_Z(0.0, -2.5), 1e-12);
  CHECK_DOUBLECOMPLEX_NEAR(zy[3], TEST_Z(2.5, -5.75), 1e-12);
  CHECK_STATUS(perflibs_spmat_destroy(complex_mat));

  return EXIT_SUCCESS;
}

int main() {
  if (test_spadd_spmm_spelmm_sddmm() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (test_spsv_real_and_complex_spmv() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
