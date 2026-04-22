/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "perflibs_sparse.h"
#include "test_utils.h"

#include <stdlib.h>

static double access_dense_rowmajor_d(const double *A, perflibs_int_t n,
                                      enum perflibs_sparse_hint_value trans,
                                      perflibs_int_t i, perflibs_int_t j) {
  if (trans == PERFLIBS_SPARSE_OPERATION_NOTRANS) {
    return A[i * n + j];
  }
  return A[j * n + i];
}

static void
dense_mm_expected(const double *A, perflibs_int_t Am, perflibs_int_t An,
                  enum perflibs_sparse_hint_value transA, const double *B,
                  perflibs_int_t Bm, perflibs_int_t Bn,
                  enum perflibs_sparse_hint_value transB, double alpha,
                  double beta, const double *C0, double *Cexp) {
  const perflibs_int_t m =
      transA == PERFLIBS_SPARSE_OPERATION_NOTRANS ? Am : An;
  const perflibs_int_t n =
      transA == PERFLIBS_SPARSE_OPERATION_NOTRANS ? An : Am;

  for (perflibs_int_t i = 0; i < m; ++i) {
    for (perflibs_int_t j = 0; j < n; ++j) {
      const double aval = access_dense_rowmajor_d(A, An, transA, i, j);
      const double bval = access_dense_rowmajor_d(B, Bn, transB, i, j);
      Cexp[i * n + j] =
          alpha * (aval * bval) + beta * (C0 ? C0[i * n + j] : 0.0);
    }
  }
}

static perflibs_doublecomplex_t
access_dense_rowmajor_z(const perflibs_doublecomplex_t *A, perflibs_int_t n,
                        enum perflibs_sparse_hint_value trans, perflibs_int_t i,
                        perflibs_int_t j) {
  if (trans == PERFLIBS_SPARSE_OPERATION_NOTRANS) {
    return A[i * n + j];
  }
  if (trans == PERFLIBS_SPARSE_OPERATION_TRANS) {
    return A[j * n + i];
  }
  return test_conj(A[j * n + i]);
}

static void dense_matvec_expected_z(const perflibs_doublecomplex_t *A,
                                    perflibs_int_t m, perflibs_int_t n,
                                    enum perflibs_sparse_hint_value trans,
                                    const perflibs_doublecomplex_t *x,
                                    perflibs_doublecomplex_t *y) {
  const perflibs_int_t rows =
      trans == PERFLIBS_SPARSE_OPERATION_NOTRANS ? m : n;
  const perflibs_int_t cols =
      trans == PERFLIBS_SPARSE_OPERATION_NOTRANS ? n : m;
  for (perflibs_int_t i = 0; i < rows; ++i) {
    perflibs_doublecomplex_t sum = TEST_Z(0.0, 0.0);
    for (perflibs_int_t j = 0; j < cols; ++j) {
      sum = test_cadd(
          sum, test_cmul(access_dense_rowmajor_z(A, n, trans, i, j), x[j]));
    }
    y[i] = sum;
  }
}

static void rowmajor_to_colmajor_z(const perflibs_doublecomplex_t *src_rm,
                                   perflibs_int_t m, perflibs_int_t n,
                                   perflibs_doublecomplex_t *dst_cm) {
  for (perflibs_int_t i = 0; i < m; ++i) {
    for (perflibs_int_t j = 0; j < n; ++j) {
      dst_cm[j * m + i] = src_rm[i * n + j];
    }
  }
}

static int test_dense_spmv_layout_and_transpose() {
  const double vals_rm[] = {1.0, 2.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0,
                            5.0, 0.0, 6.0, 0.0, 0.0, 0.0, 7.0, 8.0};
  const double vals_cm[] = {1.0, 0.0, 5.0, 0.0, 2.0, 3.0, 0.0, 0.0,
                            0.0, 4.0, 6.0, 7.0, 0.0, 0.0, 0.0, 8.0};
  const double x_notrans[] = {1.0, -1.0, 2.0, 3.0};
  const double x_trans[] = {2.0, -1.0, 0.5, 3.0};
  const double y_init_notrans[] = {10.0, 20.0, 30.0, 40.0};
  double y[4] = {0.0, 0.0, 0.0, 0.0};
  perflibs_spmat_t A_rm = NULL;
  perflibs_spmat_t A_cm = NULL;

  CHECK_STATUS(perflibs_spmat_create_dense_d(&A_rm, PERFLIBS_ROW_MAJOR, 4, 4, 4,
                                             vals_rm, 0));
  CHECK_STATUS(perflibs_spmat_create_dense_d(&A_cm, PERFLIBS_COL_MAJOR, 4, 4, 4,
                                             vals_cm, 0));

  y[0] = y_init_notrans[0];
  y[1] = y_init_notrans[1];
  y[2] = y_init_notrans[2];
  y[3] = y_init_notrans[3];
  CHECK_STATUS(perflibs_spmv_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS, 2.0,
                                    A_rm, x_notrans, 0.5, y));
  CHECK_DOUBLE_NEAR(y[0], 3.0, 1e-12);
  CHECK_DOUBLE_NEAR(y[1], 20.0, 1e-12);
  CHECK_DOUBLE_NEAR(y[2], 49.0, 1e-12);
  CHECK_DOUBLE_NEAR(y[3], 96.0, 1e-12);

  y[0] = y_init_notrans[0];
  y[1] = y_init_notrans[1];
  y[2] = y_init_notrans[2];
  y[3] = y_init_notrans[3];
  CHECK_STATUS(perflibs_spmv_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS, 2.0,
                                    A_cm, x_notrans, 0.5, y));
  CHECK_DOUBLE_NEAR(y[0], 3.0, 1e-12);
  CHECK_DOUBLE_NEAR(y[1], 20.0, 1e-12);
  CHECK_DOUBLE_NEAR(y[2], 49.0, 1e-12);
  CHECK_DOUBLE_NEAR(y[3], 96.0, 1e-12);

  y[0] = 0.0;
  y[1] = 0.0;
  y[2] = 0.0;
  y[3] = 0.0;
  CHECK_STATUS(perflibs_spmv_exec_d(PERFLIBS_SPARSE_OPERATION_TRANS, 1.0, A_rm,
                                    x_trans, 0.0, y));
  CHECK_DOUBLE_NEAR(y[0], 4.5, 1e-12);
  CHECK_DOUBLE_NEAR(y[1], 1.0, 1e-12);
  CHECK_DOUBLE_NEAR(y[2], 20.0, 1e-12);
  CHECK_DOUBLE_NEAR(y[3], 24.0, 1e-12);

  CHECK_STATUS(perflibs_spmat_destroy(A_rm));
  CHECK_STATUS(perflibs_spmat_destroy(A_cm));
  return EXIT_SUCCESS;
}

static int test_dense_spelmm_transpose_modes() {
  const double A[] = {1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 4.0,
                      5.0, 0.0, 6.0, 0.0, 0.0, 7.0, 0.0, 8.0};
  const double B[] = {5.0, 9.0,  13.0, 1.0, 6.0, 10.0, 14.0, 2.0,
                      7.0, 11.0, 15.0, 3.0, 8.0, 12.0, 16.0, 4.0};
  const double B_logical[] = {5.0,  6.0,  7.0,  8.0,  9.0, 10.0, 11.0, 12.0,
                              13.0, 14.0, 15.0, 16.0, 1.0, 2.0,  3.0,  4.0};
  double C_expected[16];
  double *C_out = NULL;
  perflibs_int_t m = -1;
  perflibs_int_t n = -1;
  perflibs_spmat_t A_sp = NULL;
  perflibs_spmat_t B_sp = NULL;
  perflibs_spmat_t C_sp = NULL;

  CHECK_STATUS(
      perflibs_spmat_create_dense_d(&A_sp, PERFLIBS_ROW_MAJOR, 4, 4, 4, A, 0));
  CHECK_STATUS(
      perflibs_spmat_create_dense_d(&B_sp, PERFLIBS_COL_MAJOR, 4, 4, 4, B, 0));
  C_sp = perflibs_spmat_create_null(4, 4);
  CHECK_TRUE(C_sp != NULL, "spmat_create_null failed");

  CHECK_STATUS(perflibs_spelmm_optimize(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                                        PERFLIBS_SPARSE_OPERATION_NOTRANS,
                                        PERFLIBS_SPARSE_SCALAR_ANY, A_sp, B_sp,
                                        PERFLIBS_SPARSE_SCALAR_ZERO, C_sp));

  dense_mm_expected(A, 4, 4, PERFLIBS_SPARSE_OPERATION_NOTRANS, B_logical, 4, 4,
                    PERFLIBS_SPARSE_OPERATION_NOTRANS, 1.25, 0.0, NULL,
                    C_expected);
  CHECK_STATUS(perflibs_spelmm_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                                      PERFLIBS_SPARSE_OPERATION_NOTRANS, 1.25,
                                      A_sp, B_sp, 0.0, C_sp));
  CHECK_STATUS(
      perflibs_spmat_export_dense_d(C_sp, PERFLIBS_ROW_MAJOR, &m, &n, &C_out));
  CHECK_DOUBLE_ARRAY(C_out, C_expected, 16, 1e-12);
  free(C_out);
  C_out = NULL;

  CHECK_STATUS(perflibs_spmat_destroy(C_sp));
  C_sp = perflibs_spmat_create_null(4, 4);
  CHECK_TRUE(C_sp != NULL, "spmat_create_null failed");
  dense_mm_expected(A, 4, 4, PERFLIBS_SPARSE_OPERATION_NOTRANS, B_logical, 4, 4,
                    PERFLIBS_SPARSE_OPERATION_TRANS, 1.0, 0.0, NULL,
                    C_expected);
  CHECK_STATUS(perflibs_spelmm_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                                      PERFLIBS_SPARSE_OPERATION_TRANS, 1.0,
                                      A_sp, B_sp, 0.0, C_sp));
  CHECK_STATUS(
      perflibs_spmat_export_dense_d(C_sp, PERFLIBS_ROW_MAJOR, &m, &n, &C_out));
  CHECK_DOUBLE_ARRAY(C_out, C_expected, 16, 1e-12);
  free(C_out);

  CHECK_STATUS(perflibs_spmat_destroy(A_sp));
  CHECK_STATUS(perflibs_spmat_destroy(B_sp));
  CHECK_STATUS(perflibs_spmat_destroy(C_sp));
  return EXIT_SUCCESS;
}

static int test_dense_complex_spsv_and_conjtranspose() {
  const perflibs_doublecomplex_t A_rm[] = {
      TEST_Z(1.0, 1.0), TEST_Z(0.0, 0.0), TEST_Z(0.0, 0.0),
      TEST_Z(0.0, 0.0), TEST_Z(2.0, 0.0), TEST_Z(2.0, -1.0),
      TEST_Z(0.0, 0.0), TEST_Z(0.0, 0.0), TEST_Z(0.0, -1.0),
      TEST_Z(1.0, 1.0), TEST_Z(3.0, 0.0), TEST_Z(0.0, 0.0),
      TEST_Z(0.0, 0.0), TEST_Z(0.0, 0.0), TEST_Z(2.0, -1.0),
      TEST_Z(4.0, 0.0)};
  perflibs_doublecomplex_t A_cm[16];
  const perflibs_doublecomplex_t x_expected[] = {
      TEST_Z(1.0, 1.0), TEST_Z(-2.0, 1.0), TEST_Z(0.5, -0.5),
      TEST_Z(1.5, 0.25)};
  perflibs_doublecomplex_t rhs_notrans[4];
  perflibs_doublecomplex_t rhs_trans[4];
  perflibs_doublecomplex_t rhs_conj[4];
  perflibs_doublecomplex_t x[4] = {0};
  const perflibs_doublecomplex_t expected_conjT[] = {
      TEST_Z(1.0, -1.0), TEST_Z(2.0, -0.0), TEST_Z(0.0, 1.0),
      TEST_Z(0.0, -0.0), TEST_Z(0.0, -0.0), TEST_Z(2.0, 1.0),
      TEST_Z(1.0, -1.0), TEST_Z(0.0, -0.0), TEST_Z(0.0, -0.0),
      TEST_Z(0.0, -0.0), TEST_Z(3.0, -0.0), TEST_Z(2.0, 1.0),
      TEST_Z(0.0, -0.0), TEST_Z(0.0, -0.0), TEST_Z(0.0, -0.0),
      TEST_Z(4.0, -0.0)};
  perflibs_doublecomplex_t *dense_out = NULL;
  perflibs_int_t m = -1;
  perflibs_int_t n = -1;
  perflibs_spmat_t A_sp = NULL;

  rowmajor_to_colmajor_z(A_rm, 4, 4, A_cm);
  dense_matvec_expected_z(A_rm, 4, 4, PERFLIBS_SPARSE_OPERATION_NOTRANS,
                          x_expected, rhs_notrans);
  dense_matvec_expected_z(A_rm, 4, 4, PERFLIBS_SPARSE_OPERATION_TRANS,
                          x_expected, rhs_trans);
  dense_matvec_expected_z(A_rm, 4, 4, PERFLIBS_SPARSE_OPERATION_CONJTRANS,
                          x_expected, rhs_conj);

  CHECK_STATUS(perflibs_spmat_create_dense_z(&A_sp, PERFLIBS_COL_MAJOR, 4, 4, 4,
                                             A_cm, 0));
  CHECK_STATUS(perflibs_spmat_hint(A_sp, PERFLIBS_SPARSE_HINT_STRUCTURE,
                                   PERFLIBS_SPARSE_STRUCTURE_TRIANGULAR));
  CHECK_STATUS(perflibs_spmat_hint(A_sp, PERFLIBS_SPARSE_HINT_SPSV_OPERATION,
                                   PERFLIBS_SPARSE_OPERATION_NOTRANS));
  CHECK_STATUS(perflibs_spsv_optimize(A_sp));

  CHECK_STATUS(perflibs_spsv_exec_z(PERFLIBS_SPARSE_OPERATION_NOTRANS, A_sp, x,
                                    TEST_Z(1.0, 0.0), rhs_notrans));
  CHECK_DOUBLECOMPLEX_ARRAY(x, x_expected, 4, 1e-12);

  x[0] = TEST_Z(0.0, 0.0);
  x[1] = TEST_Z(0.0, 0.0);
  x[2] = TEST_Z(0.0, 0.0);
  x[3] = TEST_Z(0.0, 0.0);
  CHECK_STATUS(perflibs_spsv_exec_z(PERFLIBS_SPARSE_OPERATION_TRANS, A_sp, x,
                                    TEST_Z(1.0, 0.0), rhs_trans));
  CHECK_DOUBLECOMPLEX_ARRAY(x, x_expected, 4, 1e-12);

  x[0] = TEST_Z(0.0, 0.0);
  x[1] = TEST_Z(0.0, 0.0);
  x[2] = TEST_Z(0.0, 0.0);
  x[3] = TEST_Z(0.0, 0.0);
  CHECK_STATUS(perflibs_spsv_exec_z(PERFLIBS_SPARSE_OPERATION_CONJTRANS, A_sp,
                                    x, TEST_Z(1.0, 0.0), rhs_conj));
  CHECK_DOUBLECOMPLEX_ARRAY(x, x_expected, 4, 1e-12);

  CHECK_STATUS(
      perflibs_sptranspose_exec_z(PERFLIBS_SPARSE_OPERATION_CONJTRANS, A_sp));
  CHECK_STATUS(perflibs_spmat_export_dense_z(A_sp, PERFLIBS_ROW_MAJOR, &m, &n,
                                             &dense_out));
  CHECK_DOUBLECOMPLEX_ARRAY(dense_out, expected_conjT, 16, 1e-12);
  free(dense_out);

  CHECK_STATUS(perflibs_spmat_destroy(A_sp));
  return EXIT_SUCCESS;
}

int main() {
  if (test_dense_spmv_layout_and_transpose() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (test_dense_spelmm_transpose_modes() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (test_dense_complex_spsv_and_conjtranspose() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
