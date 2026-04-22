/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "perflibs_sparse.h"
#include "test_utils.h"

#include <stdlib.h>

#define N 100

static int test_sparse_vector_lifecycle_and_real_ops() {
  perflibs_spvec_t x = NULL;
  perflibs_spvec_t gathered = NULL;
  perflibs_int_t index_base = -1, n = -1, nnz = -1;
  perflibs_int_t indx_storage[3] = {0, 0, 0};
  double val_storage[3] = {0.0, 0.0, 0.0};
  const perflibs_int_t indx[3] = {1, 4, 6};
  const double vals[3] = {1.5, -2.0, 3.0};
  const double dense_input[N] = {[1] = 4.0, [3] = -1.0, [5] = 2.0};
  double scattered[N] = {0.0};
  double dense_y[N] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  double wax_y[N];
  double wax_out[N] = {0.0};
  double dot_result = 0.0;
  const perflibs_int_t update_indx[2] = {1, 5};
  const double update_vals[2] = {10.0, -2.0};
  const perflibs_int_t gathered_expected_indx[3] = {1, 3, 5};
  const double gathered_expected_vals[3] = {4.0, -1.0, 2.0};
  const double scattered_expected[N] = {[1] = 4.0, [3] = -1.0, [5] = 2.0};
  const double updated_scatter_expected[N] = {
      [1] = 10.0, [3] = -1.0, [5] = -2.0};
  const double axpby_expected[N] = {-1.0, 18.0, -3.0, -6.0, -5.0, -10.0};
  double waxpby_expected[N];
  double rot_y_expected[N];

  for (perflibs_int_t i = 0; i < N; ++i) {
    wax_y[i] = 1.0;
    waxpby_expected[i] = 2.0;
    rot_y_expected[i] = 2.0;
  }
  waxpby_expected[1] = 7.0;
  waxpby_expected[3] = 1.5;
  waxpby_expected[5] = 1.0;
  rot_y_expected[1] = -10.0;
  rot_y_expected[3] = 1.0;

  CHECK_STATUS(perflibs_spvec_create_d(&x, 1, N, 3, indx, vals, 0));
  CHECK_STATUS(perflibs_spvec_query(x, &index_base, &n, &nnz));
  CHECK_INT_EQ(index_base, 1);
  CHECK_INT_EQ(n, N);
  CHECK_INT_EQ(nnz, 3);
  CHECK_STATUS(perflibs_spvec_export_d(x, &index_base, &n, &nnz, indx_storage,
                                       val_storage));
  CHECK_INT_EQ(index_base, 1);
  CHECK_INT_ARRAY(indx_storage, indx, 3);
  CHECK_DOUBLE_ARRAY(val_storage, vals, 3, 1e-12);
  CHECK_STATUS(perflibs_spvec_destroy(x));
  x = NULL;

  CHECK_STATUS(perflibs_spvec_gather_d(dense_input, 0, N, &gathered, 0));
  CHECK_STATUS(perflibs_spvec_query(gathered, &index_base, &n, &nnz));
  CHECK_INT_EQ(index_base, 0);
  CHECK_INT_EQ(n, N);
  CHECK_INT_EQ(nnz, 3);
  CHECK_STATUS(perflibs_spvec_export_d(gathered, &index_base, &n, &nnz,
                                       indx_storage, val_storage));
  CHECK_INT_ARRAY(indx_storage, gathered_expected_indx, 3);
  CHECK_DOUBLE_ARRAY(val_storage, gathered_expected_vals, 3, 1e-12);

  CHECK_STATUS(perflibs_spvec_scatter_d(gathered, scattered));
  CHECK_DOUBLE_ARRAY(scattered, scattered_expected, N, 1e-12);

  CHECK_STATUS(perflibs_spvec_update_d(gathered, 2, update_indx, update_vals));
  CHECK_STATUS(perflibs_spvec_scatter_d(gathered, scattered));
  CHECK_DOUBLE_ARRAY(scattered, updated_scatter_expected, N, 1e-12);

  CHECK_STATUS(perflibs_spdot_exec_d(gathered, dense_y, &dot_result));
  CHECK_DOUBLE_NEAR(dot_result, 4.0, 1e-12);

  CHECK_STATUS(perflibs_spaxpby_exec_d(2.0, gathered, -1.0, dense_y));
  CHECK_DOUBLE_ARRAY(dense_y, axpby_expected, N, 1e-12);

  CHECK_STATUS(perflibs_spwaxpby_exec_d(0.5, gathered, 2.0, wax_y, wax_out));
  CHECK_DOUBLE_ARRAY(wax_out, waxpby_expected, N, 1e-12);

  CHECK_STATUS(perflibs_sprot_exec_d(gathered, wax_out, 0.0, 1.0));
  CHECK_DOUBLE_ARRAY(wax_out, rot_y_expected, N, 1e-12);

  CHECK_STATUS(perflibs_spvec_destroy(gathered));
  return EXIT_SUCCESS;
}

static int test_complex_vector_ops() {
  perflibs_spvec_t x = NULL;
  perflibs_spvec_t gathered = NULL;
  const perflibs_int_t indx[2] = {0, 2};
  const perflibs_doublecomplex_t vals[2] = {TEST_Z(1.0, 1.0),
                                            TEST_Z(2.0, -1.0)};
  const perflibs_doublecomplex_t dense[N] = {
      [1] = TEST_Z(3.0, 0.0), [2] = TEST_Z(0.0, 4.0)};
  perflibs_doublecomplex_t y[N] = {
      [0] = TEST_Z(1.0, 0.0), [1] = TEST_Z(0.0, 2.0), [2] = TEST_Z(3.0, -1.0)};
  perflibs_doublecomplex_t w[N] = {0};
  perflibs_doublecomplex_t result = TEST_Z(0.0, 0.0);
  perflibs_doublecomplex_t scattered[N] = {0};

  CHECK_STATUS(perflibs_spvec_create_z(&x, 0, N, 2, indx, vals, 0));
  CHECK_STATUS(perflibs_spdotu_exec_z(x, y, &result));
  CHECK_DOUBLECOMPLEX_NEAR(result, TEST_Z(6.0, -4.0), 1e-12);
  CHECK_STATUS(perflibs_spdotc_exec_z(x, y, &result));
  CHECK_DOUBLECOMPLEX_NEAR(result, TEST_Z(8.0, 0.0), 1e-12);
  CHECK_STATUS(
      perflibs_spaxpby_exec_z(TEST_Z(1.0, 0.0), x, TEST_Z(1.0, 0.0), y));
  CHECK_DOUBLECOMPLEX_NEAR(y[0], TEST_Z(2.0, 1.0), 1e-12);
  CHECK_DOUBLECOMPLEX_NEAR(y[1], TEST_Z(0.0, 2.0), 1e-12);
  CHECK_DOUBLECOMPLEX_NEAR(y[2], TEST_Z(5.0, -2.0), 1e-12);
  CHECK_STATUS(
      perflibs_spwaxpby_exec_z(TEST_Z(0.5, 0.0), x, TEST_Z(2.0, 0.0), y, w));
  CHECK_DOUBLECOMPLEX_NEAR(w[0], TEST_Z(4.5, 2.5), 1e-12);
  CHECK_DOUBLECOMPLEX_NEAR(w[1], TEST_Z(0.0, 4.0), 1e-12);
  CHECK_DOUBLECOMPLEX_NEAR(w[2], TEST_Z(11.0, -4.5), 1e-12);
  CHECK_STATUS(perflibs_spvec_destroy(x));

  CHECK_STATUS(perflibs_spvec_gather_z(dense, 0, N, &gathered, 0));
  CHECK_STATUS(perflibs_spvec_scatter_z(gathered, scattered));
  CHECK_DOUBLECOMPLEX_NEAR(scattered[0], TEST_Z(0.0, 0.0), 1e-12);
  CHECK_DOUBLECOMPLEX_NEAR(scattered[1], TEST_Z(3.0, 0.0), 1e-12);
  CHECK_DOUBLECOMPLEX_NEAR(scattered[2], TEST_Z(0.0, 4.0), 1e-12);
  CHECK_STATUS(perflibs_spvec_destroy(gathered));

  return EXIT_SUCCESS;
}

int main() {
  if (test_sparse_vector_lifecycle_and_real_ops() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (test_complex_vector_ops() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
