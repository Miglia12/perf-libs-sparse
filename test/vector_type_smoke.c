/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "perflibs_sparse.h"
#include "test_utils.h"

#include <stdlib.h>

#define N_DENSE 100

static int test_float_vector_smoke() {
  perflibs_spvec_t x = NULL;
  perflibs_spvec_t gathered = NULL;
  perflibs_int_t index_base = -1;
  perflibs_int_t n = -1;
  perflibs_int_t nnz = -1;
  const perflibs_int_t create_indx[3] = {1, 3, 5};
  const float create_vals[3] = {1.5f, -2.0f, 3.0f};
  perflibs_int_t export_indx[3] = {0, 0, 0};
  float export_vals[3] = {0.0f, 0.0f, 0.0f};
  const float dense_input[N_DENSE] = {[1] = 4.0f, [3] = -1.0f, [5] = 2.0f};
  float scattered[N_DENSE] = {0};
  const float dense_y[N_DENSE] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  float y_axpby[N_DENSE] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  float wax_y[N_DENSE];
  float wax_out[N_DENSE] = {0.0f};
  const perflibs_int_t update_indx[2] = {1, 5};
  const float update_vals[2] = {10.0f, -2.0f};
  const perflibs_int_t invalid_indx = 2;
  const float invalid_val = 7.0f;

  CHECK_STATUS(
      perflibs_spvec_create_s(&x, 1, N_DENSE, 3, create_indx, create_vals, 0));
  CHECK_STATUS(perflibs_spvec_query(x, &index_base, &n, &nnz));
  CHECK_INT_EQ(index_base, 1);
  CHECK_INT_EQ(n, N_DENSE);
  CHECK_INT_EQ(nnz, 3);
  CHECK_STATUS(perflibs_spvec_export_s(x, &index_base, &n, &nnz, export_indx,
                                       export_vals));
  CHECK_INT_EQ(index_base, 1);
  CHECK_INT_ARRAY(export_indx, create_indx, 3);
  CHECK_FLOAT_ARRAY(export_vals, create_vals, 3, 1e-6f);
  CHECK_STATUS(perflibs_spvec_destroy(x));
  x = NULL;

  for (perflibs_int_t i = 0; i < N_DENSE; ++i) {
    wax_y[i] = 1.0f;
  }

  CHECK_STATUS(perflibs_spvec_gather_s(dense_input, 0, N_DENSE, &gathered, 0));
  CHECK_STATUS(perflibs_spvec_query(gathered, &index_base, &n, &nnz));
  CHECK_INT_EQ(index_base, 0);
  CHECK_INT_EQ(n, N_DENSE);
  CHECK_INT_EQ(nnz, 3);
  CHECK_STATUS(perflibs_spvec_export_s(gathered, &index_base, &n, &nnz,
                                       export_indx, export_vals));
  const perflibs_int_t gathered_indx_expected[3] = {1, 3, 5};
  const float gathered_vals_expected[3] = {4.0f, -1.0f, 2.0f};
  CHECK_INT_ARRAY(export_indx, gathered_indx_expected, 3);
  CHECK_FLOAT_ARRAY(export_vals, gathered_vals_expected, 3, 1e-6f);
  CHECK_STATUS(perflibs_spvec_scatter_s(gathered, scattered));
  CHECK_FLOAT_ARRAY(scattered, dense_input, N_DENSE, 1e-6f);

  CHECK_STATUS(perflibs_spvec_update_s(gathered, 2, update_indx, update_vals));
  CHECK_STATUS(perflibs_spvec_scatter_s(gathered, scattered));
  const float updated_scatter_expected[N_DENSE] = {
      [1] = 10.0f, [3] = -1.0f, [5] = -2.0f};
  CHECK_FLOAT_ARRAY(scattered, updated_scatter_expected, N_DENSE, 1e-6f);

  float dot_result = -1.0f;
  CHECK_STATUS(perflibs_spdot_exec_s(gathered, dense_y, &dot_result));
  CHECK_DOUBLE_NEAR((double)dot_result, 4.0, 1e-6);

  CHECK_STATUS(perflibs_spaxpby_exec_s(2.0f, gathered, -1.0f, y_axpby));
  const float axpby_expected[N_DENSE] = {-1.0f, 18.0f, -3.0f,
                                         -6.0f, -5.0f, -10.0f};
  CHECK_FLOAT_ARRAY(y_axpby, axpby_expected, N_DENSE, 1e-6f);

  CHECK_STATUS(perflibs_spwaxpby_exec_s(0.5f, gathered, 2.0f, wax_y, wax_out));
  float waxpby_expected[N_DENSE];
  float rot_expected[N_DENSE];
  for (perflibs_int_t i = 0; i < N_DENSE; ++i) {
    waxpby_expected[i] = 2.0f;
    rot_expected[i] = 2.0f;
  }
  waxpby_expected[1] = 7.0f;
  waxpby_expected[3] = 1.5f;
  waxpby_expected[5] = 1.0f;
  CHECK_FLOAT_ARRAY(wax_out, waxpby_expected, N_DENSE, 1e-6f);

  CHECK_STATUS(perflibs_sprot_exec_s(gathered, wax_out, 0.0f, 1.0f));
  rot_expected[1] = -10.0f;
  rot_expected[3] = 1.0f;
  CHECK_FLOAT_ARRAY(wax_out, rot_expected, N_DENSE, 1e-6f);

  CHECK_STATUS_EQ(
      perflibs_spvec_update_s(gathered, 1, &invalid_indx, &invalid_val),
      PERFLIBS_STATUS_INPUT_PARAMETER_ERROR);
  perflibs_spvec_print_err(gathered);

  CHECK_STATUS(perflibs_spvec_destroy(gathered));
  return EXIT_SUCCESS;
}

static int test_complex_vector_smoke() {
  perflibs_spvec_t cvec = NULL;
  perflibs_spvec_t gathered = NULL;
  perflibs_int_t index_base = -1;
  perflibs_int_t n = -1;
  perflibs_int_t nnz = -1;
  const perflibs_int_t create_indx[3] = {0, 2, 4};
  const perflibs_singlecomplex_t create_vals[3] = {
      TEST_C(1.0f, 2.0f), TEST_C(-3.0f, 1.0f), TEST_C(2.0f, -0.5f)};
  perflibs_int_t export_indx[3] = {0, 0, 0};
  perflibs_singlecomplex_t export_vals[3] = {0};
  perflibs_singlecomplex_t scatter_buffer[N_DENSE] = {0};
  const perflibs_singlecomplex_t scatter_expected[N_DENSE] = {
      [0] = TEST_C(1.0f, 2.0f),
      [2] = TEST_C(-3.0f, 1.0f),
      [4] = TEST_C(2.0f, -0.5f)};
  const perflibs_singlecomplex_t dense_source[N_DENSE] = {
      [0] = TEST_C(1.0f, 0.0f),
      [1] = TEST_C(0.0f, 2.0f),
      [3] = TEST_C(3.0f, -1.0f)};
  perflibs_singlecomplex_t dense_scatter[N_DENSE] = {0};
  const perflibs_singlecomplex_t scatter_after_update_expected[N_DENSE] = {
      [0] = TEST_C(1.0f, 0.0f),
      [1] = TEST_C(0.0f, 2.0f),
      [3] = TEST_C(5.0f, 2.0f)};
  const perflibs_int_t update_indx = 3;
  const perflibs_singlecomplex_t update_vals = TEST_C(5.0f, 2.0f);
  perflibs_singlecomplex_t dot_y[N_DENSE] = {[0] = TEST_C(1.0f, 0.0f),
                                             [1] = TEST_C(2.0f, 1.0f),
                                             [3] = TEST_C(4.0f, 4.0f)};
  perflibs_singlecomplex_t dot_result = TEST_C(0.0f, 0.0f);
  perflibs_singlecomplex_t y_axpby[N_DENSE] = {[0] = TEST_C(1.0f, 1.0f),
                                               [1] = TEST_C(2.0f, -1.0f),
                                               [2] = TEST_C(3.0f, 0.0f),
                                               [3] = TEST_C(4.0f, 2.0f)};
  const perflibs_singlecomplex_t wax_y[N_DENSE] = {[0] = TEST_C(1.0f, 0.0f),
                                                   [1] = TEST_C(1.0f, 0.0f),
                                                   [2] = TEST_C(1.0f, 0.0f),
                                                   [3] = TEST_C(1.0f, 0.0f)};
  perflibs_singlecomplex_t wax_out[N_DENSE] = {0};

  CHECK_STATUS(perflibs_spvec_create_c(&cvec, 0, N_DENSE, 3, create_indx,
                                       create_vals, 0));
  CHECK_STATUS(perflibs_spvec_query(cvec, &index_base, &n, &nnz));
  CHECK_INT_EQ(index_base, 0);
  CHECK_INT_EQ(n, N_DENSE);
  CHECK_INT_EQ(nnz, 3);
  CHECK_STATUS(perflibs_spvec_export_c(cvec, &index_base, &n, &nnz, export_indx,
                                       export_vals));
  CHECK_INT_EQ(index_base, 0);
  CHECK_INT_ARRAY(export_indx, create_indx, 3);
  CHECK_SINGLECOMPLEX_ARRAY(export_vals, create_vals, 3, 1e-6f);
  CHECK_STATUS(perflibs_spvec_scatter_c(cvec, scatter_buffer));
  CHECK_SINGLECOMPLEX_ARRAY(scatter_buffer, scatter_expected, N_DENSE, 1e-6f);
  CHECK_STATUS(perflibs_spvec_destroy(cvec));
  cvec = NULL;

  CHECK_STATUS(perflibs_spvec_gather_c(dense_source, 0, N_DENSE, &gathered, 0));
  CHECK_STATUS(perflibs_spvec_query(gathered, &index_base, &n, &nnz));
  CHECK_INT_EQ(index_base, 0);
  CHECK_INT_EQ(n, N_DENSE);
  CHECK_INT_EQ(nnz, 3);
  CHECK_STATUS(perflibs_spvec_export_c(gathered, &index_base, &n, &nnz,
                                       export_indx, export_vals));
  const perflibs_int_t gathered_indx_expected[3] = {0, 1, 3};
  const perflibs_singlecomplex_t gathered_vals_expected[3] = {
      TEST_C(1.0f, 0.0f), TEST_C(0.0f, 2.0f), TEST_C(3.0f, -1.0f)};
  CHECK_INT_ARRAY(export_indx, gathered_indx_expected, 3);
  CHECK_SINGLECOMPLEX_ARRAY(export_vals, gathered_vals_expected, 3, 1e-6f);
  CHECK_STATUS(perflibs_spvec_scatter_c(gathered, dense_scatter));
  CHECK_SINGLECOMPLEX_ARRAY(dense_scatter, dense_source, N_DENSE, 1e-6f);

  CHECK_STATUS(
      perflibs_spvec_update_c(gathered, 1, &update_indx, &update_vals));
  CHECK_STATUS(perflibs_spvec_scatter_c(gathered, dense_scatter));
  CHECK_SINGLECOMPLEX_ARRAY(dense_scatter, scatter_after_update_expected,
                            N_DENSE, 1e-6f);

  CHECK_STATUS(perflibs_spdotu_exec_c(gathered, dot_y, &dot_result));
  CHECK_SINGLECOMPLEX_NEAR(dot_result, TEST_C(11.0f, 32.0f), 1e-5f);
  CHECK_STATUS(perflibs_spdotc_exec_c(gathered, dot_y, &dot_result));
  CHECK_SINGLECOMPLEX_NEAR(dot_result, TEST_C(31.0f, 8.0f), 1e-5f);

  perflibs_singlecomplex_t alpha_axpby = TEST_C(1.0f, 0.0f);
  perflibs_singlecomplex_t beta_axpby = TEST_C(0.5f, -0.5f);
  CHECK_STATUS(
      perflibs_spaxpby_exec_c(alpha_axpby, gathered, beta_axpby, y_axpby));
  const perflibs_singlecomplex_t axpby_expected[N_DENSE] = {
      TEST_C(2.0f, 0.0f), TEST_C(0.5f, 0.5f), TEST_C(1.5f, -1.5f),
      TEST_C(8.0f, 1.0f)};
  CHECK_SINGLECOMPLEX_ARRAY(y_axpby, axpby_expected, N_DENSE, 1e-6f);

  const perflibs_singlecomplex_t alpha_waxpby = TEST_C(0.5f, 0.5f);
  const perflibs_singlecomplex_t beta_waxpby = TEST_C(1.0f, 0.0f);
  CHECK_STATUS(perflibs_spwaxpby_exec_c(alpha_waxpby, gathered, beta_waxpby,
                                        wax_y, wax_out));
  const perflibs_singlecomplex_t waxpby_expected[N_DENSE] = {
      TEST_C(1.5f, 0.5f), TEST_C(0.0f, 1.0f), TEST_C(1.0f, 0.0f),
      TEST_C(2.5f, 3.5f)};
  CHECK_SINGLECOMPLEX_ARRAY(wax_out, waxpby_expected, N_DENSE, 1e-6f);

  CHECK_STATUS(
      perflibs_sprot_exec_c(gathered, wax_out, 0.0f, TEST_C(1.0f, 0.0f)));
  const perflibs_singlecomplex_t rot_expected[N_DENSE] = {
      TEST_C(-1.0f, 0.0f), TEST_C(0.0f, -2.0f), TEST_C(1.0f, 0.0f),
      TEST_C(-5.0f, -2.0f)};
  CHECK_SINGLECOMPLEX_ARRAY(wax_out, rot_expected, N_DENSE, 1e-6f);

  CHECK_STATUS(perflibs_spvec_destroy(gathered));
  return EXIT_SUCCESS;
}

static int test_double_vector_smoke() {
  perflibs_spvec_t x = NULL;
  perflibs_spvec_t gathered = NULL;
  perflibs_int_t index_base = -1;
  perflibs_int_t n = -1;
  perflibs_int_t nnz = -1;
  const perflibs_int_t create_indx[3] = {1, 3, 5};
  const double create_vals[3] = {1.5, -2.0, 3.0};
  perflibs_int_t export_indx[3] = {0, 0, 0};
  double export_vals[3] = {0.0, 0.0, 0.0};
  const double dense_input[N_DENSE] = {[1] = 4.0, [3] = -1.0, [5] = 2.0};
  double scattered[N_DENSE] = {0};
  const double dense_y[N_DENSE] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  double y_axpby[N_DENSE] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  double wax_y[N_DENSE];
  double wax_out[N_DENSE] = {0.0};
  const perflibs_int_t update_indx[2] = {1, 5};
  const double update_vals[2] = {10.0, -2.0};
  const perflibs_int_t invalid_indx = 2;
  const double invalid_val = 7.0;

  CHECK_STATUS(
      perflibs_spvec_create_d(&x, 1, N_DENSE, 3, create_indx, create_vals, 0));
  CHECK_STATUS(perflibs_spvec_query(x, &index_base, &n, &nnz));
  CHECK_INT_EQ(index_base, 1);
  CHECK_INT_EQ(n, N_DENSE);
  CHECK_INT_EQ(nnz, 3);
  CHECK_STATUS(perflibs_spvec_export_d(x, &index_base, &n, &nnz, export_indx,
                                       export_vals));
  CHECK_INT_EQ(index_base, 1);
  CHECK_INT_ARRAY(export_indx, create_indx, 3);
  CHECK_DOUBLE_ARRAY(export_vals, create_vals, 3, 1e-12);
  CHECK_STATUS(perflibs_spvec_destroy(x));
  x = NULL;

  for (perflibs_int_t i = 0; i < N_DENSE; ++i) {
    wax_y[i] = 1.0;
  }

  CHECK_STATUS(perflibs_spvec_gather_d(dense_input, 0, N_DENSE, &gathered, 0));
  CHECK_STATUS(perflibs_spvec_query(gathered, &index_base, &n, &nnz));
  CHECK_INT_EQ(index_base, 0);
  CHECK_INT_EQ(n, N_DENSE);
  CHECK_INT_EQ(nnz, 3);
  CHECK_STATUS(perflibs_spvec_export_d(gathered, &index_base, &n, &nnz,
                                       export_indx, export_vals));
  const perflibs_int_t gathered_indx_expected[3] = {1, 3, 5};
  const double gathered_vals_expected[3] = {4.0, -1.0, 2.0};
  CHECK_INT_ARRAY(export_indx, gathered_indx_expected, 3);
  CHECK_DOUBLE_ARRAY(export_vals, gathered_vals_expected, 3, 1e-12);
  CHECK_STATUS(perflibs_spvec_scatter_d(gathered, scattered));
  CHECK_DOUBLE_ARRAY(scattered, dense_input, N_DENSE, 1e-12);

  CHECK_STATUS(perflibs_spvec_update_d(gathered, 2, update_indx, update_vals));
  CHECK_STATUS(perflibs_spvec_scatter_d(gathered, scattered));
  const double updated_scatter_expected[N_DENSE] = {
      [1] = 10.0, [3] = -1.0, [5] = -2.0};
  CHECK_DOUBLE_ARRAY(scattered, updated_scatter_expected, N_DENSE, 1e-12);

  double dot_result = -1.0;
  CHECK_STATUS(perflibs_spdot_exec_d(gathered, dense_y, &dot_result));
  CHECK_DOUBLE_NEAR(dot_result, 4.0, 1e-12);

  CHECK_STATUS(perflibs_spaxpby_exec_d(2.0, gathered, -1.0, y_axpby));
  const double axpby_expected[N_DENSE] = {-1.0, 18.0, -3.0, -6.0, -5.0, -10.0};
  CHECK_DOUBLE_ARRAY(y_axpby, axpby_expected, N_DENSE, 1e-12);

  CHECK_STATUS(perflibs_spwaxpby_exec_d(0.5, gathered, 2.0, wax_y, wax_out));
  double waxpby_expected[N_DENSE];
  for (perflibs_int_t i = 0; i < N_DENSE; ++i) {
    waxpby_expected[i] = 2.0;
  }
  waxpby_expected[1] = 7.0;
  waxpby_expected[3] = 1.5;
  waxpby_expected[5] = 1.0;
  CHECK_DOUBLE_ARRAY(wax_out, waxpby_expected, N_DENSE, 1e-12);

  CHECK_STATUS(perflibs_sprot_exec_d(gathered, wax_out, 0.0, 1.0));
  double rot_expected[N_DENSE];
  for (perflibs_int_t i = 0; i < N_DENSE; ++i) {
    rot_expected[i] = 2.0;
  }
  rot_expected[1] = -10.0;
  rot_expected[3] = 1.0;
  CHECK_DOUBLE_ARRAY(wax_out, rot_expected, N_DENSE, 1e-12);

  CHECK_STATUS_EQ(
      perflibs_spvec_update_d(gathered, 1, &invalid_indx, &invalid_val),
      PERFLIBS_STATUS_INPUT_PARAMETER_ERROR);
  perflibs_spvec_print_err(gathered);
  CHECK_STATUS(perflibs_spvec_destroy(gathered));
  return EXIT_SUCCESS;
}

static int test_complex_double_vector_smoke() {
  perflibs_spvec_t cvec = NULL;
  perflibs_spvec_t gathered = NULL;
  perflibs_int_t index_base = -1;
  perflibs_int_t n = -1;
  perflibs_int_t nnz = -1;
  const perflibs_int_t create_indx[3] = {0, 2, 4};
  const perflibs_doublecomplex_t create_vals[3] = {
      TEST_Z(1.0, 2.0), TEST_Z(-3.0, 1.0), TEST_Z(2.0, -0.5)};
  perflibs_int_t export_indx[3] = {0, 0, 0};
  perflibs_doublecomplex_t export_vals[3] = {0};
  perflibs_doublecomplex_t scatter_buffer[N_DENSE] = {0};
  const perflibs_doublecomplex_t scatter_expected[N_DENSE] = {
      [0] = TEST_Z(1.0, 2.0), [2] = TEST_Z(-3.0, 1.0), [4] = TEST_Z(2.0, -0.5)};
  const perflibs_doublecomplex_t dense_source[N_DENSE] = {
      [0] = TEST_Z(1.0, 0.0), [1] = TEST_Z(0.0, 2.0), [3] = TEST_Z(3.0, -1.0)};
  perflibs_doublecomplex_t dense_scatter[N_DENSE] = {0};
  const perflibs_doublecomplex_t scatter_after_update_expected[N_DENSE] = {
      [0] = TEST_Z(1.0, 0.0), [1] = TEST_Z(0.0, 2.0), [3] = TEST_Z(5.0, 2.0)};
  const perflibs_int_t update_indx = 3;
  const perflibs_doublecomplex_t update_vals = TEST_Z(5.0, 2.0);
  const perflibs_doublecomplex_t dot_y[N_DENSE] = {
      [0] = TEST_Z(1.0, 0.0), [1] = TEST_Z(2.0, 1.0), [3] = TEST_Z(4.0, 4.0)};
  perflibs_doublecomplex_t dot_result = TEST_Z(0.0, 0.0);
  perflibs_doublecomplex_t y_axpby[N_DENSE] = {[0] = TEST_Z(1.0, 1.0),
                                               [1] = TEST_Z(2.0, -1.0),
                                               [2] = TEST_Z(3.0, 0.0),
                                               [3] = TEST_Z(4.0, 2.0)};
  const perflibs_doublecomplex_t wax_y[N_DENSE] = {[0] = TEST_Z(1.0, 0.0),
                                                   [1] = TEST_Z(1.0, 0.0),
                                                   [2] = TEST_Z(1.0, 0.0),
                                                   [3] = TEST_Z(1.0, 0.0)};
  perflibs_doublecomplex_t wax_out[N_DENSE] = {0};

  CHECK_STATUS(perflibs_spvec_create_z(&cvec, 0, N_DENSE, 3, create_indx,
                                       create_vals, 0));
  CHECK_STATUS(perflibs_spvec_query(cvec, &index_base, &n, &nnz));
  CHECK_INT_EQ(index_base, 0);
  CHECK_INT_EQ(n, N_DENSE);
  CHECK_INT_EQ(nnz, 3);
  CHECK_STATUS(perflibs_spvec_export_z(cvec, &index_base, &n, &nnz, export_indx,
                                       export_vals));
  CHECK_INT_EQ(index_base, 0);
  CHECK_INT_ARRAY(export_indx, create_indx, 3);
  CHECK_DOUBLECOMPLEX_ARRAY(export_vals, create_vals, 3, 1e-12);
  CHECK_STATUS(perflibs_spvec_scatter_z(cvec, scatter_buffer));
  CHECK_DOUBLECOMPLEX_ARRAY(scatter_buffer, scatter_expected, N_DENSE, 1e-12);
  CHECK_STATUS(perflibs_spvec_destroy(cvec));
  cvec = NULL;

  CHECK_STATUS(perflibs_spvec_gather_z(dense_source, 0, N_DENSE, &gathered, 0));
  CHECK_STATUS(perflibs_spvec_query(gathered, &index_base, &n, &nnz));
  CHECK_INT_EQ(index_base, 0);
  CHECK_INT_EQ(n, N_DENSE);
  CHECK_INT_EQ(nnz, 3);
  CHECK_STATUS(perflibs_spvec_export_z(gathered, &index_base, &n, &nnz,
                                       export_indx, export_vals));
  const perflibs_int_t gathered_indx_expected_z[3] = {0, 1, 3};
  const perflibs_doublecomplex_t gathered_vals_expected_z[3] = {
      TEST_Z(1.0, 0.0), TEST_Z(0.0, 2.0), TEST_Z(3.0, -1.0)};
  CHECK_INT_ARRAY(export_indx, gathered_indx_expected_z, 3);
  CHECK_DOUBLECOMPLEX_ARRAY(export_vals, gathered_vals_expected_z, 3, 1e-12);
  CHECK_STATUS(perflibs_spvec_scatter_z(gathered, dense_scatter));
  CHECK_DOUBLECOMPLEX_ARRAY(dense_scatter, dense_source, N_DENSE, 1e-12);

  CHECK_STATUS(
      perflibs_spvec_update_z(gathered, 1, &update_indx, &update_vals));
  CHECK_STATUS(perflibs_spvec_scatter_z(gathered, dense_scatter));
  CHECK_DOUBLECOMPLEX_ARRAY(dense_scatter, scatter_after_update_expected,
                            N_DENSE, 1e-12);

  CHECK_STATUS(perflibs_spdotu_exec_z(gathered, dot_y, &dot_result));
  CHECK_DOUBLECOMPLEX_NEAR(dot_result, TEST_Z(11.0, 32.0), 1e-12);
  CHECK_STATUS(perflibs_spdotc_exec_z(gathered, dot_y, &dot_result));
  CHECK_DOUBLECOMPLEX_NEAR(dot_result, TEST_Z(31.0, 8.0), 1e-12);

  const perflibs_doublecomplex_t alpha_axpby = TEST_Z(1.0, 0.0);
  const perflibs_doublecomplex_t beta_axpby = TEST_Z(0.5, -0.5);
  CHECK_STATUS(
      perflibs_spaxpby_exec_z(alpha_axpby, gathered, beta_axpby, y_axpby));
  const perflibs_doublecomplex_t axpby_expected_z[N_DENSE] = {
      TEST_Z(2.0, 0.0), TEST_Z(0.5, 0.5), TEST_Z(1.5, -1.5), TEST_Z(8.0, 1.0)};
  CHECK_DOUBLECOMPLEX_ARRAY(y_axpby, axpby_expected_z, N_DENSE, 1e-12);

  const perflibs_doublecomplex_t alpha_waxpby = TEST_Z(0.5, 0.5);
  const perflibs_doublecomplex_t beta_waxpby = TEST_Z(1.0, 0.0);
  CHECK_STATUS(perflibs_spwaxpby_exec_z(alpha_waxpby, gathered, beta_waxpby,
                                        wax_y, wax_out));
  const perflibs_doublecomplex_t waxpby_expected_z[N_DENSE] = {
      TEST_Z(1.5, 0.5), TEST_Z(0.0, 1.0), TEST_Z(1.0, 0.0), TEST_Z(2.5, 3.5)};
  CHECK_DOUBLECOMPLEX_ARRAY(wax_out, waxpby_expected_z, N_DENSE, 1e-12);

  CHECK_STATUS(perflibs_sprot_exec_z(gathered, wax_out, 0.0, TEST_Z(1.0, 0.0)));
  const perflibs_doublecomplex_t rot_expected_z[N_DENSE] = {
      TEST_Z(-1.0, 0.0), TEST_Z(0.0, -2.0), TEST_Z(1.0, 0.0),
      TEST_Z(-5.0, -2.0)};
  CHECK_DOUBLECOMPLEX_ARRAY(wax_out, rot_expected_z, N_DENSE, 1e-12);

  CHECK_STATUS(perflibs_spvec_destroy(gathered));
  return EXIT_SUCCESS;
}

int main() {
  if (test_float_vector_smoke() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (test_double_vector_smoke() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (test_complex_vector_smoke() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (test_complex_double_vector_smoke() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
