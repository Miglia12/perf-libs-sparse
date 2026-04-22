/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#ifndef TEST_TEST_UTILS_H
#define TEST_TEST_UTILS_H

#include "perflibs_sparse.h"

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
static inline perflibs_singlecomplex_t test_make_c(float re, float im) {
  perflibs_singlecomplex_t z;
  z._Val[0] = re;
  z._Val[1] = im;
  return z;
}

static inline perflibs_doublecomplex_t test_make_z(double re, double im) {
  perflibs_doublecomplex_t z;
  z._Val[0] = re;
  z._Val[1] = im;
  return z;
}

static inline float test_crealf(perflibs_singlecomplex_t z) {
  return z._Val[0];
}
static inline float test_cimagf(perflibs_singlecomplex_t z) {
  return z._Val[1];
}
static inline double test_creal(perflibs_doublecomplex_t z) {
  return z._Val[0];
}
static inline double test_cimag(perflibs_doublecomplex_t z) {
  return z._Val[1];
}
#else
static inline perflibs_singlecomplex_t test_make_c(float re, float im) {
  return re + im * I;
}

static inline perflibs_doublecomplex_t test_make_z(double re, double im) {
  return re + im * I;
}

static inline float test_crealf(perflibs_singlecomplex_t z) {
  return crealf(z);
}
static inline float test_cimagf(perflibs_singlecomplex_t z) {
  return cimagf(z);
}
static inline double test_creal(perflibs_doublecomplex_t z) { return creal(z); }
static inline double test_cimag(perflibs_doublecomplex_t z) { return cimag(z); }
#endif

static inline perflibs_singlecomplex_t test_caddf(perflibs_singlecomplex_t a,
                                                  perflibs_singlecomplex_t b) {
  return test_make_c(test_crealf(a) + test_crealf(b),
                     test_cimagf(a) + test_cimagf(b));
}

static inline perflibs_doublecomplex_t test_cadd(perflibs_doublecomplex_t a,
                                                 perflibs_doublecomplex_t b) {
  return test_make_z(test_creal(a) + test_creal(b),
                     test_cimag(a) + test_cimag(b));
}

static inline perflibs_singlecomplex_t test_cmulf(perflibs_singlecomplex_t a,
                                                  perflibs_singlecomplex_t b) {
  const float ar = test_crealf(a);
  const float ai = test_cimagf(a);
  const float br = test_crealf(b);
  const float bi = test_cimagf(b);
  return test_make_c(ar * br - ai * bi, ar * bi + ai * br);
}

static inline perflibs_doublecomplex_t test_cmul(perflibs_doublecomplex_t a,
                                                 perflibs_doublecomplex_t b) {
  const double ar = test_creal(a);
  const double ai = test_cimag(a);
  const double br = test_creal(b);
  const double bi = test_cimag(b);
  return test_make_z(ar * br - ai * bi, ar * bi + ai * br);
}

static inline perflibs_singlecomplex_t test_conjf(perflibs_singlecomplex_t z) {
  return test_make_c(test_crealf(z), -test_cimagf(z));
}

static inline perflibs_doublecomplex_t test_conj(perflibs_doublecomplex_t z) {
  return test_make_z(test_creal(z), -test_cimag(z));
}

#define TEST_C(re, im) test_make_c((float)(re), (float)(im))
#define TEST_Z(re, im) test_make_z((double)(re), (double)(im))

static inline long long test_i64(perflibs_int_t value) {
  return (long long)value;
}

static inline int
check_singlecomplex_near_impl(const char *file, int line,
                              perflibs_singlecomplex_t got,
                              perflibs_singlecomplex_t expected, float tol) {
  const float diff_re = test_crealf(got) - test_crealf(expected);
  const float diff_im = test_cimagf(got) - test_cimagf(expected);
  if (hypotf(diff_re, diff_im) > tol) {
    fprintf(stderr,
            "CHECK_SINGLECOMPLEX_NEAR failed at %s:%d: got %.9g%+.9gi expected "
            "%.9g%+.9gi (tol %.3g)\n",
            file, line, (double)test_crealf(got), (double)test_cimagf(got),
            (double)test_crealf(expected), (double)test_cimagf(expected),
            (double)tol);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

#define CHECK_SINGLECOMPLEX_NEAR(actual, expected, tol)                        \
  do {                                                                         \
    if (check_singlecomplex_near_impl(__FILE__, __LINE__, (actual),            \
                                      (expected),                              \
                                      (float)(tol)) != EXIT_SUCCESS) {         \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  } while (0)

static inline int check_singlecomplex_array_impl(
    const char *file, int line, const perflibs_singlecomplex_t *got,
    const perflibs_singlecomplex_t *expected, perflibs_int_t n, float tol) {
  for (perflibs_int_t i = 0; i < n; ++i) {
    if (check_singlecomplex_near_impl(file, line, got[i], expected[i], tol) !=
        EXIT_SUCCESS) {
      fprintf(stderr, "  at index %lld\n", test_i64(i));
      return EXIT_FAILURE;
    }
  }
  return EXIT_SUCCESS;
}

#define CHECK_SINGLECOMPLEX_ARRAY(got, expected, n, tol)                       \
  do {                                                                         \
    if (check_singlecomplex_array_impl(__FILE__, __LINE__, (got), (expected),  \
                                       (perflibs_int_t)(n),                    \
                                       (float)(tol)) != EXIT_SUCCESS) {        \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  } while (0)

static inline int
check_doublecomplex_near_impl(const char *file, int line,
                              perflibs_doublecomplex_t got,
                              perflibs_doublecomplex_t expected, double tol) {
  const double diff_re = test_creal(got) - test_creal(expected);
  const double diff_im = test_cimag(got) - test_cimag(expected);
  if (hypot(diff_re, diff_im) > tol) {
    fprintf(stderr,
            "CHECK_DOUBLECOMPLEX_NEAR failed at %s:%d: got %.17g%+.17gi "
            "expected %.17g%+.17gi (tol %.3g)\n",
            file, line, test_creal(got), test_cimag(got), test_creal(expected),
            test_cimag(expected), tol);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

#define CHECK_DOUBLECOMPLEX_NEAR(actual, expected, tol)                        \
  do {                                                                         \
    if (check_doublecomplex_near_impl(__FILE__, __LINE__, (actual),            \
                                      (expected),                              \
                                      (double)(tol)) != EXIT_SUCCESS) {        \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  } while (0)

static inline int check_doublecomplex_array_impl(
    const char *file, int line, const perflibs_doublecomplex_t *got,
    const perflibs_doublecomplex_t *expected, perflibs_int_t n, double tol) {
  for (perflibs_int_t i = 0; i < n; ++i) {
    if (check_doublecomplex_near_impl(file, line, got[i], expected[i], tol) !=
        EXIT_SUCCESS) {
      fprintf(stderr, "  at index %lld\n", test_i64(i));
      return EXIT_FAILURE;
    }
  }
  return EXIT_SUCCESS;
}

#define CHECK_DOUBLECOMPLEX_ARRAY(got, expected, n, tol)                       \
  do {                                                                         \
    if (check_doublecomplex_array_impl(__FILE__, __LINE__, (got), (expected),  \
                                       (perflibs_int_t)(n),                    \
                                       (double)(tol)) != EXIT_SUCCESS) {       \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  } while (0)

#define CHECK_TRUE(cond, fmt, ...)                                             \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "CHECK_TRUE failed at %s:%d: " fmt "\n", __FILE__,       \
              __LINE__, ##__VA_ARGS__);                                        \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  } while (0)

#define CHECK_STATUS(expr)                                                     \
  do {                                                                         \
    perflibs_status_t _status = (expr);                                        \
    if (_status != PERFLIBS_STATUS_SUCCESS) {                                  \
      fprintf(stderr, "CHECK_STATUS failed at %s:%d: %s returned %d\n",        \
              __FILE__, __LINE__, #expr, (int)_status);                        \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  } while (0)

#define CHECK_STATUS_EQ(expr, expected)                                        \
  do {                                                                         \
    perflibs_status_t _status = (expr);                                        \
    if (_status != (expected)) {                                               \
      fprintf(                                                                 \
          stderr,                                                              \
          "CHECK_STATUS_EQ failed at %s:%d: %s returned %d, expected %d\n",    \
          __FILE__, __LINE__, #expr, (int)_status, (int)(expected));           \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  } while (0)

#define CHECK_INT_EQ(actual, expected)                                         \
  do {                                                                         \
    if ((actual) != (expected)) {                                              \
      fprintf(stderr,                                                          \
              "CHECK_INT_EQ failed at %s:%d: got %lld expected %lld\n",        \
              __FILE__, __LINE__, test_i64((actual)), test_i64((expected)));   \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  } while (0)

static inline int check_double_near_impl(const char *file, int line, double got,
                                         double expected, double tol) {
  if (fabs(got - expected) > tol) {
    fprintf(stderr,
            "CHECK_DOUBLE_NEAR failed at %s:%d: got %.17g expected %.17g "
            "(tol %.3g)\n",
            file, line, got, expected, tol);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

#define CHECK_DOUBLE_NEAR(actual, expected, tol)                               \
  do {                                                                         \
    if (check_double_near_impl(__FILE__, __LINE__, (double)(actual),           \
                               (double)(expected),                             \
                               (double)(tol)) != EXIT_SUCCESS) {               \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  } while (0)

static inline int check_float_array_impl(const char *file, int line,
                                         const float *got,
                                         const float *expected,
                                         perflibs_int_t n, float tol) {
  for (perflibs_int_t i = 0; i < n; ++i) {
    if (fabsf(got[i] - expected[i]) > tol) {
      fprintf(stderr,
              "CHECK_FLOAT_ARRAY failed at %s:%d index %d: got %.9g expected "
              "%.9g (tol %.3g)\n",
              file, line, (int)i, got[i], expected[i], (double)tol);
      return EXIT_FAILURE;
    }
  }
  return EXIT_SUCCESS;
}

#define CHECK_FLOAT_ARRAY(got, expected, n, tol)                               \
  do {                                                                         \
    if (check_float_array_impl(__FILE__, __LINE__, (got), (expected),          \
                               (perflibs_int_t)(n),                            \
                               (float)(tol)) != EXIT_SUCCESS) {                \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  } while (0)

static inline float test_access_dense_s(const float *data, perflibs_int_t cols,
                                        enum perflibs_sparse_hint_value trans,
                                        perflibs_int_t i, perflibs_int_t j) {
  if (trans == PERFLIBS_SPARSE_OPERATION_NOTRANS) {
    return data[i * cols + j];
  }
  return data[j * cols + i];
}

static inline double test_access_dense_d(const double *data,
                                         perflibs_int_t cols,
                                         enum perflibs_sparse_hint_value trans,
                                         perflibs_int_t i, perflibs_int_t j) {
  if (trans == PERFLIBS_SPARSE_OPERATION_NOTRANS) {
    return data[i * cols + j];
  }
  return data[j * cols + i];
}

static inline perflibs_singlecomplex_t
test_access_dense_c(const perflibs_singlecomplex_t *data, perflibs_int_t cols,
                    enum perflibs_sparse_hint_value trans, perflibs_int_t i,
                    perflibs_int_t j) {
  if (trans == PERFLIBS_SPARSE_OPERATION_NOTRANS) {
    return data[i * cols + j];
  }
  return data[j * cols + i];
}

static inline perflibs_doublecomplex_t
test_access_dense_z(const perflibs_doublecomplex_t *data, perflibs_int_t cols,
                    enum perflibs_sparse_hint_value trans, perflibs_int_t i,
                    perflibs_int_t j) {
  if (trans == PERFLIBS_SPARSE_OPERATION_NOTRANS) {
    return data[i * cols + j];
  }
  return data[j * cols + i];
}

static inline void test_csr_to_dense_s(perflibs_int_t rows, perflibs_int_t cols,
                                       const perflibs_int_t *row_ptr,
                                       const perflibs_int_t *col_indx,
                                       const float *vals,
                                       perflibs_int_t index_base, float *out) {
  const perflibs_int_t total = rows * cols;
  for (perflibs_int_t idx = 0; idx < total; ++idx) {
    out[idx] = 0.0f;
  }
  for (perflibs_int_t row = 0; row < rows; ++row) {
    const perflibs_int_t start = row_ptr[row] - index_base;
    const perflibs_int_t end = row_ptr[row + 1] - index_base;
    for (perflibs_int_t nz = start; nz < end; ++nz) {
      const perflibs_int_t col = col_indx[nz] - index_base;
      out[row * cols + col] = vals[nz];
    }
  }
}

static inline void test_csr_to_dense_d(perflibs_int_t rows, perflibs_int_t cols,
                                       const perflibs_int_t *row_ptr,
                                       const perflibs_int_t *col_indx,
                                       const double *vals,
                                       perflibs_int_t index_base, double *out) {
  const perflibs_int_t total = rows * cols;
  for (perflibs_int_t idx = 0; idx < total; ++idx) {
    out[idx] = 0.0;
  }
  for (perflibs_int_t row = 0; row < rows; ++row) {
    const perflibs_int_t start = row_ptr[row] - index_base;
    const perflibs_int_t end = row_ptr[row + 1] - index_base;
    for (perflibs_int_t nz = start; nz < end; ++nz) {
      const perflibs_int_t col = col_indx[nz] - index_base;
      out[row * cols + col] = vals[nz];
    }
  }
}

static inline void test_csr_to_dense_c(perflibs_int_t rows, perflibs_int_t cols,
                                       const perflibs_int_t *row_ptr,
                                       const perflibs_int_t *col_indx,
                                       const perflibs_singlecomplex_t *vals,
                                       perflibs_int_t index_base,
                                       perflibs_singlecomplex_t *out) {
  const perflibs_int_t total = rows * cols;
  memset(out, 0, (size_t)total * sizeof(*out));
  for (perflibs_int_t row = 0; row < rows; ++row) {
    const perflibs_int_t start = row_ptr[row] - index_base;
    const perflibs_int_t end = row_ptr[row + 1] - index_base;
    for (perflibs_int_t nz = start; nz < end; ++nz) {
      const perflibs_int_t col = col_indx[nz] - index_base;
      out[row * cols + col] = vals[nz];
    }
  }
}

static inline void test_csr_to_dense_z(perflibs_int_t rows, perflibs_int_t cols,
                                       const perflibs_int_t *row_ptr,
                                       const perflibs_int_t *col_indx,
                                       const perflibs_doublecomplex_t *vals,
                                       perflibs_int_t index_base,
                                       perflibs_doublecomplex_t *out) {
  const perflibs_int_t total = rows * cols;
  memset(out, 0, (size_t)total * sizeof(*out));
  for (perflibs_int_t row = 0; row < rows; ++row) {
    const perflibs_int_t start = row_ptr[row] - index_base;
    const perflibs_int_t end = row_ptr[row + 1] - index_base;
    for (perflibs_int_t nz = start; nz < end; ++nz) {
      const perflibs_int_t col = col_indx[nz] - index_base;
      out[row * cols + col] = vals[nz];
    }
  }
}

static inline int check_int_array_impl(const char *file, int line,
                                       const perflibs_int_t *got,
                                       const perflibs_int_t *expected,
                                       perflibs_int_t n) {
  for (perflibs_int_t i = 0; i < n; ++i) {
    if (got[i] != expected[i]) {
      fprintf(stderr,
              "CHECK_INT_ARRAY failed at %s:%d index %lld: got %lld expected "
              "%lld\n",
              file, line, test_i64(i), test_i64(got[i]), test_i64(expected[i]));
      return EXIT_FAILURE;
    }
  }
  return EXIT_SUCCESS;
}

#define CHECK_INT_ARRAY(got, expected, n)                                      \
  do {                                                                         \
    if (check_int_array_impl(__FILE__, __LINE__, (got), (expected),            \
                             (perflibs_int_t)(n)) != EXIT_SUCCESS) {           \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  } while (0)

static inline int check_double_array_impl(const char *file, int line,
                                          const double *got,
                                          const double *expected,
                                          perflibs_int_t n, double tol) {
  for (perflibs_int_t i = 0; i < n; ++i) {
    if (fabs(got[i] - expected[i]) > tol) {
      fprintf(stderr,
              "CHECK_DOUBLE_ARRAY failed at %s:%d index %d: got %.17g "
              "expected %.17g (tol %.3g)\n",
              file, line, (int)i, got[i], expected[i], tol);
      return EXIT_FAILURE;
    }
  }
  return EXIT_SUCCESS;
}

#define CHECK_DOUBLE_ARRAY(got, expected, n, tol)                              \
  do {                                                                         \
    if (check_double_array_impl(__FILE__, __LINE__, (got), (expected),         \
                                (perflibs_int_t)(n),                           \
                                (double)(tol)) != EXIT_SUCCESS) {              \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  } while (0)

#endif
