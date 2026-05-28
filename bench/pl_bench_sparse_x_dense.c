/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

/*
 * To compile on Linux with gcc:
 *
 * Using a build of perf-libs-sparse installed in <install_dir>:
 *   # gcc -fopenmp -DUSE_PERFLIBS_NAMES -O3 -std=c99 -I \
 *         <install_dir>/include/plsparse_lp64 \
 *         pl_bench_sparse_x_dense.c \
 *         <install_dir>/lib/libplsparse_lp64_mp.a \
 *         -o pl_bench_sparse_x_dense \
 *         $ARMPL_DIR/lib/libarmpl_mp.a -lm -lstdc++
 * Note: using ArmPL here to satisfy the CBLAS dependency of
 * perf-libs-sparse
 *
 * Using ArmPL:
 *   # gcc -fopenmp -O3 -std=c99 -I $ARMPL_DIR/include \
 *         pl_bench_sparse_x_dense.c \
 *         $ARMPL_DIR/lib/libplsparse_lp64_mp.a \
 *         -o pl_bench_sparse_x_dense \
 *         $ARMPL_DIR/lib/libarmpl_mp.a -lm
 */

#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

/* Build this example either against ArmPL or directly against
 * perf-libs-sparse by remapping the small subset of API names used here.
 */
#ifdef USE_PERFLIBS_NAMES
#include "perflibs_sparse.h"
#define armpl_int_t perflibs_int_t
#define armpl_spmat_t perflibs_spmat_t
#define armpl_status_t perflibs_status_t
#define ARMPL_STATUS_SUCCESS PERFLIBS_STATUS_SUCCESS
#define ARMPL_ROW_MAJOR PERFLIBS_ROW_MAJOR
#define ARMPL_SPARSE_OPERATION_NOTRANS PERFLIBS_SPARSE_OPERATION_NOTRANS
#define ARMPL_SPARSE_SCALAR_ONE PERFLIBS_SPARSE_SCALAR_ONE
#define ARMPL_SPARSE_SCALAR_ZERO PERFLIBS_SPARSE_SCALAR_ZERO
#define armpl_spmat_create_csr_d perflibs_spmat_create_csr_d
#define armpl_spmat_create_dense_d perflibs_spmat_create_dense_d
#define armpl_spmat_create_null perflibs_spmat_create_null
#define armpl_spmat_destroy perflibs_spmat_destroy
#define armpl_spmm_optimize perflibs_spmm_optimize
#define armpl_spmm_exec_d perflibs_spmm_exec_d
#else
#include "armpl.h"
#endif

struct entry {
  armpl_int_t r, c;
  double v;
};

struct csr_matrix {
  armpl_int_t m, n;
  size_t nnz;
  armpl_int_t *row_ptr, *col_indx;
  double *vals;
};

static int cmp_entry(const void *pa, const void *pb) {
  const struct entry *a = pa, *b = pb;
  if (a->r != b->r)
    return (a->r < b->r) ? -1 : 1;
  return (a->c != b->c) ? ((a->c < b->c) ? -1 : 1) : 0;
}

static void fail(const char *msg) {
  fprintf(stderr, "error: %s\n", msg);
  exit(1);
}

static void require(int ok, const char *msg) {
  if (!ok)
    fail(msg);
}

static void check(armpl_status_t s, const char *msg) {
  if (s != ARMPL_STATUS_SUCCESS)
    fail(msg);
}

/* Use a platform monotonic timer: QueryPerformanceCounter on Windows and
 * clock_gettime(CLOCK_MONOTONIC) elsewhere.
 */
static double now_seconds(void) {
#ifdef _WIN32
  static LARGE_INTEGER freq;
  static int initialized = 0;
  LARGE_INTEGER counter;

  if (!initialized) {
    require(QueryPerformanceFrequency(&freq) != 0,
            "QueryPerformanceFrequency failed");
    require(freq.QuadPart > 0, "QueryPerformanceFrequency returned zero");
    initialized = 1;
  }

  require(QueryPerformanceCounter(&counter) != 0,
          "QueryPerformanceCounter failed");
  return (double)counter.QuadPart / (double)freq.QuadPart;
#else
  struct timespec t;
  require(clock_gettime(CLOCK_MONOTONIC, &t) == 0, "clock_gettime failed");
  return (double)t.tv_sec + 1.0e-9 * (double)t.tv_nsec;
#endif
}

static void read_matrix_market_csr(const char *path, struct csr_matrix *csr) {
  FILE *f = fopen(path, "r");
  char line[512], field[64], symmetry[64];
  long long m_ll = 0, n_ll = 0, nnz_ll = 0;
  size_t used = 0, pos;
  struct entry *entries;
  armpl_int_t *next;
  size_t m_sz, cap;

  require(f != NULL, "failed to open .mtx file");
  require(fgets(line, sizeof(line), f) != NULL,
          "failed to read Matrix Market header");
  require(sscanf(line, "%%%%MatrixMarket matrix coordinate %63s %63s", field,
                 symmetry) == 2,
          "expected MatrixMarket coordinate file");
  require(strcmp(field, "complex") != 0,
          "complex Matrix Market files not supported");

  do {
    require(fgets(line, sizeof(line), f) != NULL, "missing size line");
  } while (line[0] == '%');

  require(sscanf(line, "%lld %lld %lld", &m_ll, &n_ll, &nnz_ll) == 3,
          "bad size line");
  require(m_ll > 0 && n_ll > 0 && nnz_ll >= 0, "invalid matrix dimensions");

  csr->m = (armpl_int_t)m_ll;
  csr->n = (armpl_int_t)n_ll;
  m_sz = (size_t)csr->m;
  cap = (size_t)nnz_ll * (strcmp(symmetry, "general") ? 2u : 1u);

  entries = malloc(cap * sizeof(*entries));
  require(entries != NULL, "entries allocation failed");

  /* This compact example assumes a well-formed Matrix Market file and
   * expands symmetric-style inputs into a plain CSR matrix.
   */
  for (long long i = 0; i < nnz_ll; ++i) {
    long long r, c;
    armpl_int_t rr, cc;
    double v = 1.0;

    if (!strcmp(field, "pattern")) {
      require(fscanf(f, "%lld %lld", &r, &c) == 2, "bad pattern entry");
    } else {
      require(fscanf(f, "%lld %lld %lf", &r, &c, &v) == 3, "bad numeric entry");
    }

    require(r >= 1 && r <= m_ll && c >= 1 && c <= n_ll,
            "entry index out of range");

    --r;
    --c;
    rr = (armpl_int_t)r;
    cc = (armpl_int_t)c;

    entries[used].r = rr;
    entries[used].c = cc;
    entries[used].v = v;
    ++used;

    if (!strcmp(symmetry, "symmetric") && r != c) {
      entries[used].r = cc;
      entries[used].c = rr;
      entries[used].v = v;
      ++used;
    } else if (!strcmp(symmetry, "skew-symmetric") && r != c) {
      entries[used].r = cc;
      entries[used].c = rr;
      entries[used].v = -v;
      ++used;
    } else if (!strcmp(symmetry, "hermitian") && r != c) {
      entries[used].r = cc;
      entries[used].c = rr;
      entries[used].v = v;
      ++used;
    }
  }
  fclose(f);

  qsort(entries, used, sizeof(*entries), cmp_entry);

  csr->row_ptr = calloc(m_sz + 1, sizeof(*csr->row_ptr));
  csr->col_indx = malloc(used * sizeof(*csr->col_indx));
  csr->vals = malloc(used * sizeof(*csr->vals));
  next = calloc(m_sz, sizeof(*next));

  require(csr->row_ptr && csr->col_indx && csr->vals && next,
          "CSR allocation failed");

  for (pos = 0; pos < used; ++pos) {
    require(entries[pos].r >= 0 && entries[pos].r < csr->m,
            "row index out of range");
    csr->row_ptr[entries[pos].r + 1]++;
  }

  for (pos = 1; pos <= m_sz; ++pos)
    csr->row_ptr[pos] += csr->row_ptr[pos - 1];

  memcpy(next, csr->row_ptr, m_sz * sizeof(*next));

  for (pos = 0; pos < used; ++pos) {
    armpl_int_t p = next[entries[pos].r]++;

    require(p >= 0 && (size_t)p < used, "CSR write index out of range");
    csr->col_indx[p] = entries[pos].c;
    csr->vals[p] = entries[pos].v;
  }

  free(next);
  free(entries);
  csr->nnz = used;
}

int main(int argc, char **argv) {
  const int iters = 10;
  const char *path = (argc == 2) ? argv[1] : NULL;
  struct csr_matrix csr = {0, 0, 0, NULL, NULL, NULL};
  armpl_spmat_t A = NULL, B = NULL, C = NULL;
  armpl_int_t cols = 1024;
  double t0, t1, sec;
  double *vals_B;
  size_t pos, b_len;

  require(path != NULL, "usage: pl_bench_sparse_x_dense matrix.mtx");

  read_matrix_market_csr(path, &csr);

  {
    const uint64_t max_len =
        (uint64_t)sqrt((long double)(SIZE_MAX / sizeof(double)));

    while ((uint64_t)csr.n * (uint64_t)cols > max_len && cols > 1)
      cols = cols / 8 > 0 ? cols / 8 : 1;
  }

  b_len = (size_t)csr.n * (size_t)cols;
  vals_B = malloc(b_len * sizeof(*vals_B));
  require(vals_B != NULL, "dense B allocation failed");

  srand(1);
  for (pos = 0; pos < b_len; ++pos)
    vals_B[pos] = 2.0 * rand() / (double)RAND_MAX - 1.0;

  check(armpl_spmat_create_csr_d(&A, csr.m, csr.n, csr.row_ptr, csr.col_indx,
                                 csr.vals, 0),
        "create A failed");

  check(armpl_spmat_create_dense_d(&B, ARMPL_ROW_MAJOR, csr.n, cols, cols,
                                   vals_B, 0),
        "create B failed");

  C = armpl_spmat_create_null(csr.m, cols);
  require(C != NULL, "create C failed");

  check(armpl_spmm_optimize(
            ARMPL_SPARSE_OPERATION_NOTRANS, ARMPL_SPARSE_OPERATION_NOTRANS,
            ARMPL_SPARSE_SCALAR_ONE, A, B, ARMPL_SPARSE_SCALAR_ZERO, C),
        "optimize failed");

  t0 = now_seconds();

  for (int i = 0; i < iters; ++i) {
    check(armpl_spmm_exec_d(ARMPL_SPARSE_OPERATION_NOTRANS,
                            ARMPL_SPARSE_OPERATION_NOTRANS, 1.0, A, B, 0.0, C),
          "exec failed");
  }

  t1 = now_seconds();

  sec = t1 - t0;

  printf("A: %lld x %lld nnz=%zu\n", (long long)csr.m, (long long)csr.n,
         csr.nnz);
  printf("B: %lld x %lld\n", (long long)csr.n, (long long)cols);
  printf("C: %lld x %lld\n", (long long)csr.m, (long long)cols);
  printf("iters=%d total_s=%.6f avg_s=%.6f\n", iters, sec, sec / iters);

  check(armpl_spmat_destroy(C), "destroy C failed");
  check(armpl_spmat_destroy(B), "destroy B failed");
  check(armpl_spmat_destroy(A), "destroy A failed");

  free(vals_B);
  free(csr.vals);
  free(csr.col_indx);
  free(csr.row_ptr);
  return 0;
}
