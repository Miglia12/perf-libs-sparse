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
 *         pl_bench_spsm.c \
 *         <install_dir>/lib/libplsparse_lp64_mp.a \
 *         -o pl_bench_spsm \
 *         $ARMPL_DIR/lib/libarmpl_mp.a -lm -lstdc++
 * Note: using ArmPL here to satisfy the CBLAS dependency of
 * perf-libs-sparse
 *
 * Using ArmPL:
 *   # gcc -fopenmp -O3 -std=c99 -I $ARMPL_DIR/include \
 *         pl_bench_spsm.c \
 *         $ARMPL_DIR/lib/libplsparse_lp64_mp.a \
 *         -o pl_bench_spsm \
 *         $ARMPL_DIR/lib/libarmpl_mp.a -lm
 *
 * Run as:
 *   ./pl_bench_spsm matrix.mtx [lower|upper] [nrhs] [iters]
 *
 * Arguments:
 *   matrix.mtx    MatrixMarket coordinate file for a square matrix with an
 *                 explicit non-zero diagonal.
 *   lower|upper   Optional triangle to benchmark. For general MatrixMarket
 *                 input, only the requested triangle is kept. For symmetric
 *                 or hermitian input, the requested triangle is formed from
 *                 the stored half without expanding to full storage. If
 *                 omitted, the input must already be triangular and the
 *                 benchmark infers lower or upper from the entries.
 *   nrhs          Optional number of dense RHS columns. Default: 1024, reduced
 *                 automatically if omitted and too large for allocation.
 *   iters         Optional iteration count. Default: 10.
 */

#define _POSIX_C_SOURCE 200809L
#include <limits.h>
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
#define ARMPL_SPARSE_HINT_STRUCTURE PERFLIBS_SPARSE_HINT_STRUCTURE
#define ARMPL_SPARSE_STRUCTURE_TRIANGULAR PERFLIBS_SPARSE_STRUCTURE_TRIANGULAR
#define ARMPL_SPARSE_OPERATION_NOTRANS PERFLIBS_SPARSE_OPERATION_NOTRANS
#define ARMPL_SPARSE_SCALAR_ONE PERFLIBS_SPARSE_SCALAR_ONE
#define armpl_spmat_create_csr_d perflibs_spmat_create_csr_d
#define armpl_spmat_create_dense_d perflibs_spmat_create_dense_d
#define armpl_spmat_destroy perflibs_spmat_destroy
#define armpl_spmat_export_dense_d perflibs_spmat_export_dense_d
#define armpl_spmat_hint perflibs_spmat_hint
#define armpl_spsm_optimize perflibs_spsm_optimize
#define armpl_spsm_exec_d perflibs_spsm_exec_d
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

enum triangle_kind { TRIANGLE_LOWER, TRIANGLE_UPPER };

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

static int is_triangle_arg(const char *arg) {
  return arg != NULL && (!strcmp(arg, "lower") || !strcmp(arg, "upper"));
}

static enum triangle_kind parse_triangle_arg(const char *arg) {
  require(is_triangle_arg(arg), "triangle must be 'lower' or 'upper'");
  return !strcmp(arg, "upper") ? TRIANGLE_UPPER : TRIANGLE_LOWER;
}

static int entry_in_triangle(enum triangle_kind triangle, armpl_int_t r,
                             armpl_int_t c) {
  return triangle == TRIANGLE_LOWER ? r >= c : r <= c;
}

static long long parse_positive_ll(const char *arg, const char *what) {
  char *end = NULL;
  long long value;

  require(arg != NULL && *arg != '\0', what);
  value = strtoll(arg, &end, 10);
  require(end != arg && end != NULL && *end == '\0', what);
  require(value > 0, what);
  return value;
}

static long long max_armpl_int_value(void) {
  return sizeof(armpl_int_t) == sizeof(int64_t) ? (long long)INT64_MAX
                                                : (long long)INT32_MAX;
}

static void read_matrix_market_csr(const char *path, struct csr_matrix *csr,
                                   int have_triangle,
                                   enum triangle_kind triangle) {
  FILE *f = fopen(path, "r");
  char line[512], field[64], symmetry[64];
  long long m_ll = 0, n_ll = 0, nnz_ll = 0;
  size_t used = 0, pos;
  struct entry *entries;
  armpl_int_t *next;
  size_t m_sz, cap;
  int symmetric_storage;

  require(f != NULL, "failed to open .mtx file");
  require(fgets(line, sizeof(line), f) != NULL,
          "failed to read Matrix Market header");
  require(sscanf(line, "%%%%MatrixMarket matrix coordinate %63s %63s", field,
                 symmetry) == 2,
          "expected MatrixMarket coordinate file");
  require(strcmp(field, "complex") != 0,
          "complex Matrix Market files not supported");
  require(!strcmp(symmetry, "general") || !strcmp(symmetry, "symmetric") ||
              !strcmp(symmetry, "skew-symmetric") ||
              !strcmp(symmetry, "hermitian"),
          "unsupported Matrix Market symmetry");
  require(strcmp(symmetry, "skew-symmetric") != 0,
          "skew-symmetric Matrix Market files are not supported for SpSM "
          "because they have zero diagonal entries");

  symmetric_storage = strcmp(symmetry, "general") != 0;
  require(!symmetric_storage || have_triangle,
          "symmetric Matrix Market files require a lower/upper argument");

  do {
    require(fgets(line, sizeof(line), f) != NULL, "missing size line");
  } while (line[0] == '%');

  require(sscanf(line, "%lld %lld %lld", &m_ll, &n_ll, &nnz_ll) == 3,
          "bad size line");
  require(m_ll > 0 && n_ll > 0 && nnz_ll > 0, "invalid matrix dimensions");
  require(m_ll <= max_armpl_int_value() && n_ll <= max_armpl_int_value(),
          "matrix dimensions exceed armpl_int_t range");
  require((unsigned long long)nnz_ll <=
              (unsigned long long)(SIZE_MAX / sizeof(*entries)),
          "nnz is too large");

  csr->m = (armpl_int_t)m_ll;
  csr->n = (armpl_int_t)n_ll;
  m_sz = (size_t)csr->m;
  cap = (size_t)nnz_ll;

  entries = malloc(cap * sizeof(*entries));
  require(entries != NULL, "entries allocation failed");

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

    if (symmetric_storage) {
      if (rr != cc && !entry_in_triangle(triangle, rr, cc)) {
        const armpl_int_t tmp = rr;
        rr = cc;
        cc = tmp;
      }
    } else if (have_triangle && !entry_in_triangle(triangle, rr, cc)) {
      continue;
    }

    entries[used].r = rr;
    entries[used].c = cc;
    entries[used].v = v;
    ++used;
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

static enum triangle_kind analyze_triangular_csr(const struct csr_matrix *csr) {
  int has_lower = 0;
  int has_upper = 0;

  require(csr->m == csr->n, "SpSM benchmark requires a square matrix");

  for (armpl_int_t i = 0; i < csr->m; ++i) {
    int has_diag = 0;
    for (armpl_int_t p = csr->row_ptr[i]; p < csr->row_ptr[i + 1]; ++p) {
      const armpl_int_t j = csr->col_indx[p];
      if (j < i) {
        has_lower = 1;
      } else if (j > i) {
        has_upper = 1;
      } else if (csr->vals[p] != 0.0) {
        has_diag = 1;
      }
    }
    require(has_diag, "matrix must have an explicit non-zero diagonal");
  }

  require(!(has_lower && has_upper),
          "matrix must be triangular (strictly lower or strictly upper)");
  return has_upper ? TRIANGLE_UPPER : TRIANGLE_LOWER;
}

static void validate_triangular_csr(const struct csr_matrix *csr,
                                    enum triangle_kind triangle) {
  require(csr->m == csr->n, "SpSM benchmark requires a square matrix");

  for (armpl_int_t i = 0; i < csr->m; ++i) {
    int has_diag = 0;
    for (armpl_int_t p = csr->row_ptr[i]; p < csr->row_ptr[i + 1]; ++p) {
      const armpl_int_t j = csr->col_indx[p];

      require(entry_in_triangle(triangle, i, j),
              "matrix contains entries outside the requested triangle");
      if (j == i && csr->vals[p] != 0.0)
        has_diag = 1;
    }
    require(has_diag, "matrix must have an explicit non-zero diagonal");
  }
}

static double csr_inf_norm(const struct csr_matrix *csr) {
  double max_row_sum = 0.0;

  for (armpl_int_t i = 0; i < csr->m; ++i) {
    double row_sum = 0.0;
    for (armpl_int_t p = csr->row_ptr[i]; p < csr->row_ptr[i + 1]; ++p)
      row_sum += fabs(csr->vals[p]);
    if (row_sum > max_row_sum)
      max_row_sum = row_sum;
  }

  return max_row_sum;
}

static void report_residuals(const struct csr_matrix *csr, const double *X,
                             const double *Y, armpl_int_t nrhs, double alpha) {
  double max_abs_resid = 0.0;
  double max_rel_resid = 0.0;
  double sum_abs_resid = 0.0;
  double sum_rel_resid = 0.0;
  const double a_inf = csr_inf_norm(csr);

  printf("Residuals per RHS (inf-norm):\n");
  for (armpl_int_t rhs = 0; rhs < nrhs; ++rhs) {
    double x_inf = 0.0;
    double y_inf = 0.0;
    double resid_inf = 0.0;
    double rel_resid;

    for (armpl_int_t i = 0; i < csr->m; ++i) {
      double ax_i = 0.0;
      const double y_i = alpha * Y[(size_t)i * (size_t)nrhs + (size_t)rhs];
      const double x_i = X[(size_t)i * (size_t)nrhs + (size_t)rhs];

      if (fabs(y_i) > y_inf)
        y_inf = fabs(y_i);
      if (fabs(x_i) > x_inf)
        x_inf = fabs(x_i);

      for (armpl_int_t p = csr->row_ptr[i]; p < csr->row_ptr[i + 1]; ++p) {
        const armpl_int_t j = csr->col_indx[p];
        ax_i += csr->vals[p] * X[(size_t)j * (size_t)nrhs + (size_t)rhs];
      }

      if (fabs(ax_i - y_i) > resid_inf)
        resid_inf = fabs(ax_i - y_i);
    }

    {
      const double denom = a_inf * x_inf + y_inf;
      rel_resid = denom > 0.0 ? resid_inf / denom : resid_inf;
    }

    if (resid_inf > max_abs_resid)
      max_abs_resid = resid_inf;
    if (rel_resid > max_rel_resid)
      max_rel_resid = rel_resid;
    sum_abs_resid += resid_inf;
    sum_rel_resid += rel_resid;

    // abs_inf = ||A x_k - alpha y_k||_inf
    // rel_inf = abs_inf / (||A||_inf ||x_k||_inf + ||alpha y_k||_inf)
    printf("  rhs[%lld]: abs_inf=%.3e rel_inf=%.3e\n", (long long)rhs,
           resid_inf, rel_resid);
  }

  printf("Residual summary: max_abs_inf=%.3e avg_abs_inf=%.3e "
         "max_rel_inf=%.3e avg_rel_inf=%.3e\n",
         max_abs_resid, sum_abs_resid / (double)nrhs, max_rel_resid,
         sum_rel_resid / (double)nrhs);
}

int main(int argc, char **argv) {
  const int default_iters = 10;
  const armpl_int_t default_nrhs = 1024;
  const char *path = (argc >= 2) ? argv[1] : NULL;
  struct csr_matrix csr = {0, 0, 0, NULL, NULL, NULL};
  enum triangle_kind triangle = TRIANGLE_LOWER;
  armpl_spmat_t A = NULL, X = NULL, Y = NULL;
  armpl_int_t nrhs = default_nrhs;
  int iters = default_iters;
  double t0, t1, sec;
  double *vals_X = NULL, *vals_Y = NULL, *vals_X_out = NULL;
  armpl_int_t x_out_m = 0, x_out_n = 0;
  size_t pos, dense_len;
  uint64_t max_cols;
  int have_triangle = 0;
  int next_arg = 2;
  int user_nrhs;
  int user_iters;

  require(path != NULL,
          "usage: pl_bench_spsm matrix.mtx [lower|upper] [nrhs] [iters]");

  if (argc >= 3 && is_triangle_arg(argv[2])) {
    triangle = parse_triangle_arg(argv[2]);
    have_triangle = 1;
    next_arg = 3;
  }

  require(argc <= next_arg + 2,
          "usage: pl_bench_spsm matrix.mtx [lower|upper] [nrhs] [iters]");
  user_nrhs = argc >= next_arg + 1;
  user_iters = argc >= next_arg + 2;

  if (user_nrhs) {
    const long long parsed_nrhs =
        parse_positive_ll(argv[next_arg], "nrhs must be a positive integer");
    require(parsed_nrhs <= max_armpl_int_value(), "nrhs is too large");
    nrhs = (armpl_int_t)parsed_nrhs;
  }

  if (user_iters) {
    const long long parsed_iters = parse_positive_ll(
        argv[next_arg + 1], "iters must be a positive integer");
    require(parsed_iters <= (long long)INT_MAX, "iters is too large");
    iters = (int)parsed_iters;
  }

  read_matrix_market_csr(path, &csr, have_triangle, triangle);
  if (have_triangle) {
    validate_triangular_csr(&csr, triangle);
  } else {
    triangle = analyze_triangular_csr(&csr);
  }

  require((uint64_t)csr.n > 0, "invalid matrix size");
  max_cols = (uint64_t)SIZE_MAX / sizeof(double) / (uint64_t)csr.n / 2u;
  require(max_cols > 0, "matrix is too large for dense RHS allocation");

  if (!user_nrhs) {
    while ((uint64_t)nrhs > max_cols && nrhs > 1) {
      nrhs = nrhs / 8 > 0 ? nrhs / 8 : 1;
    }
  } else {
    require((uint64_t)nrhs <= max_cols, "requested nrhs is too large");
  }

  dense_len = (size_t)csr.n * (size_t)nrhs;
  vals_X = calloc(dense_len, sizeof(*vals_X));
  vals_Y = malloc(dense_len * sizeof(*vals_Y));
  require(vals_X != NULL && vals_Y != NULL, "dense RHS allocation failed");

  srand(1);
  for (pos = 0; pos < dense_len; ++pos)
    vals_Y[pos] = 2.0 * rand() / (double)RAND_MAX - 1.0;

  check(armpl_spmat_create_csr_d(&A, csr.m, csr.n, csr.row_ptr, csr.col_indx,
                                 csr.vals, 0),
        "create A failed");
  check(armpl_spmat_hint(A, ARMPL_SPARSE_HINT_STRUCTURE,
                         ARMPL_SPARSE_STRUCTURE_TRIANGULAR),
        "set A triangular hint failed");

  check(armpl_spmat_create_dense_d(&X, ARMPL_ROW_MAJOR, csr.n, nrhs, nrhs,
                                   vals_X, 0),
        "create X failed");
  check(armpl_spmat_create_dense_d(&Y, ARMPL_ROW_MAJOR, csr.n, nrhs, nrhs,
                                   vals_Y, 0),
        "create Y failed");

  check(armpl_spsm_optimize(ARMPL_SPARSE_OPERATION_NOTRANS, A, X,
                            ARMPL_SPARSE_SCALAR_ONE, Y),
        "optimize failed");

  t0 = now_seconds();

  for (int i = 0; i < iters; ++i) {
    check(armpl_spsm_exec_d(ARMPL_SPARSE_OPERATION_NOTRANS, A, X, 1.0, Y),
          "exec failed");
  }

  t1 = now_seconds();

  sec = t1 - t0;

  printf("A: %lld x %lld nnz=%zu triangular=%s\n", (long long)csr.m,
         (long long)csr.n, csr.nnz,
         triangle == TRIANGLE_UPPER ? "upper" : "lower");
  printf("Y: %lld x %lld (row-major random RHS)\n", (long long)csr.n,
         (long long)nrhs);
  printf("X: %lld x %lld (row-major output)\n", (long long)csr.n,
         (long long)nrhs);
  printf("iters=%d total_s=%.6f avg_s=%.6f\n", iters, sec, sec / iters);
  check(armpl_spmat_export_dense_d(X, ARMPL_ROW_MAJOR, &x_out_m, &x_out_n,
                                   &vals_X_out),
        "export X failed");
  require(x_out_m == csr.n && x_out_n == nrhs, "exported X has wrong shape");
  report_residuals(&csr, vals_X_out, vals_Y, nrhs, 1.0);

  check(armpl_spmat_destroy(Y), "destroy Y failed");
  check(armpl_spmat_destroy(X), "destroy X failed");
  check(armpl_spmat_destroy(A), "destroy A failed");

  free(vals_X_out);
  free(vals_Y);
  free(vals_X);
  free(csr.vals);
  free(csr.col_indx);
  free(csr.row_ptr);
  return 0;
}
