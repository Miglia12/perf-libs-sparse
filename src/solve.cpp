/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "solve.hpp"
#include "block_sparse_rows.hpp"
#include "compressed_sparse_columns.hpp"
#include "compressed_sparse_rows.hpp"
#include "convert.hpp"
#include "dense.hpp"
#include "matrix_state.hpp"
#include "matvec.hpp"
#include "pod_vector.hpp"
#include "supernodal.hpp"
#include "timer.hpp"
#include "types.hpp"

#include "cblas_wrappers.hpp"
#include "statistics.hpp"

#include <cstring>
#include <inttypes.h>

namespace perflibs::sparse {

template <typename T>
perflibs_status_t call_spsv(perflibs_sparse_hint_value trans,
                            perflibs_spmat_impl_t<T> *impl, T *x, T alpha,
                            const T *y) {

  // Early return for alpha == 0.
  if (alpha == T(0)) {
    std::memset(reinterpret_cast<void *>(x), 0, sizeof(T) * (impl->n));
    return PERFLIBS_STATUS_SUCCESS;
  }

  // transpose is passed into the exec call
  auto i_trans = (sparse_hint_value_internal)trans;

  // uplo and diag are discovered on matrix creation, and are treated as
  // properties of the matrix alongside m, n, nnz.
  auto i_uplo = (sparse_hint_value_internal)impl->shape;
  auto i_diag = (sparse_hint_value_internal)impl->diag;

  if (impl->spmat_format == perflibs_format_csr) {
    spsv_csr<T>(impl->csr, i_trans, i_uplo, i_diag, x, y, alpha);
  } else if (impl->spmat_format == perflibs_format_csc) {
    spsv_csc<T>(impl->csc, i_trans, i_uplo, i_diag, x, y, alpha);
  } else if (impl->spmat_format == perflibs_format_coo ||
             impl->spmat_format == perflibs_format_scs) {
    convert(perflibs_format_csr, impl);
    spsv_csr<T>(impl->csr, i_trans, i_uplo, i_diag, x, y, alpha);
  } else if (impl->spmat_format == perflibs_format_bsr) {
    spsv_bsr<T>(impl->bsr, i_trans, i_uplo, i_diag, x, y, alpha);
  } else if (impl->spmat_format == perflibs_format_supernodal) {
    spsv_supernodal<T>(impl->supernodal, trans, x, y, alpha);
  } else if (impl->spmat_format == perflibs_format_dense) {
    // Because dense trsv does not do scaling of y by alpha,
    // we scale y by alpha first and copy into x using axpby.
    // TODO: We could use out-of-place scale instead, but gescal_out_of_place
    // does not yet support z and c datatypes.
    const perflibs_int_t inc = 1;
    const T zero = T(0);
    cblas_axpby<T>(impl->n, alpha, y, inc, zero, x, inc);
    spsv_trsv<T>(impl->dense.layout, i_trans, i_uplo, i_diag, impl->n,
                 impl->dense.vals_ptr, impl->dense.lda, x);
  } else if (impl->spmat_format == perflibs_format_identity) {
    const perflibs_int_t inc = 1;
    const T zero = T(0);
    cblas_axpby<T>(impl->n, alpha, y, inc, zero, x, inc);
  }
  return PERFLIBS_STATUS_SUCCESS;
};

template <typename T>
perflibs_status_t spsv_exec_impl(perflibs_sparse_hint_value trans,
                                 perflibs_spmat_impl_t<T> *impl, T *x, T alpha,
                                 const T *y) {

  // Early return for null matrix format
  if (impl->spmat_format == perflibs_format_null) {
    impl->error_handle.perflibs_error_type =
        PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    impl->error_handle.perflibs_error_code = 2;
    impl->error_handle.err_msg =
        "It is not possible to perform a triangular solve using a null matrix.";
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  } else if (impl->spmat_format != perflibs_format_identity) {
    // Handle incorrect hint passed in for first arg
    if (trans != PERFLIBS_SPARSE_OPERATION_NOTRANS &&
        trans != PERFLIBS_SPARSE_OPERATION_TRANS &&
        trans != PERFLIBS_SPARSE_OPERATION_CONJTRANS) {
      impl->error_handle.perflibs_error_type =
          PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
      impl->error_handle.perflibs_error_code = 1;
      impl->error_handle.err_msg = "Incorrect transpose option.";
      return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    } else if (impl->shape == PERFLIBS_SPARSE_SHAPE_RECTANGULAR ||
               impl->diag == PERFLIBS_SPARSE_DIAG_ZERO) {
      impl->error_handle.perflibs_error_type =
          PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
      impl->error_handle.perflibs_error_code = 2;

      if (impl->shape == PERFLIBS_SPARSE_SHAPE_RECTANGULAR) {
        impl->error_handle.err_msg =
            "The matrix passed to perflibs_spsv_exec is not triangular. The "
            "matrix used to perform a "
            "triangular solve must be either upper or lower triangular. Upper "
            "triangular matrices "
            "must have no non-zero values below the diagonal, and lower "
            "triangular matrices must "
            "have no non-zero values above the diagonal.";
      } else {
        impl->error_handle.err_msg =
            "The matrix passed to perflibs_spsv_exec contains at least one "
            "zero "
            "on the diagonal. The "
            "matrix used to perform a triangular solve must contain non-zero "
            "values in every "
            "position along the diagonal.";
      }

      return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    }
  }
  return perflibs::sparse::call_spsv(trans, impl, x, alpha, y);
};

template <typename T>
perflibs_status_t spsv_exec(perflibs_sparse_hint_value trans,
                            perflibs_spmat_top_t *A, T *x, T alpha,
                            const T *y) {

  auto impl = reinterpret_cast<perflibs_spmat_impl_t<T> *>(A->impl);

  // Fall back to CSR if the hint used for optimization disagrees with the one
  // used here for exec, and invalidate any parallel setup (if it was already
  // CSR). That is, except for the supernodal format where the hint needs to
  // match.
  if (trans != impl->userhint_spsv_op) {
    if (impl->spmat_format == perflibs_format_supernodal) {
      return PERFLIBS_STATUS_EXECUTION_FAILURE;
    }
    if (impl->spmat_format != perflibs_format_identity &&
        impl->spmat_format != perflibs_format_null) {
      convert(perflibs_format_csr, impl);
      impl->userhint_spsv_op = trans;
    }
    impl->csr.par_sv = {};
  }

  return spsv_exec_impl<T>(trans, impl, x, alpha, y);
};
template perflibs_status_t spsv_exec<float>(perflibs_sparse_hint_value trans,
                                            perflibs_spmat_top_t *A, float *x,
                                            float alpha, const float *y);
template perflibs_status_t spsv_exec<double>(perflibs_sparse_hint_value trans,
                                             perflibs_spmat_top_t *A, double *x,
                                             double alpha, const double *y);
template perflibs_status_t
spsv_exec<std::complex<float>>(perflibs_sparse_hint_value trans,
                               perflibs_spmat_top_t *A, std::complex<float> *x,
                               std::complex<float> alpha,
                               const std::complex<float> *y);
template perflibs_status_t spsv_exec<std::complex<double>>(
    perflibs_sparse_hint_value trans, perflibs_spmat_top_t *A,
    std::complex<double> *x, std::complex<double> alpha,
    const std::complex<double> *y);

template <typename T>
perflibs_status_t spsm_exec(perflibs_sparse_hint_value trans,
                            perflibs_spmat_top_t *A, perflibs_spmat_top_t *X,
                            T alpha, perflibs_spmat_top_t *Y) {
  auto ensure_dense_col_major = [](perflibs_spmat_impl_t<T> *impl,
                                   bool require_writable) {
    if (impl->spmat_format != perflibs_format_dense) {
      auto ret = convert(perflibs_format_dense, impl);
      if (ret != PERFLIBS_STATUS_SUCCESS) {
        return ret;
      }
    }

    // If we need write access and matrix values are borrowing user storage,
    // materialize our own copy.
    if (require_writable && impl->dense.vals.empty()) {
      impl->dense.make_writable();
      impl->no_copy = false;
    }

    if (impl->dense.layout != PERFLIBS_COL_MAJOR) {
      const auto m = impl->dense.m;
      const auto n = impl->dense.n;
      const auto lda = impl->dense.lda;
      const auto *vals_ptr = impl->dense.vals_ptr;

      std::vector<T> vals_col((size_t)m * (size_t)n);

      // row-major -> col-major
      for (perflibs_int_t i = 0; i < n; i++) {
        for (perflibs_int_t j = 0; j < m; j++) {
          vals_col[(size_t)i * (size_t)m + (size_t)j] = vals_ptr[j * lda + i];
        }
      }

      impl->dense.vals = std::move(vals_col);
      impl->dense.vals_ptr = impl->dense.vals.data();
      impl->dense.layout = PERFLIBS_COL_MAJOR;
      impl->dense.lda = m;
      impl->no_copy = false;
    } else if (require_writable && impl->dense.vals.empty()) {
      // Force writable storage even when already in column-major layout.
      impl->dense.make_writable();
      impl->dense.vals_ptr = impl->dense.vals.data();
      impl->no_copy = false;
    } else if (!impl->dense.vals.empty()) {
      impl->dense.vals_ptr = impl->dense.vals.data();
    }

    return PERFLIBS_STATUS_SUCCESS;
  };

  auto set_input_error = [](perflibs_spmat_impl_t<T> *impl, perflibs_int_t code,
                            const char *msg) {
    impl->error_handle.perflibs_error_type =
        PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    impl->error_handle.perflibs_error_code = code;
    impl->error_handle.err_msg = msg;
  };

  auto impl_A = reinterpret_cast<perflibs_spmat_impl_t<T> *>(A->impl);

  // Reuse core SPSV parameter checks once per SpSM call.
  if (impl_A->spmat_format == perflibs_format_null) {
    set_input_error(impl_A, 2,
                    "It is not possible to perform a triangular solve using a "
                    "null matrix.");
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }
  if (impl_A->spmat_format != perflibs_format_identity) {
    if (trans != PERFLIBS_SPARSE_OPERATION_NOTRANS &&
        trans != PERFLIBS_SPARSE_OPERATION_TRANS &&
        trans != PERFLIBS_SPARSE_OPERATION_CONJTRANS) {
      set_input_error(impl_A, 1, "Incorrect transpose option.");
      return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    }
    if (impl_A->shape == PERFLIBS_SPARSE_SHAPE_RECTANGULAR ||
        impl_A->diag == PERFLIBS_SPARSE_DIAG_ZERO) {
      if (impl_A->shape == PERFLIBS_SPARSE_SHAPE_RECTANGULAR) {
        set_input_error(
            impl_A, 2,
            "The matrix passed to perflibs_spsm_exec is not triangular. "
            "The matrix used to perform a triangular solve must be either "
            "upper or lower triangular.");
      } else {
        set_input_error(
            impl_A, 2,
            "The matrix passed to perflibs_spsm_exec contains at least one "
            "zero on the diagonal.");
      }
      return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    }
  }

  auto impl_X = reinterpret_cast<perflibs_spmat_impl_t<T> *>(X->impl);
  auto ret = ensure_dense_col_major(impl_X, true);
  if (ret != PERFLIBS_STATUS_SUCCESS) {
    return ret;
  }

  auto impl_Y = reinterpret_cast<perflibs_spmat_impl_t<T> *>(Y->impl);
  ret = ensure_dense_col_major(impl_Y, false);
  if (ret != PERFLIBS_STATUS_SUCCESS) {
    return ret;
  }

  if (impl_A->m != impl_A->n) {
    set_input_error(impl_A, 3,
                    "The matrix passed to perflibs_spsm_exec must be square.");
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }

  if (impl_X->n != impl_Y->n) {
    set_input_error(
        impl_A, 4,
        "Matrices X and Y passed to perflibs_spsm_exec must have the same "
        "number of columns.");
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }

  if (impl_X->m != impl_Y->m) {
    set_input_error(
        impl_A, 5,
        "Matrices X and Y passed to perflibs_spsm_exec must have the same "
        "number of rows.");
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }

  if (impl_A->m != impl_X->m || impl_A->m != impl_Y->m) {
    set_input_error(
        impl_A, 6,
        "Row dimensions of A, X and Y passed to perflibs_spsm_exec must "
        "match.");
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }

  // If exec uses a different transpose op than the one optimized for A,
  // refresh optimization for the requested mode.
  if (trans != impl_A->userhint_spsm_op) {
    ret = set_hint(impl_A, PERFLIBS_SPARSE_HINT_SPSM_OPERATION, trans);
    if (ret != PERFLIBS_STATUS_SUCCESS) {
      return ret;
    }
    ret = set_hint(impl_A, PERFLIBS_SPARSE_HINT_SPSV_OPERATION, trans);
    if (ret != PERFLIBS_STATUS_SUCCESS) {
      return ret;
    }
    ret = spsv_optimize(impl_A);
    if (ret != PERFLIBS_STATUS_SUCCESS) {
      return ret;
    }
  }

  // For now, we just loop over SpSV
  const auto nrhs = impl_X->n;
  const auto ldX = impl_X->dense.lda;
  const auto ldY = impl_Y->dense.lda;
  for (perflibs_int_t i = 0; i < nrhs; i++) {
    T *x = impl_X->dense.vals.data() + i * ldX;
    const T *y = impl_Y->dense.vals_ptr + i * ldY;
    ret = call_spsv<T>(trans, impl_A, x, alpha, y);
    if (ret != PERFLIBS_STATUS_SUCCESS) {
      return ret;
    }
  }
  return PERFLIBS_STATUS_SUCCESS;
}
template perflibs_status_t
spsm_exec<float>(perflibs_sparse_hint_value trans, perflibs_spmat_top_t *A,
                 perflibs_spmat_top_t *X, float alpha, perflibs_spmat_top_t *Y);
template perflibs_status_t spsm_exec<double>(perflibs_sparse_hint_value trans,
                                             perflibs_spmat_top_t *A,
                                             perflibs_spmat_top_t *X,
                                             double alpha,
                                             perflibs_spmat_top_t *Y);
template perflibs_status_t
spsm_exec<std::complex<float>>(perflibs_sparse_hint_value trans,
                               perflibs_spmat_top_t *A, perflibs_spmat_top_t *X,
                               std::complex<float> alpha,
                               perflibs_spmat_top_t *Y);
template perflibs_status_t spsm_exec<std::complex<double>>(
    perflibs_sparse_hint_value trans, perflibs_spmat_top_t *A,
    perflibs_spmat_top_t *X, std::complex<double> alpha,
    perflibs_spmat_top_t *Y);
/**
 * This function successively times spsv_exec using half the value of
 * impl->cs?.par_sv.nthreads, stopping when it is no longer faster to use fewer
 * threads. The side effect of this function is to update
 * impl->cs?.par_sv.nthreads with the value that was found to give the best
 * performance.
 */
template <typename T>
void spsv_thread_throttle(perflibs_spmat_impl_t<T> *impl) {
  // We don't want thread throttling at all for baremetal builds due to the use
  // of timers, and we shouldn't get here anyway since this function is only
  // called with omp_get_max_threads() > 1
  constexpr bool verbose = false;
  if (verbose)
    printf("\n ** Performing thread throttling for spsv **\n");

  auto tt_0 = timer_start();

  // Get a pointer to nthreads, the value we want to optimize
  // The matrix is setup either as CSR or CSC (see spsv_optimize)
  int *nthreads =
      &(impl->spmat_format == perflibs_format_csr ? impl->csr.par_sv.nthreads
                                                  : impl->csc.par_sv.nthreads);

  if (verbose)
    printf("starting nthreads = %d\n", *nthreads);
  int nthreads_best = *nthreads;
  // Candidate values of nthreads to benchmark are half the best value
  int nthreads_candidate = std::max(nthreads_best >> 1, 1);

  // Dummy data for the exec call
  T alpha = T(1.0);
  auto trans = impl->userhint_spsv_op;

  // Set x and y to ones, same as alpha
  auto tt_1 = timer_start();
  perflibs::sparse::pod_vector<T> x(impl->n, alpha);
  perflibs::sparse::pod_vector<T> y(impl->n, alpha);
  const double vector_init = timer_end(std::move(tt_1));

  // An array to write to in order to flush cache before each timed exec call -
  // sized as 1MB (Neo.V1 max L2 cache size) for each core
  // Flushing cache avoids skewing results towards the previous run, in which we
  // have warmed up cache
  const size_t flush_size = 1024 * 1024 * nthreads_best;
  perflibs::sparse::pod_vector<double> dummy(flush_size);
  auto flush_cache = [&]() {
#pragma omp parallel for
    for (size_t i = 0; i < flush_size; i++) {
      dummy[i] = 1.0;
    }
  };
  double flush_secs_total = 0.0;
  auto t_flush = timer_start();
  flush_cache(); // initialize the dummy so that the first experiment is not
                 // unfairly affected
  double flush_secs = timer_end(std::move(t_flush));
  flush_secs_total += flush_secs;

  // Variables for the timing
  // target_secs is the time to try to stay within for each execution of
  // time_exec below set this to min of 10 times the first execution cost, or
  // 0.5sec
  auto t0 = timer_start();
  spsv_exec_impl(trans, impl, x.data(), alpha, y.data());
  const double target_secs = std::min(timer_end(std::move(t0)) * 10, 0.5);
  if (verbose)
    printf("Thread throttling target_secs = %e\n", target_secs);
  // The margin of probability used to exit the timing loop early. If the
  // probability that the candidate is better than the current best is `1.0 -
  // margin`, we exit, or if the probability is as low as `margin` we also exit.
  const double margin = 0.3;
  // Timing distributions to build up - the best and the current candidate
  perflibs::sparse::statistics::normal_distribution best_dist, candidate_dist;

  double exec_secs_total = 0.0;

  auto time_exec =
      [&](perflibs::sparse::statistics::normal_distribution &candidate_dist) {
        double elapsed_secs = 0.0;
        do {

          // Flush cache
          t_flush = timer_start();
          flush_cache();
          flush_secs = timer_end(std::move(t_flush));
          flush_secs_total += flush_secs;

          // Time SpSV exec call
          auto t_candidate = timer_start();
          spsv_exec_impl(trans, impl, x.data(), alpha, y.data());
          double candidate_diff_secs = timer_end(std::move(t_candidate));
          exec_secs_total += candidate_diff_secs;

          // Do stats, exit if probability is in either tail of the distribution
          candidate_dist =
              perflibs::sparse::statistics::sample_normal_incremental(
                  candidate_dist, candidate_diff_secs);
          auto prob = perflibs::sparse::statistics::welch_t_test(best_dist,
                                                                 candidate_dist)
                          .v1_p;
          elapsed_secs += candidate_diff_secs + flush_secs;
          if (verbose)
            printf("exec: (t=%e u=%e p=%f) @ %e/%e\n", candidate_diff_secs,
                   candidate_dist.mean, prob, elapsed_secs, target_secs);
          if (prob < margin || prob > (1 - margin)) {
            break;
          }

        } while (elapsed_secs < target_secs);
      };

  // If the candidate is the same as the best then we have no work to do, just
  // return!
  if (nthreads_candidate != nthreads_best) {
    // ... otherwise, we get our starting best timing results with the default
    // setup
    time_exec(best_dist);

    // Next set the candidate value, time it, if it is better then repeat
    // with a new candidate; if it is not better then the candidate stays fixed,
    // which means we'll exit  the loop
    while (nthreads_candidate != nthreads_best) {

      *nthreads = nthreads_candidate;
      if (verbose)
        printf("trying %" PRId64 "\n", (int64_t)nthreads_candidate);
      time_exec(candidate_dist);

      // Only accept the result if there is a significant improvement
      auto prob =
          perflibs::sparse::statistics::welch_t_test(best_dist, candidate_dist)
              .v1_p;
      if (prob > (1 - margin)) {
        best_dist = std::move(candidate_dist);
        nthreads_best = nthreads_candidate;
        nthreads_candidate = std::max(nthreads_best >> 1, 1);
      } else {
        nthreads_candidate = nthreads_best;
      }
    }
    // Make sure to save the best when finished
    *nthreads = nthreads_best;
  }
  const double throttling_overhead = timer_end(std::move(tt_0));
  if (verbose) {
    printf("\nProfile:\nTime spent flushing = %e (%.2f%%)\n", flush_secs_total,
           100.0 * flush_secs_total / throttling_overhead);
    printf("Time spent initializing vectors = %e (%.2f%%)\n", vector_init,
           100.0 * vector_init / throttling_overhead);
    printf("Time spent executing = %e (%.2f%%)\n", exec_secs_total,
           100.0 * exec_secs_total / throttling_overhead);
    double other =
        throttling_overhead - vector_init - flush_secs_total - exec_secs_total;
    printf("Time spent doing other things = %e (%.2f%%)\n", other,
           100.0 * other / throttling_overhead);
    printf("Total time spent thread throttling = %e\n\n", throttling_overhead);

    printf("nthreads_best = %" PRId64 "\n", (int64_t)*nthreads);
    printf(" ** End of thread throttling for spsv **\n\n");
  }
}

template <typename T>
perflibs_status_t spsv_optimize(perflibs_spmat_impl_t<T> *impl) {

  // Early return for null matrix format
  if (impl->spmat_format == perflibs_format_null) {
    impl->error_handle.perflibs_error_type =
        PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    impl->error_handle.perflibs_error_code = 1;
    impl->error_handle.err_msg =
        "It is not possible to perform a triangular solve using a null matrix.";
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  } else if (impl->spmat_format != perflibs_format_identity) {
    if (impl->shape == PERFLIBS_SPARSE_SHAPE_RECTANGULAR ||
        impl->diag == PERFLIBS_SPARSE_DIAG_ZERO) {
      impl->error_handle.perflibs_error_type =
          PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
      impl->error_handle.perflibs_error_code = 1;
      if (impl->shape == PERFLIBS_SPARSE_SHAPE_RECTANGULAR) {
        impl->error_handle.err_msg =
            "The matrix passed to perflibs_spsv_optimize is not triangular. "
            "The "
            "matrix used to perform "
            "a triangular solve must be either upper or lower triangular. "
            "Upper triangular matrices "
            "must have no non-zero values below the diagonal, and lower "
            "triangular matrices must "
            "have no non-zero values above the diagonal.";
      } else if (impl->diag == PERFLIBS_SPARSE_DIAG_ZERO) {
        impl->error_handle.err_msg =
            "The matrix passed to perflibs_spsv_optimize contains at least one "
            "zero on the diagonal. "
            "The matrix used to perform a triangular solve must contain "
            "non-zero values in every "
            "position along the diagonal.";
      }
      return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    }
  }

  auto decide_diag = [&]() {
    // If the diagonals are all in the 'expected' position for CSR/CSC matrices,
    // then we can use a faster kernel which does not check whether every
    // element is a diagonal
    bool known_diag = impl->spmat_format == perflibs_format_csr
                          ? impl->csr.diag_in_place(impl->shape)
                      : impl->spmat_format == perflibs_format_csc
                          ? impl->csc.diag_in_place(impl->shape)
                          : false;
    if (known_diag) {
      impl->diag = impl->diag == PERFLIBS_SPARSE_DIAG_UNIT
                       ? PERFLIBS_SPARSE_DIAG_KNOWN_UNIT
                       : PERFLIBS_SPARSE_DIAG_KNOWN_NON_UNIT;
    }
  };

  // Optimize if we have multiple iterations multiple threads, and the matrix is
  // under our control
  if (impl->userhint_spsv_invocations == PERFLIBS_SPARSE_INVOCATIONS_MANY &&
      !impl->no_copy) {

    // Recursively optimize the Supernodal blocks
    if (impl->spmat_format == perflibs_format_supernodal) {

      // Execute these optimizations in parallel in order to
      // have the setup done with (hopefully!) the same number of nested
      // threads as will be used during execute calls.
      int pinfo = 0;
#pragma omp parallel for reduction(+ : pinfo)
      for (auto &mat : impl->supernodal.mats_diag) {
        auto diag = reinterpret_cast<perflibs_spmat_impl_t<T> *>(mat->impl);
        pinfo += (int)spsv_optimize(diag);
      }

      if ((perflibs_status_t)pinfo != PERFLIBS_STATUS_SUCCESS) {
        return PERFLIBS_STATUS_EXECUTION_FAILURE;
      }

#pragma omp parallel for reduction(+ : pinfo)
      for (auto &mat : impl->supernodal.mats_sep) {
        auto sep_blk = reinterpret_cast<perflibs_spmat_impl_t<T> *>(mat->impl);
        // Force all threads to use the same strategy: optimized SCS
        // This is best for load-balancing
        sep_blk->use_C = 16;
        sep_blk->use_sigma = 200;
        pinfo += (int)spmv_optimize(sep_blk);
      }

      if ((perflibs_status_t)pinfo != PERFLIBS_STATUS_SUCCESS) {
        return PERFLIBS_STATUS_EXECUTION_FAILURE;
      }

      auto sep = reinterpret_cast<perflibs_spmat_impl_t<T> *>(
          impl->supernodal.separator->impl);
      auto info = spsv_optimize(sep);
      if (info != PERFLIBS_STATUS_SUCCESS) {
        return info;
      }

      return PERFLIBS_STATUS_SUCCESS;
    }
    if (perflibs::sparse::omp::get_max_threads() > 1) {
      // If the format is dense
      bool is_dense = impl->spmat_format == perflibs_format_dense;
      // If the format is CSR
      bool is_csr = impl->spmat_format == perflibs_format_csr;
      // If the format is CSC
      bool is_csc = impl->spmat_format == perflibs_format_csc;

      // If the format is none of the above, we have a sparse matrix and
      // we will convert to CSR to allow parallelism
      if (!(is_dense || is_csr || is_csc)) {
        convert(perflibs_format_csr, impl);
        is_csr = impl->spmat_format == perflibs_format_csr;
      }

      if (is_csr || is_csc) {

        /*
            Parallel setup for CSR & CSC
            ============================

            The parallel decomposition works over "levels" which have
           dependencies between them which depend upon the sparsity pattern. For
           notrans solves, levels contain rows. For trans solves, levels contain
           columns.

            The functions gen_parallel_decomp_sv called from this function are
           responsible for the setup. These entry points for this are defined in
           the CSR & CSC .cpp files, but they both call through to the same
           function which does most of the work, gen_parallel_decomp_sv_csx,
           which is defined in sparse_sv.hpp.

            The function which executes the parallel solve is defined in the CSR
           .cpp file, and this is where both CSR & CSC parallel executions end
           up - spsv_csr_parallel.

            We handle transposes by switching between CSR and CSC, as
           appropriate. This is described here:

            CSR input:
            1. For CSR && notrans, since levels contain rows, set up a CSR
           parallel decomp: call impl->csr.template gen_parallel_decomp_s

            2. For CSR && trans, since levels contain columns, convert to CSC,
               and then set up a CSC parallel decomp: impl->csc.template
           gen_parallel_decomp_sv We then fall into the spsv_csc later on, where
           we effect the transpose by calling spsv_csr with our CSC arrays,
           passing the notrans option.

            CSC input:
            3. For CSC && notrans, since levels contain rows, convert to CSR,
               and then set up a CSR parallel decomp: impl->csr.template
           gen_parallel_decomp_sv We then fall into the spsv_csr later on.

            4. For CSC && trans, since levels contain columns, set up a CSC
           parallel decomp: impl->csc.template gen_parallel_decomp_sv We then
           fall into the spsv_csc later on, where we effect the transpose by
           calling spsv_csr with our CSC arrays, passing the notrans option.

            In each of the above cases we need the structure in the 'opposite'
           format since that gives us information about dependencies. For 1 and
           4 above we need to do this explicitly with a call to csr2csc and
           csc2csr, respectively. For 2 and 3 above we are already doing a
           conversion, so we just use the arrays from the input format.
        */

        bool is_trans =
            impl->userhint_spsv_op != PERFLIBS_SPARSE_OPERATION_NOTRANS;

        // Parallel setup works with zero indexing, which avoids repeated
        // subtraction. This means that the par_sv_t has a base index of 0,
        // always, for both level_ptr and par_rows
        auto decrement = [&](perflibs_int_t *ptrs, perflibs_int_t *indxs) {
          for (perflibs_int_t i = 0; i < impl->n + 1; i++) {
            ptrs[i]--;
          }

          for (perflibs_int_t i = 0; i < impl->nnz; i++) {
            indxs[i]--;
          }
        };

        if (is_csr && !is_trans) {

          // We use the col_ptr and row_indx arrays to track dependencies inside
          // gen_parallel_decomp_sv.
          auto [vcol_ptr, vrow_indx] = csr2csc_struct(
              impl->n, impl->n, impl->csr.col_indx_ptr, impl->csr.row_ptr_ptr);

          if (impl->index_base) {
            decrement(vcol_ptr.data(), vrow_indx.data());
          }

          impl->csr.template gen_parallel_decomp_sv<uint8_t>(
              impl->shape, vcol_ptr.data(), vrow_indx.data());
        } else if (is_csr && is_trans) {

          // copy the CSR structure in order to set up the parallel decomp
          std::vector<perflibs_int_t> vrow_ptr(
              impl->csr.row_ptr_ptr, &impl->csr.row_ptr_ptr[impl->n + 1]);
          std::vector<perflibs_int_t> vcol_indx(
              impl->csr.col_indx_ptr, &impl->csr.col_indx_ptr[impl->nnz]);

          if (impl->index_base) {
            decrement(vrow_ptr.data(), vcol_indx.data());
          }

          // convert to CSC since we're working with the transpose
          convert(perflibs_format_csc, impl);

          impl->csc.template gen_parallel_decomp_sv<uint8_t>(
              impl->shape, vrow_ptr.data(), vcol_indx.data());
        } else if (is_csc && !is_trans) {

          // copy the CSC structure in order to set up the parallel decomp
          std::vector<perflibs_int_t> vcol_ptr(
              impl->csc.col_ptr_ptr, &impl->csc.col_ptr_ptr[impl->n + 1]);
          std::vector<perflibs_int_t> vrow_indx(
              impl->csc.row_indx_ptr, &impl->csc.row_indx_ptr[impl->nnz]);

          if (impl->index_base) {
            decrement(vcol_ptr.data(), vrow_indx.data());
          }

          // convert to CSR since we're working with the transpose
          convert(perflibs_format_csr, impl);

          impl->csr.template gen_parallel_decomp_sv<uint8_t>(
              impl->shape, vcol_ptr.data(), vrow_indx.data());
        } else if (is_csc && is_trans) {

          // We use the row_ptr and col_indx arrays to track dependencies inside
          // gen_parallel_decomp_sv.
          auto [vrow_ptr, vcol_indx] = csc2csr_struct(
              impl->n, impl->n, impl->csc.row_indx_ptr, impl->csc.col_ptr_ptr);

          if (impl->index_base) {
            decrement(vrow_ptr.data(), vcol_indx.data());
          }

          impl->csc.template gen_parallel_decomp_sv<uint8_t>(
              impl->shape, vrow_ptr.data(), vcol_indx.data());
        }

        // After the parallel setup and before thread throttling, find out
        // whether diagonal elements are all in known positions or not
        decide_diag();

        // If we are not in a parallel region already then apply thread
        // throttling If we are in a parallel region, then do not, since we most
        // likely want all threads using the same number of nested threads (this
        // is true in the supernodal solve use case)
        if (perflibs::sparse::omp::get_num_threads() == 1) {
          spsv_thread_throttle(impl);
        }
      }
    }
  }

  else {
    // Find out whether diagonal elements are all in known positions or not
    decide_diag();
  }

  return PERFLIBS_STATUS_SUCCESS;
};

template perflibs_status_t
spsv_optimize<float>(perflibs_spmat_impl_t<float> *impl);
template perflibs_status_t
spsv_optimize<double>(perflibs_spmat_impl_t<double> *impl);
template perflibs_status_t spsv_optimize<std::complex<float>>(
    perflibs_spmat_impl_t<std::complex<float>> *impl);
template perflibs_status_t spsv_optimize<std::complex<double>>(
    perflibs_spmat_impl_t<std::complex<double>> *impl);

} // namespace perflibs::sparse
