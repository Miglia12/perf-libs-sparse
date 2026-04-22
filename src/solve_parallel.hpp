/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#pragma once

#include "int.hpp"
#include "pod_vector.hpp"
#include "util.hpp"

#include <algorithm>
#include <arm_neon.h>

// This header exists so CSR and CSC triangular-solve setup can share the same
// low-level parallel decomposition support without forcing higher-level solve
// declarations into the format headers.

namespace perflibs::sparse {

/// This defines SpTRSV parallelism. There are levels of rows/cols that can be
/// solved in parallel. It is a bit like a CSR/CSC structure.
struct par_sv_t {
  /// The number of threads to use for SpTRSV operations
  int nthreads = perflibs::sparse::omp::get_max_threads();

  /// The number of levels in the parallel decomposition. Rows in a level can be
  /// solved in parallel
  int64_t n_levels;

  /// Indices of the start of each level
  perflibs::sparse::pod_vector<perflibs_int_t> level_ptr;

  /// Lists of consecutive rows for each level, where levels are demarked by
  /// level_ptr
  perflibs::sparse::pod_vector<perflibs_int_t> par_rows;

  /// For each row in par_rows, the number of consecutive rows which need to be
  /// performed by the same thread
  perflibs::sparse::pod_vector<perflibs_int_t> chain_len;
};

template <typename T> struct next_int {
  using type = uint64_t;
};
template <> struct next_int<uint8_t> {
  using type = uint16_t;
};
template <> struct next_int<uint16_t> {
  using type = uint32_t;
};

inline uint8_t vwork_check_sv(const uint8_t *ndeps) {
  constexpr uint8x16_t zero = {0};
  const auto next_rows1 = vld1q_u8(ndeps);
  const auto next_rows2 = vld1q_u8(&ndeps[16]);
  const auto zero_rows01 = vceqq_u8(
      next_rows1, zero); // bits set when element is zero, otherwise bits unset
  const auto zero_rows02 = vceqq_u8(next_rows2, zero);
  return vaddvq_u8(zero_rows01 + zero_rows02);
}
inline uint16_t vwork_check_sv(const uint16_t *ndeps) {
  constexpr uint16x8_t zero = {0};
  const auto next_rows1 = vld1q_u16(ndeps);
  const auto next_rows2 = vld1q_u16(&ndeps[8]);
  const auto zero_rows01 = vceqq_u16(next_rows1, zero);
  const auto zero_rows02 = vceqq_u16(next_rows2, zero);
  return vaddvq_u16(zero_rows01 + zero_rows02);
}
inline uint32_t vwork_check_sv(const uint32_t *ndeps) {
  constexpr uint32x4_t zero = {0};
  const auto next_rows1 = vld1q_u32(ndeps);
  const auto next_rows2 = vld1q_u32(&ndeps[4]);
  const auto zero_rows01 = vceqq_u32(next_rows1, zero);
  const auto zero_rows02 = vceqq_u32(next_rows2, zero);
  return vaddvq_u32(zero_rows01 + zero_rows02);
}
inline uint64_t vwork_check_sv(const uint64_t *ndeps) {
  constexpr uint64x2_t zero = {0};
  const auto next_rows1 = vld1q_u64(ndeps);
  const auto next_rows2 = vld1q_u64(&ndeps[2]);
  const auto zero_rows01 = vceqq_u64(next_rows1, zero);
  const auto zero_rows02 = vceqq_u64(next_rows2, zero);
  return vaddvq_u64(zero_rows01 + zero_rows02);
}

/**
 * The core of the parallel setup. Sited here so that it can be inlined from
 * both CSR and CSC.
 *
 * This function iteratively looks for rows which have no dependencies ('root
 * nodes') and puts them into a 'level set'. Level sets can be processed in
 * parallel in the exec call. After finding root nodes for the current level we
 * decrement the number of dependencies to any rows in this level, and then
 * repeat, finding a new set of root nodes for the next level. This is repeated
 * until all rows have been processed.
 *
 * The hot-loop in this process is the one which searches for root nodes. This
 * is vectorized and parallelized to keep the cost of the setup to a minimum.
 *
 * A further optimization is 'chaining'. This looks for rows which are only
 * dependent on the immediate neighboring row; if one is found we add the
 * dependent row into the current level as part of a chain that is processed
 * serially after its neighbor. We then try to step further up the chain to
 * include the next neighbor. This has the effect of avoiding levels with a
 * single row, improving performance of the setup function itself due to fewer
 * levels being processed, and it also improves data locality in the execution,
 * since we avoid some instances where different threads are processed by
 * neighboring rows.
 *
 * @param [in] shape The shape of the matrix (we need to know which direction we
 * are processing the matrix to find our neighbor when chaining).
 * @param [in] n The dimension of the matrix (the matrix must be square for
 * sptrsv - this is checked before reaching this function).
 * @param [in] col_ptr pointer to the starting points of the columns in the
 * @row_indx matrix.
 * @param [in] row_indx row indices of the matrix. Note that @col_ptr and
 * @row_indx variables are named with a CSR-viewpoint - they are actually just
 * the structural arrays in the 'opposite' format from where this function is
 * called from, since the 'opposite' view gives information about dependencies
 * between rows/cols. So, when this function is called from the CSC entry point
 * (csc.gen_parallel_decomp_sv) they are actually 'row_ptr' and 'col_indx'. It's
 * easiest to think of the behavior of this function purely in terms of CSR,
 * however, so go with with the CSR names.
 */
template <typename T>
void gen_parallel_decomp_sv_csx(perflibs_sparse_matrix_shape_t shape,
                                perflibs_int_t n, const perflibs_int_t *col_ptr,
                                const perflibs_int_t *row_indx, T *ndeps,
                                par_sv_t &par_sv) {

  // Direction of travel for chains
  perflibs_int_t inc = shape == PERFLIBS_SPARSE_SHAPE_UPPER_TRIANGULAR ? -1 : 1;

  // Reset the number of threads to match what is available in the next parallel
  // region If we are optimizing in a parallel region, this sets the correct
  // nthreads for nested parallelism
  par_sv.nthreads = perflibs::sparse::omp::get_max_threads();

  // Number of levels
  perflibs_int_t n_levels = 0;
  // Allocate space for the array which points to each level
  par_sv.level_ptr.resize(n + 1);
  auto level_ptr = par_sv.level_ptr.data();
  level_ptr[0] = 0;

  // Number of rows
  perflibs_int_t np_rows = 0;
  // Allocate space for the array which contains rows in each level (pointee of
  // level_ptr)
  par_sv.par_rows.resize(n);
  auto par_rows = par_sv.par_rows.data();

  // Allocate space for the array which counts, for each matching element (row)
  // in par_rows, how many neighboring rows are to be processed by the same
  // thread, in a chain. This is to ensure some level of data locality is
  // maintained compared with the serial approach.
  par_sv.chain_len.resize(n);
  auto chain_len = par_sv.chain_len.data();

  constexpr int unroll =
      2 * 16 /
      sizeof(T); // 2 Neon vectors, each containing 128/sizeof(T) ndeps elements
  constexpr int unroll1 = unroll - 1;
  const int tail_start = n - n % unroll;

  constexpr perflibs_int_t max_chain_len = 64;
  perflibs_int_t totchains = 0;

// Experimentation shows that it's not worth using more than around 8 threads in
// pre-processing here
#pragma omp parallel shared(np_rows, n_levels, totchains),                     \
    firstprivate(n, inc, shape)                                                \
    num_threads(std::min<int>(perflibs::sparse::omp::get_max_threads(), 8))
  {
    perflibs::sparse::pod_vector<perflibs_int_t> my_par_rows(n);

    [[maybe_unused]] auto last = 1;
    while (np_rows + totchains < n) {

      // Assert that we're making progress
      assert(last != np_rows + totchains);
      last = np_rows + totchains;

      perflibs_int_t my_np_rows = 0;

// Find root nodes: those which have not been seen yet and have no dependencies
// Add to the par_rows array
#pragma omp for
      for (perflibs_int_t i = 0; i < n - unroll1; i += unroll) {
        if (vwork_check_sv((T *)&ndeps[i])) { // fire if we have some bits set
          for (auto j = 0; j < unroll; j++) {
            if (ndeps[i + j] == 0) { // If there are no dependencies in this row
              my_par_rows[my_np_rows++] = i + j;
            }
          }
        }
      }

// Update shared view, par_rows
#pragma omp critical
      {
        for (perflibs_int_t i = 0; i < my_np_rows; i++) {
          par_rows[np_rows + i] = my_par_rows[i];
        }
        np_rows += my_np_rows;
      }

#pragma omp barrier // wait for all threads to have updated par_rows

// A single thread processes the tail loop, and then updates ndeps and level_ptr
#pragma omp single
      {
        // tail loop
        for (perflibs_int_t i = tail_start; i < n; i++) {
          if (ndeps[i] == 0) {
            // If there are no dependencies in this row
            par_rows[np_rows++] = i;
          }
        }

        // Find any chains coming from the new root nodes. We need to process
        // chains completely before the root nodes themselves (next loop)
        // because this loop reads/updates ndeps.
        for (perflibs_int_t j = level_ptr[n_levels]; j < np_rows; j++) {
          auto no_dep = par_rows[j];
          auto no_dep1 = no_dep + inc;
          chain_len[j] = 1; // the root node is the first link in the chain

          bool alive = true;
          while (
              alive &&
              chain_len[j] <
                  max_chain_len) { // only keep going if we have an unbroken
                                   // chain, and the chain hasn't got too long
            alive = false;
            // find out if the neighboring row is dependent on the current row
            for (perflibs_int_t ii = col_ptr[no_dep]; ii < col_ptr[no_dep + 1];
                 ii++) {
              // if the row iterate (row_indx[ii]) in the list of dependents for
              // the current row (no_dep) is the direct neighbor of the current
              // row, and it only has one dependency, then that dependency must
              // be on the current row, so the chain can continue
              if (row_indx[ii] == no_dep1 &&
                  (perflibs_int_t)ndeps[no_dep1] == 1) {
                chain_len[j]++;
                totchains++;
                alive = shape == PERFLIBS_SPARSE_SHAPE_UPPER_TRIANGULAR
                            ? no_dep1 + inc >= 0
                            : no_dep1 + inc < n;
                break;
              }
            }
            // repeat, treating the 'neighbor' as the 'current'
            no_dep += inc;
            no_dep1 += inc;
          }
        }

        // Update the number of dependencies for each row. This needs to be done
        // after the chain has been decided.
        for (perflibs_int_t j = level_ptr[n_levels]; j < np_rows; j++) {
          auto no_dep = par_rows[j];

          for (perflibs_int_t ii = 0; ii < chain_len[j]; ii++) {
            for (perflibs_int_t k = col_ptr[no_dep]; k < col_ptr[no_dep + 1];
                 k++) {
              ndeps[row_indx[k]]--;
            }
            no_dep += inc;
          }
        }

        level_ptr[++n_levels] = np_rows;

      } // end omp single, implied barrier is important!

    } // end while

  } // end omp parallel

  par_sv.n_levels = n_levels;
}

} // namespace perflibs::sparse
