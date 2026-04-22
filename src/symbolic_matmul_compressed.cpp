/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "symbolic_matmul_compressed.hpp"
#include "util.hpp"
#include <cstdlib>

namespace perflibs::sparse {

symbolic_matmul_compressed::symbolic_matmul_compressed(
    perflibs_int_t m, const perflibs_int_t *row_ptr,
    const perflibs_int_t *col_indx)
    : cs(m), csi(m), cs_len(m), cs_buf(row_ptr[m] - row_ptr[0]),
      csi_buf(row_ptr[m] - row_ptr[0]) {

  auto index_base = row_ptr[0];

#pragma omp parallel for
  for (perflibs_int_t i = 0; i < m; i++) {
    cs[i] = cs_buf.data() + (row_ptr[i] - index_base);
    csi[i] = csi_buf.data() + (row_ptr[i] - index_base);
    perflibs_int_t k = 0;

    bool first = true;
    for (perflibs_int_t j = row_ptr[i] - index_base;
         j < row_ptr[i + 1] - index_base; j++) {
      int64_t indx = col_indx[j] - index_base;
      perflibs_int_t csi_val = indx >> 6; // divide by 64
      uint64_t cs_val =
          1ULL
          << (indx &
              63ULL); // bitwise AND with 63 to get (col_indx mod 64) and then
                      // shift to get the bit set that we need to OR into cs

      // If we haven't seen the csi_val this element belongs to, append a new
      // entry...
      if (first || csi_val != csi[i][k - 1]) {
        csi[i][k] = csi_val;
        cs[i][k++] = cs_val;
      }
      // otherwise OR in the bit that's set for this element to the last value,
      // we can do this since we know the indices are ordered
      else {
        cs[i][k - 1] |= cs_val;
      }
      first = false;
    }

    cs_len[i] = k;
  }
}

std::pair<bool, symbolic_matmul_compressed>
make_symbolic_matmul_compressed(perflibs_int_t m, const perflibs_int_t *row_ptr,
                                const perflibs_int_t *col_indx) {
  auto index_base = row_ptr[0];
  // If rows are not ordered, get out of here. Fall back to single-shot for now.
  for (perflibs_int_t i = 0; i < m; i++) {
    for (perflibs_int_t j = row_ptr[i] + 1 - index_base;
         j < row_ptr[i + 1] - index_base; j++) {
      if (col_indx[j - 1] > col_indx[j]) {
        return {false, {}};
      }
    }
  }

  return {true, symbolic_matmul_compressed(m, row_ptr, col_indx)};
}

/// For each element of cs_C, expand into col_indx_C each set bit as a separate
/// value, using the csi_C offset
inline void compressed_to_csr_row(perflibs_int_t index_base,
                                  perflibs_int_t *col_indx_C,
                                  const uint64_t *cs_C,
                                  const perflibs_int_t *csi_C,
                                  perflibs_int_t cs_C_len) {

  perflibs_int_t row_ptr_C_cur = 0;
  // Iterate over the cs_C, csi_C vectors, translating into integer index values
  for (perflibs_int_t i = 0; i < cs_C_len; i++) {
    perflibs_int_t offset = 64 * csi_C[i];
    uint64_t mask = cs_C[i];
    while (mask) {
      auto indx = __builtin_ctzll(mask);
      col_indx_C[row_ptr_C_cur++] = offset + indx + index_base; // set the index
      mask &= ~(1ULL << indx); // remove the leading set bit
    }
  }
}

/// If base indices of A and B are the same return the value, otherwise return 0
/// (this is what we do in make_index_base_equal)
inline perflibs_int_t get_common_index_base(perflibs_int_t iA,
                                            perflibs_int_t iB) {
  return iA == iB ? iA : 0;
}

/// Process a row of A, merging the compressed row of B (bitwise-OR) to build up
/// a compressed representation of the row of C
inline perflibs_int_t spmm_bitwise(perflibs_int_t index_base_A,
                                   perflibs_int_t rpA_start,
                                   perflibs_int_t rpA_end,
                                   const perflibs_int_t *col_indxA,
                                   const symbolic_matmul_compressed &comp_B,
                                   std::vector<uint64_t> &cs_C_n64,
                                   std::vector<perflibs_int_t> &csi_C_n64) {
  perflibs_int_t nentries = 0;

  // For each element of A in the current row...
  for (auto j = rpA_start; j < rpA_end; j++) {
    auto rB = col_indxA[j] - index_base_A; // The row of B we're looking at

    size_t row_len_B = comp_B.cs_len[rB];
    if (csi_C_n64.size() < nentries + row_len_B) {
      csi_C_n64.resize(nentries + row_len_B);
    }

    // Iterate over each 64-bit cs_C, csi_C entry in B and bitwise-OR into cs_C
    // where we find matching csi_C values
    for (size_t jj = 0; jj < row_len_B; jj++) {
      auto cs_B_val = comp_B.cs[rB][jj];
      auto csi_B_val = comp_B.csi[rB][jj];
      if (!cs_C_n64[csi_B_val]) {
        csi_C_n64[nentries++] = csi_B_val;
      }
      cs_C_n64[csi_B_val] |= cs_B_val;
    }
  }

  return nentries;
}

void get_symbolic_mm_structure(
    perflibs_int_t m, perflibs_int_t n, const perflibs_int_t *row_ptrA,
    const perflibs_int_t *col_indxA, const symbolic_matmul_compressed &comp_B,
    perflibs_int_t index_base_B, std::vector<perflibs_int_t> &row_ptrC,
    perflibs::sparse::pod_vector<perflibs_int_t> &col_indxC_out) {

  // The length of each compressed row of C
  perflibs::sparse::pod_vector<perflibs_int_t> cs_C_len(m);
  // A pointer to the array containing the bitmask for each row of compressed C
  perflibs::sparse::pod_vector<uint64_t *> cs_C(m);
  // A pointer to the array containing the multiple of 64 offset for each
  // element in each array in cs_C
  perflibs::sparse::pod_vector<perflibs_int_t *> csi_C(m);

  // Vectors to handle building up cs_C and csi_C efficiently - we use 2
  // pod_vectors by default, but if they are not large enough we will allocate
  // memory
  constexpr int row_buf_len = 8;
  // A default buffer of fixed size for each row for the cs_C values
  perflibs::sparse::pod_vector<uint64_t> cs_C_buf(m * row_buf_len);
  // A default buffer of fixed size for each row of the csi_C values
  perflibs::sparse::pod_vector<perflibs_int_t> csi_C_buf(m * row_buf_len);
  // A vector to track any extra memory we need beyond the default buffers
  std::vector<bool> free_cs_C(m);

  auto index_base_A = row_ptrA[0];
  auto index_base_C = get_common_index_base(index_base_A, index_base_B);

  // n64 is just used to size cs_C_n64 with an element for each run of 64
  // entries
  auto n64 = n >> 6;
  n64++;

#pragma omp parallel
  {

    std::vector<uint64_t> cs_C_n64(n64);
    constexpr int init_buflen = 4096;
    std::vector<perflibs_int_t> csi_C_n64(init_buflen);

// Apply Gustavson's algorithm: iterate over elements of A, merging rows j of B
// for all non-zero columns j in the current row of A
#pragma omp for
    for (perflibs_int_t i = 0; i < m; i++) {

      auto nentries = spmm_bitwise(index_base_A, row_ptrA[i] - index_base_A,
                                   row_ptrA[i + 1] - index_base_A, col_indxA,
                                   comp_B, cs_C_n64, csi_C_n64);

      // Admittedly, this is ugly. But it is fast, and this loop is
      // performance-critical. We tried hiding this functionality behind a
      // small_ptr_vector type, but the performance was much worse seemingly due
      // to poorer code gen with the type in the way.
      if (nentries < row_buf_len) {
        cs_C[i] = cs_C_buf.data() + i * row_buf_len;
        csi_C[i] = csi_C_buf.data() + i * row_buf_len;
      } else {
        cs_C[i] = (uint64_t *)std::malloc(sizeof(uint64_t) * nentries);
        csi_C[i] =
            (perflibs_int_t *)std::malloc(sizeof(perflibs_int_t) * nentries);
        free_cs_C[i] = true;
      }

      cs_C_len[i] = nentries;

      // Compute row_ptr for C and also store the compressed form of C for use
      // in calculating col_indx values below
      for (perflibs_int_t j = 0; j < nentries; j++) {
        row_ptrC[i + 1] += __builtin_popcountll(cs_C_n64[csi_C_n64[j]]);
        cs_C[i][j] = cs_C_n64[csi_C_n64[j]];
        csi_C[i][j] = csi_C_n64[j];
        cs_C_n64[csi_C_n64[j]] = 0ULL; // reset for next iter
      }
    }
  }

  // Turn row lengths into row pointer values
  row_ptrC[0] = index_base_C;
  for (perflibs_int_t i = 0; i < m; i++) {
    row_ptrC[i + 1] += row_ptrC[i];
  }

  // Now we can accurately size col_indx array
  col_indxC_out.resize(row_ptrC[m] - row_ptrC[0]);

// Compute column indices from compressed form of C
#pragma omp parallel for
  for (perflibs_int_t i = 0; i < m; i++) {
    compressed_to_csr_row(index_base_C,
                          &col_indxC_out[row_ptrC[i] - index_base_C], cs_C[i],
                          csi_C[i], cs_C_len[i]);
    if (free_cs_C[i]) {
      std::free(cs_C[i]);
      std::free(csi_C[i]);
    }
  }
}

void get_symbolic_mm_structure_alloc_only(
    perflibs_int_t m, perflibs_int_t n, const perflibs_int_t *row_ptrA,
    const perflibs_int_t *col_indxA, const symbolic_matmul_compressed &comp_B,
    perflibs_int_t index_base_B, std::vector<perflibs_int_t> &row_ptrC,
    perflibs::sparse::pod_vector<perflibs_int_t> &col_indxC_out) {

  auto index_base_A = row_ptrA[0];
  auto index_base_C = get_common_index_base(index_base_A, index_base_B);

  // n64 is just used to size cs_C_n64 with an element for each run of 64
  // entries
  auto n64 = n >> 6;
  n64++;

#pragma omp parallel
  {

    std::vector<uint64_t> cs_C_n64(n64);
    constexpr int init_buflen = 4096;
    std::vector<perflibs_int_t> csi_C_n64(init_buflen);

// Apply Gustavson's algorithm: iterate over elements of A, merging rows j of B
// for all non-zero columns j in the current row of A
#pragma omp for
    for (perflibs_int_t i = 0; i < m; i++) {

      auto nentries = spmm_bitwise(index_base_A, row_ptrA[i] - index_base_A,
                                   row_ptrA[i + 1] - index_base_A, col_indxA,
                                   comp_B, cs_C_n64, csi_C_n64);

      // Compute row_ptr for C
      for (perflibs_int_t j = 0; j < nentries; j++) {
        row_ptrC[i + 1] += __builtin_popcountll(cs_C_n64[csi_C_n64[j]]);
        cs_C_n64[csi_C_n64[j]] = 0ULL; // reset for next iter
      }
    }
  }

  // Turn row lengths into row pointer values
  row_ptrC[0] = index_base_C;
  for (perflibs_int_t i = 0; i < m; i++) {
    row_ptrC[i + 1] += row_ptrC[i];
  }

  col_indxC_out.resize(row_ptrC[m] - row_ptrC[0]);
}

} // end namespace perflibs::sparse
