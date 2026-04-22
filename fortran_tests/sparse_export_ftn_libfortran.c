/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

/*
 * Wrappers for exporting sparse matrices in Fortran using the C
 * interface. These are required when using a Fortran runtime which
 * does not match the version with which libperflibs has been built.
 */

#include <perflibs_sparse.h>
#include <stdlib.h>

/*
        Compile this file with your toolchain and call this function
        to determine whether or not you should call into the sparse
        matrix export/deallocate wrappers defined in this file.
*/
void query_mismatch_(perflibs_int_t *mismatch) {
// Only detect mismatch if we're not gcc (i.e. LLVM/clang or NVHPC)
// or if we are gcc but earlier than gcc 8
#if defined(__clang__) || defined(__NVCOMPILER) ||                             \
    (!defined(__clang__) && __GNUC__ < 8)
  *mismatch = 1; // Mismatch is true. Wrappers below are required.
#else
  // Wrappers are not required; you should use the documented Fortran
  // sparse export functions and you can disregard this file.
  *mismatch = 0;
#endif
}

// Deallocation of arrays returned by the export functions below
void perflibs_spmat_export_deallocate_wrapper_(void **p) { free(*p); }

/* ---- CSR exports ---- */
void perflibs_spmat_export_csr_wrapper_s_(
    perflibs_const_spmat_t *A, perflibs_int_t *index_base, perflibs_int_t *m,
    perflibs_int_t *n, perflibs_int_t **row_ptr, perflibs_int_t **col_indx,
    float **vals, perflibs_status_t *info) {

  *info = perflibs_spmat_export_csr_s(*A, *index_base, m, n, row_ptr, col_indx,
                                      vals);
}

void perflibs_spmat_export_csr_wrapper_d_(
    perflibs_const_spmat_t *A, perflibs_int_t *index_base, perflibs_int_t *m,
    perflibs_int_t *n, perflibs_int_t **row_ptr, perflibs_int_t **col_indx,
    double **vals, perflibs_status_t *info) {

  *info = perflibs_spmat_export_csr_d(*A, *index_base, m, n, row_ptr, col_indx,
                                      vals);
}

void perflibs_spmat_export_csr_wrapper_c_(
    perflibs_const_spmat_t *A, perflibs_int_t *index_base, perflibs_int_t *m,
    perflibs_int_t *n, perflibs_int_t **row_ptr, perflibs_int_t **col_indx,
    perflibs_singlecomplex_t **vals, perflibs_status_t *info) {

  *info = perflibs_spmat_export_csr_c(*A, *index_base, m, n, row_ptr, col_indx,
                                      vals);
}

void perflibs_spmat_export_csr_wrapper_z_(
    perflibs_const_spmat_t *A, perflibs_int_t *index_base, perflibs_int_t *m,
    perflibs_int_t *n, perflibs_int_t **row_ptr, perflibs_int_t **col_indx,
    perflibs_doublecomplex_t **vals, perflibs_status_t *info) {

  *info = perflibs_spmat_export_csr_z(*A, *index_base, m, n, row_ptr, col_indx,
                                      vals);
}

/* ---- CSC exports ---- */
void perflibs_spmat_export_csc_wrapper_s_(
    perflibs_const_spmat_t *A, perflibs_int_t *index_base, perflibs_int_t *m,
    perflibs_int_t *n, perflibs_int_t **row_indx, perflibs_int_t **col_ptr,
    float **vals, perflibs_status_t *info) {

  *info = perflibs_spmat_export_csc_s(*A, *index_base, m, n, row_indx, col_ptr,
                                      vals);
}

void perflibs_spmat_export_csc_wrapper_d_(
    perflibs_const_spmat_t *A, perflibs_int_t *index_base, perflibs_int_t *m,
    perflibs_int_t *n, perflibs_int_t **row_indx, perflibs_int_t **col_ptr,
    double **vals, perflibs_status_t *info) {

  *info = perflibs_spmat_export_csc_d(*A, *index_base, m, n, row_indx, col_ptr,
                                      vals);
}

void perflibs_spmat_export_csc_wrapper_c_(
    perflibs_const_spmat_t *A, perflibs_int_t *index_base, perflibs_int_t *m,
    perflibs_int_t *n, perflibs_int_t **row_indx, perflibs_int_t **col_ptr,
    perflibs_singlecomplex_t **vals, perflibs_status_t *info) {

  *info = perflibs_spmat_export_csc_c(*A, *index_base, m, n, row_indx, col_ptr,
                                      vals);
}

void perflibs_spmat_export_csc_wrapper_z_(
    perflibs_const_spmat_t *A, perflibs_int_t *index_base, perflibs_int_t *m,
    perflibs_int_t *n, perflibs_int_t **row_indx, perflibs_int_t **col_ptr,
    perflibs_doublecomplex_t **vals, perflibs_status_t *info) {

  *info = perflibs_spmat_export_csc_z(*A, *index_base, m, n, row_indx, col_ptr,
                                      vals);
}

/* ---- BSR exports ---- */
void perflibs_spmat_export_bsr_wrapper_s_(
    perflibs_const_spmat_t *A, enum perflibs_dense_layout *layout,
    perflibs_int_t *index_base, perflibs_int_t *m, perflibs_int_t *n,
    perflibs_int_t *block_size, perflibs_int_t **row_ptr,
    perflibs_int_t **col_indx, float **vals, perflibs_status_t *info) {

  *info = perflibs_spmat_export_bsr_s(*A, *layout, *index_base, m, n,
                                      block_size, row_ptr, col_indx, vals);
}

void perflibs_spmat_export_bsr_wrapper_d_(
    perflibs_const_spmat_t *A, enum perflibs_dense_layout *layout,
    perflibs_int_t *index_base, perflibs_int_t *m, perflibs_int_t *n,
    perflibs_int_t *block_size, perflibs_int_t **row_ptr,
    perflibs_int_t **col_indx, double **vals, perflibs_status_t *info) {

  *info = perflibs_spmat_export_bsr_d(*A, *layout, *index_base, m, n,
                                      block_size, row_ptr, col_indx, vals);
}

void perflibs_spmat_export_bsr_wrapper_c_(
    perflibs_const_spmat_t *A, enum perflibs_dense_layout *layout,
    perflibs_int_t *index_base, perflibs_int_t *m, perflibs_int_t *n,
    perflibs_int_t *block_size, perflibs_int_t **row_ptr,
    perflibs_int_t **col_indx, perflibs_singlecomplex_t **vals,
    perflibs_status_t *info) {

  *info = perflibs_spmat_export_bsr_c(*A, *layout, *index_base, m, n,
                                      block_size, row_ptr, col_indx, vals);
}

void perflibs_spmat_export_bsr_wrapper_z_(
    perflibs_const_spmat_t *A, enum perflibs_dense_layout *layout,
    perflibs_int_t *index_base, perflibs_int_t *m, perflibs_int_t *n,
    perflibs_int_t *block_size, perflibs_int_t **row_ptr,
    perflibs_int_t **col_indx, perflibs_doublecomplex_t **vals,
    perflibs_status_t *info) {

  *info = perflibs_spmat_export_bsr_z(*A, *layout, *index_base, m, n,
                                      block_size, row_ptr, col_indx, vals);
}

/* ---- COO exports ---- */
void perflibs_spmat_export_coo_wrapper_s_(
    perflibs_const_spmat_t *A, perflibs_int_t *m, perflibs_int_t *n,
    perflibs_int_t *nnz, perflibs_int_t **row_indx, perflibs_int_t **col_indx,
    float **vals, perflibs_status_t *info) {

  *info = perflibs_spmat_export_coo_s(*A, m, n, nnz, row_indx, col_indx, vals);
}

void perflibs_spmat_export_coo_wrapper_d_(
    perflibs_const_spmat_t *A, perflibs_int_t *m, perflibs_int_t *n,
    perflibs_int_t *nnz, perflibs_int_t **row_indx, perflibs_int_t **col_indx,
    double **vals, perflibs_status_t *info) {

  *info = perflibs_spmat_export_coo_d(*A, m, n, nnz, row_indx, col_indx, vals);
}

void perflibs_spmat_export_coo_wrapper_c_(
    perflibs_const_spmat_t *A, perflibs_int_t *m, perflibs_int_t *n,
    perflibs_int_t *nnz, perflibs_int_t **row_indx, perflibs_int_t **col_indx,
    perflibs_singlecomplex_t **vals, perflibs_status_t *info) {

  *info = perflibs_spmat_export_coo_c(*A, m, n, nnz, row_indx, col_indx, vals);
}

void perflibs_spmat_export_coo_wrapper_z_(
    perflibs_const_spmat_t *A, perflibs_int_t *m, perflibs_int_t *n,
    perflibs_int_t *nnz, perflibs_int_t **row_indx, perflibs_int_t **col_indx,
    perflibs_doublecomplex_t **vals, perflibs_status_t *info) {

  *info = perflibs_spmat_export_coo_z(*A, m, n, nnz, row_indx, col_indx, vals);
}

/* ---- Dense exports ---- */
void perflibs_spmat_export_dense_wrapper_s_(perflibs_const_spmat_t *A,
                                            enum perflibs_dense_layout *layout,
                                            perflibs_int_t *m,
                                            perflibs_int_t *n, float **vals,
                                            perflibs_status_t *info) {

  *info = perflibs_spmat_export_dense_s(*A, *layout, m, n, vals);
}

void perflibs_spmat_export_dense_wrapper_d_(perflibs_const_spmat_t *A,
                                            enum perflibs_dense_layout *layout,
                                            perflibs_int_t *m,
                                            perflibs_int_t *n, double **vals,
                                            perflibs_status_t *info) {

  *info = perflibs_spmat_export_dense_d(*A, *layout, m, n, vals);
}

void perflibs_spmat_export_dense_wrapper_c_(perflibs_const_spmat_t *A,
                                            enum perflibs_dense_layout *layout,
                                            perflibs_int_t *m,
                                            perflibs_int_t *n,
                                            perflibs_singlecomplex_t **vals,
                                            perflibs_status_t *info) {

  *info = perflibs_spmat_export_dense_c(*A, *layout, m, n, vals);
}

void perflibs_spmat_export_dense_wrapper_z_(perflibs_const_spmat_t *A,
                                            enum perflibs_dense_layout *layout,
                                            perflibs_int_t *m,
                                            perflibs_int_t *n,
                                            perflibs_doublecomplex_t **vals,
                                            perflibs_status_t *info) {

  *info = perflibs_spmat_export_dense_z(*A, *layout, m, n, vals);
}
