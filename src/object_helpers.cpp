/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "object_helpers.hpp"
#include "block_sparse_rows.hpp"
#include "compressed_sparse_columns.hpp"
#include "compressed_sparse_rows.hpp"
#include "coordinate_list.hpp"
#include "dense.hpp"
#include "pod_vector.hpp"
#include "supernodal.hpp"
#include "types.hpp"
#include "util.hpp"
#include "vector_storage.hpp"

namespace perflibs::sparse {

template <> perflibs_datatype get_datatype<float>() {
  return PERFLIBS_DATATYPE_SINGLE;
}
template <> perflibs_datatype get_datatype<double>() {
  return PERFLIBS_DATATYPE_DOUBLE;
}
template <> perflibs_datatype get_datatype<std::complex<float>>() {
  return PERFLIBS_DATATYPE_CPLXSINGLE;
}
template <> perflibs_datatype get_datatype<std::complex<double>>() {
  return PERFLIBS_DATATYPE_CPLXDOUBLE;
}

template <typename T, template <typename...> typename V>
void copy_from_vector_or_ptr(const T **ptr, V<T> &vec, const T *other_ptr,
                             const V<T> &other_vec, perflibs_int_t len) {
  static_assert(perflibs::sparse::is_vec_type_v<V<T>>,
                "Template parameter V must be a vector type");
  if (!other_vec.empty()) {
    vec = other_vec;
  } else {
    vec = V<T>(other_ptr, other_ptr + len);
  }
  *ptr = vec.data();
}
template void copy_from_vector_or_ptr(
    const perflibs_int_t **ptr, std::vector<perflibs_int_t> &vec,
    const perflibs_int_t *other_ptr,
    const std::vector<perflibs_int_t> &other_vec, perflibs_int_t len);
template void copy_from_vector_or_ptr(const float **ptr,
                                      std::vector<float> &vec,
                                      const float *other_ptr,
                                      const std::vector<float> &other_vec,
                                      perflibs_int_t len);
template void copy_from_vector_or_ptr(const double **ptr,
                                      std::vector<double> &vec,
                                      const double *other_ptr,
                                      const std::vector<double> &other_vec,
                                      perflibs_int_t len);
template void copy_from_vector_or_ptr(
    const std::complex<float> **ptr, std::vector<std::complex<float>> &vec,
    const std::complex<float> *other_ptr,
    const std::vector<std::complex<float>> &other_vec, perflibs_int_t len);
template void copy_from_vector_or_ptr(
    const std::complex<double> **ptr, std::vector<std::complex<double>> &vec,
    const std::complex<double> *other_ptr,
    const std::vector<std::complex<double>> &other_vec, perflibs_int_t len);

template void copy_from_vector_or_ptr(
    const perflibs_int_t **ptr,
    perflibs::sparse::pod_vector<perflibs_int_t> &vec,
    const perflibs_int_t *other_ptr,
    const perflibs::sparse::pod_vector<perflibs_int_t> &other_vec,
    perflibs_int_t len);
template void copy_from_vector_or_ptr(
    const float **ptr, perflibs::sparse::pod_vector<float> &vec,
    const float *other_ptr,
    const perflibs::sparse::pod_vector<float> &other_vec, perflibs_int_t len);
template void copy_from_vector_or_ptr(
    const double **ptr, perflibs::sparse::pod_vector<double> &vec,
    const double *other_ptr,
    const perflibs::sparse::pod_vector<double> &other_vec, perflibs_int_t len);
template void copy_from_vector_or_ptr(
    const std::complex<float> **ptr,
    perflibs::sparse::pod_vector<std::complex<float>> &vec,
    const std::complex<float> *other_ptr,
    const perflibs::sparse::pod_vector<std::complex<float>> &other_vec,
    perflibs_int_t len);
template void copy_from_vector_or_ptr(
    const std::complex<double> **ptr,
    perflibs::sparse::pod_vector<std::complex<double>> &vec,
    const std::complex<double> *other_ptr,
    const perflibs::sparse::pod_vector<std::complex<double>> &other_vec,
    perflibs_int_t len);

template <typename T>
std::unique_ptr<perflibs_spmat_top_t> spmat_copy(perflibs_spmat_t A) {
  auto ret = std::make_unique<perflibs_spmat_top_t>();
  ret->impl = new perflibs_spmat_impl_t<T>(
      *reinterpret_cast<perflibs_spmat_impl_t<T> *>(A->impl));
  ret->datatype = A->datatype;
  return ret;
}
template std::unique_ptr<perflibs_spmat_top_t>
spmat_copy<float>(perflibs_spmat_t A);
template std::unique_ptr<perflibs_spmat_top_t>
spmat_copy<double>(perflibs_spmat_t A);
template std::unique_ptr<perflibs_spmat_top_t>
spmat_copy<std::complex<float>>(perflibs_spmat_t A);
template std::unique_ptr<perflibs_spmat_top_t>
spmat_copy<std::complex<double>>(perflibs_spmat_t A);

template <typename T>
std::unique_ptr<perflibs_spmat_top_t> spmat_copy(perflibs_const_spmat_t A) {
  auto ret = std::make_unique<perflibs_spmat_top_t>();
  ret->impl = new perflibs_spmat_impl_t<T>(
      *reinterpret_cast<perflibs_spmat_impl_t<T> *>(A->impl));
  ret->datatype = A->datatype;
  return ret;
}
template std::unique_ptr<perflibs_spmat_top_t>
spmat_copy<float>(perflibs_const_spmat_t A);
template std::unique_ptr<perflibs_spmat_top_t>
spmat_copy<double>(perflibs_const_spmat_t A);
template std::unique_ptr<perflibs_spmat_top_t>
spmat_copy<std::complex<float>>(perflibs_const_spmat_t A);
template std::unique_ptr<perflibs_spmat_top_t>
spmat_copy<std::complex<double>>(perflibs_const_spmat_t A);

template <typename T>
std::unique_ptr<perflibs_spmat_top_t> create_new_matrix() {
  auto A = std::make_unique<perflibs_spmat_top_t>();
  A->datatype = get_datatype<T>();
  A->impl = new perflibs_spmat_impl_t<T>;
  return A;
}
template std::unique_ptr<perflibs_spmat_top_t> create_new_matrix<float>();
template std::unique_ptr<perflibs_spmat_top_t> create_new_matrix<double>();
template std::unique_ptr<perflibs_spmat_top_t>
create_new_matrix<std::complex<float>>();
template std::unique_ptr<perflibs_spmat_top_t>
create_new_matrix<std::complex<double>>();

template <typename T>
std::unique_ptr<perflibs_spvec_top_t> create_new_vector() {
  auto x = std::make_unique<perflibs_spvec_top_t>();
  x->datatype = get_datatype<T>();
  x->impl = new perflibs_spvec_impl_t<T>;
  return x;
}
template std::unique_ptr<perflibs_spvec_top_t> create_new_vector<float>();
template std::unique_ptr<perflibs_spvec_top_t> create_new_vector<double>();
template std::unique_ptr<perflibs_spvec_top_t>
create_new_vector<std::complex<float>>();
template std::unique_ptr<perflibs_spvec_top_t>
create_new_vector<std::complex<double>>();

template <typename T>
perflibs_status_t create_spmat_top_csr(perflibs_spmat_top_t **A,
                                       perflibs_int_t m, perflibs_int_t n,
                                       const perflibs_int_t *row_ptr,
                                       const perflibs_int_t *col_indx,
                                       const T *vals, perflibs_int_t flags) {

  auto Atop = create_new_matrix<T>().release();
  *A = Atop;
  return perflibs::sparse::fill_initial_data_csr(
      Atop, m, n, row_ptr, col_indx, vals,
      flags & PERFLIBS_SPARSE_CREATE_NOCOPY);
};
template perflibs_status_t
create_spmat_top_csr<float>(perflibs_spmat_top_t **A, perflibs_int_t m,
                            perflibs_int_t n, const perflibs_int_t *row_ptr,
                            const perflibs_int_t *col_indx, const float *vals,
                            perflibs_int_t flags);
template perflibs_status_t
create_spmat_top_csr<double>(perflibs_spmat_top_t **A, perflibs_int_t m,
                             perflibs_int_t n, const perflibs_int_t *row_ptr,
                             const perflibs_int_t *col_indx, const double *vals,
                             perflibs_int_t flags);
template perflibs_status_t create_spmat_top_csr<std::complex<float>>(
    perflibs_spmat_top_t **A, perflibs_int_t m, perflibs_int_t n,
    const perflibs_int_t *row_ptr, const perflibs_int_t *col_indx,
    const std::complex<float> *vals, perflibs_int_t flags);
template perflibs_status_t create_spmat_top_csr<std::complex<double>>(
    perflibs_spmat_top_t **A, perflibs_int_t m, perflibs_int_t n,
    const perflibs_int_t *row_ptr, const perflibs_int_t *col_indx,
    const std::complex<double> *vals, perflibs_int_t flags);

template <typename T>
perflibs_status_t create_spmat_top_csc(perflibs_spmat_top_t **A,
                                       perflibs_int_t m, perflibs_int_t n,
                                       const perflibs_int_t *row_indx,
                                       const perflibs_int_t *col_ptr,
                                       const T *vals, perflibs_int_t flags) {

  auto Atop = create_new_matrix<T>().release();
  *A = Atop;
  return perflibs::sparse::fill_initial_data_csc(
      Atop, m, n, row_indx, col_ptr, vals,
      flags & PERFLIBS_SPARSE_CREATE_NOCOPY);
};
template perflibs_status_t
create_spmat_top_csc<float>(perflibs_spmat_top_t **A, perflibs_int_t m,
                            perflibs_int_t n, const perflibs_int_t *row_indx,
                            const perflibs_int_t *col_ptr, const float *vals,
                            perflibs_int_t flags);
template perflibs_status_t
create_spmat_top_csc<double>(perflibs_spmat_top_t **A, perflibs_int_t m,
                             perflibs_int_t n, const perflibs_int_t *row_indx,
                             const perflibs_int_t *col_ptr, const double *vals,
                             perflibs_int_t flags);
template perflibs_status_t create_spmat_top_csc<std::complex<float>>(
    perflibs_spmat_top_t **A, perflibs_int_t m, perflibs_int_t n,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_ptr,
    const std::complex<float> *vals, perflibs_int_t flags);
template perflibs_status_t create_spmat_top_csc<std::complex<double>>(
    perflibs_spmat_top_t **A, perflibs_int_t m, perflibs_int_t n,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_ptr,
    const std::complex<double> *vals, perflibs_int_t flags);

template <typename T>
perflibs_status_t
create_spmat_top_coo(perflibs_spmat_top_t **A, perflibs_int_t m,
                     perflibs_int_t n, perflibs_int_t nnz,
                     perflibs_int_t index_base, const perflibs_int_t *row_indx,
                     const perflibs_int_t *col_indx, const T *vals,
                     perflibs_int_t flags) {

  auto Atop = create_new_matrix<T>().release();
  *A = Atop;
  return perflibs::sparse::fill_initial_data_coo(
      Atop, m, n, nnz, index_base, row_indx, col_indx, vals,
      flags & PERFLIBS_SPARSE_CREATE_NOCOPY);
};
template perflibs_status_t create_spmat_top_coo<float>(
    perflibs_spmat_top_t **A, perflibs_int_t m, perflibs_int_t n,
    perflibs_int_t nnz, perflibs_int_t index_base,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_indx,
    const float *vals, perflibs_int_t flags);
template perflibs_status_t create_spmat_top_coo<double>(
    perflibs_spmat_top_t **A, perflibs_int_t m, perflibs_int_t n,
    perflibs_int_t nnz, perflibs_int_t index_base,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_indx,
    const double *vals, perflibs_int_t flags);
template perflibs_status_t create_spmat_top_coo<std::complex<float>>(
    perflibs_spmat_top_t **A, perflibs_int_t m, perflibs_int_t n,
    perflibs_int_t nnz, perflibs_int_t index_base,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_indx,
    const std::complex<float> *vals, perflibs_int_t flags);
template perflibs_status_t create_spmat_top_coo<std::complex<double>>(
    perflibs_spmat_top_t **A, perflibs_int_t m, perflibs_int_t n,
    perflibs_int_t nnz, perflibs_int_t index_base,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_indx,
    const std::complex<double> *vals, perflibs_int_t flags);

template <typename T>
perflibs_status_t
create_spmat_top_dense(perflibs_spmat_top_t **A, perflibs_dense_layout layout,
                       perflibs_int_t m, perflibs_int_t n, perflibs_int_t lda,
                       perflibs_int_t index_base, const T *vals,
                       perflibs_int_t flags) {

  auto Atop = create_new_matrix<T>().release();
  *A = Atop;
  return perflibs::sparse::fill_initial_data_dense(
      Atop, layout, m, n, lda, index_base, vals,
      flags & PERFLIBS_SPARSE_CREATE_NOCOPY);
}
template perflibs_status_t create_spmat_top_dense<float>(
    perflibs_spmat_top_t **A, perflibs_dense_layout layout, perflibs_int_t m,
    perflibs_int_t n, perflibs_int_t lda, perflibs_int_t index_base,
    const float *vals, perflibs_int_t flags);
template perflibs_status_t create_spmat_top_dense<double>(
    perflibs_spmat_top_t **A, perflibs_dense_layout layout, perflibs_int_t m,
    perflibs_int_t n, perflibs_int_t lda, perflibs_int_t index_base,
    const double *vals, perflibs_int_t flags);
template perflibs_status_t create_spmat_top_dense<std::complex<float>>(
    perflibs_spmat_top_t **A, perflibs_dense_layout layout, perflibs_int_t m,
    perflibs_int_t n, perflibs_int_t lda, perflibs_int_t index_base,
    const std::complex<float> *vals, perflibs_int_t flags);
template perflibs_status_t create_spmat_top_dense<std::complex<double>>(
    perflibs_spmat_top_t **A, perflibs_dense_layout layout, perflibs_int_t m,
    perflibs_int_t n, perflibs_int_t lda, perflibs_int_t index_base,
    const std::complex<double> *vals, perflibs_int_t flags);

template <typename T>
perflibs_status_t create_spmat_top_bsr(perflibs_spmat_top_t **A,
                                       perflibs_dense_layout block_layout,
                                       perflibs_int_t m, perflibs_int_t n,
                                       perflibs_int_t block_size,
                                       const perflibs_int_t *row_ptr,
                                       const perflibs_int_t *col_indx,
                                       const T *vals, perflibs_int_t flags) {

  auto Atop = create_new_matrix<T>().release();
  *A = Atop;
  return perflibs::sparse::fill_initial_data_bsr(
      Atop, block_layout, m, n, block_size, row_ptr, col_indx, vals,
      flags & PERFLIBS_SPARSE_CREATE_NOCOPY);
}

template perflibs_status_t create_spmat_top_bsr<float>(
    perflibs_spmat_top_t **A, perflibs_dense_layout block_layout,
    perflibs_int_t m, perflibs_int_t n, perflibs_int_t block_size,
    const perflibs_int_t *row_ptr, const perflibs_int_t *col_indx,
    const float *vals, perflibs_int_t flags);
template perflibs_status_t create_spmat_top_bsr<double>(
    perflibs_spmat_top_t **A, perflibs_dense_layout block_layout,
    perflibs_int_t m, perflibs_int_t n, perflibs_int_t block_size,
    const perflibs_int_t *row_ptr, const perflibs_int_t *col_indx,
    const double *vals, perflibs_int_t flags);
template perflibs_status_t create_spmat_top_bsr<std::complex<float>>(
    perflibs_spmat_top_t **A, perflibs_dense_layout block_layout,
    perflibs_int_t m, perflibs_int_t n, perflibs_int_t block_size,
    const perflibs_int_t *row_ptr, const perflibs_int_t *col_indx,
    const std::complex<float> *vals, perflibs_int_t flags);
template perflibs_status_t create_spmat_top_bsr<std::complex<double>>(
    perflibs_spmat_top_t **A, perflibs_dense_layout block_layout,
    perflibs_int_t m, perflibs_int_t n, perflibs_int_t block_size,
    const perflibs_int_t *row_ptr, const perflibs_int_t *col_indx,
    const std::complex<double> *vals, perflibs_int_t flags);

template <typename T>
perflibs_status_t create_spmat_top_supernodal(
    perflibs_spmat_top_t **A, perflibs_int_t m, perflibs_int_t n,
    perflibs_int_t nsuper, perflibs_int_t nparts,
    const perflibs_int_t *super_row_ptr, const perflibs_int_t *super_col_indx,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_ptr,
    const T *vals, const perflibs_int_t *part_indx, perflibs_int_t flags) {

  auto Atop = create_new_matrix<T>().release();
  *A = Atop;
  return perflibs::sparse::fill_initial_data_supernodal(
      Atop, m, n, nsuper, nparts, super_row_ptr, super_col_indx, row_indx,
      col_ptr, vals, part_indx, flags);
}

template perflibs_status_t create_spmat_top_supernodal<float>(
    perflibs_spmat_top_t **A, perflibs_int_t m, perflibs_int_t n,
    perflibs_int_t nsuper, perflibs_int_t nparts,
    const perflibs_int_t *super_row_ptr, const perflibs_int_t *super_col_indx,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_ptr,
    const float *vals, const perflibs_int_t *part_indx, perflibs_int_t flags);
template perflibs_status_t create_spmat_top_supernodal<double>(
    perflibs_spmat_top_t **A, perflibs_int_t m, perflibs_int_t n,
    perflibs_int_t nsuper, perflibs_int_t nparts,
    const perflibs_int_t *super_row_ptr, const perflibs_int_t *super_col_indx,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_ptr,
    const double *vals, const perflibs_int_t *part_indx, perflibs_int_t flags);
template perflibs_status_t create_spmat_top_supernodal<std::complex<float>>(
    perflibs_spmat_top_t **A, perflibs_int_t m, perflibs_int_t n,
    perflibs_int_t nsuper, perflibs_int_t nparts,
    const perflibs_int_t *super_row_ptr, const perflibs_int_t *super_col_indx,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_ptr,
    const std::complex<float> *vals, const perflibs_int_t *part_indx,
    perflibs_int_t flags);
template perflibs_status_t create_spmat_top_supernodal<std::complex<double>>(
    perflibs_spmat_top_t **A, perflibs_int_t m, perflibs_int_t n,
    perflibs_int_t nsuper, perflibs_int_t nparts,
    const perflibs_int_t *super_row_ptr, const perflibs_int_t *super_col_indx,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_ptr,
    const std::complex<double> *vals, const perflibs_int_t *part_indx,
    perflibs_int_t flags);

template <typename T>
perflibs_status_t create_spvec_top(perflibs_spvec_top_t **x,
                                   perflibs_int_t index_base, perflibs_int_t n,
                                   perflibs_int_t nnz,
                                   const perflibs_int_t *indx, const T *vals,
                                   perflibs_int_t flags) {

  auto xtop = create_new_vector<T>().release();
  *x = xtop;
  return perflibs::sparse::fill_initial_data_vec(
      xtop, index_base, n, nnz, indx, vals,
      flags & PERFLIBS_SPARSE_CREATE_NOCOPY);
};
template perflibs_status_t
create_spvec_top<float>(perflibs_spvec_top_t **x, perflibs_int_t index_base,
                        perflibs_int_t n, perflibs_int_t nnz,
                        const perflibs_int_t *indx, const float *vals,
                        perflibs_int_t flags);
template perflibs_status_t
create_spvec_top<double>(perflibs_spvec_top_t **x, perflibs_int_t index_base,
                         perflibs_int_t n, perflibs_int_t nnz,
                         const perflibs_int_t *indx, const double *vals,
                         perflibs_int_t flags);
template perflibs_status_t create_spvec_top<std::complex<float>>(
    perflibs_spvec_top_t **x, perflibs_int_t index_base, perflibs_int_t n,
    perflibs_int_t nnz, const perflibs_int_t *indx,
    const std::complex<float> *vals, perflibs_int_t flags);
template perflibs_status_t create_spvec_top<std::complex<double>>(
    perflibs_spvec_top_t **x, perflibs_int_t index_base, perflibs_int_t n,
    perflibs_int_t nnz, const perflibs_int_t *indx,
    const std::complex<double> *vals, perflibs_int_t flags);

} // namespace perflibs::sparse
