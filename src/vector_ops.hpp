/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#pragma once

#include "util.hpp"

namespace perflibs::sparse {

/**
 * Gather from a dense vector x_d to a sparse vector x_s.
 * @param [in]  x_d		Dense vector used to populate the sparse vector
 * @param [in]  index_base	base index - must be either 0 or 1
 * @param [in]  n		Number of elements
 * @param [out] x_s		Sparse vector to be populated
 * @param [in]  no_copy		Currently ignored
 */
template <typename T>
perflibs_status_t gather_vec(const T *x_d, perflibs_int_t index_base,
                             perflibs_int_t n, perflibs_spvec_t x_s,
                             bool no_copy);

/**
 * Scatter from a sparse vector x_s to a dense vector x_d.
 * @param [in]  x_s		Source sparse vector
 * @param [out] x_d		Dense vector to be populated
 */
template <typename T>
perflibs_status_t scatter_vec(perflibs_spvec_t x_s, T *x_d);

/**
 * Perform a dot product of a sparse vector x and a dense vector y.
 * @param [in]  x	Sparse vector x
 * @param [in]  y	Dense vector y
 * @param [out] result	Result = x^T . y or x^* . y
 */
template <bool IsConj, typename T>
perflibs_status_t dot_exec(perflibs_spvec_t x, const T *y, T *result);

/**
 * Perform a scalar times a sparse vector accumulated into a scalar times a
 * dense vector.
 * @param [in]  alpha	Scalar parameter alpha
 * @param [in]  x	Sparse vector x
 * @param [in]  beta	Scalar parameter beta
 * @param [in]  y	Input dense vector y
 * @param [out] w	Output dense vector w
 */
template <typename T>
perflibs_status_t waxpby_exec(const T alpha, perflibs_spvec_t x, const T beta,
                              const T *y, T *w);

/**
 * Perform a plane rotation
 * @param [inout]  x	Sparse vector x
 * @param [inout]  y	Dense vector y
 * @param [in]     c	Cosine element of the rotation matrix
 * @param [in]     s	Sine element of the rotation matrix
 */
template <typename T1, typename T2>
perflibs_status_t rot_exec(perflibs_spvec_t x, T1 *y,
                           perflibs::sparse::remove_complex_t<T1> c, T2 s);

} // end namespace perflibs::sparse
