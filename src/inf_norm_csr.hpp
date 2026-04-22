/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#pragma once

#include "compressed_sparse_rows.hpp"

namespace perflibs::sparse {

template <typename T>
void spnorm_inf_csr(const perflibs_csr<T> &csr,
                    perflibs::sparse::remove_complex_t<T> *result);

} // namespace perflibs::sparse
