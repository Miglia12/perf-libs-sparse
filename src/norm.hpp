/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#pragma once

#include "util.hpp"

#include <vector>

struct perflibs_spmat_top_t;

namespace perflibs::sparse {

template <typename T> T spnorm_max(const std::vector<T> rowsums);

template <typename T>
perflibs_status_t spnorm_exec(perflibs_spmat_top_t *A, perflibs_sparse_norm nrm,
                              perflibs::sparse::remove_complex_t<T> *result);

} // namespace perflibs::sparse
