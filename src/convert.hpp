/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#pragma once

#include "types.hpp"

namespace perflibs::sparse {

template <typename T>
perflibs_status_t convert(spmat_format_t dest_format,
                          perflibs_spmat_impl_t<T> *mat_impl);
template <typename T>
perflibs_status_t convert(spmat_format_t dest_format,
                          perflibs_spmat_impl_t<T> &mat_impl);

} // namespace perflibs::sparse
