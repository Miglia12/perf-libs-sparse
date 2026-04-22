/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#pragma once

namespace perflibs::sparse {

struct sparse_matrix_statistics {
  /// mgmd = mean geometric mean distance (geomean of distance between
  /// non-zeroes in a row, averaged over all rows)
  double mgmd;
};

template <typename T> bool get_if_mgmd_is_required_default();

} // end namespace perflibs::sparse
