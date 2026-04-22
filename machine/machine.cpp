/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

namespace perflibs::sparse {

bool get_sve() {
#ifdef __ARM_FEATURE_SVE
  return true;
#else
  return false;
#endif
}

} // namespace perflibs::sparse
