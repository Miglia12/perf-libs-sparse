/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#pragma once

#include <cstdint>

// Local definition of integer types for sparse internals.
// Keep it compatible with `include/perflibs_int.h` to avoid macro/typedef
// collisions.
#ifndef PERFLIBS_INT_T
#define PERFLIBS_INT_T

#ifdef INTEGER64
using perflibs_int_t = int64_t;
#else
using perflibs_int_t = int32_t;
#endif

#endif
